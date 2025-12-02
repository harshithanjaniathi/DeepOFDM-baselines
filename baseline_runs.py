import os
# Choose GPU (or export CUDA_VISIBLE_DEVICES before running)
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # <-- set to your GPU id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF C++ logs

try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
        print("Installing Sionna and restarting the runtime. Please run the cell again.")
        os.system("pip install sionna")
        os.kill(os.getpid(), 5)
    else:
        raise e

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
sn.phy.config.seed = 42  

from sionna.phy.ofdm import ResourceGrid,RemoveNulledSubcarriers,ResourceGridMapper,OFDMModulator,OFDMDemodulator,LSChannelEstimator,PilotPattern,ZFEqualizer,LMMSEEqualizer
from sionna.phy.mapping import BinarySource,Mapper,Demapper,SymbolDemapper,Constellation
from sionna.phy.channel import OFDMChannel,GenerateOFDMChannel,cir_to_ofdm_channel,ApplyOFDMChannel,subcarrier_frequencies,ApplyTimeChannel,time_lag_discrete_time_channel,cir_to_time_channel,time_to_ofdm_channel
from sionna.phy.channel.tr38901 import TDL,CDL,AntennaArray
from sionna.phy.utils import ebnodb2no,expand_to_rank,PlotBER
from sionna.phy.fec.ldpc.decoding import LDPC5GEncoder,LDPC5GDecoder
# ----------------------------- Logging utils -----------------------------
import csv, io, json, contextlib, logging, argparse
from datetime import datetime

def _make_run_dir(base_dir="runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = os.path.join(base_dir, ts)
    os.makedirs(p, exist_ok=True)
    return p

def _subdir(root, name):
    p = os.path.join(root, name)
    os.makedirs(p, exist_ok=True)
    return p

def _setup_logging(run_dir: str, name="runner", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

def _save_csv(path, ebnos, ber, bler):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EbNo_dB", "BER", "BLER"])
        for e, b, bl in zip(np.asarray(ebnos).ravel(),
                            np.asarray(ber).ravel(),
                            np.asarray(bler).ravel()):
            w.writerow([float(e), float(b), float(bl)])

def _save_meta(path, model, ebnos, batch_size, num_target_block_errors, max_mc_iter, legend):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tf_version": tf.__version__,
        "sionna_version": getattr(sn, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": len(tf.config.list_physical_devices('GPU')),
        "legend": legend,
        "ebnos": list(map(float, np.asarray(ebnos).ravel())),
        "batch_size": int(batch_size),
        "num_target_block_errors": int(num_target_block_errors),
        "max_mc_iter": int(max_mc_iter),
        "perfect_csi": bool(getattr(model, "perfect_csi", False)),
        "num_subcarriers": int(getattr(model, "num_effective_subcarriers", 0)),
        "num_symbols": int(getattr(model, "num_ofdm_symbols", 0)),
        "coderate": float(getattr(model, "coderate", 0.0)),
        "bits_per_symbol": int(getattr(model, "num_bits_per_symbol", 0)),
        "pilot_idx": list(getattr(model, "pilot_idx", [])),
        "estimator": getattr(model, "estimator", ""),
        "num_iterations": int(getattr(model, "num_iterations", 0)),
        "R_path": R_PATH,
        "domain": getattr(model, "domain", ""),
        "channel_type": getattr(model, "channel_type", ""),
        "channel_model": getattr(model, "channel_model", ""),
        "speed": float(getattr(model, "speed", 0.0)),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# ----------------------------- Config: R path -----------------------------
# R_PATH will be computed at runtime based on CHANNEL and SPEED (or overridden
# by the environment variable R_PATH). Initialize empty here and set in main.
R_PATH = os.environ.get("R_PATH", "")

# --------------------------- Base OFDM System ----------------------------
class OFDMSystemBase(Model):
    def __init__(self, estimator: str, domain: str, num_subcarriers: int, num_symbols: int, perfect_csi: bool, num_iterations:int=4, speed: float=100.0, channel_type: str="TDL", channel_model: str="A"):
        """
        estimator in {"ls", "lmmse", "iter_lmmse"}
        speed: mobility speed in m/s (default 100.0)
        channel_type: "TDL" or "CDL" (default "TDL")
        channel_model: "A", "B", "C", "D", "E" (default "A")
        """
        self.estimator = estimator
        self.num_iterations = num_iterations
        self.speed = speed
        self.domain = domain
        self.channel_type = channel_type
        self.channel_model = channel_model
        #========== Resource Grid (common) =================#
        self.num_effective_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_symbols
        self.pilot_idx = [2,11]
        self.carrier_spacing = 15e3
        self.carrier_frequency = 2.6e9
        self.cyclic_prefix_length = 6

        NUM_UT = 1
        NUM_UT_ANT = 1
        NUM_STREAMS_PER_TX = NUM_UT_ANT
        RX_TX_ASSOCIATION = np.array([[1]])
        self.STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

        # Mask-based pilot pattern: pilots on alternate subcarriers in two OFDM symbols
        # mask = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, self.num_ofdm_symbols, self.num_effective_subcarriers])
        # s1, s2 = self.pilot_idx
        # mask[0, 0, s1, 1::2] = True
        # mask[0, 0, s2, 1::2] = True
        # nP = int(np.sum(mask[0,0]))
        # pilots = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, nP], np.complex64)
        # pilots[0, 0, :] = (1+1j)/np.sqrt(2)
        # pilot_pattern = PilotPattern(mask, pilots)

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.num_effective_subcarriers,
            subcarrier_spacing=self.carrier_spacing,
            num_tx=NUM_UT,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=[0,0],
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_idx
        )
        self.rm = RemoveNulledSubcarriers(self.rg)

        #========== Coding params (common) ==========#
        self.perfect_csi = perfect_csi
        self.num_bits_per_symbol = 6
        self.coderate = 0.5
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = self.n // 2
        self.num_codewords = 1
        self.num_data_symbols = self.rg.num_data_symbols
        syms_per_codeword = self.n // self.num_bits_per_symbol
        total_codeword_syms = self.num_codewords * syms_per_codeword
        pad_syms = self.num_data_symbols - total_codeword_syms
        assert pad_syms >= 0, "Grid too small for the given num_codewords"
        self.pad_bits = pad_syms * self.num_bits_per_symbol

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)

        if estimator == "iter_lmmse" or estimator == "lmmse":
            self.decoder_app = LDPC5GDecoder(self.encoder, hard_out=False, return_infobits=False, num_iter= 20 if estimator == "iter_lmmse" else 40)
            self.decoder_hard = LDPC5GDecoder(self.encoder, hard_out=True, return_infobits=True, num_iter=20 if estimator == "iter_lmmse" else 40)
        else:
            self.decoder_hard = LDPC5GDecoder(self.encoder, return_infobits=False, hard_out=True, num_iter=40)

        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)

        self.rg_mapper  = ResourceGridMapper(self.rg)
        self._ls_est    = LSChannelEstimator(self.rg, interpolation_type="nn")

        self.tau_rms = 150e-9
        
        # Build channel based on type (TDL or CDL)
        if self.channel_type.upper() == "CDL":
            # CDL channel requires antenna arrays
            ut_array = AntennaArray(
                num_rows=1,
                num_cols=1,
                polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=self.carrier_frequency
            )
            bs_array = AntennaArray(
                num_rows=1,
                num_cols=1,
                polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=self.carrier_frequency
            )
            self.channel_obj = CDL(
                model=self.channel_model,
                delay_spread=self.tau_rms,
                carrier_frequency=self.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                min_speed=self.speed,
                max_speed=None
            )
        else:  # TDL (default)
            self.channel_obj = TDL(
                model=self.channel_model,
                delay_spread=self.tau_rms,
                carrier_frequency=self.carrier_frequency,
                min_speed=self.speed,
                max_speed=None
            )

        self.channel = OFDMChannel(self.channel_obj, self.rg, normalize_channel=True, return_channel=True)

        self.zf_equ = ZFEqualizer(self.rg, self.STREAM_MANAGEMENT)
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.STREAM_MANAGEMENT)
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self._l_min, self._l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self.rg.num_time_samples,
                                                      l_tot=self._l_tot,
                                                      add_awgn=True)
        self._modulator = OFDMModulator(self.cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(self.num_effective_subcarriers, self._l_min, self.cyclic_prefix_length)

    def _pilot_mask_flat(self):
        mask = self.rg.pilot_pattern.mask
        flat_mask = tf.reshape(mask, [-1])
        return tf.cast(flat_mask, tf.bool)

    def lmmse_est_R(self, y_rg, no):
        if not os.path.exists(R_PATH):
            raise FileNotFoundError(f"R not found: {R_PATH}")
        data = np.load(R_PATH, allow_pickle=True)
        R_np = data["R"]
        R = tf.convert_to_tensor(R_np, dtype=tf.complex64)  # [N,N]

        y = tf.reshape(y_rg, [self.batch_size, -1])  # [B,N]

        flat_mask = self._pilot_mask_flat()
        pilot_idx = tf.reshape(tf.where(flat_mask), [-1])  # [nP]

        pvals = tf.cast(self.rg.pilot_pattern.pilots, tf.complex64)
        p = tf.scatter_nd(tf.expand_dims(pilot_idx, 1),
                          tf.reshape(pvals, [-1]),
                          [tf.size(flat_mask)])  # [N]
        p_P = tf.gather(p, pilot_idx)                 # [nP]
        y_P = tf.gather(y, pilot_idx, axis=1)         # [B, nP]

        p_conj = tf.math.conj(p)
        Left = R * p_conj[None, :]                    # [N,N]
        Left = tf.gather(Left, pilot_idx, axis=1)     # [N,nP]

        R_rows = tf.gather(R, pilot_idx, axis=0)      # [nP,N]
        R_pp = tf.gather(R_rows, pilot_idx, axis=1)   # [nP,nP]

        Middle = (p_P[:, None] * R_pp) * tf.math.conj(p_P)[None, :]
        Middle = Middle + tf.cast(no, Middle.dtype) * tf.eye(tf.shape(pilot_idx)[0], dtype=Middle.dtype)
        Middle = 0.5 * (Middle + tf.linalg.adjoint(Middle))
        md = tf.reduce_mean(tf.math.real(tf.linalg.diag_part(Middle)))
        eps = tf.cast(1e-7, tf.float32) * tf.maximum(md, 1.0)
        Middle = tf.linalg.set_diag(Middle, tf.linalg.diag_part(Middle) + tf.complex(eps, 0.0))

        L = tf.linalg.cholesky(Middle)
        Middle_inv = tf.linalg.cholesky_solve(L, tf.eye(tf.shape(pilot_idx)[0], dtype=Middle.dtype))

        tmp = Middle_inv @ tf.transpose(y_P, perm=[1, 0])  # [nP,B]
        h_hat = Left @ tmp                                  # [N,B]
        h_hat = tf.transpose(h_hat, perm=[1, 0])           # [B,N]

        Right = p_P[:, None] * R_rows                       # [nP,N]
        R_tilde = R - Left @ (Middle_inv @ Right)           # [N,N]
        err_var = tf.linalg.diag_part(R_tilde) + tf.cast(no, R_tilde.dtype)
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, tf.cast(1e-10, err_var.dtype))
        return h_hat, err_var

    def get_prior_dist(self, llr_p):
        llr_p = tf.reshape(llr_p, [self.batch_size, self.num_data_symbols, self.num_bits_per_symbol])
        indices = tf.range(2**self.num_bits_per_symbol, dtype=tf.int32)
        labels = (tf.bitwise.right_shift(tf.expand_dims(indices, -1),
                                         tf.range(self.num_bits_per_symbol-1, -1, -1)) & 1)
        labels = tf.cast(labels, tf.float32)  # [M,m]
        logits = tf.einsum('bnm,sm->bns', llr_p, labels)
        return tf.nn.softmax(logits, axis=-1)  # [B,n,M]

    def lmmse_est_iterative(self, y_rg, no, priors):
        # R
        if not os.path.exists(R_PATH):
            raise FileNotFoundError(f"R not found: {R_PATH}")
        data = np.load(R_PATH, allow_pickle=True)
        R_np = data["R"]
        R = tf.convert_to_tensor(R_np, dtype=tf.complex64)  # [N,N]
        N = tf.shape(R)[0]

        y = tf.reshape(y_rg, [self.batch_size, -1])  # [B,N]

        const = self.mapper.constellation.points  # [M]
        priors = tf.cast(priors, tf.complex64)    # [B,n,M]
        mu_x  = tf.einsum('bnm,m->bn', priors, const)  # [B,n]
        mu_x2 = tf.einsum('bnm,m->bn', priors, tf.cast(tf.abs(const)**2, tf.complex64))  # [B,n]

        P, _, _ = self.get_pilot_data(y)
        P = tf.cast(P, tf.complex64)         # [1,1,nsym,fft]
        p = tf.reshape(P, [-1])              # [N]

        data_mask = tf.equal(p, tf.complex(0.0, 0.0))
        data_idx  = tf.cast(tf.where(data_mask)[:, 0], tf.int32)  # [N_data]
        N_total   = tf.size(p)
        N_data    = tf.shape(data_idx)[0]

        p_full = tf.tile(p[tf.newaxis, :], [self.batch_size, 1])  # [B,N]

        b = tf.range(self.batch_size, dtype=tf.int32)[:, tf.newaxis]    # [B,1]
        lin = tf.reshape(b * N_total + data_idx[tf.newaxis, :], [-1, 1])  # [B*N_data,1]

        vals_mu = tf.reshape(mu_x, [-1])
        zeros_flat = tf.zeros([self.batch_size * N_total], dtype=mu_x.dtype)
        data_full_flat = tf.tensor_scatter_nd_update(zeros_flat, lin, vals_mu)
        data_full = tf.reshape(data_full_flat, [self.batch_size, N_total])

        mu_full = p_full + data_full  # [B,N]

        pilot_e2_1d = tf.cast(tf.abs(p)**2, tf.complex64)   # [N]
        pilot_e2 = tf.tile(pilot_e2_1d[tf.newaxis, :], [self.batch_size, 1])  # [B,N]

        vals_e2 = tf.reshape(mu_x2, [-1])
        zeros_e2 = tf.zeros([self.batch_size * N_total], dtype=tf.complex64)
        data_e2_flat = tf.tensor_scatter_nd_update(zeros_e2, lin, vals_e2)
        data_e2 = tf.reshape(data_e2_flat, [self.batch_size, N_total])

        e2_full = tf.cast(pilot_e2 + data_e2, tf.float32)   # [B,N]
        var_full = tf.maximum(e2_full - tf.square(tf.abs(mu_full)), 0.0)  # [B,N]

        outer_mu = tf.einsum('bi,bk->bik', mu_full, tf.math.conj(mu_full))        # [B,N,N]
        E_xxH = outer_mu + tf.linalg.diag(tf.cast(var_full, tf.complex64))        # [B,N,N]

        mu_conj = tf.math.conj(mu_full)           
        R_exp = tf.expand_dims(R, axis=0)      
        Left = R_exp * mu_conj[:, tf.newaxis, :]  

        A = tf.math.multiply(tf.expand_dims(R, 0), E_xxH)
        A = A + tf.cast(no, tf.complex64) * tf.eye(N, dtype=tf.complex64)[tf.newaxis, ...]
        A = 0.5 * (A + tf.linalg.adjoint(A))
        A = A + 1e-10 * tf.eye(N, dtype=tf.complex64)[tf.newaxis, :]

        L = tf.linalg.cholesky(A)
        v = tf.linalg.cholesky_solve(L, y[..., tf.newaxis]) 
        v = tf.squeeze(v, -1)                               

        h_hat = tf.einsum('bij,bj->bi', Left, v)             

        Right = R[tf.newaxis, ...] * mu_full[:, :, tf.newaxis]  
        Z = tf.linalg.cholesky_solve(L, Right)                  
        R_tilde = R[tf.newaxis, ...] - tf.matmul(Left, Z)      
        err_var = tf.linalg.diag_part(R_tilde) + tf.cast(no, tf.complex64)  
        return h_hat, err_var 
  
    @tf.function
    def __call__(self, batch_size: int, ebno_db: float):
        self.batch_size = batch_size

        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.coderate,
                       resource_grid=self.rg)

        # Source & coding
        bits = self.binary_source([batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_codewords*self.k])
        if self.estimator == "iter_lmmse":
            bits = tf.reshape(bits, [batch_size, self.num_codewords, self.k])
            c = self.encoder(bits)                                # [B,CW,N]
            c_flat = tf.reshape(c, [batch_size, self.num_codewords*self.n])
            pad_bits = self.binary_source([batch_size, self.pad_bits])
            coded_frame = tf.concat([c_flat, pad_bits], axis=-1)
        else:
            bits = tf.reshape(bits, [batch_size, self.num_codewords, self.k])
            c = self.encoder(bits)                                # [B,CW,N]
            c_flat = tf.reshape(c, [batch_size, self.num_codewords*self.n])
            pad_bits = self.binary_source([batch_size, self.pad_bits])
            coded_frame = tf.concat([c_flat, pad_bits], axis=-1)

        x_syms = self.mapper(coded_frame)
        x_syms = tf.reshape(x_syms, [batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_data_symbols])
        x_rg = self.rg_mapper(x_syms)

        # Channel
        if self.domain == "freq":
            y_rg, h_freq = self.channel(x_rg, no)

        else:  # time domain channel
            a, tau = self.channel_obj(self.batch_size, self.rg.num_time_samples * 1 + self._l_tot - 1, self.rg.bandwidth)  
            h_time = cir_to_time_channel(self.rg.bandwidth, a, tau, l_min=self._l_min, l_max=self._l_max, normalize=True)
            h_freq = time_to_ofdm_channel(h_time, self.rg, self._l_min)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)
            y_rg = self._demodulator(y_time)

       ##Ignore perfefct CSI case here 
        # if self.perfect_csi:
        #     h_hat = self.rm(h_freq)
        #     err_var = tf.fill(tf.shape(h_hat)[1:], tf.cast(no, tf.float32))
        #     err_var = tf.expand_dims(err_var, axis=2)
        #     no_vec = tf.fill([batch_size], no)
        #     x_hat, no_eff = self.lmmse_equ(y_rg, h_hat, err_var, no_vec)
        #     no_eff = expand_to_rank(no_eff, tf.rank(x_hat))
        #     llr = self.demapper(x_hat, no_eff)
        #     llr = tf.squeeze(llr, axis=[1, 2])
        #     llr_coded = llr[:, :self.num_codewords * self.n]
        #     llr_cw = tf.reshape(llr_coded, [self.batch_size, self.num_codewords, self.n])
        #     u_hat = self.decoder_hard(llr_cw)
        #     u = tf.reshape(c, [self.batch_size, self.num_codewords, self.n])
        #     return u, u_hat

        if self.estimator == "ls":
            # LS estimate
            h_hat_ls, err_ls = self._ls_est(y_rg, no)           # err_ls: [rx,rx_ant,nsym,fft]
            target = tf.shape(h_freq)
            h_hat = tf.reshape(h_hat_ls, target)
            err_var = tf.reshape(err_ls, target[1:])
            err_var = tf.expand_dims(err_var, axis=2)
            no_vec = tf.fill([batch_size], no)
            x_hat, no_eff = self.lmmse_equ(y_rg, h_hat, err_var, no_vec)  # LMMSE equalizer
            no_eff = expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self.demapper(x_hat, no_eff)
            llr = tf.squeeze(llr, axis=[1, 2])
            llr_coded = llr[:, :self.num_codewords * self.n]
            llr_cw = tf.reshape(llr_coded, [self.batch_size, self.num_codewords, self.n])
            u_hat = self.decoder_hard(llr_cw)
            u = tf.reshape(c, [self.batch_size, self.num_codewords, self.n])
            return u, u_hat

        # elif self.estimator == "lmmse":
        #     # Non-iter LMMSE with R
        #     h_hat_vec, err_vec = self.lmmse_est_R(y_rg, no)     # [B,N], [N]
        #     target = tf.shape(h_freq)
        #     h_hat = tf.reshape(h_hat_vec, target)
        #     err_var = tf.reshape(err_vec, target[1:])
        #     err_var = tf.expand_dims(err_var, axis=2)
        #     no_vec = tf.fill([batch_size], no)
        #     x_hat, no_eff = self.lmmse_equ(y_rg, h_hat, err_var, no_vec)
        #     no_eff = expand_to_rank(no_eff, tf.rank(x_hat))
        #     llr = self.demapper(x_hat, no_eff)
        #     llr = tf.squeeze(llr, axis=[1, 2])
        #     llr_coded = llr[:, :self.num_codewords * self.n]
        #     llr_cw = tf.reshape(llr_coded, [self.batch_size, self.num_codewords, self.n])
        #     u_hat = self.decoder_hard(llr_cw)
        #     u = tf.reshape(c, [self.batch_size, self.num_codewords, self.n])
        #     return u, u_hat

        else:  # iterative LMMSE
            demapllr_prev = tf.zeros([self.batch_size, self.rg.num_tx, self.rg.num_streams_per_tx,
                                      self.num_data_symbols, self.num_bits_per_symbol], dtype=tf.float32)
            decllr_prev  = tf.zeros_like(demapllr_prev, dtype=tf.float32)

            for _ in range(self.num_iterations):
                priors = self.get_prior_dist(decllr_prev)  
                h_hat_vec, err_vec = self.lmmse_est_iterative(y_rg, no, priors) 
                target = tf.shape(h_freq)
                h_hat = tf.reshape(h_hat_vec, target)
                err_var = tf.reshape(tf.math.real(err_vec), target)[:, :, :1, :, :] 

                no2 = tf.fill([batch_size], no)
                x_hat, no_eff = self.zf_equ(y_rg, h_hat, err_var, no2)  
                no_eff = expand_to_rank(no_eff, tf.rank(x_hat))

                llr = self.demapper(x_hat, no_eff, decllr_prev)  
                llr = tf.squeeze(llr, axis=[1, 2])
                llr_coded = llr[:, :self.num_codewords * self.n]
                llr_cw = tf.reshape(llr_coded, [self.batch_size, self.num_codewords, self.n])

                prev_app = tf.reshape(decllr_prev, [self.batch_size, -1])
                prev_app = prev_app[:, :self.num_codewords * self.n]
                prev_app = tf.reshape(prev_app, [self.batch_size, self.num_codewords, self.n])

                llr_e = llr_cw - prev_app
                dec_llr = self.decoder_app(llr_e)
                decllr_prev = tf.reshape(dec_llr, [self.batch_size, -1])
                decllr_prev = tf.pad(decllr_prev, [[0, 0], [0, self.pad_bits]])
                decllr_prev = tf.reshape(decllr_prev, [self.batch_size, self.rg.num_tx, self.rg.num_streams_per_tx,
                                                       self.num_data_symbols, self.num_bits_per_symbol])

            u_hat = self.decoder_hard(llr_e)
            u = tf.reshape(bits, [self.batch_size, self.num_codewords, self.k])
            return u, u_hat

    def get_pilot_data(self, y):
        pilot_pattern = self.rg.pilot_pattern
        pilot_vals = pilot_pattern.pilots
        mask = pilot_pattern.mask
        flatten_mask = tf.reshape(mask, [-1])
        flatten_mask = tf.cast(flatten_mask, tf.bool)
        idx = tf.where(flatten_mask)
        pilot_vals = tf.cast(pilot_vals, tf.complex64)
        pilot_vals = tf.reshape(pilot_vals, [-1])
        N = tf.size(flatten_mask)
        P_flat = tf.scatter_nd(indices=idx, updates=pilot_vals, shape=[N])
        P = tf.reshape(P_flat, tf.shape(mask))
        y_p = tf.gather(y, idx, axis=1)
        return P, y_p, idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", help="Channel designation (e.g. TDL-D, CDL-C) used for R_PATH lookup",
                        default=os.environ.get("CHANNEL", "TDL-D"))
    parser.add_argument("--channel-type", help="Channel type: TDL or CDL (default: TDL)",
                        default=os.environ.get("CHANNEL_TYPE", "TDL"), choices=["TDL", "CDL"])
    parser.add_argument("--channel-model", help="Channel model variant: A, B, C, D, E (default: A)",
                        default=os.environ.get("CHANNEL_MODEL", "A"))
    parser.add_argument("--speed", help="Speed in m/s (e.g. 10,40,60,100)",
                        default=os.environ.get("SPEED", "10"))
    parser.add_argument("--run-base", help="Base run dir name", default=os.environ.get("RUN_BASE", "cp6_timedomain_01122025"))
    args = parser.parse_args()

    # Compute R_PATH if not provided via env
    channel = args.channel
    speed = args.speed
    if not R_PATH:
        import glob
        speed_tag = f"{float(speed):.1f}"

        # Search directories to look for R mats (prefer the R_mats_timedomain dir)
        repo_dir = os.path.dirname(__file__)
        search_dirs = [os.path.join(repo_dir, "R_mats_timedomain"), repo_dir, os.getcwd()]

        ch_up = channel.upper()
        is_tdl = ch_up.startswith("TDL") or ch_up.startswith("T")
        is_cdl = ch_up.startswith("CDL") or ch_up.startswith("C")
        preferred_letter = None
        # Try to extract a short letter if provided (e.g. "TDL-D" -> "D", "TDL-C" -> "C")
        if "-" in ch_up:
            parts = ch_up.split("-")
            if len(parts) > 1 and len(parts[1]) == 1:
                preferred_letter = parts[1]

        candidates = []
        pat_speed = f"*Speed{speed_tag}*.npz"
        for d in search_dirs:
            if not d:
                continue
            # priority patterns depending on type
            patterns = []
            if is_tdl:
                patterns += [os.path.join(d, f"TDL_R_*Speed{speed_tag}*.npz"),
                             os.path.join(d, f"TDL_R_CIR_*Speed{speed_tag}*.npz"),
                             os.path.join(d, pat_speed)]
            elif is_cdl:
                patterns += [os.path.join(d, f"CDL_R_*Speed{speed_tag}*.npz"),
                             os.path.join(d, f"CDL_R_CIR_*Speed{speed_tag}*.npz"),
                             os.path.join(d, pat_speed)]
            else:
                patterns += [os.path.join(d, pat_speed)]

            for p in patterns:
                matches = glob.glob(p)
                for m in matches:
                    if m not in candidates:
                        candidates.append(m)

        # If multiple candidates, try to pick the one matching preferred_letter (C/D/A etc.)
        chosen = None
        if candidates:
            if preferred_letter:
                for c in candidates:
                    # look for patterns like _D_, _D., _D-N or CIR_D
                    name = os.path.basename(c).upper()
                    if f"_{preferred_letter}" in name or f"CIR_{preferred_letter}" in name or name.endswith(f"{preferred_letter}.NPZ"):
                        chosen = c
                        break
            # fallback heuristics: prefer files in R_mats_timedomain, then repo_dir
            if not chosen:
                for d in search_dirs:
                    for c in candidates:
                        if os.path.dirname(c) == d:
                            chosen = c
                            break
                    if chosen:
                        break
            # final fallback: take first
            if not chosen:
                chosen = candidates[0]

            R_PATH = chosen
        else:
            R_PATH = os.environ.get("R_PATH", "")
    else:
        # R_PATH already provided by environment
        pass

    ROOT = _make_run_dir(args.run_base)
    LOGGER = _setup_logging(ROOT, name="all_estimators")

    # Env info
    LOGGER.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    try:
        details = []
        for i, dev in enumerate(tf.config.list_physical_devices('GPU')):
            nm = tf.config.experimental.get_device_details(dev).get('device_name', f'GPU{i}')
            details.append(nm)
        LOGGER.info(f"GPUs: {details if details else 'CPU'}")
    except Exception as e:
        LOGGER.warning(f"GPU detail query failed: {e}")
    LOGGER.info(f"R_PATH: {R_PATH}")
    LOGGER.info(f"Channel: {channel}, Speed: {speed}")

    # Common sweep config
    EBNOS = np.arange(0, 18, 2)
    BATCH = 64
    TARGET_BLK_ERR = 100
    MAX_FRAMES = 5000

    specs = [
        ("ls",          "LS"),
        ("lmmse",       "LMMSE"),
        ("iter_lmmse",  "Iterative-LMMSE"),
    ]

    curves = {}  # name -> (ber, bler)
    channel_safe = channel.replace('-', '').replace(' ', '_')
    for est_key, legend in specs:
        sub = _subdir(ROOT, f"{est_key}_speed{speed}_{channel_safe}")
        LOGGER.info(f"=== Running {legend} ===")

        Model = OFDMSystemBase(estimator=est_key,
                               num_subcarriers=128,
                               num_symbols=14,
                               perfect_csi=False,
                               num_iterations=4 if est_key=="iter_lmmse" else 1,
                               domain="time",
                               speed=float(speed),
                               channel_type=args.channel_type,
                               channel_model=args.channel_model)

        ber_plots = PlotBER(f"OFDM over 3GPP {args.channel_type}-{args.channel_model} ({legend})")

        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            ber, bler = ber_plots.simulate(
                Model,
                ebno_dbs=EBNOS,
                batch_size=BATCH,
                num_target_block_errors=TARGET_BLK_ERR,
                target_bler=1e-4,
                early_stop=True,
                soft_estimates=True,
                max_mc_iter=MAX_FRAMES,
                legend=legend,
                show_fig=False
            )
        console = _buf.getvalue()

        # Save artifacts
        _save_csv(os.path.join(sub, "ber_bler.csv"), EBNOS, ber, bler)
        with open(os.path.join(sub, "simulate_console.log"), "w") as f:
            f.write(console)
        _save_meta(os.path.join(sub, "meta.json"), Model, EBNOS, BATCH, TARGET_BLK_ERR, MAX_FRAMES, legend)

        LOGGER.info(f"Saved: {sub}/ber_bler.csv, simulate_console.log, meta.json")
        curves[legend] = (np.asarray(ber).ravel(), np.asarray(bler).ravel())
        
        # Clear GPU memory between estimator runs
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    # Save a combined CSV containing all estimators for quick identification
    # Filename contains the channel and speed
    safe_channel = channel.replace(' ', '_')
    comb_csv_name = f"results_{safe_channel}_speed{speed}.csv"
    comb_csv_path = os.path.join(ROOT, comb_csv_name)
    # Prepare header columns in the order of specs
    headers = ["EbNo_dB"]
    for _, legend in specs:
        key = legend.replace('-', '_')
        headers.append(f"{key}_BER")
        headers.append(f"{key}_BLER")

    # Compose rows
    rows = []
    num_points = len(EBNOS)
    for i in range(num_points):
        row = [float(EBNOS[i])]
        for _, legend in specs:
            ber_vals, bler_vals = curves.get(legend, (np.full(num_points, np.nan), np.full(num_points, np.nan)))
            row.append(float(ber_vals[i]))
            row.append(float(bler_vals[i]))
        rows.append(row)

    with open(comb_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    LOGGER.info(f"Saved combined results CSV: {comb_csv_path}")

    # Combined plot
    plt.figure(figsize=(10, 6))
    for legend, (ber, _) in curves.items():
        plt.semilogy(EBNOS, ber, 'o-', label=legend)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    comb_path = os.path.join(ROOT, "ber_curves_all.png")
    plt.savefig(comb_path, dpi=220)
    plt.close()
    LOGGER.info(f"Saved combined plot: {comb_path}")