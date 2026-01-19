# -*- coding: utf-8 -*-
"""ofdm_lmmse_with_logging.py
- Uses your pilot-based LMMSE estimator (with channel correlation R)
- Logs to runs/<timestamp> (console + file)
- Saves CSV (BER/BLER), meta.json, and a BER plot
"""

# ----------------------------- Env / Imports -----------------------------
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # change GPU id if needed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # silence TF C++ logs

# Third-party
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

sn.phy.config.seed = 42  # reproducibility

from sionna.phy.ofdm import (
    ResourceGrid, RemoveNulledSubcarriers, ResourceGridMapper,
    LSChannelEstimator, PilotPattern, ZFEqualizer, LMMSEEqualizer
)
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.channel import OFDMChannel, ApplyOFDMChannel, subcarrier_frequencies
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.utils import ebnodb2no, expand_to_rank, PlotBER
from sionna.phy.fec.ldpc.decoding import LDPC5GEncoder, LDPC5GDecoder

# ----------------------------- Logging utils -----------------------------
import csv, io, json, contextlib, logging
from datetime import datetime

def _make_run_dir(base_dir="runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = os.path.join(base_dir, ts)
    os.makedirs(p, exist_ok=True)
    return p

def _setup_logging(run_dir: str, level=logging.INFO):
    logger = logging.getLogger("ofdm_logger")
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
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# ----------------------------- Config: R path -----------------------------
# Set the path to your precomputed correlation matrix npz
R_PATH = os.environ.get(
    "R_PATH",
    "/home/ha26227/neural_precoder-reproduce/TDL_R_A_Nsc128_Nsym14_Nsamp1000000_Speed40.0.npz"  # change to your path
)

# --------------------------- Your OFDMSystem ------------------------------
class OFDMSystem(Model):
    def __init__(self, num_subcarriers : int, num_symbols : int, perfect_csi):

        #========== Resource Grid =================#
        self.num_effective_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_symbols
        self.pilot_idx = [2,11]
        self.carrier_spacing = 15e3
        self.carrier_frequency = 2.6e9
        self.cyclic_prefix_length = 6

        # SISO
        NUM_UT = 1
        NUM_UT_ANT = 1
        NUM_STREAMS_PER_TX = NUM_UT_ANT

        RX_TX_ASSOCIATION = np.array([[1]])
        STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

        # Pilot mask (alternate subcarriers in two OFDM symbols)
        mask = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, self.num_ofdm_symbols, self.num_effective_subcarriers])
        sym_idx_1, sym_idx_2 = self.pilot_idx
        mask[0, 0, sym_idx_1, 1::2] = True
        mask[0, 0, sym_idx_2, 1::2] = True
        num_pilot_re = int(np.sum(mask[0,0]))
        pilots = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, num_pilot_re], np.complex64)
        pilots[0, 0, :] = (1+1j)/np.sqrt(2)
        pilot_pattern = PilotPattern(mask, pilots)

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.num_effective_subcarriers,
            subcarrier_spacing=self.carrier_spacing,
            num_tx=NUM_UT,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=[0,0],
            dc_null=False,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=self.pilot_idx
        )

        self.rm = RemoveNulledSubcarriers(self.rg)

        #========== Coding params ==========#
        self.perfect_csi = perfect_csi
        self.num_bits_per_symbol = 6
        self.coderate = 2/3
        self.n = 1944
        self.num_codewords = 2
        self.k = int(self.n * self.coderate)
        self.num_data_symbols = self.rg.num_data_symbols
        syms_per_codeword = self.n // self.num_bits_per_symbol
        total_codeword_syms = self.num_codewords * syms_per_codeword
        pad_syms = self.num_data_symbols - total_codeword_syms
        assert pad_syms >= 0, "Grid too small for the give num_codewords"
        self.pad_bits = pad_syms * self.num_bits_per_symbol

        #========== Blocks ==========#
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, return_infobits=False, hard_out=True, num_iter=40)

        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)

        self.rg_mapper  = ResourceGridMapper(self.rg)
        self._ls_est    = LSChannelEstimator(self.rg, interpolation_type="nn")

        self.tau_rms = 150e-9
        self.TDL = TDL(model="A",
                       delay_spread=self.tau_rms,
                       carrier_frequency=self.carrier_frequency,
                       min_speed=40,
                       max_speed=None)

        self.channel = OFDMChannel(self.TDL, self.rg,
                                   normalize_channel=True,
                                   return_channel=True)

        self.zf_equ = ZFEqualizer(self.rg, STREAM_MANAGEMENT)
        self.lmmse_equ = LMMSEEqualizer(self.rg, STREAM_MANAGEMENT)
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)

    #============ Pilot matrix helper (kept for completeness) ============
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

    #================ LMMSE estimator using channel correlation R =================#
    def lmmse_est(self, y_rg, no):
        """
        Returns:
            h_hat: [batch_size, N] then reshaped to match h_freq
            err_var: [N] (per-RE), then reshaped; finally expanded for equalizer
        """
        # Load R
        if not os.path.exists(R_PATH):
            raise FileNotFoundError(
                f"Correlation matrix file not found: {R_PATH}\n"
                "Set R_PATH env var or edit the script."
            )
        data = np.load(R_PATH, allow_pickle=True)
        R_loaded = data["R"]
        R = tf.convert_to_tensor(R_loaded, dtype=tf.complex64)  # [N, N]
        N = tf.shape(R)[0]

        # Vectorize received grid per-batch
        y = tf.reshape(y_rg, [self.batch_size, -1])              # [B, N]

        # Pilot structure from grid
        pilot_pattern = self.rg.pilot_pattern
        pilot_vals = pilot_pattern.pilots  # [num_tx, num_streams, num_pilots]
        mask = pilot_pattern.mask          # [num_tx, num_streams, num_syms, fft_size]

        flatten_mask = tf.reshape(mask, [-1])
        flatten_mask = tf.cast(flatten_mask, tf.bool)
        pilot_idx = tf.where(flatten_mask)
        pilot_idx = tf.reshape(pilot_idx, [-1])  # [n_P]
        n_P = tf.shape(pilot_idx)[0]

        pilot_vals = tf.cast(pilot_vals, tf.complex64)
        p = tf.scatter_nd(indices=tf.expand_dims(pilot_idx, 1),
                          updates=tf.reshape(pilot_vals, [-1]),
                          shape=[tf.size(flatten_mask)])          # [N]

        # Received pilots
        y_P = tf.gather(y, pilot_idx, axis=1)                    # [B, n_P]
        p_P = tf.gather(p, pilot_idx)                            # [n_P]

        # Build LMMSE maps
        p_conj = tf.math.conj(p)
        Left = R * p_conj[None, :]                               # [N, N]
        Left = tf.gather(Left, pilot_idx, axis=1)                # [N, n_P]

        # R submatrices
        R_pilot_rows = tf.gather(R, pilot_idx, axis=0)           # [n_P, N]
        R_pilot = tf.gather(R_pilot_rows, pilot_idx, axis=1)     # [n_P, n_P]

        # Middle = diag(p_P) R_pp diag(p_P)^H + σ²I
        Middle = (p_P[:, None] * R_pilot) * tf.math.conj(p_P)[None, :]
        Middle = Middle + tf.cast(no, Middle.dtype) * tf.eye(n_P, dtype=Middle.dtype)

        # Stabilize
        Middle = 0.5 * (Middle + tf.linalg.adjoint(Middle))
        mean_diag = tf.reduce_mean(tf.math.real(tf.linalg.diag_part(Middle)))
        eps = tf.cast(1e-7, tf.float32) * tf.maximum(mean_diag, 1.0)
        Middle = tf.linalg.set_diag(Middle, tf.linalg.diag_part(Middle) + tf.complex(eps, 0.0))

        # Solve for each batch: Middle_inv @ y_P^T
        L = tf.linalg.cholesky(Middle)                           # [n_P, n_P]
        I_np = tf.eye(n_P, dtype=Middle.dtype)
        Middle_inv = tf.linalg.cholesky_solve(L, I_np)           # [n_P, n_P]

        tmp = Middle_inv @ tf.transpose(y_P, perm=[1, 0])        # [n_P, B]
        h_hat = Left @ tmp                                       # [N, B]
        h_hat = tf.transpose(h_hat, perm=[1, 0])                 # [B, N]

        # Error covariance diag: R - Left @ Middle_inv @ Right
        Right = p_P[:, None] * R_pilot_rows                      # [n_P, N]
        R_tilde = R - Left @ (Middle_inv @ Right)                # [N, N]
        err_var = tf.linalg.diag_part(R_tilde) + tf.cast(no, R_tilde.dtype)  # [N]
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, tf.cast(1e-10, err_var.dtype))
        return h_hat, err_var

    @tf.function
    def __call__(self, batch_size: int, ebno_db: float):
        self.batch_size = batch_size

        # Eb/N0 -> noise variance
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.coderate,
                       resource_grid=self.rg)

        # Source + encode
        bits = self.binary_source([batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_codewords*self.k])
        bits = tf.reshape(bits, [batch_size, self.num_codewords, self.k])

        c = self.encoder(bits)
        c = tf.reshape(c, [batch_size, self.num_codewords*self.n])

        pad_bits = self.binary_source([batch_size, self.pad_bits])
        coded_frame = tf.concat([c, pad_bits], axis=-1)

        # Map and grid
        x_syms = self.mapper(coded_frame)
        x_syms = tf.reshape(x_syms, [batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_data_symbols])
        x_rg = self.rg_mapper(x_syms)

        # Channel
        y_rg, h_freq = self.channel(x_rg, no)

        # Estimation
        if self.perfect_csi:
            h_hat = self.rm(h_freq)
            err_var = tf.fill(tf.shape(h_hat)[1:], tf.cast(no, tf.float32))
        else:
            h_hat_vec, err_var_vec = self.lmmse_est(y_rg, no)     # [B,N], [N]
            target_shape = tf.shape(h_freq)
            h_hat = tf.reshape(h_hat_vec, target_shape)           # match channel shape
            err_var = tf.reshape(err_var_vec, target_shape[1:])   # no batch dim

        if not self.perfect_csi:
            err_var = tf.expand_dims(err_var, axis=2)             # add stream dim if needed

        # Equalize -> Demap -> Decode
        no_vector = tf.fill([batch_size], no)
        x_hat, no_eff = self.lmmse_equ(y_rg, h_hat, err_var, no_vector)
        no_eff = expand_to_rank(no_eff, tf.rank(x_hat))

        llr = self.demapper(x_hat, no_eff)
        llr = tf.squeeze(llr, axis=[1, 2])
        llr_coded = llr[:, :self.num_codewords * self.n]
        llr_cw = tf.reshape(llr_coded, [self.batch_size, self.num_codewords, self.n])
        u_hat = self.decoder(llr_cw)

        u = tf.reshape(c, [self.batch_size, self.num_codewords, self.n])
        return u, u_hat

# ----------------------------- Run & Log ---------------------------------
if __name__ == "__main__":
    RUN_DIR = _make_run_dir("lmmse_runs")
    LOGGER = _setup_logging(RUN_DIR)

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

    # Model
    Model_lmmse = OFDMSystem(num_subcarriers=128, num_symbols=14, perfect_csi=False)

    # Sweep config
    EBNOS = np.arange(0, 20, 1)
    BATCH = 100
    TARGET_BLK_ERR = 100
    MAX_FRAMES = 5000
    LEGEND = "LMMSE (pilot-corr-R)"

    ber_plots = PlotBER("OFDM over 3GPP TDL")

    LOGGER.info("Starting Eb/N0 sweep...")
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        ber, bler = ber_plots.simulate(
            Model_lmmse,
            ebno_dbs=EBNOS,
            batch_size=BATCH,
            num_target_block_errors=TARGET_BLK_ERR,
            soft_estimates=True,
            max_mc_iter=MAX_FRAMES,
            legend=LEGEND,
            show_fig=False
        )
    console_log = _buf.getvalue()
    with open(os.path.join(RUN_DIR, "simulate_console.log"), "w") as f:
        f.write(console_log)
    LOGGER.info("Sweep done.")

    # Save artifacts
    _save_csv(os.path.join(RUN_DIR, "ber_bler.csv"), EBNOS, ber, bler)
    _save_meta(os.path.join(RUN_DIR, "meta.json"), Model_lmmse, EBNOS, BATCH, TARGET_BLK_ERR, MAX_FRAMES, LEGEND)

    # Plot & save
    plt.figure(figsize=(10, 6))
    plt.semilogy(EBNOS, np.asarray(ber).ravel(), 'o-', label=LEGEND)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(RUN_DIR, "ber_curve.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    LOGGER.info(f"Saved CSV: {os.path.join(RUN_DIR, 'ber_bler.csv')}")
    LOGGER.info(f"Saved console log: {os.path.join(RUN_DIR, 'simulate_console.log')}")
    LOGGER.info(f"Saved meta: {os.path.join(RUN_DIR, 'meta.json')}")
    LOGGER.info(f"Saved plot: {fig_path}")
    print("Artifacts saved in:", RUN_DIR)