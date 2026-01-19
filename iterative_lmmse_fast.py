import os
import sionna as sn
import sionna.phy

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import Model

# Reproducibility
sionna.phy.config.seed = 42

# Sionna imports
from sionna.phy.ofdm import (
    ResourceGrid, RemoveNulledSubcarriers, ResourceGridMapper,
    OFDMModulator, OFDMDemodulator, LSChannelEstimator, PilotPattern,
    ZFEqualizer, LMMSEEqualizer
)
from sionna.phy.mapping import BinarySource, Mapper, Demapper, SymbolDemapper, Constellation
from sionna.phy.channel import (
    OFDMChannel, GenerateOFDMChannel, cir_to_ofdm_channel, ApplyOFDMChannel,
    subcarrier_frequencies, ApplyTimeChannel, time_lag_discrete_time_channel,
    cir_to_time_channel, time_to_ofdm_channel
)
from sionna.phy.channel.tr38901 import TDL, CDL, AntennaArray
from sionna.phy.utils import ebnodb2no, expand_to_rank, PlotBER
from sionna.phy.fec.ldpc.decoding import LDPC5GEncoder, LDPC5GDecoder

import csv, io, json, contextlib
from datetime import datetime

def _make_run_dir(base_dir="runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = os.path.join(base_dir, ts)
    os.makedirs(p, exist_ok=True)
    return p

def _save_csv(path, ebnos, ber, bler):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EbNo_dB", "BER", "BLER"])
        for e, b, bl in zip(np.asarray(ebnos).ravel(),
                            np.asarray(ber).ravel(),
                            np.asarray(bler).ravel()):
            w.writerow([float(e), float(b), float(bl)])

class OFDMSystem(Model):
    def __init__(self, num_subcarriers : int, num_symbols : int, perfect_csi):

        # ---------- config ----------
        self.num_effective_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_symbols
        self.pilot_idx = [2,11]
        self.carrier_spacing = 15e3
        self.carrier_frequency = 2.6e9
        self.cyclic_prefix_length = 6

        # logging knobs
        self.verbose = True
        self.log_stream = None  # set later from main

        # ---------- SISO / stream mgmt ----------
        NUM_UT = 1
        NUM_BS = 1
        NUM_UT_ANT = 1
        NUM_BS_ANT = 1
        NUM_STREAMS_PER_TX = NUM_UT_ANT
        RX_TX_ASSOCIATION = np.array([[1]])
        STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

        # ---------- pilot pattern ----------
        mask = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, self.num_ofdm_symbols, self.num_effective_subcarriers])
        sym_idx_1 = self.pilot_idx[0]
        sym_idx_2 = self.pilot_idx[1]
        mask[0, 0, sym_idx_1, 1::2] = True
        mask[0, 0, sym_idx_2, 1::2] = True
        num_pilot_symbols = int(np.sum(mask[0,0]))
        pilots = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, num_pilot_symbols], np.complex64)
        pilots[0, 0, :] = (1+1j)/np.sqrt(2)
        pilot_pattern = PilotPattern(mask, pilots)

        self.rg = ResourceGrid(
            num_ofdm_symbols = self.num_ofdm_symbols,
            fft_size = self.num_effective_subcarriers,
            subcarrier_spacing = self.carrier_spacing,
            num_tx= NUM_UT,
            num_streams_per_tx = NUM_STREAMS_PER_TX,
            cyclic_prefix_length = self.cyclic_prefix_length,
            num_guard_carriers = [0,0],
            dc_null = False,
            pilot_pattern = pilot_pattern,                    # << use the pattern we built
            pilot_ofdm_symbol_indices= self.pilot_idx
        )

        self.rm = RemoveNulledSubcarriers(self.rg)

        # ---------- coding ----------
        self.perfect_csi = perfect_csi
        self.num_bits_per_symbol = 6
        self.coderate = 1/2
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.num_codewords = 1
        self.k = int(self.n // 2)
        self.num_data_symbols = self.rg.num_data_symbols
        self.syms_per_codeword = self.n // self.num_bits_per_symbol
        total_codeword_syms = self.num_codewords * self.syms_per_codeword
        pad_syms = self.num_data_symbols - total_codeword_syms
        assert pad_syms >= 0, "Grid too small for the give num_codewords"
        self.pad_bits = pad_syms * self.num_bits_per_symbol

        # ---------- blocks ----------
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out= False ,return_infobits = False, num_iter = 20)
        self.dec_bits = LDPC5GDecoder(self.encoder, hard_out= True ,return_infobits = True, num_iter = 20)

        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)

        self.rg_mapper  = ResourceGridMapper(self.rg)
        self._ls_est    = LSChannelEstimator(self.rg, interpolation_type="nn")

        self.tau_rms  = 150e-9

        self.TDL = TDL(model="A",
                  delay_spread=self.tau_rms,
                  carrier_frequency= self.carrier_frequency,
                  min_speed = 40.0,
                  max_speed = None)

        self.channel = OFDMChannel(self.TDL,
                                   self.rg,
                                   normalize_channel = True,
                                   return_channel = True)

        self.zf_equ = ZFEqualizer(self.rg, STREAM_MANAGEMENT)
        self.lmmse_equ = LMMSEEqualizer(self.rg, STREAM_MANAGEMENT)
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self._l_min, self._l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self.rg.num_time_samples,
                                              l_tot=self._l_tot,
                                              add_awgn=True)
        self._modulator = OFDMModulator(self.cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(self.num_effective_subcarriers, self._l_min, self.cyclic_prefix_length)

    # ---- logging helper (works inside tf.function) ----
    @tf.function
    def _log(self, *msg):
        if self.verbose:
            stream = self.log_stream if self.log_stream is not None else 'stdout'
            tf.print(*msg, output_stream=stream)

    #============ Pilot matrix P ============#
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

      #==== collect pilot positions in y =====#
      y_p = tf.gather(y, idx, axis = 1)
      return P,y_p,idx

    #============== Priors computation =================#
    def get_prior_dist(self, llr_p):
      llr_p = tf.reshape(llr_p, [self.batch_size, self.num_data_symbols, self.num_bits_per_symbol])
      indices = tf.range(2**self.num_bits_per_symbol, dtype=tf.int32)   # [M]
      labels = tf.bitwise.right_shift(
                        tf.expand_dims(indices, axis=-1),
                        tf.range(self.num_bits_per_symbol-1, -1, -1)) & 1   # [M, m]

      labels = tf.cast(labels, tf.float32)
      logits = tf.einsum('bnm,sm->bns', llr_p, labels)  # sum_i c_j^(i) * LLR(k,i)  -> [B, n, M]
      P_sym  = tf.nn.softmax(logits, axis=-1)           # [B, n, M]   symbol prior probs
      return P_sym

    #========================== LMMSE Estimator ===============================#
    def lmmse_est(self, y_rg, no, priors):
      t0 = tf.timestamp()

      data = np.load("/home/ha26227/neural_precoder-reproduce/TDL_R_A_Nsc128_Nsym14_Nsamp1000000_Speed40.0.npz", allow_pickle=True)
      R_loaded = data["R"]
      R = tf.convert_to_tensor(R_loaded, dtype=tf.complex64)

      y = tf.reshape(y_rg,[self.batch_size,-1])

      const_points = self.mapper.constellation.points
      priors = tf.cast(priors, tf.complex64)
      mu_x  = tf.einsum('bnm,m->bn', priors, const_points)
      mu_x2 = tf.einsum('bnm,m->bn', priors, tf.cast(tf.abs(const_points)**2, tf.complex64))

      P, y_p, pilot_idx = self.get_pilot_data(y)
      P = tf.cast(P, tf.complex64)
      p = tf.reshape(P, [-1])

      data_mask = tf.equal(p, tf.complex(0.0, 0.0))
      data_idx  = tf.where(data_mask)[:, 0]
      data_idx = tf.cast(data_idx, tf.int32)

      N_total  = tf.size(p)
      N_data   = tf.shape(data_idx)[0]

      # logs: pilot/data counts (once per call)
      self._log("iterLMMSE: N_total=", N_total, " nPilots=", tf.size(p) - tf.cast(N_data, tf.int32), " nData=", N_data)

      p_full = tf.tile(p[tf.newaxis, :], [self.batch_size, 1])  # [B, N_total], complex64

      b = tf.range(self.batch_size, dtype=tf.int32)[:, tf.newaxis]         # [B,1]
      lin = b * tf.cast(N_total, tf.int32) + data_idx[tf.newaxis, :]       # [B, N_data]
      lin = tf.reshape(lin, [-1, 1])                                       # [B*N_data, 1]

      vals = tf.reshape(mu_x, [-1])
      zeros_flat = tf.zeros([self.batch_size * N_total], dtype=mu_x.dtype)
      data_full_flat = tf.tensor_scatter_nd_update(zeros_flat, lin, vals)
      data_full = tf.reshape(data_full_flat, [self.batch_size, N_total])

      mu_full = p_full + data_full

      pilot_e2_1d = tf.cast(tf.abs(p)**2, tf.complex64)
      pilot_e2    = tf.tile(pilot_e2_1d[tf.newaxis, :], [self.batch_size, 1])

      vals = tf.reshape(mu_x2, [-1])
      zeros_flat   = tf.zeros([self.batch_size * N_total], dtype=tf.complex64)
      data_e2_flat = tf.tensor_scatter_nd_update(zeros_flat, lin, vals)
      data_e2      = tf.reshape(data_e2_flat, [self.batch_size, N_total])

      e2_full = tf.cast(pilot_e2 + data_e2, tf.float32)
      var_full = tf.maximum(e2_full - tf.square(tf.abs(mu_full)), 0.0 )

      outer_mu = tf.einsum('bi,bk->bik', mu_full, tf.math.conj(mu_full))
      E_xxH = outer_mu + tf.linalg.diag(tf.cast(var_full, tf.complex64))

      mu_conj = tf.math.conj(mu_full)
      R_exp = tf.expand_dims(R, axis=0)

      Left = R_exp * mu_conj[:, tf.newaxis, :]

      A = tf.math.multiply(tf.expand_dims(R, 0), E_xxH)
      N = tf.shape(R)[0]
      A = A + tf.cast(no, tf.complex64) * tf.eye(N, dtype=tf.complex64)[tf.newaxis, ...]
      A = 0.5 * (A + tf.linalg.adjoint(A))
      A = A + 1e-10 * tf.eye(N, dtype=tf.complex64)[tf.newaxis, :]

      t1 = tf.timestamp()
      L = tf.linalg.cholesky(A)
      v = tf.linalg.cholesky_solve(L, y[..., tf.newaxis])
      v = tf.squeeze(v, -1)
      t2 = tf.timestamp()

      h_hat = tf.einsum('bij,bj->bi', Left, v)

      Right = R[tf.newaxis, ...] * mu_full[:, :, tf.newaxis]
      Z = tf.linalg.cholesky_solve(L, Right)
      R_tilde = R[tf.newaxis, ...] - tf.matmul(Left, Z)
      err_var = tf.linalg.diag_part(R_tilde)
      err_var = err_var + tf.cast(no, err_var.dtype)

      # log timings
      self._log("iterLMMSE: build(ms)=", tf.round((t1 - t0)*1000.0),
                " solve(ms)=", tf.round((t2 - t1)*1000.0))

      return h_hat,err_var

    @tf.function
    def __call__(self, batch_size: int, ebno_db: float):
      self.batch_size = batch_size

      no = ebnodb2no(ebno_db,
              num_bits_per_symbol=self.num_bits_per_symbol,
              coderate=self.coderate,
              resource_grid = self.rg)

      # call-level header
      self._log("\n=== CALL: Eb/N0(dB)=", ebno_db, " batch=", batch_size, " no=", no, " ===")

      bits = self.binary_source([batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_codewords*self.k])
      bits = tf.reshape(bits, [batch_size, self.num_codewords, self.k])

      c = self.encoder(bits)
      c = tf.reshape(c, [batch_size, self.num_codewords*self.n])

      pad_bits = self.binary_source([batch_size, self.pad_bits])
      coded_frame = tf.concat([c, pad_bits], axis = -1)

      x_syms = self.mapper(coded_frame)
      x_syms = tf.reshape(x_syms, [batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, self.num_data_symbols])
      x_rg = self.rg_mapper(x_syms)

      y_rg, h_freq = self.channel(x_rg,no)

      decllr_prev = tf.zeros([self.batch_size, self.rg.num_tx,self.rg.num_streams_per_tx, self.num_data_symbols, self.num_bits_per_symbol], dtype=tf.float32)
      u_hat = tf.zeros([batch_size, self.num_codewords, self.k], dtype=tf.float32)

      for iter in tf.range(4):
        it0 = tf.timestamp()
        self._log("  [it", iter, "] start")

        priors = self.get_prior_dist(decllr_prev)

        if(self.perfect_csi):
            h_hat, err_var = self.rm(h_freq), 0.0
        else:
            h_hat, err_var = self.lmmse_est(y_rg, no, priors)

        if(not self.perfect_csi):
          target_shape = tf.shape(h_freq)
          h_hat = tf.reshape(h_hat, target_shape)
          err_var = tf.reshape(err_var, target_shape)

        # quick stats on estimate
        self._log("  [it", iter, "] E|h_hat|^2~", tf.reduce_mean(tf.abs(h_hat)**2),
                  "  E[err_var]~", tf.reduce_mean(err_var))

        no2 = tf.fill([batch_size], no)
        t_eq0 = tf.timestamp()
        x_hat, no_eff = self.zf_equ(y_rg, h_hat, err_var, no2)
        no_eff = expand_to_rank(no_eff, tf.rank(x_hat))
        t_eq1 = tf.timestamp()

        llr = self.demapper(x_hat, no_eff, decllr_prev)
        llr = tf.squeeze(llr, axis=[1, 2])
        llr_coded = llr[:, :self.num_codewords * self.n]
        llr_cw = tf.reshape(llr_coded, [self.batch_size,self.num_codewords, self.n])

        # a-priori used this iter (for logging)
        apr = tf.reshape(decllr_prev, [self.batch_size,-1])[:, :self.num_codewords * self.n]
        apr = tf.reshape(apr, [self.batch_size,self.num_codewords, self.n])

        # extrinsic
        llr_e   = llr_cw - apr
        t_dec0 = tf.timestamp()
        dec_llr  = self.decoder(llr_e)
        t_dec1 = tf.timestamp()

        # magnitudes snapshot
        self._log("  [it", iter, "] ||no_eff||~", tf.reduce_mean(no_eff),
                  "  mean|LLR_in|~", tf.reduce_mean(tf.abs(llr_cw)),
                  "  mean|LLR_apr|~", tf.reduce_mean(tf.abs(apr)),
                  "  mean|LLR_dec|~", tf.reduce_mean(tf.abs(dec_llr)),
                  "  eq(ms)=", tf.round((t_eq1 - t_eq0)*1000.0),
                  "  dec(ms)=", tf.round((t_dec1 - t_dec0)*1000.0))

        if tf.equal(iter, 3):
          u_hat = self.dec_bits(llr_e)

        decllr_prev = tf.reshape(dec_llr,[self.batch_size,-1])
        decllr_prev = tf.pad(decllr_prev, paddings=[[0, 0], [0, self.pad_bits]])
        decllr_prev = tf.reshape(decllr_prev, [self.batch_size, self.rg.num_tx,self.rg.num_streams_per_tx, self.num_data_symbols, self.num_bits_per_symbol])

        it1 = tf.timestamp()
        self._log("  [it", iter, "] done. iter(ms)=", tf.round((it1 - it0)*1000.0))

      u = tf.reshape(bits, [self.batch_size, self.num_codewords, self.k])
      self._log("=== CALL done ===")
      return u, u_hat

# ------------------------- run sweep (with extra logs) -------------------------
EBN0_DB_MIN = 0
EBN0_DB_MAX = 20
RUN_DIR = _make_run_dir(base_dir="runs")
EBNOS = np.arange(EBN0_DB_MIN, EBN0_DB_MAX, 1)

Model_lmmse = OFDMSystem(num_subcarriers=128, num_symbols=14, perfect_csi= False)

# route tf.print logs to a file too (you still see them on stdout if self.log_stream is None)
Model_lmmse.log_stream = f"file:{os.path.join(RUN_DIR, 'iterative_debug.log')}"

ber_plots = PlotBER("OFDM over 3GPP CDL")

# Capture the console table from PlotBER.simulate (your original behavior)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    ber, bler = ber_plots.simulate(
        Model_lmmse,
        ebno_dbs=EBNOS,
        batch_size=128,
        num_target_block_errors=100,
        legend="Iterative_LMMSE",
        soft_estimates=True,
        max_mc_iter=5000,
        show_fig=False
    )
_console = _buf.getvalue()

# Save CSV + console log + small metadata
csv_path = os.path.join(RUN_DIR, "ber_bler.csv")
log_path = os.path.join(RUN_DIR, "simulate_console.log")
meta_path = os.path.join(RUN_DIR, "meta.json")

_save_csv(csv_path, EBNOS, ber, bler)
with open(log_path, "w") as f:
    f.write(_console)

meta = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "ebnos": EBNOS.tolist(),
    "batch_size": 128,
    "num_target_block_errors": 100,
    "max_mc_iter": 5000,
    "legend": "LMMSE",
    "perfect_csi": bool(getattr(Model_lmmse, "perfect_csi", False)),
    "num_subcarriers": int(getattr(Model_lmmse, "num_effective_subcarriers", 0)),
    "num_symbols": int(getattr(Model_lmmse, "num_ofdm_symbols", 0)),
    "coderate": float(getattr(Model_lmmse, "coderate", 0.0)),
    "bits_per_symbol": int(getattr(Model_lmmse, "num_bits_per_symbol", 0)),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved CSV: {csv_path}")
print(f"Saved console log: {log_path}")
print(f"Saved metadata: {meta_path}")
print(f"Iterative LMMSE debug log: {os.path.join(RUN_DIR, 'iterative_debug.log')}")