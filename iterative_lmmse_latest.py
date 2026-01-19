# -*- coding: utf-8 -*-
"""Iterative LMMSE Estimator - Corrected Implementation"""

import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna as sn
import sionna.phy
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import Model

sionna.phy.config.seed = 42

from sionna.phy.ofdm import (ResourceGrid, RemoveNulledSubcarriers, ResourceGridMapper, 
                             OFDMModulator, OFDMDemodulator, LSChannelEstimator, 
                             PilotPattern, ZFEqualizer, LMMSEEqualizer)
from sionna.phy.mapping import BinarySource, Mapper, Demapper, SymbolDemapper, Constellation
from sionna.phy.channel import (OFDMChannel, GenerateOFDMChannel, cir_to_ofdm_channel, 
                                ApplyOFDMChannel, subcarrier_frequencies, ApplyTimeChannel, 
                                time_lag_discrete_time_channel, cir_to_time_channel, 
                                time_to_ofdm_channel)
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
    def __init__(self, num_subcarriers: int, num_symbols: int, perfect_csi):
        super().__init__()
        
        # Resource grid setup
        self.num_effective_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_symbols
        self.pilot_idx = [2, 11]
        self.carrier_spacing = 15e3
        self.carrier_frequency = 2.6e9
        self.cyclic_prefix_length = 6

        # SISO configuration
        NUM_UT = 1
        NUM_BS = 1
        NUM_UT_ANT = 1
        NUM_BS_ANT = 1
        NUM_STREAMS_PER_TX = NUM_UT_ANT

        RX_TX_ASSOCIATION = np.array([[1]])
        STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

        # Create pilot pattern
        mask = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, self.num_ofdm_symbols, 
                        self.num_effective_subcarriers])
        sym_idx_1 = self.pilot_idx[0]
        sym_idx_2 = self.pilot_idx[1]
        mask[0, 0, sym_idx_1, 1::2] = True
        mask[0, 0, sym_idx_2, 1::2] = True
        num_pilot_symbols = int(np.sum(mask[0, 0]))
        pilots = np.zeros([NUM_UT, NUM_STREAMS_PER_TX, num_pilot_symbols], np.complex64)
        pilots[0, 0, :] = (1+1j)/np.sqrt(2)
        pilot_pattern = PilotPattern(mask, pilots)

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.num_effective_subcarriers,
            subcarrier_spacing=self.carrier_spacing,
            num_tx=NUM_UT,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=[0, 0],
            dc_null=False,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=self.pilot_idx
        )

        self.rm = RemoveNulledSubcarriers(self.rg)

        # Coding parameters
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

        assert pad_syms >= 0, "Grid too small for the given num_codewords"

        self.pad_bits = pad_syms * self.num_bits_per_symbol

        # Instantiate blocks
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=False, return_infobits=False, num_iter=20)
        self.dec_bits = LDPC5GDecoder(self.encoder, hard_out=True, return_infobits=True, num_iter=20)

        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)

        self.rg_mapper = ResourceGridMapper(self.rg)
        self._ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")

        self.tau_rms = 150e-9

        self.TDL = TDL(model="A",
                      delay_spread=self.tau_rms,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=40.0,
                      max_speed=None)

        self.channel = OFDMChannel(self.TDL,
                                   self.rg,
                                   normalize_channel=True,
                                   return_channel=True)

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
        self._demodulator = OFDMDemodulator(self.num_effective_subcarriers, self._l_min, 
                                           self.cyclic_prefix_length)

    def get_pilot_data(self, y):
        """Extract pilot information from received signal"""
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

    def get_prior_dist(self, llr_prior):
        """
        Compute prior symbol distribution from bit LLRs using equation (11).
        
        Args:
            llr_prior: [batch, num_data_symbols, num_bits_per_symbol] - prior LLRs
        
        Returns:
            P_sym: [batch, num_data_symbols, M] - prior probabilities
        """
        # Verify input shape
        tf.debugging.assert_rank(llr_prior, 3, message="llr_prior must be rank 3")
        tf.debugging.assert_equal(
            tf.shape(llr_prior)[1],
            self.num_data_symbols,
            message="llr_prior must have num_data_symbols in dim 1"
        )
        
        M = 2**self.num_bits_per_symbol  # constellation size
        
        # Generate bit labels for all constellation symbols
        indices = tf.range(M, dtype=tf.int32)  # [M]
        
        # Extract bit representation
        bit_labels = []
        for i in range(self.num_bits_per_symbol):
            bit_i = tf.bitwise.bitwise_and(
                tf.bitwise.right_shift(indices, self.num_bits_per_symbol - 1 - i),
                1
            )
            bit_labels.append(bit_i)
        
        bit_labels = tf.stack(bit_labels, axis=-1)  # [M, m]
        bit_labels = tf.cast(bit_labels, tf.float32)
        
        # Compute logits: sum_i c_u^(i) * LLR_P(k,i)
        logits = tf.einsum('bsm,um->bsu', llr_prior, bit_labels)
        
        # Apply softmax (equation 12)
        P_sym = tf.nn.softmax(logits, axis=-1)  # [batch, num_symbols, M]
        
        return P_sym

    def lmmse_est_iterative(self, y_rg, no, priors_data, R_matrix=None, reg_strength_override: float = None, debug: bool = False):
        """Iterative LMMSE channel estimator using prior symbol distributions.

        Implements equations (13), (14) and (15) from the paper in a
        numerically-stable way. The implementation forms the observation
        covariance Cov_y = R · E[x x^H] + sigma^2 I, regularizes it when
        necessary, solves the linear system and computes the standard
        LMMSE estimate and error covariance diagonal.

        Args:
            y_rg: Received resource grid [B, ...] (will be flattened internally)
            no: Noise variance (scalar)
            priors_data: [batch, num_data_symbols, M] - prior probabilities

        Returns:
            h_hat: [batch, N] Channel estimate
            err_var: [batch, N] Error variance per RE
        """
        # Load channel correlation matrix R (unless provided)
        if R_matrix is None:
            data = np.load("/home/ha26227/neural_precoder-reproduce/TDL_R_A_Nsc128_Nsym14_Nsamp1000000_Speed40.0.npz",
                           allow_pickle=True)
            R_loaded = data["R"]
            R = tf.convert_to_tensor(R_loaded, dtype=tf.complex64)  # [N, N]
            # Ensure Hermitian and numerically positive-definite: symmetrize and add tiny diagonal regularization
            R = 0.5 * (R + tf.linalg.adjoint(R))
            max_diag_R = tf.reduce_max(tf.math.real(tf.linalg.diag_part(R)))
            reg_R = tf.cast(1e-6, tf.complex64) * tf.cast(max_diag_R, tf.complex64)
            R = R + reg_R * tf.eye(int(R.shape[0]), dtype=tf.complex64)
        else:
            # allow passing either numpy array or TF tensor
            R = tf.convert_to_tensor(R_matrix, dtype=tf.complex64)

        N_total_static = int(R.shape[0])

        # Flatten received resource grid to [B, N]
        y = tf.reshape(y_rg, [self.batch_size, -1])  # [B, N]

        # Get pilot information (P contains pilot symbols, zeros elsewhere)
        P, _, pilot_idx = self.get_pilot_data(y)
        P = tf.cast(P, tf.complex64)
        p = tf.reshape(P, [-1])  # [N]

        # Data positions are where P == 0
        pilot_idx = tf.reshape(pilot_idx, [-1])
        pilot_idx = tf.cast(pilot_idx, tf.int32)
        data_mask = tf.equal(p, tf.complex(0.0, 0.0))
        data_idx = tf.where(data_mask)[:, 0]
        data_idx = tf.cast(data_idx, tf.int32)

        # Validate priors shape
        tf.debugging.assert_equal(tf.shape(priors_data)[1], tf.shape(data_idx)[0],
                                  message="priors_data dimension mismatch with data positions")

        # Constellation points
        const_points = tf.cast(self.mapper.constellation.points, tf.complex64)  # [M]

        # Compute E[x_k] and E[|x_k|^2] for data symbols
        mu_x_data = tf.einsum('bnu,u->bn', tf.cast(priors_data, tf.complex64), const_points)
        const_power = tf.cast(tf.abs(const_points) ** 2, tf.float32)
        e2_x_data = tf.einsum('bnu,u->bn', priors_data, const_power)

        # Build full-length vectors (including pilots)
        mu_x_full = tf.zeros([self.batch_size, N_total_static], dtype=tf.complex64)
        e2_x_full = tf.zeros([self.batch_size, N_total_static], dtype=tf.float32)

        # Scatter data expectations into full vectors
        b_idx = tf.range(self.batch_size, dtype=tf.int32)[:, tf.newaxis]
        data_indices = b_idx * N_total_static + data_idx[tf.newaxis, :]
        data_indices = tf.reshape(data_indices, [-1, 1])

        mu_x_full_flat = tf.reshape(mu_x_full, [-1])
        mu_x_data_flat = tf.reshape(mu_x_data, [-1])
        mu_x_full_flat = tf.tensor_scatter_nd_update(mu_x_full_flat, data_indices, mu_x_data_flat)
        mu_x_full = tf.reshape(mu_x_full_flat, [self.batch_size, N_total_static])

        e2_x_full_flat = tf.reshape(e2_x_full, [-1])
        e2_x_data_flat = tf.reshape(e2_x_data, [-1])
        e2_x_full_flat = tf.tensor_scatter_nd_update(e2_x_full_flat, data_indices, e2_x_data_flat)
        e2_x_full = tf.reshape(e2_x_full_flat, [self.batch_size, N_total_static])

        # Insert pilot symbols into expectations (pilots are deterministic)
        pilot_indices = b_idx * N_total_static + pilot_idx[tf.newaxis, :]
        pilot_indices = tf.reshape(pilot_indices, [-1, 1])

        p_pilots = tf.gather(p, pilot_idx)
        p_pilots_batch = tf.tile(p_pilots[tf.newaxis, :], [self.batch_size, 1])
        p_pilots_flat = tf.reshape(p_pilots_batch, [-1])

        mu_x_full_flat = tf.reshape(mu_x_full, [-1])
        mu_x_full_flat = tf.tensor_scatter_nd_update(mu_x_full_flat, pilot_indices, p_pilots_flat)
        mu_x_full = tf.reshape(mu_x_full_flat, [self.batch_size, N_total_static])

        p_power = tf.cast(tf.abs(p_pilots) ** 2, tf.float32)
        p_power_batch = tf.tile(p_power[tf.newaxis, :], [self.batch_size, 1])
        p_power_flat = tf.reshape(p_power_batch, [-1])

        e2_x_full_flat = tf.reshape(e2_x_full, [-1])
        e2_x_full_flat = tf.tensor_scatter_nd_update(e2_x_full_flat, pilot_indices, p_power_flat)
        e2_x_full = tf.reshape(e2_x_full_flat, [self.batch_size, N_total_static])

        # Build E[x x^H] per equation (14)
        outer_mu = tf.einsum('bi,bk->bik', mu_x_full, tf.math.conj(mu_x_full))  # [B,N,N]
        mask_offdiag = 1.0 - tf.eye(N_total_static, dtype=tf.float32)
        outer_mu = outer_mu * tf.cast(mask_offdiag, tf.complex64)[tf.newaxis, :, :]
        E_xxH = outer_mu + tf.linalg.diag(tf.cast(e2_x_full, tf.complex64))  # [B,N,N]

        # --- Matrix LMMSE (full) with stronger regularization for stability ---
        R_batch = R[tf.newaxis, :, :]  # [1, N, N]

        # Optionally apply shrinkage to E_xxH if covariance appears ill-conditioned
        # Compute a simple condition estimate from diag(E_xxH)
        diag_E = tf.linalg.diag_part(E_xxH)  # [B, N]
        max_diag_E = tf.reduce_max(tf.math.real(diag_E), axis=-1, keepdims=True)
        min_diag_E = tf.reduce_min(tf.math.real(diag_E), axis=-1, keepdims=True)
        cond_E = max_diag_E / (min_diag_E + 1e-12)
        # shrink_lambda increases with ill-conditioning; cap at 0.95
        shrink_lambda_tf = tf.clip_by_value(cond_E / 1e5, 0.0, 0.95)
        shrink_lambda = tf.reduce_mean(tf.cast(shrink_lambda_tf, tf.float32))
        if debug:
            print("shrink_lambda chosen:", float(shrink_lambda))
        if shrink_lambda > 1e-6:
            # build diagonal-only version of E_xxH per batch
            diag_matrix = tf.linalg.diag(tf.linalg.diag_part(E_xxH))
            # cast shrink_lambda to complex for mixing with complex matrices
            sl_c = tf.cast(shrink_lambda, tf.complex64)
            E_xxH_use = tf.cast(1.0, tf.complex64) * E_xxH - sl_c * E_xxH + sl_c * diag_matrix
        else:
            E_xxH_use = E_xxH

        # Covariance of observations: R · E[x x^H] + sigma^2 I
        Cov_y = tf.matmul(R_batch, E_xxH_use)  # [B, N, N]
        Cov_y = Cov_y + tf.cast(no, tf.complex64) * tf.eye(N_total_static, dtype=tf.complex64)

        # Symmetrize
        Cov_y = 0.5 * (Cov_y + tf.linalg.adjoint(Cov_y))

        if debug:
            # Diagnostics: diag and conditioning of Cov_y
            diag_cov = tf.linalg.diag_part(Cov_y)
            diag_real = tf.math.real(diag_cov)
            print("Cov_y diag (real) min/max/mean:", tf.reduce_min(diag_real).numpy(),
                  tf.reduce_max(diag_real).numpy(), tf.reduce_mean(diag_real).numpy())
            try:
                s = tf.linalg.svd(Cov_y, compute_uv=False)
                # s shape [B, N]
                s_min = tf.reduce_min(s, axis=-1)
                s_max = tf.reduce_max(s, axis=-1)
                print("Cov_y SVD min/max per batch:", s_min.numpy(), s_max.numpy())
            except Exception as e:
                print("SVD failed:", e)

        # Stronger adaptive regularization (scale with max diagonal)
        diag_cov = tf.linalg.diag_part(Cov_y)
        max_diag = tf.reduce_max(tf.math.real(diag_cov), axis=-1, keepdims=True)
        # Adaptive regularization: compute a simple condition estimate from diag(Cov_y)
        # and set a regularization strength that increases with ill-conditioning.
        # Allow the caller to override via reg_strength_override.
        if reg_strength_override is None:
            # min diagonal (real) per batch
            min_diag = tf.reduce_min(tf.math.real(diag_cov), axis=-1, keepdims=True)
            # condition estimate (ratio). Add tiny eps to avoid divide-by-zero.
            cond_est = max_diag / (min_diag + 1e-12)
            # Map cond_est to a regularization in [1e-3, 1.0] using a scaling factor.
            # Scaling factor chosen so that cond_est ~ 1e4 -> reg ~ 1.0; cond_est ~ 1e2 -> reg ~ 1e-2.
            reg_strength_tf = tf.clip_by_value(cond_est / 1e4, 1e-3, 1.0)
            reg_strength = tf.reduce_mean(tf.cast(reg_strength_tf, tf.float32))
        else:
            reg_strength = float(reg_strength_override)
        regularization = tf.cast(reg_strength, tf.complex64) * tf.cast(max_diag, tf.complex64)
        Cov_y = Cov_y + regularization[:, tf.newaxis] * tf.eye(N_total_static, dtype=tf.complex64)

        # Solve linear system v = Cov_y^{-1} y
        y_expanded = tf.expand_dims(y, axis=-1)  # [B, N, 1]
        v = tf.linalg.solve(Cov_y, y_expanded)  # [B, N, 1]
        v = v[:, :, 0]  # [B, N]

        if debug:
            v_norm = tf.norm(v, axis=-1)
            print("v norm per batch:", v_norm.numpy())

        # Numerator: R·diag(mu_x)^H (multiply each column j of R by conj(mu_x[j]))
        mu_x_conj = tf.math.conj(mu_x_full)  # [B, N]
        R_mu_conj = R[tf.newaxis, :, :] * mu_x_conj[:, tf.newaxis, :]  # [B, N, N]

        # Final estimate: h_hat = R_mu_conj · v
        h_hat = tf.einsum('bij,bj->bi', R_mu_conj, v)  # [B, N]

        # Error covariance diagonal using matrix expression
        mu_R = mu_x_full[:, :, tf.newaxis] * R[tf.newaxis, :, :]  # diag(mu_x)·R  [B,N,N]
        Z = tf.linalg.solve(Cov_y, mu_R)  # [B,N,N]
        R_tilde = R[tf.newaxis, :, :] - tf.matmul(R_mu_conj, Z)  # [B,N,N]

        if debug:
            R_tilde_diag = tf.math.real(tf.linalg.diag_part(R_tilde))
            print("R_tilde diag min/max/mean before adding noise:",
                  tf.reduce_min(R_tilde_diag).numpy(), tf.reduce_max(R_tilde_diag).numpy(),
                  tf.reduce_mean(R_tilde_diag).numpy())

        err_var = tf.math.real(tf.linalg.diag_part(R_tilde))
        # Add noise variance and ensure error variance is not smaller than noise
        err_var = err_var + tf.cast(no, err_var.dtype)
        err_var = tf.maximum(err_var, tf.cast(no, err_var.dtype))
        # Also enforce a small absolute floor to avoid exactly zero
        err_var = tf.maximum(err_var, tf.cast(1e-6, err_var.dtype))

        if debug:
            print("err_var after adding noise and flooring min/max/mean:",
                  tf.reduce_min(err_var).numpy(), tf.reduce_max(err_var).numpy(),
                  tf.reduce_mean(err_var).numpy())

        return h_hat, err_var

    @tf.function
    def __call__(self, batch_size: int, ebno_db: float):
        self.batch_size = batch_size
        
        # Eb/N0 to N0
        no = ebnodb2no(ebno_db,
                      num_bits_per_symbol=self.num_bits_per_symbol,
                      coderate=self.coderate,
                      resource_grid=self.rg)
        
        # Encoding
        bits = self.binary_source([batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, 
                                    self.num_codewords * self.k])
        bits = tf.reshape(bits, [batch_size, self.num_codewords, self.k])
        
        c = self.encoder(bits)
        c = tf.reshape(c, [batch_size, self.num_codewords * self.n])
        
        pad_bits = self.binary_source([batch_size, self.pad_bits])
        coded_frame = tf.concat([c, pad_bits], axis=-1)
        
        x_syms = self.mapper(coded_frame)
        x_syms = tf.reshape(x_syms, [batch_size, self.rg.num_tx, self.rg.num_streams_per_tx, 
                                      self.num_data_symbols])
        x_rg = self.rg_mapper(x_syms)
        
        # Channel simulation
        a, tau = self.TDL(self.batch_size, 
                          self.rg.num_time_samples * 1 + self._l_tot - 1, 
                          self.rg.bandwidth)
        
        h_time = cir_to_time_channel(self.rg.bandwidth, a, tau, 
                                      l_min=self._l_min, l_max=self._l_max, normalize=True)
        h_freq = time_to_ofdm_channel(h_time, self.rg, self._l_min)
        
        x_time = self._modulator(x_rg)
        y_time = self._channel_time(x_time, h_time, no)
        y_rg = self._demodulator(y_time)
        
        # Turbo iterations
        NUM_TURBO_ITER = 4
        
        # Initialize: zero extrinsic LLRs from decoder for CODED BITS ONLY
        llr_ext_dec = tf.zeros([self.batch_size, self.num_codewords * self.n], dtype=tf.float32)
        
        for iter_idx in range(NUM_TURBO_ITER):
            # ===== STEP 1: Prepare priors for channel estimation =====
            if self.pad_bits > 0:
                pad_llr = tf.zeros([self.batch_size, self.pad_bits], dtype=tf.float32)
                llr_prior_all_bits = tf.concat([llr_ext_dec, pad_llr], axis=-1)
            else:
                llr_prior_all_bits = llr_ext_dec
            
            # Reshape to [batch, num_data_symbols, num_bits_per_symbol]
            llr_prior_for_est = tf.reshape(llr_prior_all_bits, 
                                           [self.batch_size, self.num_data_symbols, 
                                            self.num_bits_per_symbol])
            
            # ===== STEP 2: Compute prior distribution (Equation 11) =====
            priors_data = self.get_prior_dist(llr_prior_for_est)
            
            # ===== STEP 3: Channel estimation (Equations 13-15) =====
            if self.perfect_csi:
                h_hat = self.rm(h_freq)
                err_var = tf.fill(tf.shape(h_hat), tf.cast(no, tf.float32))
            else:
                h_hat, err_var = self.lmmse_est_iterative(y_rg, no, priors_data)
                
                # Reshape to resource grid dimensions
                target_shape = tf.shape(h_freq)
                h_hat = tf.reshape(h_hat, target_shape)
                err_var = tf.reshape(err_var, target_shape)
            
            # ===== STEP 4: Equalization =====
            no_vec = tf.fill([batch_size], no)
            x_hat, no_eff = self.zf_equ(y_rg, h_hat, err_var, no_vec)
            no_eff = expand_to_rank(no_eff, tf.rank(x_hat))
            
            # ===== STEP 5: Demapping (Equation 16) =====
            llr_prior_for_demap = tf.reshape(llr_prior_all_bits,
                                             [self.batch_size, self.rg.num_tx, 
                                              self.rg.num_streams_per_tx, self.num_data_symbols,
                                              self.num_bits_per_symbol])
            
            # Demapper computes LLR(k,i) which includes the prior
            llr_demap = self.demapper(x_hat, no_eff, prior=llr_prior_for_demap)
            llr_demap = tf.squeeze(llr_demap, axis=[1, 2])
            
            # Flatten and extract coded bits
            llr_demap_flat = tf.reshape(llr_demap, [self.batch_size, -1])
            llr_demap_coded = llr_demap_flat[:, :self.num_codewords * self.n]
            
            # ===== STEP 6: Compute extrinsic from demapper =====
            llr_ext_demap = llr_demap_coded - llr_ext_dec
            llr_ext_demap = tf.reshape(llr_ext_demap, [self.batch_size, self.num_codewords, self.n])
            
            # ===== STEP 7: Decode =====
            llr_dec_out = self.decoder(llr_ext_demap)
            
            if iter_idx == NUM_TURBO_ITER - 1:
                u_hat = self.dec_bits(llr_ext_demap)
            
            # ===== STEP 8: Compute extrinsic from decoder =====
            llr_dec_out_flat = tf.reshape(llr_dec_out, [self.batch_size, self.num_codewords * self.n])
            llr_ext_demap_flat = tf.reshape(llr_ext_demap, [self.batch_size, self.num_codewords * self.n])
            llr_ext_dec = llr_dec_out_flat - llr_ext_demap_flat
        
        u = tf.reshape(bits, [self.batch_size, self.num_codewords, self.k])
        
        return u, u_hat


# ===== Main Execution =====
if __name__ == "__main__":
    Model_lmmse = OFDMSystem(num_subcarriers=128, num_symbols=14, perfect_csi=False)

    EBN0_DB_MIN = 0
    EBN0_DB_MAX = 20

    ber_plots = PlotBER("OFDM over 3GPP TDL-A")

    RUN_DIR = _make_run_dir(base_dir="runs")
    EBNOS = np.arange(EBN0_DB_MIN, EBN0_DB_MAX, 2)

    # Capture console output
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        ber, bler = ber_plots.simulate(
            Model_lmmse,
            ebno_dbs=EBNOS,
            batch_size=128,
            num_target_block_errors=100,
            legend="IEDD-LMMSE",
            soft_estimates=True,
            max_mc_iter=5000,
            show_fig=False
        )
    _console = _buf.getvalue()

    # Save results
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
        "legend": "IEDD-LMMSE",
        "perfect_csi": bool(Model_lmmse.perfect_csi),
        "num_subcarriers": int(Model_lmmse.num_effective_subcarriers),
        "num_symbols": int(Model_lmmse.num_ofdm_symbols),
        "coderate": float(Model_lmmse.coderate),
        "bits_per_symbol": int(Model_lmmse.num_bits_per_symbol),
        "num_turbo_iter": 4,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {RUN_DIR}")
    print(f"{'='*60}")
    print(f"CSV file: {csv_path}")
    print(f"Console log: {log_path}")
    print(f"Metadata: {meta_path}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("BER/BLER Summary:")
    print("-" * 60)
    for e, b, bl in zip(EBNOS, ber, bler):
        print(f"Eb/N0 = {e:5.1f} dB: BER = {b:.6e}, BLER = {bl:.6e}")
    print("-" * 60)