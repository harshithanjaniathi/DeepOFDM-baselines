# # -*- coding: utf-8 -*-
# import os
# import numpy as np
# import tensorflow as tf
# import argparse
# # -----------------------------------------------------------
# # 1) Parse GPU argument BEFORE importing heavy TF modules
# # -----------------------------------------------------------
# parser = argparse.ArgumentParser(description="Generate CDL correlation matrix R")
# parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
# args = parser.parse_args()

# # -----------------------------------------------------------
# # 2) Set visible device(s) and quiet TensorFlow logs
# # -----------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # -----------------------------------------------------------
# # 3) Configure TensorFlow GPU memory growth
# # -----------------------------------------------------------
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"✅ Using GPU: {gpus}")
#     except RuntimeError as e:
#         print("⚠️ TensorFlow GPU setup failed:", e)
# else:
#     print("⚠️ No GPU detected — running on CPU")

# import sionna as sn
# from sionna.phy.ofdm import ResourceGrid
# from sionna.phy.channel import GenerateOFDMChannel
# from sionna.phy.channel.tr38901 import CDL, AntennaArray, TDL

# sn.phy.config.seed = 101  # reproducibility

# # ================== CONFIG ==================
# NUM_SUBCARRIERS   = 128
# NUM_OFDM_SYMBOLS  = 14
# CARRIER_SPACING   = 15e3
# CARRIER_FREQ      = 2.6e9
# CP_LEN            = 6
# NUM_TX            = 1
# NUM_STREAMS_TX    = 1

# # Channel model and param
# CHANNEL      = "TDL"
# CHANNEL_MODEL   = "A"
# TAU_RMS_S   = 150e-9
# MIN_SPEED   = 40.0
# MAX_SPEED   = None
# DIRECTION   = "uplink"

# # Monte-Carlo
# NUM_REALIZATIONS = int(1e6)   # total channel draws
# CHUNK_SIZE       = 512        # process in chunks to be memory-safe

# # Output folder & filename
# OUTPUT_DIR = "./R_mat_latest"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# FNAME = f"{CHANNEL}_R_{CHANNEL_MODEL}_Nsc{NUM_SUBCARRIERS}_Nsym{NUM_OFDM_SYMBOLS}_Nsamp{NUM_REALIZATIONS}_Speed{MIN_SPEED}.npz"
# OUTPATH = os.path.join(OUTPUT_DIR, FNAME)
# # ===========================================

# def build_rg():
#     # No pilots needed for R (it’s a channel statistic), keep a simple SISO grid
#     return ResourceGrid(
#         num_ofdm_symbols      = NUM_OFDM_SYMBOLS,
#         fft_size              = NUM_SUBCARRIERS,
#         subcarrier_spacing    = CARRIER_SPACING,
#         num_tx                = NUM_TX,
#         num_streams_per_tx    = NUM_STREAMS_TX,
#         cyclic_prefix_length  = CP_LEN,
#         num_guard_carriers    = [0, 0],
#         dc_null               = False,
#         pilot_pattern         = None,          # not used
#         pilot_ofdm_symbol_indices = None
#     )

# def build_channel():
#     if CHANNEL == "CDL":
#         ut = AntennaArray(num_rows=1, num_cols=1, polarization="single",
#                       polarization_type="V", antenna_pattern="38.901",
#                       carrier_frequency=CARRIER_FREQ)
#         bs = AntennaArray(num_rows=1, num_cols=1, polarization="single",
#                       polarization_type="V", antenna_pattern="38.901",
#                       carrier_frequency=CARRIER_FREQ)
#         channel = CDL(model = CHANNEL_MODEL,
#               delay_spread = TAU_RMS_S,
#               carrier_frequency=CARRIER_FREQ,
#               ut_array=ut,
#               bs_array=bs,
#               direction=DIRECTION,
#               min_speed=MIN_SPEED,
#               max_speed=MAX_SPEED)
#     elif CHANNEL == "TDL":
#         channel = TDL(model = CHANNEL_MODEL,
#               delay_spread = TAU_RMS_S,
#               carrier_frequency=CARRIER_FREQ,
#               min_speed=MIN_SPEED,
#               max_speed=MAX_SPEED)
#     return channel

# def compute_R(num_draws=NUM_REALIZATIONS, chunk=CHUNK_SIZE):
#     rg  = build_rg()
#     channel = build_channel()
#     chan = GenerateOFDMChannel(channel, rg, normalize_channel=True)

#     N = NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS
#     R_sum = tf.zeros((N, N), dtype=tf.complex64)

#     remaining = num_draws
#     while remaining > 0:
#         b = min(chunk, remaining)
#         # h_freq shape: [B, 1, 1, 1, 1, nS, nT] (Sionna); flatten to [B, N]
#         h_freq = chan(b)
#         h_vec  = tf.reshape(h_freq, [b, -1])                 # [b, N]
#         # accumulate conjugate outer products across the batch
#         R_sum += tf.einsum('bn,bm->nm', tf.math.conj(h_vec), h_vec)
#         remaining -= b

#     R = R_sum / tf.cast(num_draws, tf.complex64)
#     return R

# def main():
#     R = compute_R()
#     R_np = R.numpy()

#     np.savez_compressed(
#         OUTPATH,
#         R=R_np,
#         meta=dict(
#             model=CHANNEL_MODEL,
#             n_sc=NUM_SUBCARRIERS,
#             n_sym=NUM_OFDM_SYMBOLS,
#             carrier_spacing=CARRIER_SPACING,
#             carrier_frequency=CARRIER_FREQ,
#             tau_rms=TAU_RMS_S,
#             min_speed=MIN_SPEED,
#             max_speed=MAX_SPEED,
#             num_realizations=NUM_REALIZATIONS,
#             normalize_channel=True,
#         ),
#     )
#     print("Saved R to:", OUTPATH)
#     print("R shape:", R_np.shape, "dtype:", R_np.dtype)

#     # quick read-back check
#     data = np.load(OUTPATH, allow_pickle=True)
#     R_loaded = data["R"]
#     print("Reload OK & equal?", np.allclose(R_np, R_loaded))

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-