import os
import numpy as np
import tensorflow as tf
import argparse
# -----------------------------------------------------------
# 1) Parse GPU argument BEFORE importing heavy TF modules
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate TDL correlation matrix R")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
args = parser.parse_args()

# -----------------------------------------------------------
# 2) Set visible device(s) and quiet TensorFlow logs
# -----------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------
# 3) Configure TensorFlow GPU memory growth
# -----------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using GPU: {gpus}")
    except RuntimeError as e:
        print("⚠️ TensorFlow GPU setup failed:", e)
else:
    print("⚠️ No GPU detected — running on CPU")

import sionna as sn
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import TDL

sn.phy.config.seed = 101  # reproducibility

# ================== CONFIG ==================
NUM_SUBCARRIERS   = 128
NUM_OFDM_SYMBOLS  = 14
CARRIER_SPACING   = 15e3
CARRIER_FREQ      = 2.6e9
CP_LEN            = 13
NUM_TX            = 1
NUM_STREAMS_TX    = 1
NUM_RX            = 1
NUM_RX_ANT        = 1

# Channel model and param
CHANNEL_MODEL   = "C"
TAU_RMS_S   = 150e-9
MIN_SPEED   = 10.0
MAX_SPEED   = None

NUM_REALIZATIONS = int(1e6)   # total channel draws
CHUNK_SIZE       = 1024       # increased for efficiency

# Output folder & filename
OUTPUT_DIR = "./R_mats"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FNAME = f"CDL_R_{CHANNEL_MODEL}_Nsc{NUM_SUBCARRIERS}_Nsym{NUM_OFDM_SYMBOLS}_Nsamp{NUM_REALIZATIONS}_CP{CP_LEN}_Speed{MIN_SPEED}.npz"
OUTPATH = os.path.join(OUTPUT_DIR, FNAME)
# ===========================================

def build_rg():
    """Build resource grid matching the OFDM system configuration"""
    return ResourceGrid(
        num_ofdm_symbols      = NUM_OFDM_SYMBOLS,
        fft_size              = NUM_SUBCARRIERS,
        subcarrier_spacing    = CARRIER_SPACING,
        num_tx                = NUM_TX,
        num_streams_per_tx    = NUM_STREAMS_TX,
        cyclic_prefix_length  = CP_LEN,
        num_guard_carriers    = [0, 0],
        dc_null               = False,
        pilot_pattern         = None,
        pilot_ofdm_symbol_indices = None
    )

def build_channel():
    """Build TDL channel model"""
    channel = TDL(
        model = CHANNEL_MODEL,
        delay_spread = TAU_RMS_S,
        carrier_frequency = CARRIER_FREQ,
        min_speed = MIN_SPEED,
        max_speed = MAX_SPEED
    )
    return channel

def compute_R(num_draws=NUM_REALIZATIONS, chunk=CHUNK_SIZE):
    rg = build_rg()
    channel = build_channel()
    chan = GenerateOFDMChannel(channel, rg, normalize_channel=True)

    # Total dimension size
    N = NUM_RX * NUM_RX_ANT * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS
    R_sum = tf.zeros((N, N), dtype=tf.complex64)
    
    # Track channel power for verification
    power_sum = 0.0

    remaining = num_draws
    processed = 0
    
    print(f"Computing R with {num_draws} realizations...")
    
    while remaining > 0:
        b = min(chunk, remaining)
        h_freq = chan(b)
        
        h_freq = tf.squeeze(h_freq, axis=[3, 4])
        h_vec = tf.reshape(h_freq, [b, -1])  # [batch, N]
    
        assert h_vec.shape[1] == N, f"Dimension mismatch: got {h_vec.shape[1]}, expected {N}"
        
        R_batch = tf.einsum('bn,bm->nm', h_vec, tf.math.conj(h_vec))
        R_sum += R_batch
        
        power_sum += tf.reduce_sum(tf.abs(h_vec)**2).numpy()
        
        remaining -= b
        processed += b
        
        if processed % (10 * chunk) == 0:
            print(f"  Processed {processed}/{num_draws} realizations...")

    R = R_sum / tf.cast(num_draws, tf.complex64)
    avg_power_per_re = power_sum / (num_draws * N)
    
    R_H = tf.linalg.adjoint(R)
    hermitian_error = tf.reduce_max(tf.abs(R - R_H)).numpy()
    
    print(f"Average channel power per RE: {avg_power_per_re:.6f}")
    print(f"Expected (normalized): ~1.0")
    print(f"Hermitian symmetry error: {hermitian_error:.2e}")   
    
    return R, avg_power_per_re

def main():
    R, avg_power = compute_R()
    R_np = R.numpy()

    # Save with metadata
    np.savez_compressed(
        OUTPATH,
        R=R_np,
        meta=dict(
            model=CHANNEL_MODEL,
            n_sc=NUM_SUBCARRIERS,
            n_sym=NUM_OFDM_SYMBOLS,
            n_rx=NUM_RX,
            n_rx_ant=NUM_RX_ANT,
            carrier_spacing=CARRIER_SPACING,
            carrier_frequency=CARRIER_FREQ,
            tau_rms=TAU_RMS_S,
            min_speed=MIN_SPEED,
            max_speed=MAX_SPEED,
            num_realizations=NUM_REALIZATIONS,
            normalize_channel=True,
            avg_channel_power_per_re=avg_power,
            vectorization_order="[rx, rx_ant, symbol, subcarrier]"
        ),
    )
    
    print(f"R shape: {R_np.shape}, dtype: {R_np.dtype}")
    print(f"R memory size: {R_np.nbytes / 1e6:.1f} MB")

    # Verification: load and check
    data = np.load(OUTPATH, allow_pickle=True)
    R_loaded = data["R"]
    meta = data["meta"].item()


if __name__ == "__main__":
    main()