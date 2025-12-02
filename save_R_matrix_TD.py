import os
import numpy as np
import tensorflow as tf
import argparse

# -----------------------------------------------------------
# 1) Parse GPU argument BEFORE importing heavy TF modules
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate correlation matrix R from CIR")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
parser.add_argument("--channel_type", type=str, default="TDL", choices=["TDL", "CDL"], 
                    help="Channel model type: TDL or CDL (default: TDL)")
parser.add_argument("--channel_model", type=str, default="A", 
                    help="Channel model variant (e.g., A, B, C, D, E) (default: A)")
parser.add_argument("--speed", type=float, default=10.0, 
                    help="Minimum user speed in m/s (default: 10.0)")
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
from sionna.phy.channel.tr38901 import TDL, CDL, AntennaArray
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.channel import OFDMChannel,GenerateOFDMChannel,ApplyTimeChannel,time_lag_discrete_time_channel,cir_to_time_channel,time_to_ofdm_channel


sn.phy.config.seed = 101  # reproducibility

# ================== CONFIG ==================
NUM_SUBCARRIERS   = 128
NUM_OFDM_SYMBOLS  = 14
CARRIER_SPACING   = 15e3
CARRIER_FREQ      = 2.6e9
NUM_TX            = 1
NUM_STREAMS_TX    = 1
NUM_RX            = 1
NUM_RX_ANT        = 1
CP_LEN            = 6

# Channel model configuration
CHANNEL_TYPE    = args.channel_type  # "TDL" or "CDL"
CHANNEL_MODEL   = args.channel_model  # "A", "B", "C", "D", "E"

# TDL-specific parameters
TDL_TAU_RMS_S   = 150e-9
TDL_MIN_SPEED   = args.speed
TDL_MAX_SPEED   = None

# CDL-specific parameters
CDL_DELAY_SPREAD = 150e-9
CDL_MIN_SPEED    = args.speed
CDL_MAX_SPEED    = None
CDL_DIRECTION    = "uplink"  # "uplink" or "downlink"

NUM_REALIZATIONS = int(1e6)
CHUNK_SIZE       = 512

# Output folder & filename
OUTPUT_DIR = "./R_mats_TD_011225"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if CHANNEL_TYPE == "TDL":
    FNAME = f"TDL_R_CIR_{CHANNEL_MODEL}_Nsc{NUM_SUBCARRIERS}_Nsym{NUM_OFDM_SYMBOLS}_Nsamp{NUM_REALIZATIONS}_CP{CP_LEN}_Speed{TDL_MIN_SPEED}.npz"
else:
    FNAME = f"CDL_R_CIR_{CHANNEL_MODEL}_Nsc{NUM_SUBCARRIERS}_Nsym{NUM_OFDM_SYMBOLS}_Nsamp{NUM_REALIZATIONS}__CP{CP_LEN}_Speed{CDL_MIN_SPEED}_{CDL_DIRECTION}.npz"

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
        cyclic_prefix_length  = CP_LEN,  # Not needed for CIR->freq conversion
        num_guard_carriers    = [0, 0],
        dc_null               = False,
        pilot_pattern         = None,
        pilot_ofdm_symbol_indices = None
    )

def build_antenna_arrays():
    """Build antenna arrays for CDL channel (SISO configuration)"""
    # User Terminal (UT) antenna array - single antenna
    ut_array = AntennaArray(
        num_rows=1,
        num_cols=NUM_RX_ANT,  # 1 for SISO
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=CARRIER_FREQ
    )
    
    # Base Station (BS) antenna array - single antenna
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=NUM_TX,  # 1 for SISO
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=CARRIER_FREQ
    )
    
    return ut_array, bs_array

def build_channel():
    """Build TDL or CDL channel model"""
    if CHANNEL_TYPE == "TDL":
        print(f"🔧 Building TDL-{CHANNEL_MODEL} channel model")
        print(f"   Delay spread: {TDL_TAU_RMS_S*1e9:.1f} ns")
        print(f"   Speed range: {TDL_MIN_SPEED} - {TDL_MAX_SPEED} m/s")
        
        channel = TDL(
            model = CHANNEL_MODEL,
            delay_spread = TDL_TAU_RMS_S,
            carrier_frequency = CARRIER_FREQ,
            min_speed = TDL_MIN_SPEED,
            max_speed = TDL_MAX_SPEED
        )
    
    elif CHANNEL_TYPE == "CDL":
        print(f"🔧 Building CDL-{CHANNEL_MODEL} channel model")
        print(f"   Delay spread: {CDL_DELAY_SPREAD*1e9:.1f} ns")
        print(f"   Speed range: {CDL_MIN_SPEED} - {CDL_MAX_SPEED} m/s")
        print(f"   Direction: {CDL_DIRECTION}")
        
        # Build antenna arrays for CDL
        ut_array, bs_array = build_antenna_arrays()
        print(f"   UT Array: {NUM_RX_ANT} antenna(s)")
        print(f"   BS Array: {NUM_TX} antenna(s)")
        
        channel = CDL(
            model = CHANNEL_MODEL,
            delay_spread = CDL_DELAY_SPREAD,
            carrier_frequency = CARRIER_FREQ,
            ut_array = ut_array,
            bs_array = bs_array,
            direction = CDL_DIRECTION,
            min_speed = CDL_MIN_SPEED,
            max_speed = CDL_MAX_SPEED
        )
    
    else:
        raise ValueError(f"Unknown channel type: {CHANNEL_TYPE}")
    
    return channel

def compute_R(num_draws=NUM_REALIZATIONS, chunk=CHUNK_SIZE):
    rg = build_rg()
    channel = build_channel()
    
    # Compute subcarrier frequencies for CIR to OFDM conversion
    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    
    # Total dimension size
    N = NUM_RX * NUM_RX_ANT * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS
    R_sum = tf.zeros((N, N), dtype=tf.complex64)
    
    # Track channel power for verification
    power_sum = 0.0

    remaining = num_draws
    processed = 0
    l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
    l_tot = l_max - l_min + 1

    num_delay = 1
    print(f"\n📊 Computing R from CIR with {num_draws} realizations...")
    
    while remaining > 0:
        b = min(chunk, remaining)
        a, tau = channel(b, rg.num_time_samples * num_delay + l_tot - 1, rg.bandwidth)
                
        h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min = l_min, l_max = l_max, normalize=True) 
        h_freq = time_to_ofdm_channel(h_time, rg, l_min)
        
        # Squeeze out tx dimensions (we have 1 TX, 1 stream)
        h_freq = tf.squeeze(h_freq, axis=[3, 4])
        # Now: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        
        # Vectorize: [batch, N] where N = rx * rx_ant * symbols * subcarriers
        h_vec = tf.reshape(h_freq, [b, -1])
    
        assert h_vec.shape[1] == N, f"Dimension mismatch: got {h_vec.shape[1]}, expected {N}"
        
        # Compute outer product and accumulate
        R_batch = tf.einsum('bn,bm->nm', h_vec, tf.math.conj(h_vec))
        R_sum += R_batch
        
        power_sum += tf.reduce_sum(tf.abs(h_vec)**2).numpy()
        
        remaining -= b
        processed += b
        
        if processed % (10 * chunk) == 0:
            print(f"  Processed {processed}/{num_draws} realizations...")

    R = R_sum / tf.cast(num_draws, tf.complex64)
    avg_power_per_re = power_sum / (num_draws * N)
    
    # Check Hermitian symmetry
    R_H = tf.linalg.adjoint(R)
    hermitian_error = tf.reduce_max(tf.abs(R - R_H)).numpy()
    
    print(f"\n✅ Computation complete!")
    print(f"   Average channel power per RE: {avg_power_per_re:.6f}")
    print(f"   Expected (normalized): ~1.0")
    print(f"   Hermitian symmetry error: {hermitian_error:.2e}")   
    
    return R, avg_power_per_re

def main():
    print("="*60)
    print(f"Channel Correlation Matrix Generator (CIR Method)")
    print(f"Channel Type: {CHANNEL_TYPE}-{CHANNEL_MODEL}")
    print(f"Configuration: SISO ({NUM_TX}x{NUM_RX_ANT})")
    print("="*60)
    
    R, avg_power = compute_R()
    R_np = R.numpy()

    # Build metadata dictionary
    meta_dict = dict(
        channel_type=CHANNEL_TYPE,
        model=CHANNEL_MODEL,
        n_sc=NUM_SUBCARRIERS,
        n_sym=NUM_OFDM_SYMBOLS,
        n_rx=NUM_RX,
        n_rx_ant=NUM_RX_ANT,
        n_tx=NUM_TX,
        carrier_spacing=CARRIER_SPACING,
        carrier_frequency=CARRIER_FREQ,
        num_realizations=NUM_REALIZATIONS,
        normalize_channel=True,
        avg_channel_power_per_re=avg_power,
        vectorization_order="[rx, rx_ant, symbol, subcarrier]",
        generation_method="CIR_to_OFDM"
    )
    
    # Add channel-specific parameters
    if CHANNEL_TYPE == "TDL":
        meta_dict.update(dict(
            tau_rms=TDL_TAU_RMS_S,
            min_speed=TDL_MIN_SPEED,
            max_speed=TDL_MAX_SPEED
        ))
    else:  # CDL
        meta_dict.update(dict(
            delay_spread=CDL_DELAY_SPREAD,
            min_speed=CDL_MIN_SPEED,
            max_speed=CDL_MAX_SPEED,
            direction=CDL_DIRECTION,
            antenna_pattern="38.901",
            polarization="single"
        ))
    
    # Save with metadata
    np.savez_compressed(
        OUTPATH,
        R=R_np,
        meta=meta_dict
    )
    
    print(f"\n💾 Saved to: {OUTPATH}")
    print(f"   R shape: {R_np.shape}, dtype: {R_np.dtype}")
    print(f"   R memory size: {R_np.nbytes / 1e6:.1f} MB")

    # Verification: load and check
    data = np.load(OUTPATH, allow_pickle=True)
    R_loaded = data["R"]
    meta = data["meta"].item()
    print(f"✅ Verified: R loaded successfully")
    print(f"   Channel type: {meta['channel_type']}-{meta['model']}")
    print(f"   Generation method: {meta['generation_method']}")
    print("="*60)

if __name__ == "__main__":
    main()