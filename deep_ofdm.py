"""
OFDM Neural Receiver System
A clean implementation supporting multiple receiver architectures:
- Neural Receiver (NRX): Neural receiver only
- Neural Modulation (NM): Neural precoder + neural receiver
- Symbol-level Information Precoding (SIP): SIP precoder + neural receiver
- Neural Modulation with SIP (NM_SIP): Both neural precoding and SIP
"""

import os
import time
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure GPU
def setup_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU {gpu_id} configured successfully")
        except RuntimeError as e:
            logging.error(f"GPU setup error: {e}")

import sionna
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper
from sionna.ofdm import LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder
from sionna.channel.tr38901 import AntennaArray, TDL
from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel, time_to_ofdm_channel
from sionna.channel import ApplyTimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber, insert_dims, expand_to_rank
from tensorflow.keras.optimizers.schedules import CosineDecay

# Import neural network models
from models_end import (NeuralPrecoder, NeuralPrecoder1D, 
                        NeuralPrecoder_1_layer_linear, NeuralPrecoder_1_layer_relu,
                        NeuralReceiver, NeuralReceiverPilot, NeuralReceiverPilotChannel,
                        NeuralPrecoderSIP, NeuralReceiverSIP)


class OFDMConfig:
    """Configuration class for OFDM system parameters"""
    def __init__(self, args):
        # Parse and store all arguments
        self.args = args
        
        # Basic OFDM parameters
        self.num_subcarriers = args.num_subcarriers
        self.subcarrier_spacing = args.subcarrier_spacing
        self.fft_size = args.num_subcarriers
        self.cyclic_prefix_length = args.cyclic_prefix_length
        self.num_ofdm_symbols = 14
        
        # Antenna configuration
        self.num_ut_ant = args.num_ut_ant
        self.num_bs_ant = args.num_bs_ant
        self.num_streams_per_tx = self.num_ut_ant
        
        # Pilot configuration
        self.pilot_ofdm_symbol_indices = self._parse_pilot_indices(args.pilot_loc)
        self.has_pilots = self.pilot_ofdm_symbol_indices is not None
        
        # Receiver type: 'pilot' or 'pilot_channel' (includes h_freq)
        self.receiver_type = getattr(args, 'receiver_type', 'pilot')
        
        # Channel parameters
        self.carrier_frequency = (args.ul_center_frequency + args.dl_center_frequency) / 2
        self.direction = args.direction
        self.cdl_model = args.cdl_model
        self.delay_spread = args.delay_spread
        self.min_speed = args.min_speed
        self.max_speed = args.max_speed
        
        # Coding and modulation
        self.coderate = args.coderate
        self.num_bits_per_symbol = args.num_bits_per_symbol
        self.modulation = args.modulation
        
        # Neural network architecture
        self.precoder_mode = args.precoder_mode
        self.precoder_arch = args.precoder_arch
        self.num_conv_channels_precoder = args.num_conv_channels_precoder
        self.num_conv_channels_receiver = args.num_conv_channels_receiver
        self.kernel_size = args.kernel_size
        
    def _parse_pilot_indices(self, pilot_str):
        """Parse pilot location string to list of indices"""
        if not pilot_str or pilot_str.strip() == "":
            return None
        return [int(x) for x in pilot_str.split(',')]
    
    def get_resource_grid(self):
        """Create and return resource grid"""
        pilot_pattern = "kronecker" if self.has_pilots else None
        
        rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self.num_streams_per_tx,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=[0, 0],
            dc_null=False,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices
        )
        return rg
    
    def get_code_params(self, rg):
        """Get coding parameters based on resource grid"""
        n = int(rg.num_data_symbols * self.num_bits_per_symbol)
        k = int(n * self.coderate)
        return k, n
    
    def compute_code_length(self, rg):
        """Compute code length n for use in traced functions"""
        # This can be called during tracing
        return int(rg.num_data_symbols * self.num_bits_per_symbol)


class BaselineReceiver(tf.keras.Model):
    """Baseline receiver with perfect CSI or LS channel estimation"""
    def __init__(self, config, rg, system_type="perfect-csi"):
        super().__init__()
        self.config = config
        self.rg = rg
        self.system_type = system_type
        
        # Stream management
        self.sm = StreamManagement(np.array([[1]]), config.num_streams_per_tx)
        
        # Mapper and demapper
        self.mapper = Mapper("qam", config.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", config.num_bits_per_symbol)
        
        # Channel estimation and equalization
        if system_type == "ls-estimation":
            self.ls_est = LSChannelEstimator(rg, interpolation_type="nn")
        elif system_type == "perfect-csi":
            self.remove_nulled_scs = RemoveNulledSubcarriers(rg)
        
        self.lmmse_equ = LMMSEEqualizer(rg, self.sm)
        
    def call(self, y, h_freq, no, g=None):
        """
        Process received signal
        Args:
            y: Received signal after OFDM demodulation
            h_freq: Frequency domain channel (for perfect CSI)
            no: Noise variance (per sample)
            g: Effective channel after precoding (for downlink)
        Returns:
            llr: Log-likelihood ratios
        """
        batch_size = tf.shape(y)[0]
        no_batch = tf.fill([batch_size], no)
        
        if self.system_type == "perfect-csi":
            if self.config.direction == "uplink":
                h_hat = self.remove_nulled_scs(h_freq)
            else:  # downlink
                h_hat = g
            err_var = 0.0
        elif self.system_type == "ls-estimation":
            h_hat, err_var = self.ls_est([y, no_batch])
        
        # LMMSE equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no_batch])
        no_eff_ = expand_to_rank(no_eff, tf.rank(x_hat))
        
        # Demapping
        llr = self.demapper([x_hat, no_eff_])
        return llr


class NeuralReceiverSystem(tf.keras.Model):
    """Neural receiver system with optional neural precoding"""
    def __init__(self, config, rg):
        super().__init__()
        self.config = config
        self.rg = rg
        
        # Pre-compute code length (avoid calling get_code_params in graph mode)
        self.k, self.n = config.get_code_params(rg)
        
        # Stream management
        self.sm = StreamManagement(np.array([[1]]), config.num_streams_per_tx)
        
        # Resource grid demapper
        self.rg_demapper = ResourceGridDemapper(rg, self.sm)
        
        # Build neural precoder based on mode
        self._build_precoder()
        
        # Build neural receiver based on mode and pilot availability
        self._build_receiver()
        
        # Mapper (can be trainable for neural modulation)
        self._build_mapper()
        
    def _build_precoder(self):
        """Build neural precoder based on configuration"""
        self.has_neural_precoder = False
        self.has_sip = False
        
        if self.config.precoder_mode == "nm":
            self.has_neural_precoder = True
            if self.config.precoder_arch == "full":
                self.neural_precoder = NeuralPrecoder(
                    self.config.num_conv_channels_precoder,
                    self.config.num_bs_ant
                )
            elif self.config.precoder_arch == "linear_1l":
                self.neural_precoder = NeuralPrecoder_1_layer_linear(
                    self.config.kernel_size,
                    self.config.num_bs_ant
                )
            elif self.config.precoder_arch == "linear_1l_relu":
                self.neural_precoder = NeuralPrecoder_1_layer_relu(
                    self.config.kernel_size,
                    self.config.num_bs_ant
                )
            elif self.config.precoder_arch == "time_1D":
                self.neural_precoder = NeuralPrecoder1D(
                    self.config.kernel_size,
                    self.config.num_bs_ant,
                    direction="time"
                )
            elif self.config.precoder_arch == "freq_1D":
                self.neural_precoder = NeuralPrecoder1D(
                    self.config.kernel_size,
                    self.config.num_bs_ant,
                    direction="freq"
                )
        
        elif self.config.precoder_mode == "sip":
            self.has_sip = True
            self.sip_precoder = NeuralPrecoderSIP(
                self.config.num_conv_channels_precoder,
                self.config.num_bs_ant
            )
        
        elif self.config.precoder_mode == "nm_sip":
            self.has_neural_precoder = True
            self.has_sip = True
            self.neural_precoder = NeuralPrecoder(
                self.config.num_conv_channels_precoder,
                self.config.num_bs_ant
            )
            self.sip_precoder = NeuralPrecoderSIP(
                self.config.num_conv_channels_precoder,
                self.config.num_bs_ant
            )
        
        # else: nrx mode - no precoder needed
    
    def _build_receiver(self):
        """Build neural receiver based on configuration"""
        if self.has_sip:
            # SIP mode uses special receiver
            self.neural_receiver = NeuralReceiverSIP(
                self.config.num_conv_channels_receiver,
                self.config.num_bits_per_symbol
            )
        else:
            # NRX or NM mode
            if self.config.has_pilots:
                if self.config.receiver_type == 'pilot_channel':
                    # Use receiver with h_freq input
                    self.neural_receiver = NeuralReceiverPilotChannel(
                        self.config.num_conv_channels_receiver,
                        self.config.num_bits_per_symbol
                    )
                else:
                    self.neural_receiver = NeuralReceiverPilot(
                        self.config.num_conv_channels_receiver,
                        self.config.num_bits_per_symbol
                    )
            else:
                self.neural_receiver = NeuralReceiver(
                    self.config.num_conv_channels_receiver,
                    self.config.num_bits_per_symbol
                )
    
    def _build_mapper(self):
        """Build mapper (trainable constellation for neural modulation)"""
        if self.config.modulation == "neural":
            from sionna.mapping import Constellation
            constellation = Constellation(
                "qam",
                self.config.num_bits_per_symbol,
                trainable=True,
                center=True,
                normalize=True
            )
            self.constellation = constellation
            self.mapper = Mapper("custom", self.config.num_bits_per_symbol, constellation=constellation)
        else:
            self.mapper = Mapper("qam", self.config.num_bits_per_symbol)
            self.constellation = None
    
    def _apply_neural_precoding(self, x_rg, pilot_indices=None):
        """
        Apply neural precoding to the transmitted signal
        
        Args:
            x_rg: Resource grid [batch, 1, num_streams, num_symbols, num_subcarriers]
            pilot_indices: Indices of pilot symbols (if any)
        
        Returns:
            x_rg_processed: Processed resource grid
        """
        batch_size = tf.shape(x_rg)[0]
        num_symbols = tf.shape(x_rg)[3]
        num_subcarriers = tf.shape(x_rg)[4]
        
        # Compute input energy for normalization
        energy_input = tf.reduce_mean(tf.abs(x_rg)**2, axis=[1,2,3,4], keepdims=True)
        
        if self.config.precoder_mode == "nm":
            # Neural modulation: only process data symbols, leave pilots unchanged
            if pilot_indices is not None:
                # Extract data symbol indices
                all_indices = tf.range(num_symbols)
                data_mask = tf.ones(num_symbols, dtype=tf.bool)
                for pilot_idx in pilot_indices:
                    data_mask = tf.tensor_scatter_nd_update(
                        data_mask,
                        [[pilot_idx]],
                        [False]
                    )
                data_indices = tf.boolean_mask(all_indices, data_mask)
                
                # Extract data symbols
                x_rg_data = tf.gather(x_rg, data_indices, axis=3)
                
                # Compute energy of data symbols only
                energy_input_data = tf.reduce_mean(tf.abs(x_rg_data)**2, axis=[1,2,3,4], keepdims=True)
                
                # Apply neural precoder to data symbols
                x_rg_data = self.neural_precoder(tf.squeeze(x_rg_data, axis=1))
                
                # Normalize data symbols
                energy_output_data = tf.reduce_mean(tf.abs(x_rg_data)**2, axis=[1,2,3,4], keepdims=True)
                scale = tf.cast(tf.sqrt(energy_input_data / (energy_output_data + 1e-12)), x_rg.dtype)
                x_rg_data = x_rg_data * scale
                
                # Scatter processed data symbols back into full grid
                x_rg_data = tf.expand_dims(x_rg_data, axis=1)  # Add back dimension
                
                # Create scatter indices
                num_data_symbols = tf.shape(x_rg_data)[3]
                batch_idx = tf.range(batch_size)[:, None, None, None, None]
                batch_idx = tf.tile(batch_idx, [1, 1, 1, num_data_symbols, num_subcarriers])
                
                dim1_idx = tf.zeros_like(batch_idx)
                dim2_idx = tf.zeros_like(batch_idx)
                
                symbol_idx = data_indices[None, None, None, :, None]
                symbol_idx = tf.tile(symbol_idx, [batch_size, 1, 1, 1, num_subcarriers])
                
                subcarrier_idx = tf.range(num_subcarriers)[None, None, None, None, :]
                subcarrier_idx = tf.tile(subcarrier_idx, [batch_size, 1, 1, num_data_symbols, 1])
                
                indices = tf.stack([batch_idx, dim1_idx, dim2_idx, symbol_idx, subcarrier_idx], axis=-1)
                indices = tf.reshape(indices, [-1, 5])
                updates = tf.reshape(x_rg_data, [-1])
                
                x_rg_out = tf.tensor_scatter_nd_update(x_rg, indices, updates)
                
            else:
                # No pilots - process entire grid
                x_rg_out = self.neural_precoder(tf.squeeze(x_rg, axis=1))
                
                # Normalize
                energy_output = tf.reduce_mean(tf.abs(x_rg_out)**2, axis=[1,2,3,4], keepdims=True)
                scale = tf.cast(tf.sqrt(energy_input / (energy_output + 1e-12)), x_rg.dtype)
                x_rg_out = x_rg_out * scale
                
                x_rg_out = tf.expand_dims(x_rg_out, axis=1)
        
        elif self.config.precoder_mode == "sip":
            # SIP only
            x_rg_out = self.sip_precoder(tf.squeeze(x_rg, axis=1))
            
            # Normalize
            energy_output = tf.reduce_mean(tf.abs(x_rg_out)**2, axis=[1,2,3,4], keepdims=True)
            scale = tf.cast(tf.sqrt(energy_input / (energy_output + 1e-12)), x_rg.dtype)
            x_rg_out = x_rg_out * scale
            
            x_rg_out = tf.expand_dims(x_rg_out, axis=1)
        
        elif self.config.precoder_mode == "nm_sip":
            # Neural modulation followed by SIP
            x_rg_out = self.neural_precoder(tf.squeeze(x_rg, axis=1))
            x_rg_out = self.sip_precoder(x_rg_out)
            
            # Normalize
            energy_output = tf.reduce_mean(tf.abs(x_rg_out)**2, axis=[1,2,3,4], keepdims=True)
            scale = tf.cast(tf.sqrt(energy_input / (energy_output + 1e-12)), x_rg.dtype)
            x_rg_out = x_rg_out * scale
            
            x_rg_out = tf.expand_dims(x_rg_out, axis=1)
        
        return x_rg_out
    
    def call(self, y, x_rg_pilots, no, h_freq=None):
        """
        Process received signal through neural receiver
        
        Args:
            y: Received signal [batch, num_streams, num_symbols, num_subcarriers]
            x_rg_pilots: Pilot symbols (if available)
            no: Noise variance
            h_freq: Channel frequency response (if using pilot_channel receiver)
        
        Returns:
            llr: Log-likelihood ratios
        """
        batch_size = tf.shape(y)[0]
        
        # Neural receiver processing
        if self.config.has_pilots:
            if self.config.receiver_type == 'pilot_channel' and h_freq is not None:
                llr = self.neural_receiver([y, x_rg_pilots, h_freq, no])
            else:
                llr = self.neural_receiver([y, x_rg_pilots, no])
        else:
            llr = self.neural_receiver([y, no])
        
        # Reshape for decoder
        llr = insert_dims(llr, 2, 1)
        llr = self.rg_demapper(llr)
        
        # Use pre-computed code length
        llr = tf.reshape(llr, [batch_size, 1, 1, self.n])
        
        return llr


class OFDMSystem(tf.keras.Model):
    """Complete OFDM communication system"""
    def __init__(self, config, system_type="neural-receiver", training=False, inference_speed=None):
        super().__init__()
        self.config = config
        self.system_type = system_type
        self.training_mode = training
        
        # Create resource grid
        self.rg = config.get_resource_grid()
        self.k, self.n = config.get_code_params(self.rg)
        
        # Channel configuration
        self._setup_channel(inference_speed)
        
        # Source and coding
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        
        # Resource grid mapper
        self.rg_mapper = ResourceGridMapper(self.rg)
        
        # OFDM modulator/demodulator
        self.l_min, self.l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.l_tot = self.l_max - self.l_min + 1
        self.modulator = OFDMModulator(config.cyclic_prefix_length)
        self.demodulator = OFDMDemodulator(config.fft_size, self.l_min, config.cyclic_prefix_length)
        
        # Time domain channel
        self.channel_time = ApplyTimeChannel(
            self.rg.num_time_samples,
            l_tot=self.l_tot,
            add_awgn=True
        )
        
        # Build transmitter/receiver
        if "baseline" in system_type:
            self.receiver = BaselineReceiver(config, self.rg, system_type.replace("baseline-", ""))
            self.mapper = self.receiver.mapper
        else:
            self.receiver = NeuralReceiverSystem(config, self.rg)
            self.mapper = self.receiver.mapper
        
        # ZF precoder for downlink
        if config.direction == "downlink":
            self.sm = StreamManagement(np.array([[1]]), config.num_streams_per_tx)
            self.zf_precoder = ZFPrecoder(self.rg, self.sm, return_effective_channel=True)
    
    def _setup_channel(self, inference_speed):
        """Setup channel model"""
        speed_min = self.config.min_speed if inference_speed is None else inference_speed
        speed_max = self.config.max_speed if inference_speed is None else inference_speed
        
        self.channel = TDL(
            model=self.config.cdl_model,
            delay_spread=self.config.delay_spread,
            carrier_frequency=self.config.carrier_frequency,
            min_speed=speed_min,
            max_speed=speed_max
        )
    
    def _extract_pilots(self, x_rg):
        """Extract pilot symbols from resource grid"""
        if not self.config.has_pilots:
            return None
        
        # Create pilot mask
        pilot_mask = np.zeros((self.config.num_ofdm_symbols,), dtype=np.float32)
        pilot_mask[self.config.pilot_ofdm_symbol_indices] = 1.0
        pilot_mask = tf.constant(pilot_mask)
        pilot_mask = tf.reshape(pilot_mask, [1, 1, 1, self.config.num_ofdm_symbols, 1])
        pilot_mask = tf.cast(pilot_mask, dtype=x_rg.dtype)
        
        # Extract pilots
        x_rg_pilots = x_rg * pilot_mask
        return x_rg_pilots
    
    @tf.function
    def call(self, batch_size, ebno_db):
        """
        Execute one forward pass
        
        Args:
            batch_size: Number of samples in batch
            ebno_db: Eb/N0 in dB
        
        Returns:
            If training: rate (bits per channel use)
            If inference: (b, b_hat) transmitted and decoded bits
        """
        # Compute noise variance
        no = ebnodb2no(ebno_db, self.config.num_bits_per_symbol, self.config.coderate, self.rg)
        
        # Generate bits
        if self.training_mode:
            c = self.binary_source([batch_size, 1, self.config.num_streams_per_tx, self.n])
        else:
            b = self.binary_source([batch_size, 1, self.config.num_streams_per_tx, self.k])
            c = self.encoder(b)
        
        # Map to symbols
        x = self.mapper(c)
        
        # Map to resource grid
        x_rg = self.rg_mapper(x)
        
        # Extract pilots
        x_rg_pilots = self._extract_pilots(x_rg)
        
        # Generate channel
        num_time_samples = self.rg.num_time_samples + self.l_tot - 1
        a, tau = self.channel(batch_size, num_time_samples, self.rg.bandwidth)
        
        # Convert to time and frequency domain
        h_time = cir_to_time_channel(
            self.rg.bandwidth, a, tau,
            l_min=self.l_min, l_max=self.l_max,
            normalize=True
        )
        h_freq = time_to_ofdm_channel(h_time, self.rg, self.l_min)
        
        # Apply downlink precoding if needed
        g = None
        if self.config.direction == "downlink":
            x_rg, g = self.zf_precoder([x_rg, h_freq])
        
        # Apply neural precoding if needed
        if isinstance(self.receiver, NeuralReceiverSystem):
            if self.receiver.has_neural_precoder or self.receiver.has_sip:
                x_rg = self.receiver._apply_neural_precoding(
                    x_rg,
                    self.config.pilot_ofdm_symbol_indices
                )
        
        # OFDM modulation
        x_time = self.modulator(x_rg)
        
        # Apply channel
        y_time = self.channel_time([x_time, h_time, no])
        
        # OFDM demodulation
        y = self.demodulator(y_time)
        
        # Receiver processing
        if isinstance(self.receiver, BaselineReceiver):
            llr = self.receiver(y, h_freq, no, g)
        else:
            # Remove batch dimension for neural receiver
            y_squeezed = tf.squeeze(y, axis=1)
            if x_rg_pilots is not None:
                x_rg_pilots_squeezed = tf.squeeze(x_rg_pilots, axis=1)
            else:
                x_rg_pilots_squeezed = None
            no_batch = tf.fill([batch_size], no)
            
            # Pass h_freq if using pilot_channel receiver
            if self.config.receiver_type == 'pilot_channel':
                h_freq_squeezed = tf.squeeze(h_freq, axis=[1, 2])  # [batch, num_ofdm_symbols, num_subcarriers]
                llr = self.receiver(y_squeezed, x_rg_pilots_squeezed, no_batch, h_freq_squeezed)
            else:
                llr = self.receiver(y_squeezed, x_rg_pilots_squeezed, no_batch)
        
        # Return rate for training, bits for inference
        if self.training_mode:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce / tf.math.log(2.)
            return rate
        else:
            b_hat = self.decoder(llr)
            return b, b_hat
    
    def count_parameters(self):
        """Count and log model parameters"""
        try:
            total = self.count_params()
            trainable = sum(tf.keras.backend.count_params(w) for w in self.trainable_weights)
            non_trainable = total - trainable
            
            logging.info(f"Total parameters: {total:,}")
            logging.info(f"Trainable parameters: {trainable:,}")
            logging.info(f"Non-trainable parameters: {non_trainable:,}")
            
            # Count receiver parameters
            if hasattr(self.receiver, 'neural_receiver'):
                receiver_params = sum(
                    tf.keras.backend.count_params(w) 
                    for w in self.receiver.neural_receiver.trainable_weights
                )
                logging.info(f"Neural receiver parameters: {receiver_params:,}")
            
            # Count precoder parameters
            if hasattr(self.receiver, 'neural_precoder'):
                precoder_params = sum(
                    tf.keras.backend.count_params(w) 
                    for w in self.receiver.neural_precoder.trainable_weights
                )
                logging.info(f"Neural precoder parameters: {precoder_params:,}")
                
        except Exception as e:
            logging.warning(f"Could not count parameters: {e}")


def setup_logging(results_folder, params, inf_speed):
    """Setup logging configuration"""
    log_dir = os.path.join(results_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{params}_{inf_speed}_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def create_results_folder(args):
    """Create results folder structure based on parameters"""
    # Build folder name components
    precoder_str = args.precoder_mode
    subcarrier_str = int(args.subcarrier_spacing / 1e3)
    delay_str = int(args.delay_spread * 1e9)
    
    # Convolutional channels string
    if args.precoder_mode == "nrx":
        conv_str = f"{args.num_conv_channels_receiver}"
    elif args.precoder_mode == "nm" and args.precoder_arch in ["linear_1l", "linear_1l_relu"]:
        conv_str = f"kernel_{args.kernel_size}"
    else:
        conv_str = f"{args.num_conv_channels_precoder}_{args.num_conv_channels_receiver}"
    
    # Pilot string
    if args.pilot_loc and args.pilot_loc.strip():
        pilot_list = [int(x) for x in args.pilot_loc.split(',')]
        pilot_str = len(pilot_list)
    else:
        pilot_str = 0
    
    # Precoder architecture string
    precoder_arch_str = f"precoder_arch_{args.precoder_arch}" if args.precoder_mode == "nm" else ""
    
    # Build full path
    results_folder = os.path.join(
        args.results_folder,
        f"seed_{args.random_seed}",
        precoder_str,
        args.direction,
        precoder_arch_str,
        f"num_subcarriers_{args.num_subcarriers}_sub_{subcarrier_str}_delay_{delay_str}_cp_{args.cyclic_prefix_length}",
        f"n_conv_{conv_str}_lr_{args.initial_learning_rate}_cosine",
        f"coderate_{args.coderate}_ant_{args.num_ut_ant}x{args.num_bs_ant}_mod_{args.modulation}_{args.num_bits_per_symbol}_p_{pilot_str}"
    )
    
    os.makedirs(results_folder, exist_ok=True)
    return results_folder


def save_checkpoint(model, optimizer, iteration, weights_folder, params, run_id):
    """Save training checkpoint"""
    import pickle
    
    ckpt_filename = f"checkpoint_{params}_{run_id}_iter_{iteration}.pkl"
    ckpt_filepath = os.path.join(weights_folder, ckpt_filename)
    
    model_weights_filename = f"model_weights_{params}_{run_id}_iter_{iteration}.pkl"
    model_weights_filepath = os.path.join(weights_folder, model_weights_filename)
    
    try:
        optimizer_weights = optimizer.get_weights()
    except AttributeError:
        optimizer_weights = None
    
    checkpoint = {
        'iteration': iteration,
        'model_weights': model.get_weights(),
        'optimizer_weights': optimizer_weights
    }
    
    with open(ckpt_filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    with open(model_weights_filepath, 'wb') as f:
        pickle.dump(model.get_weights(), f)
    
    logging.info(f"Saved checkpoint at iteration {iteration}")


def load_latest_checkpoint(weights_folder, params):
    """Load the latest checkpoint"""
    import glob
    import re
    
    pattern = f"checkpoint_{params}_*.pkl"
    files = glob.glob(os.path.join(weights_folder, pattern))
    
    if not files:
        return None, 0
    
    # Find file with highest iteration number
    latest = None
    latest_iter = -1
    for filepath in files:
        filename = os.path.basename(filepath)
        m = re.search(r'_iter_(\d+)', filename)
        if m:
            iter_num = int(m.group(1))
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest = filepath
    
    if latest:
        logging.info(f"Loading checkpoint from {latest}")
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint, checkpoint['iteration']
    
    return None, 0


def load_latest_model_weights(weights_folder, params):
    """Load latest model weights for inference"""
    import glob
    import re
    
    pattern = f"model_weights_{params}_*.pkl"
    files = glob.glob(os.path.join(weights_folder, pattern))
    
    if not files:
        return None, 0
    
    # Find file with highest iteration number
    latest = None
    latest_iter = -1
    for filepath in files:
        filename = os.path.basename(filepath)
        m = re.search(r'_iter_(\d+)', filename)
        if m:
            iter_num = int(m.group(1))
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest = filepath
    
    if latest:
        logging.info(f"Loading model weights from {latest}")
        with open(latest, 'rb') as f:
            weights = pickle.load(f)
        return weights, latest_iter
    
    return None, 0


def train(args, config, results_folder):
    """Training phase"""
    # Set random seed
    sionna.config.seed = args.random_seed
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create model
    model = OFDMSystem(config, system_type="neural-receiver", training=True)
    
    # Learning rate schedule
    lr_schedule = CosineDecay(
        initial_learning_rate=args.initial_learning_rate,
        decay_steps=args.num_training_iterations,
        alpha=0.05
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Build model
    model(1, tf.constant(10.0, tf.float32))
    model.count_parameters()
    
    # Load checkpoint if continuing training
    start_iter = 0
    weights_folder = os.path.join(results_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)
    
    params = f"qam_{args.num_bits_per_symbol}_cdl_{args.cdl_model}_speed_{args.min_speed}_{args.max_speed}"
    
    if args.continue_training:
        checkpoint, start_iter = load_latest_checkpoint(weights_folder, params)
        if checkpoint:
            model.set_weights(checkpoint['model_weights'])
            if checkpoint.get('optimizer_weights'):
                optimizer.set_weights(checkpoint['optimizer_weights'])
            start_iter += 1
            logging.info(f"Resuming from iteration {start_iter}")
        else:
            logging.info("No checkpoint found, starting from scratch")
    
    # Training loop
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in tqdm(range(start_iter, args.num_training_iterations)):
        # Random Eb/N0 for training
        ebno_db = tf.random.uniform(
            shape=[],
            minval=args.training_ebno_min,
            maxval=args.training_ebno_max
        )
        
        # Training step
        with tf.GradientTape() as tape:
            rate = model(args.training_batch_size, ebno_db)
            loss = -rate
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Logging
        if i % 1000 == 0:
            print(f'Iteration {i}/{args.num_training_iterations}  Rate: {rate.numpy():.4f} bit', end='\r')
            logging.info(f'Iteration {i}/{args.num_training_iterations}  Rate: {rate.numpy():.4f} bit')
        
        # Save checkpoint
        if i % 10000 == 0 and i > 0:
            save_checkpoint(model, optimizer, i, weights_folder, params, run_id)
        
        # Validation
        if i % (args.num_training_iterations // args.num_vals) == 0:
            current_lr = optimizer.learning_rate.numpy()
            logging.info(f'Step {i}: Learning Rate = {current_lr:.2e}')
            
            # Save constellation if using trainable constellation
            if hasattr(model.receiver, 'constellation') and model.receiver.constellation is not None:
                fig = model.receiver.constellation.show()
                fig.savefig(os.path.join(results_folder, f"constellation_{params}.pdf"))
                plt.close(fig)
    
    # Final checkpoint
    save_checkpoint(model, optimizer, args.num_training_iterations, weights_folder, params, run_id)
    logging.info("Training completed")


def inference(args, config, results_folder):
    """Inference phase"""
    params = f"qam_{args.num_bits_per_symbol}_cdl_{args.cdl_model}_speed_{args.min_speed}_{args.max_speed}"
    
    # Load model weights
    weights_folder = os.path.join(results_folder, "weights")
    
    if args.weights_path:
        logging.info(f"Loading weights from {args.weights_path}")
        with open(args.weights_path, 'rb') as f:
            weights = pickle.load(f)
        # Extract iteration from filename
        import re
        latest_iter = int(re.search(r'_(\d+)\.pkl$', args.weights_path).group(1))
    else:
        weights, latest_iter = load_latest_model_weights(weights_folder, params)
        if weights is None:
            raise ValueError("No saved model found for inference")
    
    # Eb/N0 range
    ebno_dbs = np.arange(args.inference_ebno_min, args.inference_ebno_max, args.inference_ebno_step)
    
    BER = {}
    BLER = {}
    
    # Neural receiver
    logging.info("Evaluating neural receiver...")
    model = OFDMSystem(
        config,
        system_type="neural-receiver",
        training=False,
        inference_speed=args.inf_speed
    )
    model(1, tf.constant(10.0, tf.float32))
    model.set_weights(weights)
    model.count_parameters()
    
    # Save constellation
    if hasattr(model.receiver, 'constellation') and model.receiver.constellation is not None:
        fig = model.receiver.constellation.show()
        fig.savefig(os.path.join(results_folder, f"constellation_{params}.pdf"))
        plt.close(fig)
    
    ber, bler = sim_ber(
        model, ebno_dbs,
        batch_size=args.inference_batch_size,
        max_mc_iter=args.max_mc_iter,
        num_target_block_errors=args.num_target_block_errors,
        target_bler=args.target_bler
    )
    BER['neural-receiver'] = ber.numpy()
    BLER['neural-receiver'] = bler.numpy()
    
    # Baseline: Perfect CSI
    logging.info("Evaluating baseline (perfect CSI)...")
    model = OFDMSystem(
        config,
        system_type="baseline-perfect-csi",
        training=False,
        inference_speed=args.inf_speed
    )
    
    ber, bler = sim_ber(
        model, ebno_dbs,
        batch_size=args.inference_batch_size,
        max_mc_iter=args.max_mc_iter,
        num_target_block_errors=args.num_target_block_errors,
        target_bler=args.target_bler
    )
    BER['baseline-perfect-csi'] = ber.numpy()
    BLER['baseline-perfect-csi'] = bler.numpy()
    
    # Baseline: LS estimation (if pilots available)
    if config.has_pilots:
        logging.info("Evaluating baseline (LS estimation)...")
        model = OFDMSystem(
            config,
            system_type="baseline-ls-estimation",
            training=False,
            inference_speed=args.inf_speed
        )
        
        ber, bler = sim_ber(
            model, ebno_dbs,
            batch_size=args.inference_batch_size,
            max_mc_iter=args.max_mc_iter,
            num_target_block_errors=args.num_target_block_errors,
            target_bler=args.target_bler
        )
        BER['baseline-ls-estimation'] = ber.numpy()
        BLER['baseline-ls-estimation'] = bler.numpy()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c='C0', label='Baseline - Perfect CSI')
    if config.has_pilots:
        plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c='C1', label='Baseline - LS Estimation')
    plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c='C2', label='Neural receiver')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"bler_{params}_{args.inf_speed}_{latest_iter}_bs_{args.inference_batch_size}.pdf"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, BER['baseline-perfect-csi'], 'o-', c='C0', label='Baseline - Perfect CSI')
    if config.has_pilots:
        plt.semilogy(ebno_dbs, BER['baseline-ls-estimation'], 'x--', c='C1', label='Baseline - LS Estimation')
    plt.semilogy(ebno_dbs, BER['neural-receiver'], 's-.', c='C2', label='Neural receiver')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"ber_{params}_{args.inf_speed}_{latest_iter}_bs_{args.inference_batch_size}.pdf"))
    plt.close()
    
    # Log results
    logging.info(f"EB/N0 (dB): {list(ebno_dbs)}")
    logging.info(f"BER: {BER}")
    logging.info(f"BLER: {BLER}")
    
    print("\n" + "="*80)
    print(f"EB/N0 (dB): {list(ebno_dbs)}")
    print(f"BER: {BER}")
    print(f"BLER: {BLER}")
    print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OFDM Neural Receiver System")
    
    # GPU configuration
    parser.add_argument('--gpu', type=int, default=0, help="GPU id")
    parser.add_argument('--debug', action='store_true', help="Run in eager mode")
    
    # Training configuration
    parser.add_argument('--training', action='store_true', help="Run training phase")
    parser.add_argument('--continue_training', action='store_true', help="Continue from checkpoint")
    parser.add_argument('--num_training_iterations', type=int, default=100000, help="Training iterations")
    parser.add_argument('--training_batch_size', type=int, default=128, help="Training batch size")
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--num_vals', type=int, default=5, help="Number of validation points")
    
    # Architecture configuration
    parser.add_argument('--precoder_mode', choices=['nrx', 'nm', 'sip', 'nm_sip'], 
                        default='nrx', help='Precoding mode')
    parser.add_argument('--precoder_arch', choices=['full', 'linear_1l', 'linear_1l_relu', 'time_1D', 'freq_1D'],
                        default='full', help='Precoder architecture')
    parser.add_argument('--receiver_type', choices=['pilot', 'pilot_channel'],
                        default='pilot', help='Receiver type: pilot (y+pilot) or pilot_channel (y+pilot+h_freq)')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for 1D architectures')
    parser.add_argument('--num_conv_channels_precoder', type=int, default=128, help='Precoder channels')
    parser.add_argument('--num_conv_channels_receiver', type=int, default=128, help='Receiver channels')
    
    # System configuration
    parser.add_argument('--modulation', type=str, default='qam', help='Modulation (qam/neural)')
    parser.add_argument('--num_bits_per_symbol', type=int, default=6, help='Bits per symbol')
    parser.add_argument('--coderate', type=float, default=0.5, help='Code rate')
    parser.add_argument('--num_ut_ant', type=int, default=1, help='UE antennas')
    parser.add_argument('--num_bs_ant', type=int, default=4, help='BS antennas')
    parser.add_argument('--direction', type=str, default='uplink', help='uplink/downlink')
    
    # OFDM configuration
    parser.add_argument('--num_subcarriers', type=int, default=32, help='Number of subcarriers')
    parser.add_argument('--subcarrier_spacing', type=float, default=30e3, help='Subcarrier spacing (Hz)')
    parser.add_argument('--cyclic_prefix_length', type=int, default=6, help='CP length')
    parser.add_argument('--pilot_loc', type=str, default='2,11', help='Pilot symbol indices (comma-separated)')
    
    # Channel configuration
    parser.add_argument('--cdl_model', type=str, default='A', help='CDL model')
    parser.add_argument('--delay_spread', type=float, default=100e-9, help='Delay spread (s)')
    parser.add_argument('--min_speed', type=float, default=20.0, help='Min speed (m/s)')
    parser.add_argument('--max_speed', type=float, default=20.0, help='Max speed (m/s)')
    parser.add_argument('--inf_speed', type=float, default=20.0, help='Inference speed (m/s)')
    parser.add_argument('--ul_center_frequency', type=float, default=1.91e9, help='UL center freq (Hz)')
    parser.add_argument('--dl_center_frequency', type=float, default=2.11e9, help='DL center freq (Hz)')
    parser.add_argument('--num_delay', type=int, default=1, help='Delay slots')
    parser.add_argument('--perfect_csi', type=bool, default=False, help='Perfect CSI flag')
    
    # Training Eb/N0 range
    parser.add_argument('--training_ebno_min', type=float, default=0.0, help='Training Eb/N0 min (dB)')
    parser.add_argument('--training_ebno_max', type=float, default=20.0, help='Training Eb/N0 max (dB)')
    parser.add_argument('--training_ebno_step', type=float, default=4.0, help='Training Eb/N0 step (dB)')
    
    # Inference Eb/N0 range
    parser.add_argument('--inference_ebno_min', type=float, default=0.0, help='Inference Eb/N0 min (dB)')
    parser.add_argument('--inference_ebno_max', type=float, default=20.0, help='Inference Eb/N0 max (dB)')
    parser.add_argument('--inference_ebno_step', type=float, default=2.0, help='Inference Eb/N0 step (dB)')
    
    # Inference configuration
    parser.add_argument('--inference_batch_size', type=int, default=1024, help='Inference batch size')
    parser.add_argument('--max_mc_iter', type=int, default=1000, help='Max MC iterations')
    parser.add_argument('--num_target_block_errors', type=int, default=100, help='Target block errors')
    parser.add_argument('--target_bler', type=float, default=1e-3, help='Target BLER')
    
    # Output configuration
    parser.add_argument('--results_folder', type=str, default='results_claude', help='Results folder')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to weights file')
    parser.add_argument('--random_seed', type=int, default=101, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup GPU
    setup_gpu(args.gpu)
    
    # Configure TensorFlow execution
    tf.config.run_functions_eagerly(args.debug)
    
    # Create configuration
    config = OFDMConfig(args)
    
    # Create results folder
    results_folder = create_results_folder(args)
    
    # Setup logging
    params = f"qam_{args.num_bits_per_symbol}_cdl_{args.cdl_model}_speed_{args.min_speed}_{args.max_speed}"
    setup_logging(results_folder, params, args.inf_speed)
    
    logging.info("="*80)
    logging.info("OFDM Neural Receiver System")
    logging.info("="*80)
    logging.info(f"Precoder mode: {args.precoder_mode}")
    logging.info(f"Direction: {args.direction}")
    logging.info(f"Results folder: {results_folder}")
    logging.info("="*80)
    
    start_time = time.time()
    
    # Training phase
    if args.training:
        logging.info("Starting training phase...")
        train(args, config, results_folder)
        logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Inference phase
    logging.info("Starting inference phase...")
    inference(args, config, results_folder)
    logging.info(f"Inference completed in {time.time() - start_time:.2f} seconds")
    
    logging.info("="*80)
    logging.info("All tasks completed successfully")
    logging.info("="*80)


if __name__ == "__main__":
    main()