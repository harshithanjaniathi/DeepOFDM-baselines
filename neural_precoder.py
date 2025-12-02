import os
import io
import time
import pickle
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

import re

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.run_functions_eagerly(False)  # Enable graph mode by default

# Pre-parse GPU argument before importing TensorFlow
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--gpu', type=int, default=0, help="GPU id")
pre_args, _ = pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(pre_args.gpu)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import sionna
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel, time_to_ofdm_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import BinarySource, ebnodb2no, sim_ber, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.utils.metrics import compute_ber
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from tensorflow.keras.optimizers.schedules import CosineDecay

from models_end import NeuralPrecoder, NeuralPrecoder1D, NeuralPrecoder_1_layer_linear, NeuralPrecoder_1_layer_relu, NeuralReceiver, NeuralReceiverPilot, NeuralPrecoderSIP, NeuralReceiverSIP

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="OFDM MIMO simulation with neural receiver and baseline receivers using argparse.")
    # Training/inference mode and learning parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU id")
    parser.add_argument('--training', action='store_true',
                        help="Run the training phase if specified.")
    parser.add_argument('--debug', action='store_true',
                        help="Run in eager mode.")
    parser.add_argument('--continue_training', action='store_true',
                        help="Continue training from the latest saved checkpoint if available. Default is to train from scratch.")
    parser.add_argument('--num_training_iterations', type=int, default=100000,
                        help="Number of training iterations.")
    parser.add_argument('--training_batch_size', type=int, default=128,
                        help="Training batch size.")
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3,
                        help="Initial learning rate.")
    # Note: decay_steps and decay_rate are no longer used by the cosine scheduler.
    parser.add_argument('--num_vals', type=int, default=5,
                        help="Number of validation points.")

    # Simulation parameters
    parser.add_argument('--precoder_arch', choices=['full', 'linear_1l', 'linear_1l_relu', 'time_1D', 'freq_1D'], 
                        default="full", help="Precoding scheme (full/linear_1l/linear_1l_relu/time_1D/freq_1D).")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Kernel size for convolutional layers in the neural precoder.")
    parser.add_argument(    '--precoder_mode', choices=['nrx','nm','nm_sip','sip'], 
                        default='nrx', help='Precoding scheme (nrx/nm/nm_sip/sip).')
    parser.add_argument('--modulation', type=str, default="qam",
                        help="Modulation scheme (qam/neural).")
    parser.add_argument('--coderate', type=float, default=0.5,
                        help="Channel coding rate")
    parser.add_argument('--num_ut_ant', type=int, default=1,
                        help="Number of UE antenna.")
    parser.add_argument('--num_bs_ant', type=int, default=4,
                        help="Number of BS antenna.")
    parser.add_argument('--num_bits_per_symbol', type=int, default=6,
                        help="Number of bits per symbol.")
    parser.add_argument('--min_speed', type=float, default=20.0,
                        help="Minimum speed (m/s).")
    parser.add_argument('--max_speed', type=float, default=20.0,
                        help="Maximum speed (m/s).")
    parser.add_argument('--inf_speed', type=float, default=20.0,
                        help="Infinite speed for inference (m/s).")
    parser.add_argument('--cdl_model', type=str, default="C",
                        help="CDL model to use.")
    parser.add_argument('--direction', type=str, default="uplink",
                        help="Transmission direction (uplink/downlink).")
    parser.add_argument('--delay_spread', type=float, default=100e-9,
                        help="Nominal delay spread (s).")
    parser.add_argument('--perfect_csi', type=bool, default=False,
                        help="Use perfect CSI (True/False).")
    parser.add_argument('--cyclic_prefix_length', type=int, default=6,
                        help="Cyclic prefix length.")
    parser.add_argument('--pilot_loc', type=str, default="2,11",
                        help="Comma-separated list of pilot OFDM symbol indices.")
    parser.add_argument('--num_subcarriers', type=int, default=32,
                        help="Number of subcarriers.")
    parser.add_argument('--subcarrier_spacing', type=float, default=30e3,
                        help="Subcarrier spacing (Hz).")
    parser.add_argument("--ul_center_frequency", type=float, default=1.91e9,
                        help="Uplink center frequency")
    parser.add_argument("--dl_center_frequency", type=float, default=2.11e9,
                        help="Downlink center frequency")
    parser.add_argument('--num_delay', type=int, default=1,
                        help="Number of delay slots for CSI.")

    # Neural precoder parameter
    parser.add_argument('--num_conv_channels_precoder', type=int, default=128,
                        help="Number of convolutional channels in the neural receiver.")

    # Neural receiver parameter
    parser.add_argument('--num_conv_channels_receiver', type=int, default=128,
                        help="Number of convolutional channels in the neural receiver.")

    # Random seed
    parser.add_argument('--random_seed', type=int, default=101,
                        help="Random seed for reproducibility.")

    # Eb/No simulation parameters for training validation and inference
    parser.add_argument('--training_ebno_min', type=float, default=0.0,
                        help="Minimum Eb/No (dB) for training validation.")
    parser.add_argument('--training_ebno_max', type=float, default=20.0,
                        help="Maximum Eb/No (dB) for training validation.")
    parser.add_argument('--training_ebno_step', type=float, default=4.0,
                        help="Step for Eb/No (dB) in training validation.")
    parser.add_argument('--inference_ebno_min', type=float, default=-0.0,
                        help="Minimum Eb/No (dB) for inference.")
    parser.add_argument('--inference_ebno_max', type=float, default=20.0,
                        help="Maximum Eb/No (dB) for inference.")
    parser.add_argument('--inference_ebno_step', type=float, default=2.0,
                        help="Step for Eb/No (dB) in inference.")

    # Inference sim_ber parameters
    parser.add_argument('--inference_batch_size', type=int, default=1024,
                        help="Batch size for inference simulation.")
    parser.add_argument('--max_mc_iter', type=int, default=1000,
                        help="Maximum Monte Carlo iterations for inference.")
    parser.add_argument('--num_target_block_errors', type=int, default=100,
                        help="Number of target block errors for inference.")
    parser.add_argument('--target_bler', type=float, default=1e-3,
                        help="Target BLER for inference.")

    # Results and logging parameters
    parser.add_argument('--results_folder', type=str, default="harshitha_exp",
                        help="Base folder for results.")
    parser.add_argument('--weights_path', type=str, default=None,
                        help="Base folder for results.")
    
    # parser.add_argument('--results_folder', type=str, default="inf_models_end_nrx_pilot",
    #                     help="Base folder for results.")

    args = parser.parse_args()
    tf.config.run_functions_eagerly(args.debug)  # Enable eager mode if args.debug.


    # Process pilot_loc argument from comma-separated string to list of ints.
    if not args.pilot_loc or args.pilot_loc.strip() == "":
        args.pilot_loc = None
    else:
        pilot_loc = [int(x) for x in args.pilot_loc.split(',')]
        args.pilot_loc = pilot_loc

    subcarrier_spacing = args.subcarrier_spacing
    num_delay = args.num_delay

    ul_center_frequency = args.ul_center_frequency
    dl_center_frequency = args.dl_center_frequency
    num_subcarriers = args.num_subcarriers

    carrier_frequency = (ul_center_frequency + dl_center_frequency) / 2
    total_num_subcarriers = (dl_center_frequency - ul_center_frequency + num_subcarriers * subcarrier_spacing) // subcarrier_spacing
    fft_size = num_subcarriers
    args.carrier_frequency = carrier_frequency
    args.total_num_subcarriers = total_num_subcarriers
    args.fft_size = fft_size

    # Create results folder based on parameters
    precoder_str = args.precoder_mode

    subcarrier_str = int(subcarrier_spacing / 1e3)
    delay_str = int(args.delay_spread * 1e9)

    if args.precoder_mode == "nrx":
        conv_str = f"{args.num_conv_channels_receiver}"
    elif args.precoder_mode == "nm" and args.precoder_arch == "linear_1l" and args.precoder_arch == "linear_1l_relu":
        conv_str = f"kernel_{args.kernel_size}"
    else:
        conv_str = f"{args.num_conv_channels_precoder}_{args.num_conv_channels_receiver}"

    if args.pilot_loc is not None:
        pilot_str = len(args.pilot_loc)
    else:
        pilot_str = 0

    if args.precoder_mode == "nm":
        precoder_arch_str = f"precoder_arch_{args.precoder_arch}"
    else:
        precoder_arch_str = ""

    results_folder = os.path.join(args.results_folder,
                                  f"seed_{args.random_seed}",
                                  precoder_str,
                                  args.direction,
                                  precoder_arch_str,
                                  f"num_subcarriers_{args.num_subcarriers}_sub_{subcarrier_str}_delay_{delay_str}_cp_{args.cyclic_prefix_length}",
                                  f"n_conv_{conv_str}_lr_{args.initial_learning_rate}_cosine",
                                  f"coderate_{args.coderate}_ant_{args.num_ut_ant}x{args.num_bs_ant}_mod_{args.modulation}_{args.num_bits_per_symbol}_p_{pilot_str}",)
    os.makedirs(results_folder, exist_ok=True)

    # Create logs folder inside results_folder
    log_dir = os.path.join(results_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)

    params = f"qam_{args.num_bits_per_symbol}_cdl_{args.cdl_model}_speed_{args.min_speed}_{args.max_speed}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{params}_{args.inf_speed}_{timestamp}.txt")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    logging.info("Script started.")

    ############################################
    ## Training configuration
    num_training_iterations = args.num_training_iterations + 1  # To include the last iteration
    num_vals = args.num_vals
    training_batch_size = args.training_batch_size

    ############################################

    weights_folder = os.path.join(results_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    def get_latest_file(pattern):
        files = glob.glob(os.path.join(weights_folder, pattern))
        if not files:
            return None

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

        # If we found at least one file with an _iter_ number, return the highest
        if latest:
            return latest

        # Otherwise fall back to most recently modified
        return max(files, key=os.path.getmtime)

    class Model(tf.keras.Model):
        def __init__(self,
                     domain,
                     direction,
                     cdl_model,
                     delay_spread,
                     perfect_csi,
                     min_speed,
                     max_speed,
                     cyclic_prefix_length,
                     pilot_ofdm_symbol_indices,
                     carrier_frequency,
                     subcarrier_spacing,
                     num_subcarriers,
                     total_num_subcarriers,
                     num_delay,
                     system="neural-receiver",
                     training=False):
            super().__init__()
            self._system = system
            self._training = training

            self._domain = domain
            self._direction = direction
            self._cdl_model = cdl_model
            self._delay_spread = delay_spread
            self._perfect_csi = perfect_csi
            self._min_speed = min_speed
            self._max_speed = max_speed
            self._cyclic_prefix_length = cyclic_prefix_length
            self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

            self._precoder_mode = args.precoder_mode
            self._precoder_arch = args.precoder_arch
            self._carrier_frequency = carrier_frequency
            self._subcarrier_spacing = subcarrier_spacing
            self._fft_size = num_subcarriers
            self._num_subcarriers = num_subcarriers
            self._total_num_subcarriers = total_num_subcarriers
            self._num_delay = num_delay
            self._num_ofdm_symbols = 14
            self._num_ut_ant = args.num_ut_ant
            self._num_bs_ant = args.num_bs_ant
            self._num_streams_per_tx = self._num_ut_ant
            self._dc_null = False
            self._num_guard_carriers = [0, 0]
            if self._pilot_ofdm_symbol_indices == None:
                self._pilot_pattern = None
            else:
                self._pilot_pattern = "kronecker"
            self._num_bits_per_symbol = args.num_bits_per_symbol
            self._num_conv_channels_precoder = args.num_conv_channels_precoder
            self._num_conv_channels_receiver = args.num_conv_channels_receiver
            self._kernel_size = args.kernel_size
            self._num_filters = args.num_bs_ant
            self._coderate = args.coderate

            self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

            self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                    fft_size=self._fft_size,
                                    subcarrier_spacing=self._subcarrier_spacing,
                                    num_tx=1,
                                    num_streams_per_tx=self._num_streams_per_tx,
                                    cyclic_prefix_length=self._cyclic_prefix_length,
                                    num_guard_carriers=self._num_guard_carriers,
                                    dc_null=self._dc_null,
                                    pilot_pattern=self._pilot_pattern,
                                    pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

            self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
            self._k = int(self._n * self._coderate)

            if self._num_ut_ant % 2 == 0:
                self._ut_array = AntennaArray(num_rows=1,
                                              num_cols=int(self._num_ut_ant / 2),
                                              polarization="dual",
                                              polarization_type="cross",
                                              antenna_pattern="38.901",
                                              carrier_frequency=self._carrier_frequency)
            else:
                self._ut_array = AntennaArray(num_rows=1,
                                              num_cols=self._num_ut_ant,
                                              polarization="single",
                                              polarization_type="V",
                                              antenna_pattern="38.901",
                                              carrier_frequency=self._carrier_frequency)

            if self._num_bs_ant % 2 == 0:
                self._bs_array = AntennaArray(num_rows=1,
                                              num_cols=int(self._num_bs_ant / 2),
                                              polarization="dual",
                                              polarization_type="cross",
                                              antenna_pattern="38.901",
                                              carrier_frequency=self._carrier_frequency)
            else:
                self._bs_array = AntennaArray(num_rows=1,
                                              num_cols=self._num_bs_ant,
                                              polarization="single",
                                              polarization_type="V",
                                              antenna_pattern="38.901",
                                              carrier_frequency=self._carrier_frequency)

            self._cdl = CDL(model=self._cdl_model,
                            delay_spread=self._delay_spread,
                            carrier_frequency=self._carrier_frequency,
                            ut_array=self._ut_array,
                            bs_array=self._bs_array,
                            direction=self._direction,
                            min_speed=self._min_speed,
                            max_speed=self._max_speed)

            self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
            self._ul_frequencies = self._frequencies[:self._num_subcarriers]
            self._dl_frequencies = self._frequencies[-self._num_subcarriers:]

            if self._domain == "freq":
                self._channel_freq = ApplyOFDMChannel(add_awgn=True)
            elif self._domain == "time":
                self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
                self._l_tot = self._l_max - self._l_min + 1
                self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                      l_tot=self._l_tot,
                                                      add_awgn=True)
                self._modulator = OFDMModulator(self._cyclic_prefix_length)
                self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

                if "baseline" in system:
                    if system == 'baseline-perfect-csi':
                        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)
                    elif system == 'baseline-ls-estimation':
                        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
                    self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
                    self._mapper = Mapper("qam", self._num_bits_per_symbol)
                    self._demapper = Demapper("app", "qam", args.num_bits_per_symbol)
                    
                elif system == "neural-receiver":
                    
                    if args.pilot_loc is not None:
                        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
                        
                    # neural modulation / just sip / neural modulation with sip
                    if self._precoder_mode == "nm":
                        if self._precoder_arch == "full":
                            self._neural_mod = NeuralPrecoder(self._num_conv_channels_precoder, self._num_filters)
                            # TODO: print the number of parameters in the self._neural_mod
                        elif args.precoder_arch == "linear_1l":
                            self._neural_mod = NeuralPrecoder_1_layer_linear(self._kernel_size, self._num_filters)
                        elif args.precoder_arch == "linear_1l_relu":
                            self._neural_mod = NeuralPrecoder_1_layer_relu(self._kernel_size, self._num_filters)
                        elif args.precoder_arch == "time_1D":
                            self._neural_mod = NeuralPrecoder1D(self._kernel_size, self._num_filters, direction="time")
                        elif args.precoder_arch == "freq_1D":
                            self._neural_mod = NeuralPrecoder1D(self._kernel_size, self._num_filters, direction="freq")

                    elif self._precoder_mode == "sip":
                        self._neural_mod_sip = NeuralPrecoderSIP(self._num_conv_channels_precoder, self._num_filters)

                    elif self._precoder_mode == "nm_sip":
                        self._neural_mod = NeuralPrecoder(self._num_conv_channels_precoder, self._num_filters)
                        self._neural_mod_sip = NeuralPrecoderSIP(self._num_conv_channels_precoder, self._num_filters)


                    if self._precoder_mode == "sip" or self._precoder_mode =="nm_sip":
                        self._neural_receiver = NeuralReceiverSIP(self._num_conv_channels_receiver, self._num_bits_per_symbol)

                    else:
                        if args.pilot_loc is not None:
                            self._neural_receiver = NeuralReceiverPilot(self._num_conv_channels_receiver, self._num_bits_per_symbol)
                        else:
                            self._neural_receiver = NeuralReceiver(self._num_conv_channels_receiver, self._num_bits_per_symbol)
                            # TODO: print the number of parameters in the self._neural_receiver
                    
                    
                    self._rg_demapper = ResourceGridDemapper(self._rg, self._sm)
                    if args.modulation == "neural":
                        constellation = Constellation("qam", self._num_bits_per_symbol, trainable=True, center=True, normalize=True)
                        self.constellation = constellation
                        self._mapper = Mapper("custom", self._num_bits_per_symbol, constellation=constellation)
                    else:
                        self._mapper = Mapper("qam", self._num_bits_per_symbol)
            
            self._binary_source = BinarySource()
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._rg_mapper = ResourceGridMapper(self._rg)

            if self._direction == "downlink":
                self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)

            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        # --- helper: count and log params of submodules once they're built ---
        def _count_layer_params(self, layer):
            if layer is None:
                return 0
            # Ensure variables exist; if not built yet, this will be 0 and we’ll just skip
            tvars = getattr(layer, "trainable_variables", [])
            ntvars = getattr(layer, "non_trainable_variables", [])
            return int(sum(tf.keras.backend.count_params(v) for v in (tvars + ntvars)))

        def log_component_param_counts(self, prefix=""):
            try:
                if hasattr(self, "_neural_mod") and self._neural_mod is not None:
                    n = self._count_layer_params(self._neural_mod)
                    logging.info("%sNeural precoder params: %d", prefix, n)
                    print(f"{prefix}Neural precoder params: {n}")
                if hasattr(self, "_neural_receiver") and self._neural_receiver is not None:
                    n = self._count_layer_params(self._neural_receiver)
                    logging.info("%sNeural receiver params: %d", prefix, n)
                    print(f"{prefix}Neural receiver params: {n}")
            except Exception as e:
                logging.warning("Could not log component param counts: %s", e)

        @tf.function
        def call(self, batch_size, ebno_db):
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)

            if self._training:
                c = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
            else:
                b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
                c = self._encoder(b)

            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            
            if args.pilot_loc is None:
                x_rg_data = x_rg

            # extract the data from ofdm frame
            non_pilot_symbols = np.setdiff1d(np.arange(self._num_ofdm_symbols), self._pilot_ofdm_symbol_indices)
            if args.pilot_loc is not None and self._precoder_mode == "nm":
                x_rg_data = tf.gather(x_rg, non_pilot_symbols, axis=-2)

            # extact the pilots from ofdm frame
            pilot_mask = np.zeros((self._num_ofdm_symbols,), dtype=np.float32)
            pilot_mask[self._pilot_ofdm_symbol_indices] = 1.0
            pilot_mask = tf.constant(pilot_mask)          # shape: [num_symbols]

            pilot_mask = tf.reshape(
                pilot_mask,
                [1, 1, 1, self._num_ofdm_symbols, 1]
            )

            pilot_mask = tf.cast(pilot_mask, dtype=x_rg.dtype)
            x_rg_pilot = x_rg * pilot_mask

            if self._domain == "time":
                a, tau = self._cdl(batch_size, self._rg.num_time_samples * self._num_delay + self._l_tot - 1, self._rg.bandwidth)
                
                h_time = cir_to_time_channel(self._rg.bandwidth, a, tau, l_min=self._l_min, l_max=self._l_max, normalize=True) 
                h_freq = time_to_ofdm_channel(h_time, self._rg, self._l_min)
                
                if self._direction == "downlink":
                    x_rg, g = self._zf_precoder([x_rg, h_freq])
                
                if self._precoder_mode == "nm" or self._precoder_mode == "nm_sip" or self._precoder_mode == "sip":
                    energy_input = tf.reduce_mean(tf.abs(x_rg)**2, axis=[1,2,3,4], keepdims=True)
                    energy_input_data = tf.reduce_mean(tf.abs(x_rg_data)**2, axis=[1,2,3,4], keepdims=True)
                                
                if self._system == "neural-receiver":
                    if self._precoder_mode == "nm":

                        x_rg_data = self._neural_mod(tf.squeeze(x_rg_data, axis=1))
                        
                        energy_output_data = tf.reduce_mean(tf.abs(x_rg_data)**2, axis=[1, 2, 3, 4], keepdims=True)
                        scale = tf.cast(tf.sqrt(energy_input_data / (energy_output_data + 1e-12)), x_rg.dtype)
                        x_rg_data = x_rg_data * scale
                        
                        batch_size = tf.shape(x_rg)[0]
                        num_subcarriers = tf.shape(x_rg)[-1]
                        num_data_symbols = tf.shape(x_rg_data)[3]

                        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
                        batch_indices = tf.tile(batch_indices, [1, num_data_symbols, num_subcarriers])
                        dim1_indices = tf.zeros_like(batch_indices)
                        dim2_indices = tf.zeros_like(batch_indices)
                        symbol_indices = tf.convert_to_tensor(non_pilot_symbols, dtype=tf.int32)
                        symbol_indices = symbol_indices[tf.newaxis, :, tf.newaxis]
                        symbol_indices = tf.tile(symbol_indices, [batch_size, 1, num_subcarriers])
                        subcarrier_indices = tf.range(num_subcarriers)[tf.newaxis, tf.newaxis, :]
                        subcarrier_indices = tf.tile(subcarrier_indices, [batch_size, num_data_symbols, 1])
                        scatter_indices = tf.stack(
                            [batch_indices, dim1_indices, dim2_indices, symbol_indices, subcarrier_indices], axis=-1
                        )
                        scatter_indices = tf.reshape(scatter_indices, [-1, 5])
                        updates = tf.reshape(x_rg_data, [-1])
                        x_rg = tf.tensor_scatter_nd_update(
                            tensor=x_rg,
                            indices=scatter_indices,
                            updates=updates
                        )

                    elif self._precoder_mode == "sip" or self._precoder_mode == "nm_sip":
                        
                        # modulation
                        if self._precoder_mode == "nm_sip":
                            x_rg = self._neural_mod(tf.squeeze(x_rg, axis=1))

                        # sip
                        x_rg = self._neural_mod_sip(tf.squeeze(x_rg, axis=1))

                        energy_output = tf.reduce_mean(tf.abs(x_rg)**2, axis=[1, 2, 3, 4], keepdims=True)
                        scale = tf.cast(tf.sqrt(energy_input / (energy_output + 1e-12)), x_rg.dtype)
                        x_rg = x_rg * scale
                    
                x_time = self._modulator(x_rg)
                y_time = self._channel_time([x_time, h_time, no])
                y = self._demodulator(y_time)

                no = tf.fill([batch_size], no)
                if "baseline" in self._system:
                    if self._system == 'baseline-perfect-csi':
                        if self._direction == "uplink":
                            h_hat = self._remove_nulled_scs(h_freq)
                        elif self._direction == "downlink":
                            h_hat = g
                        err_var = 0.0
                    elif self._system == 'baseline-ls-estimation':
                        h_hat, err_var = self._ls_est([y, no])
                    x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
                    no_eff_ = expand_to_rank(no_eff, tf.rank(x_hat))
                    llr = self._demapper([x_hat, no_eff_])
                elif self._system == "neural-receiver":
                    y = tf.squeeze(y, axis=1)
                    if args.pilot_loc is not None:
                        llr = self._neural_receiver([y, tf.squeeze(x_rg_pilot, axis=[1]), no])
                    else:
                        llr = self._neural_receiver([y, no])

                    llr = insert_dims(llr, 2, 1)
                    llr = self._rg_demapper(llr)
                    llr = tf.reshape(llr, [batch_size, 1, 1, self._n])

            if self._training:
                bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
                bce = tf.reduce_mean(bce)
                rate = tf.constant(1.0, tf.float32) - bce / tf.math.log(2.)
                return rate
            else:
                b_hat = self._decoder(llr)
                return b, b_hat

    start_time = time.time()

    # Training Phase
    if args.training:
        # Set random seed for reproducibility - ONLY DURING TRAINING.
        sionna.config.seed = args.random_seed
        # By default, training starts from scratch unless --continue_training is specified.
        model = Model(domain="time",
                      direction=args.direction,
                      cdl_model=args.cdl_model,
                      delay_spread=args.delay_spread,
                      perfect_csi=args.perfect_csi,
                      min_speed=args.min_speed,
                      max_speed=args.max_speed,
                      cyclic_prefix_length=args.cyclic_prefix_length,
                      pilot_ofdm_symbol_indices=args.pilot_loc,
                      carrier_frequency=args.carrier_frequency,
                      subcarrier_spacing=args.subcarrier_spacing,
                      total_num_subcarriers=args.total_num_subcarriers,
                      num_delay=args.num_delay,
                      num_subcarriers=args.num_subcarriers,
                      training=True)
        # Use a cosine annealing scheduler to decay from 1e-3 to 5e-5.
        # Note: alpha = final_lr / initial_lr = 5e-5/1e-3 = 0.05.
        lr_schedule = CosineDecay(
            initial_learning_rate=args.initial_learning_rate,
            decay_steps=num_training_iterations,
            alpha=0.05
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        start_iter = 0
        model(1, tf.constant(10.0, tf.float32))
        model.log_component_param_counts(prefix="[INFER] ")
        # If continue_training is specified, search for the latest checkpoint file.
        if args.continue_training:
            latest_ckpt = get_latest_file(f"checkpoint_{params}_*.pkl")
            if latest_ckpt is not None:
                
                # print the latest checkpoint file
                logging.info("Loading latest checkpoint from %s", latest_ckpt)
                model(1, tf.constant(10.0, tf.float32))
                with open(latest_ckpt, 'rb') as f:
                    checkpoint = pickle.load(f)
                start_iter = checkpoint['iteration'] + 1
                model.set_weights(checkpoint['model_weights'])
                if checkpoint.get('optimizer_weights') is not None:
                    optimizer.set_weights(checkpoint['optimizer_weights'])
                logging.info("Resuming training from iteration %d", start_iter)
            else:
                logging.info("No checkpoint found. Training from scratch.")
        else:
            logging.info("Training from scratch.")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in tqdm(range(start_iter, num_training_iterations)):
            ebno_db = tf.random.uniform(shape=[], minval=args.training_ebno_min, maxval=args.training_ebno_max)
            with tf.GradientTape() as tape:
                rate = model(training_batch_size, ebno_db)
                loss = -rate
            weights = model.trainable_weights
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

            if i == start_iter:
                # capture & print/log a wider summary
                buf = io.StringIO()
                model.summary(print_fn=lambda x: buf.write(x+"\n"), line_length=100, positions=[0.30,0.65,1.0])
                summary = buf.getvalue()
                print(summary)
                logging.info("\n%s", summary)

                # log simple param counts
                total = model.count_params()
                trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
                logging.info("Params → total=%d, trainable=%d, non-trainable=%d",
                            total, trainable, total - trainable)



            if i % 1000 == 0:
                print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')
                logging.info('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()))
                
                if i % 10000 == 0:
                    # Generate unique filenames for this checkpoint
                    ckpt_filename = f"checkpoint_{params}_{run_id}_iter_{i}.pkl"
                    ckpt_filepath = os.path.join(weights_folder, ckpt_filename)
                    model_weights_filename = f"model_weights_{params}_{run_id}_iter_{i}.h5"
                    model_weights_filepath = os.path.join(weights_folder, model_weights_filename)
                    # Save checkpoint (model & optimizer state, iteration)
                    try:
                        optimizer_weights = optimizer.get_weights()
                    except AttributeError:
                        optimizer_weights = None
                    checkpoint = {
                        'iteration': i,
                        'model_weights': model.get_weights(),
                        'optimizer_weights': optimizer_weights
                    }
                    with open(ckpt_filepath, 'wb') as f:
                        pickle.dump(checkpoint, f)
                    # Also save model weights separately for inference
                    with open(model_weights_filepath, 'wb') as f:
                        pickle.dump(model.get_weights(), f)

                if i % (num_training_iterations // args.num_vals) == 0:
                    current_lr = optimizer.learning_rate.numpy()
                    logging.info('Step %d: Learning Rate = %.2e', i, current_lr)
                    constellation_fig = model._mapper.constellation.show()
                    constellation_fig.savefig(os.path.join(results_folder, f"constellation_{params}.pdf"))
                    
                    # model._training = False
                    # tf.config.run_functions_eagerly(False)
                    # eb_no_val = np.arange(args.training_ebno_min, args.training_ebno_max, args.training_ebno_step)
                    # ber, bler = sim_ber(model,
                    #                     eb_no_val,
                    #                     batch_size=256,
                    #                     max_mc_iter=100,
                    #                     num_target_block_errors=100,
                    #                     target_bler=1e-3)
                    # logging.info("EB/N0 (dB): %s", eb_no_val)
                    # logging.info("BER: %s", ber.numpy())
                    # logging.info("BLER: %s", bler.numpy())
                    # model._training = True
                    # tf.config.run_functions_eagerly(False)

    ### Inference Phase ###
    tf.config.run_functions_eagerly(args.debug)
    BER = {}
    BLER = {}

    ebno_dbs = np.arange(args.inference_ebno_min, args.inference_ebno_max, args.inference_ebno_step)

    # During inference, search for the latest model weights file saved.
    # During inference, search for the latest model weights file saved.
    if args.weights_path is not None:
        latest_model = args.weights_path
    else:
        latest_model = get_latest_file(f"model_weights_{params}_*.h5")
    
        if latest_model is None:
            raise ValueError("No saved model found for inference.")
        else:
            logging.info("Loading latest model weights from %s", latest_model)
            # get the iteration number from model_name 
            latest_iter = int(re.search(r'_(\d+)\.h5$', latest_model).group(1))

    model = Model(domain="time",
                  direction=args.direction,
                  cdl_model=args.cdl_model,
                  delay_spread=args.delay_spread,
                  perfect_csi=args.perfect_csi,
                  min_speed=args.inf_speed,
                  max_speed=None,
                  cyclic_prefix_length=args.cyclic_prefix_length,
                  pilot_ofdm_symbol_indices=args.pilot_loc,
                  carrier_frequency=args.carrier_frequency,
                  subcarrier_spacing=args.subcarrier_spacing,
                  num_subcarriers=args.num_subcarriers,
                  total_num_subcarriers=args.total_num_subcarriers,
                  num_delay=args.num_delay,
                  system="neural-receiver")
    # Build model layers
    model(1, tf.constant(10.0, tf.float32))
    model.log_component_param_counts(prefix="[TRAIN] ")

    with open(latest_model, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    constellation_fig = model._mapper.constellation.show()
    constellation_fig.savefig(os.path.join(results_folder, f"constellation_{params}.pdf"))

    ber, bler = sim_ber(model, ebno_dbs,
                        batch_size=args.inference_batch_size,
                        max_mc_iter=args.max_mc_iter,
                        num_target_block_errors=args.num_target_block_errors,
                        target_bler=args.target_bler)

    BER['neural-receiver'] = ber.numpy()
    BLER['neural-receiver'] = bler.numpy()

    # Perfect CSI baseline
    model = Model(domain="time",
                  direction=args.direction,
                  cdl_model=args.cdl_model,
                  delay_spread=args.delay_spread,
                  perfect_csi=args.perfect_csi,
                  min_speed=args.inf_speed,
                  max_speed=None,
                  cyclic_prefix_length=args.cyclic_prefix_length,
                  pilot_ofdm_symbol_indices=args.pilot_loc,
                  carrier_frequency=args.carrier_frequency,
                  subcarrier_spacing=args.subcarrier_spacing,
                  num_subcarriers=args.num_subcarriers,
                  total_num_subcarriers=args.total_num_subcarriers,
                  num_delay=args.num_delay,
                  system="baseline-perfect-csi")
    ber, bler = sim_ber(model, ebno_dbs,
                        batch_size=args.inference_batch_size,
                        max_mc_iter=args.max_mc_iter,
                        num_target_block_errors=args.num_target_block_errors,
                        target_bler=args.target_bler)
    BER['baseline-perfect-csi'] = ber.numpy()
    BLER['baseline-perfect-csi'] = bler.numpy()

    # if args.pilot_loc is not None:

    #     # LS estimation baseline
    #     model = Model(domain="time",
    #                 direction=args.direction,
    #                 cdl_model=args.cdl_model,
    #                 delay_spread=args.delay_spread,
    #                 perfect_csi=args.perfect_csi,
    #                 min_speed=args.inf_speed,
    #                 max_speed=None,
    #                 cyclic_prefix_length=args.cyclic_prefix_length,
    #                 pilot_ofdm_symbol_indices=args.pilot_loc,
    #                 carrier_frequency=args.carrier_frequency,
    #                 subcarrier_spacing=args.subcarrier_spacing,
    #                 num_subcarriers=args.num_subcarriers,
    #                 total_num_subcarriers=args.total_num_subcarriers,
    #                 num_delay=args.num_delay,
    #                 system="baseline-ls-estimation")
    #     ber, bler = sim_ber(model, ebno_dbs,
    #                         batch_size=args.inference_batch_size,
    #                         max_mc_iter=args.max_mc_iter,
    #                         num_target_block_errors=args.num_target_block_errors,
    #                         target_bler=args.target_bler)
    #     BER['baseline-ls-estimation'] = ber.numpy()
    #     BLER['baseline-ls-estimation'] = bler.numpy()

    # Plotting BLER
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c='C0', label='Baseline - Perfect CSI')
    # if args.pilot_loc is not None:
    #     plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c='C1', label='Baseline - LS Estimation')
    plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c='C2', label='Neural receiver')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"bler_{params}_{args.inf_speed}_{latest_iter}_bs_{args.inference_batch_size}.pdf"))
    
    # Plotting BER
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, BER['baseline-perfect-csi'], 'o-', c='C0', label='Baseline - Perfect CSI')
    # if args.pilot_loc is not None:
    #     plt.semilogy(ebno_dbs, BER['baseline-ls-estimation'], 'x--', c='C1', label='Baseline - LS Estimation')
    plt.semilogy(ebno_dbs, BER['neural-receiver'], 's-.', c='C2', label='Neural receiver')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"ber_{params}_{args.inf_speed}_{latest_iter}_bs_{args.inference_batch_size}.pdf"))
    
    print("EB/N0 (dB):", list(ebno_dbs))
    print("BER:", {key: list(value) for key, value in BER.items()})
    print("BLER:", {key: list(value) for key, value in BLER.items()})
    
    logging.info("EB/N0 (dB): %s", list(ebno_dbs))
    logging.info("BER: %s", {key: list(value) for key, value in BER.items()})
    logging.info("BLER: %s", {key: list(value) for key, value in BLER.items()})

if __name__ == "__main__":
    main()
