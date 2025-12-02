import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv1D, SeparableConv2D, SeparableConv1D, BatchNormalization
from tensorflow.keras.activations import relu
from sionna.utils import log10, insert_dims

###############################################################################
# ResidualBlock with Projection Skip
###############################################################################
class ResidualBlock(Layer):
    r"""
    This Keras layer implements a residual block:
       BN -> ReLU -> SeparableConv2D -> BN -> ReLU -> SeparableConv2D
    plus a skip connection.

    It supports an optional 1x1 projection on the skip path if the input
    channels differ from the output channels.

    Parameters:
      in_channels: Number of channels in the input tensor.
      out_channels: Number of channels produced by this block.
      kernel_size: Kernel size for the SeparableConv2D layers.
      dilation_rate: Dilation rate for the SeparableConv2D layers.

    Input shape:  [batch, H, W, in_channels]
    Output shape: [batch, H, W, out_channels]
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3,3),
                 dilation_rate=(1,1),
                 **kwargs):
        super().__init__(**kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate

    def build(self, input_shape):
        # First BN -> ReLU -> Conv2D
        self._bn_1 = BatchNormalization()
        self._conv_1 = SeparableConv2D(filters=self._out_channels,
                                       kernel_size=self._kernel_size,
                                       dilation_rate=self._dilation_rate,
                                       padding='same',
                                       activation=None)
        # Second BN -> ReLU -> SeparableConv2D
        self._bn_2 = BatchNormalization()
        self._conv_2 = SeparableConv2D(filters=self._out_channels,
                                       kernel_size=self._kernel_size,
                                       dilation_rate=self._dilation_rate,
                                       padding='same',
                                       activation=None)
        # If input channels differ from output channels, project via 1x1 conv.
        if self._in_channels != self._out_channels:
            self._skip_conv = Conv2D(filters=self._out_channels,
                                     kernel_size=1,
                                     padding='same',
                                     use_bias=False)
        else:
            self._skip_conv = None

    def call(self, inputs):
        z = self._bn_1(inputs)
        z = relu(z)
        z = self._conv_1(z)

        z = self._bn_2(z)
        z = relu(z)
        z = self._conv_2(z)

        # Apply projection on the skip path if needed.
        if self._skip_conv is not None:
            inputs = self._skip_conv(inputs)
        return z + inputs
    
# class ResidualBlock(Layer):
#     r"""
#     This Keras layer implements a residual block:
#        BN -> ReLU -> Conv2D -> BN -> ReLU -> Conv2D
#     plus a skip connection.

#     It supports an optional 1x1 projection on the skip path if the input
#     channels differ from the output channels.

#     Parameters:
#       in_channels: Number of channels in the input tensor.
#       out_channels: Number of channels produced by this block.
#       kernel_size: Kernel size for the Conv2D layers.
#       dilation_rate: Dilation rate for the Conv2D layers.

#     Input shape:  [batch, H, W, in_channels]
#     Output shape: [batch, H, W, out_channels]
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=(3,3),
#                  dilation_rate=(1,1),
#                  **kwargs):
#         super().__init__(**kwargs)
#         self._in_channels = in_channels
#         self._out_channels = out_channels
#         self._kernel_size = kernel_size
#         self._dilation_rate = dilation_rate

#     def build(self, input_shape):
#         # First BN -> ReLU -> Conv2D
#         self._bn_1 = BatchNormalization()
#         self._conv_1 = Conv2D(filters=self._out_channels,
#                                        kernel_size=self._kernel_size,
#                                        dilation_rate=self._dilation_rate,
#                                        padding='same',
#                                        activation=None)
#         # Second BN -> ReLU -> Conv2D
#         self._bn_2 = BatchNormalization()
#         self._conv_2 = Conv2D(filters=self._out_channels,
#                                        kernel_size=self._kernel_size,
#                                        dilation_rate=self._dilation_rate,
#                                        padding='same',
#                                        activation=None)
#         # If input channels differ from output channels, project via 1x1 conv.
#         if self._in_channels != self._out_channels:
#             self._skip_conv = Conv2D(filters=self._out_channels,
#                                      kernel_size=1,
#                                      padding='same',
#                                      use_bias=False)
#         else:
#             self._skip_conv = None

#     def call(self, inputs):
#         z = self._bn_1(inputs)
#         z = relu(z)
#         z = self._conv_1(z)

#         z = self._bn_2(z)
#         z = relu(z)
#         z = self._conv_2(z)

#         # Apply projection on the skip path if needed.
#         if self._skip_conv is not None:
#             inputs = self._skip_conv(inputs)
#         return z + inputs

###############################################################################
# NeuralReceiver
###############################################################################
class NeuralReceiver(Layer):
    r"""
    A residual convolutional neural receiver.
      - An input Conv2D with self._num_conv_channels filters.
      - 5 ResidualBlocks, each operating at 2*self._num_conv_channels channels.
        (Kernel sizes & dilation rates are as defined by your table.)
      - An output Conv2D mapping to num_bits_per_symbol filters.

    Input:
      y : [batch, num_rx_antenna, num_ofdm_symbols, num_subcarriers], tf.complex
      no : [batch], tf.float32

    Output:
      [batch, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol]
    """
    def __init__(self, num_conv_channels, num_bits_per_symbol, **kwargs):
        super().__init__(**kwargs)
        self._num_conv_channels = num_conv_channels
        self._num_bits_per_symbol = num_bits_per_symbol

    def build(self, input_shape):
        # 1) Input conv: uses self._num_conv_channels filters.
        self._input_conv = Conv2D(filters=self._num_conv_channels,
                                  kernel_size=(3,3),
                                  dilation_rate=(1,1),
                                  padding='same',
                                  activation=None)
        # 2) Residual Blocks: each will output 2*self._num_conv_channels channels.
        # Block 1: projects from self._num_conv_channels to 2*self._num_conv_channels.
        out = 2*self._num_conv_channels
        self._res_block_1 = ResidualBlock(in_channels=self._num_conv_channels,
                                          out_channels=out,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,2))
        # Blocks 2-5: maintain 2*self._num_conv_channels channels.
        # self._res_block_2 = ResidualBlock(in_channels=out,
        #                                   out_channels=out,
        #                                   kernel_size=(7,5),
        #                                   dilation_rate=(7,1))
        self._res_block_3 = ResidualBlock(in_channels=out,
                                          out_channels=out,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_4 = ResidualBlock(in_channels=out,
                                          out_channels=out,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_5 = ResidualBlock(in_channels=out,
                                          out_channels=out,
                                          kernel_size=(3,3),
                                          dilation_rate=(1,1))
        # 3) Output conv: maps to num_bits_per_symbol filters.
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol,
                                   kernel_size=(1,1),
                                   dilation_rate=(1,1),
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        y, no = inputs
        # Convert noise variance to log10 scale.
        no = log10(no)
        # Rearrange y from [batch, num_rx_ant, sym, subc] to [batch, sym, subc, num_rx_ant].
        y = tf.transpose(y, [0, 2, 3, 1])
        # Expand no to match spatial dimensions and tile.
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, tf.shape(y)[1], tf.shape(y)[2], 1])
        # Concatenate real and imaginary parts of y with noise.
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Pass through the input conv and the residual blocks.
        z = self._input_conv(z)                        # [batch, sym, subc, num_conv_channels]
        z = self._res_block_1(z)                         # -> [batch, sym, subc, 2*num_conv_channels]
        # z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        z = self._res_block_5(z)
        z = self._output_conv(z)                         # [batch, sym, subc, num_bits_per_symbol]
        return z

###############################################################################
# NeuralPrecoder
###############################################################################
class NeuralPrecoder(Layer):
    r"""
    A neural precoder-like network with:
      - An input Conv2D for x with self._num_conv_channels filters.
      - A ResidualBlock that projects from self._num_conv_channels to 2*self._num_conv_channels.
      - A projection Conv2D that outputs 2*self._num_conv_channels filters.
      - 4 main ResidualBlocks operating at 2*self._num_conv_channels channels.
      - A final Conv2D that produces 2*num_filters channels (for real and imaginary parts).
      - A skip connection from the original x.

    All layers assume channels-last format: [batch, sym, subc, channel].

    Input:
      x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

    Output:
      [batch, 1, num_ofdm_symbols, num_subcarriers], tf.complex
    """
    def __init__(self, num_conv_channels, num_filters, **kwargs):
        super().__init__(**kwargs)
        self._num_conv_channels = num_conv_channels
        self._num_filters = num_filters

    def build(self, input_shape):
        # Branch for x:
        # 1) Input conv: uses self._num_conv_channels filters.
        self._input_conv_x = Conv2D(filters=self._num_conv_channels,
                                    kernel_size=(3,3),
                                    dilation_rate=(1,1),
                                    padding='same',
                                    activation=None)
        # 2) Residual block for x: projects from self._num_conv_channels to 2*self._num_conv_channels.
        out = 2*self._num_conv_channels
        self._res_block_x = ResidualBlock(in_channels=self._num_conv_channels,
                                          out_channels= out,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,2))
        # 3) Projection conv: now outputs 2*self._num_conv_channels filters.
        self._projection_conv = Conv2D(filters=out,
                                       kernel_size=(1,1),
                                       padding='same',
                                       activation=None)
        # 4) Four main residual blocks (maintaining 2*self._num_conv_channels).
        self._res_block_1 = ResidualBlock(in_channels=out,
                                          out_channels=out,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,1))
        self._res_block_2 = ResidualBlock(in_channels=out,
                                          out_channels=out,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        # self._res_block_3 = ResidualBlock(in_channels=out,
        #                                   out_channels=out,
        #                                   kernel_size=(5,3),
        #                                   dilation_rate=(1,2))
        # self._res_block_4 = ResidualBlock(in_channels=out,
        #                                   out_channels=out,
        #                                   kernel_size=(3,3),
        #                                   dilation_rate=(1,1))
        # 5) Final output conv: produces 2*num_filters channels (for real and imaginary parts).
        self._output_conv = Conv2D(filters=self._num_filters * 2,
                                   kernel_size=(1,1),
                                   dilation_rate=(1,1),
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        """
        Input:
          x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        """
        x = inputs  # [batch, tx_ant, sym, subc]
        # Move tx_ant to the last dimension: [batch, sym, subc, tx_ant]
        x = tf.transpose(x, [0, 2, 3, 1])
        # Separate real and imaginary parts.
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        # Concatenate real and imaginary parts along the channel dimension.
        x_in = tf.concat([x_real, x_imag], axis=-1)
        # Process the branch: input conv then residual block.
        x_in = self._input_conv_x(x_in)         # [batch, sym, subc, num_conv_channels]
        x_in = self._res_block_x(x_in)            # -> [batch, sym, subc, 2*num_conv_channels]
        # Merge features using the projection conv.
        z = self._projection_conv(x_in)           # [batch, sym, subc, 2*num_conv_channels]
        # Pass through the four main residual blocks.
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        # z = self._res_block_3(z)
        # z = self._res_block_4(z)
        # Final output conv produces 2*num_filters channels.
        z = self._output_conv(z)
        # Split channels into real and imaginary parts.
        z_real = z[..., :self._num_filters]
        z_imag = z[..., self._num_filters:]
        z_complex = tf.complex(z_real, z_imag)
        # Add skip connection from original x (using its real and imaginary parts).
        skip_x = tf.complex(x_real, x_imag)
        z_complex = z_complex + skip_x
        # Rearrange to [batch, 1, sym, subc]
        z_complex = tf.transpose(z_complex, [0, 3, 1, 2])
        z_complex = tf.expand_dims(z_complex, axis=1)
        return z_complex



###############################################################################
# NeuralPrecoder with SIP
###############################################################################
class NeuralPrecoderSIP(Layer):
    r"""
    A neural precoder-like network with:
      - An input Conv2D for x with self._num_conv_channels filters.
      - A ResidualBlock that projects from self._num_conv_channels to 2*self._num_conv_channels.
      - A projection Conv2D that outputs 2*self._num_conv_channels filters.
      - 4 main ResidualBlocks operating at 2*self._num_conv_channels channels.
      - A final Conv2D that produces 2*num_filters channels (for real and imaginary parts).
      - A skip connection from the original x.

    All layers assume channels-last format: [batch, sym, subc, channel].

    Input:
      x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

    Output:
      [batch, 1, num_ofdm_symbols, num_subcarriers], tf.complex
    """
    def __init__(self, num_conv_channels, num_filters, **kwargs):
        super().__init__(**kwargs)
        self._num_conv_channels = num_conv_channels
        self._num_filters = num_filters

    def build(self, input_shape):
        # Create the trainable pilot allocation matrix.
        self._A = tf.Variable(tf.random.normal([1, 14, 128], mean=-1, stddev=0.1), trainable=True)
        
        # Compute sip_pilots once based on the fixed dimensions from input_shape.
        # input_shape is [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers].
        tx_range = tf.range(input_shape[1])         # Number of transmit antennas.
        ofdm_range = tf.range(input_shape[2])         # Number of OFDM symbols.
        subcarrier_range = tf.range(input_shape[3])   # Number of subcarriers.
        tx, ofdm, subcarrier = tf.meshgrid(tx_range, ofdm_range, subcarrier_range, indexing='ij')
        
        # For positions where (tx + ofdm + subcarrier) is even, use -1.0; otherwise, use +1.0.
        alternating_real = tf.where((tx + ofdm + subcarrier) % 2 == 0,
                                     tf.cast(-1.0, tf.float32),
                                     tf.cast(1.0, tf.float32))
        # Create an imaginary part filled with zeros.
        alternating_imag = tf.zeros_like(alternating_real)

        self._sip_pilots = tf.expand_dims(tf.complex(alternating_real, alternating_imag), axis=0)

    def call(self, inputs):
        """
        Input:
          x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        """
        A_processed = tf.cast(tf.sigmoid(self._A), dtype=inputs.dtype)
        x = tf.sqrt(1 - A_processed) * inputs + tf.sqrt(A_processed) * self._sip_pilots

        x = tf.expand_dims(x, axis=1)
        return x


###############################################################################
# NeuralReceiver
###############################################################################
class NeuralReceiverSIP(Layer):
    r"""
    A residual convolutional neural receiver.
      - An input Conv2D with self._num_conv_channels filters.
      - 5 ResidualBlocks, each operating at 2*self._num_conv_channels channels.
        (Kernel sizes & dilation rates are as defined by your table.)
      - An output Conv2D mapping to num_bits_per_symbol filters.

    Input:
      y : [batch, num_rx_antenna, num_ofdm_symbols, num_subcarriers], tf.complex
      no : [batch], tf.float32

    Output:
      [batch, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol]
    """
    def __init__(self, num_conv_channels, num_bits_per_symbol, **kwargs):
        super().__init__(**kwargs)
        self._num_conv_channels = num_conv_channels
        self._num_bits_per_symbol = num_bits_per_symbol

    def build(self, input_shape):
        

        # Compute sip_pilots once based on the fixed dimensions from input_shape.
        # input_shape is [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers].
        tx_range = tf.range(1)#input_shape[1])         # Number of transmit antennas.
        ofdm_range = tf.range(14)#input_shape[2])         # Number of OFDM symbols.
        subcarrier_range = tf.range(128)#input_shape[3])   # Number of subcarriers.
        tx, ofdm, subcarrier = tf.meshgrid(tx_range, ofdm_range, subcarrier_range, indexing='ij')
        
        # For positions where (tx + ofdm + subcarrier) is even, use -1.0; otherwise, use +1.0.
        alternating_real = tf.where((tx + ofdm + subcarrier) % 2 == 0,
                                     tf.cast(-1.0, tf.float32),
                                     tf.cast(1.0, tf.float32))
        # Create an imaginary part filled with zeros.
        alternating_imag = tf.zeros_like(alternating_real)

        self._sip_pilots = tf.expand_dims(tf.complex(alternating_real, alternating_imag), axis=0)


        # 1) Input conv: uses self._num_conv_channels filters.
        self._input_conv = Conv2D(filters=self._num_conv_channels,
                                  kernel_size=(3,3),
                                  dilation_rate=(1,1),
                                  padding='same',
                                  activation=None)
        # 2) Residual Blocks: each will output 2*self._num_conv_channels channels.
        # Block 1: projects from self._num_conv_channels to 2*self._num_conv_channels.
        self._res_block_1 = ResidualBlock(in_channels=self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,2))
        # Blocks 2-5: maintain 2*self._num_conv_channels channels.
        self._res_block_2 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,1))
        self._res_block_3 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_4 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_5 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(3,3),
                                          dilation_rate=(1,1))
        # 3) Output conv: maps to num_bits_per_symbol filters.
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol,
                                   kernel_size=(1,1),
                                   dilation_rate=(1,1),
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        y, no = inputs
        # Convert noise variance to log10 scale.
        no = log10(no)
        # Rearrange y from [batch, num_rx_ant, sym, subc] to [batch, sym, subc, num_rx_ant].
        y = tf.transpose(y, [0, 2, 3, 1])

        # Expand no to match spatial dimensions and tile.
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, tf.shape(y)[1], tf.shape(y)[2], 1])

        pilots = tf.tile(self._sip_pilots, [tf.shape(y)[0], 1, 1, 1])

        pilots = tf.transpose(pilots, [0, 2, 3, 1])

        # Concatenate real and imaginary parts of y with noise.
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                        tf.math.real(pilots),
                        tf.math.imag(pilots),
                       no], axis=-1)
        # Pass through the input conv and the residual blocks.
        z = self._input_conv(z)                        # [batch, sym, subc, num_conv_channels]
        z = self._res_block_1(z)                         # -> [batch, sym, subc, 2*num_conv_channels]
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        z = self._res_block_5(z)
        z = self._output_conv(z)                         # [batch, sym, subc, num_bits_per_symbol]
        return z
    

class NeuralReceiverPilot(Layer):
    r"""
    A residual convolutional neural receiver that also takes pilot symbols as input.
      - Input: y, pilot [batch, num_rx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
               no    [batch], tf.float32
      - Same architecture as NeuralReceiver, but input conv sees
        [ real(y), imag(y), real(pilot), imag(pilot), log10(no) ].
      - Output: [batch, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol]
    """
    def __init__(self, num_conv_channels, num_bits_per_symbol, **kwargs):
        super().__init__(**kwargs)
        self._num_conv_channels    = num_conv_channels
        self._num_bits_per_symbol  = num_bits_per_symbol

    def build(self, input_shape):
        # exactly the same blocks as NeuralReceiver
        self._input_conv = Conv2D(filters=self._num_conv_channels,
                                  kernel_size=(3,3),
                                  dilation_rate=(1,1),
                                  padding='same',
                                  activation=None)
        self._res_block_1 = ResidualBlock(in_channels=self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,2))
        self._res_block_2 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(7,5),
                                          dilation_rate=(7,1))
        self._res_block_3 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_4 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(5,3),
                                          dilation_rate=(1,2))
        self._res_block_5 = ResidualBlock(in_channels=2*self._num_conv_channels,
                                          out_channels=2*self._num_conv_channels,
                                          kernel_size=(3,3),
                                          dilation_rate=(1,1))
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol,
                                   kernel_size=(1,1),
                                   dilation_rate=(1,1),
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        # unpack the three inputs
        y, pilot, no = inputs

        # 1) convert noise to log10 and tile
        no = log10(no)
        no = insert_dims(no, 3, 1)   # -> [batch, 1] -> [batch, 1,1,1]
        no = tf.tile(no, [1,
                          tf.shape(y)[2],   # num_ofdm_symbols
                          tf.shape(y)[3],   # num_subcarriers
                          1])               # -> [batch, sym, subc, 1]

        # 2) transpose both y & pilot to spatial layout
        #    [batch, num_rx_ant, sym, subc] -> [batch, sym, subc, num_rx_ant]
        y     = tf.transpose(y,     [0, 2, 3, 1])
        pilot = tf.transpose(pilot, [0, 2, 3, 1])

        # 3) split into real/imag and concat: y, pilot, noise
        z = tf.concat([
            tf.math.real(y),
            tf.math.imag(y),
            tf.math.real(pilot),
            tf.math.imag(pilot),
            no
        ], axis=-1)  # final channel dim = 4*num_rx_ant + 1

        # 4) run through the same residual CNN stack
        z = self._input_conv(z)
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        z = self._res_block_5(z)
        z = self._output_conv(z)

        return z  # shape [batch, sym, subc, num_bits_per_symbol]
    



class NeuralPrecoder_1_layer_linear(Layer):
    r"""
    A simplified neural precoder with a single Conv2D layer.

    Input:
      x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

    Output:
      [batch, 1, num_ofdm_symbols, num_subcarriers], tf.complex
    """
    def __init__(self, kernel_size, num_filters, **kwargs):
        super().__init__(**kwargs)
        self._kernel_size = kernel_size
        self._num_filters = num_filters

    def build(self, input_shape):
        # Single Conv2D layer to process the input
        self._conv = Conv2D(filters=self._num_filters * 2,  # Real and imaginary parts
                            kernel_size=(self._kernel_size, self._kernel_size),
                            padding='same',
                            activation=None)
        # self._bn_1 = BatchNormalization()
        self._bn_1 = BatchNormalization(center=False, scale=False,
                                axis=-1,  # Key change: normalize per-channel
                                momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        """
        Input:
          x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        """
        x = inputs  # [batch, tx_ant, sym, subc]
        # Move tx_ant to the last dimension: [batch, sym, subc, tx_ant]
        x = tf.transpose(x, [0, 2, 3, 1])
        # Separate real and imaginary parts.
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        # Concatenate real and imaginary parts along the channel dimension.
        x_in = tf.concat([x_real, x_imag], axis=-1)
        # Process the input through the single Conv2D layer.
        z = self._conv(x_in)
        z = self._bn_1(z)
        
        # Split channels into real and imaginary parts.
        z_real = z[..., :self._num_filters]
        z_imag = z[..., self._num_filters:]
        z_complex = tf.complex(z_real, z_imag)
        # Rearrange to [batch, 1, sym, subc]
        z_complex = tf.transpose(z_complex, [0, 3, 1, 2])
        z_complex = tf.expand_dims(z_complex, axis=1)
        return z_complex


class NeuralPrecoder_1_layer_relu(Layer):
    r"""
    A simplified neural precoder with a single Conv2D layer.

    Input:
      x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

    Output:
      [batch, 1, num_ofdm_symbols, num_subcarriers], tf.complex
    """
    def __init__(self, kernel_size, num_filters, **kwargs):
        super().__init__(**kwargs)
        self._kernel_size = kernel_size
        self._num_filters = num_filters

    def build(self, input_shape):
        # Single Conv2D layer to process the input
        self._conv = Conv2D(filters=self._num_filters * 2,  # Real and imaginary parts
                            kernel_size=(self._kernel_size, self._kernel_size),
                            padding='same',
                            activation=None)
        # self._bn_1 = BatchNormalization()
        self._bn_1 = BatchNormalization(center=False, scale=False,
                                axis=-1,  # Key change: normalize per-channel
                                momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        """
        Input:
          x : [batch, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        """
        x = inputs  # [batch, tx_ant, sym, subc]
        # Move tx_ant to the last dimension: [batch, sym, subc, tx_ant]
        x = tf.transpose(x, [0, 2, 3, 1])
        # Separate real and imaginary parts.
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        # Concatenate real and imaginary parts along the channel dimension.
        x_in = tf.concat([x_real, x_imag], axis=-1)
        # Process the input through the single Conv2D layer.
        z = self._conv(x_in)
        z = self._bn_1(z)
        z = tf.nn.relu(z)
        
        # Split channels into real and imaginary parts.
        z_real = z[..., :self._num_filters]
        z_imag = z[..., self._num_filters:]
        z_complex = tf.complex(z_real, z_imag)
        # Rearrange to [batch, 1, sym, subc]
        z_complex = tf.transpose(z_complex, [0, 3, 1, 2])
        z_complex = tf.expand_dims(z_complex, axis=1)
        return z_complex


class ResidualBlockDirectional(Layer):
    """
    A 2D residual block that performs separable convolutions along a single axis:
      BN -> ReLU -> SeparableConv2D -> BN -> ReLU -> SeparableConv2D
    Supports 'time' (conv along the first spatial axis) or 'freq' (conv along the second).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation_rate=1,
                 direction='time',
                 **kwargs):
        super().__init__(**kwargs)
        if direction not in ('time', 'freq'):
            raise ValueError("direction must be 'time' or 'freq'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.direction = direction

    def build(self, input_shape):
        # Determine 2D kernel and dilation shapes
        if self.direction == 'time':
            ks = (self.kernel_size, 1)
            dr = (self.dilation_rate, 1)
        else:  # freq
            ks = (1, self.kernel_size)
            dr = (1, self.dilation_rate)

        self.bn1 = BatchNormalization()
        self.conv1 = SeparableConv2D(
            filters=self.out_channels,
            kernel_size=ks,
            dilation_rate=dr,
            padding='same',
            activation=None)
        self.bn2 = BatchNormalization()
        self.conv2 = SeparableConv2D(
            filters=self.out_channels,
            kernel_size=ks,
            dilation_rate=dr,
            padding='same',
            activation=None)

        # Optional 1x1 projection on skip path if channel count changes
        if self.in_channels != self.out_channels:
            self.skip_conv = Conv2D(
                filters=self.out_channels,
                kernel_size=1,
                padding='same',
                use_bias=False)
        else:
            self.skip_conv = None

    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        shortcut = self.skip_conv(inputs) if self.skip_conv is not None else inputs
        return x + shortcut


class NeuralPrecoder1D(Layer):
    """
    Neural precoder that applies 2D separable convolutions along a single axis.
    direction='time' performs convs with kernel=(k,1); 'freq' uses (1,k).

    Input:  tf.complex tensor of shape [batch, num_tx_ant, T, F]
    Output: tf.complex tensor of shape [batch, 1, T, F]
    """
    def __init__(self, num_conv_channels, num_filters, direction='time', **kwargs):
        super().__init__(**kwargs)
        if direction not in ('time', 'freq'):
            raise ValueError("direction must be 'time' or 'freq'")
        self.num_conv_channels = num_conv_channels
        self.num_filters = num_filters
        self.direction = direction

    def build(self, input_shape):
        # Channel dimension after concatenating real+imag will be 2*input_tx_ant
        tx_ant = input_shape[-1]
        self.initial_channels = tx_ant * 2

        # Input projection
        # Use Conv2D with 1x1 to adjust channel dimension first
        self.input_proj = Conv2D(
            filters=self.num_conv_channels,
            kernel_size=1,
            padding='same',
            activation=None)

        # Residual blocks sequence
        self.res_blocks = []
        # First block: in -> 2*conv_channels
        self.res_blocks.append(
            ResidualBlockDirectional(
                in_channels=self.num_conv_channels,
                out_channels=2 * self.num_conv_channels,
                kernel_size=7,
                dilation_rate=7,
                direction=self.direction))
        # Projection to maintain channel count
        self.proj_conv = Conv2D(
            filters=2 * self.num_conv_channels,
            kernel_size=1,
            padding='same',
            activation=None)
        # Four subsequent blocks
        self.res_blocks.extend([
            ResidualBlockDirectional(
                in_channels=2 * self.num_conv_channels,
                out_channels=2 * self.num_conv_channels,
                kernel_size=7,
                dilation_rate=1,
                direction=self.direction),
            ResidualBlockDirectional(
                in_channels=2 * self.num_conv_channels,
                out_channels=2 * self.num_conv_channels,
                kernel_size=5,
                dilation_rate=2,
                direction=self.direction),
            ResidualBlockDirectional(
                in_channels=2 * self.num_conv_channels,
                out_channels=2 * self.num_conv_channels,
                kernel_size=5,
                dilation_rate=2,
                direction=self.direction),
            ResidualBlockDirectional(
                in_channels=2 * self.num_conv_channels,
                out_channels=2 * self.num_conv_channels,
                kernel_size=3,
                dilation_rate=1,
                direction=self.direction)
        ])

        # Final output conv: produce 2*num_filters channels
        self.output_conv = Conv2D(
            filters=2 * self.num_filters,
            kernel_size=1,
            padding='same',
            activation=None)

    def call(self, inputs, training=None):
        # inputs: [batch, tx_ant, T, F], complex
        x = tf.transpose(inputs, [0, 2, 3, 1])  # [batch, T, F, tx_ant]
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        z = tf.concat([x_real, x_imag], axis=-1)  # [batch, T, F, 2*tx_ant]

        # Project to conv channels
        z = self.input_proj(z)

        # Apply first residual block
        z = self.res_blocks[0](z, training=training)
        # Projection to maintain channel count
        z = self.proj_conv(z)

        # Apply remaining residual blocks
        for block in self.res_blocks[1:]:
            z = block(z, training=training)

        # Final conv and split into real/imag
        z = self.output_conv(z)
        z_real = z[..., :self.num_filters]
        z_imag = z[..., self.num_filters:]
        z_complex = tf.complex(z_real, z_imag)

        # Skip connection from input
        skip = tf.complex(x_real, x_imag)
        z_complex = z_complex + skip

        # Restore shape [batch, 1, T, F]
        z_complex = tf.transpose(z_complex, [0, 3, 1, 2])
        z_complex = tf.expand_dims(z_complex, axis=1)
        return z_complex
