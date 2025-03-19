import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization,\
                         concatenate, Activation, Input, Conv2DTranspose, Lambda, LSTM, GRU,Reshape, Embedding, GlobalAveragePooling1D,\
                         Multiply,Bidirectional,Layer, MaxPool1D, Conv1DTranspose
import keras.backend as K
from keras import layers
import tensorflow as tf
import numpy as np
from scipy import signal

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D, BatchNormalization, Multiply, Add, Activation
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    x = ExpandDimsLayer(axis=2)(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x

##########################################################################

###### MODULES #######
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import constraints, activations, initializers, regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.constraints import Constraint

class DynamicTanh(tf.keras.layers.Layer):
    def __init__(self, normalized_shape, alpha_init_value=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init_value = alpha_init_value
        self.normalized_shape = normalized_shape

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.alpha_init_value),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.normalized_shape,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.normalized_shape,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        x = tf.math.tanh(self.alpha * inputs)
        return self.gamma * x + self.beta

class FANLayer(tf.keras.layers.Layer):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of the p component (default: True).
        gated (bool): If True, applies gating to the output.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias weights.
    """
    
    def __init__(self, 
                 output_dim, 
                 p_ratio=0.25, 
                 activation='gelu', 
                 use_p_bias=True, 
                 gated=False, 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 **kwargs):
        super(FANLayer, self).__init__(**kwargs)
        
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        
        self.p_ratio = p_ratio
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_p_bias = use_p_bias
        self.gated = gated
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        # Compute output dimensions for p and g components
        self.p_output_dim = int(output_dim * self.p_ratio)
        self.g_output_dim = output_dim - 2 * self.p_output_dim  # Account for cosine and sine
        
        # Layers for linear transformations
        self.input_linear_p = layers.Dense(self.p_output_dim, 
                                    use_bias=self.use_p_bias, 
                                    kernel_regularizer=self.kernel_regularizer, 
                                    bias_regularizer=self.bias_regularizer)
        self.input_linear_g = layers.Dense(self.g_output_dim, 
                                    kernel_regularizer=self.kernel_regularizer, 
                                    bias_regularizer=self.bias_regularizer)
        
        if self.gated:
            self.gate = self.add_weight(name='gate', 
                                        shape=(1,), 
                                        initializer=initializers.RandomNormal(), 
                                        trainable=True, 
                                            regularizer=None, 
                                            constraint=NonNeg())

    def call(self, inputs):
        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(inputs))
        
        # Apply the linear transformation for the p component
        p = self.input_linear_p(inputs)
        
        if self.gated:
            gate = tf.sigmoid(self.gate)
            output = tf.concat([gate * tf.cos(p), gate * tf.sin(p), (1 - gate) * g], axis=-1)
        else:
            output = tf.concat([tf.cos(p), tf.sin(p), g], axis=-1)
        
        return output

    def get_config(self):
        config = super(FANLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "p_ratio": self.p_ratio,
            "activation": activations.serialize(self.activation),
            "use_p_bias": self.use_p_bias,
            "gated": self.gated,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer)
        })
        return config

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Lambda



import tensorflow as tf


def Conv1DTranspose2(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    x = Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)  # (batch, time, 1) -> (batch, time, 1, 1)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)  # (batch, time, 1, filters) -> (batch, time, filters)
    return x

def CNN_DAE(signal_size=512):
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.
    input_shape = (signal_size, 1)
    input_tensor = Input(shape=input_shape)

    # **Encoder (Conv1D Layers)**
    x = Conv1D(filters=40, kernel_size=16, activation='elu', strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)

    x = Conv1D(filters=20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=20, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=40, kernel_size=16, activation='elu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=1, kernel_size=16, activation='elu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # **Decoder (Conv1DTranspose2 Layers)**
    x = Conv1DTranspose2(input_tensor=x, filters=40, kernel_size=16, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x, filters=20, kernel_size=16, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x, filters=20, kernel_size=16, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x, filters=20, kernel_size=16, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x, filters=40, kernel_size=16, activation='elu', strides=2, padding='same')
    x = BatchNormalization()(x)

    # **Fully Connected Layers (Latent Representation)**
    x = Dense(signal_size, activation='elu')(x)  # 512차원으로 맞춤
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)

    # **출력 Shape 맞추기 (Conv1D로 조정)**
    x = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')(x)  # 최종 출력 (batch_size, 512, 1)

    model = Model(inputs=input_tensor, outputs=x)
    return model





def LANLFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 8),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB0 = Conv1D(filters=int(layers / 8),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 8),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 8),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 8),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module_dilated(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x


###### MODELS #######

def deep_filter_I_LANL():
    # TODO: Make the doc

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_model_I_LANL_dilated():
    # TODO: Make the doc

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def FCN_DAE():
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.

    input_shape = (512, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    x = Conv1DTranspose(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=1,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    predictions = Conv1DTranspose(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='linear',
                        strides=1,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model


def DRRN_denoising():
    # Implementation of DRNN approach presented in
    # Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    # arXiv preprint arXiv:1807.11551.    

    model = Sequential()
    model.add(LSTM(64, input_shape=(512, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model

sigLen = 512

##########################################################################
class SpatialGateDep(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(SpatialGateDep, self).__init__()
        self.transpose = transpose
        self.conv = Conv1D(filters, kernel_size, input_shape=input_shape, activation=activation)

    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=1)
        out = self.conv(x)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out

class SpatialGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(SpatialGate, self).__init__()
        self.transpose = transpose
        self.conv = Conv1D(filters, kernel_size, padding='same',
                           input_shape=input_shape, activation=activation)

    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=-1)
        out = self.conv(x)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out

########################################################
############## IMPLEMENT CHANNEL #######################
########################################################

class ChannelGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(ChannelGate, self).__init__()
        self.transpose = transpose
        self.conv = Conv1D(filters, kernel_size,
                           input_shape=input_shape,
                           activation=activation,
                           padding='same')

    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        x = tf.reduce_mean(x, axis=1, keepdims=True)
        x = tf.transpose(x, [0, 2, 1])
        out = self.conv(x)
        out = tf.transpose(out, [0, 2, 1])
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out

########################################################
################# IMPLEMENT CBAM #######################
########################################################

class CBAM(tf.keras.layers.Layer):
    def __init__(self, c_filters, c_kernel, c_input, c_transpose,
                 s_filters, s_kernel, s_input, s_transpose, spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.channel_attention = ChannelGate(c_filters, c_kernel, input_shape=c_input, transpose=c_transpose)
        self.spatial_attention = SpatialGate(s_filters, s_kernel, input_shape=s_input, transpose=s_transpose)

    def call(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x
    

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation=None):
        super(AttentionBlock, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None  # Activation은 이후에 명시적으로 적용
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=None  # Activation은 이후에 명시적으로 적용
            )
        self.activation = tf.keras.layers.LeakyReLU() if activation == 'LeakyReLU' else None
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        if self.activation is not None:
            output = self.activation(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output


class AttentionBlockBN(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None):
        super(AttentionBlockBN, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None  # Activation은 별도로 정의
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=None  # Activation은 별도로 정의
            )
        self.activation = tf.keras.layers.LeakyReLU()
        self.bn = BatchNormalization()
        self.dp = Dropout(rate=0.001)  # rate=0.1 for qtdb
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation=None):
        super(EncoderBlock, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None  # Activation은 이후에 명시적으로 적용
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=None  # Activation은 이후에 명시적으로 적용
            )
        self.activation = tf.keras.layers.LeakyReLU() if activation == 'LeakyReLU' else None
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        if self.activation is not None:
            output = self.activation(output)
        output = self.maxpool(output)
        return output


class AttentionDeconv(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation=None,
                 strides=2, padding='same'):
        super(AttentionDeconv, self).__init__()
        self.deconv = keras.layers.Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=None  # Activation은 이후에 명시적으로 적용
        )
        self.activation = tf.keras.layers.LeakyReLU() if activation == 'LeakyReLU' else None
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )

    def call(self, x):
        output = self.deconv(x)
        if self.activation is not None:
            output = self.activation(output)
        output = self.attention(output)
        return output


class AttentionDeconvBN(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation=None,
                 strides=2, padding='same'):
        super(AttentionDeconvBN, self).__init__()
        self.deconv = keras.layers.Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
        )
        self.bn = BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU() if activation == 'LeakyReLU' else None
        self.dp = Dropout(rate=0.001)  # rate=0.1 for qtdb
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )

    def call(self, x):
        output = self.deconv(x)
        output = self.bn(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dp(output)
        output = self.attention(output)
        return output


class AttentionDeconvECA(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvECA, self).__init__()
        self.deconv = keras.layers.Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        )
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False,
            spatial=False
        )

    def call(self, x):
        output = self.attention(self.deconv(x))
        return output

def AttentionSkipDAE(signal_size=512):
    input_shape = (signal_size, 1)
    inputs = Input(shape=input_shape)
    
    # 인코더 부분
    enc1 = AttentionBlock(signal_size, 16, input_size=(signal_size, 1))(inputs)
    enc2 = AttentionBlock(signal_size//2, 32)(enc1)
    enc3 = AttentionBlock(signal_size//4, 64)(enc2)
    enc4 = AttentionBlock(signal_size//8, 64)(enc3)
    enc5 = AttentionBlock(signal_size//16, 1)(enc4)  # 원래 32
    
    # 디코더 부분 (Skip connections 포함)
    dec5 = AttentionDeconv(signal_size//16, 64)(enc5)
    dec4 = AttentionDeconv(signal_size//8, 64)(Add()([dec5, enc4]))
    dec3 = AttentionDeconv(signal_size//4, 32)(Add()([dec4, enc3]))
    dec2 = AttentionDeconv(signal_size//2, 16)(Add()([dec3, enc2]))
    dec1 = AttentionDeconv(signal_size, 1, activation='linear')(Add()([dec2, enc1]))
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=dec1)
    
    return model

# 모델 사용 예시:
# model = AttentionSkipDAE(signal_size=512)
    
##TCDAE

def transformer_encoder(inputs,head_size,num_heads,ff_dim,dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    # # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x) 
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res
    # FAN Layer 적용 (Feed Forward 부분)
    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    # x = FANLayer(output_dim=ff_dim, p_ratio=0.25, activation="gelu", gated=True)(x)
    # x = layers.Dropout(dropout)(x)
    # x = layers.Dense(inputs.shape[-1])(x)  # 최종 출력 크기 조정
    # return x + res  # Residual connection
# def FANformer_encoder(inputs,head_size,num_heads,ff_dim,dropout=0):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs
#     # FAN Layer 적용 (Feed Forward 부분)
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = FANLayer(output_dim=ff_dim, p_ratio=0.25, activation="gelu", gated=False)(x)
#     x = layers.Dropout(dropout)(x)
#     # x = layers.Dense(inputs.shape[-1])(x)  # 최종 출력 크기 조정
#     x = FANLayer(output_dim=inputs.shape[-1], p_ratio=0.25, activation="linear", gated=False)(x)
#     return x + res  # Residual connection

def FANformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # DyT 대신 LayerNorm
    x = DynamicTanh(normalized_shape=inputs.shape[-1])(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed-forward
    x = DynamicTanh(normalized_shape=res.shape[-1])(res)
    x = FANLayer(output_dim=ff_dim, p_ratio=0.25, activation="gelu", gated=False)(x)
    x = layers.Dropout(dropout)(x)
    x = FANLayer(output_dim=inputs.shape[-1], p_ratio=0.25, activation="linear", gated=False)(x)

    return x + res


# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x1 = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x1, x1)

#     x2 = layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding="same")(inputs)
#     x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
#     x2 = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x2, x2)

#     x3 = layers.Conv1D(filters=inputs.shape[-1], kernel_size=5, padding="same")(inputs)
#     x3 = layers.LayerNormalization(epsilon=1e-6)(x3)
#     x3 = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x3, x3)

#     # Multi-Scale Feature Combination
#     x = layers.Concatenate()([x1, x2, x3])

#     # FFN 적용
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

#     return x + inputs

ks = 13   #orig 13
ks1 = 7

# 학습 중 노이즈를 추가하여 강건성을 높이는 역할
class AddGatedNoise(layers.Layer):
    def __init__(self, **kwargs):
        super(AddGatedNoise, self).__init__(**kwargs)

    def call(self, x, training=None):
        # training 매개변수를 직접 사용
        if training is None:
            training = False
        
        noise = tf.random.uniform(shape=tf.shape(x), minval=-1, maxval=1)
        return tf.cond(tf.cast(training, tf.bool), 
                       lambda: x * (1 + noise), 
                       lambda: x)
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb

class TFPositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(TFPositionalEncoding1D, self).__init__()

        self.channels = int(np.ceil(channels / 2) * 2)
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = tf.expand_dims(get_emb(sin_inp_x), 0)
        emb = emb[0]  # A bit of a hack
        self.cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )

        return self.cached_penc
        
def Transformer_DAE(signal_size = sigLen,head_size=64,num_heads=2,ff_dim=64,num_transformer_blocks=2, dropout=0):   ###paper 1 model

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x0 = Conv1D(filters=16,
                input_shape=(input_shape, 1),
                kernel_size=ks,
                activation='linear',  
                strides=2,
                padding='same')(input)

    x0 = AddGatedNoise()(x0)

    x0 = layers.Activation('sigmoid')(x0)
    # x0 = Dropout(0.3)(x0)
    x0_ = Conv1D(filters=16,
               input_shape=(input_shape, 1),
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(input)
    # x0_ = Dropout(0.3)(x0_)
    xmul0 = Multiply()([x0,x0_])

    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32,
                kernel_size=ks,
                activation='linear',  # 使用线性激活函数
                strides=2,
                padding='same')(xmul0)

    x1 = AddGatedNoise()(x1)

    x1 = layers.Activation('sigmoid')(x1)

    # x1 = Dropout(0.3)(x1)
    x1_ = Conv1D(filters=32,
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(xmul0)
    # x1_ = Dropout(0.3)(x1_)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64,
               kernel_size=ks,
               activation='linear',
               strides=2,
               padding='same')(xmul1)
    x2 = AddGatedNoise()(x2)

    x2 = layers.Activation('sigmoid')(x2)
    # x2 = Dropout(0.3)(x2)
    x2_ = Conv1D(filters=64,
               kernel_size=ks,
               activation='elu',
               strides=2,
               padding='same')(xmul1)
    # x2_ = Dropout(0.3)(x2_)
    xmul2 = Multiply()([x2, x2_])

    xmul2 = BatchNormalization()(xmul2)

    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = xmul2+position_embed(xmul2)
    #
    for _ in range(num_transformer_blocks):
        x3 = transformer_encoder(x3,head_size,num_heads,ff_dim, dropout)
    # x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    # x4 = x4+xmul2
    x4 = x3
    x5 = Conv1DTranspose(input_tensor=x4,
                        filters=64,
                        kernel_size=ks,
                        activation='elu',
                        strides=1,
                        padding='same')
    x5 = x5+xmul2
    x5 = BatchNormalization()(x5)

    x6 = Conv1DTranspose(input_tensor=x5,
                        filters=32,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')
    x6 = x6+xmul1
    x6 = BatchNormalization()(x6)

    x7 = Conv1DTranspose(input_tensor=x6,
                        filters=16,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')

    x7 = x7 + xmul0 #res

    x8 = BatchNormalization()(x7)
    predictions = Conv1DTranspose(
                        input_tensor=x8,
                        filters=1,
                        kernel_size=ks,
                        activation='linear',
                        strides=2,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model

ks = 13   #orig 13
ks1 = 7
# def frequency_branch(input_tensor, filters, kernel_size=13):
#     # 첫 번째 FAN Layer + Conv1D(stride=2) 추가
#     x0 = FANLayer(output_dim=filters, p_ratio=0.25, activation="gelu", gated=True)(input_tensor)
#     x0 = Activation('sigmoid')(x0)
#     x0 = Conv1D(filters=filters, kernel_size=1, strides=2, padding='same')(x0)  # Stride 적용

#     x0_ = FANLayer(output_dim=filters, p_ratio=0.25, activation=None, gated=True)(input_tensor)
#     x0_ = Conv1D(filters=filters, kernel_size=1, strides=2, padding='same')(x0_)  # Stride 적용
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)

#     # 두 번째 FAN Layer + Conv1D(stride=2) 추가
#     x1 = FANLayer(output_dim=filters * 2, p_ratio=0.25, activation="gelu", gated=True)(xmul0)
#     x1 = Activation('sigmoid')(x1)
#     x1 = Conv1D(filters=filters * 2, kernel_size=1, strides=2, padding='same')(x1)  # Stride 적용

#     x1_ = FANLayer(output_dim=filters * 2, p_ratio=0.25, activation=None, gated=True)(xmul0)
#     x1_ = Conv1D(filters=filters * 2, kernel_size=1, strides=2, padding='same')(x1_)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)

#     # 세 번째 FAN Layer + Conv1D(stride=2) 추가
#     x2 = FANLayer(output_dim=filters * 4, p_ratio=0.25, activation="gelu", gated=True)(xmul1)
#     x2 = Activation('sigmoid')(x2)
#     x2 = Conv1D(filters=filters * 4, kernel_size=1, strides=2, padding='same')(x2)  # Stride 적용

#     x2_ = FANLayer(output_dim=filters * 4, p_ratio=0.25, activation='elu', gated=True)(xmul1)
#     x2_ = Conv1D(filters=filters * 4, kernel_size=1, strides=2, padding='same')(x2_)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)

#     return xmul2


# def Dual_FreqDAE(signal_size = sigLen, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=8, dropout=0):
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)

#     # 시간 도메인 인코더
#     x0 = Conv1D(16, ks, activation='sigmoid', strides=2, padding='same')(time_input)
#     x0_ = Conv1D(16, ks, activation=None, strides=2, padding='same')(time_input)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)

#     x1 = Conv1D(32, ks, activation='sigmoid', strides=2, padding='same')(xmul0)
#     x1_ = Conv1D(32, ks, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)

#     x2 = Conv1D(64, ks, activation='sigmoid', strides=2, padding='same')(xmul1)
#     x2_ = Conv1D(64, ks, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)

#     # 주파수 도메인 인코더 (FAN 기반으로 수정)
#     f2 = frequency_branch(freq_input, 16, 13)

#     # Cross Attention: 시간→주파수
#     cross_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(xmul2, f2)
#     combined = layers.Concatenate()([xmul2, cross_out])

#     # Position Encoding + Transformer Encoder
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = combined + position_embed(combined)

#     for _ in range(num_transformer_blocks):
#         x3 = FANformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     # 디코더
#     x4 = x3
#     x5 = Conv1DTranspose(x4, filters=64, kernel_size=ks, activation='elu', strides=1, padding='same')
#     x5 = Add()([x5, xmul2])
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(x5, filters=32, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x6 = Add()([x6, xmul1])
#     x6 = BatchNormalization()(x6)

#     x7 = Conv1DTranspose(x6, filters=16, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x7 = Add()([x7, xmul0])
#     x8 = BatchNormalization()(x7)

#     predictions = Conv1DTranspose(x8, filters=1, kernel_size=ks, activation='linear', strides=2, padding='same')

#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model

# this is real one
def frequency_branch(input_tensor, filters, kernel_size=13):
    # 첫 번째 Conv1D 및 Gated Noise 추가
    x0 = Conv1D(filters=filters, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(input_tensor)
    # x0 = AddGatedNoise()(x0)
    x0 = Activation('sigmoid')(x0)

    x0_ = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, strides=2, padding='same')(input_tensor)
    xmul0 = Multiply()([x0, x0_])
    xmul0 = BatchNormalization()(xmul0)

    # 두 번째 Conv1D 및 Gated Noise 추가
    x1 = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul0)
    # x1 = AddGatedNoise()(x1)
    x1 = Activation('sigmoid')(x1)

    x1_ = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation=None, strides=2, padding='same')(xmul0)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    # 세 번째 Conv1D 및 Gated Noise 추가
    x2 = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul1)
    # x2 = AddGatedNoise()(x2)
    x2 = Activation('sigmoid')(x2)

    x2_ = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='elu', strides=2, padding='same')(xmul1)
    xmul2 = Multiply()([x2, x2_])
    # xmul2 = BatchNormalization()(xmul2)

    return xmul2

def Dual_FreqDAE(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=8, dropout=0):   ###paper 1 model
    input_shape = (signal_size, 1)
    time_input = Input(shape=input_shape)
    
    # 주파수 도메인 입력
    freq_input = Input(shape=input_shape)

    x0 = Conv1D(filters=16,
                input_shape=(input_shape, 1),
                kernel_size=ks,
                activation='linear', 
                strides=2,
                padding='same')(time_input)

    x0 = layers.Activation('sigmoid')(x0)
    # x0 = Dropout(0.3)(x0)
    x0_ = Conv1D(filters=16,
               input_shape=(input_shape, 1),
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(time_input)
    # x0_ = Dropout(0.3)(x0_)
    xmul0 = Multiply()([x0,x0_])

    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32,
                kernel_size=ks,
                activation='linear',
                strides=2,
                padding='same')(xmul0)

    x1 = layers.Activation('sigmoid')(x1)

    # x1 = Dropout(0.3)(x1)
    x1_ = Conv1D(filters=32,
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(xmul0)
    # x1_ = Dropout(0.3)(x1_)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64,
               kernel_size=ks,
               activation='linear',
               strides=2,
               padding='same')(xmul1)

    x2 = layers.Activation('sigmoid')(x2)
    # x2 = Dropout(0.3)(x2)
    x2_ = Conv1D(filters=64,
               kernel_size=ks,
               activation='elu',
               strides=2,
               padding='same')(xmul1)
    # x2_ = Dropout(0.3)(x2_)
    xmul2 = Multiply()([x2, x2_])

    xmul2 = BatchNormalization()(xmul2)
    
    f2 = frequency_branch(freq_input, 16, 13)

    # 시간 및 주파수 도메인 특성 결합
    combined = layers.Concatenate()([xmul2, f2])    
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = combined+position_embed(combined)
    #
    for _ in range(num_transformer_blocks):
        x3 = FANformer_encoder(x3,head_size,num_heads,ff_dim, dropout)
    # for _ in range(num_transformer_blocks):
    #     xmul2 = FANformer_encoder(xmul2, head_size, num_heads, ff_dim, dropout)
    #     f2 = FANformer_encoder(f2, head_size, num_heads, ff_dim, dropout)

    # # 결합
    # x3 = layers.Concatenate()([xmul2, f2])

    x4 = x3
    x5 = Conv1DTranspose(input_tensor=x4,
                        filters=64,
                        kernel_size=ks,
                        activation='elu',
                        strides=1,
                        padding='same')
    x5 = x5+xmul2
    x5 = BatchNormalization()(x5)

    x6 = Conv1DTranspose(input_tensor=x5,
                        filters=32,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')
    x6 = x6+xmul1
    x6 = BatchNormalization()(x6)

    x7 = Conv1DTranspose(input_tensor=x6,
                        filters=16,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')

    x7 = x7 + xmul0 #res

    x8 = BatchNormalization()(x7)
    predictions = Conv1DTranspose(
                        input_tensor=x8,
                        filters=1,
                        kernel_size=ks,
                        activation='linear',
                        strides=2,
                        padding='same')

    model = Model(inputs=[time_input, freq_input], outputs=predictions)
    return model


# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, Activation, Multiply, BatchNormalization, Concatenate,  Add, Dense, Dropout, LayerNormalization
# from tensorflow.keras.models import Model



# import tensorflow as tf
# from tensorflow.keras.layers import Layer

# class AutoCorrelationAttentionLayer(Layer):
#     def call(self, x):
#         # x: (batch, seq_len, d_model)
#         # tf.zeros_like(x)와 tf.complex는 call 내부에서 사용되므로 KerasTensor 문제가 발생하지 않습니다.
#         x_complex = tf.complex(x, tf.zeros_like(x))
#         fft_x = tf.signal.fft(x_complex)
#         # auto-correlation = inverse FFT(FFT(x) * conj(FFT(x)))
#         ac = tf.signal.ifft(fft_x * tf.math.conj(fft_x))
#         ac = tf.math.real(ac)
#         return ac


# def autoformer_attention(x):
#     ac = AutoCorrelationAttentionLayer()(x)
#     weights = tf.keras.layers.Lambda(lambda t: tf.nn.softmax(t, axis=1))(ac)
#     out = weights * x
#     return out



# class SeriesDecompLayer(Layer):
#     def __init__(self, kernel_size, **kwargs):
#         super(SeriesDecompLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#         self.padding = (kernel_size - 1) // 2
#         # AveragePooling1D는 커널 크기와 stride=1, valid padding으로 설정
#         self.pool = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')

#     def call(self, x):
#         # tf.pad를 레이어 내부에서 사용
#         x_padded = tf.pad(x, [[0, 0], [self.padding, self.padding], [0, 0]], mode='REFLECT')
#         moving_mean = self.pool(x_padded)
#         # 추세(trend)와 잔차(residual) 계산
#         trend = moving_mean
#         residual = x - moving_mean
#         return residual, trend

#     def compute_output_shape(self, input_shape):
#         # 두 개의 텐서를 반환하므로 튜플로 반환 (각각의 output shape)
#         batch_size, seq_len, channels = input_shape
#         # AveragePooling1D의 결과는 seq_len 그대로 유지 (padding='valid'와 직접 pad했으므로)
#         output_shape = (batch_size, seq_len, channels)
#         return output_shape, output_shape

# # Autoformer 스타일 블록
# def AutoformerBlock(x, kernel_size=25, dropout=0.0):
#     # 기존 series_decomp 대신 사용자 정의 레이어 사용
#     residual, trend = SeriesDecompLayer(kernel_size)(x)
    
#     # Auto-correlation 기반 attention 적용 (이 부분은 기존 코드 유지)
#     attn_out = autoformer_attention(residual)
#     attn_out = tf.keras.layers.Dropout(dropout)(attn_out)
    
#     # Residual 연결 및 Layer Normalization
#     x = tf.keras.layers.Add()([x, attn_out])
#     x = tf.keras.layers.LayerNormalization()(x)
    
#     # Feed-Forward 네트워크
#     ff = tf.keras.layers.Dense(x.shape[-1], activation='relu')(x)
#     ff = tf.keras.layers.Dropout(dropout)(ff)
#     ff = tf.keras.layers.Dense(x.shape[-1])(ff)
#     x = tf.keras.layers.Add()([x, ff])
#     x = tf.keras.layers.LayerNormalization()(x)
    
#     return x


# # 원래의 COMBDAE 구조에서 AutoformerBlock을 적용한 모델 (Encoder/Decoder 구조는 기존과 동일)
# def Transformer_COMBDAE(signal_size=sigLen, 
#                       patch_size=4,         # 패치 길이 (하이퍼파라미터)
#                       embed_dim=128,        # 패치 임베딩 차원
#                       head_size=64, num_heads=16, ff_dim=64, 
#                       num_transformer_blocks=4, dropout=0):
#     # 입력 정의 (시간 도메인, 주파수 도메인)
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)
    
#     # --- 인코더: 시간 도메인 처리 (기존 Conv1D 기반) ---
#     x0 = Conv1D(filters=16, kernel_size=ks, activation='linear', strides=2, padding='same')(time_input)
#     x0 = Activation('sigmoid')(x0)
#     x0_ = Conv1D(filters=16, kernel_size=ks, activation=None, strides=2, padding='same')(time_input)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)
    
#     x1 = Conv1D(filters=32, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul0)
#     x1 = Activation('sigmoid')(x1)
#     x1_ = Conv1D(filters=32, kernel_size=ks, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)
    
#     x2 = Conv1D(filters=64, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul1)
#     x2 = Activation('sigmoid')(x2)
#     x2_ = Conv1D(filters=64, kernel_size=ks, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)
    
#     # --- 주파수 도메인 분기: 기존 frequency_branch 사용 (별도 구현되어 있다고 가정) ---
#     f2 = frequency_branch(freq_input, 16, 13)
    
#     # --- 시간 및 주파수 도메인 특징 결합 + 위치 임베딩 (TFPositionalEncoding1D는 별도 구현) ---
#     combined = Concatenate()([xmul2, f2])
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = combined + position_embed(combined)
    
#     # --- Autoformer Block 여러 개 적용 ---
#     x = x3
#     for _ in range(num_transformer_blocks):
#         x = AutoformerBlock(x, kernel_size=25, dropout=dropout)
    
#     # --- 디코더: Conv1DTranspose 기반 복원 ---
#     x4 = x
#     x5 = Conv1DTranspose(input_tensor=x4, filters=64, kernel_size=ks, activation='elu', strides=1, padding='same')
#     x5 = Add()([x5, xmul2])
#     x5 = BatchNormalization()(x5)
    
#     x6 = Conv1DTranspose(input_tensor=x5, filters=32, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x6 = Add()([x6, xmul1])
#     x6 = BatchNormalization()(x6)
    
#     x7 = Conv1DTranspose(input_tensor=x6, filters=16, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x7 = Add()([x7, xmul0])
#     x8 = BatchNormalization()(x7)
    
#     predictions = Conv1DTranspose(input_tensor=x8, filters=1, kernel_size=ks, activation='linear', strides=2, padding='same')
    
#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, Activation, Multiply, BatchNormalization, Concatenate,  Add, Dense, Dropout, LayerNormalization
# from tensorflow.keras.models import Model

# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Input, Conv1D, Activation, Multiply, BatchNormalization, Concatenate,  Add, Dense, Dropout, LayerNormalization
# from tensorflow.keras.models import Model

# # Series Decomposition Layer (기존 코드와 동일)
# class SeriesDecompLayer(Layer):
#     def __init__(self, kernel_size, **kwargs):
#         super(SeriesDecompLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#         self.padding = (kernel_size - 1) // 2
#         self.pool = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')

#     def call(self, x):
#         x_padded = tf.pad(x, [[0, 0], [self.padding, self.padding], [0, 0]], mode='REFLECT')
#         moving_mean = self.pool(x_padded)
#         trend = moving_mean
#         residual = x - moving_mean
#         return residual, trend

#     def compute_output_shape(self, input_shape):
#         batch_size, seq_len, channels = input_shape
#         output_shape = (batch_size, seq_len, channels)
#         return output_shape, output_shape

# import tensorflow as tf
# from tensorflow.keras.layers import Layer

# class FedformerAttentionLayer(Layer):
#     def call(self, x):
#         # x: (batch, seq_len, d_model)
#         # 복소수 변환은 call 내부에서 실행되므로 KerasTensor 문제가 발생하지 않습니다.
#         x_complex = tf.complex(x, tf.zeros_like(x))
#         fft_x = tf.signal.fft(x_complex)
#         amplitude = tf.abs(fft_x)
#         phase = tf.math.angle(fft_x)

        
#         # amplitude에 softmax 적용 (여기서는 axis=1: 주파수 축)
#         weights = tf.nn.softmax(amplitude, axis=1)
        
#         # 가중치 곱을 통해 FFT 값을 필터링한 후 재구성
#         fft_x_filtered = tf.complex(
#             weights * amplitude * tf.cos(phase),
#             weights * amplitude * tf.sin(phase)
#         )
#         # 역 FFT를 통해 필터링된 시계열 복원
#         x_filtered = tf.signal.ifft(fft_x_filtered)
#         x_filtered = tf.math.real(x_filtered)
#         return x_filtered
# def fedformer_attention(x):
#     return FedformerAttentionLayer()(x)

# # Fedformer 스타일 블록
# def FedformerBlock(x, kernel_size=25, dropout=0.0):
#     # Series Decomposition을 통해 추세와 계절성(잔차)를 분리합니다.
#     residual, trend = SeriesDecompLayer(kernel_size)(x)
    
#     # 잔차 부분에 대해 Fedformer Attention 적용
#     attn_out = fedformer_attention(residual)
#     attn_out = Dropout(dropout)(attn_out)
    
#     # 입력과 attention 결과를 합산하고 normalization 적용
#     x = Add()([x, attn_out])
#     x = LayerNormalization()(x)
    
#     # Feed-Forward 네트워크
#     ff = Dense(x.shape[-1], activation='relu')(x)
#     ff = Dropout(dropout)(ff)
#     ff = Dense(x.shape[-1])(ff)
#     x = Add()([x, ff])
#     x = LayerNormalization()(x)
    
#     return x

# # Transformer_COMBDAE 모델에서 FedformerBlock 적용
# def Transformer_COMBDAE(signal_size=sigLen, 
#                          head_size=64, num_heads=16, ff_dim=64, 
#                          num_transformer_blocks=4, dropout=0):
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)
    
#     # 인코더: 시간 도메인 처리 (기존 Conv1D 기반)
#     x0 = Conv1D(filters=16, kernel_size=ks, activation='linear', strides=2, padding='same')(time_input)
#     x0 = Activation('sigmoid')(x0)
#     x0_ = Conv1D(filters=16, kernel_size=ks, activation=None, strides=2, padding='same')(time_input)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)
    
#     x1 = Conv1D(filters=32, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul0)
#     x1 = Activation('sigmoid')(x1)
#     x1_ = Conv1D(filters=32, kernel_size=ks, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)
    
#     x2 = Conv1D(filters=64, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul1)
#     x2 = Activation('sigmoid')(x2)
#     x2_ = Conv1D(filters=64, kernel_size=ks, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)
    
#     # 주파수 도메인 분기: 기존 frequency_branch 사용 (별도 구현되어 있다고 가정)
#     f2 = frequency_branch(freq_input, 16, 13)
    
#     # 시간 및 주파수 도메인 특징 결합 + 위치 임베딩 (TFPositionalEncoding1D는 별도 구현)
#     combined = Concatenate()([xmul2, f2])
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = combined + position_embed(combined)
    
#     # Fedformer Block 여러 개 적용 (기존 AutoformerBlock 대신 FedformerBlock 사용)
#     x = x3
#     for _ in range(num_transformer_blocks):
#         x = FedformerBlock(x, kernel_size=25, dropout=dropout)
    
#     x4 = x
#     x5 = Conv1DTranspose(input_tensor=x4, filters=64, kernel_size=ks, activation='elu', strides=1, padding='same')
#     x5 = Add()([x5, xmul2])
#     x5 = BatchNormalization()(x5)
    
#     x6 = Conv1DTranspose(input_tensor=x5, filters=32, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x6 = Add()([x6, xmul1])
#     x6 = BatchNormalization()(x6)
    
#     x7 = Conv1DTranspose(input_tensor=x6, filters=16, kernel_size=ks, activation='elu', strides=2, padding='same')
#     x7 = Add()([x7, xmul0])
#     x8 = BatchNormalization()(x7)
    
#     predictions = Conv1DTranspose(input_tensor=x8, filters=1, kernel_size=ks, activation='linear', strides=2, padding='same')
    
#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model
