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

def LFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 4),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)


    x = concatenate([LB0, LB1, LB2, LB3])

    return x


def NLFilter_module(x, layers):

    NLB0 = Conv1D(filters=int(layers / 4),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='relu',
                strides=1,
                padding='same')(x)


    x = concatenate([NLB0, NLB1, NLB2, NLB3])

    return x


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

def deep_filter_vanilla_linear():

    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     input_shape=(512, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_vanilla_Nlinear():
    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     input_shape=(512, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_I_linear():
    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = LFilter_module(input, 64)
    tensor = LFilter_module(tensor, 64)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 16)
    tensor = LFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_I_Nlinear():
    input_shape = (512, 1)
    input = Input(shape=input_shape)

    tensor = NLFilter_module(input, 64)
    tensor = NLFilter_module(tensor, 64)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 16)
    tensor = NLFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


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

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x) 
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

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
        
def Transformer_DAE(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   ###paper 1 model

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
    xmul2 = BatchNormalization()(xmul2)

    return xmul2


def Transformer_COMBDAE(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   ###paper 1 model
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

    # x0 = AddGatedNoise()(x0)

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

    # x1 = AddGatedNoise()(x1)
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
    # x2 = AddGatedNoise()(x2)
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
    # f1 = frequency_branch(f0, 32)
    # f2 = frequency_branch(f1, 64)    
    # 시간 및 주파수 도메인 특성 결합
    combined = layers.Concatenate()([xmul2, f2])    
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = combined+position_embed(combined)
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

    model = Model(inputs=[time_input, freq_input], outputs=predictions)
    return model


# from tensorflow.keras import layers, Model, Input
# from tensorflow.keras.layers import Conv1D, BatchNormalization, Multiply, Add, Activation

# def frequency_branch_processing(freq_input, filters):
#     """
#     이미 Fourier 처리된 주파수 입력을 더 깊게 처리하는 Dynamic Convolution 방식
#     """
#     x = Conv1D(filters, kernel_size=1, activation='linear', padding='same')(freq_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     return x

# def token_mixer_local_global(time_input, freq_features):
#     """
#     시간 도메인과 주파수 도메인의 특징을 통합
#     """
#     # 시간 도메인에서의 로컬 처리
#     local_features = Conv1D(filters=16, kernel_size=3, padding="same", activation="relu")(time_input)
    
#     # 주파수 도메인에서 글로벌 특징 처리
#     global_features = Conv1D(filters=16, kernel_size=1, padding="same", activation="relu")(freq_features)
    
#     # 로컬과 글로벌 특징 통합
#     combined = layers.Concatenate()([local_features, global_features])
#     combined = BatchNormalization()(combined)
#     return combined

# def transformer_encoder_with_freq(inputs, head_size, num_heads, ff_dim, dropout=0):
#     """
#     Transformer 블록에 주파수 도메인 처리를 추가한 버전
#     """
#     # Normalization and Multi-Head Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs

#     # Frequency Enhanced FFN
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=3, activation="relu", padding="same")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, padding="same")(x)
#     return x + res

# # def Transformer_COMBDAE_FreqEnhanced(signal_size=512, head_size=64, num_heads=8, ff_dim=64,
# #                                       num_transformer_blocks=6, dropout=0):
# #     """
# #     시간과 주파수 도메인을 모두 사용하는 SFHformer 기반 COMBDAE
# #     """
# def Transformer_FreqDAE(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   ###paper 1 model
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)

#     # 주파수 도메인 입력
#     freq_input = Input(shape=input_shape)
#     freq_features = frequency_branch_processing(freq_input, filters=16)

#     # 시간 및 주파수 특징 결합
#     combined_features = token_mixer_local_global(time_input, freq_features)

#     # Transformer Blocks
#     x = combined_features
#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder_with_freq(x, head_size, num_heads, ff_dim, dropout)

#     # Decoder Path
#     x = Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
#     x = BatchNormalization()(x)

#     # 최종 예측
#     predictions = Conv1D(filters=1, kernel_size=3, activation="linear", padding="same")(x)

#     # 모델 정의
#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Multiply, Add, Activation, Concatenate, Dropout
# from tensorflow.keras.models import Model
# import numpy as np

# from tensorflow.keras.layers import Layer
# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, Activation
# from tensorflow.keras.activations import gelu

# class FourierUnit1D(tf.keras.layers.Layer):
#     def __init__(self, filters, groups=4):
#         super(FourierUnit1D, self).__init__()
#         self.filters = filters
#         self.groups = groups

#         # Convolution layers for processing
#         self.conv1 = tf.keras.layers.Conv1D(filters=self.filters * 2, kernel_size=1, strides=1, padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.conv2 = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, strides=1, padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization()

#         # Dynamic weighting layer
#         self.dynamic_weight = tf.keras.Sequential([
#             tf.keras.layers.Conv1D(filters=self.groups, kernel_size=1, strides=1, padding="same"),
#             tf.keras.layers.Softmax(axis=-1)
#         ])

#     def call(self, inputs):
#         # FFT Transformation
#         fft_features = tf.signal.rfft(inputs)  # Perform FFT (outputs complex numbers)
#         real_part = tf.math.real(fft_features)
#         imag_part = tf.math.imag(fft_features)

#         # Combine real and imaginary parts
#         combined_fft = tf.concat([real_part, imag_part], axis=-1)

#         # Process through convolutional layers
#         processed_fft = self.conv1(combined_fft)
#         processed_fft = self.bn1(processed_fft)
#         processed_fft = tf.keras.activations.gelu(processed_fft)

#         # Dynamic weighting
#         dynamic_weights = self.dynamic_weight(combined_fft)

#         # Apply dynamic weighting
#         weighted_fft = processed_fft * dynamic_weights

#         # Process back to real domain
#         processed_real = tf.signal.irfft(tf.complex(weighted_fft[..., :self.filters], weighted_fft[..., self.filters:]),
#                                          fft_length=[inputs.shape[1]])
#         return processed_real


# def FrequencyBranch(input_tensor, filters, kernel_size=13):
#     # 첫 번째 Conv1D 및 Fourier Unit 추가
#     x0 = Conv1D(filters=filters, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(input_tensor)
#     x0 = Activation('sigmoid')(x0)

#     x0_ = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, strides=2, padding='same')(input_tensor)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)

#     # Fourier Unit 추가 (첫 번째 단계)
#     fourier1 = FourierUnit1D(filters)
#     xmul0 = fourier1(xmul0)

#     # 두 번째 Conv1D 및 Fourier Unit 추가
#     x1 = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul0)
#     x1 = Activation('sigmoid')(x1)

#     x1_ = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)

#     # Fourier Unit 추가 (두 번째 단계)
#     fourier2 = FourierUnit1D(filters * 2)
#     xmul1 = fourier2(xmul1)

#     # 세 번째 Conv1D 및 Fourier Unit 추가
#     x2 = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul1)
#     x2 = Activation('sigmoid')(x2)

#     x2_ = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)

#     # Fourier Unit 추가 (세 번째 단계)
#     fourier3 = FourierUnit1D(filters * 4)
#     xmul2 = fourier3(xmul2)

#     return xmul2

# def Transformer_FreqDAE(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#     # Time-domain Input
#     input_time = Input(shape=(signal_size, 1))

#     # Time-domain Branch
#     x0 = Conv1D(filters=16, kernel_size=13, strides=2, padding='same', activation='linear')(input_time)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('sigmoid')(x0)

#     x1 = Conv1D(filters=32, kernel_size=13, strides=2, padding='same', activation='linear')(x0)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('sigmoid')(x1)

#     x2 = Conv1D(filters=64, kernel_size=13, strides=2, padding='same', activation='linear')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('sigmoid')(x2)

#     # Frequency-domain Branch
#     freq_branch = FrequencyBranch(input_tensor=input_time, filters=16, kernel_size=13)

#     # Combine Time and Frequency Features
#     combined_features = tf.keras.layers.Concatenate(axis=-1)([x2, freq_branch])

#     # Positional Encoding
#     position_embed = TFPositionalEncoding1D(combined_features.shape[1])
#     x3 = combined_features + position_embed(combined_features)

#     # Transformer Encoder Blocks
#     for _ in range(num_transformer_blocks):
#         x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     # Decoder Part (Upsampling)
#     x4 = Conv1DTranspose(filters=64, kernel_size=13, strides=1, padding='same', activation='elu')(x3)
#     x4 = BatchNormalization()(x4)

#     x5 = Conv1DTranspose(filters=32, kernel_size=13, strides=2, padding='same', activation='elu')(x4)
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(filters=16, kernel_size=13, strides=2, padding='same', activation='elu')(x5)
#     x6 = BatchNormalization()(x6)

#     predictions = Conv1DTranspose(filters=1, kernel_size=13, strides=2, padding='same', activation='linear')(x6)

#     # Model
#     model = Model(inputs=[input_time], outputs=predictions)
#     return model

# class FrequencyBranch(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, strides):
#         super(FrequencyBranch, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides

#         # Fourier Unit for Frequency Features
#         self.fourier_unit = FourierUnit1D(filters=filters)

#         # Convolutional layers for additional feature extraction
#         self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', activation='relu')
#         self.bn1 = BatchNormalization()

#         self.conv2 = Conv1D(filters * 2, kernel_size, strides=strides, padding='same', activation='relu')
#         self.bn2 = BatchNormalization()

#     def call(self, inputs):
#         # Process through FourierUnit
#         fft_features = self.fourier_unit(inputs)

#         # Additional convolutional feature extraction
#         x = self.conv1(fft_features)
#         x = self.bn1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)

#         return x

# def Transformer_FreqDAE(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#     input_shape = (signal_size, 1)
#     input_time = Input(shape=input_shape)

#     # Time-domain Branch
#     x0 = Conv1D(filters=16, kernel_size=13, strides=2, padding='same', activation='linear')(input_time)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('sigmoid')(x0)

#     x1 = Conv1D(filters=32, kernel_size=13, strides=2, padding='same', activation='linear')(x0)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('sigmoid')(x1)

#     x2 = Conv1D(filters=64, kernel_size=13, strides=2, padding='same', activation='linear')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('sigmoid')(x2)  # Shape: (None, 64, 64)

#     # Frequency-domain Branch
#     freq_branch = FrequencyBranch(filters=64, kernel_size=13, strides=2)  # Define with required arguments
#     freq_features = freq_branch(input_time)  # Output shape: (None, 128, 128)

#     # Align shapes of time-domain and frequency-domain features
#     freq_features_resized = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='same')(freq_features)
#     freq_features_resized = tf.keras.layers.Reshape((64, 64))(freq_features_resized)  # Reshape to match x2

#     # Combine Time and Frequency features
#     combined_features = tf.keras.layers.Concatenate(axis=-1)([x2, freq_features_resized])  # Shape: (None, 64, 128)

#     # Positional Encoding
#     position_embed = TFPositionalEncoding1D(128)  # Ensure embedding matches concatenated dimension
#     position_encoding = position_embed(combined_features)
#     x3 = combined_features + position_encoding  # Both shapes: (None, 64, 128)

#     # Transformer Encoder Blocks
#     for _ in range(num_transformer_blocks):
#         x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     # Upsampling and Reconstruction
#     x4 = Conv1DTranspose(filters=64, kernel_size=13, strides=1, padding='same', activation='elu')(x3)
#     x4 = BatchNormalization()(x4)

#     x5 = Conv1DTranspose(filters=32, kernel_size=13, strides=2, padding='same', activation='elu')(x4)
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(filters=16, kernel_size=13, strides=2, padding='same', activation='elu')(x5)
#     x6 = BatchNormalization()(x6)

#     predictions = Conv1DTranspose(filters=1, kernel_size=13, strides=2, padding='same', activation='linear')(x6)

#     model = Model(inputs=[input_time], outputs=predictions)
#     return model




# def Transformer_FreqDAE(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#     input_shape = (signal_size, 1)
#     time_input = tf.keras.Input(shape=input_shape)

#     # Time-domain processing
#     x0 = Conv1D(16, kernel_size=13, strides=2, padding='same', activation='linear')(time_input)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('sigmoid')(x0)

#     x1 = Conv1D(32, kernel_size=13, strides=2, padding='same', activation='linear')(x0)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('sigmoid')(x1)

#     x2 = Conv1D(64, kernel_size=13, strides=2, padding='same', activation='linear')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('sigmoid')(x2)

#     # Frequency-domain processing
#     # Frequency-domain Branch
#     # Frequency-domain Branch
#     freq_branch_layer = FrequencyBranch(filters=16, kernel_size=13, strides=2, target_length=64)
#     freq_features = freq_branch_layer(time_input)

#     # Combine Time and Frequency features
#     combined_features = tf.keras.layers.Concatenate(axis=-1)([x2, freq_features])  # Combine features
#     # Positional Encoding
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = combined_features + position_embed(combined_features)

#     # Transformer blocks
#     for _ in range(num_transformer_blocks):
#         x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     x4 = x3
#     x5 = Conv1DTranspose(input_tensor=x4,
#                         filters=64,
#                         kernel_size=ks,
#                         activation='elu',
#                         strides=1,
#                         padding='same')
#     x5 = x5+x2
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(input_tensor=x5,
#                         filters=32,
#                         kernel_size=ks,
#                         activation='elu',
#                         strides=2,
#                         padding='same')
#     x6 = x6+x1
#     x6 = BatchNormalization()(x6)

#     x7 = Conv1DTranspose(input_tensor=x6,
#                         filters=16,
#                         kernel_size=ks,
#                         activation='elu',
#                         strides=2,
#                         padding='same')

#     x7 = x7 + x0 #res

#     x8 = BatchNormalization()(x7)
#     predictions = Conv1DTranspose(
#                         input_tensor=x8,
#                         filters=1,
#                         kernel_size=ks,
#                         activation='linear',
#                         strides=2,
#                         padding='same')
#     model = Model(inputs=[time_input], outputs=predictions)
#     return model

# class FrequencyBranch(Layer):
#     def __init__(self, filters, kernel_size, strides):
#         super(FrequencyBranch, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides

#         # Initialize Conv1D and BatchNorm layers
#         self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', activation='relu')
#         self.bn1 = BatchNormalization()
#         self.conv2 = Conv1D(filters * 2, kernel_size, strides=strides, padding='same', activation='relu')
#         self.bn2 = BatchNormalization()
#         self.conv3 = Conv1D(filters * 4, kernel_size, strides=strides, padding='same', activation='relu')
#         self.bn3 = BatchNormalization()

#     def call(self, inputs):
#         # FFT to extract frequency domain features
#         fft_output = tf.signal.rfft(inputs)
#         real_part = tf.math.real(fft_output)
#         imag_part = tf.math.imag(fft_output)
#         fft_features = tf.concat([real_part, imag_part], axis=-1)  # Combine real and imaginary parts

#         # Apply Conv1D and BatchNorm layers
#         x = self.conv1(fft_features)
#         x = self.bn1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)

#         # Reshape to match Time-domain shape
#         x = tf.reshape(x, (-1, 64, 64))  # Adjust shape as needed
#         return x

# def Transformer_FreqDAE(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#     input_shape = (signal_size, 1)
#     input = Input(shape=input_shape)

#     # Time-domain branch
#     x0 = Conv1D(filters=16, kernel_size=13, strides=2, padding='same', activation='linear')(input)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('sigmoid')(x0)

#     x1 = Conv1D(filters=32, kernel_size=13, strides=2, padding='same', activation='linear')(x0)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('sigmoid')(x1)

#     x2 = Conv1D(filters=64, kernel_size=13, strides=2, padding='same', activation='linear')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('sigmoid')(x2)

#     # Frequency-domain branch
#     freq_branch_layer = FrequencyBranch(filters=16, kernel_size=13, strides=2)
#     freq_features = freq_branch_layer(input)  # Frequency-domain features

#     # Combine Time-domain and Frequency-domain features
#     combined_features = Concatenate(axis=-1)([x2, freq_features])  # Combined output shape: (batch, 64, 128)

#     # Positional Encoding
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = combined_features + position_embed(combined_features)

#     # Transformer Encoder Blocks
#     for _ in range(num_transformer_blocks):
#         x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     # Decoder
#     x4 = x3
#     x5 = Conv1DTranspose(input_tensor=x4, filters=64, kernel_size=13, activation='elu', strides=1, padding='same')
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(input_tensor=x5, filters=32, kernel_size=13, activation='elu', strides=2, padding='same')
#     x6 = BatchNormalization()(x6)

#     x7 = Conv1DTranspose(input_tensor=x6, filters=16, kernel_size=13, activation='elu', strides=2, padding='same')
#     x7 = BatchNormalization()(x7)

#     predictions = Conv1DTranspose(input_tensor=x7, filters=1, kernel_size=13, activation='linear', strides=2, padding='same')

#     model = Model(inputs=[input], outputs=predictions)
#     return model


# import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Multiply, Input, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras import layers

# # Sub-Pixel Convolution layer for upsampling
# class SubPixelConv1D(tf.keras.layers.Layer):
#     def __init__(self, scale):
#         super(SubPixelConv1D, self).__init__()
#         self.scale = scale

#     def call(self, inputs):
#         batch_size, length, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
#         reshaped = tf.reshape(inputs, (batch_size, length, channels // self.scale, self.scale))
#         return tf.reshape(tf.transpose(reshaped, perm=[0, 1, 3, 2]), (batch_size, length * self.scale, channels // self.scale))

# # Fusion Network for combining time and frequency domain features
# class FusionNetwork(tf.keras.layers.Layer):
#     def __init__(self, filters):
#         super(FusionNetwork, self).__init__()
#         self.conv1 = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')
#         self.conv2 = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')

#     def call(self, time_features, freq_features):
#         combined = Concatenate()([time_features, freq_features])
#         x = self.conv1(combined)
#         x = self.conv2(x)
#         return x

# def frequency_branch_updated(input_tensor, filters, kernel_size=13):
#     x0 = Conv1D(filters=filters, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(input_tensor)
#     x0 = Activation('sigmoid')(x0)

#     x0_ = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, strides=2, padding='same')(input_tensor)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)

#     x1 = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul0)
#     x1 = Activation('sigmoid')(x1)

#     x1_ = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)

#     x2 = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul1)
#     x2 = Activation('sigmoid')(x2)

#     x2_ = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)

#     return xmul2

# def Transformer_COMBDAE_updated(signal_size=sigLen, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0.1):
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)

#     # Conv1D layers for time domain
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

#     # Frequency branch
#     f2 = frequency_branch_updated(freq_input, filters=16, kernel_size=ks)

#     # Fusion Network
#     fusion_output = FusionNetwork(128)(xmul2, f2)

#     # Transformer Encoder Blocks
#     # Assuming TFPositionalEncoding1D and transformer_encoder are properly defined elsewhere
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x3 = fusion_output + position_embed(fusion_output)

#     for _ in range(num_transformer_blocks):
#         x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

#     # Upsampling with Sub-Pixel Convolution
#     x4 = SubPixelConv1D(scale=2)(x3)
#     x4 = Conv1D(filters=64, kernel_size=ks, activation='elu', padding='same')(x4)
#     x4 = BatchNormalization()(x4)

#     x5 = SubPixelConv1D(scale=2)(x4)
#     x5 = Conv1D(filters=32, kernel_size=ks, activation='elu', padding='same')(x5)
#     x5 = BatchNormalization()(x5)

#     x6 = SubPixelConv1D(scale=2)(x5)
#     x6 = Conv1D(filters=16, kernel_size=ks, activation='elu', padding='same')(x6)
#     x6 = BatchNormalization()(x6)

#     predictions = Conv1D(filters=1, kernel_size=ks, activation='linear', padding='same')(x6)

#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model


import tensorflow as tf
from tensorflow.keras import layers


class MLPTemporalFretsLayer(layers.Layer):
    def __init__(self, fft_length, embed_size, sparsity_threshold, scale=0.02, **kwargs):
        super(MLPTemporalFretsLayer, self).__init__(**kwargs)
        self.fft_length = fft_length  # IFFT를 위한 길이
        self.embed_size = embed_size  # 임베딩 크기
        self.scale = scale  # 가중치 초기화 스케일
        self.sparsity_threshold = sparsity_threshold
    def build(self, input_shape):
        # Keras의 add_weight 메서드를 사용하여 학습 가능한 변수로 등록
        self.r = self.add_weight(
            shape=(self.embed_size, self.embed_size),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=self.scale),
            trainable=True,
            name="r_weight"
        )
        self.i = self.add_weight(
            shape=(self.embed_size, self.embed_size),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=self.scale),
            trainable=True,
            name="i_weight"
        )
        self.rb = self.add_weight(
            shape=(self.embed_size,),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=self.scale),
            trainable=True,
            name="rb_bias"
        )
        self.ib = self.add_weight(
            shape=(self.embed_size,),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=self.scale),
            trainable=True,
            name="ib_bias"
        )
        self.embeddings = tf.Variable(
            initial_value=tf.random.normal([1, self.embed_size]),
            trainable=True,
            name="embeddings"
        )
        self.fc = tf.keras.Sequential([
            layers.Dense(512 * 1, activation='relu'),  # 입력을 (512 * 1)로 변환
            layers.Reshape((512, 1))  # 차원을 (512, 1)로 재구성
        ])
        super(MLPTemporalFretsLayer, self).build(input_shape)
        
    def tokenEmb(self, inputs):
        # 임베딩 확장: [Batch, Input length, Channel]
        x = tf.transpose(inputs, perm=[0, 2, 1])
        # 입력 텐서 차원: (batch_size, 1,512)
        x = tf.expand_dims(x, axis=-1)  # 차원 확장
        # x after expand_dims: (None, 1, 512, 1)
        #print(f'x after expand_dims: {x.shape}')
        y = self.embeddings
        return x * y
    
    def call(self, inputs):
        # 입력 텐서 차원: (batch_size, 512, 1)
        # FFT 적용 (주파수 도메인으로 변환)
        # x = tf.signal.rfft(inputs)  # (batch_size, 512, 1)
        x = inputs        
        x = self.tokenEmb(x)
        bias = x
        # 입력 텐서 차원: (batch_size, 1,512, emd_size)
        # FreMLP_temporal 적용
        x = self.FreMLP_temporal(x, self.r, self.i, self.rb, self.ib, self.embed_size)
        print(f'x: {x.shape}')
        # x: (None, 1, 512, 32)
        #x = x + bias  # Bias 추가        
        x = x + tf.cast(bias, dtype=tf.complex64)
        # IFFT로 시간 도메인으로 복원
        # x = tf.signal.irfft(x, fft_length=[self.fft_length])
        x = tf.squeeze(x, axis=1)
        print(f'x after squeeze: {x.shape}')
        x = tf.reshape(x, [tf.shape(x)[0], 512, 1])
        print(f'x after reshape: {x.shape}')
        x = self.fc(x)  # fc 레이어 적용하여 (512, 1)로 변환
        print(f'x after fc: {x.shape}')
        return x

    def FreMLP_temporal(self, x, r, i, rb, ib, embed_size):
        # 시계열 길이 추출 (512)
        # 입력 텐서 차원: (batch_size, 1,512, emd_size)
        #print(f'x: {x.shape}')  
        #v x: (None, 1, 512, 32)
        time_steps = tf.shape(x)[2]
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        # 실수 및 허수 성분의 출력을 미리 초기화
        o1_real = tf.zeros([tf.shape(x)[0], 1, time_steps // 2 +1, embed_size], dtype=tf.float32)
        o1_imag = tf.zeros([tf.shape(x)[0], 1, time_steps // 2 +1, embed_size], dtype=tf.float32)
        # 입력 텐서 차원: (batch_size, 1, 257, emd_size)
        # 실수 및 허수 성분에 대한 가중치 연산
        o1_real = tf.nn.relu(
            tf.einsum('bijd,dd->bijd', x_real, r) - 
            tf.einsum('bijd,dd->bijd', x_imag, i) + rb
        )
        # 입력 텐서 차원: (batch_size, 1, 257, emd_size)
        o1_imag = tf.nn.relu(
            tf.einsum('bijd,dd->bijd', x_imag, r) + 
            tf.einsum('bijd,dd->bijd', x_real, i) + ib
        )
        y = tf.stack([o1_real, o1_imag], axis=-1)
                # 입력 텐서 차원: (batch_size, 1, 257, emd_size,2)
        print(f'y: {y.shape}')
        # y: (None, 512, 128, 2)
        # print(f'y: {y}')
        # softshrink 연산을 직접 구현하여 희소성 추가
        y = self.softshrink(y, lambd=self.sparsity_threshold)
        # print(f'y after softshrink: {y.shape}')
        # print(f'y after softshrink: {y}')
#         y: Tensor("model/mlp_temporal_frets_layer/stack:0", shape=(None, 512, 128, 2), dtype=float32)
# y after softshrink: (None, 512, 128, 2)
# y after softshrink: Tensor("model/mlp_temporal_frets_layer/SelectV2:0", shape=(None, 512, 128, 2), dtype=float32)
#         print(f'y[..., 0]: {y[..., 0]}')
#         print(f'y[..., 1]: {y[..., 1]}')
#         y[..., 0]: Tensor("model/mlp_temporal_frets_layer/strided_slice_3:0", shape=(None, 512, 128), dtype=float32)
# y[..., 1]: Tensor("model/mlp_temporal_frets_layer/strided_slice_4:0", shape=(None, 512, 128), dtype=float32)
        # (batch_size, time_steps // 2 + 1, embed_size, 2)에서 복소수 표현으로 변환
        y = tf.complex(y[..., 0], y[..., 1])
        #         y after complex: (None, 1, 512, embed)
        print(f'y after complex: {y.shape}')
#         print(f'y after complex: {y}')
#         y after complex: (None, 512, 128)
# y after complex: Tensor("model/mlp_temporal_frets_layer/Complex:0", shape=(None, 512, 128), dtype=complex64)
        return y
    
    def softshrink(self, inputs, lambd):
        # Softshrink 연산을 직접 구현
        return tf.where(
            tf.abs(inputs) > lambd, 
            inputs - tf.sign(inputs) * lambd, 
            tf.zeros_like(inputs)
        )
# Transformer_COMBDAE 모델에 MLP_temporal 적용
def Transformer_COMBDAE_FreTS(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
    input_shape = (signal_size, 1)
    time_input = Input(shape=input_shape)
    freq_input = Input(shape=input_shape)
    # print(f'time_input: {time_input.shape}')
    # time_input: (None, 512, 1)
    # FreTS MLP_temporal 적용 (주파수 도메인에서 학습)
    freq_output = MLPTemporalFretsLayer(fft_length=1, embed_size=32, sparsity_threshold=0.01, scale=0.02)(freq_input)
    # print(f'time_input after MLPTEMP: {time_output.shape}')
    # time_input after MLPTEMP: (None, 512, 1)
    # # Custom Keras Layer for MLP_temporal_frets
    # time_input = MLPTemporalFretsLayer(r, i, rb, ib, 1, 128)(time_input)
    # Time-domain 처리
    x0 = Conv1D(filters=16, kernel_size=13, activation='linear', strides=2, padding='same')(time_input)
    x0 = AddGatedNoise()(x0)
    x0 = Activation('sigmoid')(x0)
    x0_ = Conv1D(filters=16, kernel_size=13, activation=None, strides=2, padding='same')(time_input)
    xmul0 = Multiply()([x0, x0_])
    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32, kernel_size=13, activation='linear', strides=2, padding='same')(xmul0)
    x1 = AddGatedNoise()(x1)
    x1 = Activation('sigmoid')(x1)
    x1_ = Conv1D(filters=32, kernel_size=13, activation=None, strides=2, padding='same')(xmul0)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64, kernel_size=13, activation='linear', strides=2, padding='same')(xmul1)
    x2 = AddGatedNoise()(x2)
    x2 = Activation('sigmoid')(x2)
    x2_ = Conv1D(filters=64, kernel_size=13, activation='elu', strides=2, padding='same')(xmul1)
    xmul2 = Multiply()([x2, x2_])
    xmul2 = BatchNormalization()(xmul2)
    # print(f'f2 after MLPTEMP: {f2.shape}')
    # f2 after MLPTEMP: (None, 512, 16)
    f2 = frequency_branch(freq_output , 16, 13)
    # print(f'f2 after branch: {f2.shape}')
    # f2 after branch: (None, 64, 64)
    # 시간 및 주파수 도메인 결합
    combined = layers.Concatenate()([xmul2, f2])
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = combined + position_embed(combined)

    for _ in range(num_transformer_blocks):
        x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

    x4 = x3
    x5 = Conv1DTranspose(input_tensor=x4, filters=64, kernel_size=13, activation='elu', strides=1, padding='same')
    x5 = x5 + xmul2
    x5 = BatchNormalization()(x5)

    x6 = Conv1DTranspose(input_tensor=x5, filters=32, kernel_size=13, activation='elu', strides=2, padding='same')
    x6 = x6 + xmul1
    x6 = BatchNormalization()(x6)

    x7 = Conv1DTranspose(input_tensor=x6, filters=16, kernel_size=13, activation='elu', strides=2, padding='same')
    x7 = x7 + xmul0
    x8 = BatchNormalization()(x7)

    predictions = Conv1DTranspose(input_tensor=x8, filters=1, kernel_size=13, activation='linear', strides=2, padding='same')

    model = Model(inputs=[time_input, freq_input], outputs=predictions)
    return model











import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Multiply, BatchNormalization, Activation, Dense, Concatenate, Add
from tensorflow.keras.models import Model

def cross_attention_transformer(query,key,value,head_size,num_heads,ff_dim,dropout=0):
    # 레이어 정규화
    query_norm = layers.LayerNormalization(epsilon=1e-6)(query)
    key_norm = layers.LayerNormalization(epsilon=1e-6)(key)
    value_norm = layers.LayerNormalization(epsilon=1e-6)(value)

    # 멀티헤드 어텐션
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        query=query_norm, key=key_norm, value=value_norm)
    x = layers.Dropout(dropout)(x)
    res = x + query  # 잔차 연결 (Residual connection)

    # 피드포워드 네트워크
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=query.shape[-1], kernel_size=1)(x)
    return x + res

# time domain을 query로, freq domain을 key, value로  # AutoEncoder 구조, 각각의 domain은 conv + self-attention을 거친 후 cross-attention
sigLen=512
def Transformer_COMBDAE_updated(signal_size = sigLen,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   # proposed3 model
    input_shape = (signal_size, 1)
    time_input = Input(shape=input_shape)
    
    # 주파수 도메인 입력
    freq_input = Input(shape=input_shape)
    freq_input_sliced = Lambda(lambda x: x[:, :256, :])(freq_input)

    x0 = Conv1D(filters=16,
                input_shape=(input_shape, 1),
                kernel_size=ks,
                activation='linear', 
                strides=2,
                padding='same')(time_input)
    

    x0 = AddGatedNoise()(x0)

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

    time_domain = xmul2
    
    # time_domain을 Self-Attention
    pos_encoding = TFPositionalEncoding1D(signal_size)
    time_domain = time_domain + pos_encoding(time_domain) 

    for _ in range(num_transformer_blocks): # self-Attention
        time_domain = transformer_encoder(time_domain, head_size, num_heads, ff_dim, dropout)
    
    freq_domain = frequency_branch(freq_input_sliced, 16, 13)
    pos_encoding = TFPositionalEncoding1D(signal_size)
    freq_domain = freq_domain + pos_encoding(freq_domain) 

    for _ in range(num_transformer_blocks): # self-Attention
        freq_domain = transformer_encoder(freq_domain, head_size, num_heads, ff_dim, dropout)
    
    # time을 query로, freq을 key, value로 self-attention
    for _ in range(num_transformer_blocks):
        x = cross_attention_transformer(time_domain, freq_domain, freq_domain, head_size, num_heads, ff_dim, dropout)
    
    # f1 = frequency_branch(f0, 32)
    # f2 = frequency_branch(f1, 64)    
    # 시간 및 주파수 도메인 특성 결합
    # combined = layers.Concatenate()([xmul2, f2])    
    
    # position_embed = TFPositionalEncoding1D(signal_size)
    # x3 = combined+position_embed(combined)
    

    # x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    # x4 = x4+xmul2
    
    x4 = x
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

# def Transformer_Gated_CombDAE_freqencoder(signal_size=sigLen, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0):
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)

#     # 시간 도메인 처리
#     x0 = Conv1D(filters=16, kernel_size=13, activation='linear', strides=2, padding='same')(time_input)
#     x0 = AddGatedNoise()(x0)
#     x0 = Activation('sigmoid')(x0)
#     x0_ = Conv1D(filters=16, kernel_size=13, activation=None, strides=2, padding='same')(time_input)
#     xmul0 = Multiply()([x0, x0_])
#     xmul0 = BatchNormalization()(xmul0)

#     x1 = Conv1D(filters=32, kernel_size=13, activation='linear', strides=2, padding='same')(xmul0)
#     x1 = AddGatedNoise()(x1)
#     x1 = Activation('sigmoid')(x1)
#     x1_ = Conv1D(filters=32, kernel_size=13, activation=None, strides=2, padding='same')(xmul0)
#     xmul1 = Multiply()([x1, x1_])
#     xmul1 = BatchNormalization()(xmul1)

#     x2 = Conv1D(filters=64, kernel_size=13, activation='linear', strides=2, padding='same')(xmul1)
#     x2 = AddGatedNoise()(x2)
#     x2 = Activation('sigmoid')(x2)
#     x2_ = Conv1D(filters=64, kernel_size=13, activation='elu', strides=2, padding='same')(xmul1)
#     xmul2 = Multiply()([x2, x2_])
#     xmul2 = BatchNormalization()(xmul2)

#     # Transformer encoder를 시간 도메인에만 적용
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x2_pos = position_embed(xmul2)  # Positional encoding 추가
#     for _ in range(num_transformer_blocks):
#         x2_pos = transformer_encoder(x2_pos, head_size, num_heads, ff_dim, dropout)

#     # 주파수 도메인 처리 (Conv1D만 적용, attention 없음)
#     f2 = frequency_branch(freq_input, 16, 13)
    
#     for _ in range(num_transformer_blocks):
#         f2 = frequency_encoder(f2, ff_dim, kernel_size=3, dropout=0.1)
    
#     gated_output = layers.Concatenate()([x2_pos, f2])
#     # Deconvolution 과정
#     x4 = Conv1DTranspose(input_tensor=gated_output, filters=64, kernel_size=13, activation='elu', strides=1, padding='same')
#     x4 = Add()([x4, xmul2])
#     x4 = BatchNormalization()(x4)

#     x5 = Conv1DTranspose(input_tensor=x4, filters=32, kernel_size=13, activation='elu', strides=2, padding='same')
#     x5 = Add()([x5, xmul1])
#     x5 = BatchNormalization()(x5)

#     x6 = Conv1DTranspose(input_tensor=x5, filters=16, kernel_size=13, activation='elu', strides=2, padding='same')
#     x6 = Add()([x6, xmul0])
#     x6 = BatchNormalization()(x6)

#     x7 = Conv1DTranspose(input_tensor=x6, filters=1, kernel_size=13, activation='linear', strides=2, padding='same')

#     model = Model(inputs=[time_input, freq_input], outputs=x7)
#     return model
# def apply_frequency_filter(time_features, freq_filters):
#     phase_shifts = np.linspace(0, np.pi, num=freq_filters.shape[-1])  # 주파수 필터를 위한 위상 이동 값 생성
#     cos_filter = np.cos(phase_shifts)  # 코사인 함수를 이용해 필터 생성
#     freq_applied = tf.einsum('btf,f->btf', time_features, cos_filter)  # 시간 특성에 주파수 필터 적용
#     return freq_applied
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Add, Multiply, BatchNormalization, Activation, LayerNormalization, Concatenate, GlobalAveragePooling1D, Reshape, Dense, Lambda
# from tensorflow.keras.models import Model

# def squeeze_excite_block(input_tensor, ratio=16):
#     """Squeeze and Excitation block."""
#     filters = input_tensor.shape[-1]
#     se = GlobalAveragePooling1D()(input_tensor)
#     se = Reshape((1, filters))(se)
#     se = Dense(filters // ratio, activation='relu')(se)
#     se = Dense(filters, activation='sigmoid')(se)
#     return Multiply()([input_tensor, se])

# def resize_sequence(x, target):
#     """Resize sequence length of x to match the target length using linear interpolation."""
#     target_length = tf.shape(target)[1]
#     resized = tf.image.resize(x, [target_length, tf.shape(x)[-1]])
#     return resized

# def match_batch_size(x, target):
#     """Repeat or slice x to match the batch size of the target."""
#     batch_size_target = tf.shape(target)[0]
#     batch_size_x = tf.shape(x)[0]

#     def repeat_fn():
#         repeat_count = tf.math.floordiv(batch_size_target, batch_size_x)
#         remainder = tf.math.mod(batch_size_target, batch_size_x)
#         x_repeated = tf.tile(x, [repeat_count, 1, 1])
#         x_remainder = tf.cond(
#             tf.math.greater(remainder, 0),
#             lambda: x[:remainder, :, :],
#             lambda: tf.zeros_like(x[:0, :, :])
#         )
#         return tf.concat([x_repeated, x_remainder], axis=0)

#     def slice_fn():
#         return x[:batch_size_target, :, :]

#     return tf.cond(tf.math.less(batch_size_x, batch_size_target), repeat_fn, slice_fn)

# def dense_residual_block(input_tensor, filters, kernel_size, strides=1, activation='relu', block_name='block'):
#     """A dense residual block."""
#     x = Conv1D(filters, kernel_size, strides=strides, padding='same', activation=activation, name=f'{block_name}_conv')(input_tensor)
#     x = BatchNormalization(name=f'{block_name}_bn')(x)
    
#     # Resize input_tensor to match x's shape (sequence length and channel dimension)
#     input_resized = Conv1D(filters, kernel_size=1, padding='same', name=f'{block_name}_resize')(input_tensor)
#     input_resized = Lambda(lambda inputs: resize_sequence(inputs[0], inputs[1]))([input_resized, x])
    
#     # Match the batch size explicitly if needed
#     input_resized = Lambda(lambda inputs: match_batch_size(inputs[0], inputs[1]))([input_resized, x])
    
#     return Add()([x, input_resized])  # Residual Connection

