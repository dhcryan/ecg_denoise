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


def Transformer_COMBDAE(signal_size = sigLen,head_size=64,num_heads=16,ff_dim=64,num_transformer_blocks=4, dropout=0):   ###paper 1 model
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
        x3 = transformer_encoder(x3,head_size,num_heads,ff_dim, dropout)

    
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
