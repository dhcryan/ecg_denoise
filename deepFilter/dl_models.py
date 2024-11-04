import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization,\
                         concatenate, Activation, Input, Conv2DTranspose, Lambda, LSTM, GRU,Reshape, Embedding, GlobalAveragePooling1D,\
                         Multiply,Bidirectional
import keras.backend as K
from keras import layers
import tensorflow as tf
import numpy as np
from scipy import signal

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
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
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x

##########################################################################

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
        noise = tf.random.uniform(shape=tf.shape(x), minval=-1, maxval=1)
        return tf.keras.backend.in_train_phase(x * (1 + noise), x, training=training)
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

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Multiply, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# Sub-Pixel Convolution layer for upsampling
class SubPixelConv1D(tf.keras.layers.Layer):
    def __init__(self, scale):
        super(SubPixelConv1D, self).__init__()
        self.scale = scale

    def call(self, inputs):
        batch_size, length, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        reshaped = tf.reshape(inputs, (batch_size, length, channels // self.scale, self.scale))
        return tf.reshape(tf.transpose(reshaped, perm=[0, 1, 3, 2]), (batch_size, length * self.scale, channels // self.scale))

# Fusion Network for combining time and frequency domain features
class FusionNetwork(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(FusionNetwork, self).__init__()
        self.conv1 = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')
        self.conv2 = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')

    def call(self, time_features, freq_features):
        combined = Concatenate()([time_features, freq_features])
        x = self.conv1(combined)
        x = self.conv2(x)
        return x

def frequency_branch_updated(input_tensor, filters, kernel_size=13):
    x0 = Conv1D(filters=filters, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(input_tensor)
    x0 = Activation('sigmoid')(x0)

    x0_ = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, strides=2, padding='same')(input_tensor)
    xmul0 = Multiply()([x0, x0_])
    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul0)
    x1 = Activation('sigmoid')(x1)

    x1_ = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation=None, strides=2, padding='same')(xmul0)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='linear', strides=2, padding='same')(xmul1)
    x2 = Activation('sigmoid')(x2)

    x2_ = Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='elu', strides=2, padding='same')(xmul1)
    xmul2 = Multiply()([x2, x2_])
    xmul2 = BatchNormalization()(xmul2)

    return xmul2

def Transformer_COMBDAE_updated(signal_size=sigLen, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0.1):
    input_shape = (signal_size, 1)
    time_input = Input(shape=input_shape)
    freq_input = Input(shape=input_shape)

    # Conv1D layers for time domain
    x0 = Conv1D(filters=16, kernel_size=ks, activation='linear', strides=2, padding='same')(time_input)
    x0 = Activation('sigmoid')(x0)
    x0_ = Conv1D(filters=16, kernel_size=ks, activation=None, strides=2, padding='same')(time_input)
    xmul0 = Multiply()([x0, x0_])
    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul0)
    x1 = Activation('sigmoid')(x1)
    x1_ = Conv1D(filters=32, kernel_size=ks, activation=None, strides=2, padding='same')(xmul0)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64, kernel_size=ks, activation='linear', strides=2, padding='same')(xmul1)
    x2 = Activation('sigmoid')(x2)
    x2_ = Conv1D(filters=64, kernel_size=ks, activation='elu', strides=2, padding='same')(xmul1)
    xmul2 = Multiply()([x2, x2_])
    xmul2 = BatchNormalization()(xmul2)

    # Frequency branch
    f2 = frequency_branch_updated(freq_input, filters=16, kernel_size=ks)

    # Fusion Network
    fusion_output = FusionNetwork(128)(xmul2, f2)

    # Transformer Encoder Blocks
    # Assuming TFPositionalEncoding1D and transformer_encoder are properly defined elsewhere
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = fusion_output + position_embed(fusion_output)

    for _ in range(num_transformer_blocks):
        x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout)

    # Upsampling with Sub-Pixel Convolution
    x4 = SubPixelConv1D(scale=2)(x3)
    x4 = Conv1D(filters=64, kernel_size=ks, activation='elu', padding='same')(x4)
    x4 = BatchNormalization()(x4)

    x5 = SubPixelConv1D(scale=2)(x4)
    x5 = Conv1D(filters=32, kernel_size=ks, activation='elu', padding='same')(x5)
    x5 = BatchNormalization()(x5)

    x6 = SubPixelConv1D(scale=2)(x5)
    x6 = Conv1D(filters=16, kernel_size=ks, activation='elu', padding='same')(x6)
    x6 = BatchNormalization()(x6)

    predictions = Conv1D(filters=1, kernel_size=ks, activation='linear', padding='same')(x6)

    model = Model(inputs=[time_input, freq_input], outputs=predictions)
    return model


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



# def Transformer_COMBDAE_updated(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_blocks=6, dropout=0.1, ks=3):
#     input_shape = (signal_size, 1)
#     time_input = Input(shape=input_shape)
#     freq_input = Input(shape=input_shape)

#     # Initial Convolutional Layer
#     x0 = Conv1D(filters=16, kernel_size=ks, activation='linear', strides=2, padding='same')(time_input)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('relu')(x0)

#     # Depthwise Separable Convolution Layer
#     x0 = Conv1D(filters=16, kernel_size=ks, strides=2, padding='same')(x0)
#     x0 = BatchNormalization()(x0)
#     x0 = Activation('relu')(x0)

#     # Squeeze-and-Excitation Block
#     x0 = squeeze_excite_block(x0)

#     # Residual Block 1 with Dilated Convolution
#     x1 = Conv1D(filters=32, kernel_size=ks, strides=1, dilation_rate=2, padding='same')(x0)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('relu')(x1)
#     x1 = squeeze_excite_block(x1)
    
#     # Match the dimensions using 1x1 Conv1D
#     x0_resized = Conv1D(filters=32, kernel_size=1, padding='same')(x0)
#     x1 = Add()([x0_resized, x1])  # Residual connection

#     # Residual Block 2 with Depthwise Separable Convolution
#     x2 = Conv1D(filters=64, kernel_size=ks, strides=2, padding='same')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('relu')(x2)
#     x2 = squeeze_excite_block(x2)

#     # Match the sequence length using tf.image.resize
#     x1_resized = Conv1D(filters=64, kernel_size=1, padding='same')(x1)
#     x1_resized = Lambda(lambda inputs: resize_sequence(inputs[0], inputs[1]))([x1_resized, x2])

#     # Match the batch size explicitly
#     x2 = Lambda(lambda inputs: match_batch_size(inputs[0], inputs[1]))([x2, x1_resized])
    
#     # Perform the addition
#     x2 = Add()([x1_resized, x2])

#     # Adding Position Encoding
#     position_embed = TFPositionalEncoding1D(signal_size)
#     x2 = x2 + position_embed(x2)

#     f2 = frequency_branch(freq_input, 16, 13)

#     # Combining Time and Frequency Domains
#     combined = Concatenate()([x2, f2])
#     combined = position_embed(combined)

#     # Transformer Blocks
#     for _ in range(num_transformer_blocks):
#         combined = transformer_encoder(combined, head_size, num_heads, ff_dim, dropout)

#     x4 = combined

#     # Upsampling with Transposed Convolutions
#     x5 = Conv1DTranspose(filters=64, kernel_size=ks, activation='elu', strides=1, padding='same')(x4)
#     x5 = BatchNormalization()(x5)
#     x2_resized = Conv1D(filters=64, kernel_size=1, padding='same')(x2)
#     x5_resized = Lambda(lambda inputs: resize_sequence(inputs[0], inputs[1]))([x5, x2_resized])
#     x5 = Add()([x2_resized, x5_resized])

#     x6 = Conv1DTranspose(filters=32, kernel_size=ks, activation='elu', strides=2, padding='same')(x5)
#     x6 = BatchNormalization()(x6)
#     x1_resized = Conv1D(filters=32, kernel_size=1, padding='same')(resize_sequence(x1, x6))
#     x1_resized = Lambda(lambda inputs: match_batch_size(inputs[0], inputs[1]))([x1_resized, x6])
#     x6_resized = Lambda(lambda inputs: resize_sequence(inputs[0], inputs[1]))([x1_resized, x6])
#     x6 = Add()([x6_resized, x6])

#     x7 = Conv1DTranspose(filters=16, kernel_size=ks, activation='elu', strides=2, padding='same')(x6)
#     x7 = BatchNormalization()(x7)
#     x0_resized = Conv1D(filters=16, kernel_size=1, padding='same')(x0)
#     x7_resized = Lambda(lambda inputs: resize_sequence(inputs[0], inputs[1]))([x7, x0_resized])
#     x7_resized = Lambda(lambda inputs: match_batch_size(inputs[0], inputs[1]))([x7_resized, x0_resized])
#     x7 = Add()([x0_resized, x7_resized])

#     x8 = BatchNormalization()(x7)
#     predictions = Conv1DTranspose(filters=1, kernel_size=ks, activation='linear', strides=2, padding='same')(x8)

#     model = Model(inputs=[time_input, freq_input], outputs=predictions)
#     return model
