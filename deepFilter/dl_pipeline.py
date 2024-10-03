import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from datetime import datetime
from deepFilter.dl_models import *
import os

current_date = datetime.now().strftime('%m%d')
# Custom loss SSD
def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)

# Combined loss SSD + MSE
def combined_ssd_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-2) * 500 + K.sum(K.square(y_true - y_pred), axis=-2)

def combined_ssd_mad_loss(y_true, y_pred):
    return K.max(K.square(y_true - y_pred), axis=-2) * 50 + K.sum(K.square(y_true - y_pred), axis=-2)

# Custom loss SAD
def sad_loss(y_true, y_pred):
    return K.sum(K.sqrt(K.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-2)

def train_dl(Dataset, experiment):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))
    
    if experiment == 'Transformer_COMBDAE_with_band_encoding':
        [X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset

        # F_train_x, F_val_x, F_train_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)
            # 시간 도메인과 주파수 도메인 데이터를 함께 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
        F_train_x, F_val_x, F_train_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)
    else:
        [X_train, y_train, X_test, y_test] = Dataset
        # 일반 모델들을 위한 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
            
    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'Transformer_DAE':
        # Transformer_FDAE
        model = Transformer_DAE()
        model_label = 'Transformer_DAE'
    elif experiment == 'Transformer_COMBDAE':
        model = Transformer_COMBDAE()
        model_label = 'Transformer_COMBDAE'
    elif experiment == 'Transformer_COMBDAE_with_band_encoding':
        model = Transformer_COMBDAE_with_band_encoding()
        model_label = 'Transformer_COMBDAE_with_band_encoding'
    print('\n ' + model_label + '\n ')

    model.summary()

    epochs = int(1e2)  # 100000
    # epochs = 100
    batch_size = 128
    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-7


    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = keras.losses.mean_squared_error

    elif experiment == 'FCN-DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss


    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    # Keras Callbacks

    # 체크포인트 파일 경로 설정
    model_dir = current_date
    model_filepath = os.path.join(model_dir, model_label + '_weights.best.hdf5')

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  # on acc has to go max
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                factor=0.5,           # 학습률 감소 비율은 그대로 유지
                                min_delta=0.001,      # min_delta를 0.05에서 0.001로 줄여 작은 개선도 감지
                                mode='min',           # val_loss 최소화를 목표로 함
                                patience=10,          # patience를 2에서 10으로 늘려 학습률 감소 시점을 늦춤
                                verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",  
                            min_delta=0.0001,       # 개선 판단을 위한 최소 변화량
                            mode='min',             # val_loss 최소화를 목표로 함
                            patience=20,            # patience를 50에서 20으로 줄여 더 빠른 조기 종료
                            verbose=1)
    tb_log_dir = './runs_' + current_date +'/' + model_label
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, write_grads=False,
                         write_images=False, embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs_new

    # GPU
    if experiment == 'Transformer_COMBDAE_with_band_encoding':
        model.fit(x=[X_train, F_train_x], y=y_train,
                validation_data=([X_val, F_val_x], y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop,
                            reduce_lr,
                            checkpoint,
                            tboard])
    else:
        model.fit(x=X_train, y=y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop,
                            reduce_lr,
                            checkpoint,
                            tboard])
    K.clear_session()



def test_dl(Dataset, experiment):

    print('Deep Learning pipeline: Testing the model')

    if experiment == 'Transformer_COMBDAE_with_band_encoding':
        [X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset

        # F_train_x, F_val_x, F_train_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)
            # 시간 도메인과 주파수 도메인 데이터를 함께 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
        F_train_x, F_val_x, F_train_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)
    else:
        [X_train, y_train, X_test, y_test] = Dataset
        # 일반 모델들을 위한 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
      
    batch_size = 32

    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'Transformer_DAE':
        # Transformer_FDAE
        model = Transformer_DAE()
        model_label = 'Transformer_DAE'
    elif experiment == 'Transformer_COMBDAE':
        model = Transformer_COMBDAE()
        model_label = 'Transformer_COMBDAE'
    elif experiment == 'Transformer_COMBDAE_with_band_encoding':
        model = Transformer_COMBDAE_with_band_encoding()
        model_label = 'Transformer_COMBDAE_with_band_encoding'
    print('\n ' + model_label + '\n ')

    model.summary()

    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = 'mse'

    elif experiment == 'FCN-DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss

    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])
    # 체크포인트 파일 경로 설정
    model_dir = current_date
    model_filepath = os.path.join(model_dir, model_label + '_weights.best.hdf5')

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # load weights
    model.load_weights(model_filepath)
    
    if experiment == 'Transformer_COMBDAE_with_band_encoding':
        # Test score
        y_pred = model.predict([X_test, F_test_x], batch_size=batch_size, verbose=1)

    else:
        # Test score
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
        
    K.clear_session()

    return [X_test, y_test, y_pred]