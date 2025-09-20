import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
import tensorflow as tf 
from datetime import datetime
from deepFilter.dl_models import *
import os
import shap
import time
from tabulate import tabulate

current_date = datetime.now().strftime('%m%d')
# Global dictionary to store inference times and FLOPs for comparison
inference_times = {}
# Custom loss SSD
import tensorflow as tf

def ssd_loss(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_pred - y_true), axis=-2)

# Combined loss SSD + MSE
def combined_ssd_mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-2) * 500 + tf.reduce_sum(tf.square(y_true - y_pred), axis=-2)

def combined_ssd_mad_loss(y_true, y_pred):
    return tf.reduce_max(tf.square(y_true - y_pred), axis=-2) * 50 + tf.reduce_sum(tf.square(y_true - y_pred), axis=-2)

# Custom loss SAD
def sad_loss(y_true, y_pred):
    return tf.reduce_sum(tf.sqrt(tf.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return tf.reduce_max(tf.square(y_pred - y_true), axis=-2)


def train_dl(Dataset, experiment):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))
    
    train_start_time = time.time()  # Training start time
    
    if experiment in ['Dual_FreqDAE']:
        [X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
        F_train_x, F_val_x, F_train_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)
    else:
        [X_train, y_train, X_test, y_test] = Dataset
        # 일반 모델들을 위한 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
            
    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'CNN_DAE':
        # FCN_DAE
        model = CNN_DAE()
        model_label = 'CNN_DAE'
        
    if experiment == 'FCN_DAE':
        # FCN_DAE
        model = FCN_DAE()
        model_label = 'FCN_DAE'

    if experiment == 'DRNN':
        # DRNN
        model = DRRN_denoising()
        model_label = 'DRNN'

    if experiment == 'DeepFilter':
        # Inception-like linear and non linear dilated
        model = deep_filter_model_I_LANL_dilated()
        model_label = 'DeepFilter'
        
    if experiment == 'AttentionSkipDAE':
        # Inception-like linear and non linear dilated
        model = AttentionSkipDAE()
        model_label = 'AttentionSkipDAE'
    
    if experiment == 'Transformer_DAE':
        # Transformer_FDAE
        model = Transformer_DAE()
        model_label = 'Transformer_DAE'
        
    if experiment == 'Dual_FreqDAE':
        model = Dual_FreqDAE()
        model_label = 'Dual_FreqDAE'
        
    print('\n ' + model_label + '\n ')

    model.summary()
    epochs = int(1e5)  # 100000
    batch_size = 128  #128

    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-10

    # Loss function selection according to method implementation
    if experiment == 'DRNN' or experiment == 'AttentionSkipDAE':
        criterion = keras.losses.mean_squared_error

    elif experiment == 'FCN-DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss

    model.compile(loss=criterion,
                  optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    # Keras Callbacks

    # 체크포인트 파일 경로 설정
    model_dir = current_date
    model_filepath = os.path.join(model_dir, f"{model_label}_weights.best.weights.h5")

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  # on acc has to go max
                                 save_weights_only=True)

    # reduce_lr = ReduceLROnPlateau(monitor="val_loss",
    #                             factor=0.5,           # 학습률 감소 비율은 그대로 유지
    #                             min_delta=0.005,      # min_delta를 0.05에서 0.001로 줄여 작은 개선도 감지
    #                             mode='min',           # val_loss 최소화를 목표로 함
    #                             patience=10,          # patience를 2에서 10으로 늘려 학습률 감소 시점을 늦춤
    #                             verbose=1)

    # early_stop = EarlyStopping(monitor="val_loss",  
    #                         min_delta=0.001,       # 개선 판단을 위한 최소 변화량
    #                         mode='min',             # val_loss 최소화를 목표로 함
    #                         patience=40,            # patience를 50에서 20으로 줄여 더 빠른 조기 종료
    #                         verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',  # on acc has to go max
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",  # "val_loss"
                               min_delta=0.05,
                               mode='min',  # on acc has to go max
                               patience=10,
                               verbose=1)
    tb_log_dir = './runs_' + current_date +'/' + model_label
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, 
                         write_images=False, embeddings_freq=0,
                        #  embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs_new

    if experiment in ['Dual_FreqDAE']:
        history = model.fit(x=[X_train, F_train_x], y=y_train,
                validation_data=([X_val, F_val_x], y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop,
                            reduce_lr,
                            checkpoint,
                            tboard])
    else:
        history = model.fit(x=X_train, y=y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop,
                            reduce_lr,
                            checkpoint,
                            tboard])
    # model.save(model_filepath)
    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    actual_epochs = len(history.epoch)
    avg_epoch_time = training_time / actual_epochs if actual_epochs > 0 else 0
    inference_times[experiment] = inference_times.get(experiment, {})
    inference_times[experiment]['training_time'] = avg_epoch_time
    print(f"Average epoch training time for {experiment}: {avg_epoch_time:.4f} seconds (total: {training_time:.4f}s over {actual_epochs} epochs)")
    
    K.clear_session()



def test_dl(Dataset, experiment):

    print('Deep Learning pipeline: Testing the model')
# 여기선 x_test, y_test만 사용됨
    if experiment in ['Dual_FreqDAE']:
        [X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset

    else:
        [X_train, y_train, X_test, y_test] = Dataset
        # 일반 모델들을 위한 데이터 분할
    batch_size = 128

    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'CNN_DAE':
        # FCN_DAE
        model = CNN_DAE()
        model_label = 'CNN_DAE'
        
    if experiment == 'FCN_DAE':
        # FCN_DAE
        model = FCN_DAE()
        model_label = 'FCN_DAE'

    if experiment == 'DRNN':
        # DRNN
        model = DRRN_denoising()
        model_label = 'DRNN'

    if experiment == 'DeepFilter':
        # Inception-like linear and non linear dilated
        model = deep_filter_model_I_LANL_dilated()
        model_label = 'DeepFilter'

    if experiment == 'AttentionSkipDAE':
        # Inception-like linear and non linear dilated
        model = AttentionSkipDAE()
        model_label = 'AttentionSkipDAE'
        
    if experiment == 'Transformer_DAE':
        model = Transformer_DAE()
        model_label = 'Transformer_DAE'
        
    if experiment == 'Dual_FreqDAE':
        model = Dual_FreqDAE()
        model_label = 'Dual_FreqDAE'
            
    print('\n ' + model_label + '\n ')

    model.summary()

    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = 'mse'

    elif experiment == 'FCN_DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss
        # criterion = combined_huber_freq_loss

    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])
    # # for 125
    #model_dir = '0221_FIXED'
    model_dir = current_date
    model_filepath = os.path.join(model_dir, model_label + '_weights.best.weights.h5')

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.load_weights(model_filepath)
    
    # Measure inference time and parameters
    start_time = time.time()
    
    if experiment in ['Dual_FreqDAE']:
        # Test score
        y_pred = model.predict([X_test, F_test_x], batch_size=batch_size, verbose=1)
    else:
        # Test score
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Get number of parameters
    params = model.count_params()
    
    # FLOPs calculation not available for TensorFlow models in this setup
    flops = 'N/A'
    
    inference_times[experiment] = {'time': inference_time, 'params': params, 'flops': flops}
    print(f"Inference time for {experiment}: {inference_time:.4f} seconds")
    print(f"Parameters for {experiment}: {params}")
    print(f"FLOPs for {experiment}: {flops}")
    
    K.clear_session()

    return [X_test, y_test, y_pred]


def print_inference_time_table():
    """
    Print a table comparing average epoch training time, inference times, parameters, and FLOPs of different models.
    """
    if not inference_times:
        print("No data recorded.")
        return
    
    table_data = []
    for model, data in inference_times.items():
        training_time = data.get('training_time', 'N/A')
        time_taken = data.get('time', 'N/A')
        params = data.get('params', 'N/A')
        flops = data.get('flops', 'N/A')
        table_data.append([
            model,
            f"{training_time:.4f} seconds" if isinstance(training_time, float) else training_time,
            f"{time_taken:.4f} seconds" if isinstance(time_taken, float) else time_taken,
            params,
            flops
        ])
    
    print("\nComputational Analysis Table:")
    print(tabulate(table_data, headers=["Model", "Average Epoch Training Time (s)", "Inference Time", "Parameters", "FLOPs"], tablefmt="grid"))
    
    # Suggestions for model compression
    print("\nSuggestions for Model Compression:")
    print("- Pruning: Remove unnecessary weights to reduce model size and computation.")
    print("- Quantization: Use lower precision (e.g., int8) for weights and activations.")
    print("- Knowledge Distillation: Train a smaller model to mimic the larger one.")
    print("- Lightweight Architectures: Consider MobileNet or EfficientNet variants for real-time deployment.")