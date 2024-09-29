import numpy as np
from scipy.fft import fft
import glob
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle


def make_fourier(inputs, n, fs):
    """
    주파수 도메인 정보 추출 및 time-domain과 같은 shape으로 만듦.
    
    Parameters:
    inputs: 입력 신호 (원본 신호, 2D 배열 - (배치 크기, 샘플 수))
    n: FFT 샘플 수
    fs: 샘플링 주파수 (예: 360 Hz)
    
    Returns:
    주파수 도메인에서 얻은 신호 (FFT), time-domain과 동일한 크기
    """
    T = n / fs
    k = np.arange(n)
    freq = k / T
    freq = freq[range(int(n / 2))]

    signal_list = []
    for i in range(inputs.shape[0]):
        y = inputs[i, :]
        Y = fft(y) / n  # FFT 수행 후 정규화
        Y = np.abs(Y[range(int(n / 2))])
        # Magnitude 값을 두 배로 늘려 time-domain과 동일한 shape으로 맞춤 (512)
        Y_full = np.hstack([Y, Y])  # Duplicate to match time-domain size
        signal_list.append(Y_full)

    return np.asarray(signal_list)

def Data_Preparation_with_Fourier(samples, channel_ratio, fs=360):
    print('Getting the Data ready ...')

    # Set random seed for reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        qtdb = pickle.load(input)

    print(f"[INFO] Loaded QTDatabase with {len(qtdb.keys())} signals")

    # Load combined noise
    with open('data/CombinedNoise.pkl', 'rb') as input:
        combined_noise = pickle.load(input)
    print(f"[INFO] Loaded CombinedNoise with {len(combined_noise)} channels")

    #####################################
    # Data split
    #####################################
    test_set = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 
                'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32', 'sel49', 
                'sel14046', 'sel15814']

    beats_train = []
    beats_test = []
    fourier_train_x = []
    fourier_test_x = []
    fourier_train_y = []
    fourier_test_y = []
    valid_train_indices = []  # To keep track of valid indices in training data
    valid_test_indices = []   # To keep track of valid indices in test data
    # 노이즈 인덱스 저장 리스트
    noise_indices_train = []
    noise_indices_test = []
    sn_train = []
    sn_test = []
    
    skip_beats = 0
    qtdb_keys = list(qtdb.keys())

    print(f"[INFO] Processing QTDatabase, {len(qtdb.keys())} signals to process.")

    for signal_name in qtdb_keys:
        for b_idx, b in enumerate(qtdb[signal_name]):
            b_np = np.zeros(samples)
            b_sq = np.array(b)

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            # Fourier 변환 적용 (주파수 도메인 정보, time-domain과 동일한 shape으로)
            fourier_transformed_y = make_fourier(b_np.reshape(1, -1), samples, fs)

            if signal_name in test_set:
                beats_test.append(b_np)
                fourier_test_y.append(fourier_transformed_y[0])  # Append the single batch
                valid_test_indices.append(len(beats_test) - 1)  # Track valid test beat index
            else:
                beats_train.append(b_np)
                fourier_train_y.append(fourier_transformed_y[0])  # Append the single batch
                valid_train_indices.append(len(beats_train) - 1)  # Track valid train beat index

        print(f"[DEBUG] Processed signal {signal_name}, total beats in train: {len(beats_train)}, total beats in test: {len(beats_test)}")

    #####################################
    # Adding noise to train and test sets
    #####################################
    print(f"[INFO] Adding noise to train and test sets")
    # Random scaling factor for train and test
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    noise_index = 0
    # Adding noise to train
    for beat_idx, beat in enumerate(beats_train):
        if np.random.rand() < channel_ratio:
            noise_combination_idx = np.random.randint(1, 8)  # 8 types of noise combinations
            noise = combined_noise[0][:, noise_combination_idx]
            noise_segment = noise[noise_index:noise_index + samples]
            beat_max_value = np.max(beat) - np.min(beat)
            noise_max_value = np.max(noise_segment) - np.min(noise_segment)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_train[beat_idx] / Ase
            signal_noise = beat + alpha * noise_segment
            sn_train.append(signal_noise)
            fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
            fourier_train_x.append(fourier_transformed_x[0])  # Append the single batch
            noise_indices_train.append(noise_combination_idx)  # 노이즈 인덱스 저장
            noise_index += samples
            if noise_index > (len(noise) - samples):
                noise_index = 0
            valid_train_indices.append(beat_idx)  # Only track valid beats
        else:
            noise_combination_idx = np.random.randint(1, 8)
            noise = combined_noise[1][:, noise_combination_idx]
            noise_segment = noise[noise_index:noise_index + samples]
            beat_max_value = np.max(beat) - np.min(beat)
            noise_max_value = np.max(noise_segment) - np.min(noise_segment)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_train[beat_idx] / Ase
            signal_noise = beat + alpha * noise_segment
            sn_train.append(signal_noise)
            fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
            fourier_train_x.append(fourier_transformed_x[0])  # Append the single batch
            noise_indices_train.append(noise_combination_idx)  # 노이즈 인덱스 저장            
            noise_index += samples
            if noise_index > (len(noise) - samples):
                noise_index = 0
            valid_train_indices.append(beat_idx)

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    np.save('rnd_test.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
        
    for beat_idx, beat in enumerate(beats_test):
        if np.random.rand() < channel_ratio:
            noise_combination_idx = np.random.randint(1, 8)
            noise = combined_noise[0][:, noise_combination_idx]
            noise_segment = noise[noise_index:noise_index + samples]
            beat_max_value = np.max(beat) - np.min(beat)
            noise_max_value = np.max(noise_segment) - np.min(noise_segment)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_test[beat_idx] / Ase
            signal_noise = beat + alpha * noise_segment
            sn_test.append(signal_noise)
            fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
            fourier_test_x.append(fourier_transformed_x[0])  # Append the single batch
            noise_indices_test.append(noise_combination_idx)  # 노이즈 인덱스 저장
            noise_index += samples
            if noise_index > (len(noise) - samples):
                noise_index = 0
            valid_test_indices.append(beat_idx)
        else:
            noise_combination_idx = np.random.randint(1, 8)
            noise = combined_noise[1][:, noise_combination_idx]
            noise_segment = noise[noise_index:noise_index + samples]
            beat_max_value = np.max(beat) - np.min(beat)
            noise_max_value = np.max(noise_segment) - np.min(noise_segment)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_test[beat_idx] / Ase
            signal_noise = beat + alpha * noise_segment
            sn_test.append(signal_noise)
            fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
            fourier_test_x.append(fourier_transformed_x[0])  # Append the single batch
            noise_indices_test.append(noise_combination_idx)  # 노이즈 인덱스 저장            
            noise_index += samples
            if noise_index > (len(noise) - samples):
                noise_index = 0
            valid_test_indices.append(beat_idx)

    X_train = np.array(sn_train)[valid_train_indices]  # Match noisy and original beats
    X_test = np.array(sn_test)[valid_test_indices]

    y_train = np.array(beats_train)[valid_train_indices]  # Match noisy and original beats
    y_test = np.array(beats_test)[valid_test_indices]

    # Fourier 정보도 포함된 주파수 도메인 데이터셋 생성
    F_train_x = np.array(fourier_train_x)[valid_train_indices]
    F_test_x = np.array(fourier_test_x)[valid_test_indices]
    F_train_y = np.array(fourier_train_y)[valid_train_indices]
    F_test_y = np.array(fourier_test_y)[valid_test_indices]

    # Shape을 time-domain과 동일하게 확장
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    F_train_x = np.expand_dims(F_train_x, axis=2)
    F_train_y = np.expand_dims(F_train_y, axis=2)
    
    F_test_x = np.expand_dims(F_test_x, axis=2)
    F_test_y = np.expand_dims(F_test_y, axis=2)

    Dataset = [X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y]
    
    print(f"[INFO] Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"[INFO] Fourier shapes -> F_train_x: {F_train_x.shape}, F_train_y: {F_train_y.shape}, F_test_x: {F_test_x.shape}, F_test_y: {F_test_y.shape}")
    print('Dataset ready to use.')

    return Dataset, valid_train_indices, valid_test_indices, noise_indices_train, noise_indices_test
