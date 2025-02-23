import numpy as np
from scipy.fft import fft
import glob
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle
from scipy.fft import fft, ifft
import numpy as np

def make_fourier_ifft(inputs, n, fs):
    """
    FFT 변환 후 IFFT를 수행하여 주파수 도메인 정보를 Time-domain과 동일한 형태로 변환.

    Parameters:
    inputs: 원본 신호 (2D 배열 - (배치 크기, 샘플 수))
    n: FFT 샘플 수
    fs: 샘플링 주파수 (예: 360 Hz)

    Returns:
    ifft_transformed: IFFT 결과 (원본 신호와 동일한 shape)
    """
    T = n / fs
    k = np.arange(n)
    freq = k / T
    freq = freq[range(int(n / 2))]

    signal_list = []
    for i in range(inputs.shape[0]):
        y = inputs[i, :]
        
        # 1️⃣ FFT 변환 (복소수 형태 유지)
        Y = fft(y) / n  
        
        # 2️⃣ IFFT 변환 (Time-domain으로 복원)
        Y_ifft = ifft(Y)  
        
        # 3️⃣ 실수 부분만 사용하여 원본 형태로 복원
        ifft_transformed = np.real(Y_ifft)

        signal_list.append(ifft_transformed)

    return np.asarray(signal_list)


# from scipy.fft import fft, ifft

# def make_fourier(inputs, n, fs):
#     """
#     특정 주파수 대역만 유지하는 FFT 변환.
    
#     Parameters:
#     inputs: 원본 신호 (2D 배열 - (배치 크기, 샘플 수))
#     n: FFT 샘플 수
#     fs: 샘플링 주파수 (예: 360 Hz)
#     low_cut: 유지할 주파수 대역의 하한 (Hz)
#     high_cut: 유지할 주파수 대역의 상한 (Hz)
    
#     Returns:
#     특정 주파수 대역만 유지한 신호 (IFFT 후 time-domain으로 변환)
#     """
#     print(f"Input shape: {inputs.shape}")
    
#     T = n / fs
#     k = np.arange(n)
#     freq = k / T
#     freq = freq[range(int(n / 2))]  # Nyquist 주파수까지 선택
#     print(f"Freq shape: {freq.shape}")
    
#     signal_list = []
#     for i in range(inputs.shape[0]):
#         y = inputs[i, :]
#         # print(f"Signal {i} shape: {y.shape}")
#         # Signal 0 shape: (512,)
#         Y = fft(y) / n  # FFT 수행 후 정규화
#         # print(f"FFT output shape: {Y.shape}")
#         # FFT output shape: (512,)
#         # 특정 주파수 대역 필터링
#         Y_filtered = np.zeros_like(Y)
#         mask = (freq >= 0.5) & (freq <= 50)
#         # print(f"Mask shape: {mask.shape}")
#         # Mask shape: (256,)
#         Y_filtered[:len(mask)][mask] = Y[:len(mask)][mask]  # 특정 대역만 유지
# #         print(f"Filtered FFT shape: {Y_filtered.shape}")
# # Filtered FFT shape: (512,)
#         # IFFT로 복원
#         # Y_ifft = ifft(Y_filtered)  
#         # ifft_transformed = np.real(Y_ifft)
#         # print(f"IFFT output shape: {ifft_transformed.shape}")
#         # IFFT output shape: (512,)
#         # IFFT output shape: (512,)
#         signal_list.append(Y_filtered)

#     output = np.asarray(signal_list)
#     # print(f"Final output shape: {output.shape}")
#     # Final output shape: (1, 512)

#     return output

# import numpy as np
# from scipy.fft import fft

# def make_fourier(inputs, n, fs):
#     """
#     주파수 대역별 가중치를 적용한 Fourier 변환.
#     inputs: 입력 신호 (2D 배열 - (배치 크기, 샘플 수))
#     n: FFT 샘플 수
#     fs: 샘플링 주파수 (예: 360 Hz)
#     """
#     T = n / fs
#     k = np.arange(n)
#     freq = k / T  # 주파수 배열 (0 ~ fs/2)
#     freq = freq[range(int(n / 2))]  # Nyquist 주파수까지 선택 (0 ~ 180 Hz)

#     # 주파수 대역별 가중치 정의 (이미지 및 코드 참고)
#     weights = np.ones_like(freq)  # 기본 가중치 1
#     # 신호가 강한 저주파수 대역(0~20 Hz) 강조 (가중치 1.5)
#     weights[(freq >= 0) & (freq <= 20)] = 1.5
#     # 전력선 노이즈(50~60 Hz) 억제 (가중치 0.1)
#     weights[(freq >= 50) & (freq <= 60)] = 0.1
#     # 불필요한 고주파수(80~180 Hz) 억제 (가중치 0.05)
#     weights[(freq >= 80) & (freq <= 180)] = 0.05

#     signal_list = []
#     for i in range(inputs.shape[0]):
#         y = inputs[i, :]
#         Y = fft(y) / n  # FFT 수행 후 정규화
#         Y_mag = np.abs(Y[range(int(n / 2))]) * weights  # 주파수 대역별 가중치 적용
#         Y_full = np.hstack([Y_mag, Y_mag])  # 시간 도메인 크기(512)로 확장
#         signal_list.append(Y_full)

#     return np.asarray(signal_list)

# 사용 예시
# X_test = np.random.randn(13316, 512, 1)  # 예시 데이터
# F_test_x_weighted = make_fourier_weighted(X_test.squeeze(-1), 512, 360)
# print("Weighted Fourier shape:", F_test_x_weighted.shape)  # (13316, 512)
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

def Data_Preparation_with_Fourier(samples, fs=360):
    print('Getting the Data ready ...')

    # Set random seed for reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        qtdb = pickle.load(input)

    print(f"[INFO] Loaded QTDatabase with {len(qtdb.keys())} signals")
    # # Load combined noise
    # 650000 samples
    with open('data/CombinedNoise_Train.pkl', 'rb') as input:
        combined_noise = pickle.load(input)
    with open('data/CombinedNoise_Test.pkl', 'rb') as input:
        static_noise = pickle.load(input)
    # with open('data/Mixed_Noise_SNR_3.pkl', 'rb') as input:
    #     static_noise = pickle.load(input)
    print(f"[INFO] Loaded CombinedNoise with {len(combined_noise)} channels")
    total_length = combined_noise.shape[0]  # 650000 samples
    half_length = total_length // 2
    train_noise_1 = combined_noise
    # Test Noise:
    test_noise_1 = static_noise
    # test_noise_1 = np.squeeze(static_noise)[0]

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
            # # ablation study를 위해 IFFT 적용 (model D)
            # fourier_transformed_y = make_fourier_ifft(b_np.reshape(1, -1), samples, fs)
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
    # To ensure equal selection of channels
    # Adding noise to train
    for beat_idx, beat in enumerate(beats_train):
        noise_source = train_noise_1  # Upper half of channel 1
        noise_segment = noise_source[noise_index:noise_index + samples]
        signal_noise = beat + noise_segment
        sn_train.append(signal_noise)
        # print(signal_noise.shape) 512
        fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
        fourier_train_x.append(fourier_transformed_x[0])  # Append the single batch
        # print(f'fourier_transformed_x shape: {fourier_transformed_x.shape}') (1, 512)
        noise_index += samples
        # 노이즈 크기 650000 넘어가면 초기화
        if noise_index > (len(noise_source) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    np.save('rnd_test.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
        
    for beat_idx, beat in enumerate(beats_test):
        # if np.random.rand() < channel_ratio:
        noise_source = test_noise_1  # Lower half of channel 1
        noise_segment = noise_source[noise_index:noise_index + samples]
        signal_noise = beat + noise_segment
        sn_test.append(signal_noise)
        fourier_transformed_x = make_fourier(signal_noise.reshape(1, -1), samples, fs)  # X에 대한 Fourier 변환
        fourier_test_x.append(fourier_transformed_x[0])  # Append the single batch
        # noise_indices_test.append(noise_combination_idx)  # 노이즈 인덱스 저장
        noise_index += samples
        # if noise_index > (len(noise) - samples):
        #     noise_index = 0
        if noise_index > (len(noise_source) - samples):
            noise_index = 0

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

    return Dataset, valid_train_indices, valid_test_indices
