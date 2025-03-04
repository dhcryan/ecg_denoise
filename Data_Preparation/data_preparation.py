import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle
import numpy as np
import pickle
import wfdb

# def Data_Preparation(samples):
#     with open('data/CombinedNoise_Test.pkl', 'rb') as input:
#         static_noise = pickle.load(input)
#     # with open('data/Mixed_Noise_SNR_-3.pkl', 'rb') as input:
#     #     static_noise = pickle.load(input)
#     test_noise_1 = static_noise
#     # test_noise_1 = np.squeeze(static_noise)[0]
def Data_Preparation(samples, noise_index=0):
    print('Getting the Data ready ...')
    # Set random seed for reproducibility
    with open('data/CombinedNoise_Test_125.pkl', 'rb') as input:
        all_test_noises = pickle.load(input)  # (125, 650000)
    test_noise_1 = all_test_noises[noise_index]  # 현재 실험할 test_noise 선택
    seed = 1234
    np.random.seed(seed=seed)
    # Load QT Database
    
    with open('data/QTDatabase.pkl', 'rb') as input:
        qtdb = pickle.load(input)

    # 650000 samples
    with open('data/CombinedNoise_Train.pkl', 'rb') as input:
        combined_noise = pickle.load(input)


    print(f"[INFO] Loaded CombinedNoise with {len(combined_noise)} channels")
    total_length = combined_noise.shape[0]  # 650000 samples
    half_length = total_length // 2
    train_noise_1 = combined_noise
    # Test Noise:

    #####################################
    # Data split
    #####################################
    test_set = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 
                'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32', 'sel49', 
                'sel14046', 'sel15814']

    beats_train = []
    beats_test = []
    valid_train_indices = []  # To keep track of valid indices in training data
    valid_test_indices = []   # To keep track of valid indices in test data
    sn_train = []
    sn_test = []
    
    skip_beats = 0
    # samples = 512
    qtdb_keys = list(qtdb.keys())

    print(f"[INFO] Processing QTDatabase, {len(qtdb_keys)} signals to process.")
# b_np.shape는 (512,)로, 패딩을 포함한 전체 샘플 크기가 512임을 알 수 있습니다.
    for signal_name in qtdb_keys:
        for b_idx, b in enumerate(qtdb[signal_name]):
            b_np = np.zeros(samples)
            b_sq = np.array(b)

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue
# 이 평균값을 b_sq의 각 값에서 빼는 과정은 신호의 중앙화 작업입니다. 즉, 신호의 값들이 배열의 양 끝 값의 평균을 기준으로 대칭적으로 배치되도록 변환됩니다.
# 이 계산을 통해 신호의 첫 값과 마지막 값에 대한 편향을 제거하고, 신호를 중앙으로 이동시키는 효과가 있습니다.
            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
                valid_test_indices.append(len(beats_test) - 1)  # Track valid test beat index
            else:
                beats_train.append(b_np)
                valid_train_indices.append(len(beats_train) - 1)  # Track valid train beat index

        print(f"[DEBUG] Processed signal {signal_name}, total beats in train: {len(beats_train)}, total beats in test: {len(beats_test)}")
    # Compute ECG Power for Train & Test
    #####################################
    # Adding noise to train and test sets
    #####################################
    print(f"[INFO] Adding noise to train and test sets")
    # size=len(beats_train): beats_train의 길이만큼 난수를 생성합니다. 즉, beats_train에 있는 심박 데이터의 개수와 동일한 수의 난수를 생성합니다.
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    noise_index = 0
    for beat_idx, beat in enumerate(beats_train):
        noise_source = train_noise_1  # Upper half of channel 1
        noise_segment = noise_source[noise_index:noise_index + samples]
        signal_noise = beat + noise_segment
        sn_train.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_source) - samples):
            noise_index = 0
    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save('rnd_test.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    
    for beat_idx, beat in enumerate(beats_test):
        noise_source = test_noise_1  # Lower half of channel 1
        noise_segment = noise_source[noise_index:noise_index + samples]
        signal_noise = beat + noise_segment
        sn_test.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_source) - samples):
            noise_index = 0
    X_train = np.array(sn_train)[valid_train_indices]  # Match noisy and original beats
    X_test = np.array(sn_test)[valid_test_indices]

    y_train = np.array(beats_train)[valid_train_indices]  # Match noisy and original beats
    y_test = np.array(beats_test)[valid_test_indices]

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    Dataset = [X_train, y_train, X_test, y_test]
    print(f"[INFO] Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    print('Dataset ready to use.')
# [INFO] Final shapes -> X_train: (72002, 512, 1), y_train: (72002, 512, 1), X_test: (13316, 512, 1), y_test: (13316, 512, 1)
    return Dataset, valid_train_indices, valid_test_indices


# import glob
# import numpy as np
# from scipy.signal import resample_poly
# import wfdb
# import math
# import _pickle as pickle


# def Data_Preparation(samples):
#     print('Getting the Data ready ...')

#     # Set random seed for reproducibility
#     seed = 1234
#     np.random.seed(seed=seed)

#     # Load QT Database
#     with open('data/QTDatabase.pkl', 'rb') as input:
#         qtdb = pickle.load(input)

#     print(f"[INFO] Loaded QTDatabase with {len(qtdb.keys())} signals")

#     # # Load combined noise
#     with open('data/CombinedNoise_Train.pkl', 'rb') as input:
#         combined_noise = pickle.load(input)
#     with open('data/CombinedNoise_Test.pkl', 'rb') as input:
#         static_noise = pickle.load(input)
#     print(f"[INFO] Loaded CombinedNoise with {len(combined_noise)} channels")
#     total_length = combined_noise.shape[0]  # 650000 samples
#     # half_length = total_length // 2

#     # Train Noise:
#     train_noise_1 = combined_noise[:total_length, 0]  # Upper half of channel 1
#     # train_noise_2 = combined_noise[half_length:, 0]  # Lower half of channel 2

#     # Test Noise:
#     test_noise_1 = static_noise[:total_length, 0]  # Lower half of channel 1
#     #####################################
#     # Data split
#     #####################################
#     test_set = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 
#                 'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32', 'sel49', 
#                 'sel14046', 'sel15814']

#     beats_train = []
#     beats_test = []
#     valid_train_indices = []  # To keep track of valid indices in training data
#     valid_test_indices = []   # To keep track of valid indices in test data
#     sn_train = []
#     sn_test = []
#     noise_indices_train = []
#     noise_indices_test = []    
    
#     skip_beats = 0
#     # samples = 512
#     qtdb_keys = list(qtdb.keys())

#     print(f"[INFO] Processing QTDatabase, {len(qtdb_keys)} signals to process.")
# # b_np.shape는 (512,)로, 패딩을 포함한 전체 샘플 크기가 512임을 알 수 있습니다.
#     for signal_name in qtdb_keys:
#         for b_idx, b in enumerate(qtdb[signal_name]):
#             b_np = np.zeros(samples)
#             b_sq = np.array(b)

#             init_padding = 16
#             if b_sq.shape[0] > (samples - init_padding):
#                 skip_beats += 1
#                 continue
# # 이 평균값을 b_sq의 각 값에서 빼는 과정은 신호의 중앙화 작업입니다. 즉, 신호의 값들이 배열의 양 끝 값의 평균을 기준으로 대칭적으로 배치되도록 변환됩니다.
# # 이 계산을 통해 신호의 첫 값과 마지막 값에 대한 편향을 제거하고, 신호를 중앙으로 이동시키는 효과가 있습니다.
#             b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

#             if signal_name in test_set:
#                 beats_test.append(b_np)
#                 valid_test_indices.append(len(beats_test) - 1)  # Track valid test beat index
#             else:
#                 beats_train.append(b_np)
#                 valid_train_indices.append(len(beats_train) - 1)  # Track valid train beat index

#         print(f"[DEBUG] Processed signal {signal_name}, total beats in train: {len(beats_train)}, total beats in test: {len(beats_test)}")

#     #####################################
#     # Adding noise to train and test sets
#     #####################################
#     print(f"[INFO] Adding noise to train and test sets")
#     # Random scaling factor for train and test
#     # size=len(beats_train): beats_train의 길이만큼 난수를 생성합니다. 즉, beats_train에 있는 심박 데이터의 개수와 동일한 수의 난수를 생성합니다.
#     rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
#     noise_index = 0
#     # Adding noise to train
#     # https://chatgpt.com/g/g-cKXjWStaE-python/c/66e1471b-57b4-8006-b921-233e7803fcab
#     for beat_idx, beat in enumerate(beats_train):
#         # if beat_idx % 2 == 0:
#         #     noise_source = train_noise_1  # Upper half of channel 1
#         noise_source = train_noise_1  # Upper half of channel 1
#         # else:
#         #     noise_source = train_noise_2  # Lower half of channel 2
#         # 노이즈 조합도 순차적으로 선택, 주기적으로 변화를 줌 (매 8회 주기)
#         # noise_combination_idx = (beat_idx % 7) + 1  # 1부터 7까지 순차적으로 선택
#         # noise_combination_idx = 0         
#         # noise = combined_noise[selected_channel][:, noise_combination_idx]
#         noise_segment = noise_source[noise_index:noise_index + samples]
#         # beat_max_value = np.max(beat) - np.min(beat)
#         # noise_max_value = np.max(noise_segment) - np.min(noise_segment)
#         # if noise_max_value == 0:
#         #     Ase = 1  # 기본값 설정
#         # else:
#         #     Ase = noise_max_value / beat_max_value
#         # alpha = rnd_train[beat_idx] / Ase
#         signal_noise = beat + noise_segment
#         # signal_noise = beat + alpha * noise_segment
#         sn_train.append(signal_noise)
#         # noise_indices_train.append(noise_combination_idx)  # 노이즈 인덱스 저장
#         noise_index += samples
#         # print(f"[DEBUG] Beat train{beat_idx}, noise index: {noise_index}")
#         # print(len(noise_source))
#         if noise_index > (len(noise_source) - samples):
#             noise_index = 0

                
#     # Adding noise to test
#     noise_index = 0
#     rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
#     # Saving the random array so we can use it on the amplitude segmentation tables
#     np.save('rnd_test.npy', rnd_test)
#     print('rnd_test shape: ' + str(rnd_test.shape))
    
#     for beat_idx, beat in enumerate(beats_test):
#         # if np.random.rand() < channel_ratio:
#         noise_source = test_noise_1  # Lower half of channel 1
#         # else:
#         #     noise_source = test_noise_2  # Upper half of channel 2
#         # 노이즈 조합도 순차적으로 선택, 주기적으로 변화를 줌 (매 8회 주기)
#         # noise_combination_idx = (beat_idx % 7) + 1  # 1부터 7까지 순차적으로 선택
#         # noise_combination_idx = 0  # 1부터 7까지 순차적으로 선택
#         # # noise = combined_noise[selected_channel][:, noise_combination_idx]
#         # noise = static_noise[selected_channel][:, noise_combination_idx]
#         noise_segment = noise_source[noise_index:noise_index + samples]
#         # beat_max_value = np.max(beat) - np.min(beat)
#         # noise_max_value = np.max(noise_segment) - np.min(noise_segment)
#         # if noise_max_value == 0:
#         #     Ase = 1  # 기본값 설정
#         # else:
#         #     Ase = noise_max_value / beat_max_value
#         # alpha = rnd_train[beat_idx] / Ase
#         signal_noise = beat + noise_segment
#         # signal_noise = beat + alpha * noise_segment
#         # signal_noise = beat + noise_segment
#         sn_test.append(signal_noise)
#         # noise_indices_test.append(noise_combination_idx)  # 노이즈 인덱스 저장
#         noise_index += samples
#         # print(f"[DEBUG] Beat test {beat_idx}, noise index: {noise_index}")
#         # print(len(noise_source))
#         if noise_index > (len(noise_source) - samples):
#             noise_index = 0
#     X_train = np.array(sn_train)[valid_train_indices]  # Match noisy and original beats
#     X_test = np.array(sn_test)[valid_test_indices]

#     y_train = np.array(beats_train)[valid_train_indices]  # Match noisy and original beats
#     y_test = np.array(beats_test)[valid_test_indices]

#     X_train = np.expand_dims(X_train, axis=2)
#     y_train = np.expand_dims(y_train, axis=2)

#     X_test = np.expand_dims(X_test, axis=2)
#     y_test = np.expand_dims(y_test, axis=2)

#     Dataset = [X_train, y_train, X_test, y_test]
#     print(f"[INFO] Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
#     print('Dataset ready to use.')

#     return Dataset, valid_train_indices, valid_test_indices

