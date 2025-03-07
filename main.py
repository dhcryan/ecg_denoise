import _pickle as pickle
from datetime import datetime
import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, firwin, filtfilt, butter
from utils.metrics import MAD, SSD, PRD, COS_SIM, SNR
from utils.visualization import visualize_multiple_beats, visualize_signals, plot_ecg_comparison_separate
from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier
from Data_Preparation.data_preparation_only_fourier import Data_Preparation_only_Fourier
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset, FIRRemoveBL, IIRRemoveBL
from deepFilter.dl_pipeline import train_dl, test_dl

if __name__ == "__main__":

    # dl_experiments = [ 'DRNN', 'DeepFilter','CNN_DAE','FCN_DAE','AttentionSkipDAE','Transformer_DAE']
    dl_experiments = ['Dual_FreqDAE']
    train_time_list = []
    test_time_list = []
    
    # Get the current date in 'MMDD' format
    current_date = datetime.now().strftime('%m%d')
    for experiment in dl_experiments:
        
        # 데이터 준비 단계
        if experiment in ['Dual_FreqDAE']:
            Dataset, valid_train_indices, valid_test_indices = Data_Preparation_with_Fourier(samples=512, fs=360)
            X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y = Dataset         
        else:
            Dataset, valid_train_indices, valid_test_indices = Data_Preparation(samples=512)
            X_train, y_train, X_test, y_test = Dataset
        
        train_dl(Dataset, experiment)

        [X_test, y_test, y_pred] = test_dl(Dataset, experiment)

        save_dir = current_date
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_results = [X_test, y_test, y_pred]
        with open(os.path.join(save_dir, 'test_results_' + experiment + '.pkl'), 'wb') as output:
            pickle.dump(test_results, output)
        print('Results from experiment ' + experiment + ' saved')

# import _pickle as pickle
# import os
# import pandas as pd
# from datetime import datetime
# from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier
# from deepFilter.dl_pipeline import test_dl

# if __name__ == "__main__":

#     dl_experiments = ['Transformer_DAE']
#     current_date = datetime.now().strftime('%m%d')
#     save_dir = f"test_results_125_noise_Transformer_DAE_{current_date}"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ✅ SNR 조합 정보 불러오기
#     snr_combinations_df = pd.read_csv('data/SNR_Combinations_125.csv')

#     for noise_idx in range(125):  # 125개 노이즈에 대해 반복
#         bw_snr = snr_combinations_df.loc[noise_idx, "BW_SNR"]
#         em_snr = snr_combinations_df.loc[noise_idx, "EM_SNR"]
#         ma_snr = snr_combinations_df.loc[noise_idx, "MA_SNR"]

#         print(f"\n[INFO] Testing Noise {noise_idx+1}/125")
#         print(f"[INFO] SNR Levels: BW={bw_snr}, EM={em_snr}, MA={ma_snr}")
#         for experiment in dl_experiments:
#         # ✅ 데이터셋 생성 (각 노이즈 적용)
#             if experiment in ['Dual_FreqDAE']:
#                 Dataset, valid_train_indices, valid_test_indices = Data_Preparation_with_Fourier(samples=512, fs=360, noise_index=noise_idx)
#                 X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y = Dataset         
#             else:
#                 Dataset, valid_train_indices, valid_test_indices = Data_Preparation(samples=512, noise_index=noise_idx)
#                 X_train, y_train, X_test, y_test = Dataset
#             # ✅ 모델 테스트 수행
#             [X_test, y_test, y_pred] = test_dl(Dataset, 'Transformer_DAE')

#             # ✅ 결과 저장
#             result_filename = f'test_results_BW{bw_snr}_EM{em_snr}_MA{ma_snr}.pkl'
#             with open(os.path.join(save_dir, result_filename), 'wb') as output:
#                 pickle.dump([X_test, y_test, y_pred], output)

#             print(f"[INFO] Test result saved: {result_filename}")
