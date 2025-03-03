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
        
        # 훈련 시간 측정 및 모델 훈련
        start_train = datetime.now()
        train_dl(Dataset, experiment)
        end_train = datetime.now()
        train_time_list.append(end_train - start_train)

        # 테스트 시간 측정 및 모델 테스트
        start_test = datetime.now()
        [X_test, y_test, y_pred] = test_dl(Dataset, experiment)
        end_test = datetime.now()
        test_time_list.append(end_test - start_test)

        # 시각화 및 결과 저장
        save_dir = current_date
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_results = [X_test, y_test, y_pred]
        with open(os.path.join(save_dir, 'test_results_' + experiment + '.pkl'), 'wb') as output:
            pickle.dump(test_results, output)
        print('Results from experiment ' + experiment + ' saved')

        # # 타이밍 저장
        # timing = [train_time_list, test_time_list]
        # with open(os.path.join(save_dir, 'timing_' + experiment + '.pkl'), 'wb') as output:
        #     pickle.dump(timing, output)
        # print('Timing from experiment ' + experiment + ' saved')    
