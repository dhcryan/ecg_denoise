import _pickle as pickle
from datetime import datetime
import time
import numpy as np
import os
from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils.visualization import visualize_multiple_beats, visualize_signals
# from utils import visualization as vs
from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from deepFilter.dl_pipeline import train_dl, test_dl

if __name__ == "__main__":
    # Get the current date in 'MMDD' format
    current_date = datetime.now().strftime('%m%d')
    # Prepare the dataset
    Dataset, valid_train_indices, valid_test_indices, noise_indices_train, noise_indices_test = Data_Preparation(samples=512, channel_ratio=0.5)

    train_time_list = []
    test_time_list = []

    # FIR Filter
    print('Running FIR filter on the test set. This will take a while (2h)...')
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
    end_test = datetime.now()
    
    # FIR: 학습 시간은 0으로 처리
    train_time_list.append(0)
    test_time_list.append(end_test - start_test)

    test_results_FIR = [X_test_f, y_test_f, y_filter]

    save_dir = current_date
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save Results
    with open(os.path.join(save_dir, 'test_results_FIR' + '.pkl'), 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_FIR, output)
    print('Results from experiment ' + 'test_results_FIR' + ' saved')
    
    # IIR Filter
    print('Running IIR filter on the test set. This will take a while (25 mins)...')
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
    end_test = datetime.now()

    # IIR: 학습 시간은 0으로 처리
    train_time_list.append(0)
    test_time_list.append(end_test - start_test)

    test_results_IIR = [X_test_f, y_test_f, y_filter]

    save_dir = current_date
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save Results
    with open(os.path.join(save_dir, 'test_results_IIR' + '.pkl'), 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_IIR, output)
    print('Results from experiment ' + 'test_results_IIR' + ' saved')

    # Save timing information (학습 시간은 0, 테스트 시간은 기록)
    timing = [train_time_list, test_time_list]
    with open('timing.pkl', 'wb') as output:
        pickle.dump(timing, output)
    print('Timing for FIR and IIR filters saved')

