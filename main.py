import _pickle as pickle
from datetime import datetime
import time
import numpy as np
import os

from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils import visualization as vs
# from utils import visualization as vs
from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from deepFilter.dl_pipeline import train_dl, test_dl


if __name__ == "__main__":

    # dl_experiments = [
    #                   'DRNN',
    #                   'FCN-DAE',
    #                   'Vanilla L',
    #                   'Vanilla NL',
    #                   'Multibranch LANL',
    #                   'Multibranch LANLD',
    #                   'Transformer_DAE',
    #                   'Transformer_FDAE']
    dl_experiments = ['Transformer_DAE', 'Transformer_COMBDAE']

    train_time_list = []
    test_time_list = []
    
    # Get the current date in 'MMDD' format
    current_date = datetime.now().strftime('%m%d')
    
    for experiment in range(len(dl_experiments)):
        if experiment == 'Transformer_COMBDAE':
            Dataset, valid_train_indices, valid_test_indices, noise_indices_train, noise_indices_test = Data_Preparation_with_Fourier(samples=512, channel_ratio=0.5, fs=360)
            start_train = datetime.now()
            train_dl(Dataset, dl_experiments[experiment])
            end_train = datetime.now()
            train_time_list.append(end_train - start_train)

            start_test = datetime.now()
            [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])
            end_test = datetime.now()
            test_time_list.append(end_test - start_test)

            test_results = [X_test, y_test, y_pred]
            # 폴더 경로 설정
            save_dir = current_date

            # 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save Results
            with open(os.path.join(save_dir, 'test_results_' + dl_experiments[experiment] + '.pkl'), 'wb') as output:  # Overwrites any existing file.
                pickle.dump(test_results, output)
            print('Results from experiment ' + dl_experiments[experiment] + ' saved')
        
        else:
            Dataset, valid_train_indices, valid_test_indices, noise_indices_train, noise_indices_test = Data_Preparation(samples=512, channel_ratio=0.5)
            start_train = datetime.now()
            train_dl(Dataset, dl_experiments[experiment])
            end_train = datetime.now()
            train_time_list.append(end_train - start_train)

            start_test = datetime.now()
            [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])
            end_test = datetime.now()
            test_time_list.append(end_test - start_test)

            test_results = [X_test, y_test, y_pred]
            # 폴더 경로 설정
            save_dir = current_date

            # 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save Results
            with open(os.path.join(save_dir, 'test_results_' + dl_experiments[experiment] + '.pkl'), 'wb') as output:  # Overwrites any existing file.
                pickle.dump(test_results, output)
            print('Results from experiment ' + dl_experiments[experiment] + ' saved')