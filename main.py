import _pickle as pickle
from datetime import datetime
import time
import numpy as np
import os
from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils.visualization import visualize_multiple_beats, visualize_signals, plot_ecg_comparison_separate
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
    #                   'Multibranch LANLD', ]
    dl_experiments = ['Transformer_DAE', 'Transformer_COMBDAE']

    train_time_list = []
    test_time_list = []
    
    # Get the current date in 'MMDD' format
    current_date = datetime.now().strftime('%m%d')
    
    for experiment in range(len(dl_experiments)):
        if dl_experiments[experiment] in ['Transformer_COMBDAE','Transformer_COMBDAE_with_band_encoding','Transformer_COMBDAE_FreTS']:
            Dataset, valid_train_indices, valid_test_indices, noise_indices_train, noise_indices_test = Data_Preparation_with_Fourier(samples=512, channel_ratio=0.5, fs=360)
            X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y = Dataset
            start_train = datetime.now()
            train_dl(Dataset, dl_experiments[experiment])
            end_train = datetime.now()
            train_time_list.append(end_train - start_train)

            start_test = datetime.now()
            [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])
            end_test = datetime.now()
            test_time_list.append(end_test - start_test)
            # 시각화 호출 예시
            visualize_multiple_beats(X_train, y_train, noise_indices_train, num_samples=5)
            visualize_signals(y_train, X_train, fs=360, num_samples=5, signal_length=512, save_dir='visualizations')
            # Visualize the comparison for training data and save the plots
            plot_ecg_comparison_separate(X_train, y_train, valid_train_indices, "Training Set", num_beats=5, save_dir='visualizations')
            # Visualize the comparison for testing data and save the plots
            plot_ecg_comparison_separate(X_test, y_test, valid_test_indices, "Testing Set", num_beats=5, save_dir='visualizations')
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
            X_train, y_train, X_test, y_test = Dataset
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
            
        timing = [train_time_list, test_time_list]
        with open(os.path.join(save_dir, 'timing_' + dl_experiments[experiment] + '.pkl'), 'wb') as output:
            pickle.dump(timing, output)
        print('Timing from experiment ' + dl_experiments[experiment] + ' saved')