import _pickle as pickle
from datetime import datetime
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from Data_Preparation import data_preparation as dp

if __name__ == "__main__":
    
    # Prepare the dataset
    Dataset, valid_train_indices, valid_test_indices = dp.Data_Preparation(samples=512, channel_ratio=0.5)

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

    # Save FIR filter results
    with open('test_results_FIR.pkl', 'wb') as output:
        pickle.dump(test_results_FIR, output)
    print('Results from FIR filter saved')

    # IIR Filter
    print('Running IIR filter on the test set. This will take a while (25 mins)...')
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
    end_test = datetime.now()

    # IIR: 학습 시간은 0으로 처리
    train_time_list.append(0)
    test_time_list.append(end_test - start_test)

    test_results_IIR = [X_test_f, y_test_f, y_filter]

    # Save IIR filter results
    with open('test_results_IIR.pkl', 'wb') as output:
        pickle.dump(test_results_IIR, output)
    print('Results from IIR filter saved')

    # Save timing information (학습 시간은 0, 테스트 시간은 기록)
    timing = [train_time_list, test_time_list]
    with open('timing.pkl', 'wb') as output:
        pickle.dump(timing, output)
    print('Timing for FIR and IIR filters saved')
