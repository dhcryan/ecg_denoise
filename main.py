import _pickle as pickle
from datetime import datetime
import time
import numpy as np

from utils.metrics import MAD, SSD, PRD, COS_SIM
# from utils import visualization as vs
from Data_Preparation import data_preparation as dp

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
    
    dl_experiments = ['MemoryTransformer_DAE']
    
    Dataset, valid_train_indices, valid_test_indices = dp.Data_Preparation(samples=512, channel_ratio=0.5)

    train_time_list = []
    test_time_list = []

    for experiment in range(len(dl_experiments)):
        start_train = datetime.now()
        train_dl(Dataset, dl_experiments[experiment])
        end_train = datetime.now()
        train_time_list.append((end_train - start_train).total_seconds())

        start_test = datetime.now()
        [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])
        end_test = datetime.now()
        test_time_list.append((end_test - start_test).total_seconds())

        test_results = [X_test, y_test, y_pred]

        # Save Results
        with open('0920/test_results_' + dl_experiments[experiment] + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(test_results, output)
        print('Results from experiment ' + dl_experiments[experiment] + ' saved')
    
    # Saving timing list
    timing = [train_time_list, test_time_list]
    with open('timing.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(timing, output)
    print('Timing saved')
    
    

# import _pickle as pickle
# from datetime import datetime
# import time
# import numpy as np

# from utils.metrics import MAD, SSD, PRD, COS_SIM
# from utils import visualization as vs
# from Data_Preparation import data_preparation as dp

# from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
# from deepFilter.dl_pipeline import train_dl, test_dl

# import tensorflow as tf

# if __name__ == "__main__":

#     # Set up strategy for multi-GPU
#     strategy = tf.distribute.MirroredStrategy()

#     # Print available devices
#     print(f"Number of devices: {strategy.num_replicas_in_sync}")

#     dl_experiments = [
#         'DRNN',
#         'FCN-DAE',
#         'Vanilla L',
#         'Vanilla NL',
#         'Multibranch LANL',
#         'Multibranch LANLD',
#         'Transformer_DAE',
#         'Transformer_FDAE'
#     ]
    
#     # Prepare the dataset
#     Dataset, valid_train_indices, valid_test_indices = dp.Data_Preparation(samples=512, channel_ratio=0.5)

#     train_time_list = []
#     test_time_list = []

#     # Loop through each experiment
#     for experiment in range(len(dl_experiments)):

#         # Start training timer
#         start_train = datetime.now()

#         # Distribute the model and dataset training across multiple GPUs
#         with strategy.scope():
#             train_dl(Dataset, dl_experiments[experiment])

#         # End training timer
#         end_train = datetime.now()
#         train_time_list.append((end_train - start_train).total_seconds())

#         # Start testing timer
#         start_test = datetime.now()

#         # Distribute the model testing across multiple GPUs
#         with strategy.scope():
#             [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])

#         # End testing timer
#         end_test = datetime.now()
#         test_time_list.append((end_test - start_test).total_seconds())

#         # Save the test results
#         test_results = [X_test, y_test, y_pred]
#         with open('0920/test_results_' + dl_experiments[experiment] + '.pkl', 'wb') as output:  # Overwrites any existing file.
#             pickle.dump(test_results, output)
#         print(f'Results from experiment {dl_experiments[experiment]} saved')

#     # Save timing information
#     timing = [train_time_list, test_time_list]
#     with open('timing.pkl', 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(timing, output)
#     print('Timing saved')
