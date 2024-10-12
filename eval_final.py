# Define Metric functions
from sklearn.metrics.pairwise import cosine_similarity
import _pickle as pickle
from datetime import datetime
import time
import numpy as np
import os
from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils.visualization import generate_hboxplot, generate_violinplots, generate_barplot, generate_boxplot
from utils.visualization import generate_table, generate_table_time, ecg_view_3d, ecg_view
# from utils import visualization as vs
from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier
from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from deepFilter.dl_pipeline import train_dl, test_dl


if __name__ == "__main__":
    # Get the current date in 'MMDD' format
    current_date = datetime.now().strftime('%m%d')
    dl_experiments = [  'DRNN',
                        'FCN-DAE',
                        'Vanilla L',
                        'Vanilla NL',
                        'Multibranch LANL',
                        'Multibranch LANLD','Transformer_DAE','Transformer_COMBDAE','Transformer_COMBDAE_FreTS']

    # def SSD(y, y_pred):
    #     return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


    # def MAD(y, y_pred):
    #     return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


    # def PRD(y, y_pred):
    #     N = np.sum(np.square(y_pred - y), axis=1)
    #     D = np.sum(np.square(y_pred - np.mean(y)), axis=1)
    #     PRD = np.sqrt(N/D) * 100
    #     return PRD

    # def COS_SIM(y, y_pred):
    #     cos_sim = []
    #     y = np.squeeze(y, axis=-1)
    #     y_pred = np.squeeze(y_pred, axis=-1)
    #     for idx in range(len(y)):
    #         kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
    #         cos_sim.append(kl_temp)
    #     cos_sim = np.array(cos_sim)
    #     return cos_sim

    ####### LOAD EXPERIMENTS #######

    # Load Results DRNN
    with open('1006/test_results_' + dl_experiments[0] + '.pkl', 'rb') as input:
        test_DRNN = pickle.load(input)

    # Load Results FCN_DAE
    with open('1006/test_results_' + dl_experiments[1] + '.pkl', 'rb') as input:
        test_FCN_DAE = pickle.load(input)

    # Load Results Vanilla L
    with open('1006/test_results_' + dl_experiments[2] + '.pkl', 'rb') as input:
        test_Vanilla_L = pickle.load(input)

    # Load Results Exp Vanilla NL
    with open('1006/test_results_' + dl_experiments[3] + '.pkl', 'rb') as input:
        test_Vanilla_NL = pickle.load(input)

    # Load Results Multibranch LANL
    with open('1006/test_results_' + dl_experiments[4] + '.pkl', 'rb') as input:
        test_Multibranch_LANL = pickle.load(input)

    # Load Results Multibranch LANLD
    with open('1006/test_results_' + dl_experiments[5] + '.pkl', 'rb') as input:
        test_Multibranch_LANLD = pickle.load(input)
    # Load Results Transformer_DAE

    with open('1010/test_results_' + dl_experiments[6] + '.pkl', 'rb') as input:
        test_Transformer_DAE = pickle.load(input)

    # # Load Results Transformer_FDAE
    # with open('0920/test_results_' + dl_experiments[7] + '.pkl', 'rb') as input:
    #     test_Transformer_FDAE = pickle.load(input)
        # Transformer_COMBDAE_with_CrossDomainAttention
    # Load Results Transformer_FDAE
    with open('1010/test_results_' + dl_experiments[7] + '.pkl', 'rb') as input:
        test_Transformer_COMBDAE = pickle.load(input)
            
    with open('1012/test_results_' + dl_experiments[8] + '.pkl', 'rb') as input:
        test_Transformer_COMBDAE_FreTS = pickle.load(input)   
        
        
    # Load Result FIR Filter
    with open('1005/test_results_FIR.pkl', 'rb') as input:
        test_FIR = pickle.load(input)

    # Load Result IIR Filter
    with open('1005/test_results_IIR.pkl', 'rb') as input:
        test_IIR = pickle.load(input)

    ####### Calculate Metrics #######

    print('Calculating metrics ...')

    # DL Metrics

    # Exp FCN-DAE

    [X_test, y_test, y_pred] = test_DRNN

    SSD_values_DL_DRNN = SSD(y_test, y_pred)

    MAD_values_DL_DRNN = MAD(y_test, y_pred)

    PRD_values_DL_DRNN = PRD(y_test, y_pred)

    COS_SIM_values_DL_DRNN = COS_SIM(y_test, y_pred)

    # Exp FCN-DAE

    [X_test, y_test, y_pred] = test_FCN_DAE

    SSD_values_DL_FCN_DAE = SSD(y_test, y_pred)

    MAD_values_DL_FCN_DAE = MAD(y_test, y_pred)

    PRD_values_DL_FCN_DAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_FCN_DAE = COS_SIM(y_test, y_pred)

    # Vanilla L

    [X_test, y_test, y_pred] = test_Vanilla_L

    SSD_values_DL_exp_1 = SSD(y_test, y_pred)

    MAD_values_DL_exp_1 = MAD(y_test, y_pred)

    PRD_values_DL_exp_1 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_1 = COS_SIM(y_test, y_pred)

    # Vanilla_NL

    [X_test, y_test, y_pred] = test_Vanilla_NL

    SSD_values_DL_exp_2 = SSD(y_test, y_pred)

    MAD_values_DL_exp_2 = MAD(y_test, y_pred)

    PRD_values_DL_exp_2 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_2 = COS_SIM(y_test, y_pred)

    # Multibranch_LANL

    [X_test, y_test, y_pred] = test_Multibranch_LANL

    SSD_values_DL_exp_3 = SSD(y_test, y_pred)

    MAD_values_DL_exp_3 = MAD(y_test, y_pred)

    PRD_values_DL_exp_3 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_3 = COS_SIM(y_test, y_pred)

    # Multibranch_LANLD

    [X_test, y_test, y_pred] = test_Multibranch_LANLD

    SSD_values_DL_exp_4 = SSD(y_test, y_pred)

    MAD_values_DL_exp_4 = MAD(y_test, y_pred)

    PRD_values_DL_exp_4 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_4 = COS_SIM(y_test, y_pred)


    # Transformer_DAE

    [X_test, y_test, y_pred] = test_Transformer_DAE

    SSD_values_DL_exp_5 = SSD(y_test, y_pred)

    MAD_values_DL_exp_5 = MAD(y_test, y_pred)

    PRD_values_DL_exp_5 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_5 = COS_SIM(y_test, y_pred)

    # Transformer_FDAE

    [X_test, y_test, y_pred] = test_Transformer_COMBDAE

    SSD_values_DL_exp_6 = SSD(y_test, y_pred)

    MAD_values_DL_exp_6 = MAD(y_test, y_pred)

    PRD_values_DL_exp_6 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_6 = COS_SIM(y_test, y_pred)


    # Transformer_FDAE

    [X_test, y_test, y_pred] = test_Transformer_COMBDAE_FreTS

    SSD_values_DL_exp_7 = SSD(y_test, y_pred)

    MAD_values_DL_exp_7 = MAD(y_test, y_pred)

    PRD_values_DL_exp_7 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_7 = COS_SIM(y_test, y_pred)

    # Digital Filtering

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR

    SSD_values_FIR = SSD(y_test, y_filter)

    MAD_values_FIR = MAD(y_test, y_filter)

    PRD_values_FIR = PRD(y_test, y_filter)

    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)

    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR

    SSD_values_IIR = SSD(y_test, y_filter)

    MAD_values_IIR = MAD(y_test, y_filter)

    PRD_values_IIR = PRD(y_test, y_filter)

    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)

    ####### Results Visualization #######

    SSD_all = [SSD_values_FIR,
                SSD_values_IIR,
                SSD_values_DL_FCN_DAE,
                SSD_values_DL_DRNN,
                SSD_values_DL_exp_1,
                SSD_values_DL_exp_2,
                SSD_values_DL_exp_3,
                SSD_values_DL_exp_4,
                SSD_values_DL_exp_5,
                SSD_values_DL_exp_6,
                SSD_values_DL_exp_7]

    MAD_all = [MAD_values_FIR,
            MAD_values_IIR,
            MAD_values_DL_FCN_DAE,
            MAD_values_DL_DRNN,
            MAD_values_DL_exp_1,
            MAD_values_DL_exp_2,
            MAD_values_DL_exp_3,
            MAD_values_DL_exp_4,
            MAD_values_DL_exp_5,
            MAD_values_DL_exp_6,
            MAD_values_DL_exp_7
            ]

    PRD_all = [PRD_values_FIR,
            PRD_values_IIR,
            PRD_values_DL_FCN_DAE,
            PRD_values_DL_DRNN,
            PRD_values_DL_exp_1,
            PRD_values_DL_exp_2,
            PRD_values_DL_exp_3,
            PRD_values_DL_exp_4,
            PRD_values_DL_exp_5,
            PRD_values_DL_exp_6,
            PRD_values_DL_exp_7
            ]

    CORR_all = [COS_SIM_values_FIR,
                COS_SIM_values_IIR,
                COS_SIM_values_DL_FCN_DAE,
                COS_SIM_values_DL_DRNN,
                COS_SIM_values_DL_exp_1,
                COS_SIM_values_DL_exp_2,
                COS_SIM_values_DL_exp_3,
                COS_SIM_values_DL_exp_4,
                COS_SIM_values_DL_exp_5,
                COS_SIM_values_DL_exp_6,
                COS_SIM_values_DL_exp_7
                ]

    Exp_names = ['FIR Filter', 'IIR Filter'] + dl_experiments

    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, CORR_all]

    # Metrics table
    generate_table(metrics, metric_values, Exp_names)

    
    # timing_var = ['training', 'test']
    # generate_table_time(timing_var, timing, Exp_names, gpu=True)
    
    
    rnd_test = np.load('rnd_test.npy')

    # rnd_test = np.concatenate([rnd_test, rnd_test])

    segm = [0.2, 0.6, 1.0, 1.5, 2.0]  # real number of segmentations is len(segmentations) - 1
    SSD_seg_all = []
    MAD_seg_all = []
    PRD_seg_all = []
    COS_SIM_seg_all = []

    for idx_exp in range(len(Exp_names)):
        SSD_seg = [None] * (len(segm) - 1)
        MAD_seg = [None] * (len(segm) - 1)
        PRD_seg = [None] * (len(segm) - 1)
        COS_SIM_seg = [None] * (len(segm) - 1)
        for idx_seg in range(len(segm) - 1):
            SSD_seg[idx_seg] = []
            MAD_seg[idx_seg] = []
            PRD_seg[idx_seg] = []
            COS_SIM_seg[idx_seg] = []
            for idx in range(len(rnd_test)):
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)

                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)

                # PRD
                oua = PRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    PRD_seg[idx_seg].append(oua)

                # COS SIM
                oua = CORR_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    COS_SIM_seg[idx_seg].append(oua)

        # Processing the last index
        # SSD
        SSD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SSD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SSD_seg[-1].append(oua)

        SSD_seg_all.append(SSD_seg)  # [exp][seg][item]

        # MAD
        MAD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAD_seg[-1].append(oua)

        MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

        # PRD
        PRD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = PRD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                PRD_seg[-1].append(oua)

        PRD_seg_all.append(PRD_seg)  # [exp][seg][item]

        # COS SIM
        COS_SIM_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = CORR_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                COS_SIM_seg[-1].append(oua)

        COS_SIM_seg_all.append(COS_SIM_seg)  # [exp][seg][item]

    # Printing Tables
    seg_table_column_name = []
    for idx_seg in range(len(segm) - 1):
        column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
        seg_table_column_name.append(column_name)

    # SSD Table
    SSD_seg_all = np.array(SSD_seg_all)
    SSD_seg_all = np.swapaxes(SSD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the SSD metric')
    generate_table(seg_table_column_name, SSD_seg_all, Exp_names)

    # MAD Table
    MAD_seg_all = np.array(MAD_seg_all)
    MAD_seg_all = np.swapaxes(MAD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the MAD metric')
    generate_table(seg_table_column_name, MAD_seg_all, Exp_names)

    # PRD Table
    PRD_seg_all = np.array(PRD_seg_all)
    PRD_seg_all = np.swapaxes(PRD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the PRD metric')
    generate_table(seg_table_column_name, PRD_seg_all, Exp_names)

    # COS SIM Table
    COS_SIM_seg_all = np.array(COS_SIM_seg_all)
    COS_SIM_seg_all = np.swapaxes(COS_SIM_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the COS SIM metric')
    generate_table(seg_table_column_name, COS_SIM_seg_all, Exp_names)

    # 저장 경로 설정
    save_directory = 'plots'

    # 파일 이름 설정
    filename_ssd_hbox = 'ssd_hboxplot.png'
    filename_ssd_violin = 'ssd_violinplot.png'
    filename_ssd_bar = 'ssd_barplot.png'
    filename_ssd_box = 'ssd_boxplot.png'

    filename_mad_hbox = 'mad_hboxplot.png'
    filename_mad_violin = 'mad_violinplot.png'
    filename_mad_bar = 'mad_barplot.png'
    filename_mad_box = 'mad_boxplot.png'

    filename_prd_hbox = 'prd_hboxplot.png'
    filename_prd_violin = 'prd_violinplot.png'
    filename_prd_bar = 'prd_barplot.png'
    filename_prd_box = 'prd_boxplot.png'

    filename_cos_hbox = 'cos_hboxplot.png'
    filename_cos_violin = 'cos_violinplot.png'
    filename_cos_bar = 'cos_barplot.png'
    filename_cos_box = 'cos_boxplot.png'

    # SSD 그래프들 생성
    print("Generating SSD plots...")
    generate_hboxplot(SSD_all, Exp_names, 'SSD (au)', log=False, save_dir=save_directory, filename=filename_ssd_hbox, set_x_axis_size=(0, 100.1))
    generate_violinplots(SSD_all, Exp_names, 'SSD (au)', log=False, save_dir=save_directory, filename=filename_ssd_violin, set_x_axis_size=(0, 100.1))
    generate_barplot(SSD_all, Exp_names, 'SSD (au)', log=False, save_dir=save_directory, filename=filename_ssd_bar, set_x_axis_size=(0, 100.1))
    generate_boxplot(SSD_all, Exp_names, 'SSD (au)', log=False, save_dir=save_directory, filename=filename_ssd_box, set_x_axis_size=(0, 100.1))

    # MAD 그래프들 생성
    print("Generating MAD plots...")
    generate_hboxplot(MAD_all, Exp_names, 'MAD (au)', log=False, save_dir=save_directory, filename=filename_mad_hbox, set_x_axis_size=(0, 3.01))
    generate_violinplots(MAD_all, Exp_names, 'MAD (au)', log=False, save_dir=save_directory, filename=filename_mad_violin, set_x_axis_size=(0, 3.01))
    generate_barplot(MAD_all, Exp_names, 'MAD (au)', log=False, save_dir=save_directory, filename=filename_mad_bar, set_x_axis_size=(0, 3.01))
    generate_boxplot(MAD_all, Exp_names, 'MAD (au)', log=False, save_dir=save_directory, filename=filename_mad_box, set_x_axis_size=(0, 3.01))

    # PRD 그래프들 생성
    print("Generating PRD plots...")
    generate_hboxplot(PRD_all, Exp_names, 'PRD (au)', log=False, save_dir=save_directory, filename=filename_prd_hbox, set_x_axis_size=(0, 150.1))
    generate_violinplots(PRD_all, Exp_names, 'PRD (au)', log=False, save_dir=save_directory, filename=filename_prd_violin, set_x_axis_size=(0, 150.1))
    generate_barplot(PRD_all, Exp_names, 'PRD (au)', log=False, save_dir=save_directory, filename=filename_prd_bar, set_x_axis_size=(0, 150.1))
    generate_boxplot(PRD_all, Exp_names, 'PRD (au)', log=False, save_dir=save_directory, filename=filename_prd_box, set_x_axis_size=(0, 150.1))

    # Cosine Similarity 그래프들 생성
    print("Generating Cosine Similarity plots...")
    generate_hboxplot(CORR_all, Exp_names, 'Cosine Similarity (0-1)', log=False, save_dir=save_directory, filename=filename_cos_hbox, set_x_axis_size=(0, 1))
    generate_violinplots(CORR_all, Exp_names, 'Cosine Similarity (0-1)', log=False, save_dir=save_directory, filename=filename_cos_violin, set_x_axis_size=(0, 1))
    generate_barplot(CORR_all, Exp_names, 'Cosine Similarity (0-1)', log=False, save_dir=save_directory, filename=filename_cos_bar, set_x_axis_size=(0, 1))
    generate_boxplot(CORR_all, Exp_names, 'Cosine Similarity (0-1)', log=False, save_dir=save_directory, filename=filename_cos_box, set_x_axis_size=(0, 1))
    
    # Test signal plotting
    signals_index = np.array([110, 210, 410, 810, 1610, 3210, 6410, 12810]) + 10
    noise_indices = np.random.randint(1, 8, size=len(signals_index))  # 노이즈 인덱스 임의 설정

    ecg_signals2plot = []
    ecgbl_signals2plot = []
    dl_signals2plot = []
    fil_signals2plot = []

    [X_test, y_test, y_pred] = test_Transformer_COMBDAE
    for id in signals_index:
        ecgbl_signals2plot.append(X_test[id])
        ecg_signals2plot.append(y_test[id])
        dl_signals2plot.append(y_pred[id])

    [X_test, y_test, y_filter] = test_IIR
    for id in signals_index:
        fil_signals2plot.append(y_filter[id])

    # Plotting signals with noise information
    for i in range(len(signals_index)):
        ecg_view(ecg=ecg_signals2plot[i],
                ecg_blw=ecgbl_signals2plot[i],
                ecg_dl=dl_signals2plot[i],
                ecg_f=fil_signals2plot[i],
                noise_index=noise_indices[i],
                signal_name=None,
                beat_no=i,
                save_dir='view')
        
    # 테스트 신호를 사용한 3D 플로팅
    for i in range(len(signals_index)):
        ecg_view_3d(ecg=ecg_signals2plot[i],
                    ecg_blw=ecgbl_signals2plot[i],
                    ecg_dl=dl_signals2plot[i],
                    ecg_f=fil_signals2plot[i],
                    noise_index=noise_indices[i],
                    signal_name=None,
                    beat_no=i,
                    save_dir='view')