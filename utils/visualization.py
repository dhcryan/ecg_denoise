#============================================================
#
#  Deep Learning BLW Filtering
#  Data Visualization
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy.fft import fft
from scipy.signal import spectrogram, stft
import random
import os

# 노이즈 조합에 대한 설명을 미리 정의
NOISE_COMBINATIONS = {
    1: "Baseline Wander (BW) only",
    2: "Electrode Motion (EM) only",
    3: "Muscle Artifact (MA) only",
    4: "BW + EM",
    5: "BW + MA",
    6: "EM + MA",
    7: "BW + EM + MA"
}
def ensure_directory(directory):
    """ 디렉토리가 존재하지 않으면 생성 """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def ecg_view_noisy_comparison(ecg_clean, ecg_noisy, noise_index, signal_name=None, sample_index=None, save_dir='visualizations'):
    """
    원래 신호와 노이즈가 추가된 신호를 시각화하고, 어떤 노이즈 조합이 사용되었는지 표시합니다.
    
    Parameters:
    ecg_clean: 노이즈가 없는 원래 ECG 신호.
    ecg_noisy: 노이즈가 추가된 ECG 신호.
    noise_index: 8가지 노이즈 조합 중 어떤 조합이 사용되었는지 나타내는 인덱스 (1~8).
    signal_name: 시각화할 신호의 이름 (Optional).
    sample_index: 선택된 샘플 번호 (Optional).
    """
    ensure_directory(save_dir)  # 디렉토리 생성
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 원래 신호 (Clean ECG)
    plt.plot(ecg_clean, 'g', label='ECG Original (Clean)')
    
    # 노이즈가 추가된 신호 (Noisy ECG)
    plt.plot(ecg_noisy, 'r', label='ECG Noisy')
    
    plt.grid(True)
    plt.ylabel('Amplitude (au)')
    plt.xlabel('Samples')
    
    # 범례 표시
    leg = ax.legend()

    # 노이즈 인덱스에 해당하는 노이즈 조합 정보 가져오기
    noise_description = NOISE_COMBINATIONS.get(noise_index, "Unknown Noise Combination")

    # 추가적인 시각화 정보 계산
    amplitude_range_clean = np.max(ecg_clean) - np.min(ecg_clean)
    amplitude_range_noisy = np.max(ecg_noisy) - np.min(ecg_noisy)
    mean_clean = np.mean(ecg_clean)
    mean_noisy = np.mean(ecg_noisy)

    # 신호 및 샘플 번호에 대한 제목 설정 (노이즈 인덱스 포함)
    if signal_name is not None and sample_index is not None:
        plt.title(f"Signal {signal_name}, Sample Index: {sample_index}, Noise Index: {noise_index} ({noise_description})")
    else:
        plt.title(f"ECG Signal Comparison with Noise Index: {noise_index} ({noise_description})")

    # # 추가 정보 표시 (진폭 범위와 평균값 비교)
    # textstr = f"Amplitude Range (Clean): {amplitude_range_clean:.2f}\n"\
    #           f"Amplitude Range (Noisy): {amplitude_range_noisy:.2f}\n"\
    #           f"Mean (Clean): {mean_clean:.2f}\n"\
    #           f"Mean (Noisy): {mean_noisy:.2f}"
    
    # # 그래프 안에 텍스트로 추가 정보 표시
    # plt.gcf().text(0.15, 0.85, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # plt.show()
    # 이미지 파일 저장
    filename = f"ECG_Comparison_{signal_name}_Sample_{sample_index}_Noise_{noise_index}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # 창을 닫아 메모리 해제

    print(f"Saved: {filepath}")

# 예시 데이터 및 시각화 호출
def visualize_multiple_beats(X_train, y_train, noise_indices_train, num_samples=5):
    """
    여러 개의 신호에 대해 원래 신호와 노이즈가 추가된 신호를 비교 시각화합니다.
    
    Parameters:
    X_train: 노이즈가 추가된 신호 (학습용).
    y_train: 노이즈 없는 원래 신호 (학습용).
    noise_indices_train: 각 신호에 대해 사용된 노이즈 조합 인덱스 리스트.
    num_samples: 시각화할 샘플 개수.
    """
    # 노이즈가 추가된 신호에 맞춰 샘플 개수 조정 (노이즈 인덱스에 맞게 제한)
    max_index = len(noise_indices_train)
    
    # 랜덤 샘플 선택 (노이즈가 추가된 비트만 선택)
    sample_indices = np.random.choice(max_index, num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):  # 선택된 샘플에 대해 시각화
        ecg_clean = y_train[idx].flatten()  # 원래 신호
        ecg_noisy = X_train[idx].flatten()  # 노이즈가 추가된 신호
        noise_idx = noise_indices_train[idx]  # 해당 신호에 추가된 노이즈 인덱스
        
        # 시각화 함수 호출 (샘플 인덱스 추가)
        ecg_view_noisy_comparison(ecg_clean, ecg_noisy, noise_index=noise_idx, signal_name="SampleECG", sample_index=idx, save_dir='visualizations/ecg_view_noisy_comparison')


# 시간 도메인에서 신호 시각화
def plot_time_domain(signal, title='Signal in Time Domain', save_dir='visualizations', filename='time_domain.png'):
    ensure_directory(save_dir)
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Signal')
    plt.title(title)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 파일 저장
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

# 주파수 도메인에서 신호 시각화 (FFT)
def plot_frequency_domain(signal, fs, title='Signal in Frequency Domain', save_dir='visualizations', filename='frequency_domain.png'):
    ensure_directory(save_dir)
    N = len(signal)
    T = 1.0 / fs
    yf = fft(signal)
    xf = np.fft.fftfreq(N, T)[:N//2]
    
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # 파일 저장
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

# STFT 시각화 함수
def plot_stft(signal, fs, title='STFT Magnitude', save_dir='visualizations', filename='stft.png', freq_min=0.01, freq_max=50):
    ensure_directory(save_dir)
    f, t, Zxx = stft(signal, fs=fs, nperseg=128)
    f_limit_idx = np.where((f >= freq_min) & (f <= freq_max))
    f_limited = f[f_limit_idx]
    Zxx_limited = np.abs(Zxx[f_limit_idx])
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f_limited, Zxx_limited, shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')

    # 파일 저장
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

# Spectrogram 시각화 함수
def plot_spectrogram(signal, fs, title='Spectrogram', save_dir='visualizations', filename='spectrogram.png', freq_min=0.01, freq_max=50):
    ensure_directory(save_dir)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=128)
    f_limit_idx = np.where((f >= freq_min) & (f <= freq_max))
    f_limited = f[f_limit_idx]
    Sxx_limited = Sxx[f_limit_idx]
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f_limited, 10 * np.log10(Sxx_limited), shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency (dB/Hz)')

    # 파일 저장
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def visualize_signals(y_train, X_train, fs, num_samples=3, signal_length=512, save_dir='visualizations/plot_signals'):
    """
    여러 신호에 대해 다양한 도메인에서 시각화하고, 이미지를 파일로 저장합니다.
    
    Parameters:
    y_train: 클린 신호 데이터셋
    X_train: 노이즈가 추가된 신호 데이터셋
    fs: 샘플링 주파수
    num_samples: 시각화할 샘플 개수 (기본값 3)
    signal_length: 신호 길이 (기본값 512)
    save_dir: 이미지가 저장될 디렉토리
    """
    ensure_directory(save_dir)  # 디렉토리 생성

    # 가능한 샘플 인덱스에서 무작위로 샘플 선택
    max_index = min(len(y_train), len(X_train))
    sample_indices = np.random.choice(max_index, num_samples, replace=False)  # 무작위로 샘플 인덱스 선택
    
    # 선택된 샘플에 대해 시각화
    for i, idx in enumerate(sample_indices):
        clean_signal = y_train[idx][:signal_length].flatten()  # 클린 신호
        noisy_signal = X_train[idx][:signal_length].flatten()  # 노이즈가 추가된 신호
        
        print(f"Visualizing Sample {idx}:")  # 실제 선택된 sample 번호 출력
        
        # 시간 도메인에서 신호 비교
        plot_time_domain(
            clean_signal, 
            title=f'Time Domain of Clean Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'time_domain_clean_{idx}.png'
        )
        plot_time_domain(
            noisy_signal, 
            title=f'Time Domain of Noisy Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'time_domain_noisy_{idx}.png'
        )
        
        # 주파수 도메인에서 신호 비교 (FFT)
        plot_frequency_domain(
            clean_signal, 
            fs, 
            title=f'Frequency Domain of Clean Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'frequency_domain_clean_{idx}.png'
        )
        plot_frequency_domain(
            noisy_signal, 
            fs, 
            title=f'Frequency Domain of Noisy Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'frequency_domain_noisy_{idx}.png'
        )
        
        # STFT 시각화
        plot_stft(
            clean_signal, 
            fs, 
            title=f'STFT of Clean Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'stft_clean_{idx}.png'
        )
        plot_stft(
            noisy_signal, 
            fs, 
            title=f'STFT of Noisy Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'stft_noisy_{idx}.png'
        )
        
        # Spectrogram 시각화
        plot_spectrogram(
            clean_signal, 
            fs, 
            title=f'Spectrogram of Clean Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'spectrogram_clean_{idx}.png'
        )
        plot_spectrogram(
            noisy_signal, 
            fs, 
            title=f'Spectrogram of Noisy Signal (Sample {idx})', 
            save_dir=save_dir, 
            filename=f'spectrogram_noisy_{idx}.png'
        )
############################################################################################################
def generate_hboxplot(np_data, description, ylabel, log, save_dir, filename, set_x_axis_size=None):
    # Process the results and store in Pandas DataFrame
    ensure_directory(save_dir)  # 디렉토리 생성
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(15, 6))

    ax = sns.boxplot(data=pd_df, orient="h", width=0.4)  # 가로로 누운 boxplot

    if log:
        ax.set_xscale("log")

    if set_x_axis_size is not None:
        ax.set_xlim(set_x_axis_size)

    ax.set(ylabel='Models/Methods', xlabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    # Save plot to file
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def generate_violinplots(np_data, description, ylabel, log, save_dir, filename, set_x_axis_size=None):
    # Process the results and store in Pandas DataFrame
    ensure_directory(save_dir)  # 디렉토리 생성
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 6))
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=pd_df, palette="Set3", bw=.2, cut=1, linewidth=1, orient="h")  # 가로로 누운 violinplot

    if log:
        ax.set_xscale("log")

    if set_x_axis_size is not None:
        ax.set_xlim(set_x_axis_size)

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    # Save plot to file
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def generate_barplot(np_data, description, ylabel, log, save_dir, filename, set_x_axis_size=None):
    # Process the results and store in Pandas DataFrame
    ensure_directory(save_dir)  # 디렉토리 생성
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=pd_df, orient="h")  # 가로로 누운 barplot

    if log:
        ax.set_xscale("log")

    if set_x_axis_size is not None:
        ax.set_xlim(set_x_axis_size)

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    # Save plot to file
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def generate_boxplot(np_data, description, ylabel, log, save_dir, filename, set_x_axis_size=None):
    # Process the results and store in Pandas DataFrame
    ensure_directory(save_dir)  # 디렉토리 생성
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 6))
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=pd_df, orient="h")  # 가로로 누운 boxplot

    if log:
        ax.set_xscale("log")

    if set_x_axis_size is not None:
        ax.set_xlim(set_x_axis_size)

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    # Save plot to file
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

############################################################################################################
# Function to visualize original and noisy beats separately
def plot_ecg_comparison_separate(X_data, y_data, indices, title, num_beats=5, save_dir='visualizations'):
    """
    Visualizes the original and noisy ECG beats separately and saves the plot.
    
    Parameters:
    X_data: Noisy ECG beats (X_train or X_test)
    y_data: Original ECG beats (y_train or y_test)
    indices: Valid indices for the beats
    title: Title for the plot
    num_beats: Number of beats to visualize
    save_dir: Directory to save the visualization
    """
    ensure_directory(save_dir)  # 디렉토리 생성

    # Randomly select a subset of valid indices for visualization
    if len(indices) < num_beats:
        num_beats = len(indices)  # Ensure we don't sample more than available valid beats
    selected_indices = random.sample(indices, num_beats)

    # Create a figure
    plt.figure(figsize=(15, num_beats * 4))

    for i, idx in enumerate(selected_indices):
        # Original beat plot
        plt.subplot(num_beats, 2, 2*i + 1)
        plt.plot(y_data[idx], label="Original Beat", color="blue")
        plt.title(f"{title} - Original Beat {idx}")
        plt.legend()

        # Noisy beat plot
        plt.subplot(num_beats, 2, 2*i + 2)
        plt.plot(X_data[idx], label="Noisy Beat", color="orange", linestyle="--")
        plt.title(f"{title} - Noisy Beat {idx}")
        plt.legend()

    plt.tight_layout()

    # Save the plot to the file
    filename = f"{title.replace(' ', '_')}_comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Saved: {filepath}")

def ecg_view(ecg, ecg_blw, ecg_dl, ecg_f, noise_index=None, signal_name=None, beat_no=None, save_dir='view'):
    """
    ECG 신호와 필터링된 신호들을 시각화하고, 노이즈 조합 정보를 제목에 표시하며 저장합니다.
    
    Parameters:
    ecg: 원본 ECG 신호
    ecg_blw: 노이즈가 포함된 ECG 신호 (Baseline Wander 포함)
    ecg_dl: DL로 필터링된 ECG 신호
    ecg_f: IIR 필터로 필터링된 ECG 신호
    noise_index: 노이즈 인덱스 (Optional)
    signal_name: 시각화할 신호의 이름 (Optional)
    beat_no: 시각화할 비트 번호 (Optional)
    save_dir: 시각화한 이미지를 저장할 디렉토리 (Optional)
    """
    ensure_directory(save_dir)  # 디렉토리 생성

    # 노이즈 인덱스에 해당하는 노이즈 조합 정보 가져오기
    noise_description = NOISE_COMBINATIONS.get(noise_index, "Unknown Noise Combination")
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 신호 플롯
    plt.plot(ecg_blw, 'k--', label=f'{noise_description}', alpha=0.7)
    plt.plot(ecg, 'g-', label='ECG Original', lw=2)
    plt.plot(ecg_dl, 'b-.', label='ECG DL Filtered', lw=2)
    plt.plot(ecg_f, 'r-', label='ECG IIR Filtered', lw=1.5)
    
    plt.grid(True)
    plt.ylabel('Amplitude (au)')
    plt.xlabel('Samples')
    
    # 범례 추가
    leg = ax.legend()

    # 신호 및 비트 번호에 대한 제목 설정 (노이즈 인덱스 포함)
    if signal_name is not None and beat_no is not None:
        plt.title(f'Signal {signal_name}, Beat {beat_no}, Noise: {noise_description}')
    else:
        plt.title(f'ECG Signal Comparison with Noise: {noise_description}')

    # 이미지 파일 저장
    filename = f"ECG_Comparison_Beat_{beat_no}_Noise_{noise_index}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # 창을 닫아 메모리 해제
    
    print(f"Saved: {filepath}")

def ecg_view_3d(ecg, ecg_blw, ecg_dl, ecg_f, noise_index=None, signal_name=None, beat_no=None, save_dir='view'):
    """
    ECG 신호와 필터링된 신호들을 3D로 시각화하고, 노이즈 조합 정보를 제목에 표시하며 이미지를 저장합니다.
    
    Parameters:
    ecg: 원본 ECG 신호
    ecg_blw: 노이즈가 포함된 ECG 신호 (Baseline Wander 포함)
    ecg_dl: DL로 필터링된 ECG 신호
    ecg_f: IIR 필터로 필터링된 ECG 신호
    noise_index: 노이즈 인덱스 (Optional)
    signal_name: 시각화할 신호의 이름 (Optional)
    beat_no: 시각화할 비트 번호 (Optional)
    save_dir: 시각화한 이미지를 저장할 디렉토리 (Optional)
    """
    ensure_directory(save_dir)  # 디렉토리 생성

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 시간 축을 x로 하고 각 신호를 y, z 축으로 배치
    t = np.arange(len(ecg))

    # z 축을 크게 구분하여 신호들을 배치 (zdir='y'에 따라 구분)
    ax.plot(t, ecg_blw, zs=3, zdir='y', label='Noisy ECG Signal', color='k', linestyle='--', lw=2, alpha=0.8)
    ax.plot(t, ecg, zs=2, zdir='y', label='Clean ECG Signal', color='g', lw=2, alpha=0.8)
    ax.plot(t, ecg_dl, zs=1, zdir='y', label='ECG DL Filtered', color='b', lw=2, alpha=0.8)
    ax.plot(t, ecg_f, zs=0, zdir='y', label='ECG IIR Filtered', color='r', lw=2, alpha=0.8)

    # 3D 축 레이블 설정
    ax.set_xlabel('Samples', fontsize=14, labelpad=15)
    ax.set_ylabel('Signal Type', fontsize=14, labelpad=15)
    ax.set_zlabel('Amplitude', fontsize=14, labelpad=15)

    # 노이즈 인덱스에 해당하는 노이즈 조합 정보 가져오기
    noise_description = NOISE_COMBINATIONS.get(noise_index, "Unknown Noise Combination")
    
    # 신호 및 비트 번호에 대한 제목 설정 (노이즈 인덱스 포함)
    if signal_name is not None and beat_no is not None:
        plt.title(f'Signal {signal_name}, Beat {beat_no}, Noise: {noise_description}', fontsize=16, pad=20)
    else:
        plt.title(f'ECG Signal Comparison with Noise: {noise_description}', fontsize=16, pad=20)

    # 그리드 추가 및 투명도 조절
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 범례 표시
    ax.legend(fontsize=12)
    
    # 시각적으로 더 부드럽게 보이도록 틱 설정
    ax.tick_params(axis='both', which='major', labelsize=12, pad=10)

    # 이미지 파일 저장
    filename = f"ECG_3D_Comparison_Beat_{beat_no}_Noise_{noise_index}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # 창을 닫아 메모리 해제
    
    print(f"Saved: {filepath}")

def generate_table(metrics, metric_values, Exp_names):
    # Print tabular results in the console, in a pretty way
    tb = PrettyTable()
    ind = 0

    for exp_name in Exp_names:

        tb.field_names = ['Method/Model'] + metrics

        tb_row = []
        tb_row.append(exp_name)

        for metric in metric_values:   # metric_values[metric][model][beat]
            m_mean = np.mean(metric[ind])
            m_std = np.std(metric[ind])
            tb_row.append('{:.3f}'.format(m_mean) + ' (' + '{:.3f}'.format(m_std) + ')')

        tb.add_row(tb_row)
        ind += 1

    print(tb)


def generate_table_time(column_names, all_values, Exp_names, gpu=True):
    # Print tabular results in the console, in a pretty way

    # The FIR and IIR are the last on all_values
    # We need circular shift them to the right
    all_values[0] = all_values[0][-2::] + all_values[0][0:-2]
    all_values[1] = all_values[1][-2::] + all_values[1][0:-2]

    tb = PrettyTable()
    ind = 0

    if gpu:
        device = 'GPU'
    else:
        device = 'CPU'

    for exp_name in Exp_names:
        tb.field_names = ['Method/Model'] + [column_names[0] + '(' + device + ') h:m:s:ms'] + [
            column_names[1] + '(' + device + ') h:m:s:ms']

        tb_row = []
        tb_row.append(exp_name)
        tb_row.append(all_values[0][ind])
        tb_row.append(all_values[1][ind])

        tb.add_row(tb_row)

        ind += 1

    print(tb)

    if gpu:
        print('* For FIR and IIR Filters is CPU since scipy filters are CPU based implementations')
