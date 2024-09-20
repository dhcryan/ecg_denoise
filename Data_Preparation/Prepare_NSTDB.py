import wfdb
import numpy as np
import pickle

def prepare(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/'):
    """
    Prepares noise data from the MIT-BIH Noise Stress Test Database (NSTDB) with noise combinations as per the TCDAE study.
    Noise combinations include Baseline Wander (BW), Electrode Motion (EM), and Muscle Artifact (MA) noises.

    Parameters:
    NSTDBPath: Path to the NSTDB dataset.
    snr_range: Tuple indicating the SNR range (default: (-6, 18)).
    """
    # Load NSTDB (Baseline Wander, Electrode Motion, and Muscle Artifact noises)
    bw_signals, _ = wfdb.rdsamp(NSTDBPath + 'bw')
    em_signals, _ = wfdb.rdsamp(NSTDBPath + 'em')
    ma_signals, _ = wfdb.rdsamp(NSTDBPath + 'ma')

    # print(f"[INFO] Loaded baseline wander (BW) signals: shape {bw_signals.shape}")
    # print(f"[INFO] Loaded electrode motion (EM) signals: shape {em_signals.shape}")
    # print(f"[INFO] Loaded muscle artifact (MA) signals: shape {ma_signals.shape}")
# [INFO] Loaded baseline wander (BW) signals: shape (650000, 2)
# [INFO] Loaded electrode motion (EM) signals: shape (650000, 2)
# [INFO] Loaded muscle artifact (MA) signals: shape (650000, 2)
    # Initialize combined noise storage for pickle output
    combined_noise = []

    # Loop through both training and test noise channels
    for channel in range(2):
        # Get noise data for the current channel (both BW, EM, MA)
        bw_noise = bw_signals[:, channel]
        em_noise = em_signals[:, channel]
        ma_noise = ma_signals[:, channel]
        
        print(f"[DEBUG] Processing channel {channel+1}: BW noise shape: {bw_noise.shape}, EM noise shape: {em_noise.shape}, MA noise shape: {ma_noise.shape}")
        # 650000, 650000, 650000
        # Noise combination types based on Fig. 2 in the TCDAE paper (8 combinations of BW, EM, and MA)
        noise_combinations = [
            (0, 0, 0),  # Type 1: No noise
            (1, 0, 0),  # Type 2: BW only
            (0, 1, 0),  # Type 3: EM only
            (0, 0, 1),  # Type 4: MA only
            (1, 1, 0),  # Type 5: BW + EM
            (1, 0, 1),  # Type 6: BW + MA
            (0, 1, 1),  # Type 7: EM + MA
            (1, 1, 1)   # Type 8: BW + EM + MA (all noises)
        ]

        # Create a noise matrix to store all possible combinations of noises
        noise_matrix = np.zeros((bw_noise.shape[0], len(noise_combinations)))
        print(f"[DEBUG] Initialized noise matrix with shape {noise_matrix.shape}")
        # 650000, 8
        # Iterate over all noise combinations to fill the noise matrix
        for idx, (bw_flag, em_flag, ma_flag) in enumerate(noise_combinations):
            combined_noise_for_type = (
                bw_flag * bw_noise +
                em_flag * em_noise +
                ma_flag * ma_noise
            )
            noise_matrix[:, idx] = combined_noise_for_type
            # print(f"[DEBUG] Filled noise matrix for combination type {idx+1}, shape: {noise_matrix[:, idx].shape}")
            #650000
        # Append the noise matrix for this channel to the combined noise list
        combined_noise.append(noise_matrix)
        # print(f"[INFO] Combined noise for channel {channel+1} has shape {noise_matrix.shape}")
        # 650000,8
    # Save combined noise (BW, EM, MA, and their combinations) as a pickle file
    with open('data/CombinedNoise.pkl', 'wb') as output:  # Overwrites any existing file
        pickle.dump(combined_noise, output)
    # print(f'combined noise shape : {np.array(combined_noise).shape}')
    # combined noise shape : (2, 650000, 8)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) with combined noise saved as pickle')
# 채널마다 650000개의 데이터가 있고, 8가지의 노이즈 조합이 있음을 확인할 수 있습니다.
