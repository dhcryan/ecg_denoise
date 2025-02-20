# # import wfdb
# # import numpy as np
# # import pickle

# # def prepare(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/'):
# #     """
# #     Prepares noise data from the MIT-BIH Noise Stress Test Database (NSTDB) with noise combinations as per the TCDAE study.
# #     Noise combinations include Baseline Wander (BW), Electrode Motion (EM), and Muscle Artifact (MA) noises.

# #     Parameters:
# #     NSTDBPath: Path to the NSTDB dataset.
# #     snr_range: Tuple indicating the SNR range (default: (-6, 18)).
# #     """
# #     # Load NSTDB (Baseline Wander, Electrode Motion, and Muscle Artifact noises)
# #     bw_signals, _ = wfdb.rdsamp(NSTDBPath + 'bw')
# #     em_signals, _ = wfdb.rdsamp(NSTDBPath + 'em')
# #     ma_signals, _ = wfdb.rdsamp(NSTDBPath + 'ma')

# #     # print(f"[INFO] Loaded baseline wander (BW) signals: shape {bw_signals.shape}")
# #     # print(f"[INFO] Loaded electrode motion (EM) signals: shape {em_signals.shape}")
# #     # print(f"[INFO] Loaded muscle artifact (MA) signals: shape {ma_signals.shape}")
# # # [INFO] Loaded baseline wander (BW) signals: shape (650000, 2)
# # # [INFO] Loaded electrode motion (EM) signals: shape (650000, 2)
# # # [INFO] Loaded muscle artifact (MA) signals: shape (650000, 2)
# #     # Initialize combined noise storage for pickle output
# #     combined_noise = []

# #     # Loop through both training and test noise channels
# #     for channel in range(2):
# #         # Get noise data for the current channel (both BW, EM, MA)
# #         bw_noise = bw_signals[:, channel]
# #         em_noise = em_signals[:, channel]
# #         ma_noise = ma_signals[:, channel]
        
# #         print(f"[DEBUG] Processing channel {channel+1}: BW noise shape: {bw_noise.shape}, EM noise shape: {em_noise.shape}, MA noise shape: {ma_noise.shape}")
# #         # 650000, 650000, 650000
# #         # Noise combination types based on Fig. 2 in the TCDAE paper (8 combinations of BW, EM, and MA)
# #         noise_combinations = [
# #             (0, 0, 0),  # Type 1: No noise
# #             (1, 0, 0),  # Type 2: BW only
# #             (0, 1, 0),  # Type 3: EM only
# #             (0, 0, 1),  # Type 4: MA only
# #             (1, 1, 0),  # Type 5: BW + EM
# #             (1, 0, 1),  # Type 6: BW + MA
# #             (0, 1, 1),  # Type 7: EM + MA
# #             (1, 1, 1)   # Type 8: BW + EM + MA (all noises)
# #         ]

# #         # Create a noise matrix to store all possible combinations of noises
# #         noise_matrix = np.zeros((bw_noise.shape[0], len(noise_combinations)))
# #         print(f"[DEBUG] Initialized noise matrix with shape {noise_matrix.shape}")
# #         # 650000, 8
# #         # Iterate over all noise combinations to fill the noise matrix
# #         for idx, (bw_flag, em_flag, ma_flag) in enumerate(noise_combinations):
# #             combined_noise_for_type = (
# #                 bw_flag * bw_noise +
# #                 em_flag * em_noise +
# #                 ma_flag * ma_noise
# #             )
# #             noise_matrix[:, idx] = combined_noise_for_type
# #             # print(f"[DEBUG] Filled noise matrix for combination type {idx+1}, shape: {noise_matrix[:, idx].shape}")
# #             #650000
# #         # Append the noise matrix for this channel to the combined noise list
# #         combined_noise.append(noise_matrix)
# #         # print(f"[INFO] Combined noise for channel {channel+1} has shape {noise_matrix.shape}")
# #         # 650000,8
# #     # Save combined noise (BW, EM, MA, and their combinations) as a pickle file
# #     with open('data/CombinedNoise.pkl', 'wb') as output:  # Overwrites any existing file
# #         pickle.dump(combined_noise, output)
# #     # print(f'combined noise shape : {np.array(combined_noise).shape}')
# #     # combined noise shape : (2, 650000, 8)
# #     print('=========================================================')
# #     print('MIT BIH data noise stress test database (NSTDB) with combined noise saved as pickle')
# # # 채널마다 650000개의 데이터가 있고, 8가지의 노이즈 조합이 있음을 확인할 수 있습니다.
import wfdb
import numpy as np
import pickle
# CENSD 구현
def prepare_combined_noise_with_bw(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/'):
    """
    Prepares combined noise data from the MIT-BIH Noise Stress Test Database (NSTDB)
    with Baseline Wander (BW) noise always included. Electrode Motion (EM) and 
    Muscle Artifact (MA) noises are added randomly.
    Expands the final noise matrix to shape (2, 650000, 1).
    
    Parameters:
    NSTDBPath: Path to the NSTDB dataset.
    """
    # Load NSTDB noise signals (Baseline Wander, Electrode Motion, Muscle Artifact)
    bw_signals, _ = wfdb.rdsamp(NSTDBPath + 'bw')
    em_signals, _ = wfdb.rdsamp(NSTDBPath + 'em')
    ma_signals, _ = wfdb.rdsamp(NSTDBPath + 'ma')
    
    # Initialize storage for combined noise across channels
    combined_noise = []

    # Loop through both training and test noise channels
    for channel in range(2):
        # Extract channel-specific noise data
        bw_noise = bw_signals[:, channel]
        em_noise = em_signals[:, channel]
        ma_noise = ma_signals[:, channel]
        
        print(f"[DEBUG] Processing channel {channel+1}: BW noise shape: {bw_noise.shape}, EM noise shape: {ma_noise.shape}, MA noise shape: {ma_noise.shape}")
        
        # Initialize combined noise array for this channel
        combined_channel_noise = np.zeros(bw_noise.shape)

        # Divide the noise data into chunks (simulating random time intervals)
        chunk_size = 10000  # Random time chunk size
        num_chunks = len(bw_noise) // chunk_size
        seed = 1234  # 원하는 값을 설정
        np.random.seed(seed=seed)
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size

            # Include BW noise by default
            bw_chunk = bw_noise[start_idx:end_idx]

            # Randomly decide whether to include EM and MA noises
            em_flag = np.random.choice([0, 1])  # Include Electrode Motion or not
            ma_flag = np.random.choice([0, 1])  # Include Muscle Artifact or not
            
            # Generate noise for this chunk
            chunk_noise = (
                bw_chunk +  # BW noise is always included
                em_flag * em_noise[start_idx:end_idx] +
                ma_flag * ma_noise[start_idx:end_idx]
            )

            # Apply the chunk noise to the combined noise array
            combined_channel_noise[start_idx:end_idx] += chunk_noise
        
        # Expand dimensions to make it (650000, 1) for this channel
        combined_channel_noise = np.expand_dims(combined_channel_noise, axis=1)

        # Append the combined noise for this channel
        combined_noise.append(combined_channel_noise)

    # Stack the noises for all channels to shape (2, 650000, 1)
    combined_noise = np.stack(combined_noise, axis=0)

    # Save combined noise data as a pickle file
    with open('data/CombinedNoise.pkl', 'wb') as output:
        pickle.dump(combined_noise, output)
    
    print('=========================================================')
    print(f'Final combined noise shape: {combined_noise.shape}')
    print('Realistic NSTDB noise data with Baseline Wander always included saved as pickle')

