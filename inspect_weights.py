
import h5py
import os

file_path = "0221_FIXED/Dual_FreqDAE_weights.best.weights.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape}")
    # else:
    #     print(name)

if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}:")
        f.visititems(print_structure)
else:
    print(f"File not found: {file_path}")
