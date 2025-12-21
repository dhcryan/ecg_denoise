
import h5py
import shutil
import os


files_to_patch = [
    ("0221_FIXED/Transformer_DAE_weights.best.weights.h5", "0221_FIXED/Transformer_DAE_weights.best.keras3.h5"),
    ("0221_FIXED/Dual_FreqDAE_weights.best.weights.h5", "0221_FIXED/Dual_FreqDAE_weights.best.keras3.h5")
]

for src_path, dst_path in files_to_patch:
    if not os.path.exists(src_path):
        print(f"Source file not found: {src_path}")
        continue

    print(f"Copying {src_path} to {dst_path}...")
    shutil.copy(src_path, dst_path)

    print(f"Patching {dst_path} for Keras 3 compatibility...")
    with h5py.File(dst_path, 'r+') as f:
        if 'layers' in f:
            layers_grp = f['layers']
            for layer_name in layers_grp:
                if 'multi_head_attention' in layer_name:
                    mha_grp = layers_grp[layer_name]
                    
                    replacements = {
                        'key_dense': 'key',
                        'query_dense': 'query',
                        'value_dense': 'value',
                        'output_dense': 'attention_output'
                    }
                    
                    for old, new in replacements.items():
                        if old in mha_grp:
                            print(f"Renaming {layer_name}/{old} to {layer_name}/{new}")
                            mha_grp.move(old, new)


print("Patching complete.")
