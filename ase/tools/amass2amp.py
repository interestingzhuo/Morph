import glob
from tqdm import tqdm
import os
# amass_splits = {
#     'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', "bandai_1_valid", "apid_valid", "AIST_valid", "bandai_2_valid", "choreomaster_valid"],
#     'test': ['Transitions_mocap', 'SSM_synced', "bandai_1_test", "apid_test", "AIST_test", "bandai_2_test", "choreomaster_test"],
#     'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes", "DFaust_67", "bandai_1_train", "apid_train", "AIST_train", "bandai_2_train", "choreomaster_train"]  # Adding ACCAD
# }
amass_splits = {
    'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes", "DFaust_67"]  # Adding ACCAD
}

amass_split_dict = {}
for k, v in amass_splits.items():
    for d in v:
        amass_split_dict[d] = k


train_data = []
test_data = []
valid_data = []

import glob
from tqdm import tqdm
import joblib

missing = []
for name in tqdm(glob.glob("/mnt/aigen_data/albertzli/code/ASE/datasets/amass/*/*/*.npy")):

    start_name = name.split('/')[-3]
    if start_name not in amass_split_dict:
        if start_name not in missing:
            missing += [start_name]
        continue
    if amass_split_dict[start_name] == 'train':
        train_data += [name]
    elif amass_split_dict[start_name] == 'test':
        test_data += [name]
    elif amass_split_dict[start_name] == 'valid':
        valid_data += [name]
    
print(missing)

joblib.dump(train_data, f"../datasets/sample_data/amass_train.pkl")
joblib.dump(test_data, f"../datasets/sample_data/amass_test.pkl")
joblib.dump(valid_data, f"../datasets/sample_data/amass_valid.pkl")