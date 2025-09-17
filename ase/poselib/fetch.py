import joblib
import numpy as np
motion_files = joblib.load("/mnt/aigen_data/albertzli/code/ASE/datasets/sample_data/amass_train.pkl")
error_keys = []
from tqdm import tqdm
for name in  tqdm(motion_files):
    data = np.load(name, allow_pickle=True).item()
    for key in data:
        if type(data[key]) == dict and'arr' in data[key] and True in np.isnan(data[key]['arr']):
            error_keys += [name.split('/')[-1].split(".")[0]]
print(error_keys)             