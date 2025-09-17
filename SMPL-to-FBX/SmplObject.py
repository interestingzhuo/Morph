import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np


class SmplObjects(object):
    joints = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]

    def __init__(self, read_path):
        self.files = {}
        files = open("datasets/physics_optimized_filename_to_test_smpl_pkl_filename.txt").readlines()
        files = [files[i].strip().split("####")[-1] for i in range(2000)]
        #paths = sorted(glob.glob(f"{read_path}/*.pkl"))
        import pdb;pdb.set_trace()
        paths = [f"{read_path}/{file}.pkl" for file in files]
        for path in paths:
            filename = path.split("/")[-1]
            try:
                # with open(path, "rb") as fp:
                #     data = pickle.load(fp)
                import joblib
                data = joblib.load(path)
            except:
                # import pdb;pdb.set_trace()
                continue
            self.files[filename] = {
                "smpl_poses": data["smpl_poses"],
                "smpl_trans": data["smpl_trans"],
            }
        self.keys = [key for key in self.files.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]
