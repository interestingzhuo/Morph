
import os

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive
import joblib
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import glob
import torch 

def export_motion(motion, dst_path):

    # import pdb;pdb.set_trace()
    poses = motion.local_rotation
    trans = motion.root_translation
    
    bone_names_dst = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]
    bone_names_src = [
        "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Foot", "R_Hip",
        "R_Knee", "R_Ankle", "R_Foot", "Spine1", "Spine2", "Spine3",
        "Neck", "Head", "L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist",
        "L_Hand", "R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
    idx = []
    for name in bone_names_dst:
        idx += [bone_names_src.index(name)]
    poses = poses[:, idx, :]
    
    poses = poses[:, :, [1, 2, 0, 3]]
    trans = trans[:, [1, 2, 0]]
    num_frames, num_bones, _ = poses.shape
    poses = Rot.from_quat(poses.reshape(-1, 4)).as_rotvec().reshape(num_frames, 72)
   


    data = {
            'smpl_trans': trans,
            'smpl_poses': poses
    }   

    joblib.dump(data, dst_path)


import glob
import joblib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(path):
    try:
        dst_path = path.replace(".npy", ".pkl")
        motion = SkeletonMotion.from_file(path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        export_motion(motion, dst_path)
    except:
        return 


if __name__ == "__main__":
    files = glob.glob("datasets/momask_test_overfitting_repeat_time_0/*.npy")
    # 使用多进程池
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_file, files), total=len(files)))







