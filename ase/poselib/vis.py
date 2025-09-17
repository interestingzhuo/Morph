
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch




data = joblib.load("/mnt/kaiwu-group-z3/albertzli/results/pipeline/5_1.pkl")
skeleton_tree = np.load("/mnt/kaiwu-group-z3/albertzli/code/ASE/ase/data/assets/skeleton_tree_smpl.npy", allow_pickle=True).item()
trans = data['final_trans']
poses = data['final_poses']


axis_R =  Rot.from_quat(np.array([0, 0.7071, 0.7071, 0])).as_matrix()
num_frames, num_bones = poses.shape
num_bones = int(num_bones/3)

poses = Rot.from_rotvec(poses.reshape(-1,3)).as_matrix()
poses = poses.reshape(num_frames, num_bones, 3, 3)
# axis transformation
for pose in poses:#for each frame
    pose[0, :] = np.dot(axis_R, pose[0, :])

poses = poses.reshape(-1, 3, 3)
poses = Rot.from_matrix(poses).as_quat().reshape(num_frames, num_bones, 4)
trans = np.dot(axis_R, trans.T).T/100

# 坐标系变换 
poses = torch.tensor(poses[:,:,[2, 0, 1, 3]])




bone_names_src = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand", 
]
bone_names_dst = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Foot", "R_Hip",
    "R_Knee", "R_Ankle", "R_Foot", "Spine1", "Spine2", "Spine3",
    "Neck", "Head", "L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist", 
    "L_Hand", "R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
idx = []
for name in bone_names_dst:
    idx += [bone_names_src.index(name)]
import pdb
pdb.set_trace()
poses = poses[:,idx,:]
trans = torch.tensor(trans[:,[2, 0, 1]])

motion = SkeletonMotion.from_rotation_and_root_translation(
                                poses,
                                trans, 
                                skeleton_tree, 
                                is_local=False,
                                fps = 30
                            )


plot_skeleton_motion_interactive(motion)