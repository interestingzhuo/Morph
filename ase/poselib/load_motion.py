# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive
import joblib
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import glob
import torch 
skeleton_tree = np.load("ase/data/assets/skeleton_tree_smpl.npy", allow_pickle=True).item()

def load_motion(final_trans, final_poses, dst_path):

    trans = final_trans
    poses = final_poses

    num_frames, num_bones = poses.shape
    num_bones = int(num_bones/3)
    poses = poses.reshape(-1, 3).copy()
    poses = Rot.from_rotvec(poses).as_quat().reshape(num_frames, num_bones, 4)
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
    poses = poses[:, idx, :]

    poses = torch.tensor(poses[:, :, [2, 0, 1, 3]])
    trans = torch.tensor(trans[:, [2, 0, 1]])




    motion = SkeletonMotion.from_rotation_and_root_translation(
                                    poses,
                                    trans, 
                                    skeleton_tree, 
                                    is_local=True,
                                    fps = 30
                                )
  
    # 首帧倾斜约束
    local_rotation = motion.local_rotation
    root_translation = motion.root_translation
    tar_global_pos = motion.global_translation



    # 站立帧
    FOOT_INDEX = [4, 8]
    ROOT_INDEX = 0
    threshold = 10*np.pi/180
    foot_keyp3d = tar_global_pos[0,FOOT_INDEX].numpy()
    root_keyp3d = tar_global_pos[0,ROOT_INDEX].numpy()
    center = np.mean(foot_keyp3d, axis=0)

    vector = root_keyp3d - center
    vector = vector / np.linalg.norm(vector)
    theta = np.arccos(np.dot(vector, np.array([0, 0, 1]))) 

    if theta > threshold:
        # theta -= threshold
        axis = np.cross(np.array([0, 0, 1]), vector)
        offset_r = Rot.from_rotvec(-theta * axis)
        for i in range(local_rotation.shape[0]):
            rotation = Rot.from_quat(local_rotation[i,0,:])

            local_rotation[i,0,:] = torch.from_numpy((offset_r * rotation).as_quat())



    motion = SkeletonMotion.from_rotation_and_root_translation(
        local_rotation,
        root_translation,
        skeleton_tree,
        is_local=True,
        fps=30
    )

    # 首帧地面约束
    local_rotation = motion.local_rotation
    root_translation = motion.root_translation
    tar_global_pos = motion.global_translation

    min_h = torch.min(tar_global_pos[...,  2])
    root_translation[:, 2] += -min_h

    # add offset
    # root_translation[:, 2] += 0.01

    motion = SkeletonMotion.from_rotation_and_root_translation(
                                    local_rotation,
                                    root_translation, 
                                    skeleton_tree, 
                                    is_local=True,
                                    fps = 30
                                )


    motion.to_file(dst_path)


import glob
import joblib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(file):
    data = joblib.load(file)
    dst_path = file.replace(".pkl", ".npy").replace("repeat_time_0","smpl_data_generated_npy")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    final_trans = data['smpl_trans']
    final_poses = data['smpl_poses']
    load_motion(final_trans, final_poses, dst_path)

if __name__ == "__main__":
    files = glob.glob("*.pkl")
    
    # 使用多进程池
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_file, files), total=len(files)))


