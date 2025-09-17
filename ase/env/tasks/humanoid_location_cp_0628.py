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

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidLocation(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        num_joints = 15
        self._target_pos = torch.zeros([self.num_envs, num_joints, 3], device=self.device, dtype=torch.float)
        self._target_rot = torch.zeros([self.num_envs, num_joints, 4], device=self.device, dtype=torch.float)
        self._target_vel = torch.zeros([self.num_envs, num_joints, 3], device=self.device, dtype=torch.float)
        self._target_angle_vel = torch.zeros([self.num_envs, num_joints, 3], device=self.device, dtype=torch.float)

        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 474
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        

        self._prev_target_rot = self._target_rot.clone()
        self._prev_target_pos = self._target_pos.clone()
        self._prev_target_vel = self._target_vel.clone()
        self._prev_target_angle_vel= self._target_angle_vel.clone()

        return

    def post_physics_step(self):
        self._update_task()
        self._cur_ref_motion_times += self.dt
        super().post_physics_step()
        return
        
    def _update_marker(self):
        self._marker_pos[..., 0:2] = self._target_pos[:,0,0:2]
        self._marker_pos[..., 2] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + 1

        return

    def _update_task(self):
        #是否超过了序列长度 
        cur_ref_motion_times =  self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths.to(self.device)[self._reset_ref_motion_ids]

        next_task_mask =  cur_ref_motion_times < ref_motion_lengths
        next_env_ids = next_task_mask.nonzero(as_tuple=False).flatten()
       
        if len(next_env_ids) > 0:
            self._next_step(next_env_ids)
        return

    def _reset_task(self, env_ids):
        # 更新目标点,停在原地
        
        
        motion_ids = self._reset_ref_motion_ids[env_ids]
        motion_times = self._cur_ref_motion_times[env_ids] 
        if len(motion_times.nonzero(as_tuple=False).flatten()) == 0:
            pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times + self.dt)
                
            self._target_pos[env_ids] = pos
            self._target_root_rot[env_ids] = root_rot
           
        
        return 

    def _next_step(self, env_ids):
        # 更新目标点
        
        motion_ids = self._reset_ref_motion_ids[env_ids]
        motion_times = self._cur_ref_motion_times[env_ids] + self.dt     # next frame

        pos, rot, dof_pos, vel, ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        self._target_pos[env_ids] = pos
        self._target_rot[env_ids] = rot
        self._target_vel[env_ids] = vel
        self._target_angle_vel[env_ids] = ang_vel
        


    def _compute_task_obs(self, env_ids=None):
        
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            target_pos = self._target_pos
            target_rot = self._target_rot
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            target_pos = self._target_pos[env_ids]
            target_rot = self._target_rot[env_ids]

        # import pdb
        # pdb.set_trace()
        obs = compute_imitation_observations(body_pos, body_rot, target_pos, target_rot,
                                            body_vel, body_ang_vel, self._root_height_obs, self._local_root_obs)
      
        return obs

    def _compute_reward(self, actions):

        pos = self._rigid_body_pos
        rot = self._rigid_body_rot
        vel = self._rigid_body_vel
        angle_vel = self._rigid_body_ang_vel
        
        
        self.rew_buf[:] = compute_location_reward(pos, self._prev_target_pos, rot, self._prev_target_rot, vel, self._prev_target_vel, angle_vel, self._prev_target_angle_vel)

        

        reset_mask = self.reset_buf == 1
        if torch.any(reset_mask):
            self.rew_buf[reset_mask] = 0
        return

    def _draw_task(self):
        self._update_marker()
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._marker_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################



@torch.jit.script
def compute_location_observations(root_states, _target_root_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos3d = torch.cat([_target_root_pos, torch.zeros_like(_target_root_pos[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = quat_rotate(heading_rot, tar_pos3d - root_pos)
    local_tar_pos = local_tar_pos[..., 0:2]

    obs = local_tar_pos
    return obs


@torch.jit.script
def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v

@torch.jit.script
def compute_imitation_observations(body_pos, body_rot, target_pos, target_rot, body_vel, body_ang_vel, root_height_obs, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    """target"""
    # target root height    [N, 1]
    target_root_pos = target_pos[:, 0, :]
    target_root_rot = target_rot[:, 0, :]
    target_rel_root_h = root_h - target_root_pos[:, 2:3]
    # target root rotation  [N, 6]
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(target_root_rot)
    target_rel_root_rot = quat_mul(target_root_rot, quat_conjugate(root_rot))
    target_rel_root_rot_obs = torch_utils.quat_to_tan_norm(target_rel_root_rot)
    # target 2d pos [N, 2]
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]
    # target heading    [N, 2]
    target_rel_heading = target_heading - heading
    target_rel_heading_vec = heading_to_vec(target_rel_heading)

    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(target_rel_body_rot.shape[0], -1)


    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec, target_rel_body_pos, target_rel_body_rot_obs, target_pos.reshape(target_pos.shape[0], -1), target_rot.reshape(target_rot.shape[0], -1)), dim=-1)

    return obs


@torch.jit.script
def compute_location_reward(pos, _target_pos, rot, _target_rot, vel, _target_vel, avel, _target_avel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    pos_err_scale = 100
    rot_err_scale = 10
    vel_err_scale = 0.1
    avel_err_scale = 0.1
    
    pos_reward_w = 0.5
    rot_reward_w = 0.3
    vel_reward_w = 0.1
    avel_reward_w = 0.1
    

    pos_diff = _target_pos - pos
    pos_err = (pos_diff * pos_diff).mean(dim=-1).mean(dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    
    


    diff_rot = quat_mul(_target_rot, quat_conjugate(rot))
    diff_rot_angle = torch_utils.quat_to_angle_axis(diff_rot)[0]
    diff_rot_angle_dist = (diff_rot_angle ** 2).mean(dim=-1)
    _rot_reward = torch.exp(-rot_err_scale * diff_rot_angle_dist)
    

    vel_diff = _target_vel - vel
    vel_err = (vel_diff * vel_diff).mean(dim=-1).mean(dim=-1)
    _vel_reward = torch.exp(-vel_err_scale * vel_err)

    avel_diff = _target_avel - avel
    avel_err = (avel_diff * avel_diff).mean(dim=-1).mean(dim=-1)
    _avel_reward = torch.exp(-avel_err_scale * avel_err)


   


    reward = pos_reward_w * pos_reward + rot_reward_w * _rot_reward + vel_reward_w * _vel_reward + avel_reward_w * _avel_reward

    return reward