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

from env.tasks.humanoid import Humanoid, dof_to_obs
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidImitation(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]
        self._termination_dists = cfg["env"]["terminationDistance"]


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
        self._target_dof_pos = torch.zeros([self.num_envs, 28], device=self.device, dtype=torch.float)
        self._target_dof_vel = torch.zeros([self.num_envs, 28], device=self.device, dtype=torch.float)
        self.body_pos_weights = torch.ones(num_joints, device=self.device)

       
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 530
        return obs_size

    def pre_physics_step(self, actions):
        actions[self.reset_buf == 1] = 0
        super().pre_physics_step(actions)
        
        self.extras["rot"] = self._rigid_body_rot.clone()
        self.extras["pos"] = self._rigid_body_pos.clone()
        self.extras["rot_gt"] = self._target_rot.clone()
        self.extras["pos_gt"] = self._target_pos.clone()
        self._prev_target_rot = self._target_rot.clone()
        self._prev_target_pos = self._target_pos.clone()
        self._prev_target_vel = self._target_vel.clone()
        self._prev_target_angle_vel= self._target_angle_vel.clone()

        self._prev_target_dof_pos = self._target_dof_pos.clone()
        self._prev_target_dof_vel = self._target_dof_vel.clone()



        return

    def post_physics_step(self):
        self._update_task()
        self._cur_ref_motion_times += self.dt
        super().post_physics_step()
        return
        

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        return


    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        return


    def _compute_reset(self):
       
        cur_ref_motion_times =  self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths[self._reset_ref_motion_ids]

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_body_ids,self._rigid_body_pos, self._prev_target_pos, 
                                                   self.max_episode_length,self._enable_early_termination, self._termination_dists, cur_ref_motion_times, ref_motion_lengths)
        return

    def _update_task(self):
        #是否超过了序列长度 
    
        cur_ref_motion_times =  self._cur_ref_motion_times + self.dt
        
        ref_motion_lengths = self._motion_lib._motion_lengths[self._reset_ref_motion_ids]
        
        next_task_mask =  cur_ref_motion_times < ref_motion_lengths
        next_env_ids = next_task_mask.nonzero(as_tuple=False).flatten()
       
        if len(next_env_ids) > 0:
            self._next_step(next_env_ids)
        return

    def _reset_task(self, env_ids):
        # 更新目标点,停在原地
        
        
        motion_ids = self._reset_ref_motion_ids[env_ids]
        motion_times = self._cur_ref_motion_times[env_ids] + self.dt
       
        pos, rot, dof_pos, vel, ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        self._target_pos[env_ids] = pos
        self._target_rot[env_ids] = rot
        self._target_vel[env_ids] = vel
        self._target_angle_vel[env_ids] = ang_vel
        self._target_dof_pos[env_ids] = dof_pos#有自由度的约束
        self._target_dof_vel[env_ids] = dof_vel#有自由度的约束
        
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
        self._target_dof_pos[env_ids] = dof_pos#有自由度的约束
        self._target_dof_vel[env_ids] = dof_vel#有自由度的约束
        


    def _compute_task_obs(self, env_ids=None):
        
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            target_pos = self._target_pos
            target_rot = self._target_rot
            target_vel = self._target_vel
            target_dof_pos = self._target_dof_pos
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
           
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            target_pos = self._target_pos[env_ids]
            target_rot = self._target_rot[env_ids]
            target_vel = self._target_vel
            target_dof_pos = self._target_dof_pos[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            

        # import pdb
        # pdb.set_trace()
        # obs = compute_imitation_observations(body_pos, body_rot, target_pos, target_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs)
        obs = compute_phys_observations(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs)
      
        return obs

    def _compute_reward(self, actions):
        actions[self.reset_buf == 1] = 0

        '''
        pos = self._rigid_body_pos
        rot = self._rigid_body_rot
        vel = self._rigid_body_vel
        angle_vel = self._rigid_body_ang_vel
        
        
        self.rew_buf[:] = compute_location_reward(pos, self._prev_target_pos, rot, self._prev_target_rot, vel, self._prev_target_vel, angle_vel, self._prev_target_angle_vel)
        '''
        
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel
        dof_pos = self._dof_pos
        dof_vel = self._dof_vel
        target_pos = self._prev_target_pos
        target_rot = self._prev_target_rot
        target_dof_pos = self._prev_target_dof_pos
        target_dof_vel = self._prev_target_dof_vel

        reward_specs = {'k_dof': 60, 'k_vel': 0.2, 'k_pos': 100, 'k_rot': 40, 'w_dof': 0.6, 'w_vel': 0.1, 'w_pos': 0.2, 'w_rot': 0.1}
        cfg_reward_specs = reward_specs

        self.rew_buf[:], _, _ = compute_imitation_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, 
                                                                                              target_dof_pos, target_dof_vel, body_vel, body_ang_vel, self._dof_obs_size, self._dof_offsets, self.body_pos_weights, reward_specs)
        
        

        reset_mask = self.reset_buf == 1
        if torch.any(reset_mask):
            self.rew_buf[reset_mask] = 0
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
def compute_imitation_observations(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    
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
    # target target dof   [N, dof]
    target_rel_dof_pos = target_dof_pos - dof_pos
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(target_rel_body_rot.shape[0], -1)


    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec, target_rel_dof_pos, target_rel_body_pos, target_rel_body_rot_obs), dim=-1)
    return obs

@torch.jit.script
def compute_phys_observations(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    
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
    # target target dof   [N, dof]
    target_rel_dof_pos = target_dof_pos - dof_pos
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(target_rel_body_rot.shape[0], -1)


    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec, target_rel_dof_pos, target_rel_body_pos, target_rel_body_rot_obs, target_pos.reshape(target_pos.shape[0], -1), target_rot.reshape(target_rot.shape[0], -1)), dim=-1)
    return obs


@torch.jit.script
def compute_imitation_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, target_dof_vel, body_vel, body_ang_vel, dof_obs_size, dof_offsets, body_pos_weights, reward_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor, str]
    
    k_dof, k_vel, k_pos, k_rot = reward_specs['k_dof'], reward_specs['k_vel'], reward_specs['k_pos'], reward_specs['k_rot']
    w_dof, w_vel, w_pos, w_rot = reward_specs['w_dof'], reward_specs['w_vel'], reward_specs['w_pos'], reward_specs['w_rot']
    
    # dof rot reward
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    target_dof_obs = dof_to_obs(target_dof_pos, dof_obs_size, dof_offsets)
    diff_dof_obs = dof_obs - target_dof_obs
    diff_dof_obs_dist = (diff_dof_obs ** 2).mean(dim=-1)
    dof_reward = torch.exp(-k_dof * diff_dof_obs_dist)

    # velocity reward
    diff_dof_vel = target_dof_vel - dof_vel
    diff_dof_vel_dist = (diff_dof_vel ** 2).mean(dim=-1)
    vel_reward = torch.exp(-k_vel * diff_dof_vel_dist)

    # body pos reward
    diff_body_pos = target_pos - body_pos
    diff_body_pos = diff_body_pos * body_pos_weights[:, None]
    diff_body_pos_dist = (diff_body_pos ** 2).mean(dim=-1).mean(dim=-1)
    body_pos_reward = torch.exp(-k_pos * diff_body_pos_dist)

    # body rot reward
    diff_body_rot = quat_mul(target_rot, quat_conjugate(body_rot))
    diff_body_rot_angle = torch_utils.quat_to_angle_axis(diff_body_rot)[0]
    diff_body_rot_angle_dist = (diff_body_rot_angle ** 2).mean(dim=-1)
    body_rot_reward = torch.exp(-k_rot * diff_body_rot_angle_dist)

    # reward = dof_reward * vel_reward * body_pos_reward * body_rot_reward
    reward = w_dof * dof_reward + w_vel * vel_reward + w_pos * body_pos_reward + w_rot * body_rot_reward
    sub_rewards = torch.stack([dof_reward, vel_reward, body_pos_reward, body_rot_reward], dim=-1)
    sub_rewards_names = 'dof_reward,vel_reward,body_pos_reward,body_rot_reward'
    return reward, sub_rewards, sub_rewards_names



@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_body_ids, rigid_body_pos, prev_target_pos, 
                           max_episode_length, enable_early_termination, termination_dists, cur_ref_motion_times, ref_motion_lengths):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    if (enable_early_termination):
        # 转到局部坐标系
        cur_wbpos = rigid_body_pos
        e_wbpos = prev_target_pos
        diff = cur_wbpos - e_wbpos
        jpos_diffw = torch.ones_like(diff)
        jpos_diffw[:, contact_body_ids] = 0
        diff *= jpos_diffw
        jpos_diffw = jpos_diffw[0,:,0].to(torch.bool)

        
        body_diff = torch.linalg.norm(diff[:,jpos_diffw], dim=-1).mean(dim=-1)
        
        

        
        has_fallen = body_diff > termination_dists

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reach_max_length = progress_buf >= max_episode_length - 1
    reach_max_dur = cur_ref_motion_times >= ref_motion_lengths
    reset_cond = torch.logical_or(reach_max_length, reach_max_dur)
    reset = torch.where(reset_cond, torch.ones_like(reset_buf), terminated)
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

