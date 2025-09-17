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
        self.energy_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 330
        return obs_size

    def pre_physics_step(self, actions):
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
        
        # super()._compute_reset()

        cur_ref_motion_times =  self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths[self._reset_ref_motion_ids]

        self.reset_buf[:], self._terminate_buf[:]= compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_body_ids, self._root_body_id, self._rigid_body_pos, self._prev_target_pos, 
                                                   self.max_episode_length, self._enable_early_termination, self._termination_dists, cur_ref_motion_times, ref_motion_lengths)
        
        return

    
        

    def _update_task(self):
        # 是否超过了序列长度 
        # 是否跟上了，跟上再更新
        # hold_buf = compute_diff(self.reset_buf, self.progress_buf,self._contact_body_ids,self._rigid_body_pos, self._prev_target_pos,0.2)
        # self._cur_ref_motion_times[hold_buf] -= self.dt
            
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
            target_angle_vel = self._target_angle_vel
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
            target_vel = self._target_vel[env_ids]
            target_angle_vel = self._target_angle_vel[env_ids]
            
            target_dof_pos = self._target_dof_pos[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            

        
        obs = compute_location_observations(body_pos, body_rot, target_pos, target_rot, body_vel, body_ang_vel , dof_pos, dof_vel, target_vel, target_angle_vel)
    
        return obs

    def _compute_reward(self, actions):

        pos = self._rigid_body_pos
        rot = self._rigid_body_rot
        vel = self._rigid_body_vel
        angle_vel = self._rigid_body_ang_vel
        dof_pos = self._dof_pos
        dof_vel = self._dof_vel
        
        self.rew_buf[:] = compute_location_reward2d(pos, self._prev_target_pos, rot, self._prev_target_rot, vel, self._prev_target_vel, angle_vel, self._prev_target_angle_vel)

        self.energy_buf[:] = self.get_energy()
        self.rew_buf += 2 * self.energy_buf

        

        reset_mask = self.reset_buf == 1
        if torch.any(reset_mask):
            self.rew_buf[reset_mask] = 0
        return
    
    def get_energy(self):
        energy = torch.abs(torch.multiply(self._dof_force, self._dof_vel)).sum(dim=-1) 

        energy =  -0.0005*energy
        return energy



#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v

@torch.jit.script
def compute_location_observations(body_pos, body_rot, target_pos, target_rot, body_vel, body_ang_vel , dof_pos, dof_vel, target_vel, target_angle_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    pos_diff = target_pos - body_pos

    diff_rot = quat_mul(target_rot, quat_conjugate(body_rot))

    diff_rot = torch_utils.quat_to_tan_norm(diff_rot.view(-1, 4)).view(diff_rot.shape[0], diff_rot.shape[1], -1) #6dof


    vel_diff = target_vel - body_vel

    avel_diff = target_angle_vel - body_ang_vel

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_body_rot_obs = torch_utils.quat_to_tan_norm(flat_body_rot)
    body_rot = flat_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1], flat_body_rot_obs.shape[1]) #计算6dof旋转


    obs = torch.cat((diff_rot, pos_diff, vel_diff, avel_diff, target_rot, target_pos), dim=-1)
    n = obs.shape[0]
    obs = obs.reshape(n,-1)

    return obs


@torch.jit.script
def compute_diff(reset_buf, progress_buf, contact_body_ids, rigid_body_pos, prev_target_pos, termination_dists):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    terminated = torch.zeros_like(reset_buf)
    diff = rigid_body_pos - prev_target_pos
    jpos_diffw = torch.ones_like(diff)
    jpos_diffw[:, contact_body_ids] = 0
    diff *= jpos_diffw
    jpos_diffw = jpos_diffw[0,:,0].to(torch.bool)

    
    body_diff = torch.linalg.norm(diff[:,jpos_diffw], dim=-1).mean(dim=-1)
    has_fallen = body_diff > termination_dists

    has_fallen *= (progress_buf > 1)
    terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    

    
    return  terminated

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
    
    

    # 是否改用dof形式
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

@torch.jit.script
def compute_location_reward2d(pos, _target_pos, rot, _target_rot, vel, _target_vel, avel, _target_avel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    pos_err_scale = 100
    rot_err_scale = 10
    vel_err_scale = 0.1
    avel_err_scale = 0.1
    
    pos_reward_w = 0.5
    rot_reward_w = 0.3
    vel_reward_w = 0.1
    avel_reward_w = 0.1
    

    pos_diff = _target_pos[:,:,1:] - pos[:,:,1:]
    pos_err = (pos_diff * pos_diff).mean(dim=-1).mean(dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    
    

    # 是否改用dof形式
    diff_rot = quat_mul(_target_rot, quat_conjugate(rot))
    diff_rot_angle = torch_utils.quat_to_angle_axis(diff_rot)[0]
    diff_rot_angle_dist = (diff_rot_angle ** 2).mean(dim=-1)
    _rot_reward = torch.exp(-rot_err_scale * diff_rot_angle_dist)
    

    vel_diff = _target_vel[:,:,1:] - vel[:,:,1:]
    vel_err = (vel_diff * vel_diff).mean(dim=-1).mean(dim=-1)
    _vel_reward = torch.exp(-vel_err_scale * vel_err)

    avel_diff = _target_avel - avel
    avel_err = (avel_diff * avel_diff).mean(dim=-1).mean(dim=-1)
    _avel_reward = torch.exp(-avel_err_scale * avel_err)


   


    reward = pos_reward_w * pos_reward + rot_reward_w * _rot_reward + vel_reward_w * _vel_reward + avel_reward_w * _avel_reward
    
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_body_ids, root_body_id, rigid_body_pos, prev_target_pos, 
                           max_episode_length, enable_early_termination, termination_dists, cur_ref_motion_times, ref_motion_lengths):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    if (enable_early_termination):
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

    return reset, terminated


