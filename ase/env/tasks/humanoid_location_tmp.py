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
from env.tasks.humanoid import Humanoid, dof_to_obs
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
        
        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._target_pos = torch.zeros([self.num_envs, 17, 3], device=self.device, dtype=torch.float)
        self._target_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._target_dof_pos = torch.zeros([self.num_envs, 31], device=self.device, dtype=torch.float)
        self._target_root_vel = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._target_root_ang_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._target_dof_vel = torch.zeros([self.num_envs, 31], device=self.device, dtype=torch.float)
        self._target_key_pos = torch.zeros([self.num_envs, 6, 3], device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        



        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2
        return obs_size

    def pre_physics_step(self, actions):
        # 结束的action置0
        actions[self.reset_buf == 1] = 0
        super().pre_physics_step(actions)
        

        self._prev_target_rot = self._target_root_rot.clone()
        self._prev_target_pos = self._target_pos.clone()
        # previous target
        # self._prev_target_root_pos = self._target_root_pos.clone()
        # self._prev_target_root_rot = self._target_root_rot.clone()
        # self._prev_target_dof_pos = self._target_dof_pos.clone()
        # self._prev_target_root_vel = self._target_root_vel.clone()
        # self._prev_target_root_ang_vel = self._target_root_ang_vel.clone()
        # self._prev_target_dof_vel = self._target_dof_vel.clone()
        # self._prev_target_key_pos = self._target_key_pos.clone()
        # self._prev_target_rb_pos = self._target_rb_pos.clone()
        # self._prev_target_rb_rot = self._target_rb_rot.clone()
        return

    def post_physics_step(self):
        self._update_task()
        super().post_physics_step()
        self._compute_reset()

        return
    def _update_marker(self):
        self._marker_pos[..., 0:2] = self._target_root_pos
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
        
    def _compute_reset(self):
        cur_ref_motion_times =  self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths.to(self.device)[self._reset_ref_motion_ids]

        old_reset_buf = self.reset_buf.clone()
        old_terminate_buf = self._terminate_buf.clone()

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._tar_change_steps_max,
                                                                           cur_ref_motion_times, ref_motion_lengths)
        reset_mask = old_reset_buf == 1
        if torch.any(reset_mask):
            self.reset_buf[reset_mask] = 1
            self._terminate_buf[reset_mask] = old_terminate_buf[reset_mask]
        
        return

    def _next_step(self, env_ids):
        # 更新目标点
        
        motion_ids = self._reset_ref_motion_ids[env_ids]
        motion_times = self._cur_ref_motion_times[env_ids] + self.dt     # next frame

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        self._target_pos[env_ids] = root_pos
        self._target_root_rot[env_ids] = root_rot
        self._target_dof_pos[env_ids] = dof_pos
        self._target_root_vel[env_ids] = root_vel[:,0:2]
        self._target_root_ang_vel[env_ids] = root_ang_vel
        self._target_dof_vel[env_ids] = dof_vel
        self._target_key_pos[env_ids] = key_pos

    def _reset_task(self, env_ids):
        # 更新目标点,停在原地
        
        
        motion_ids = self._reset_ref_motion_ids[env_ids]
        motion_times = self._cur_ref_motion_times[env_ids] 
        if len(motion_times.nonzero(as_tuple=False).flatten()) == 0:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times + self.dt)
                
            self._target_pos[env_ids] = root_pos
            self._target_root_rot[env_ids] = root_rot
            self._target_dof_pos[env_ids] = dof_pos
            self._target_root_vel[env_ids] = root_vel[:,0:2]
            self._target_root_ang_vel[env_ids] = root_ang_vel
            self._target_dof_vel[env_ids] = dof_vel
            self._target_key_pos[env_ids] = key_pos
        
        return 



    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            _target_pos = self._target_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            _target_pos = self._target_pos[env_ids]
        
        obs = compute_location_observations(root_states, _target_pos[:,0,0:2])
        return obs

    def _compute_reward(self, actions):
        actions[self.reset_buf == 1] = 0
        root_pos = self._humanoid_root_states[..., 0:2]
        # root_rot = self._humanoid_root_states[..., 3:7]
        
        self.rew_buf[:] = compute_location_reward(root_pos, self._prev_target_pos[:,0,0:2])
        # pos = self._rigid_body_pos
        # # rot = self._rigid_body_rot
        
        
        # self.rew_buf[:] = compute_location_reward(pos, self._prev_target_pos)


        
        
        # body_root_pos = self._humanoid_root_states[..., 0:2]
        # body_root_rot = self._humanoid_root_states[..., 3:7]
        # body_vel = self._rigid_body_vel
        # body_ang_vel = self._rigid_body_ang_vel
        # dof_pos = self._dof_pos #
        # dof_vel = self._dof_vel #

        # target_root_pos = self._prev_target_root_pos
        # target_root_rot = self._prev_target_root_rot
        # target_dof_pos = self._prev_target_dof_pos
        # target_dof_vel = self._prev_target_dof_vel

        # reward_specs = {'k_dof': 60, 'k_vel': 0.2, 'k_pos': 0.5, 'k_rot': 40, 'w_dof': 0.6, 'w_vel': 0.1, 'w_pos': 0.2, 'w_rot': 0.1}
        # # cfg_reward_specs = self.cfg['env'].get('reward_specs', dict())
        # # reward_specs.update(cfg_reward_specs)

        # self.rew_buf[:] = compute_location_reward(body_root_pos, body_root_rot, target_root_pos, target_root_rot, dof_pos, dof_vel, 
        #                                         target_dof_pos, target_dof_vel, body_vel, body_ang_vel, self._dof_obs_size, self._dof_offsets, reward_specs)
        # 结束的rew置零
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
def compute_humanoid_reset(reset_buf, progress_buf,_tar_change_steps_max, cur_ref_motion_times, ref_motion_lengths):
    # type: (Tensor, Tensor, float,  Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    reach_max_length = progress_buf >= _tar_change_steps_max - 1
    reach_max_dur = cur_ref_motion_times >= ref_motion_lengths
    reset_cond = torch.logical_or(reach_max_length, reach_max_dur)
    reset = torch.where(reset_cond, torch.ones_like(reset_buf), terminated)

    return reset, terminated



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

# @torch.jit.script
# def compute_location_reward(body_root_pos, body_root_rot, target_root_pos, target_root_rot, dof_pos, dof_vel, target_dof_pos, target_dof_vel, body_vel, body_ang_vel, dof_obs_size, dof_offsets, reward_specs):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Dict[str, float]) -> Tensor
    
#     k_dof, k_vel, k_pos, k_rot = reward_specs['k_dof'], reward_specs['k_vel'], reward_specs['k_pos'], reward_specs['k_rot']
#     w_dof, w_vel, w_pos, w_rot = reward_specs['w_dof'], reward_specs['w_vel'], reward_specs['w_pos'], reward_specs['w_rot']
    
#     # dof rot reward
#     dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
#     target_dof_obs = dof_to_obs(target_dof_pos, dof_obs_size, dof_offsets)
    
#     diff_dof_obs = dof_obs - target_dof_obs
    
#     diff_dof_obs_dist = (diff_dof_obs ** 2).sum(dim=-1)
#     dof_reward = torch.exp(-k_dof * diff_dof_obs_dist)
    

#     # velocity reward
#     diff_dof_vel = target_dof_vel - dof_vel
#     diff_dof_vel_dist = (diff_dof_vel ** 2).sum(dim=-1)
#     vel_reward = torch.exp(-k_vel * diff_dof_vel_dist)
    
    
#     # body pos reward
#     # pos_err_scale = 0.5
#     # pos_diff = target_root_pos - body_root_pos
#     # pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
#     # pos_reward = torch.exp(-pos_err_scale * pos_err)
#     diff_body_pos = target_root_pos - body_root_pos
#     diff_body_pos_dist = (diff_body_pos ** 2).sum(dim=-1)
#     body_pos_reward = torch.exp(-k_pos * diff_body_pos_dist)

#     # body rot reward
#     diff_body_rot = quat_mul(target_root_rot, quat_conjugate(body_root_rot))
#     diff_body_rot_angle = torch_utils.quat_to_angle_axis(diff_body_rot)[0]
#     diff_body_rot_angle_dist = (diff_body_rot_angle ** 2).sum(dim=-1)
#     body_rot_reward = torch.exp(-k_rot * diff_body_rot_angle_dist)

#     reward = dof_reward * vel_reward * body_pos_reward * body_rot_reward
#     reward = w_dof * dof_reward + w_vel * vel_reward + w_pos * body_pos_reward + w_rot * body_rot_reward
#     sub_rewards = torch.stack([dof_reward, vel_reward, body_pos_reward, body_rot_reward], dim=-1)
#     sub_rewards_names = 'dof_reward,vel_reward,body_pos_reward,body_rot_reward'
#     return reward


@torch.jit.script
def compute_location_reward(pos, _target_pos):
    # type: (Tensor, Tensor) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1

    pos_diff = _target_pos - pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    # tar_dir = _target_root_pos - root_pos[..., 0:2]
    # tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
#     # delta_root_pos = root_pos - prev_root_pos
#     # root_vel = delta_root_pos / dt
#     # tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
#     # tar_vel_err = tar_speed - tar_dir_speed
#     # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
#     # vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
#     # speed_mask = tar_dir_speed <= 0
#     # vel_reward[speed_mask] = 0


#     # heading_rot = torch_utils.calc_heading_quat(root_rot)
#     # facing_dir = torch.zeros_like(root_pos)
#     # facing_dir[..., 0] = 1.0
#     # facing_dir = quat_rotate(heading_rot, facing_dir)
#     # facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
#     # facing_reward = torch.clamp_min(facing_err, 0.0)


#     # dist_mask = pos_err < dist_threshold
#     # facing_reward[dist_mask] = 1.0
#     # vel_reward[dist_mask] = 1.0

    reward = pos_reward

    return reward

# @torch.jit.script
# def compute_location_reward(pos, _target_pos):
#     # type: (Tensor, Tensor) -> Tensor
#     dist_threshold = 0.5

#     pos_err_scale = 0.5
#     vel_err_scale = 4.0

#     pos_reward_w = 0.5
#     vel_reward_w = 0.4
#     face_reward_w = 0.1

    
#     pos_diff = _target_pos - pos
#     pos_err = (pos_diff ** 2).sum(dim=-1).mean(dim=-1)

#     pos_reward = torch.exp(-pos_err_scale * pos_err)

#     # tar_dir = _target_root_pos - root_pos[..., 0:2]
#     # tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
# #     # delta_root_pos = root_pos - prev_root_pos
# #     # root_vel = delta_root_pos / dt
# #     # tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
# #     # tar_vel_err = tar_speed - tar_dir_speed
# #     # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
# #     # vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
# #     # speed_mask = tar_dir_speed <= 0
# #     # vel_reward[speed_mask] = 0


# #     # heading_rot = torch_utils.calc_heading_quat(root_rot)
# #     # facing_dir = torch.zeros_like(root_pos)
# #     # facing_dir[..., 0] = 1.0
# #     # facing_dir = quat_rotate(heading_rot, facing_dir)
# #     # facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
# #     # facing_reward = torch.clamp_min(facing_err, 0.0)


# #     # dist_mask = pos_err < dist_threshold
# #     # facing_reward[dist_mask] = 1.0
# #     # vel_reward[dist_mask] = 1.0

#     reward = pos_reward

#     return reward