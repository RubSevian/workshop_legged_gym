from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import get_euler_xyz
from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch, gymapi

import torch

class Go2(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Initialize indices for feet and thighs
        self.rr_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RR_foot")
        self.rl_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RL_foot")
        self.fl_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FL_foot")
        self.fr_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FR_foot")
        self.rr_thigh_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RR_thigh")
        self.rl_thigh_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RL_thigh")
        self.fl_thigh_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FL_thigh")
        self.fr_thigh_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FR_thigh")
        self.rr_calf_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RR_calf")
        self.rl_calf_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "RL_calf")
        self.fl_calf_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FL_calf")
        self.fr_calf_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "FR_calf")
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "base")
        self.body_state_buffer = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device)
        self.desired_contact_indices = torch.tensor([self.rr_foot_idx, self.rl_foot_idx], dtype=torch.long, device=self.device, requires_grad=False)
        self.undesired_contact_indices = torch.tensor([self.fl_foot_idx, self.fr_foot_idx, self.rr_thigh_idx, self.rl_thigh_idx, self.fl_thigh_idx, self.fr_thigh_idx, self.rr_calf_idx, self.rl_calf_idx, self.fl_calf_idx, self.fr_calf_idx], dtype=torch.long, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.bool, device=self.device)
        self.num_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.actor_handles[0])
        self.feet_air_time = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device)
        # Initialize stance mask for feet contact reward
        self.stance_mask = torch.zeros(self.num_bodies, dtype=torch.bool, device=self.device)
        self.stance_mask[self.desired_contact_indices] = True

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    def update_body_states(self):
        rb_states = self.gym.get_actor_rigid_body_states(self.sim, self.actor_handles[0], gymapi.STATE_ALL)
        body_states = gymtorch.wrap_tensor(rb_states).view(self.num_envs, self.num_bodies, 13)
        self.body_state_buffer.copy_(body_states)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        return noise_vec

    def _reward_tracking_pitch(self):
        # Tracking
        base_quat = self.root_states[:, 3:7]
        euler = get_euler_xyz(base_quat)
        episode_time_buf = self.episode_length_buf * self.dt
        pitch_command = episode_time_buf * self.cfg.commands.pitch / self.cfg.commands.standup_duration
        pitch_command = torch.clip(pitch_command, self.cfg.commands.pitch, 0.)
        error = torch.square(pitch_command - euler[:, 1]) + torch.square(self.cfg.commands.roll - euler[:, 0])
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_pos(self):
        hip_names = ["RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
    
    def _reward_feet_contact(self):
        contact = self.contact_forces[:, :, 2] > 50.0  # [num_envs, num_bodies]
        desired_contact = torch.sum(contact[:, self.desired_contact_indices], dim=1)  # [num_envs]
        undesired_contact = torch.sum(contact[:, self.undesired_contact_indices], dim=1)  # [num_envs]
        slip_penalty = -0.5 * torch.sum(torch.norm(self.contact_forces[:, self.desired_contact_indices, 0:2], dim=2), dim=1)  # [num_envs]
        return 1.0 * desired_contact - 3.0 * undesired_contact + slip_penalty  # [num_envs]
    
    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        error = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-5 * error)

    def _reward_com_over_support(self):
        base_pos = self.body_state_buffer[:, self.base_index, 0:3]
        rr_pos = self.body_state_buffer[:, self.rr_foot_idx, 0:3]
        rl_pos = self.body_state_buffer[:, self.rl_foot_idx, 0:3]
        support_center = 0.5 * (rr_pos + rl_pos)
        target_height = self.cfg.rewards.base_height_target
        error = (0.4 * torch.square(base_pos[:, 0] - support_center[:, 0]) +
                0.4 * torch.square(base_pos[:, 1] - support_center[:, 1]) +
                0.8 * torch.square(base_pos[:, 2] - target_height))
        return torch.exp(-5.0 * error)
    

    def _reward_rear_feet_contact_and_air(self):
        # Contact reward for rear feet
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 80.0  # [num_envs, 2]
        contact_filt = torch.logical_or(contact, self.last_contacts[:, self.desired_contact_indices])  # [num_envs, 2]
        self.last_contacts[:, self.desired_contact_indices] = contact
        contact_reward = torch.sum(1.0 * contact_filt, dim=1)  # [num_envs]

        # Air time reward for steps (dependent on commands)
        first_contact = (self.feet_air_time[:, self.desired_contact_indices] > 0.) * contact_filt  # [num_envs, 2]
        self.feet_air_time[:, self.desired_contact_indices] += self.dt
        air_time_reward = torch.sum(0.3 * (self.feet_air_time[:, self.desired_contact_indices] - 0.5) * first_contact, dim=1)  # [num_envs]
        air_time_reward *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # [num_envs]
        self.feet_air_time[:, self.desired_contact_indices] *= ~contact_filt

        return contact_reward + 0.5 * air_time_reward  # [num_envs]