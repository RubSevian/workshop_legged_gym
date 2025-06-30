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

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time  # Период цикла шага (1.6 с)
        phase = (self.episode_length_buf * self.dt / cycle_time) % 1.0  # Фаза цикла [0, 1] [num_envs]
        return phase  # [num_envs]

    def _get_gait_phase(self):
        phase = self._get_phase()  # Фаза цикла [num_envs]
        sin_pos = torch.sin(2 * torch.pi * phase)  # Синус фазы [-1, 1] [num_envs]
        gait_mask = torch.zeros((self.num_envs, len(self.desired_contact_indices)), dtype=torch.bool, device=self.device)  # Маска: True - опора, False - полёт [num_envs, 2]
        gait_mask[:, 0] = sin_pos >= 0  # RR в опоре при sin ≥ 0 (фаза [0, 0.5])
        gait_mask[:, 1] = sin_pos < 0   # RL в опоре при sin < 0 (фаза [0.5, 1.0])
        gait_mask[torch.abs(sin_pos) < self.cfg.rewards.bias] = True  # Двойная опора при |sin| < bias
        return gait_mask  # [num_envs, 2]

    
    def _reward_tracking_pitch(self):
        # Tracking
        base_quat = self.root_states[:, 3:7]  # [num_envs, 4]
        euler = get_euler_xyz(base_quat)  # Tuple of [num_envs] tensors: (roll, pitch, yaw)
        roll = euler[:,0]  # [num_envs]
        pitch = euler[:,1]  # [num_envs]
        pitch_error = torch.abs(pitch - self.cfg.commands.pitch)  # [num_envs]
        roll_error = torch.abs(roll - self.cfg.commands.roll)  # [num_envs]
        total_error = pitch_error + roll_error  # [num_envs]
        return torch.exp(-2 * total_error / self.cfg.rewards.tracking_sigma)
        # episode_time_buf = self.episode_length_buf * self.dt
        # pitch_command = episode_time_buf * self.cfg.commands.pitch / self.cfg.commands.standup_duration
        # pitch_command = torch.clip(pitch_command, self.cfg.commands.pitch, 0.)
        # error = torch.square(pitch_command - euler[:, 1]) + torch.square(self.cfg.commands.roll - euler[:, 0])
        # print (self.cfg.)
        # return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_pos(self):
        hip_names = ["RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        error = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)  # [num_envs]
        # hip_names = ["RR_hip_joint", "RL_hip_joint"]
        # self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i, name in enumerate(hip_names):
        #     self.hip_indices[i] = self.dof_names.index(name)
        # return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
    
    def _reward_feet_contact(self):
        contact = self.contact_forces[:, :, 2] > 50.0  # [num_envs, num_bodies]
        desired_contact = torch.sum(contact[:, self.desired_contact_indices], dim=1)  # [num_envs]
        undesired_contact = torch.sum(contact[:, self.undesired_contact_indices], dim=1)  # [num_envs]
        slip_penalty = -0.5 * torch.sum(torch.norm(self.contact_forces[:, self.desired_contact_indices, 0:2], dim=2), dim=1)  # [num_envs]
        return 1.0 * desired_contact - 4.0 * undesired_contact + slip_penalty  # [num_envs]
    
    def _reward_base_height(self):
                # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = torch.square(base_height - 0.45)
        return torch.exp(-2* error)

    def _reward_com_over_support(self):
        base_pos = self.body_state_buffer[:, self.base_index, 0:3]
        rr_pos = self.body_state_buffer[:, self.rr_foot_idx, 0:3]
        rl_pos = self.body_state_buffer[:, self.rl_foot_idx, 0:3]
        support_center = 0.5 * (rr_pos + rl_pos)
        target_height = 0.42
        error = (0.4 * torch.square(base_pos[:, 0] - support_center[:, 0]) +
                0.4 * torch.square(base_pos[:, 1] - support_center[:, 1]) +
                0.8 * torch.square(base_pos[:, 2] - target_height))
        return torch.exp(-5.0 * error)
    

    def _reward_rear_feet_contact_and_air(self):
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 50.0  # [num_envs, 2]
        contact_changes = torch.abs(contact.float() - self.last_contacts[:, self.desired_contact_indices].float())  # [num_envs, 2]
        self.last_contacts[:, self.desired_contact_indices] = contact
        gait_mask = self._get_gait_phase()  # [num_envs, 2]
        contact_reward = torch.sum(1.0 * contact * gait_mask, dim=1)  # Только текущие контакты
        swing_reward = torch.sum(1.0 * (~contact) * (~gait_mask), dim=1)  # Увеличен вес
        contact_change_penalty = -1.0 * torch.sum(contact_changes, dim=1)  # Штраф за частые переключения
        undesired_contact_penalty = -10.0 * torch.sum(self.contact_forces[:, self.undesired_contact_indices, 2] > 20.0, dim=1)
        return contact_reward + swing_reward + contact_change_penalty + undesired_contact_penalty 