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
        # self.last_contacts = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.bool, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.desired_contact_indices), dtype=torch.bool, device=self.device)  # [num_envs, 2]
        self.feet_air_time = torch.zeros(self.num_envs, len(self.desired_contact_indices), dtype=torch.float, device=self.device)  # [num_envs, 2]
        self.num_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.actor_handles[0])
        


    def compute_observations(self):
        """ Computes observations
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        self.compute_ref_state()
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_pos,  # 1
                                    cos_pos  # 1
                                    ),dim=-1)
        # add noise if needed
        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
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
        noise_vec[45:46] =0.
        noise_vec[46:47] = 0.
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[47:234] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def step(self,actions):
        # actions = self.ref_dof_pos*4
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        super().post_physics_step()
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        swing_mask = 1 - self._get_gait_phase().float()
        self.swing_mask = swing_mask * (1 - self.standing_command_mask.unsqueeze(1))
        self.stance_mask = 1 - self.swing_mask

        self.swing_mask_l = self.swing_mask[:, 0]
        self.swing_mask_r = self.swing_mask[:, 1]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[
            : self.num_envs * self.num_bodies, :
        ]
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.standing_command_mask = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        

        
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time  # Период цикла шага (1.6 с)
        phase = (self.episode_length_buf * self.dt / cycle_time)  # Фаза цикла [0, 1] [num_envs]
        # print(phase)
        return phase  # [num_envs]

    def _get_gait_phase(self):
        phase = self._get_phase()  # Фаза цикла [num_envs]
        sin_pos = torch.sin(2 * torch.pi * phase)  # Синус фазы [-1, 1] [num_envs]
        gait_mask = torch.zeros((self.num_envs, len(self.desired_contact_indices)), dtype=torch.bool, device=self.device)  # Маска: True - опора, False - полёт [num_envs, 2]
        gait_mask[:, 0] = sin_pos >= 0  # RR в опоре при sin ≥ 0 (фаза [0, 0.5])
        gait_mask[:, 1] = sin_pos < 0   # RL в опоре при sin < 0 (фаза [0.5, 1.0])
        gait_mask[torch.abs(sin_pos) < self.cfg.rewards.bias] = 1 # Двойная опора при |sin| < bias
        return gait_mask  # [num_envs, 2]
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # print(sin_pos)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = 2
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 6] = 0#sin_pos_l * scale_1
        self.ref_dof_pos[:, 7] = 1.57
        self.ref_dof_pos[:, 8] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 9] = 0#-sin_pos_r * scale_1
        self.ref_dof_pos[:, 10] = 1.57
        self.ref_dof_pos[:, 11] = -sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < self.cfg.rewards.bias] = 0

    
    def _reward_tracking_pitch(self):
        # Tracking
        base_quat = self.root_states[:, 3:7]
        euler = get_euler_xyz(base_quat)
        episode_time_buf = self.episode_length_buf * self.dt
        pitch_command = episode_time_buf * self.cfg.commands.pitch / self.cfg.commands.standup_duration
        # pitch_command = torch.clip(pitch_command, self.cfg.commands.pitch, 0.)
        # Clip для безопасности (между min being 0 и max=target)
        pitch_command = torch.clamp(pitch_command, 0.0, self.cfg.commands.pitch)
        error = torch.square(pitch_command - euler[:, 1]) + torch.square(self.cfg.commands.roll - euler[:, 0])
        # print(f"orient_grav{self.projected_gravity}")
        # print(f"orient[0]{euler[:,0]}")
        # print(f"orient[1]{euler[:,1]}")
        # print(f"Diff {pitch_command-euler[:,1]}")
        return torch.exp(-1*error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_pos(self):
        hip_names = ["FR_hip_joint", "FL_hip_joint","RR_hip_joint", "RL_hip_joint"]
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
    
   
    def _reward_base_height(self):
        # Penalize base height
        # height_error = self.root_states[:, 2] - self.cfg.rewards.base_height_target
        # return torch.exp(-1* torch.square(height_error / self.cfg.rewards.tracking_sigma))
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        print(f"Base_height: {base_height}")
        error = base_height - self.cfg.rewards.base_height_target
        # print(f"ERROR {error}")
        rew = torch.exp(-1* torch.square(error / (self.cfg.rewards.tracking_sigma)))
        # print(f"REW {rew}")
        return rew


    def _reward_com_over_support(self):
        base_pos = self.body_state_buffer[:, self.base_index, 0:3]
        rr_pos = self.body_state_buffer[:, self.rr_foot_idx, 0:3]
        rl_pos = self.body_state_buffer[:, self.rl_foot_idx, 0:3]
        support_center = 0.5 * (rr_pos + rl_pos)
        target_height = self.cfg.rewards.base_height_target
        error = (0.4 * torch.square(base_pos[:, 0] - support_center[:, 0]) +
                0.4 * torch.square(base_pos[:, 1] - support_center[:, 1]) +
                0.8 * torch.square(base_pos[:, 2] - target_height))
        return torch.exp(-8.0 * error)
    

    def _reward_rear_feet_contact_and_air(self):
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 50.0  # [num_envs, 2]
        contact_changes = torch.abs(contact.float() - self.last_contacts.float())  # [num_envs, 2]
        self.last_contacts = contact
        gait_mask = self._get_gait_phase()  # [num_envs, 2]
        contact_reward = torch.sum(1.0 * contact * gait_mask, dim=1)  # Только текущие контакты
        swing_reward = torch.sum(1.0 * (~contact) * (~gait_mask), dim=1)  # Увеличен вес
        contact_change_penalty = -0.9 * torch.sum(contact_changes, dim=1)  # Штраф за частые переключения
        undesired_contact_penalty = -15 * torch.sum(self.contact_forces[:, self.undesired_contact_indices, 2] > 1.0, dim=1)
        return contact_reward + swing_reward + contact_change_penalty + undesired_contact_penalty 
    

    def _reward_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.1 * torch.sum(torch.abs(self.actions), dim=1)
        return 0.5*term_1 + 0.5*term_2 + term_3
    
    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > self.cfg.rewards.command_dead)
    
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_feet_air_time_2(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        # Контакты задних лап (RR, RL)
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 50.0  # [num_envs, 2]
        # print(f"contact={contact}")
        
        # Reward long steps

        # print(contact)
        # print(self.last_contacts)
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
            # Отладка
        # print(f"FeetAirTime2: reward={rew_airTime.mean()}, air_time={self.feet_air_time.mean()}")
        return rew_airTime
    
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 5.0
        # print(f"Contact{contact}")
        # print(f"rigid_body_state {self.rigid_body_state }")
        foot_speed_norm = torch.norm(self.rigid_state[:, self.desired_contact_indices, 7:9], dim=2)
        # print(f"foot_speed_norm{foot_speed_norm}")
        rew = torch.sqrt(foot_speed_norm)
        # print(f"rew{rew}")
        rew *= contact
        # print(f"rew* {torch.sum(rew, dim=1)}")
        #print(f"Reward for feet slip (env 0): {rew}")
        return torch.sum(rew, dim=1)
    def _reward_feet_clearance(self):#鼓励抬脚高度
        """
        Поощряет подъём задних лап (RR, RL) на целевую высоту в фазе свинга для ритмичной походки.
        Использует фиксированную целевую высоту (target_foot_height) и экспоненциальную награду.
        Сбрасывает высоту при контакте и применяется только при движении (lin_vel > 0.1).
        """
        # Высота лап относительно земли
        self.feet_height = self.rigid_state[:, self.desired_contact_indices, 2] - 0.02  # [num_envs, 2]
        # print(f"Height {self.feet_height}")
        contact = self.contact_forces[:, self.desired_contact_indices, 2] > 1.0  # [num_envs, 2]
        self.feet_height *= ~contact  # Сброс высоты при контакте
        swing_mask = torch.logical_not(self._get_gait_phase())  # [num_envs, 2]
        target_height = self.cfg.rewards.target_foot_height  # Фиксированная высота (например, 0.1 м)
        # Экспоненциальная награда за близость к целевой высоте в свинге
        error = torch.abs(self.feet_height - target_height)  # [num_envs, 2]
        print(f" height { self.feet_height }")
        # print(error)
        rew = torch.exp(-error * swing_mask )  # [num_envs, 2]
        reward = torch.sum(rew, dim=1)  # [num_envs]
        # Награда только при движении
        # print(f"RearFeetClearance: reward={reward.mean()}, feet_height={self.feet_height.mean()}, target_height={target_height}, swing_mask={swing_mask.float().mean()}, moving={moving.float().mean()}, contact={contact.float().mean()}")
        return reward
    
    # def _reward_feet_contact_forces(self):
    #     return torch.sum(
    #         (
    #             torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
    #             - self.cfg.rewards.max_contact_force
    #         ).clip(0, 120),
    #         dim=1,
    #     )

    # def _reward_tracking_lin_vel(self):
    #     """
    #     Tracks linear velocity commands along the xy axes.
    #     Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
    #     """
    #     error = self.commands[:, :2] - self.base_lin_vel[:, :2]
    #     error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
    #     rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_lin).sum(dim=1)/2
    #     return rew

    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """

    #     error = self.commands[:, 2] - self.base_ang_vel[:, 2]
    #     error *= 1.0 / (1.0 + torch.abs(self.commands[:, 2]))
    #     rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_ang)
    #     # print(rew.size())
    #     return rew
    
# * ######################### HELPER FUNCTIONS ############################## * #

    def _neg_exp(self, x, a=1):
        """ shorthand helper for negative exponential e^(-x/a)
            a: range of x
        """
        return torch.exp(-(x/a)/a)

    def _neg_sqrd_exp(self, x, a=1):
        """ shorthand helper for negative squared exponential e^(-(x/a)^2)
            a: range of x
        """
        return torch.exp(-torch.square(x/a)/a)