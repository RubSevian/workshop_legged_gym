from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch
import torch


class Go2_Walk(LeggedRobot):
    """Clock-free Go2 locomotion baseline."""

    def compute_observations(self):
        """Append one proprioceptive frame and build critic-only terrain data."""
        current_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        if current_obs.shape[1] != self.cfg.env.num_single_observations:
            raise RuntimeError(
                "go2_walk single observation size does not match "
                "cfg.env.num_single_observations"
            )

        if self.add_noise:
            current_obs += (
                2 * torch.rand_like(current_obs) - 1
            ) * self.single_obs_noise_scale

        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = current_obs

        # Repeat the first valid frame after reset. Zero padding would expose an
        # artificial reset marker that the physical robot will never produce.
        init_ids = self.history_needs_init.nonzero(as_tuple=False).flatten()
        if len(init_ids) > 0:
            self.obs_history[init_ids] = current_obs[init_ids].unsqueeze(1).repeat(
                1, self.cfg.env.history_length, 1
            )
            self.history_needs_init[init_ids] = False

        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)

        if self.privileged_obs_buf is not None:
            height_scan = torch.clip(
                self.root_states[:, 2].unsqueeze(1)
                - self.cfg.rewards.base_height_target
                - self.measured_heights,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
            contacts = (
                self.contact_forces[:, self.feet_indices, 2]
                > self.cfg.rewards.contact_force_threshold
            ).float()
            self.privileged_obs_buf = torch.cat(
                (
                    self.obs_buf,
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    contacts,
                    height_scan,
                ),
                dim=-1,
            )
            if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
                raise RuntimeError(
                    "go2_walk privileged observation size does not match "
                    "cfg.env.num_privileged_obs"
                )

    def _get_noise_scale_vec(self, cfg):
        if cfg.env.history_length < 1:
            raise ValueError("cfg.env.history_length must be at least 1")
        expected_num_obs = (
            cfg.env.num_single_observations * cfg.env.history_length
        )
        if cfg.env.num_observations != expected_num_obs:
            raise ValueError(
                "cfg.env.num_observations must equal "
                "num_single_observations * history_length"
            )

        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        single_noise = torch.zeros(
            cfg.env.num_single_observations,
            dtype=self.obs_buf.dtype,
            device=self.device,
        )
        single_noise[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        single_noise[3:6] = noise_scales.gravity * noise_level
        single_noise[9:21] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        single_noise[21:33] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        self.single_obs_noise_scale = single_noise
        return single_noise.repeat(cfg.env.history_length)

    def post_physics_step(self):
        # Foot rewards need the rigid-body state from the current physics step.
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super().post_physics_step()

    def _post_physics_step_callback(self):
        """Update contact events and swing statistics for one policy step.

        These are the event-based foot terms from the reference Go2 policy.
        They do not prescribe a gait phase: every foot is evaluated only when
        it actually touches down.
        """
        super()._post_physics_step_callback()

        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        self.foot_contact_filt = torch.logical_or(contact, self.last_contacts)
        self.first_foot_contact = (
            self.feet_air_time > 0.0
        ) & self.foot_contact_filt
        self.feet_air_time += self.dt

        foot_clearance = (
            self.rigid_state[:, self.feet_indices, 2]
            - self._get_foot_terrain_heights()
            - self.cfg.rewards.foot_radius
        )
        self.swing_peak = torch.maximum(self.swing_peak, foot_clearance)
        self.last_contacts = contact

    def _resample_commands(self, env_ids):
        """Use the parent sampler, with configurable complete stand intervals."""
        super()._resample_commands(env_ids)
        zero_probability = self.cfg.commands.zero_command_probability
        if not 0.0 <= zero_probability <= 1.0:
            raise ValueError("commands.zero_command_probability must be in [0, 1]")
        if zero_probability == 0.0 or len(env_ids) == 0:
            return

        zero_mask = torch.rand(len(env_ids), device=self.device) < zero_probability
        self.commands[env_ids[zero_mask], :3] = 0.0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0
        self.history_needs_init[env_ids] = True
        self.last_contacts[env_ids] = False
        self.foot_contact_filt[env_ids] = False
        self.first_foot_contact[env_ids] = False
        self.swing_peak[env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )
        self.obs_history = torch.zeros(
            self.num_envs,
            self.cfg.env.history_length,
            self.cfg.env.num_single_observations,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.history_needs_init = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        num_feet = len(self.feet_indices)
        self.foot_contact_filt = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device
        )
        self.first_foot_contact = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device
        )
        self.swing_peak = torch.zeros(
            self.num_envs, num_feet, dtype=torch.float, device=self.device
        )
        self.pose_weights = torch.tensor(
            [1.0, 1.0, 0.1] * (self.num_dof // 3),
            dtype=torch.float,
            device=self.device,
        )

    def _get_foot_terrain_heights(self):
        """Return terrain height directly under each foot in world coordinates."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs, len(self.feet_indices), device=self.device
            )
        if self.cfg.terrain.mesh_type == "none":
            raise RuntimeError("Foot-height rewards require a terrain mesh")

        foot_xy = self.rigid_state[:, self.feet_indices, :2]
        grid_xy = (
            (foot_xy + self.terrain.cfg.border_size)
            / self.terrain.cfg.horizontal_scale
        ).long()
        px = torch.clip(grid_xy[:, :, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(grid_xy[:, :, 1], 0, self.height_samples.shape[1] - 2)
        heights = torch.stack(
            (
                self.height_samples[px, py],
                self.height_samples[px + 1, py],
                self.height_samples[px, py + 1],
            ),
            dim=-1,
        )
        return torch.min(heights, dim=-1).values * self.terrain.cfg.vertical_scale

    def _reward_foot_slip(self):
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        foot_speed_xy_sq = torch.sum(
            torch.square(self.rigid_state[:, self.feet_indices, 7:9]),
            dim=2,
        )
        moving = torch.norm(self.commands[:, :3], dim=1) > 0.01
        return torch.sum(
            foot_speed_xy_sq * contact.float(), dim=1
        ) * moving

    def _reward_feet_air_time(self):
        """Reward a touchdown after at least 0.10 s in swing, when commanded."""
        moving = torch.norm(self.commands[:, :3], dim=1) > 0.01
        reward = torch.sum(
            (self.feet_air_time - self.cfg.rewards.min_feet_air_time)
            * self.first_foot_contact.float(),
            dim=1,
        ) * moving
        self.feet_air_time *= ~self.foot_contact_filt
        return reward

    def _reward_feet_height(self):
        """Penalize touchdown if the swing peak differs from the 10 cm target."""
        moving = torch.norm(self.commands[:, :3], dim=1) > 0.01
        peak_error = torch.square(
            self.swing_peak / self.cfg.rewards.target_foot_height - 1.0
        )
        reward = torch.sum(
            peak_error * self.first_foot_contact.float(), dim=1
        ) * moving
        self.swing_peak *= ~self.foot_contact_filt
        return reward

    def _reward_feet_clearance(self):
        """Keep moving swing feet near the reference 10 cm clearance."""
        foot_clearance = (
            self.rigid_state[:, self.feet_indices, 2]
            - self._get_foot_terrain_heights()
            - self.cfg.rewards.foot_radius
        )
        foot_speed_xy = torch.linalg.vector_norm(
            self.rigid_state[:, self.feet_indices, 7:9], dim=2
        )
        return torch.sum(
            torch.abs(foot_clearance - self.cfg.rewards.target_foot_height)
            * torch.sqrt(foot_speed_xy),
            dim=1,
        )

    def _reward_pose(self):
        joint_error = torch.square(self.dof_pos - self.default_dof_pos)
        return torch.exp(-torch.sum(joint_error * self.pose_weights, dim=1))

    def _reward_torques(self):
        return torch.sqrt(torch.sum(torch.square(self.torques), dim=1)) + torch.sum(
            torch.abs(self.torques), dim=1
        )

    def _reward_energy(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_action_rate(self):
        """Standard Unitree penalty for abrupt action changes."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_stand_still(self):
        """Bounded pose cost for zero-command intervals.

        A raw summed L1 cost can dominate all positive terms and is then
        erased by only_positive_rewards. This bounded form keeps a useful
        ranking among imperfect standing poses while its optimum remains the
        configured default_dof_pos.
        """
        mean_joint_position_error = torch.mean(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        )
        zero_command = torch.norm(self.commands[:, :3], dim=1) < 0.01
        return (
            self.cfg.rewards.stand_joint_position_weight
            * torch.tanh(
                mean_joint_position_error
                / self.cfg.rewards.stand_joint_position_tolerance
            )
            * zero_command
        )
