from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

import torch


class Go2_Walk(LeggedRobot):
    """Go2 rough-terrain velocity task with a four-foot trot gait."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Explicit order used by the gait clock: diagonal pairs are FL+RR and
        # FR+RL. Do not rely on the body order in the URDF here.
        foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self.gait_foot_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in foot_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        hip_names = (
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
        )
        self.hip_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_names],
            dtype=torch.long,
            device=self.device,
        )
        # Positive signed angle points away from the center of the body.
        self.hip_outward_sign = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], device=self.device
        )
        self.terrain_center_index = (
            self.cfg.terrain.measured_points_x.index(0.0)
            * len(self.cfg.terrain.measured_points_y)
            + self.cfg.terrain.measured_points_y.index(0.0)
        )

    def compute_observations(self):
        """Append one proprioceptive frame and build critic-only terrain data."""
        phase = self._get_phase()
        phase_sin = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        phase_cos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        current_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                phase_sin,
                phase_cos,
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
                self.contact_forces[:, self.gait_foot_indices, 2]
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
        previous_actions = self.last_actions.clone()
        super().post_physics_step()
        self.last_last_actions[:] = previous_actions
        self.last_last_actions[self.reset_buf.bool()] = 0.0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0.0
        self.gait_phase[env_ids] = 0.0
        self.was_moving[env_ids] = False
        self.obs_history[env_ids] = 0.0
        self.history_needs_init[env_ids] = True

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )
        self.last_last_actions = torch.zeros_like(self.actions)
        # Unlike episode time, this clock is part of the locomotion state. It
        # is reset while standing, so zero commands always expose the same
        # sin/cos pair and cannot drive periodic leg motion.
        self.gait_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.was_moving = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.nominal_foot_xy = torch.tensor(
            list(
                zip(
                    self.cfg.rewards.nominal_foot_x,
                    self.cfg.rewards.nominal_foot_y,
                )
            ),
            dtype=torch.float,
            device=self.device,
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

    def _get_phase(self):
        return self.gait_phase

    def _post_physics_step_callback(self):
        """Run the base callback, then advance only active gait clocks."""
        super()._post_physics_step_callback()
        moving = self._is_moving_command()
        continuing = moving & self.was_moving
        self.gait_phase[~moving] = 0.0
        self.gait_phase[moving & ~self.was_moving] = 0.0
        self.gait_phase[continuing] = torch.remainder(
            self.gait_phase[continuing]
            + self.dt / self.cfg.rewards.cycle_time,
            1.0,
        )
        self.was_moving[:] = moving

    def _resample_commands(self, env_ids):
        """Sample locomotion commands with explicit full stand intervals."""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        # The base sampler removes small linear commands but not small yaw.
        # Use the same yaw dead zone as the moving/standing classifier.
        if not self.cfg.commands.heading_command:
            yaw_active = (
                torch.abs(self.commands[env_ids, 2])
                > self.cfg.commands.yaw_deadzone
            )
            self.commands[env_ids, 2] *= yaw_active

        stand = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.commands.stand_probability
        )
        stand_ids = env_ids[stand]
        self.commands[stand_ids, :3] = 0.0

    def _push_robots(self):
        """Apply training pushes only where they do not conflict with stand."""
        if self.cfg.domain_rand.push_standing:
            push_ids = torch.arange(self.num_envs, device=self.device)
        else:
            push_ids = self._is_moving_command().nonzero(
                as_tuple=False
            ).flatten()
        if len(push_ids) == 0:
            return

        max_velocity = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[push_ids, 7:9] = (
            2.0 * torch.rand(len(push_ids), 2, device=self.device) - 1.0
        ) * max_velocity
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _is_moving_command(self):
        return (
            (torch.norm(self.commands[:, :2], dim=1) > self.cfg.rewards.command_dead)
            | (torch.abs(self.commands[:, 2]) > self.cfg.rewards.command_dead)
        )

    def _get_gait_phase(self):
        """Return desired stance mask for FL, FR, RL, RR diagonal trot."""
        sin_phase = torch.sin(2 * torch.pi * self._get_phase())
        stance = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        # A small overlap around the transitions gives double support and lets
        # individual contacts adapt to uneven terrain.
        overlap = self.cfg.rewards.gait_transition_margin
        stance[:, 0] = sin_phase >= -overlap  # FL
        stance[:, 3] = sin_phase >= -overlap  # RR
        stance[:, 1] = sin_phase <= overlap   # FR
        stance[:, 2] = sin_phase <= overlap   # RL
        stance[~self._is_moving_command()] = True
        return stance

    def _get_swing_weight(self):
        """Continuous 0..1 swing weight for FL, FR, RL, RR."""
        sin_phase = torch.sin(2 * torch.pi * self._get_phase()).unsqueeze(1)
        signed_phase = torch.cat(
            (-sin_phase, sin_phase, sin_phase, -sin_phase), dim=1
        )
        swing_weight = torch.clamp(
            (signed_phase - self.cfg.rewards.gait_transition_margin)
            / (1.0 - self.cfg.rewards.gait_transition_margin),
            min=0.0,
            max=1.0,
        )
        return swing_weight * self._is_moving_command().unsqueeze(1)

    def _get_swing_progress(self):
        """Return per-foot 0..1 progress through its scheduled swing."""
        phase = self._get_phase()
        diagonal_a = torch.clamp(2.0 * (phase - 0.5), min=0.0, max=1.0)
        diagonal_b = torch.clamp(2.0 * phase, min=0.0, max=1.0)
        return torch.stack(
            (diagonal_a, diagonal_b, diagonal_b, diagonal_a), dim=1
        )

    def _get_foot_positions_in_base(self):
        """Return FL, FR, RL, RR foot positions in the base frame."""
        relative_world = (
            self.rigid_state[:, self.gait_foot_indices, :3]
            - self.root_states[:, None, :3]
        )
        base_quat = self.base_quat[:, None, :].expand(-1, 4, -1)
        return quat_rotate_inverse(
            base_quat.reshape(-1, 4), relative_world.reshape(-1, 3)
        ).view(self.num_envs, 4, 3)

    def _get_target_foot_xy(self):
        """Raibert-style touchdown targets in the base frame."""
        nominal = self.nominal_foot_xy.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        desired_velocity = self.commands[:, None, :2].expand(-1, 4, -1).clone()

        # A yaw command produces the tangential foot velocity omega x r.
        yaw_rate = self.commands[:, 2].unsqueeze(1)
        desired_velocity[:, :, 0] -= yaw_rate * nominal[:, :, 1]
        desired_velocity[:, :, 1] += yaw_rate * nominal[:, :, 0]

        velocity_error = (
            self.commands[:, :2] - self.base_lin_vel[:, :2]
        ).unsqueeze(1)
        stance_duration = 0.5 * self.cfg.rewards.cycle_time
        offset = (
            0.5 * stance_duration * desired_velocity
            + self.cfg.rewards.foot_placement_velocity_gain * velocity_error
        )
        offset = torch.clamp(
            offset,
            min=-self.cfg.rewards.foot_placement_max_offset,
            max=self.cfg.rewards.foot_placement_max_offset,
        )
        return nominal + offset

    def _get_foot_terrain_heights(self):
        """Maximum nearby terrain height for each foot."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, 4, device=self.device)

        foot_xy = self.rigid_state[:, self.gait_foot_indices, :2]
        points = (
            foot_xy + self.cfg.terrain.border_size
        ) / self.cfg.terrain.horizontal_scale
        points = points.long()

        neighboring_heights = []
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            px = torch.clamp(
                points[:, :, 0] + dx, 0, self.height_samples.shape[0] - 1
            )
            py = torch.clamp(
                points[:, :, 1] + dy, 0, self.height_samples.shape[1] - 1
            )
            neighboring_heights.append(self.height_samples[px, py])
        return (
            torch.stack(neighboring_heights, dim=-1).max(dim=-1).values
            * self.cfg.terrain.vertical_scale
        )

    # Task-specific additions to standard velocity and stability rewards.
    def _reward_gait_contact(self):
        contact = (
            self.contact_forces[:, self.gait_foot_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        gait_match = torch.mean(
            (contact == self._get_gait_phase()).float(), dim=1
        )
        return gait_match * self._is_moving_command()

    def _reward_feet_clearance(self):
        swing_weight = self._get_swing_weight()
        foot_clearance = (
            self.rigid_state[:, self.gait_foot_indices, 2]
            - self._get_foot_terrain_heights()
            - self.cfg.rewards.foot_radius
        )
        target = self.cfg.rewards.target_foot_height * torch.sqrt(swing_weight)
        return torch.mean(torch.square(foot_clearance - target) * swing_weight, dim=1)

    def _reward_foot_drag(self):
        contact = (
            self.contact_forces[:, self.gait_foot_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        foot_speed_xy = torch.norm(
            self.rigid_state[:, self.gait_foot_indices, 7:9], dim=2
        )
        return torch.mean(
            foot_speed_xy * contact.float() * self._get_swing_weight(), dim=1
        )

    def _reward_foot_slip(self):
        contact = (
            self.contact_forces[:, self.gait_foot_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        foot_speed_xy_sq = torch.sum(
            torch.square(self.rigid_state[:, self.gait_foot_indices, 7:9]),
            dim=2,
        )
        return torch.mean(foot_speed_xy_sq * contact.float(), dim=1)

    def _reward_foot_placement(self):
        """Place each foot near a velocity-aware target before touchdown."""
        foot_xy = self._get_foot_positions_in_base()[:, :, :2]
        target_xy = self._get_target_foot_xy()
        error_sq = torch.sum(torch.square(foot_xy - target_xy), dim=2)
        normalized_error = torch.clamp(
            error_sq / self.cfg.rewards.foot_placement_sigma ** 2,
            min=0.0,
            max=1.0,
        )

        # Early swing remains free for obstacle clearance. The target becomes
        # important near touchdown, when leaving a rear foot behind is harmful.
        touchdown_weight = (
            self._get_swing_weight() * torch.square(self._get_swing_progress())
        )
        return torch.mean(normalized_error * touchdown_weight, dim=1)

    def _reward_rear_foot_extension(self):
        """Soft one-sided limit preventing rear feet trailing too far back."""
        rear_x = self._get_foot_positions_in_base()[:, 2:4, 0]
        rear_limit = (
            self.nominal_foot_xy[2:4, 0]
            - self.cfg.rewards.rear_foot_max_extension
        )
        excess = torch.clamp(rear_limit.unsqueeze(0) - rear_x, min=0.0)
        normalized_excess = torch.clamp(
            torch.square(
                excess / self.cfg.rewards.rear_foot_extension_sigma
            ),
            min=0.0,
            max=1.0,
        )
        return torch.mean(normalized_excess, dim=1) * self._is_moving_command()

    def _reward_hip_inward(self):
        """Penalize only hip motion toward the body center."""
        signed_hip_angle = (
            self.dof_pos[:, self.hip_indices] * self.hip_outward_sign
        )
        inward_violation = torch.clamp(
            self.cfg.rewards.min_outward_hip_angle - signed_hip_angle,
            min=0.0,
        )
        return torch.mean(inward_violation, dim=1)

    def _reward_smoothness(self):
        first_difference = torch.sum(
            torch.square(self.actions - self.last_actions), dim=1
        )
        second_difference = torch.sum(
            torch.square(
                self.actions - 2.0 * self.last_actions + self.last_last_actions
            ),
            dim=1,
        )
        return first_difference + 0.5 * second_difference

    def _reward_base_height(self):
        terrain_height = self.measured_heights[:, self.terrain_center_index]
        return torch.square(
            self.root_states[:, 2]
            - terrain_height
            - self.cfg.rewards.base_height_target
        )

    def _reward_stand_still(self):
        zero_command = ~self._is_moving_command()
        joint_position = torch.mean(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        joint_velocity = torch.mean(torch.square(self.dof_vel), dim=1)
        foot_velocity = torch.mean(
            torch.sum(
                torch.square(
                    self.rigid_state[:, self.gait_foot_indices, 7:10]
                ),
                dim=2,
            ),
            dim=1,
        )
        base_motion = (
            torch.sum(torch.square(self.base_lin_vel), dim=1)
            + 0.5 * torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        action_rate = torch.mean(
            torch.square(self.actions - self.last_actions), dim=1
        )
        contacts = (
            self.contact_forces[:, self.gait_foot_indices, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        missing_feet = torch.mean((~contacts).float(), dim=1)
        return (
            self.cfg.rewards.stand_joint_pos_weight * joint_position
            + self.cfg.rewards.stand_joint_vel_weight * joint_velocity
            + self.cfg.rewards.stand_foot_vel_weight * foot_velocity
            + self.cfg.rewards.stand_base_vel_weight * base_motion
            + self.cfg.rewards.stand_action_rate_weight * action_rate
            + self.cfg.rewards.stand_missing_contact_weight * missing_feet
        ) * zero_command
