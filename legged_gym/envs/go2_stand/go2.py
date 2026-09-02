from legged_gym.utils.isaacgym_utils import get_euler_xyz
from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float

import torch

class Go2(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Индексы вычисляются один раз: искать body/DOF handles внутри reward
        # дорого, потому что reward вызывается для каждого policy step.
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
        self.desired_contact_indices = torch.tensor([self.rr_foot_idx, self.rl_foot_idx], dtype=torch.long, device=self.device, requires_grad=False)
        self.undesired_contact_indices = torch.tensor([self.fl_foot_idx, self.fr_foot_idx, self.rr_thigh_idx, self.rl_thigh_idx, self.fl_thigh_idx, self.fr_thigh_idx, self.rr_calf_idx, self.rl_calf_idx, self.fl_calf_idx, self.fr_calf_idx], dtype=torch.long, device=self.device, requires_grad=False)

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_names],
            dtype=torch.long,
            device=self.device,
        )
        self.rl_leg_indices = torch.tensor(
            [self.dof_names.index(f"RL_{part}_joint") for part in ("hip", "thigh", "calf")],
            dtype=torch.long,
            device=self.device,
        )
        self.rr_leg_indices = torch.tensor(
            [self.dof_names.index(f"RR_{part}_joint") for part in ("hip", "thigh", "calf")],
            dtype=torch.long,
            device=self.device,
        )

        # Собственная история только для двух задних лап. Буферы базового класса
        # last_contacts/feet_air_time оставляем нетронутыми: они имеют размер по
        # всем стопам и могут использоваться стандартными reward-функциями.
        rear_shape = (self.num_envs, len(self.desired_contact_indices))
        self.current_rear_contacts = torch.zeros(rear_shape, dtype=torch.bool, device=self.device)
        self.filtered_rear_contacts = torch.zeros(rear_shape, dtype=torch.bool, device=self.device)
        self.last_rear_contacts = torch.zeros(rear_shape, dtype=torch.bool, device=self.device)
        self.rear_contact_changes = torch.zeros(rear_shape, dtype=torch.float, device=self.device)
        self.rear_feet_air_time = torch.zeros(rear_shape, dtype=torch.float, device=self.device)

        self.body_masses = None
        self.body_local_coms = None
        if getattr(self.cfg.rewards.scales, "com_over_support", 0.0) != 0.0:
            # Массы нужны для настоящего mass-weighted CoM. Считываем их только
            # когда reward включён: при тысячах env это заметная работа на CPU.
            body_masses = []
            body_local_coms = []
            for env, actor in zip(self.envs, self.actor_handles):
                properties = self.gym.get_actor_rigid_body_properties(env, actor)
                body_masses.append([prop.mass for prop in properties])
                body_local_coms.append(
                    [[prop.com.x, prop.com.y, prop.com.z] for prop in properties]
                )
            self.body_masses = torch.tensor(body_masses, dtype=torch.float, device=self.device)
            self.body_local_coms = torch.tensor(
                body_local_coms, dtype=torch.float, device=self.device
            )

        # При pitch=-pi/2 продольное горизонтальное направление робота — -body Z,
        # а стандартная body X смотрит вверх. Эту ось используем для команд x/y.
        self.biped_forward_axis = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        if self.cfg.domain_rand.debug_randomization:
            self._debug_domain_randomization()

    def _debug_domain_randomization(self):
        """Print and validate a small sample of Baseline v1 randomization."""
        count = min(self.cfg.domain_rand.debug_randomization_envs, self.num_envs)
        if count < 1:
            return

        probe_actions_scaled = torch.zeros_like(self.dof_pos)
        probe_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        probe_dof_vel = torch.full_like(self.dof_vel, 0.1)
        probe_torques = self._compute_p_control_torques(
            probe_actions_scaled, probe_dof_pos, probe_dof_vel
        )

        print("[go2_stand] Baseline v1 domain-randomization sample:")
        for env_id in range(count):
            friction = getattr(self, "friction_coeffs", None)
            friction_value = (
                float(friction[env_id].item()) if friction is not None else None
            )
            print(
                f"  env={env_id} "
                f"base_mass={self.randomized_body_masses[env_id, 0].item():.4f} "
                f"link_mass_multiplier_mean="
                f"{self.multiplied_link_masses_ratio[env_id].mean().item():.4f} "
                f"base_com_offset={self.added_base_com[env_id].tolist()} "
                f"friction={friction_value} "
                f"kp_multiplier_mean={self.p_gains_multiplier[env_id].mean().item():.4f} "
                f"kd_multiplier_mean={self.d_gains_multiplier[env_id].mean().item():.4f} "
                f"zero_offset_mean={self.motor_zero_offsets[env_id].mean().item():.6f} "
                f"probe_torque_mean={probe_torques[env_id].mean().item():.4f}"
            )

        if count > 1:
            def require_variation(name, values):
                if torch.allclose(values[:count], values[0].expand_as(values[:count])):
                    raise RuntimeError(f"{name} does not vary across debug environments")

            if self.cfg.domain_rand.randomize_friction:
                require_variation("friction", self.friction_coeffs)
            if self.cfg.domain_rand.randomize_base_mass:
                require_variation("base mass", self.added_base_masses)
            if self.cfg.domain_rand.randomize_link_mass:
                require_variation(
                    "link mass multipliers", self.multiplied_link_masses_ratio
                )
            if self.cfg.domain_rand.randomize_base_com:
                require_variation("base CoM offset", self.added_base_com)
            if self.cfg.domain_rand.randomize_pd_gains:
                require_variation("Kp multipliers", self.p_gains_multiplier)
                require_variation("Kd multipliers", self.d_gains_multiplier)
                pd_only_probe = (
                    -self.d_gains.unsqueeze(0)
                    * self.d_gains_multiplier
                    * probe_dof_vel
                )
                require_variation("PD-randomized probe torque", pd_only_probe)
            if self.cfg.domain_rand.randomize_motor_zero_offset:
                require_variation("motor zero offsets", self.motor_zero_offsets)
                zero_only_probe = (
                    self.p_gains.unsqueeze(0) * self.motor_zero_offsets
                    - self.d_gains.unsqueeze(0) * probe_dof_vel
                )
                require_variation("zero-offset probe torque", zero_only_probe)

            randomizes_controller = (
                self.cfg.domain_rand.randomize_pd_gains
                or self.cfg.domain_rand.randomize_motor_zero_offset
            )
            if randomizes_controller and torch.allclose(
                probe_torques[:count], probe_torques[0].expand(count, -1)
            ):
                raise RuntimeError(
                    "PD/zero-offset randomization did not change probe torques"
                )
            print(
                "[go2_stand] controller torque randomization check: PASSED"
            )


    def compute_observations(self):
        """Compute one measurable frame and append it to actor history."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        # Здесь только сигналы, доступные на реальном роботе. Линейная скорость
        # остаётся simulation-only и используется reward/evaluator, но не actor.
        current_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_pos,
                cos_pos,
            ),
            dim=-1,
        )
        if current_obs.shape[1] != self.cfg.env.num_single_observations:
            raise RuntimeError(
                "go2_stand single observation size does not match "
                "cfg.env.num_single_observations"
            )

        # Шум добавляется только свежему измерению. Если шумить готовый стек,
        # старые кадры получали бы новый независимый шум на каждом policy step.
        if self.add_noise:
            current_obs += (
                2 * torch.rand_like(current_obs) - 1
            ) * self.single_obs_noise_scale

        self._append_observation_history(current_obs)

    def _append_observation_history(self, current_obs):
        """Append one frame and expose the flattened oldest-to-newest history."""
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = current_obs

        # После reset повторяем первый валидный кадр K раз. Нулевые кадры были
        # бы искусственным признаком reset, которого не будет на реальном Go2.
        init_ids = self.history_needs_init.nonzero(as_tuple=False).flatten()
        if len(init_ids) > 0:
            self.obs_history[init_ids] = current_obs[init_ids].unsqueeze(1).repeat(
                1, self.cfg.env.history_length, 1
            )
            self.history_needs_init[init_ids] = False

        # Порядок: от самого старого кадра к текущему.
        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        if cfg.env.history_length < 1:
            raise ValueError("cfg.env.history_length must be at least 1")
        if cfg.env.num_single_observations != 47:
            raise ValueError("go2_stand currently defines exactly 47 values per frame")
        expected_num_obs = cfg.env.num_single_observations * cfg.env.history_length
        if cfg.env.num_observations != expected_num_obs:
            raise ValueError(
                "cfg.env.num_observations must equal "
                "num_single_observations * history_length"
            )

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        single_noise = torch.zeros(
            self.cfg.env.num_single_observations,
            dtype=self.obs_buf.dtype,
            device=self.device,
        )
        # Layout одного кадра: ang_vel(3), gravity(3), commands(3),
        # dof_pos(12), dof_vel(12), last_action(12), phase(2) = 47.
        single_noise[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        single_noise[3:6] = noise_scales.gravity * noise_level
        single_noise[6:9] = 0.0
        single_noise[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        single_noise[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        single_noise[33:47] = 0.0
        self.single_obs_noise_scale = single_noise
        # Базовый класс ожидает vector размера полного observation. В нашем
        # compute_observations он не применяется, но сохраняем корректный shape.
        return single_noise.repeat(self.cfg.env.history_length)

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
        # Reward-функции foot_slip/feet_clearance читают rigid_state внутри
        # super().post_physics_step(), поэтому tensor должен быть свежим заранее.
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Базовый класс перезапишет last_actions в конце super(). Сохраняем
        # a_(t-1), чтобы после reward корректно получить историю a_(t-2).
        previous_actions = self.last_actions.clone()
        super().post_physics_step()
        self.last_last_actions[:] = previous_actions
        # reset_idx вызывается внутри super(); нельзя переносить старую историю
        # действий в только что сброшенный episode.
        self.last_last_actions[self.reset_buf.bool()] = 0.0

    def _post_physics_step_callback(self):
        """Prepare shared contact state once before all reward functions."""
        super()._post_physics_step_callback()

        current = (
            self.contact_forces[:, self.desired_contact_indices, 2]
            > self.cfg.rewards.rear_contact_force
        )
        self.current_rear_contacts.copy_(current)
        self.filtered_rear_contacts.copy_(torch.logical_or(current, self.last_rear_contacts))
        self.rear_contact_changes.copy_(
            torch.abs(current.float() - self.last_rear_contacts.float())
        )
        # Reward-функции только читают history; она обновляется ровно один раз,
        # поэтому результат больше не зависит от порядка rewards в config.
        self.last_rear_contacts.copy_(current)

        self.standing_command_mask.copy_(~self._is_locomotion_command())
        self.stance_mask = self._get_gait_phase()
        self.swing_mask = ~self.stance_mask
        self.swing_mask_l = self.swing_mask[:, 0]
        self.swing_mask_r = self.swing_mask[:, 1]
        # joint_pos reward (если его включить) должен видеть reference текущей,
        # а не предыдущей фазы. Callback выполняется до compute_reward().
        self.compute_ref_state()

    def reset_idx(self, env_ids):
        # Save diagnostic accumulators before the base class clears episode
        # state and reward sums.
        episode_steps = self.episode_length_buf[env_ids].float().clone()
        metric_values = {
            name: values[env_ids].clone()
            for name, values in self.baseline_metric_sums.items()
        }
        terminated = (~self.time_out_buf[env_ids]).float().clone()
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            denominator = episode_steps.clamp(min=1.0)
            for name, values in metric_values.items():
                self.extras["episode"][name] = torch.mean(values / denominator)
                self.baseline_metric_sums[name][env_ids] = 0.0
            self.extras["episode"]["episode_length_s"] = (
                torch.mean(episode_steps) * self.dt
            )
            self.extras["episode"]["terminated_count"] = torch.sum(terminated)
            self.extras["episode"]["termination_fraction"] = torch.mean(terminated)
        self.last_last_actions[env_ids] = 0.0
        self.obs_history[env_ids] = 0.0
        self.history_needs_init[env_ids] = True
        # История контактов не должна переходить из старого episode в новый.
        self.current_rear_contacts[env_ids] = False
        self.filtered_rear_contacts[env_ids] = False
        self.last_rear_contacts[env_ids] = False
        self.rear_contact_changes[env_ids] = 0.0
        self.rear_feet_air_time[env_ids] = 0.0

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
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.obs_history = torch.zeros(
            self.num_envs,
            self.cfg.env.history_length,
            self.cfg.env.num_single_observations,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.history_needs_init = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.com_support_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        metric_names = (
            "baseline_total_reward_per_step",
            "front_feet_contact_rate",
            "front_feet_contact_penalty_raw",
            "mean_abs_torque",
            "mean_abs_action",
            "com_support_distance_m",
            "forward_velocity_abs_error_m_s",
            "lateral_velocity_abs_error_m_s",
        )
        self.baseline_metric_sums = {
            name: torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device,
                requires_grad=False,
            )
            for name in metric_names
        }
        

        
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        # Locomotion clock starts only after stand-up. During stand-up actor sees
        # a fixed sin/cos phase (0, 1), not a fictitious alternating gait.
        episode_time = self.episode_length_buf * self.dt
        locomotion_time = torch.clamp(
            episode_time - self.cfg.commands.standup_duration, min=0.0
        )
        return torch.remainder(locomotion_time / cycle_time, 1.0)

    def _get_locomotion_stage_blend(self):
        """Smooth 0->1 transition after the stand-up interval."""
        episode_time = self.episode_length_buf * self.dt
        transition = self.cfg.commands.standup_transition_duration
        if transition <= 0.0:
            return (episode_time >= self.cfg.commands.standup_duration).float()
        return torch.clamp(
            (episode_time - self.cfg.commands.standup_duration) / transition,
            min=0.0,
            max=1.0,
        )

    def _is_locomotion_command(self):
        """Classify commands without mixing m/s and rad/s in one norm."""
        linear = (
            torch.norm(self.commands[:, :2], dim=1)
            > self.cfg.commands.linear_locomotion_threshold
        )
        yaw = (
            torch.abs(self.commands[:, 2])
            > self.cfg.commands.yaw_locomotion_threshold
        )
        return linear | yaw

    def _resample_commands(self, env_ids):
        """Sample commands using the same dead zones as gait-stage logic."""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)

        linear_active = (
            torch.norm(self.commands[env_ids, :2], dim=1)
            > self.cfg.commands.linear_locomotion_threshold
        )
        self.commands[env_ids, :2] *= linear_active.unsqueeze(1)
        if not self.cfg.commands.heading_command:
            yaw_active = (
                torch.abs(self.commands[env_ids, 2])
                > self.cfg.commands.yaw_locomotion_threshold
            )
            self.commands[env_ids, 2] *= yaw_active

    def _get_heading_frame_lin_vel(self):
        """Linear velocity in the horizontal frame of the upright robot.

        The standard base_lin_vel uses the complete base quaternion. At a pitch
        of -90 degrees its local X component is approximately vertical, so it
        cannot represent the user's forward/backward command correctly.
        """
        forward_world = quat_apply(self.base_quat, self.biped_forward_axis)
        forward_xy = forward_world[:, :2]
        forward_norm = torch.norm(forward_xy, dim=1, keepdim=True)
        fallback = torch.zeros_like(forward_xy)
        fallback[:, 0] = 1.0
        forward_xy = torch.where(
            forward_norm > 1e-4,
            forward_xy / forward_norm.clamp(min=1e-4),
            fallback,
        )
        lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=1)
        world_linear_velocity = self.root_states[:, 7:10]
        forward_velocity = torch.sum(world_linear_velocity[:, :2] * forward_xy, dim=1)
        lateral_velocity = torch.sum(world_linear_velocity[:, :2] * lateral_xy, dim=1)
        return torch.stack(
            (forward_velocity, lateral_velocity, world_linear_velocity[:, 2]), dim=1
        )

    def _get_gait_phase(self):
        phase = self._get_phase()  # Фаза цикла [num_envs]
        sin_pos = torch.sin(2 * torch.pi * phase)  # Синус фазы [-1, 1] [num_envs]
        gait_mask = torch.zeros((self.num_envs, len(self.desired_contact_indices)), dtype=torch.bool, device=self.device)  # Маска: True - опора, False - полёт [num_envs, 2]
        gait_mask[:, 0] = sin_pos >= 0  # RR в опоре при sin ≥ 0 (фаза [0, 0.5])
        gait_mask[:, 1] = sin_pos < 0   # RL в опоре при sin < 0 (фаза [0.5, 1.0])
        gait_mask[torch.abs(sin_pos) < self.cfg.rewards.bias] = 1 # Двойная опора при |sin| < bias
        # При нулевой команде робот должен стоять на обеих задних лапах, а не
        # продолжать принудительный цикл шагов.
        standing = ~self._is_locomotion_command()
        gait_mask[standing] = True
        return gait_mask  # [num_envs, 2]
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # print(sin_pos)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        # Reference содержит абсолютные joint positions. Для суставов, которым
        # не задаём траекторию, правильный target — default, а не ноль.
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        scale_1 = 2
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.rl_leg_indices[0]] = 0
        self.ref_dof_pos[:, self.rl_leg_indices[1]] = 1.57
        self.ref_dof_pos[:, self.rl_leg_indices[2]] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.rr_leg_indices[0]] = 0
        self.ref_dof_pos[:, self.rr_leg_indices[1]] = 1.57
        self.ref_dof_pos[:, self.rr_leg_indices[2]] = -sin_pos_r * scale_1
        # Double support phase
        double_support = torch.abs(sin_pos) < self.cfg.rewards.bias
        self.ref_dof_pos[double_support] = self.default_dof_pos

    
    def _reward_tracking_pitch(self):
        # TODO baseline ablation: replace Euler pitch/roll reward with a
        # quaternion/projected-gravity orientation reward.
        base_quat = self.root_states[:, 3:7]
        euler = get_euler_xyz(base_quat)
        episode_time_buf = self.episode_length_buf * self.dt
        pitch_command = episode_time_buf * self.cfg.commands.pitch / self.cfg.commands.standup_duration
        # Target pitch отрицательный. min/max должны быть упорядочены, иначе
        # torch.clamp(input, 0, -1.57) сразу возвращает -1.57 без stand-up ramp.
        pitch_command = torch.clamp(
            pitch_command,
            min=min(0.0, self.cfg.commands.pitch),
            max=max(0.0, self.cfg.commands.pitch),
        )
        error = torch.square(pitch_command - euler[:, 1]) + torch.square(self.cfg.commands.roll - euler[:, 0])
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def compute_reward(self):
        """Compute training reward and accumulate non-shaping debug metrics."""
        super().compute_reward()
        heading_velocity = self._get_heading_frame_lin_vel()
        front_contact = (
            torch.norm(
                self.contact_forces[:, [self.fl_foot_idx, self.fr_foot_idx], :],
                dim=2,
            )
            > self.cfg.rewards.undesired_contact_force
        ).float().mean(dim=1)
        self.baseline_metric_sums["baseline_total_reward_per_step"] += self.rew_buf
        self.baseline_metric_sums["front_feet_contact_rate"] += front_contact
        self.baseline_metric_sums["front_feet_contact_penalty_raw"] += (
            -15.0 * front_contact * 2.0
        )
        self.baseline_metric_sums["mean_abs_torque"] += torch.mean(
            torch.abs(self.torques), dim=1
        )
        self.baseline_metric_sums["mean_abs_action"] += torch.mean(
            torch.abs(self.actions), dim=1
        )
        self.baseline_metric_sums["com_support_distance_m"] += self.com_support_distance
        self.baseline_metric_sums["forward_velocity_abs_error_m_s"] += torch.abs(
            self.commands[:, 0] - heading_velocity[:, 0]
        )
        self.baseline_metric_sums["lateral_velocity_abs_error_m_s"] += torch.abs(
            self.commands[:, 1] - heading_velocity[:, 1]
        )

    def _reward_tracking_lin_vel(self):
        # Команды x/y заданы в горизонтальной системе вертикального робота.
        heading_lin_vel = self._get_heading_frame_lin_vel()
        error = torch.sum(
            torch.square(self.commands[:, :2] - heading_lin_vel[:, :2]), dim=1
        )
        reward = torch.exp(-error / self.cfg.rewards.tracking_sigma)
        return reward * self._get_locomotion_stage_blend()

    def _reward_tracking_ang_vel(self):
        # Поворот влево/вправо — вращение вокруг мировой вертикали. Компонента
        # base_ang_vel[:, 2] после pitch=-90 относится уже к другой мировой оси.
        yaw_rate_error = torch.square(self.commands[:, 2] - self.root_states[:, 12])
        reward = torch.exp(-yaw_rate_error / self.cfg.rewards.tracking_sigma)
        return reward * self._get_locomotion_stage_blend()
    
    def _reward_hip_pos(self):
        error = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)  # [num_envs]
    
   
    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = base_height - self.cfg.rewards.base_height_target
        # Отдельная sigma имеет физический смысл в метрах и не связана с
        # шириной velocity/orientation tracking rewards.
        return torch.exp(-torch.square(error / self.cfg.rewards.base_height_sigma))


    def _reward_com_over_support(self):
        """Reward CoM projection inside the support region of contacting rear feet."""
        # rigid_state содержит pose link frame. Настоящий CoM каждого звена
        # смещён на prop.com, поэтому переносим local COM offset в мировой frame.
        body_positions = self.rigid_state[:, :, 0:3]
        body_quaternions = self.rigid_state[:, :, 3:7]
        body_com_positions = body_positions + quat_apply(
            body_quaternions, self.body_local_coms
        )
        total_mass = self.body_masses.sum(dim=1, keepdim=True)
        com = torch.sum(
            body_com_positions * self.body_masses.unsqueeze(2), dim=1
        ) / total_mass

        rear_xy = self.rigid_state[:, self.desired_contact_indices, 0:2]
        rr_xy = rear_xy[:, 0]
        rl_xy = rear_xy[:, 1]
        support_segment = rl_xy - rr_xy
        segment_length_sq = torch.sum(torch.square(support_segment), dim=1).clamp(
            min=1e-8
        )
        segment_parameter = torch.sum(
            (com[:, :2] - rr_xy) * support_segment, dim=1
        ) / segment_length_sq
        segment_parameter = torch.clamp(segment_parameter, 0.0, 1.0)
        closest_on_segment = rr_xy + segment_parameter.unsqueeze(1) * support_segment

        contact_weights = self.current_rear_contacts.float()
        contact_count = torch.sum(contact_weights, dim=1)
        closest_single_foot = torch.sum(
            rear_xy * contact_weights.unsqueeze(2), dim=1
        ) / contact_count.unsqueeze(1).clamp(min=1.0)
        both_in_contact = contact_count == 2
        closest_support_point = torch.where(
            both_in_contact.unsqueeze(1), closest_on_segment, closest_single_foot
        )

        distance_to_support = torch.norm(com[:, :2] - closest_support_point, dim=1)
        self.com_support_distance.copy_(distance_to_support)
        # Стопы имеют ненулевую площадь. Внутри margin reward максимален,
        # снаружи плавно убывает с отдельной sigma в метрах.
        outside_distance = torch.clamp(
            distance_to_support - self.cfg.rewards.com_support_margin, min=0.0
        )
        normalized_error = outside_distance / self.cfg.rewards.com_support_sigma
        reward = torch.exp(-torch.square(normalized_error))
        # В полёте support region не существует, поэтому положительного бонуса нет.
        return reward * (contact_count > 0).float()
    

    def _reward_rear_feet_contact_and_air(self):
        contact = self.current_rear_contacts
        gait_mask = self._get_gait_phase()  # [num_envs, 2]
        scheduled_match = torch.sum(
            contact * gait_mask + (~contact) * (~gait_mask), dim=1
        ).float()
        # During stand-up (and for a standing command) both rear feet should be
        # loaded. Alternating contact is introduced smoothly after stand-up.
        rear_support = torch.sum(contact, dim=1).float()
        locomotion_blend = (
            self._get_locomotion_stage_blend()
            * self._is_locomotion_command().float()
        )
        contact_reward = (
            (1.0 - locomotion_blend) * rear_support
            + locomotion_blend * scheduled_match
        )
        undesired_contact = (
            torch.norm(self.contact_forces[:, self.undesired_contact_indices, :], dim=2)
            > self.cfg.rewards.undesired_contact_force
        )
        undesired_contact_penalty = -15.0 * torch.sum(undesired_contact, dim=1)
        # Каждый нормальный шаг содержит два переключения контакта. Отдельный
        # penalty за любое переключение конфликтовал с gait mask, поэтому его
        # убираем; несвоевременный контакт уже уменьшает match reward.
        return contact_reward + undesired_contact_penalty
    

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
        forward_velocity = self._get_heading_frame_lin_vel()[:, 0]
        absolute_speed = torch.abs(forward_velocity)
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            forward_velocity) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(forward_velocity)
        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return (
            reward
            * (self.commands[:, 0].abs() > self.cfg.commands.linear_locomotion_threshold)
            * self._get_locomotion_stage_blend()
        )
    
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
        # Contact history подготовлена один раз в callback; reward не изменяет
        # shared state и потому не зависит от порядка вызова reward-функций.
        contact_filt = self.filtered_rear_contacts
        first_contact = (self.rear_feet_air_time > 0.0) * contact_filt
        self.rear_feet_air_time += self.dt
        rew_airTime = torch.sum((self.rear_feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.rear_feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = (
            self.contact_forces[:, self.desired_contact_indices, 2]
            > self.cfg.rewards.slip_contact_force
        )
        foot_speed_norm = torch.norm(self.rigid_state[:, self.desired_contact_indices, 7:9], dim=2)
        # Квадрат скорости гладкий около нуля и имеет понятное направление:
        # больше проскальзывание -> больше положительный raw penalty.
        slip_penalty = torch.square(foot_speed_norm) * contact
        return torch.sum(slip_penalty, dim=1)
    def _reward_feet_clearance(self):#鼓励抬脚高度
        """
        Поощряет подъём задних лап (RR, RL) на целевую высоту в фазе свинга для ритмичной походки.
        Использует фиксированную целевую высоту (target_foot_height) и экспоненциальную награду.
        Сбрасывает высоту при контакте и применяется только при движении (lin_vel > 0.1).
        """
        # Текущий terrain = plane: env origin задаёт уровень поверхности.
        # Вычитаем радиус стопы, чтобы reward работал с нижней точкой стопы.
        ground_height = self.env_origins[:, 2].unsqueeze(1)
        feet_height = (
            self.rigid_state[:, self.desired_contact_indices, 2]
            - ground_height
            - self.cfg.rewards.foot_radius
        )
        swing_mask = torch.logical_not(self._get_gait_phase())  # [num_envs, 2]
        error = (feet_height - self.cfg.rewards.target_foot_height) / self.cfg.rewards.clearance_sigma
        # Маску умножаем СНАРУЖИ exp. Иначе в stance exp(0)=1 давал
        # постоянный максимальный бонус независимо от высоты лапы.
        reward_per_foot = torch.exp(-torch.square(error)) * swing_mask.float()
        locomotion_blend = (
            self._get_locomotion_stage_blend()
            * self._is_locomotion_command().float()
        )
        return torch.sum(reward_per_foot, dim=1) * locomotion_blend
    
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
