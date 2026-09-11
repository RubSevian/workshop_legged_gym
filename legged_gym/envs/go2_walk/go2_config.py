from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2_Walk_Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # Paper observation: angular velocity, projected gravity, command,
        # joint position/velocity and previous action. Keep five-frame history.
        num_single_observations = 45
        history_length = 5
        num_observations = num_single_observations * history_length
        # Critic-only values: true linear velocity, contacts and terrain scan.
        num_privileged_obs = num_observations + 3 + 4 + 187
        num_envs = 4096
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.9
        dynamic_friction = 0.8
        restitution = 0.

        # Height samples are used by terrain-relative rewards, but deliberately
        # are not exposed to the actor: the real robot has no height-map input.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
                             -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2,
                             0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 1
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # smooth slope, rough slope, stairs up, stairs down, discrete obstacles
        terrain_proportions = [0.2, 0.4, 0.15, 0.15, 0.1]
        slope_treshold = 0.75

        

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 1.}    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        feet_name_reward = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        contact_foot = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        flip_visual_attachments = True
        fix_base_link = False
        self_collisions = 1  # 1 disables self collisions

    class sim(LeggedRobotCfg.sim):
        dt=0.005

    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.25
        base_height_target = 0.35
        contact_force_threshold = 5.0
        soft_dof_pos_limit = 0.95
        min_feet_air_time = 0.10
        target_foot_height = 0.10
        # Radius of the Go2's spherical foot collision geometry [m].
        foot_radius = 0.022
        # At a strictly zero command, keep all 12 joints near
        # init_state.default_joint_angles. The pose cost is bounded by this
        # value, so it remains informative with only_positive_rewards.
        stand_joint_position_weight = 2.0
        stand_joint_position_tolerance = 0.15  # mean joint error [rad]
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            # Source: DerSimi/unitree_go2_sim2real walker_policy.  Positive
            # terms define motion; the remaining terms make it economical and
            # symmetric without imposing an explicit phase or gait template.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_pos_limits = -1.0
            pose = 0.5
            termination = -1.0
            stand_still = -1.0
            torques = -0.0002
            action_rate = -0.01
            energy = -0.001
            feet_clearance = -2.0
            feet_height = -0.2
            foot_slip = -0.1
            feet_air_time = 0.1

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.
        heading_command = False
        # Fraction of complete command intervals set to [vx, vy, yaw] = 0.
        # The other intervals retain the unmodified parent sampler distribution.
        zero_command_probability = 0.15
        class ranges(LeggedRobotCfg.commands.ranges):
            # Current task command envelope; sampling remains the standard
            # Legged-Gym implementation in the parent class.
            lin_vel_x = [-1., 1.]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.50, 0.50]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        debug_randomization = False
        debug_randomization_envs = 3
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_base_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

class Go2_Walk_CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_walk'
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 100
