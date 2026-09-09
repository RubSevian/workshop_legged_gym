from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2_Walk_Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # One frame contains proprioception, commands, the previous action and
        # the gait clock. Five frames give the policy short-term dynamics.
        num_single_observations = 47
        history_length = 5
        num_observations = num_single_observations * history_length
        # Actor history already contains gait sin/cos in every frame. The
        # remaining privileged values are true velocity, contacts and terrain.
        num_privileged_obs = num_observations + 3 + 4 + 187
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
        damping = {'joint': 1.}     # [N*m*s/rad]
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
        base_height_target = 0.32
        cycle_time = 0.5
        gait_transition_margin = 0.2
        command_dead = 0.1
        contact_force_threshold = 5.0
        foot_radius = 0.02
        min_outward_hip_angle = 0.05
        soft_dof_pos_limit = 0.9
        only_positive_rewards = True
        target_foot_height = 0.08
        max_contact_force = 120.
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -1.0
            gait_contact = 0.5
            feet_clearance = -15.0
            foot_drag = -0.5
            foot_slip = -0.1
            hip_inward = -0.05
            collision = -1.0
            termination = -10.0
            torques = -2e-4
            dof_acc = -2.5e-7
            dof_pos_limits = -10.0
            feet_contact_forces = -0.01
            smoothness = -0.02
            stand_still = -0.5

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.8]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        debug_randomization = False
        debug_randomization_envs = 3
        randomize_friction = True
        friction_range = [0.5, 1.15]
        randomize_base_mass = True
        added_base_mass_range = [-2, 2]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.25
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.7, 1.3]

        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]

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
        num_steps_per_env = 60 # per iteration
        max_iterations = 10000
