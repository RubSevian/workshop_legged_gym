from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 45
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # horizontal_scale = 0.1# [m]
        # vertical_scale = 0.001 # [m]
        # border_size = 25 # [m]
        # curriculum = False
        # static_friction = 0.8
        # dynamic_friction = 0.6
        # restitution = 0.
        # # rough terrain only:
        # measure_heights = False
        # measured_points_x = [-0.20, -0.15, 0., 0.15, 0.20]
        # measured_points_y = [-0.15, 0., 0.15]
        # # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # selected = False # select a unique terrain type and pass all arguments
        # terrain_kwargs = None # Dict of arguments for selected terrain
        # max_init_terrain_level = 1 # starting curriculum state
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows= 10 # number of terrain rows (levels)
        # num_cols = 20 # number of terrain cols (types)
        # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0,1,0,0,0]
        # # trimesh only:
        # slope_treshold = 0 # slopes above this threshold will be corrected to vertical surfaces
        # max_height = 0.05  # Ограничение высоты неровностей
        # curriculum_step = 0.02  # Плавное усложнение
        # success_threshold = 0.8  # Строгий порог успеха

        

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,  # [rad]
            'RL_hip_joint': 0,  # [rad]
            'FR_hip_joint': -0,  # [rad]
            'RR_hip_joint': -0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.1,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.1,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.7,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.7,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        flip_visual_attachments = True
        fix_base_link = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class sim(LeggedRobotCfg.sim):
        dt=0.005

    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.7
        base_height_target = 0.6 # Match init_state pos
        cycle_time = 0.25 #0.25
        bias = 0.2#0.1
        command_dead =0.1
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.5# Disable for standing task
            tracking_ang_vel = 1.5
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            feet_air_time = 0#1.5
            feet_air_time_2 = 0#1.5
            low_speed = 0.005
            joint_pos =2
            foot_slip = 2
            tracking_pitch = 5 # Increased
            rear_feet_contact_and_air = 4#4#1.5#1#1#3
            hip_pos=3#2#0.5#3#-1.5 #-2.#-1.0  # Activate to control rear joints
            com_over_support = 0#2#2#2#3#3#0.5#2#0.5#1.5#0.5#3.0  # Increased
            feet_contact = 0##0.8#3.0  # Reduced to balance
            orientation = 0.0
            torques = -5e-4
            dof_vel = -5e-6
            dof_acc = -2.5e-6 #es2432
            dof_pos_limits = -10#-5.0
            base_height =3#3#3 # Increased
            collision = -0.5#0.0001
            termination = -10#10
            dof_vel_limits = -10
            feet_stumble = -0.0 
            action_rate = 0#-0.01
            smoothness = -0.01
            stand_still = -0.

    class commands(LeggedRobotCfg.commands):
        pitch = -1.57
        roll = 0.
        standup_duration = 1.25
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.25]  # Поощряем движение вперёд
            lin_vel_y = [-0.25, 0.25] # Небольшое боковое движение
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.7, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2, 2]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.

class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_stand'
        num_steps_per_env = 60 # per iteration
        max_iterations = 10000