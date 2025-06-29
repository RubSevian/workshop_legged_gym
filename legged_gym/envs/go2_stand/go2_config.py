from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 45
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,  # [rad]
            'RL_hip_joint': 0,  # [rad]
            'FR_hip_joint': -0,  # [rad]
            'RR_hip_joint': -0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.3,  # [rad]
            'RL_calf_joint': -1.3,  # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 70.}  # [N*m/rad]
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
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter



    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.75
        base_height_target = 0.85 # Match init_state pos
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 0.8  # Disable for standing task
            tracking_ang_vel = 0.8
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            feet_air_time = 0#1.5
            tracking_pitch = 3  # Increased
            hip_pos = -2.5#-1.0  # Activate to control rear joints
            com_over_support = 0.5#0.5#3.0  # Increased
            feet_contact = 0.8#3.0  # Reduced to balance
            orientation = 0.0
            torques = -1e-6
            dof_vel = -5e-4
            dof_acc = -1e-7 #es2432
            dof_pos_limits = -10.0
            base_height =0# 6.0  # Increased
            action_rate = 0#-0.1
            collision = 0.01
            termination = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            stumble = 0.0
            stand_still = 0.0
            feet_contact_forces = 0 #0.05

    class commands(LeggedRobotCfg.commands):
        pitch = -1.57
        roll = 0.
        standup_duration = 3.
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # Поощряем движение вперёд
            lin_vel_y = [-0.5, 0.5] # Небольшое боковое движение
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        push_interval_s = 2
        max_push_vel_xy = 1.
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_stand'
        max_iterations = 3000
