from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2_Walk_Cfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.44] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        flip_visual_attachments = True
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.2, 1.25] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_base_mass = True
    
    class commands( LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        pitch = -1.57
        roll = 0.
        standup_duration = 3.
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class rewards( LeggedRobotCfg.rewards ):
        tracking_sigma = 0.75
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = 0
            ang_vel_xy = 0
            feet_air_time = 0
            tracking_pitch = 2.5  # Increased for stricter vertical torso
            hip_pos = -1.0
            feet_drag = 0
            collision = -5.0  # Strong penalty for undesired contacts
            feet_contact = 4.0  # Increased to strongly reward RL_foot, RR_foot contact
            orientation = -0.5  # Increased to penalize non-vertical orientation
            torques = -0.0002
            dof_pos_limits = 0
            base_height = -5.0 
            # tracking_lin_vel = 0.5
            # tracking_ang_vel = 0.5
            # lin_vel_z = 0
            # ang_vel_xy = 0
            # feet_air_time = 0
            # tracking_pitch = 4  # Increased for stricter vertical torso
            # hip_pos = -1.0
            # feet_drag = 0
            # collision = -3.0  # Strong penalty for undesired contacts
            # feet_contact = 10.0  # Increased to strongly reward RL_foot, RR_foot contact
            # orientation = -1  # Increased to penalize non-vertical orientation
            # torques = -0.0002
            # dof_pos_limits = 0
            # base_height = -10.0
            # dof_vel=0.0
            # dof_acc =0.0
            # action_rate = 0.001
            # termination = 0.0
            # dof_vel_limits = 0.0 
            # torque_limits = 0.0
            # stumble = 0.0
            # stand_still = 0.0
            # feet_contact_forces = 0.0
            

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.83
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.50

class Go2_Walk_CfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_walk'
        max_iterations = 3000 # number of policy updates

  