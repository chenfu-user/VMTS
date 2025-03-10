from legged_gym.envs.base.base_config import BaseConfig

class PointFootRoughCfg(BaseConfig):
    class depth:
        use_camera = False
        use_student = False
        use_warp = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position2 = [0.12043, 0.0322, -0.11186]  # front camera
        position1 = [0.08043, 0.0122, -0.09186]  # front camera
        angle = [1.043778179, 1.083778179]  # positive pitch down

        update_interval = 2  # 5 works without retraining, 8 worse

        original = (87, 58)
        resized = (87, 58)
        horizontal_fov = [86,88]#[86,88]
        buffer_len = 5
        
        near_clip = 0.0
        far_clip = 3.0
        dis_noise = 0.01
        
        scale = 1
        invert = True
        
    class env:
        num_envs = 8192
        # @property
        # def num_envs(self):
        #     if PointFootRoughCfg.depth.use_camera:
        #         return 256
        #     return 8192
        num_propriceptive_obs = 29
        num_terrain_map_obs = 11**2
        num_privileged_obs = num_propriceptive_obs + num_terrain_map_obs  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 6
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

        measure_lin_vel_critic = True
        
        measure_feet_height_critic = True
        measure_base_height_critic = True
        measure_contact_force_critic = False
        measure_contact_filt_critic = True
        measure_friction_coeff_critic = True   
        measure_step_delay = True #False
        measure_restitutions_critic = True#False
        measure_gravityoffset_critic = False
        measure_motor_strength = True
        measure_motor_offset_critic = False
        measure_terrain_type_critic = False
        measure_link_mass_critic = True
        measure_target_height_critic = False
        measure_target_feet_height_critic = False

        if(measure_lin_vel_critic):
            num_privileged_obs += 3
        if(measure_contact_force_critic):
            num_privileged_obs += 3*2
        
        if(measure_contact_filt_critic):
            num_privileged_obs += 2
        if(measure_feet_height_critic):
            num_privileged_obs += 2
        if(measure_base_height_critic):
            num_privileged_obs += 1
        if(measure_friction_coeff_critic):
            num_privileged_obs += 1
        if(measure_step_delay):
            num_privileged_obs += 1
        if(measure_restitutions_critic):
            num_privileged_obs += 1
        if(measure_gravityoffset_critic):
            num_privileged_obs += 3
        if(measure_motor_strength):
            num_privileged_obs += 12
        if(measure_motor_offset_critic):
            num_privileged_obs += 6
        if(measure_terrain_type_critic):
            num_privileged_obs += 1
        if(measure_link_mass_critic):
            num_privileged_obs += 9
        if(measure_target_height_critic):
            num_privileged_obs += 1
        if(measure_target_feet_height_critic):
            num_privileged_obs += 2

        filtered_imu = False
        filtered_alpha = 1.0

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.6
        restitution = 0.8
        # rough terrain only:
        measure_heights_actor = False
        measure_heights_critic = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,
                             0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]#[0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        beta = 1.0

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.66]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 40,
            "hip_L_Joint": 40,
            "knee_L_Joint": 40,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 40,
            "hip_R_Joint": 40,
            "knee_R_Joint": 40,
            "foot_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.0,
            "hip_L_Joint": 2.0,
            "knee_L_Joint": 2.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 2.0,
            "hip_R_Joint": 2.0,
            "knee_R_Joint": 2.0,
            "foot_R_Joint": 0.0,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25#0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        import os
        import sys
        robot_type = os.getenv("ROBOT_TYPE")

        # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
        if not robot_type:
            print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
            sys.exit(1)
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pointfoot/' + robot_type + '/urdf/robot.urdf'
        name = robot_type
        foot_name = 'foot'
        base_name = 'base'
        knee_name = 'knee'
        hip_name = 'hip'
        terminate_after_contacts_on = ["abad", "base"]
        penalize_contacts_on = ["base", "abad", "hip", "knee"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 0.5
        
        randomize_step_delay = False
        max_step_delay_time = 0.015

        randomize_link_mass = False
        randomize_link_rate = [-0.10, 0.10]
        
        random_pd_gains = False
        randomize_pd_rate = [-0.20, 0.20]

        randomize_link_com = False
        rand_link_vec = [0.002,0.002,0.002]

        random_inertia = False
        inertia_range = [0.95, 1.05]

        random_motor_strength = False
        motor_strength_range = [0.8, 1.2]

        random_imu_offset = False
        random_imu_range = [-1/360, 1/360]

        randomize_restitution = False
        restitution_range = [0, 1.0]

        randomize_gravity = False
        gravity_range = [-1.0, 1.0]

        randomize_motor_offset = False
        motor_offset_range = [-0.01, 0.01]

    class rewards:
        class scales:
            action_rate = -0.01
            ang_vel_xy = -0.06#-0.06
            base_height = -10.0
            collision = -30.0
            dof_acc = -2.5e-07
            # dof_pos_limits = -2.0
            # dof_vel = -0.0
            feet_air_time = 60
            feet_contact_forces = -0.01
            feet_stumble = -0.0
            lin_vel_z = -0.5
            no_fly = 1.0
            orientation = -6.0#-6.0
            # stand_still = -1.0
            termination = -0.0
            torque_limits = -0.1
            torques = -2.5e-05
            tracking_ang_vel = 4.0
            tracking_lin_vel = 7.5
            unbalance_feet_air_time = -300.0
            unbalance_feet_height = -60.0
            feet_distance = -100
            survival = 100

            tracking_orientation = 2.5 #奖励姿态跟踪

            tracking_base_height = 1.0

            target_feet_height = -2*3.#-2*3.

            # feet_vel_still = -0.00008

            feet_alignment_x = -3.0

            # feet_alignment_y = -3.0
            hip_pos = -2.# -1.0

            feet_contact_number = 2.5

            feet_contact_vel = -0.5 #惩罚足端离地一定高度的速度

            # above_vel_range = -1.0

            # unbalance_knee_height = -20.0

            lin_vel_pb = 2.5/60 #奖励线速度平滑

            # last_feet_air_time = 30.0
            # last_contact_time = 30.0
            # lift_feet_height = 0.0

            # feet_y_force = -0.1

            # action_smoothness = -0.002

        base_height_target = 0.66
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 200.  # forces above this value are penalized
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        min_feet_distance = 0.15
        min_feet_air_time = 0.25
        max_feet_air_time = 0.65
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        max_feet_height = 0.03#0.03
        
        cycle_time = 0.5

        

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            gravity = 1.0
            height_measurements = 5.0
            
            contact_force = 1.0
            friction_coeff = 1.0
            contact_filt = 1.0
            feet_height = 1.0
            base_height = 1.0   

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        add_pri_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1#0.1
            
            contact_force = 0.
            friction_coeff = 0.
            contact_filt = 0.
            feet_height = 0.0
            base_height = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PointFootRoughCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    history_length = 4

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        empirical_normalization = False
        policy_class_name = 'ActorCritic'#ActorCriticMoe,ActorCriticTS,ActorCriticEst,ActorCritic,ActorCriticMoeS,ActorCriticPie
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 8000  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'pointfoot_rough'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

        use_depth = PointFootRoughCfg.depth.use_camera
        use_student = PointFootRoughCfg.depth.use_student

    class depth_encoder:
        if_depth = PointFootRoughCfg.depth.use_camera
        depth_shape = PointFootRoughCfg.depth.resized
        buffer_len = PointFootRoughCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = PointFootRoughCfg.depth.update_interval * 24
