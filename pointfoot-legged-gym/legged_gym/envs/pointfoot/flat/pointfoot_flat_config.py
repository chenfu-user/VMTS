from legged_gym.envs import PointFootRoughCfg, PointFootRoughCfgPPO


class PointFootFlatCfg(PointFootRoughCfg):
    class env(PointFootRoughCfg.env):
        num_privileged_obs = 27

    class terrain(PointFootRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights_critic = False

    class asset(PointFootRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(PointFootRoughCfg.rewards):
        max_contact_force = 350.

        class scales(PointFootRoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 5.
            unbalance_feet_air_time = 1.0
            no_fly = 1.
            # feet_contact_forces = -0.01

    class commands(PointFootRoughCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 4.

        class ranges(PointFootRoughCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(PointFootRoughCfg.domain_rand):
        friction_range = [0.,
                          1.5]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.


class PointFootFlatCfgPPO(PointFootRoughCfgPPO):
    class policy(PointFootRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(PointFootRoughCfgPPO.runner):
        experiment_name = 'pointfoot_flat'
        max_iterations = 30000
