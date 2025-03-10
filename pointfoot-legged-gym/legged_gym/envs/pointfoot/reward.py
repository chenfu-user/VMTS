    def _compute_feet_states(self):
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.last_feet_air_time = self.feet_air_time * self.first_contact + self.last_feet_air_time * ~self.first_contact
        # 当脚抬起时，更新 last_contact_time
        self.last_contact_time = self.contact_time * self.first_air + self.last_contact_time * ~self.first_air

        self.feet_air_time *= ~self.contact_filt
        self.contact_time *= self.contact_filt
        if self._include_feet_height_rewards:
            self.last_max_feet_height = self.current_max_feet_height * self.first_contact + self.last_max_feet_height * ~self.first_contact
            self.current_max_feet_height *= ~self.contact_filt
            self.feet_height = self.feet_state[:, :, 2] - self._get_heights_below_foot()
            self.current_max_feet_height = torch.max(self.current_max_feet_height,
                                                     self.feet_height)

        if self._include_knee_height_rewards:    
            self.knee_state = self.rigid_body_states[:, self.knee_indices, :]
            self.last_max_knee_height = self.current_max_knee_height * self.first_contact + self.last_max_knee_height * ~self.first_contact
            self.current_max_knee_height *= ~self.contact_filt
            # 计算膝关节的高度
            self.knee_height = self.knee_state[:, :, 2] - self._get_heights_below_foot()
            # 更新膝关节的当前最大高度
            self.current_max_knee_height = torch.max(self.current_max_knee_height, self.knee_height)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.first_air = (self.contact_time > 0.) * ~self.contact_filt
        self.feet_air_time += self.dt

        self.contact_time += self.dt

        
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        sin_pos = torch.sin(2 * torch.pi * phase)
        cos2_pos = torch.cos(4*torch.pi*phase)
        self.ref_contact = torch.zeros_like(self.contact_filt)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        cos2_pos_l = cos2_pos.clone()
        cos2_pos_r = cos2_pos.clone()
        max_feet_height=self.cfg.rewards.max_feet_height
        # Reshape terrain_type from [8192,1] to [8192,2]
        terrain_type = self.terrain_type.clone()
        terrain_type = terrain_type.repeat(1, 2)
        # self.target_feet_height[:, 0] = 0.034+torch.max(-self.cfg.rewards.max_feet_height*sin_pos_r, torch.zeros_like(sin_pos_r))
        # self.target_feet_height[:, 1] = 0.034+torch.max(self.cfg.rewards.max_feet_height*sin_pos_l, torch.zeros_like(sin_pos_l))
        self.target_feet_height[:, 0] = torch.max(max_feet_height*(1 - cos2_pos_r)/2, torch.zeros_like(cos2_pos_r))
        self.target_feet_height[:, 1] = torch.max(max_feet_height*(1 - cos2_pos_l)/2, torch.zeros_like(cos2_pos_l))

        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_l[sin_pos_l < 0] = 1
        sin_pos_r[sin_pos_r > 0] = 1
        self.ref_contact[:, 0] = sin_pos_r
        self.ref_contact[:, 1] = sin_pos_l
        self.target_feet_height = self.target_feet_height * ~self.ref_contact + 0.03
        self.ref_contact[torch.abs(sin_pos) < 0.0000001] = 1

        
        
    def quat_to_euler(self, quat):
        """
        将四元数转换为欧拉角 (roll, pitch, yaw)，返回 shape 为 [num_envs, 3] 的 tensor。
        假设输入四元数的顺序为 (w, x, y, z)。
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # 计算 roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # 计算 pitch
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(sinp.clamp(-1, 1))

        # 计算 yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
            # 提取 roll 和 pitch 角度
        roll, pitch, _ = self.quat_to_euler(self.base_quat)

        # 根据 roll 和 pitch 计算惩罚权重
        # roll 和 pitch 越接近 0（水平），惩罚越大
        attitude_penalty_weight = torch.exp(-torch.abs(roll)/0.5 - torch.abs(pitch)/0.5)  # 指数衰减函数
        
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) #* attitude_penalty_weight

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_tracking_orientation(self):
        # Penalize non flat base orientation
        orientation_err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-orientation_err/0.1)
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        # contact = self.contact_forces[:, self.feet_indices, 2] > 2.
        contact = self.contact_filt
        stance_mask = self.ref_contact
        reward = torch.where(contact == stance_mask, 1.0, -1.0)
        return torch.mean(reward, dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # 
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        reward = torch.square(self.base_height - self.cfg.rewards.base_height_target)
        final_reward = torch.where(self.base_height > self.cfg.rewards.base_height_target, reward*1.5, reward)
        return final_reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.min_feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime
    
    def _reward_last_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.cycle_time/2,
                     torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime
    
    def _reward_last_contact_time(self):
        # Reward steps between proper duration
        rew_contact_below_min = torch.sum(
            torch.min(self.contact_time - self.cfg.rewards.cycle_time/2, 
                     torch.zeros_like(self.contact_time)) * self.first_air,
            dim=1)
        rew_contact_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.contact_time,
                      torch.zeros_like(self.contact_time)) * self.first_air,
            dim=1)
        rew_contact_time = rew_contact_below_min + rew_contact_above_max
        return rew_contact_time
    
    def _reward_lin_vel_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_tracking_lin_vel() - self.last_lin_reward)
        detalvel = -torch.sum(torch.abs(self.base_lin_vel-self.lastbase_lin_vel),dim=-1)
        return detalvel/self.dt
        return delta_phi/self.dt

    def _reward_no_fly(self):
        # contacts = self.contact_forces[:, self.feet_indices, 2] > 0.2
        # single_contact = torch.sum(1. * contacts, dim=1) == 1
        # return 1. * single_contact

        left_foot_contact = self.contact_filt[:, 0]  # 左脚接触状态
        right_foot_contact = self.contact_filt[:, 1]  # 右脚接触状态

        # 比较两只脚的接触状态是否相同
        same_state = left_foot_contact == right_foot_contact

        # 如果状态相同，给予惩罚-1.0；如果状态不同，给予1.0
        reward = torch.where(same_state, -1.0, 1.0)

        # 返回每个环境的奖励值
        return reward
    
    def _reward_tracking_base_height(self):
    # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.exp(-torch.square(base_height - self.cfg.rewards.base_height_target)/0.15)

    def _reward_unbalance_feet_air_time(self):
        return torch.var(self.last_feet_air_time, dim=-1)

    def _reward_unbalance_feet_height(self):
        return torch.var(self.last_max_feet_height, dim=-1)
    
    def _reward_unbalance_knee_height(self):
        return torch.var(self.last_max_knee_height, dim=-1)
    
    def _reward_target_feet_height(self):
        reward = torch.max(self.feet_height-self.cfg.rewards.max_feet_height-0.03, \
                  torch.zeros_like(self.feet_height))

        adaptive_reward_feet = torch.mean(reward, dim=1)
       
        return adaptive_reward_feet

        # 计算足端最大高度的惩罚
        reward_feet = torch.abs(self.feet_height - self.target_feet_height)

        adaptive_reward_feet = torch.mean(reward_feet, dim=1)

        return adaptive_reward_feet#*rate_terrain
    
    def _reward_lift_feet_height(self):
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1) + 0.01
        speed_weight = torch.exp(-1/speed)
        feet_height = torch.clamp(self.last_max_feet_height, max=0.13)
        return torch.mean(speed_weight.unsqueeze(1) * feet_height, dim=1)
    
    def _reward_hip_pos(self):
        # print("hip_indices:", self.hip_indices)
        # print("default_dof_pos shape:", self.default_dof_pos.shape)
        # print(self.dof_pos.shape)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize displacement and rotation at zero commands
        reward_lin = torch.abs(self.base_lin_vel[:, :2]) * (torch.abs(self.commands[:, :2]) < 0.1)
        reward_ang = (torch.abs(self.base_ang_vel[:, -1]) * (torch.abs(self.commands[:, 2]) < 0.1)).unsqueeze(dim=-1)
        return torch.sum(torch.cat((reward_lin, reward_ang), dim=-1), dim=-1)
    
    def _reward_feet_vel_still(self):
        # Penalize displacement and rotation at zero commands
        feet_vel_l = quat_rotate_inverse(self.base_quat, self.feet_state[:,0, 7:10])
        feet_vel_r = quat_rotate_inverse(self.base_quat, self.feet_state[:,1, 7:10])
        # 将左右脚的速度值在 x 和 y 方向分别拼接
        foot_vel_x = torch.stack((feet_vel_l[:, 0], feet_vel_r[:, 0]), dim=1)
        foot_vel_y = torch.stack((feet_vel_l[:, 1], feet_vel_r[:, 1]), dim=1)
        # left_foot_vel = feet_xy_vel[:, 0, :]  # 左脚的 xy 速度，形状 [num_envs, 2]
        # right_foot_vel = feet_xy_vel[:, 1, :]  # 右脚的 xy 速度，形状 [num_envs, 2]     
        reward_l = torch.norm(foot_vel_x, dim=1) * (torch.abs(self.commands[:, :1]) < 0.1).float()
        reward_r = torch.norm(foot_vel_y, dim=1) * (torch.abs(self.commands[:, 1:2]) < 0.1).float()
        return torch.sum(torch.cat((reward_l, reward_r), dim=-1), dim=-1)
    
    def _reward_feet_alignment_x(self):
        # 获取相对于机体坐标系的左右足端位置
        base_position = self.root_states[:, 0:3]
        feet_pos_l = quat_rotate_inverse(self.base_quat, self.feet_state[:, 0, 0:3] - base_position)
        feet_pos_r = quat_rotate_inverse(self.base_quat, self.feet_state[:, 1, 0:3] - base_position)

        # 提取左右足端在 x 方向上的位置
        feet_pos_lx = feet_pos_l[:, 0]
        feet_pos_rx = feet_pos_r[:, 0]

        # 判断左右脚是否在同一侧
        same_direction = torch.sign(feet_pos_lx) * torch.sign(feet_pos_rx) > 0

        # 计算两只脚 x 位置的差值
        feet_x_diff = torch.abs(feet_pos_lx - feet_pos_rx)

        feet_xzeors_diff = torch.abs(feet_pos_lx + feet_pos_rx)
        final_xzeors_reward = feet_xzeors_diff* (torch.abs(self.base_lin_vel[:, 1]) < 0.1).float()

        # 计算奖励和惩罚
        reward = torch.zeros_like(feet_pos_lx)
        reward[same_direction] = feet_x_diff[same_direction]  # 奖励同侧且位置接近
        reward[~same_direction] = feet_x_diff[~same_direction] * 2  # 惩罚一前一后

        # 获取机器人在 x 方向上的线速度
        linear_velocity_x = self.base_lin_vel[:, 0]#self.base_lin_vel[:, 0]
        velocity_threshold = 0.1 # 速度阈值

        # 判断机器人是否处于低速状态
        low_velocity = torch.abs(linear_velocity_x) < velocity_threshold

        # 在低速情况下应用奖励函数
        final_x_reward = reward * low_velocity.float()

        return final_x_reward #+ final_xzeors_reward
    
    def _reward_feet_alignment_y(self):
        # 获取相对于机体坐标系的左右足端位置
        base_position = self.root_states[:, 0:3]
        feet_pos_l = quat_rotate_inverse(self.base_quat, self.feet_state[:, 0, 0:3] - base_position)
        feet_pos_r = quat_rotate_inverse(self.base_quat, self.feet_state[:, 1, 0:3] - base_position)

        feet_pos_ly = feet_pos_l[:, 1]
        feet_pos_ry = feet_pos_r[:, 1]


        feet_y_diff = torch.abs(feet_pos_ly + feet_pos_ry)
        final_y_reward = feet_y_diff* (torch.abs(self.base_lin_vel[:, 1]) < 0.1).float()


        return final_y_reward
    
    def _reward_above_vel_range(self):
        reward_x = torch.max(self.base_lin_vel[:, 0] - self.command_ranges["lin_vel_x"][1], torch.zeros_like(self.base_lin_vel[:, 0])) +\
                torch.max(-self.base_lin_vel[:, 0] + self.command_ranges["lin_vel_x"][0], torch.zeros_like(self.base_lin_vel[:, 0]))
        
        reward_y = torch.max(self.base_lin_vel[:, 1] - self.command_ranges["lin_vel_y"][1], torch.zeros_like(self.base_lin_vel[:, 1])) +\
                torch.max(-self.base_lin_vel[:, 1] + self.command_ranges["lin_vel_y"][0], torch.zeros_like(self.base_lin_vel[:, 1]))
        
        return reward_x + reward_y

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_distance(self):
        reward = 0
        for i in range(self.feet_state.shape[1] - 1):
            for j in range(i + 1, self.feet_state.shape[1]):
                feet_distance = torch.norm(
                    self.feet_state[:, i, :2] - self.feet_state[:, j, :2], dim=-1
                )
            reward += torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        return (~self.reset_buf).float() * self.dt

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        return term_2
    
    def _reward_feet_y_force(self):
        """
        Penalize y-direction ground reaction forces in local frame when feet are in contact with ground.
        This encourages minimizing lateral forces during stance phase.
        """
        
        # Transform forces to base frame
        forces_local_r = quat_rotate_inverse(self.base_quat, self.contact_forces[:, 0, 0:3])
        forces_local_l = quat_rotate_inverse(self.base_quat, self.contact_forces[:, 1, 0:3])
        
        # Get y-direction forces in local frame
        foot_forces_y_r = forces_local_r[:, 1] # y-direction force in local frame
        foot_forces_y_l = forces_local_l[:, 1] # y-direction force in local frame
        
        # Only consider y-forces when foot is in contact
        y_forces_r = torch.abs(foot_forces_y_r) * self.contact_filt[:, 0].float()
        y_forces_l = torch.abs(foot_forces_y_l) * self.contact_filt[:, 1].float()
        
        return y_forces_r + y_forces_l
    
    def _reward_feet_contact_vel(self):
        """
        Penalize z-direction velocities when feet are close to the ground.
        This encourages smooth landing and prevents high-impact foot strikes.
        """
        
        # Define height threshold for velocity penalty (e.g. 0.1m above ground)
        height_threshold = 0.005 + 0.03
        
        # Get z velocities in world frame
        feet_vel = self.feet_state[:, :, 7:10] # Index 9 is z-velocity
        
        # Only penalize z velocities when feet are below threshold
        close_to_ground = self.feet_height < height_threshold
        penalized_vel = torch.norm(feet_vel, dim=-1) * close_to_ground.float()
        
        # Sum penalties across all feet
        return torch.sum(penalized_vel, dim=1)