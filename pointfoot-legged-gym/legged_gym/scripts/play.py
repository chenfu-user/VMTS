# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_restitution = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.random_pd_gains = False
    env_cfg.domain_rand.randomize_step_delay = True
    env_cfg.domain_rand.random_motor_strength = True
    env_cfg.domain_rand.max_step_delay_time = 0.015

    # env_cfg.depth.use_camera = True
    # env_cfg.control.damping = {
    #         "abad_L_Joint": 3.5,
    #         "hip_L_Joint": 3.5,
    #         "knee_L_Joint": 3.5,
    #         "foot_L_Joint": 0.0,
    #         "abad_R_Joint": 3.5,
    #         "hip_R_Joint": 3.5,
    #         "knee_R_Joint": 3.5,
    #         "foot_R_Joint": 0.0,
    #     }  # [N*m*s/rad]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    history_length = train_cfg.history_length  # 假设使用过去 5 个时刻的观测

    def get_combined_obs():
        # 将历史观测和当前观测拼接成 2D 输入
        return obs_history.view(-1, env.num_envs, env.num_obs).permute(1, 0, 2).reshape(env.num_envs, -1)
    

    obs,_ = env.get_observations()
    critic_obs = env.get_privileged_observations()
    

    obs_history = obs.unsqueeze(0).repeat(history_length + 1, 1, 1)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if train_cfg.runner.policy_class_name == 'ActorCriticMoe' or train_cfg.runner.policy_class_name == 'ActorCriticTS' or train_cfg.runner.policy_class_name == 'ActorCriticMoeS':
        policy_teacher = ppo_runner.get_teacher_policy(device=env.device)
        if env_cfg.depth.use_camera:
            policy_depth = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
            depth_encoder = ppo_runner.get_depth_encoder(device=env.device)
    elif train_cfg.runner.policy_class_name == 'ActorCriticPie':
        if env_cfg.depth.use_camera:
            policy_depth = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
            depth_encoder = ppo_runner.get_depth_encoder(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 1 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    infos = {}
    if env_cfg.depth.use_camera:
        infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -2] #if ppo_runner.if_depth else None
    infos["expert_labels"] = env.expert_labels.clone().to(ppo_runner.device)
    
    total_velocity_error = 0
    total_height_error = 0
    # total_lifetime = 0
    max_lifetime_per_robot = torch.zeros(env.num_envs, device=env.device)
    lifetime = torch.zeros(env.num_envs, device=env.device)
    num_robots = env.num_envs

    for i in range(10*int(env.max_episode_length)):
        if obs_history is None or obs_history.shape[0] == 0:  
            # 扩展 new_obs 的维度，并复制 self.history_length + 1 次作为初始化
            obs_history = obs.unsqueeze(0).repeat(history_length + 1, 1, 1)
        else:
            # 拼接新观测，并保持 obs_history 的长度不超过 self.history_length + 1
            obs = obs.unsqueeze(0)  # 扩展 new_obs 为 [1, num_envs, num_obs]
            obs_history = torch.cat([obs_history[1:], obs], dim=0)  # 移除最早的观测，拼接新观测

        # 获取组合后的历史观测数据
        combined_obs = get_combined_obs()
        # print(combined_obs.shape)
        # print(infos["depth"])

        # 使用组合后的观测数据传递给 policy 网络
        # actions = policy_teacher(critic_obs.detach())
        if train_cfg.runner.policy_class_name == 'ActorCriticMoe' or train_cfg.runner.policy_class_name == 'ActorCriticTS':
            if env_cfg.depth.use_student:
                if env_cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        depth_latent = depth_encoder(infos["depth"], combined_obs[:,-env.num_propriceptive_obs:])
                    actions = policy_depth(combined_obs.detach(), depth_latent.detach())
                    depth_encoder.detach_hidden_states()
                else:
                    actions = policy(combined_obs.detach())
            else:
                if train_cfg.runner.policy_class_name == 'ActorCriticTS':
                    actions = policy_teacher(critic_obs.detach())
                else:
                    actions = policy_teacher(critic_obs.detach(), combined_obs.detach(), infos["expert_labels"])
        elif train_cfg.runner.policy_class_name == 'ActorCriticMoeS':
            if env_cfg.depth.use_student:
                if env_cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        depth_latent = depth_encoder(infos["depth"], combined_obs[:,-env.num_propriceptive_obs:])
                    actions = policy_depth(combined_obs.detach(), depth_latent.detach(), infos["expert_labels"])
                    depth_encoder.detach_hidden_states()
                else:
                    actions = policy(combined_obs.detach())
            else:
                actions = policy_teacher(critic_obs.detach(), combined_obs.detach(), infos["expert_labels"])
        elif train_cfg.runner.policy_class_name == 'ActorCriticPie':
            if env_cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = depth_encoder(infos["depth"], combined_obs)
                    depth = infos["depth"]
                else:
                    depth_latent = depth_encoder(depth, combined_obs)
                actions = policy_depth(combined_obs.detach(), depth_latent.detach())
                depth_encoder.detach_hidden_states()
        else:
            actions = policy(combined_obs.detach())

        # actions = policy(obs.detach())
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        # 计算速度误差
        velocity_error = torch.abs(env.base_lin_vel - env.commands[:, :3]).sum(dim=1).mean()
        total_velocity_error += velocity_error.item()

        # 计算高度误差
        height_error = torch.abs(env.base_height - 0.66).mean()
        total_height_error += height_error.item()

        # lifetime = (~env.reset_buf).float() * env.dt
        lifetime = torch.where(~env.reset_buf, lifetime + (~env.reset_buf).float() * env.dt, torch.zeros_like(lifetime))
        max_lifetime_per_robot = torch.max(max_lifetime_per_robot, lifetime)
        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'base_height': env.base_height[robot_index].mean().item(),
                    'ref_contact_l':env.feet_height[robot_index, 0].item(),
                    'ref_contact_r':env.feet_height[robot_index, 1].item(),
                    'feet_air_time_l': env.target_feet_height[robot_index, 0].item(),
                    'feet_air_time_r': env.target_feet_height[robot_index, 1].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_ang_vel_x': env.proprioceptive_obs_buf[robot_index, 0].item(),
                    'filtered_ang_vel_x': 0.25*env.filtered_ang_vel[robot_index, 0].item(),
                    'base_ang_vel_y': env.proprioceptive_obs_buf[robot_index, 1].item(),
                    'filtered_ang_vel_y': 0.25*env.filtered_ang_vel[robot_index, 1].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()

            # 计算并打印总的平均误差
            avg_velocity_error = total_velocity_error / stop_state_log
            avg_height_error = total_height_error / stop_state_log
            avg_time = max_lifetime_per_robot.mean()

            print(f"Total Average Velocity Error (up to step {stop_state_log}): {avg_velocity_error:.4f}")
            print(f"Total Average Height Error (up to step {stop_state_log}): {avg_height_error:.4f}")
            print(f"Total Average avg_time (up to step {stop_state_log}): {avg_time:.4f}")

        if  0 < i < stop_rew_log:
            pass
            # if infos["episode"]:
            #     num_episodes = torch.sum(env.reset_buf).item()
            #     if num_episodes>0:
            #         logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
