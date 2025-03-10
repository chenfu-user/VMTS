from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticTS, ActorCriticMoe, ActorCriticMoeS, ActorCriticEst

import numpy as np
import torch
import copy

class ActorInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ActorInferenceWrapper, self).__init__()
        self.model = model  # 这里使用传入的 model 而不是尝试创建新的属性
    
    def forward(self, obs):
        # 如果输入是一维张量，将其扩展为二维张量
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # 在第0维增加 batch 维度
        # 直接调用 model 的 act_inference 函数
        return self.model.act_inference(obs)
    
class DeActorInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DeActorInferenceWrapper, self).__init__()
        self.model = model  # 这里使用传入的 model 而不是尝试创建新的属性
    
    def forward(self, obs):
        # 如果输入是一维张量，将其扩展为二维张量
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # 在第0维增加 batch 维度
        depth_lat = obs[:,:121]
        stu_obs = obs[:,121:]
        # 直接调用 model 的 act_inference 函数
        return self.model.act_depthlatent_inference(stu_obs, depth_lat)
    
class DepthInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DepthInferenceWrapper, self).__init__()
        self.model = model  # 这里使用传入的 model 而不是尝试创建新的属性
    
    def forward(self, depth, student_obs):
        # 如果输入是一维张量，将其扩展为二维张量
        if student_obs.dim() == 1:
            student_obs = student_obs.unsqueeze(0)  # 在第0维增加 batch 维度
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
            
        return self.model.depth_encoder(depth, student_obs[:,-29:])

def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    # print(train_cfg.runner.load_run)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print('resume_path:', resume_path)
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs*(train_cfg.history_length+1), env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, env_cfg.env.num_propriceptive_obs, env_cfg.env.num_terrain_map_obs, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    # export policy as an onnx file
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    path_depth = os.path.join(path, "policy_depth.onnx")
    path_deact = os.path.join(path, "policy_deact.onnx")
    path = os.path.join(path, "policy.onnx")
    
    model = copy.deepcopy(actor_critic).to("cpu")
    model.eval()

        # 包装 act_inference 函数
    wrapped_model = ActorInferenceWrapper(actor_critic)
    wrapped_model.eval()
    wrapped_model2 = DeActorInferenceWrapper(actor_critic)
    wrapped_model1 = DepthInferenceWrapper(actor_critic)
    wrapped_model2.eval()
    wrapped_model1.eval()

    # 将模型移动到 CPU
    wrapped_model.to("cpu")

    dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs*(train_cfg.history_length+1))
    if env_cfg.depth.use_camera:
        dummy_input1 = torch.randn(58,87)
        dummy_input2 = torch.randn(121 + env_cfg.env.num_propriceptive_obs*(train_cfg.history_length+1))

    input_names = ["nn_input"]
    output_names = ["nn_output"]

    if env_cfg.depth.use_camera:
        torch.onnx.export(
            wrapped_model1,
            (dummy_input1,dummy_input),
            path_depth,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=13,
        )
        torch.onnx.export(
            wrapped_model2,
            dummy_input2,
            path_deact,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=13,
        )
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)


if __name__ == '__main__':
    args = get_args()
    export_policy_as_onnx(args)
