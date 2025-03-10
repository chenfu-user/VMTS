import torch
import numpy as np

# 创建测试环境和数据
def test_flatten_reshape():
    # 1. 设置参数
    num_transitions_per_env = 4  # 每个环境的转换数
    num_envs = 3                 # 环境数量
    obs_shape = (6,)            # 观测空间维度
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 创建随机观测数据
    observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
    
    # 填充一些随机数据
    for i in range(num_transitions_per_env):
        for j in range(num_envs):
            observations[i,j] = torch.tensor([i+j*0.1]*6, device=device)
    
    print("原始观测数据形状:", observations.shape)
    print("原始数据示例:\n", observations[0:2])  # 打印前两个转换的数据

    # 3. 展平操作
    flattened_obs = observations.flatten(0, 1)
    print("\n展平后的形状:", flattened_obs.shape)
    print("展平后的数据示例:\n", flattened_obs[0:6])  # 打印前6个数据

    # 4. 还原操作
    restored_obs = flattened_obs.reshape(num_transitions_per_env, num_envs, *obs_shape)
    print("\n还原后的形状:", restored_obs.shape)
    print("还原后的数据示例:\n", restored_obs[0:2])  # 打印前两个转换的数据

    # 5. 验证还原是否正确
    is_equal = torch.all(observations == restored_obs)
    print("\n还原是否正确:", is_equal.item())

    # 6. 测试单个观测值的追踪
    test_transition = 1
    test_env = 2
    print(f"\n追踪特定观测值 (transition={test_transition}, env={test_env}):")
    print("原始数据:", observations[test_transition, test_env])
    flattened_index = test_transition * num_envs + test_env
    print("展平后的位置:", flattened_obs[flattened_index])
    print("还原后的数据:", restored_obs[test_transition, test_env])

    return is_equal.item()

if __name__ == "__main__":
    success = test_flatten_reshape()
    print("\n测试结果:", "通过" if success else "失败")
