import numpy as np
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
import gymnasium as gym
import os
import json
import torch
from imitation.util.util import save_policy

# 1. 处理数据
def process_data(json_data):
    states = []
    actions = []
    for step in json_data['data']:
        # 合并aruco和self状态
        state = np.concatenate([
            step['states']['aruco'],
            step['states']['self']
        ])
        states.append(state)
        actions.append(step['actions'])
    
    # 添加数据标准化
    states = np.array(states)
    actions = np.array(actions)
    
    # 计算状态的均值和标准差
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-8  # 添加小值避免除零
    
    # 标准化状态
    states = (states - state_mean) / state_std
    
    return states, actions, (state_mean, state_std)

# 2. 创建轨迹
def create_trajectory(states, actions):
    # 确保状态比动作多一个
    if len(states) != len(actions) + 1:
        # 如果状态和动作数量相等，我们需要去掉最后一个动作
        actions = actions[:-1]
    
    # 为每个动作创建一个空的info字典
    infos = [{} for _ in range(len(actions))]
    
    # 创建一个带有奖励的轨迹
    return TrajectoryWithRew(
        obs=states,  # 所有状态
        acts=actions,  # 动作数量 = 状态数量 - 1
        infos=infos,  # 为每个动作提供一个空字典
        terminal=True,  # 轨迹结束标志
        rews=np.zeros(len(actions))  # 奖励的长度和动作一样
    )

# 3. 训练行为克隆模型
def train_bc(trajectories):
    bc_trainer = bc.BC(
        observation_space=gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(6,)
        ),
        action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(2,)
        ),
        demonstrations=trajectories,
        rng=np.random.default_rng(),
        batch_size=64,  # 增加batch size
        optimizer_kwargs=dict(lr=1e-3),  # 设置学习率
    )
    
    bc_trainer.train(
        n_epochs=100,  # 增加训练轮数
        progress_bar=True
    )
    return bc_trainer

# 4. 加载所有JSON文件
def load_all_trajectories(json_dir):
    all_trajectories = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 保存标准化参数
    state_means = []
    state_stds = []
    
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r') as f:
            json_data = json.load(f)
            states, actions, (state_mean, state_std) = process_data(json_data)
            state_means.append(state_mean)
            state_stds.append(state_std)
            trajectory = create_trajectory(states, actions)
            all_trajectories.append(trajectory)
    
    # 保存标准化参数，用于推理时使用
    normalization_params = {
        'state_mean': np.mean(state_means, axis=0).tolist(),
        'state_std': np.mean(state_stds, axis=0).tolist()
    }
    
    save_dir = "trained_models/bc_policy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f)
    
    return all_trajectories

def main():
    # 1. 设置数据路径
    json_dir = "data_json/json/"
    
    # 2. 加载所有轨迹数据
    print("正在加载轨迹数据...")
    trajectories = load_all_trajectories(json_dir)
    print(f"成功加载 {len(trajectories)} 条轨迹")
    
    # 3. 训练BC模型
    print("开始训练BC模型...")
    bc_trainer = train_bc(trajectories)
    print("训练完成！")
    
    # 4. 保存模型（修改保存方式）
    save_dir = "trained_models/bc_policy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用 save_policy 保存
    save_policy(bc_trainer.policy, save_dir)
    print(f"模型已保存到 {save_dir}")

if __name__ == "__main__":
    main()
