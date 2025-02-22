import numpy as np
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import gymnasium as gym
import os
import json
import torch
from stable_baselines3 import PPO
from imitation.util.util import save_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

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

# 3. 加载所有JSON文件
def load_all_trajectories(json_dir):
    all_trajectories = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
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
    
    normalization_params = {
        'state_mean': np.mean(state_means, axis=0).tolist(),
        'state_std': np.mean(state_stds, axis=0).tolist()
    }
    
    # 修改为 gail_policy
    save_dir = os.path.abspath("trained_models/gail_policy")
    os.makedirs(save_dir, exist_ok=True)
    
    normalization_params_file = os.path.join(save_dir, 'normalization_params.json')
    with open(normalization_params_file, 'w') as f:
        json.dump(normalization_params, f)
    
    return all_trajectories

def create_env(state_dims):
    """创建自定义环境"""
    class CustomEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf,
                shape=(state_dims,)
            )
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0, 
                shape=(2,)
            )
            self.state = None
            
        def reset(self, **kwargs):
            self.state = np.zeros(state_dims)
            return self.state, {}
            
        def step(self, action):
            # 创建新的状态向量
            new_state = self.state.copy()
            dt = 0.1  # 时间步长
            
            # 只更新机器人的状态（后三个值）
            v = action[0]  # 线速度
            w = action[1]  # 角速度
            
            # 机器人运动学模型
            theta = new_state[5]  # 当前机器人朝向
            new_state[3] += v * np.cos(theta) * dt  # 机器人x坐标
            new_state[4] += v * np.sin(theta) * dt  # 机器人y坐标
            new_state[5] += w * dt                  # 机器人偏航角
            
            self.state = new_state
            reward = 0  # 奖励将由GAIL的判别器生成
            done = False
            return self.state, reward, done, False, {}

    return CustomEnv

def train_gail(trajectories, env_fn):
    """训练GAIL模型"""
    # 创建向量化环境
    venv = make_vec_env(env_fn, n_envs=16)  # 增加并行环境数量
    
    # 创建奖励网络
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    
    # 创建GAIL训练器
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=64,          # 继续减小批量
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=1,   # 减少判别器更新次数！
        venv=venv,
        gen_algo=PPO(
            policy="MlpPolicy",
            env=venv,
            batch_size=32,
            ent_coef=0.1,           # 增加熵系数，鼓励更多探索
            learning_rate=1e-3,      # 增加学习率，让生成器学习更快
            verbose=1
        ),
        reward_net=reward_net,
    )
    
    # 减少总训练步数
    gail_trainer.train(total_timesteps=200000)  # 减少总步数
    
    return gail_trainer

def main():
    try:
        # 1. 设置数据路径
        json_dir = "data_json/json/"
        
        # 2. 加载所有轨迹数据
        print("正在加载轨迹数据...")
        trajectories = load_all_trajectories(json_dir)
        print(f"成功加载 {len(trajectories)} 条轨迹")
        
        # 3. 创建环境并训练GAIL模型
        print("开始训练GAIL模型...")
        env_fn = create_env(state_dims=6)  # 根据您的状态维度修改
        gail_trainer = train_gail(trajectories, env_fn)
        
        # 4. 开始训练
        gail_trainer.train(total_timesteps=200000)
        
        # 5. 训练完成后保存模型
        save_dir = os.path.abspath("trained_models/gail_policy")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存生成器策略
        policy_path = os.path.join(save_dir, "gen_policy")
        save_policy(gail_trainer.gen_algo.policy, policy_path)
        
        print(f"训练完成！模型已保存到 {save_dir}")
        
    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前模型...")
        
        # 保存模型
        save_dir = os.path.abspath("trained_models/gail_policy")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存生成器策略
        policy_path = os.path.join(save_dir, "gen_policy")
        save_policy(gail_trainer.gen_algo.policy, policy_path)
        
        print(f"模型已保存到 {save_dir}")

if __name__ == "__main__":
    main()
