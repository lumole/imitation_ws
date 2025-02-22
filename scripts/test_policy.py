import numpy as np
import torch
from imitation.algorithms import bc
import gymnasium as gym
import os

class PolicyTester:
    def __init__(self, model_path="trained_models/bc_policy_model1"):
        """初始化策略测试器
        Args:
            model_path: BC模型的路径
        """
        # 创建与训练时相同的环境空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(6,)  # aruco(3) + self(3)
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(2,)  # v, w
        )
        
        # 加载模型
        try:
            self.policy = bc.reconstruct_policy(
                policy_path=model_path,
                device="cpu"
            )
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            raise e

    def predict(self, aruco_state, robot_state):
        """预测机器人动作
        Args:
            aruco_state: aruco标签状态 [x, y, theta]
            robot_state: 机器人状态 [x, y, theta]
        Returns:
            action: [v, w] 线速度和角速度
        """
        # 组合状态
        state = np.concatenate([aruco_state, robot_state])
        
        # 预测动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).reshape(1, -1)
            action = self.policy.predict(state_tensor)[0]
        
        return action

def main():
    # 示例使用
    tester = PolicyTester()
    
    # 测试数据
    aruco_state = np.array([1.0, 2.0, 0.5])
    robot_state = np.array([0.0, 0.0, 0.0])
    
    # 预测并输出结果
    action = tester.predict(aruco_state, robot_state)
    
    print(f"输入aruco状态: {aruco_state}")
    print(f"输入机器人状态: {robot_state}")
    print(f"预测动作 [v, w]: {action}")

if __name__ == "__main__":
    main() 