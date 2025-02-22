import numpy as np
import torch
from imitation.algorithms import bc

class BCPolicy:
    def __init__(self, model_path):
        """加载训练好的BC模型
        Args:
            model_path: 模型文件路径
        """
        # 直接加载训练好的BC模型
        self.policy = bc.reconstruct_policy(model_path)
    
    def predict(self, aruco_state, robot_state):
        """预测动作
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
            action = self.policy.predict(state.reshape(1, -1))[0]
        
        return action 