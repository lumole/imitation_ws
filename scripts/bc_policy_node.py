#!/usr/bin/env python3
import rospy
import numpy as np
import torch
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from stable_baselines3.common.policies import BasePolicy

class BCPolicyNode:
    def __init__(self):
        rospy.init_node('bc_policy_node')
        
        # 加载模型
        model_path = rospy.get_param('~model_path', 'trained_models/bc_policy')
        self.policy = BasePolicy.load(model_path)
        rospy.loginfo("模型加载成功！")
        
        # 状态变量
        self.aruco_state = None
        self.robot_state = None
        
        # 发布器和订阅器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 订阅aruco位姿
        self.aruco_sub = rospy.Subscriber(
            '/aruco_pose',  # 根据实际话题修改
            PoseStamped,
            self.aruco_callback
        )
        
        # 订阅机器人位姿
        self.odom_sub = rospy.Subscriber(
            '/odom',
            Odometry,
            self.odom_callback
        )
        
        # 控制频率（Hz）
        self.rate = rospy.Rate(10)
        
        rospy.loginfo("BC策略节点初始化完成")
    
    def aruco_callback(self, msg):
        # 提取aruco位姿 [x, y, theta]
        self.aruco_state = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            self.quaternion_to_yaw(msg.pose.orientation)
        ])
    
    def odom_callback(self, msg):
        # 提取机器人位姿 [x, y, theta]
        self.robot_state = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self.quaternion_to_yaw(msg.pose.pose.orientation)
        ])
    
    @staticmethod
    def quaternion_to_yaw(q):
        """四元数转yaw角"""
        # 使用arctan2计算yaw角
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return yaw
    
    def get_action(self):
        """使用模型预测动作"""
        if self.aruco_state is None or self.robot_state is None:
            return None
            
        # 组合状态
        state = np.concatenate([self.aruco_state, self.robot_state])
        
        # 预测动作
        with torch.no_grad():
            action = self.policy.predict(state.reshape(1, -1))[0]
        return action
    
    def publish_cmd_vel(self, action):
        """发布速度命令"""
        if action is None:
            return
            
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])  # v
        cmd_vel.angular.z = float(action[1]) # w
        self.cmd_vel_pub.publish(cmd_vel)
    
    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            # 获取动作
            action = self.get_action()
            
            # 发布速度命令
            if action is not None:
                self.publish_cmd_vel(action)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = BCPolicyNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 