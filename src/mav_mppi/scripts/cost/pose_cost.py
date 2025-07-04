import torch
import torch.nn as nn
import numpy as np

from utils.pose import Pose
from utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion, quaternion_to_matrix

# from rclpy.logging import get_logger

class PoseCost():
    def __init__(self, stage_pose_weight, stage_orientation_weight, terminal_pose_weight, terminal_orientation_weight, gamma, n_horizen, device):
        # self.logger = get_logger("PoseCost")
        self.device = device
        self.n_horizen = n_horizen
        self.gamma = gamma

        self.stage_pose_weight = stage_pose_weight
        self.stage_orientation_weight = stage_orientation_weight
        
        self.terminal_pose_weight = terminal_pose_weight
        self.terminal_orientation_weight = terminal_orientation_weight
        

    def compute_stage_cost(self, eefTraj, target_pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,:-1,0:3,3].clone()
        ee_sample_orientation = eefTraj[:,:-1,0:3,0:3].clone()
        
        diff_pose = ee_sample_pose - target_pose.pose.to(device=self.device)
        # target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(device=self.device), "ZYX")
        target_pose_ori_mat = quaternion_to_matrix(target_pose.orientation.to(device=self.device))

        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_sample_orientation), target_pose_ori_mat)
        diff_orientation = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        # gamma = self.gamma ** torch.arange(self.n_horizen-1, device=self.device)
        # stage_cost = stage_cost * gamma

        stage_cost = torch.sum(stage_cost, dim=1)
        return stage_cost


    def compute_terminal_cost(self, eefTraj, target_pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,-1,0:3,3].clone()
        ee_sample_orientation = eefTraj[:,-1,0:3,0:3].clone()

        diff_pose = ee_sample_pose - target_pose.pose.to(device=self.device)
        # target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(device=self.device), "ZYX")
        target_pose_ori_mat = quaternion_to_matrix(target_pose.orientation.to(device=self.device))

        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_sample_orientation), target_pose_ori_mat)
        diff_orientation = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        # terminal_cost = (self.gamma **self.n_horizen) * terminal_cost

        return terminal_cost
   

    def compute_prev_stage_cost(self, ee, target_pose: Pose) -> torch.Tensor:
        ee_sample_pose = ee[:,0:3,3]
        ee_sample_orientation = ee[:,0:3,0:3]

        diff_pose = ee_sample_pose - target_pose.pose
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - target_pose.rpy

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        gamma = self.gamma ** torch.arange(self.n_horizen)
        stage_cost = stage_cost * gamma

        return stage_cost


    def compute_prev_terminal_cost(self, eefTraj, target_pose: Pose) -> torch.Tensor:
        ee_terminal_pose = eefTraj[-1,0:3,3]
        ee_terminal_orientation = eefTraj[-1,0:3,0:3]

        diff_pose = ee_terminal_pose - target_pose.pose
        diff_orientation = matrix_to_euler_angles(ee_terminal_orientation, "ZYX") - target_pose.rpy

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        terminal_cost = (self.gamma ** self.n_horizen) * terminal_cost

        return terminal_cost
    