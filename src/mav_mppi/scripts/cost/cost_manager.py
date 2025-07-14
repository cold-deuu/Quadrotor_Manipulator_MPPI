import torch

# 수정필요
from cost.pose_cost import PoseCost
from cost.covar_cost import CovarCost
from cost.action_cost import ActionCost
from cost.joint_space_cost import JointSpaceCost



from utils.pose import Pose

# from rclpy.logging import get_logger

class CostManager:
    def __init__(self, n_sample, n_horizon, n_action, _lambda, device):
        # self.logger = get_logger("Cost_Manager")

        # MPPI Parameter
        self.device = device
        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.n_action = n_action
        self._lambda = _lambda
        self.alpha = 0.1
        self.gamma = 1.0

        ## Weights
        # Pose Cost Weights
        self.stage_pose_weight = 100.0
        self.stage_orientation_weight = 10.0
        self.terminal_pose_weight = 300.0
        self.terminal_orientation_weight = 30.0

        # Covariance Cost Weights
        self.covar_weight = 0.1

        # Action Cost Weights
        self.action_weight = 0.0

        # Joint Space Cost Weights
        self.centering_cost = 100.0
        self.joint_tracking_weight = 1.0

        # Cost Library
        self.pose_cost = PoseCost(self.stage_pose_weight, self.stage_orientation_weight, self.terminal_pose_weight, self.terminal_orientation_weight, self.gamma, self.n_horizon, self.device)
        self.covar_cost = CovarCost(self.covar_weight, self._lambda, self.alpha, self.device)
        self.action_cost = ActionCost(self.action_weight, self.gamma, self.n_horizon, self.device)
        self.joint_cost = JointSpaceCost(self.centering_cost, self.joint_tracking_weight, self.gamma, self.n_horizon, self.device)

        # For Pose Cost
        self.target : Pose
        self.eef_trajectories : torch.Tensor
        self.joint_trajectories : torch.Tensor
        self.qSamples : torch.Tensor
        self.uSamples : torch.Tensor

        # For Covar Cost
        self.u : torch.Tensor
        self.v : torch.Tensor
        self.sigma_matrix : torch.Tensor


    def update_pose_cost(self, qSamples: torch.Tensor, uSamples: torch.Tensor, eef_trajectories: torch.Tensor, joint_trajectories: torch.Tensor,target: Pose):
        self.target = target.clone()
        self.qSamples = qSamples.clone()
        self.uSamples = uSamples.clone()
        self.eef_trajectories = eef_trajectories.clone()
        self.joint_trajectories = joint_trajectories.clone()


    def update_covar_cost(self,  u : torch.Tensor , v : torch.Tensor , sigma_matrix : torch.Tensor):
        self.u = u.clone()
        self.v = v.clone()
        self.sigma_matrix = sigma_matrix.clone()


    def compute_all_cost(self):
        S = torch.zeros((self.n_sample), device=self.device)
        cost_stage = self.pose_cost.compute_stage_cost(self.eef_trajectories, self.target)
        cost_terminal = self.pose_cost.compute_terminal_cost(self.eef_trajectories, self.target)
        # cost_centering = self.joint_cost.compute_centering_cost(self.qSamples[...,3:])
        cost_drone_pose = self.pose_cost.compute_drone_pose_cost(self.qSamples, self.eef_trajectories)
        # cost_joint_limit = self.joint_cost.compute_joint_limit_cost(self.qSamples[..., 3:])

        # print(f"Stage Cost : {cost_stage}")
        # print(f"Terminal Cost : {cost_terminal}")

        S += cost_stage
        S += cost_terminal
        # S += cost_centering
        # S += cost_drone_pose
        # S += cost_joint_limit

        cost_log_dict = {
            "stage_cost": float(cost_stage.mean().cpu()),
            "terminal_cost": float(cost_terminal.mean().cpu()),
        }

        # 전체, 그리고 각 cost 항목도 함께 반환
        return S, cost_log_dict
    