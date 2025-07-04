import torch

# from rclpy.logging import get_logger

class JointSpaceCost:
    def __init__(self, centering_weight: float, joint_traj_weight: float, gamma: float, n_horizon: int, device):
        # self.logger = get_logger("Joint_Space_Cost")
        self.device = device
        self.n_horizon = n_horizon
        
        self.centering_weight = centering_weight
        self.joint_traj_weight = joint_traj_weight
        self.gamma = gamma

        self.qCenter =torch.tensor([0.0, 0.0, 0.0, (-3.0718-0.0698)/2, 0.0, (3.7525-0.0175)/2, 0.0], device = self.device)


    def compute_centering_cost(self, qSample: torch.Tensor) -> torch.Tensor:
        cost_center = torch.sum(torch.pow(qSample-self.qCenter, 2), dim=2)
        cost_center = self.centering_weight * cost_center

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_center = cost_center * gamma

        cost_center = torch.sum(cost_center, dim=1)
        return cost_center
    

    def compute_jointTraj_cost(self, qSample: torch.Tensor, jointTraj: torch.Tensor) -> torch.Tensor:
        # jointTraj = jointTraj.clone().unsqueeze(0).to(device = self.device)
        cost_tracking = torch.sum(torch.pow(qSample - jointTraj, 2), dim = 2)
        cost_tracking = self.joint_traj_weight * cost_tracking

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_tracking = cost_tracking * gamma

        cost_tracking = torch.sum(cost_tracking, dim=1)
        return cost_tracking
    

    def compute_prev_centering_cost(self, qSample: torch.Tensor) -> torch.Tensor:
        cost_center = torch.sum(torch.pow(qSample - self.qCenter, 2), dim=1)
        cost_center = self.centering_weight * cost_center

        cost_center = cost_center
        cost_center = torch.sum(cost_center, dim=0)
        return cost_center
    
        
    def compute_prev_jointTraj_cost(self, q: torch.Tensor, jointTraj: torch.Tensor) -> torch.Tensor:
        jointTraj = jointTraj.clone().to(device = self.device)
        # cost_tracking = torch.sum(torch.pow(q - jointTraj, 2), dim=1)
        cost_tracking = torch.norm(q - jointTraj, p=2, dim=1)
        cost_tracking = self.joint_traj_weight * cost_tracking

        cost_tracking = torch.sum(cost_tracking, dim=0)
        return cost_tracking
    
    def compute_joint_limit_cost(self, qSample: torch.Tensor) -> torch.Tensor:

        q_lower = torch.tensor([-6.2832, 0.8203, -6.2832, 0.5236, -6.2832, 1.1345, -6.2832], device=self.device)
        q_upper = torch.tensor([6.2832, 5.4629, 6.2832, 5.7596, 6.2832, 5.1487, 6.2832], device=self.device)

        below_limit = qSample < q_lower
        above_limit = qSample > q_upper
        out_of_bounds = below_limit | above_limit  # shape: (n_samples, n_horizon, dof)

        mask = out_of_bounds.any(dim=2)  # shape: (n_samples, n_horizon)

        stepwise_cost = mask.float() * 1e10  # shape: (n_samples, n_horizon)

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        stepwise_cost *= gamma 

        cost = torch.sum(stepwise_cost, dim=1)  # shape: (n_samples,)

        return cost