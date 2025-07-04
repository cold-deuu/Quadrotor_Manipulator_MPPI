import torch
# from rclpy.logging import get_logger

class ActionCost:
    def __init__(self, action_weight, gamma, n_horizon, device):
        # self.logger = get_logger("Action_Cost")
        self.device = device

        self.n_horizon = n_horizon

        self.action_weight = action_weight
        self.gamma = gamma


    def compute_action_cost(self, uSample):
        # uSample : (n_samples, n_horizon, n_action)
        cost_action = torch.sum(torch.pow(uSample, 2), dim=2) # : (n_samples, n_horizon) 각 샘플별로 타임스텝의 L2 제곱합
        cost_action = self.action_weight * cost_action 

        # 시간 감쇠 계수(discount factor) gamma 적용 : 더 가까운 미래일 수록 더 중요하게
        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = cost_action * gamma

        cost_action = torch.sum(cost_action, dim=1) # (n_samples,) 각 샘플 별로 cost 하나 씩 부여
        return cost_action
    

    def compute_prev_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=1)
        cost_action = self.action_weight * cost_action

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = cost_action * gamma

        cost_action = torch.sum(cost_action, dim=0)
        return cost_action
    