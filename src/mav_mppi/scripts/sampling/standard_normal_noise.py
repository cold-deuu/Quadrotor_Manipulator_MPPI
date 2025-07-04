import torch

# from rclpy.logging import get_logger


class StandardSamplling:
    def __init__(self, n_sample : int, n_horizon : int, n_action : int, device):
        # self.logger = get_logger("Standard_Sampling")

        # Torch GPU
        self.device = device

        # Sampling Parameter
        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.n_action = n_action
        self.sigma = torch.eye((self.n_action), device = self.device) * 0.1 # 분산 0.1

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)


    def sampling(self):
        # 표준 정규분포(0, 1)을 따르는 가우시안 noise 생성
        standard_normal_noise = torch.randn(self.n_sample, self.n_horizon, self.n_action, device=self.device)
        # sigma matrix(공분산 행렬) 확장 : (n_sample, n_horizon, n_action, n_action)
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        # 노이즈에 공분산 행렬 적용
        noise = torch.matmul(standard_normal_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise


    def get_sample_joint(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        """샘플링된 가속도를 기반으로 각 샘플 traj의 joint 위치 q를 시뮬레이션하는 함수"""

        # 초기 상태
        # 현재 상태를 샘플 수 만큼 복제
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)  # (n_sample, 1, n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)        # (n_sample, 1, n_action)

        # 속도 계산
        v = torch.cumsum(samples * dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)

        # 이전 속도: [v0, v0+..., ..., v_{N-1}]
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)

        # 누적 위치 계산: q[i] = q[i-1] + v[i-1] * dt + 0.5 * a[i] * dt^2
        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0

        return q


    def get_prev_sample_joint(self, u_prev: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).expand(1, self.n_action)  # (1, n_action)
        q0 = q.unsqueeze(0).expand(1, self.n_action)        # (1, n_action)
        v = torch.cumsum(u_prev * dt, dim=0) + qdot0  # (n_horizon, n_action)

        v_prev = torch.cat([qdot0, v[:-1, :]], dim=0)  # (n_horizon, n_action)

        dq = v_prev * dt + 0.5 * u_prev * dt**2
        q = torch.cumsum(dq, dim=0) + q0
        return q.unsqueeze(0)
    