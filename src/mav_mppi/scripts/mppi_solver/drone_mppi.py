import math
import numpy as np
import torch
import os
from filter.svg_filter import SavGolFilter
from sampling.standard_normal_noise import StandardSamplling


class MPPI():
    """drone MPPI"""
    def __init__(self):
        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float32)

        # MPPI Parameter
        self.n_samples = 1000
        self.n_timestep = 32
        self.dt = 0.01
        self.n_action = 3

        # State : (x,y,z)
        # Assumption : Rotation Fixed
        self.state = torch.zeros((self.n_samples, self.n_timestep, self.n_action), device = self.device)
        self.v_prev = torch.zeros((3), device = self.device)
        self.x_prev = torch.zeros((3), device = self.device)
        
        # Sampling Lib
        self.sample_gen = StandardSamplling(self.n_samples, self.n_timestep, self.n_action, self.device)

        # Sampling Level : Acceleration
        self.u = torch.zeros((self.n_samples, self.n_timestep, self.n_action), device = self.device)
        self.u_prev = torch.zeros((self.n_timestep, self.n_action), device = self.device)

        # Sampling Parameter : Covariance
        self.sigma = torch.eye((self.n_action), device = self.device) *30.0

        self.param_lambda = 0.1
        self.param_gamma = self.param_lambda * (1.0 - (0.9))  # constant parameter of mppi

        self.filter = SavGolFilter(self.n_action)


    def generateNoiseAndSampling(self):
        standard_normal_noise = torch.randn(self.n_samples, self.n_timestep, self.n_action, device=self.device)
        self.sigma_matrix = self.sigma.expand(self.n_samples, self.n_timestep, -1, -1)
        noise = torch.matmul(standard_normal_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise
    
    def predict_trajectory(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_samples, 1, self.n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_samples, 1, self.n_action)
        v = torch.cumsum(samples * dt, dim=1) + qdot0 

        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)

        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0
        return q


    
    def compute_stage_cost(self, trajectory, target):
        trajectory = trajectory[:,:-1, :].clone()

        err_diff = trajectory[:,:,:3] - target[:3]

        cost_diff = torch.pow(err_diff,2).sum(dim=-1).sum(dim=-1)
        cost_err = cost_diff * 100
        stage_cost = cost_err

        return stage_cost
    
    def compute_terminal_cost(self, trajectory, target):
        trajectory = trajectory[:,-1,:].clone()

        err_diff = trajectory[:,:3] - target[:3]

        cost_diff = torch.pow(err_diff,2).sum(dim=-1)

        cost_err = cost_diff * 20
        # print(f"Cost Diff : {cost_diff}")
        return cost_err


    
    def compute_weights(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute weights for each sample in a batch using PyTorch.
        
        Args:
            S (torch.Tensor): Tensor of shape (batch_size,) containing the scores (costs) for each sample.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) containing the computed weights.
        """

        rho = S.min()  # (scalar)
        print("Rho :", rho)

        scaled_S = (-1.0 / self.param_lambda) * (S - rho)  # (batch_size,)
        eta = torch.exp(scaled_S).sum()  # (scalar)

        weights = torch.exp(scaled_S) / eta  # (batch_size,)

        return weights
    
    def apply_constraint(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(
            u,
            min=torch.tensor([-10, -10, -10], device=self.device),
            max=torch.tensor([10, 10, 10], device=self.device)
        )
        return u
    
    def compute_control_input(self):
        target = torch.tensor([1.0, 2.0, 3.4], dtype = torch.float32, device = self.device)
        u = self.u_prev.clone()
        noise = self.generateNoiseAndSampling()
        noise = self.sample_gen.sampling()
        v = noise + u

        trajectory = self.predict_trajectory(v, self.x_prev, self.v_prev, self.dt)


        S = torch.zeros((self.n_samples), device = self.device)
        S += self.compute_stage_cost(trajectory, target)
        S += self.compute_terminal_cost(trajectory, target)

        w = self.compute_weights(S)
        w_expanded = w.view(-1, 1, 1)        
        w_epsilon = torch.sum(w_expanded * noise, dim=0)

        w_epsilon = self.filter.savgol_filter_torch(w_epsilon,window_size=5,polyorder=2)

        u += w_epsilon
        
        self.sample_gen.update_distribution(u, v, w, noise)

        self.u_prev = u.clone()
        self.u = u[0].clone()

        v = self.v_prev.clone() + self.dt * self.u.clone()
        x = self.x_prev.clone() + self.v_prev * self.dt + 0.5 * self.u.clone() * self.dt**2


        
        return x, v
    

    def set_state(self, x, v):
        # x :  x,  y,  z
        # v : xd, yd, zd
        self.x_prev = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.v_prev = torch.tensor(v, dtype=torch.float32, device=self.device)

        # print("Prev x : ", self.x_prev)
        # print("Prev y : ", self.v_prev)
