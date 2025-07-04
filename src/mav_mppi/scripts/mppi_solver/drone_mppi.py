import math
import numpy as np
import torch
import os
from filter.svg_filter import SavGolFilter

class MPPI():
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
    
    # def predict_trajectory(self, u):
    #     f = torch.zeros((self.n_samples, self.n_timestep, 3),device=self.device)
    #     f[:,:,2] = u[:,:,0].clone()
    #     tau = u[:,:,1:].clone()
    #     trajectory = torch.zeros((self.n_samples, self.n_timestep, 6), device = self.device)

    #     angvel = torch.zeros((self.n_samples, self.n_timestep, 3), device = self.device)
    #     transvel = torch.zeros((self.n_samples, self.n_timestep, 3), device = self.device)
    #     angvel[:,0,:] = self.v_prev[3:].clone() + self.dt * torch.einsum('ij, sj -> si', self.I_inv, tau[:,0,:].clone())
    #     J = self.compute_rotational_jacobian(self.x_prev[3:])
    #     rot_mat = self.get_rotation_matrix(self.x_prev[3:])
    #     transvel[:,0,:] = self.v_prev[:3].clone() + self.dt * (self.g + 1/self.m * (torch.einsum('ij, sj -> si',rot_mat,f[:,0,:]) - self.kd * self.v_prev[:3].clone()))
    #     trajectory[:,0,3:] = self.x_prev[3:].clone().unsqueeze(0) + self.dt * J @ self.v_prev[3:]
    #     trajectory[:,0,:3] = self.x_prev[:3].clone() + self.dt * self.v_prev[:3]
    #     for i in range(1, self.n_timestep):
    #         angvel[:,i,:] = angvel[:,i-1,:] + self.dt * torch.einsum('ij, sj -> si',self.I_inv, tau[:,i,:].clone()) 
    #         J = self.compute_rotational_jacobian_gpu(trajectory[:,i-1,3:])
    #         J_inv = torch.linalg.inv(J)
    #         rot_mat = self.get_rotation_matrix_gpu(trajectory[:,i-1, 3:])
    #         trajectory[:,i,3:] = trajectory[:,i-1,3:] + self.dt * torch.einsum("bijk,bk->bij",J_inv,angvel[:,i,:]).squeeze(1)
    #         trajectory[:, i, 3:] = torch.atan2(torch.sin(trajectory[:, i, 3:]), torch.cos(trajectory[:, i, 3:]))
    #         transvel[:,i,:] = (transvel[:,i-1,:].unsqueeze(1) + self.dt * (self.g + 1/self.m *(torch.einsum("bijk,bj->bik",rot_mat, f[:,i,:])- self.kd * transvel[:,i-1,:].unsqueeze(1)))).squeeze(1)
    #         trajectory[:,i,:3] = trajectory[:,i-1,:3].clone() + self.dt * transvel[:,i,:]
        

        
    #     return trajectory


    
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
        v = noise + u

        trajectory = self.predict_trajectory(v, self.x_prev, self.v_prev, self.dt)
        

        S = torch.zeros((self.n_samples), device = self.device)
        S += self.compute_stage_cost(trajectory, target)
        S += self.compute_terminal_cost(trajectory, target)
        # Sigma_inv = torch.linalg.inv(self.sigma_matrix)
        # quad_term = torch.matmul(u.unsqueeze(-2), Sigma_inv @ v.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (batch_size, time_step)

        # S += self.param_gamma * quad_term.sum(dim=1)  # (batch_size,)
        w = self.compute_weights(S)
        w_expanded = w.view(-1, 1, 1)        
        w_epsilon = torch.sum(w_expanded * noise, dim=0)

        w_epsilon = self.filter.savgol_filter_torch(w_epsilon,window_size=5,polyorder=2)

        u += w_epsilon
        # u = self.apply_constraint(u)
        

        self.u_prev = u.clone()
        self.u = u[0].clone()

        v = self.v_prev.clone() + self.dt * self.u.clone()
        x = self.x_prev.clone() + self.v_prev * self.dt + 0.5 * self.u.clone() * self.dt**2

        # best_idx = torch.argmin(S)
        # # print(trajectory[best_idx])

        
        return x, v
    

    def set_state(self, x, v):
        # x :  x,  y,  z
        # v : xd, yd, zd
        self.x_prev = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.v_prev = torch.tensor(v, dtype=torch.float32, device=self.device)

        # print("Prev x : ", self.x_prev)
        # print("Prev y : ", self.v_prev)
