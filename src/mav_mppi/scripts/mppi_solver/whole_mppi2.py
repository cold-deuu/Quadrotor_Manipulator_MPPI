import os
import torch
import numpy as np
import rospkg  
import math
import yaml
import time
from datetime import datetime

# Sampling Library
from robot import urdfparser as u2c
from sampling.standard_normal_noise import StandardSamplling

# Cost
from cost.cost_manager import CostManager

# URDF‐based FK (GPU)
from robot.urdf_fk import URDFFK

# TF Library
from utils.pose import Pose, pose_diff, pos_diff
from utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# Filter : MPPI
from filter.svg_filter import SavGolFilter

class whole_MPPI:
    def __init__(self, target_pose=None):
        # Torch Arguments
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[MPPI] Using device: {self.device}")
        torch.set_default_dtype(torch.float32)

        # Load URDF Parser with GPU
        rospack = rospkg.RosPack()
        root_path = rospack.get_path("aerial_manipulation")
        urdf_path = os.path.join(root_path, "urdf", "aerial_manipulator_gpu.urdf")

        # URDF 기반 GPU FK 초기화
        self.fk_urdf = URDFFK(
            self.device,
            urdf_path,
            root_link="base",
            end_link="j2s7s300_link_7",
        )

        # MPPI Hyper-Parameter
        self.n_action = 10
        self.n_manipulator_dof = 7
        self.n_mobile_dof = 3
        self.n_samples = 300
        self.n_horizon = 10
        self.dt = 0.01


        # Initial State of Each Iteration
        self.q = torch.zeros(self.n_action, device=self.device)
        self.qdot = torch.zeros(self.n_action, device=self.device)

        # Control Input of MPPI
        self.u_prev = torch.zeros(self.n_horizon, self.n_action, device=self.device)
        self.u = torch.zeros((self.n_action), device=self.device)
        

        # EEF State
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizon, 4, 4), device=self.device)
        self.ee_pose = Pose()

        # Sampling Lib
        self.sample_gen = StandardSamplling(
            self.n_samples,
            self.n_horizon,
            self.n_action,
            device=self.device
        )


        # Target states
        if target_pose is None:
            self.target_pose = Pose()
            self.target_pose.pose = torch.tensor([0.1029, 0.4055, 1.6498])
            self.target_pose.orientation = torch.tensor([-0.5, -0.5, 0.5, -0.5])
        else:
            self.target_pose = target_pose

        # if target_pose_drone is None:
        #     self.target_pose_drone = torch.tensor([1.0, 2.0, 3.4], dtype = torch.float32, device = self.device)
        # else:
        #     self.target_pose_drone = target_pose_drone

        
        self._lambda = 0.01
        self.cost_manager = CostManager(self.n_samples, self.n_horizon, self.n_action, self._lambda, self.device)

        self.svg_filter = SavGolFilter(self.n_action)


    def check_reach(self):

        fk_result = self.fk_urdf.compute_fk_cpu(self.q_des)
        # print(f"fk_result : {fk_result}")

        if isinstance(fk_result, np.ndarray):
            fk_result = torch.tensor(fk_result, dtype=torch.float32)

        self.ee_pose.from_matrix(fk_result)


        pose_err = pos_diff(self.ee_pose, self.target_pose)
        ee_ori_mat = euler_angles_to_matrix(self.ee_pose.rpy, "ZYX")
        target_ori_mat = quaternion_to_matrix(self.target_pose.orientation)

        # target_ori_mat = euler_angles_to_matrix(self.target_pose.rpy, "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_ori_mat), target_ori_mat)
        diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        diff_ori_3d = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        if pose_err < 0.005:
            return True
        else:
            return False
        
    
    def predict_trajectory(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_samples, 1, self.n_mobile_dof)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_samples, 1, self.n_mobile_dof)
        v = torch.cumsum(samples * dt, dim=1) + qdot0 

        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)

        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0
        return q 


    def compute_control_input(self):

        # 초기 설정
        u = self.u_prev.clone() # 이전 가속도

        # 샘플링
        noise = self.sample_gen.sampling()
        v = u.unsqueeze(0) + noise
        q_samples = self.sample_gen.get_sample_joint(v, self.q, self.qdot, self.dt)
        trajectory = self.fk_urdf.compute_fk_whole_gpu(q_samples)
        print(trajectory[:,:,2,3].shape) # n_samp, n_times,

    
        none_joint_trajs = torch.zeros((self.n_samples, self.n_horizon, self.n_action), device=self.device)
        self.cost_manager.update_pose_cost(q_samples, v, trajectory, none_joint_trajs, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sample_gen.sigma_matrix)
        S, _ = self.cost_manager.compute_all_cost()


        # # weight 계산 + 부드럽게 필터링
        w = self.compute_weights(S, self._lambda) # (n_samples,)
        w_expanded = w.view(-1, 1,1) # (n_samples, 1, 1) : 아래 noise(n_samples, n_horizon, n_action)와 연산하기 위해서

        w_eps = torch.sum(w_expanded * noise, dim = 0) # w_eps.shape = (n_horizon, n_action)
        w_eps = self.svg_filter.savgol_filter_torch(w_eps,window_size=5,polyorder=2)

        # # 제어 입력(가속도) 업데이트
        u+= w_eps
        self.u_prev = u.clone()
        self.u = u[0].clone() # 현재 timestep에서 실행할 가속도 명령

        self.qdot_des = self.qdot.clone() + self.u.clone() * self.dt
        self.q_des = self.q.clone() + self.qdot.clone() * self.dt + 0.5 * self.u.clone() * self.dt**2

        # print(f"qdot : {self.qdot}")

        qdes_np = self.q_des.to('cpu').numpy()
        vdes_np = self.qdot_des.to('cpu').numpy()

        # print(f"target : {self.target_pose.pose}")


        # 종료 조건 : 목표에 도달하면
        if self.check_reach():
            print("Reach !")
            return None
        
        return qdes_np, vdes_np
    

    
    def compute_weights(self, S: torch.Tensor, _lambda) -> torch.Tensor:
        # 최소값 계산 (rho)
        rho = S.min()  # (scalar)

        # eta 계산
        scaled_S = (-1.0 / _lambda) * (S - rho) # softmin 구조
        eta = torch.exp(scaled_S).sum()  # (scalar) : 정규화를 위한 전체 합

        # 각 샘플의 weight 계산
        weights = torch.exp(scaled_S) / eta  # 정규화

        return weights


    def update_state(self, q, v):
        # Floating Base Coordinate : [x,y,z, phi, theta, psi]
        # Assumption : phi = theta = psi = 0
        # Control Input : xyz,q1 - q7
        q_mppi = np.delete(q,[3,4,5])
        v_mppi = np.delete(v,[3,4,5])
        # v_mppi[:3] = np.zeros((3))
        self.q = torch.tensor(q_mppi).to(self.device)
        self.qdot = torch.tensor(v_mppi).to(self.device)

        