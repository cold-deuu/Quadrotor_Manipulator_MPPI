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
from mav_mppi.scripts.sampling.standard_normal_noise import StandardSamplling

# Cost
from cost.cost_manager import CostManager

# URDF‐based FK (GPU)
from robot.urdf_fk import URDFFK

# TF Library
from utils.pose import Pose, pose_diff, pos_diff
from utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# Filter : MPPI
from filter.svg_filter import SavGolFilter

class MPPI:
    def __init__(self):
        # Logger 및 device 설정
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[MPPI] Using device: {self.device}")
        torch.set_default_dtype(torch.float32)

        # MPPI 하이퍼파라미터
        self.n_action = 7
        self.n_manipulator_dof = 7
        self.n_mobile_dof = 0
        self.n_samples = 100
        self.n_horizon = 32
        self.dt = 0.01

        # 현재 로봇 상태 (관절 위치 및 속도)
        self._q = torch.zeros(self.n_action, device=self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.q_prev = torch.zeros(self.n_action, device=self.device)
        self.v_prev = torch.zeros(self.n_action, device=self.device)


        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizon, 4, 4), device=self.device)
        self.base_pose = torch.zeros((7),device=self.device)
        # Action
        self.u = torch.zeros((self.n_action), device=self.device)
        self.u_prev = torch.zeros((self.n_horizon, self.n_action), device = self.device)

        # 가속도 노이즈 샘플러
        self.sample_gen = StandardSamplling(
            self.n_samples,
            self.n_horizon,
            self.n_action,
            device=self.device
        )


        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([0.1029, 0.4055, 1.6498])
        self.target_pose.orientation = torch.tensor([ -0.5, -0.5, 0.5, -0.5 ])
        # self.target_pose.orientation = torch.tensor([0, -0.4871745, 0, -0.8733046])
        
        self._lambda = 0.1
        self.cost_manager = CostManager(self.n_samples, self.n_horizon, self.n_action, self._lambda, self.device)


        rospack = rospkg.RosPack()
        root_path = rospack.get_path("aerial_manipulation")
        urdf_path = os.path.join(root_path, "urdf", "aerial_manipulator_gpu.urdf")

        # URDF 기반 GPU FK 초기화
        self.fk_urdf = URDFFK(
            urdf_path,
            root_link="base",
            end_link="j2s7s300_link_7"
        )

        self.svg_filter = SavGolFilter(self.n_action)

        # Log
        self.cnt = 0

    def check_reach(self, q_full):
        # fk_result = self.fk_urdf.compute_fk_cpu(self.base_pose, self._q)

        fk_result = self.fk_urdf.compute_fk_cpu(self.base_pose, self.qdes)
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

        # print(f"Pose Err : {pose_err}")
        # print(f"Ori Err : {target_ori_mat}")

        if pose_err < 0.005:
            return True
        else:
            return False

    def compute_control_input(self):

        # 초기 설정
        u = self.u_prev.clone() # 이전 가속도
        self._qddot = u[0].clone() # 현재의 가속도

        # 샘플링
        noise = self.sample_gen.sampling()
        v = u.unsqueeze(0) + noise
        q_samples = self.sample_gen.get_sample_joint(v, self._q, self._qdot, self.dt)
        
        # Forward Kinematics - 샘플들에 대해 병렬 연산
        trajectory = self.fk_urdf.compute_fk_gpu(q_samples, self.base_pose)

        # cost 계산
        none_joint_trajs = torch.zeros((self.n_samples, self.n_horizon, self.n_action), device=self.device)
        self.cost_manager.update_pose_cost(q_samples, v, trajectory, none_joint_trajs, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sample_gen.sigma_matrix)
        S = self.cost_manager.compute_all_cost() # 최종 cost : 샘플 별로 cost 점수 하나 씩

        # weight 계산 + 부드럽게 필터링
        w = self.compute_weights(S, self._lambda) # (n_samples,)
        w_expanded = w.view(-1, 1,1) # (n_samples, 1, 1) : 아래 noise(n_samples, n_horizon, n_action)와 연산하기 위해서

        # 샘플링된 제어 노이즈들의 softmin-weighted 평균을 구하는 연산
        # w_eps : n_samples 개의 제어 노이즈 trajectory들을, 각자의 cost에 기반한 softmin weight로 평균 낸 값
        w_eps = torch.sum(w_expanded * noise, dim = 0) # w_eps.shape = (n_horizon, n_action)
        w_eps = self.svg_filter.savgol_filter_torch(w_eps,window_size=9,polyorder=2)

        # 제어 입력(가속도) 업데이트
        u+= w_eps
        self.u_prev = u.clone()
        self.u = u[0].clone() # 현재 timestep에서 실행할 가속도 명령

        # 가속도 적분해서 목표 속도와 경로 계산
        self.vdes = self._qdot + self.u * self.dt
        self.qdes = self._q + self._qddot * self.dt + 0.5 * self.u * self.dt * self.dt

        # NUMPY
        qdes_np = self.qdes.to('cpu').numpy()
        vdes_np = self.vdes.to('cpu').numpy()

        # 종료 조건 : 목표에 도달하면
        if self.check_reach(self.base_pose):
            print("Reach !")
            return qdes_np, vdes_np
        
        return qdes_np, vdes_np
    

    
    def compute_weights(self, S: torch.Tensor, _lambda) -> torch.Tensor:
        """
        Compute weights for each sample in a batch using PyTorch.
        
        Args:
            S (torch.Tensor): Tensor of shape (batch_size,) containing the scores (costs) for each sample.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) containing the computed weights.
        """
        # 최소값 계산 (rho)
        rho = S.min()  # (scalar)

        # eta 계산
        scaled_S = (-1.0 / _lambda) * (S - rho) # softmin 구조
        eta = torch.exp(scaled_S).sum()  # (scalar) : 정규화를 위한 전체 합

        # 각 샘플의 weight 계산
        weights = torch.exp(scaled_S) / eta  # 정규화

        return weights


    def update_joint(self, q_full, v_full):
        self._q = torch.tensor(q_full[7:]).to(self.device)
        self._qdot = torch.tensor(v_full[6:]).to(self.device)

        self.base_pose = torch.tensor(q_full[:7]).to(self.device)