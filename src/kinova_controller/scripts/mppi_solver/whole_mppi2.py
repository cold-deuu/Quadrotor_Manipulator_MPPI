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
        self.n_samples = 1000
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
            self.target_pose.pose = torch.tensor([0.1029, 1.4055, 1.6498])
            self.target_pose.orientation = torch.tensor([-0.5, -0.5, 0.5, -0.5])
        else:
            self.target_pose = target_pose

        
        self._lambda = 0.01
        self.cost_manager = CostManager(self.n_samples, self.n_horizon, self.n_action, self._lambda, self.device)

        self.svg_filter = SavGolFilter(self.n_action)


        # TEST
        self.q_sensor = None

    def check_reach(self):

        fk_result = self.fk_urdf.compute_fk_cpu(self.q_des)

        # if self.q_sensor is not None:
        #     qsensor=  torch.tensor(self.q_sensor)
        #     fk_result2 = self.fk_urdf.compute_fk_cpu(qsensor)
        #     print(f"oMi Torch :{fk_result2}")

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
        
    
    def warm_start(self, prev_optimal):
        n_timestep, n_dof = prev_optimal.shape
        tmp = torch.zeros((n_timestep, n_dof), device=self.device)
        tmp[:-1,:] = prev_optimal[1:,:].clone()


        print(f"self.uprev = {self.u_prev}")
        print(f"Tmp : {tmp}")
        print(f"UPrev : {prev_optimal}")



        return tmp
        


    def compute_control_input(self):
        
        # u = self.warm_start(self.u_prev)
        # print(f"U : {u}")
        u = self.u_prev.clone() 

        noise = self.sample_gen.sampling()
        v = u.unsqueeze(0) + noise
        # v = self.apply_constraint(v_)
        q_samples = self.sample_gen.get_sample_joint(v, self.q, self.qdot, self.dt)
        # print(f"Noise + u : {q_samples[0,:,:]}")

        trajectory = self.fk_urdf.compute_fk_whole_gpu(q_samples)

        print(f"oMi Traj : {trajectory[:,0,:3,3]}")
    
        none_joint_trajs = torch.zeros((self.n_samples, self.n_horizon, self.n_action), device=self.device)
        self.cost_manager.update_pose_cost(q_samples, v, trajectory, none_joint_trajs, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sample_gen.sigma_matrix)
        S, _ = self.cost_manager.compute_all_cost()
        # print(f"Cost : {S}")
        w = self.compute_weights(S, self._lambda) # (n_samples,)
        w_expanded = w.view(-1, 1,1) # (n_samples, 1, 1)

        w_eps = torch.sum(w_expanded * noise, dim = 0) # w_eps.shape = (n_horizon, n_action)
        # w_eps = self.svg_filter.savgol_filter_torch(w_eps,window_size=3,polyorder=2)

        u+= w_eps
        self.u_prev = u.clone()
        self.u = u[0].clone() 
        # print(f"U0 : {self.u}")
        self.qdot_des = self.qdot.clone() + self.u.clone() * self.dt
        self.q_des = self.q.clone() + self.qdot_des.clone() * self.dt


        qdes_np = self.q_des.to('cpu').numpy()
        vdes_np = self.qdot_des.to('cpu').numpy()
        # print(f"QDES : {qdes_np}")
        # print(f"VDES : {vdes_np}")
        if self.check_reach():
            print("Reach !")
            # return qdes_np, vdes_np
            return None
        
        return qdes_np, vdes_np
    

    
    def compute_weights(self, S: torch.Tensor, _lambda) -> torch.Tensor:
        rho = S.min() 
        # print(f"Min Cost : {rho}")
        scaled_S = (-1.0 / _lambda) * (S - rho) 
        eta = torch.exp(scaled_S).sum() 

        weights = torch.exp(scaled_S) / eta  

        return weights


    def apply_constraint(self, v):
        v = torch.clamp(
        v,
        min=torch.tensor([-3000, -3000, -3000, -100, -100, -100, -100, -100, -100, -100], device=self.device),
        max=torch.tensor([3000, 3000, 3000 , 100, 100, 100, 100, 100,  100, 100], device=self.device)
        )
        return v

    def update_state(self, q, v):
        # Floating Base Coordinate : [x,y,z, phi, theta, psi]
        # Assumption : phi = theta = psi = 0
        # Control Input : xyz,q1 - q7
        q_mppi = np.delete(q,[3,4,5])
        v_mppi = np.delete(v,[3,4,5])
        # v_mppi[:3] = np.zeros((3))
        self.q = torch.tensor(q_mppi).to(self.device)
        self.qdot = torch.tensor(v_mppi).to(self.device)


        # TEST
        self.q_sensor = q_mppi.copy()
        
        