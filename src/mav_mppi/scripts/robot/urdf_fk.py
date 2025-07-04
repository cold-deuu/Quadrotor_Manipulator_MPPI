import numpy as np
import torch
from robot import urdfparser as u2c
from .transformation_matrix import *

class URDFFK:
    def __init__(self, urdf_path: str, root_link: str = "base", end_link: str = "j2s6s200_link_7"):
        """
        URDF 기반 Forward Kinematics 초기화

        Args:
            urdf_path (str): URDF 파일 경로
            root_link (str): 로봇 루트 링크 이름
            end_link (str): 로봇 말단 링크 이름
        """
        self.root_link = root_link
        self.end_link = end_link
        print("ROOT :", root_link)
        print("end_link :", end_link)
        
        self.robot = u2c.URDFparser(root_link, [end_link])
        self.robot.from_file(urdf_path)
        self.robot._joint_chain_list = self.robot._get_joint_chain(end_link)  # 🔧 리스트 아님

        # base → j2s6s200_link_base 고정 조인트 변환 (URDF에서 정의됨)
        # xyz = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # rpy = torch.tensor([0.0, 0.0, 1.57079632679], dtype=torch.float32)
        # self.mount_transform = make_transform_matrix(xyz, rpy)

    def xyzquat_to_matrix(self, xyzquat: torch.Tensor) -> torch.Tensor:
        """
        [x, y, z, qx, qy, qz, qw] → 4x4 변환행렬로 변환

        Args:
            xyzquat (np.ndarray): 위치 + 쿼터니언, shape (7,)
        
        Returns:
            torch.Tensor: shape (4, 4)
        """
        T = torch.eye(4, dtype=torch.float32)
        
        T[:3, 3] = xyzquat[:3].clone()

        x, y, z, w = xyzquat[3:]
        qx, qy, qz, qw = x, y, z, w

        # 쿼터니언을 통한 회전행렬 계산
        R = torch.tensor([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
        ], dtype=torch.float32)

        T[:3, :3] = R
        return T




    def compute_fk_cpu(self, base_pose: torch.Tensor, state : torch.Tensor) -> np.ndarray:
        """
        World frame 기준 말단 링크 위치 계산

        Args:
            q_full (np.ndarray): shape (13,), [0:7]은 드론 pose (xyz+quat), [7:]은 로봇 관절각도

        Returns:
            np.ndarray: shape (4, 4) SE(3) 변환행렬 (EE pose in world frame)
        """
        q_arm = state.clone()
        base_tf = self.xyzquat_to_matrix(base_pose)  # 드론 base → world
        robot_tf = self.robot.forward_kinematics_cpu(q_arm, base_movement=False)  # 로봇 base → EEF

        T_world_to_ee = base_tf @ robot_tf
        return T_world_to_ee.numpy()



    def compute_fk_gpu(self,
        q_arm: torch.Tensor,           # (N, T, 7)
        base_xyzquat: torch.Tensor,      # (7,)
        base_movement: bool = False
    ) -> torch.Tensor:
        """
        병렬 GPU FK 계산: 드론 pose와 관절 궤적을 기반으로 EE 위치 계산

        Args:
            q_arm (torch.Tensor): (N, T, 7) - 로봇 관절 trajectory
            base_xyzquat (np.ndarray): (7,) - 드론 pose (xyz + quat)
            base_movement (bool): 현재 사용하지 않음

        Returns:
            torch.Tensor: (N, T, 4, 4) - EE pose in world frame
        """
        device = q_arm.device
        N, T, _ = q_arm.shape

        # 1. 드론 base pose → torch 변환행렬 (4,4)
        base_tf = self.xyzquat_to_matrix(base_xyzquat).to(device)  # (4, 4)

        # 2. (N, T, 4, 4)로 확장
        base_tf = base_tf.unsqueeze(0).unsqueeze(0).expand(N, T, 4, 4)

        # 3. manipulator FK (GPU 병렬)
        robot_tf = self.robot.forward_kinematics(q_arm, base_movement=base_movement)  # (N, T, 4, 4)

        # 4. 최종 world → EE 변환
        return base_tf @ robot_tf
