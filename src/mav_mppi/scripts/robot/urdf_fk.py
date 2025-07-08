import numpy as np
import torch
from robot import urdfparser as u2c
from .transformation_matrix import *
from utils.rotation_conversions import xyz_to_se3

class URDFFK:
    def __init__(self, device, urdf_path: str, root_link: str = "base", end_link: str = "j2s6s200_link_7"):
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
        
        self.device = device

        self.robot = u2c.URDFparser(root_link, [end_link])
        self.robot.from_file(urdf_path)
        self.robot._joint_chain_list = self.robot._get_joint_chain(end_link)  # 🔧 리스트 아님

        # base → j2s6s200_link_base 고정 조인트 변환 (URDF에서 정의됨)
        # xyz = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # rpy = torch.tensor([0.0, 0.0, 1.57079632679], dtype=torch.float32)
        # self.mount_transform = make_transform_matrix(xyz, rpy)





    def compute_fk_cpu(self, state : torch.Tensor) -> np.ndarray:

        q_arm = state[3:].clone()
        base_pose = state[:3].clone()
        
        base_se3 = torch.eye((4))
        base_se3[:3,3] = base_pose.clone()

        robot_tf = self.robot.forward_kinematics_cpu(q_arm, base_movement=False)  # 로봇 base → EEF

        
        T_world_to_ee = base_se3 @ robot_tf
        return T_world_to_ee.numpy()



    def compute_fk_gpu(self,
        q_arm: torch.Tensor,           # (N, T, 7)
        # drone_samples: torch.Tensor, # (N, T, 3)
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
    
    def compute_fk_whole_gpu(self,
        qSample: torch.Tensor,          # (N, T, 10)
        base_movement: bool = False
    ) -> torch.Tensor:

        q_base = qSample[:,:,:3].clone()
        q_arm = qSample[:,:,3:].clone()

        se3_base = xyz_to_se3(q_base)
        base_to_arm = self.robot.forward_kinematics(q_arm, base_movement=base_movement) # (N, T, 4, 4)

        print(f"se3_base : {se3_base.shape}")
        print(f"base_to_arm : {base_to_arm.shape}")

        # # 3. 최종 world → EE 변환
        return torch.matmul(se3_base, base_to_arm)
