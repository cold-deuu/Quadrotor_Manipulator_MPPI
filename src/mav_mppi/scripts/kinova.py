#!/usr/bin/env python3
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

import rospkg
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from mav_mppi.scripts.trajectory.trajManager import jointTraj, SE3Traj

from time import time
from scipy.spatial.transform import Rotation as R

from copy import deepcopy

from cvxopt import matrix, solvers

from robot.urdf_fk import URDFFK
from mppi_solver.mppi import MPPI  # mppi.py 에 정의된 MPPI 클래스
import torch


def pretty_matrix_print(matrix):
    """
    Prints a matrix in a visually appealing format.
    Args:
        matrix (list of list of float): The matrix to print.
    """
    if not matrix or not isinstance(matrix[0], list):
        print("Invalid matrix format. Please provide a 2D list.")
        return

    print("Matrix:")
    for row in matrix:
        formatted_row = " | ".join(f"{value:8.3f}" for value in row)
        print(f"[ {formatted_row} ]")


def quaternion_to_rpy(quaternion):
    """
    Converts a quaternion to roll, pitch, yaw (RPY) using ZYX rotation order.
    Args:
        quaternion (list or np.array): Quaternion [x, y, z, w].
    Returns:
        tuple: Roll, Pitch, Yaw in radians.
    """
    rotation = R.from_quat(quaternion)
    return rotation.as_euler('zyx', degrees=False)


class kinova(RobotWrapper):
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('aerial_manipulation')
        pkg_dir = package_path + '/urdf'
        urdf_path = pkg_dir + '/full_robot_floating2.urdf'  # Pinocchio 전용 URDF

        # Build Pinocchio 모델
        self.robot = self.BuildFromURDF(urdf_path)
        # 데이터 버퍼 생성
        self.data, _, _, = pin.createDatas(
            self.robot.model,
            self.robot.collision_model,
            self.robot.visual_model
        )
        self.model = self.robot.model


class controller:
    def __init__(self):
        rospy.init_node("kinova_controller", anonymous=True)
        rospy.Subscriber("/harrierD7/robot_states", JointState, self.joint_state_callback)

        self.publisher        = rospy.Publisher("/harrierD7/robot_cmd", JointState, queue_size=10)
        self.dronePosePub     = rospy.Publisher("/harrierD7/drone_pose", Float64MultiArray, queue_size=10)

        # Pinocchio 모델
        self.robot = kinova()
        self.q     = None
        self.v     = None
        self.baseSE3 = pin.SE3(1)

        # MPPI 초기화
        self.mppi = MPPI()

        # URDF‐FK (CPU) 초기화
        rospack = rospkg.RosPack()
        urdf_path = rospack.get_path("aerial_manipulation") + "/urdf/aerial_manipulator_gpu.urdf"
        self.fk_urdf = URDFFK(urdf_path, root_link="base", end_link="j2s7s300_link_7")

        # 기존 제어용 Traj
        self.jointTraj    = jointTraj(7)
        self.se3Traj      = SE3Traj()

        self.control_init     = False
        self.jointControlFlag = False
        self.rate             = rospy.Rate(100)
        self.iter             = 0
        self.qtmp             = np.zeros((4,))


    def joint_state_callback(self, msg):
        # 전체 상태
        self.q = np.array(msg.position) 
        self.v = np.array(msg.velocity) 

        # 드론 base pose
        base_xyzquat = np.array(msg.position[:7]) # 드론 위치
        self.baseSE3 = pin.XYZQUATToSE3(base_xyzquat) # 피노키오 객체로 변환
        self.v[:3]   = self.baseSE3.rotation @ self.v[:3] # 드론 선속도 -> local to world

        self.mppi.update_joint(self.q, self.v)


    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.q is None or self.v is None: # 센서 메시지 들어오지 않으면 대기
                continue

            # Pinocchio dynamics 업데이트
            pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]

            torque = np.zeros((7,))

            g = self.robot.data.nle

            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]
            if not self.jointControlFlag: # 조인트 제어가 완료되지 않은 동안
                qtarget = np.array([1.57, 1.7, 0, 4.4, 0, 4.71, 0.0]) # 조인트 target : initial pose

                if not self.control_init: # 처음에는 control_init가 False 이므로 trajectories 초기화
                    stime = time()
                    duration = 1.5
                    qinit = self.q[7:].copy()
                    # print("qinit", qinit)
                    self.jointTraj.setDuration(duration)
                    self.jointTraj.setStartTime(stime)
                    self.jointTraj.setInitSample(qinit)
                    self.jointTraj.setTargetSample(qtarget)
                    self.control_init = True
                else: # control_init가 이제 True 이니까 traj 따라서 매니퓰레이터 제어
                    ctime = time()
                    self.jointTraj.setCurrentTime(ctime)
                    qdes = self.jointTraj.computeNext()
                    qerr = qdes - self.q[7:]
                    ades = 1000 * qerr - 100 * self.v[6:]
                    torque = self.robot.data.M[6:, 6:] @ ades + g[6:]
                    if np.linalg.norm(qtarget - self.q[7:]) < 0.01: # 조인트 오차가 충분히 작으면
                            self.iter += 1
                            if self.iter > 50: # 그게 50 스텝 유지되면
                                self.jointControlFlag = True # True
                                self.control_init = False
                                print("Joint Control Finished")
 
            else:
                if not self.control_init: # SE3 제어 시작 전 traj 초기화
                    stime = time()
                    duration = 12.0
                    xyzquat = np.array([1.0, 1.0, 1.1, 0.0, 0.0, 0.0, 1.0])

                    targetSE3 = pin.SE3(1) # 목표 pose 설정
                    targetSE3.translation = xyzquat[:3].copy() + oMi.translation.copy()
                    targetSE3.rotation = oMi.rotation
                    print("targetSE3:", targetSE3)

                    oMi_init = deepcopy(oMi) 
                    self.se3Traj.setDuration(duration)
                    self.se3Traj.setStartTime(stime)
                    self.se3Traj.setInitSample(oMi_init)
                    self.se3Traj.setTargetSample(targetSE3)
                    self.qtmp[:3] = self.q[:3].copy()
                    self.qtmp[3] = quaternion_to_rpy(self.q[3:7])[2].copy()
                    self.control_init = True
                else:
                    # MPPI로 다음 target q_des 생성
                    qdes_, vdes = self.mppi.compute_control_input()     
                    # q_des에 해당하는 제어 토크 입력 계산                   
                    torque = self.robot.data.M[6:,6:] @ (400 * (qdes_ - self.q[7:]) + 40 * ( - self.v[6:])) + g[6:]

                

            # 계산된 제어 토크 입력 발행
            msg = JointState()
            msg.effort = [float(t) for t in torque[:7]]
            self.publisher.publish(msg)


if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
