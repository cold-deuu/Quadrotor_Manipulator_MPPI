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
from utils.pose import Pose
from data.data_logger import DataLogger  

from cvxopt import matrix, solvers

from robot.urdf_fk import URDFFK
from mppi_solver.mppi import MPPI
import torch

def quaternion_to_rpy(quaternion):
    rotation = R.from_quat(quaternion)
    return rotation.as_euler('zyx', degrees=False)

class kinova(RobotWrapper):
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('aerial_manipulation')
        pkg_dir = package_path + '/urdf'
        urdf_path = pkg_dir + '/full_robot_floating2.urdf'
        self.robot = self.BuildFromURDF(urdf_path)
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

        self.publisher    = rospy.Publisher("/harrierD7/robot_cmd", JointState, queue_size=10)
        self.dronePosePub = rospy.Publisher("/harrierD7/drone_pose", Float64MultiArray, queue_size=10)

        self.robot = kinova()
        self.q = None
        self.v = None
        self.baseSE3 = pin.SE3(1)

        target_pose = Pose()
        target_pose.pose = torch.tensor([0.1029, 0.4055, 1.6498])
        target_pose.orientation = torch.tensor([-0.5, -0.5, 0.5, -0.5])

        self.mppi = MPPI(target_pose=target_pose)

        rospack = rospkg.RosPack()
        urdf_path = rospack.get_path("aerial_manipulation") + "/urdf/aerial_manipulator_gpu.urdf"
        self.fk_urdf = URDFFK(urdf_path, root_link="base", end_link="j2s7s300_link_7")

        self.jointTraj = jointTraj(7)
        self.se3Traj   = SE3Traj()

        self.control_init     = False
        self.jointControlFlag = False
        self.rate             = rospy.Rate(100)
        self.iter             = 0
        self.qtmp             = np.zeros((4,))

        # DataLogger 인스턴스
        log_file = "/home/chan/aerial_ws/src/mav_mppi/scripts/data/kinova_log.csv"
        self.logger = DataLogger(log_file)
        rospy.on_shutdown(self.on_shutdown)

    def joint_state_callback(self, msg):
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)
        base_xyzquat = np.array(msg.position[:7])
        self.baseSE3 = pin.XYZQUATToSE3(base_xyzquat)
        self.v[:3]   = self.baseSE3.rotation @ self.v[:3]
        self.mppi.update_joint(self.q, self.v)

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.q is None or self.v is None:
                continue

            pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]

            torque = np.zeros((7,))
            g = self.robot.data.nle

            qtarget = np.array([1.57, 1.7, 0, 4.4, 0, 4.71, 0.0])

            if not self.jointControlFlag:
                if not self.control_init:
                    stime = time()
                    duration = 1.5
                    qinit = self.q[7:].copy()
                    self.jointTraj.setDuration(duration)
                    self.jointTraj.setStartTime(stime)
                    self.jointTraj.setInitSample(qinit)
                    self.jointTraj.setTargetSample(qtarget)
                    self.control_init = True
                else:
                    ctime = time()
                    self.jointTraj.setCurrentTime(ctime)
                    qdes = self.jointTraj.computeNext()
                    qerr = qdes - self.q[7:]
                    ades = 1000 * qerr - 100 * self.v[6:]
                    torque = self.robot.data.M[6:, 6:] @ ades + g[6:]
                    if np.linalg.norm(qtarget - self.q[7:]) < 0.01:
                        self.iter += 1
                        if self.iter > 50:
                            self.jointControlFlag = True
                            self.control_init = False
                            print("Joint Control Finished")
            else:
                if not self.control_init:
                    stime = time()
                    duration = 12.0
                    xyzquat = np.array([1.0, 1.0, 1.1, 0.0, 0.0, 0.0, 1.0])

                    targetSE3 = pin.SE3(1)
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
                    qdes, vdes, cost_log_dict = self.mppi.compute_control_input()
                    self.cost_log_dict = cost_log_dict
                    torque = self.robot.data.M[6:,6:] @ (400 * (qdes - self.q[7:]) + 40 * (-self.v[6:])) + g[6:]

            msg = JointState()
            msg.effort = [float(t) for t in torque[:7]]
            self.publisher.publish(msg)

            # DataLogger로 dict 넘김 (array는 그대로)
            # self.save_data(oMi)

    def save_data(self, oMi):
        # cost_log_dict가 없으면 아무것도 저장하지 않고 return
        if not hasattr(self, "cost_log_dict"):
            return

        now = rospy.get_time()
        q_actual = self.q[7:].copy()
        ee_pos = oMi.translation.copy()
        ee_vel = self.v[3:6].copy()
        eef_rpy = R.from_matrix(oMi.rotation).as_euler('zyx', degrees=False)
        target_pos = self.mppi.target_pose.pose.detach().cpu().numpy() if hasattr(self.mppi.target_pose.pose, "detach") else np.array(self.mppi.target_pose.pose)
        target_quat = self.mppi.target_pose.orientation.detach().cpu().numpy() if hasattr(self.mppi.target_pose.orientation, "detach") else np.array(self.mppi.target_pose.orientation)

        row_dict = {
            "time": now,
            "q": q_actual,
            "ee_pos": ee_pos,
            "ee_vel": ee_vel,
            "ee_rpy": eef_rpy,
            "ee_target_pos": target_pos,
            "ee_target_quat": target_quat,
            **self.cost_log_dict  # cost_log_dict가 반드시 존재할 때만!
        }

        self.logger.append(row_dict)

    def on_shutdown(self):
        self.logger.save()

if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
