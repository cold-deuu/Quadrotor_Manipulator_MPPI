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
from mppi_solver.whole_mppi2 import whole_MPPI 
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


def xyzquat_to_xyzrpy(xyzquat):
    quaternion = xyzquat[3:].copy()
    rotation = R.from_quat(quaternion)
    rpy = rotation.as_euler('zyx', degrees=False)

    return np.array([xyzquat[0], xyzquat[1], xyzquat[2], rpy[0], rpy[1], rpy[2]])


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

        self.publisher        = rospy.Publisher("/harrierD7/robot_cmd", JointState, queue_size=10)
        self.dronePosePublisher     = rospy.Publisher("/harrierD7/drone_pose", Float64MultiArray, queue_size=10)

        self.robot = kinova()
        self.q     = None
        self.v     = None
        self.baseSE3 = pin.SE3(1)

        self.mppi = whole_MPPI()

        rospack = rospkg.RosPack()
        urdf_path = rospack.get_path("aerial_manipulation") + "/urdf/aerial_manipulator_gpu.urdf"

        self.jointTraj    = jointTraj(7)
        self.se3Traj      = SE3Traj()

        self.control_init     = False
        self.jointControlFlag = False
        self.rate             = rospy.Rate(100)
        self.iter             = 0
        self.qtmp             = np.zeros((4,))




    def joint_state_callback(self, msg):
        self.q = np.array(msg.position) 
        self.v = np.array(msg.velocity) 

        q_drone_euler = xyzquat_to_xyzrpy(self.q[:7])
        self.q_euler = np.append(q_drone_euler, self.q[7:])

        base_xyzquat = self.q[:7].copy() 
        self.baseSE3 = pin.XYZQUATToSE3(base_xyzquat) 
        self.v[:3] = self.baseSE3.rotation @ self.v[:3] 

        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
        # pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        # pin.updateFramePlacements(self.robot.model, self.robot.data)


        self.mppi.update_state(self.q_euler, self.v)


    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.q is None or self.v is None: 
                continue

            pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]

            torque = np.zeros((7,))

            g = self.robot.data.nle

            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]
            if not self.jointControlFlag: 
                qtarget = np.array([1.57, 1.7, 0, 4.4, 0, 4.71, 0.0]) 

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
                    qerr = qdes - self.q[7:].copy()
                    ades = 1000 * qerr - 100 * self.v[6:]
                    torque = self.robot.data.M[6:, 6:] @ ades + g[6:]
                    if np.linalg.norm(qtarget - self.q[7:]) < 0.01:
                            self.iter += 1
                            if self.iter > 50: 
                                self.jointControlFlag = True # True
                                self.control_init = False
                                print("Joint Control Finished")

                    rospy.logwarn(f"oMi Current : {oMi}")
   

                msg_arm = JointState()
                msg_arm.effort = [float(t) for t in torque[:7]]
                self.publisher.publish(msg_arm)
 
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
                    self.qtmp[3] = xyzquat_to_xyzrpy(self.q[:7])[2].copy()
                    self.control_init = True

                    msg_arm = JointState()
                    msg_arm.effort = [float(t) for t in torque[:7]]
                    self.publisher.publish(msg_arm)


                else:
                    
                    q_des, v_des = self.mppi.compute_control_input()     
                    torque = self.robot.data.M[6:,6:] @ (400 * (q_des[3:] - self.q[7:]) + 40 * ( - self.v[6:])) + g[6:]
                    
                    print(f'oMi : {oMi}')
                
                    msg_arm = JointState()
                    msg_arm.effort = [float(t) for t in torque[:7]]
                    self.publisher.publish(msg_arm)

                    msg_drone = Float64MultiArray()
                    msg_drone.data = q_des[:3].tolist()
                    self.dronePosePublisher.publish(msg_drone)


if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
