import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

import rospkg
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from trajectory.trajManager import jointTraj, SE3Traj

from time import time
from scipy.spatial.transform import Rotation as R

from copy import deepcopy

from cvxopt import matrix, solvers

from robot.urdf_fk import URDFFK
# from mppi_solver.whole_mppi2 import whole_MPPI 
from mppi_solver.mppi import MPPI
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



class kinova(RobotWrapper):
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('aerial_manipulation')
        pkg_dir = package_path + '/urdf/kinova_urdf'
        urdf_path = pkg_dir + '/kinova.urdf'  

        self.robot = self.BuildFromURDF(urdf_path)
        self.data, _, _, = pin.createDatas(
            self.robot.model,
            self.robot.collision_model,
            self.robot.visual_model
        )
        self.model = self.robot.model

        print(f"Model Nq : {self.model.nq}")


class controller:
    def __init__(self):
        rospy.init_node("kinova_controller", anonymous=True)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.publisher = rospy.Publisher("/kinova_effort_controller/command", Float64MultiArray, queue_size=10)

        self.robot = kinova()
        self.q     = None
        self.v     = None

        self.mppi = MPPI()

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

        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
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

            oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]
            if not self.jointControlFlag: 
                qtarget = np.array([0, 3.14, 0,4.7,0,1.57, -6.28]) 

                if not self.control_init: 
                    stime = time()
                    duration = 3.0
                    qinit = self.q.copy()

                    self.jointTraj.setDuration(duration)
                    self.jointTraj.setStartTime(stime)
                    self.jointTraj.setInitSample(qinit)
                    self.jointTraj.setTargetSample(qtarget)
                    self.control_init = True
                else:
                    ctime = time()
                    self.jointTraj.setCurrentTime(ctime)
                    qdes = self.jointTraj.computeNext()
                    qerr = qdes - self.q.copy()
                    ades = 400 * qerr - 40 * self.v
                    torque = self.robot.data.M @ ades + g
                    print(f"norm : {np.linalg.norm(qtarget - self.q)}")
                    print(f"qcurrent : {self.q}")
                    if np.linalg.norm(qtarget - self.q) < 0.01:
                        self.jointControlFlag = True # True
                        self.control_init = False
                        print("Joint Control Finished")

                msg = Float64MultiArray()
                msg.data = torque.tolist()
                self.publisher.publish(msg)
 
            else:
                if not self.control_init:
                    self.control_init = True
                q_des, v_des = self.mppi.compute_control_input()  
                    
                torque = self.robot.data.M @ (400 * (q_des - self.q) + 40 * (- self.v)) + g
                
                print(f"oMi : {oMi}")
                msg = Float64MultiArray()
                msg.data = torque.tolist()
                self.publisher.publish(msg)

                # msg_drone = Float64MultiArray()
                # msg_drone.data = q_des[:3].tolist()
                # self.dronePosePublisher.publish(msg_drone)


if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
