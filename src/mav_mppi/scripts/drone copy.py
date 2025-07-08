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

from mppi_solver.drone_mppi import MPPI


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
    rotation = R.from_quat(quaternion)  # Create rotation object from quaternion
    rpy = rotation.as_euler('zyx', degrees=False)  # Convert to RPY (roll, pitch, yaw) using ZYX order
    return rpy

class drone(RobotWrapper):
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('aerial_manipulation')
        pkg_dir = package_path + '/urdf'
        urdf_path = pkg_dir + '/drone.urdf'

        self.robot = self.BuildFromURDF(urdf_path)

        self.data, _, _, = pin.createDatas(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.model = self.robot.model

    # def computeAllTerms(self, q, v):
    #     pin.computeAllTerms(self.model, self.data, )
        


class controller:
    def __init__(self):
        rospy.init_node("kinova_controller", anonymous=True)
        self.robot = drone()

        rospy.Subscriber("/harrierD7/robot_states", JointState, self.joint_state_callback)
        
        # self.publisher = rospy.Publisher("/harrierD7/robot_cmd", JointState, queue_size=10)
        self.dronePosePublisher = rospy.Publisher("/harrierD7/drone_pose", Float64MultiArray, queue_size=10)

        self.q = np.zeros((self.robot.model.nq))
        self.v = np.zeros((self.robot.model.nv))

        self.q = None
        self.v = None

        self.baseSE3 = pin.SE3(1)

        self.jointTraj = jointTraj(7)
        self.se3Traj = SE3Traj()


        self.control_init = False
        self.jointControlFlag = False
        self.rate = rospy.Rate(100)

        self.mppi = MPPI()

        self.iter = 0
        self.qtmp = np.zeros((4))


    def joint_state_callback(self, msg):
        self.q = np.array(msg.position[:7])
        self.v = np.array(msg.velocity[:6])
        self.q_euler = np.zeros((6))
        euler = quaternion_to_rpy(self.q[3:7])
        self.q_euler[:3] = self.q[:3].copy()
        self.q_euler[3:6] = euler.copy()

        base_xyzquat = np.array(msg.position[:7])
        self.baseSE3 = pin.XYZQUATToSE3(base_xyzquat)
        self.v[:3] = self.baseSE3.rotation @ self.v[:3]
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

    def compute_rotational_jacobian(self):
        phi, theta, psi = self.q_euler[3:]
        J = np.zeros((3,3))
        J[0,0] = 1.0
        J[0,1] = np.sin(phi) * np.tan(theta)
        J[0,2] = np.cos(phi) * np.tan(theta)
        J[1,1] = np.cos(phi)
        J[1,2] = -np.sin(phi)
        J[2,1] = np.sin(phi)/np.cos(theta)
        J[2,2] = np.cos(phi)/np.cos(theta)
        return J
    
    def get_rotation_matrix(self, rpy_world):
        roll, pitch, yaw = rpy_world[0], rpy_world[1], rpy_world[2]

        cphi = np.cos(roll)
        sphi = np.sin(roll)
        ctheta = np.cos(pitch)
        stheta = np.sin(pitch)
        cpsi = np.cos(yaw)
        spsi = np.sin(yaw)

        r00 = cpsi * ctheta
        r01 = cpsi * stheta * sphi - spsi * cphi
        r02 = cpsi * stheta * cphi + spsi * sphi

        r10 = spsi * ctheta
        r11 = spsi * stheta * sphi + cpsi * cphi
        r12 = spsi * stheta * cphi - cpsi * sphi

        r20 = -stheta
        r21 = ctheta * sphi
        r22 = ctheta * cphi

        R = np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ])

        return R
    

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.q is not None:
                trans = self.q[:3].copy()
                vel = self.v[:3].copy()
                print("trans : ", trans)
                self.mppi.set_state(trans, vel)
                xdes, _ = self.mppi.compute_control_input()
                print(f"Xdes : {xdes}")
                
                
                msg = Float64MultiArray()
                msg.data = xdes.to('cpu').tolist()
                self.dronePosePublisher.publish(msg)
                    

if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
