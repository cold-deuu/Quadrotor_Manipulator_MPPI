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

                # print(u)
                # tau = u[1:].to('cpu').numpy()
                # f = u[0].to("cpu").numpy()
                # f_ = np.array([0,0,f])

                # I_inv = np.zeros((3,3))
                # I_inv[0,0] = 1/1.57
                # I_inv[1,1] = 1/3.93
                # I_inv[2,2] = 1/2.59

                # J = self.compute_rotational_jacobian()
                # J_inv = np.linalg.inv(J)
                # g = np.array([0,0,-9.81])
                # omega_next = self.v[3:].copy() + I_inv @ tau * 0.01
                # theta_next = self.q_euler[3:].copy() + 0.01 * J_inv @ omega_next.copy()
                # rot_next = self.get_rotation_matrix(theta_next)
                # v_next = self.v[:3].copy() +0.01 * (g + rot_next @ f_)/14.7
                # x_next = self.q[:3].copy() + 0.01 * v_next

                

                # print(u)


                # frame_name=  "drone"
                # frame_id = self.robot.model.getFrameId(frame_name)
                # oMf = self.robot.data.oMf[frame_id]
                # J_local = pin.getFrameJacobian(self.robot.model, self.robot.data, frame_id, pin.ReferenceFrame.LOCAL)
                # # J_drone = np.zeros((6,4))
                # # J_drone[:,:3] = J_local[:,:3].copy()
                # # J_drone[:, -1] = J_local[:,-1].copy()
                # if not self.control_init:

                        
                #     targetSE3 = pin.XYZQUATToSE3(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
                #     targetSE3 = oMf * targetSE3
                #     oMf_init = deepcopy(oMf)
                #     self.se3Traj.setInitSample(oMf_init)
                #     self.se3Traj.setTargetSample(targetSE3)
                #     self.se3Traj.setStartTime(time())
                #     self.se3Traj.setDuration(12.0)
                #     self.control_init = True
                # else:
                #     ctime = time()
                #     self.se3Traj.setCurrentTime(ctime)
                #     target = self.se3Traj.computeNext()

                #     target_6d = pin.log(target).vector
                #     diff_6d = pin.log(oMf.inverse() * target).vector
                #     vdrone = np.zeros((4))
                #     vdrone[:3] = self.v[:3].copy()
                #     vdrone[3] = self.v[-1].copy()
                #     ades = 5 * diff_6d - 20 * J_local @ self.v
                #     qddot_des = np.linalg.pinv(J_local) @ ades

                #     print(f"target : {target_6d}")

                #     mass = self.robot.data.M
                #     # g = self.robot.data.nle
                #     g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, self.q)


                #     RR_mat = np.zeros((6,6))
                #     RR_mat[:3,:3] = RR_mat[3:,3:] = oMf.rotation.copy()

                #     qddot_des = RR_mat @ qddot_des
                #     g = RR_mat @ g
                #     wrench = mass @ qddot_des + g
                #     wrench = np.delete(target_6d,[3,4])
                
                
                msg = Float64MultiArray()
                msg.data = xdes.to('cpu').tolist()
                self.dronePosePublisher.publish(msg)

                    

if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
