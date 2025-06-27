import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

import rospkg
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
from trajManager import jointTraj, SE3Traj

from time import time
from scipy.spatial.transform import Rotation as R

from copy import deepcopy

from cvxopt import matrix, solvers



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

class kinova(RobotWrapper):
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('aerial_manipulation')
        pkg_dir = package_path + '/urdf'
        urdf_path = pkg_dir + '/full_robot_floating2.urdf'

        self.robot = self.BuildFromURDF(urdf_path)

        self.data, _, _, = pin.createDatas(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.model = self.robot.model

    # def computeAllTerms(self, q, v):
    #     pin.computeAllTerms(self.model, self.data, )
        


class controller:
    def __init__(self):
        rospy.init_node("kinova_controller", anonymous=True)
        rospy.Subscriber("/harrierD7/robot_states", JointState, self.joint_state_callback)
        
        self.publisher = rospy.Publisher("/harrierD7/robot_cmd", JointState, queue_size=10)
        self.dronePosePublisher = rospy.Publisher("/harrierD7/drone_pose", Float64MultiArray, queue_size=10)

        self.robot = kinova()
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

        self.iter = 0
        self.qtmp = np.zeros((4))
    def joint_state_callback(self, msg):
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)


        # for i in range(self.robot.model.nq):
        #     self.q[i] = msg.position[i] # xyzquat

        # for i in range(self.robot.model.nv):
        #     self.v[i] = msg.velocity[i] # xyzrpy
        
        base_xyzquat = np.array(msg.position[:7])
        self.baseSE3 = pin.XYZQUATToSE3(base_xyzquat)
        self.v[:3] = self.baseSE3.rotation @ self.v[:3]

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.q is not None:
                pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.v)
                oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]
                
                torque = np.zeros((7))
                
                # Model 
                nq = self.robot.model.nq
                nv = self.robot.model.nv    

                # Mass
                mass = self.robot.data.M    
                # Gravity
                g = self.robot.data.nle 

                joint_id = self.robot.model.getJointId("j2s7s300_joint_7")  # 조인트 ID 가져오기

                # pin.computeJointJacobian(self.robot.model, self.robot.data, self.q)
                J = pin.getJointJacobian(self.robot.model, self.robot.data, joint_id,pin.ReferenceFrame.LOCAL)  
                # J[3:5,:6] = np.zeros((2, 6)) 
                
                # J[:,6:] = np.zeros((6,7))

                # pretty_matrix_print(mass.tolist())
                # oMi
                oMi = self.robot.data.oMi[self.robot.index("j2s7s300_joint_7")]
                if not self.jointControlFlag:
                    # qtarget = np.array([0.0, 0.0, 0.0, 4.4, 0.0, 4.71, 0.0])
                    qtarget = np.array([1.57, 1.7, 0, 4.4, 0, 4.71, 0.0])

                    if not self.control_init:
                        stime = time()
                        duration = 1.5
                        # qtarget = np.zeros((nq - 7))
                        qinit = self.q[7:].copy()
                        print("qinit", qinit)
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
                        torque = self.robot.data.M[6:,6:] @ ades + g[6:]
                        if np.linalg.norm(qtarget-self.q[7:]) < 0.01:
                            self.iter += 1
                            # print("J:", J)

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
                        # print("oMi:", oMi)
                        # print("targetSE3:", pin.XYZQUATToSE3(xyzquat))
                        print("targetSE3:", targetSE3)
                        oMi_init = deepcopy(oMi)
                        # targetSE3 = pin.XYZQUATToSE3(xyzquat)
                        # targetSE3.rotation = oMi.rotation.copy()
                        self.se3Traj.setDuration(duration)
                        self.se3Traj.setStartTime(stime)
                        self.se3Traj.setInitSample(oMi_init)
                        self.se3Traj.setTargetSample(targetSE3)
                        self.qtmp[:3] = self.q[:3].copy()
                        self.qtmp[3] = quaternion_to_rpy(self.q[3:7])[2].copy()
                        self.control_init = True


                    else:
                        ctime = time()
                        self.se3Traj.setCurrentTime(ctime)
                        oMi_des = self.se3Traj.computeNext()
                        oMi_err = pin.SE3(1)
                        # oMi_err.translation = oMi_des.translation - oMi.translation
                        # oMi_err.rotation = oMi.rotation.T @ oMi_des.rotation
                        oMi_err = oMi.inverse() * oMi_des
                        xerr6d = pin.log6(oMi_err).vector
                        ades = 100 * xerr6d - 10 * J @ self.v

                        qdot = np.linalg.pinv(J) @ ades

                        qcurr = np.zeros((13))
                        qcurr[:3] = self.q[:3].copy()  # Copy the first 3 elements (position)
                        qcurr[3:6] = quaternion_to_rpy(self.q[3:7].copy())
                        qcurr[6:] = self.q[7:].copy()
                        


                        P = matrix(np.eye(13))  # Identity matrix of size 13x13
                        q = matrix(np.zeros(13))  # Zero vector of size 13                        
                        P[:6,:6] *= 0.0001  # Set the first 6x6 block to 100 times the identity matrix
                        A = np.zeros((6,13))
                        A[:6,:] = J.copy()

                        b = np.zeros((6))
                        b[:6] = ades.copy()

                        A = matrix(A)
                        b = matrix(b)
                        solution = solvers.qp(P, q, None, None, A,b)
                        x_opt = np.array(solution['x']).flatten()

                        qdes = qcurr + x_opt * 0.01  # Assuming a time step of 0.01 seconds


                        torque = self.robot.data.M@ (400 * (qdes-qcurr) - 40 * self.v ) + g

                        pose_des = np.zeros((4))
                        pose_des[:3] = qdes[:3].copy()

                        yaw_init = qdes[3]  
                        pose_des[3] = yaw_init
                        
                        torque = torque[6:]  

                        # IF
                        # torque = self.robot.data.M[6:, 6:] @ (400 * (qtarget-qcurr[6:]) - 40 * self.v[6:]) + g[6:]
                        # print(qcurr[:6])
                        # xdot_drone = 100 * (np.array([0, 1.0, 2.1, 0.0, 0.0, 0.3]) - qcurr[:6])
                        # pose_des[:3] = qcurr[:3] + xdot_drone[:3] * 0.01
                        # pose_des[3] = qcurr[5] + xdot_drone[5] * 0.01




                        # print("pose_des:", pose_des)
                        msg_drone = Float64MultiArray()
                        msg_drone.data = pose_des.tolist()
                        self.dronePosePublisher.publish(msg_drone)
                        

                msg = JointState()
                msg.effort = []


                for i in range(7):
                    msg.effort.append(torque[i])
                self.publisher.publish(msg)
            




if __name__ == "__main__":
    ctrl = controller()
    ctrl.main()
