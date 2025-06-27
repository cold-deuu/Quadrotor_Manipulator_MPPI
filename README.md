### MPPI Controller for Quadrotor Manipulator System ###
---
Simulation Base : https://github.com/RISC-NYUAD/Aerial-Manipulator-Gazebo.git

## Dependencies ##
---
* Ubuntu 20-04, ROS-Noetic
* Pinocchio (Kinematics Solver)
* CVXOPT (pip install cvxopt)

## How To Run ##
---
**Launch Simulator**
```Terminal
roslaunch aerial_manipulation aerial_manipulator
```
**Launch Controller**
```
roscd mav_mppi/scripts
python3 kinova.py
```
## Change Log ##
---
|Date|Log|
|--|--|
|2025-06-27|Test IK Control|


## To Do List ##
[ ] Insert MPPI Controller 
[ ] Data Collector Script 
[ ] Code 정리
 
