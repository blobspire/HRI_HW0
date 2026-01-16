import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import config
from robot import Panda


# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=config.cameraDistance, 
                                cameraYaw=config.cameraYaw,
                                cameraPitch=config.cameraPitch, 
                                cameraTargetPosition=config.cameraTargetPosition)

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.6, -0.2, 0.05])

# load the robot
panda = Panda(basePosition=config.baseStartPosition,
                baseOrientation=p.getQuaternionFromEuler(config.baseStartOrientationE),
                jointStartPositions=config.jointStartPositions)

# let the scene initialize
for i in range (200):
    p.stepSimulation()
    time.sleep(config.control_dt)

# run sequence of position and gripper commands
for i in range (800):
    panda.move_to_pose(ee_position=[0.6, -0.2, 0.1], ee_rotz=0., positionGain=0.01)
    p.stepSimulation()
    time.sleep(config.control_dt)

for i in range (800):
    panda.move_to_pose(ee_position=[0.6, -0.2, 0.02], ee_rotz=0., positionGain=0.01)
    p.stepSimulation()
    time.sleep(config.control_dt)    

for i in range (300):
    panda.close_gripper()
    p.stepSimulation()
    time.sleep(config.control_dt)

for i in range (800):
    panda.move_to_pose(ee_position=[0.6, -0.2, 0.4], ee_rotz=np.pi/2, positionGain=0.01)
    p.stepSimulation()
    time.sleep(config.control_dt)  