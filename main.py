import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda

from ollama import generate # Single messages

# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cube1 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.6, -0.2, 0.05])
cube2 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.4, -0.3, 0.05])


# p.quaternionFromEuler([0, 0, 0]) # Uses quaternion by default. Can convert from Euler angle since easier

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# let the scene initialize
for i in range (200):
    p.stepSimulation()
    time.sleep(control_dt)

# run sequence of position and gripper commands
# for i in range (800):
#     panda.move_to_pose(ee_position=[0.6, -0.2, 0.1], ee_rotz=0., positionGain=0.01)
#     p.stepSimulation()
#     time.sleep(control_dt)

# for i in range (800):
#     panda.move_to_pose(ee_position=[0.6, -0.2, 0.02], ee_rotz=0., positionGain=0.01)
#     p.stepSimulation()
#     time.sleep(control_dt)    

# for i in range (300):
#     panda.close_gripper()
#     p.stepSimulation()
#     time.sleep(control_dt)

# for i in range (800):
#     panda.move_to_pose(ee_position=[0.6, -0.2, 0.4], ee_rotz=np.pi/2, positionGain=0.01)
#     p.stepSimulation()
#     time.sleep(control_dt)  

# Control robot with LLM commands via Ollama
# Ignore joints 8 and 9
# Joints 10 and 11 are the gripper fingers
# Revolute = radians and prismatic = meters
# We're using this Panda robot this year

# LLM should have access to state of arm and ball. It should be given command from user and append the state
# Have some code here that can call robot functions

terminate = False

while not terminate: # Execute commands for robot
    user_command = input("Enter your command for the robot (or 'exit' to quit): ")
    if user_command.lower() == 'exit':
        terminate = True
        break

    state = panda.get_state()
    state_description = f"Joint Positions: {state['joint-position']}, End-Effector Position: {state['ee-position']}"

    cube1_pos, cube1_orn = p.getBasePositionAndOrientation(cube1)
    cube2_pos, cube2_orn = p.getBasePositionAndOrientation(cube2)
    env_description = f"Cube1 Position: {cube1_pos}, Cube1 Orientation: {cube1_orn}, Cube2 Position: {cube2_pos}, Cube2 Orientation: {cube2_orn}."

    # One shot prompt with state, user command, available functions, and expected format
    prompt = f"""
    You are controlling a robot arm to accomplish user instructions.
    The robot is a Panda robotic arm with a two-finger gripper. It is on a table with a small cube nearby.
    You will figure out the minimum sequence of robot actions that will fulfill the user's command.
    The robot has the following commands available: move_to_pose(x, y, z, rotz), open_gripper(), close_gripper().
    To pick up a cube, move the end-effector above the cube, lower it down, close the gripper, then lift up.
    To place down a cube, move the end-effector above the target position, lower it down, open the gripper, then lift up.
    The end-effector orientation should have the gripper facing downwards (i.e., rotz = 0.0) unless specified otherwise.
    The units for x, y, z, and rotz are meters and radians, respectively.
    Use the current robot and environment state to determine appropriate coordinates for the actions.
    Your response should be a semicolon separated list of the exact commands to run.
    Do not provide any additional text or explanations.

    <Start of Example>
    Current robot state: Joint Positions: [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.0, 0.0, 0.04, 0.04], End-Effector Position: (0.55, 0.00, 0.52)
    Current environment state: Cube1 Position: (0.60, -0.20, 0.025), Cube1 Orientation: (0.0, 0.0, 0.0, 1.0), Cube2 Position: (0.40, -0.30, 0.025), Cube2 Orientation: (0.0, 0.0, 0.0, 1.0)
    User command: pick up a cube
    Response: open_gripper(); move_to_pose(0.60, -0.20, 0.12, 0.0); move_to_pose(0.60, -0.20, 0.03, 0.0); close_gripper(); move_to_pose(0.60, -0.20, 0.20, 0.0)
    <End of Example>
    
    Current robot state: {state_description}
    Current environment state: {env_description}
    User command: {user_command}
    Response:"""

    print("\nPrompt to LLM:")
    print(prompt)

    response = generate("llama3", prompt=prompt)

    reponse_text = response["response"]

    print("LLM Response:")
    print(reponse_text)

    # Command options: move_to_pose(x, y, z, rotz), open_gripper(), close_gripper()
    commands = [cmd.strip() for cmd in reponse_text.split(';')]
    for cmd in commands:
        if cmd.startswith("move_to_pose"):
            # Extract parameters
            params = cmd[13:-1]  # Get the string inside the parentheses
            x, y, z, rotz = map(float, params.split(','))
            print(f"Executing: move_to_pose({x}, {y}, {z}, {rotz})")
            for i in range (800):
                panda.move_to_pose(ee_position=[x, y, z], ee_rotz=rotz, positionGain=0.01)
                p.stepSimulation()
                time.sleep(control_dt)
        elif cmd == "open_gripper()":
            print("Executing: open_gripper()")
            for i in range (300):
                panda.open_gripper()
                p.stepSimulation()
                time.sleep(control_dt)
        elif cmd == "close_gripper()":
            print("Executing: close_gripper()")
            for i in range (300):
                panda.close_gripper()
                p.stepSimulation()
                time.sleep(control_dt)
        else:
            print(f"Unknown command: {cmd}")

    state = panda.get_state()