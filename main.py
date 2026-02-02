import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda

import ollama

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
# Randomize cube position to ensure genuine reasoning
cube1 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.6 + np.random.uniform(-0.05, 0.05), -0.2 + np.random.uniform(-0.05, 0.05), 0.05])
cube2 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.4 + np.random.uniform(-0.05, 0.05), -0.3 + np.random.uniform(-0.05, 0.05), 0.05])

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# Tool configuration
def move_to_pose(x: float, y: float, z: float, rotz: float) -> str:
    """
    Move robot end-effector to a Cartesian pose.

    Args:
        x: target x (meters)
        y: target y (meters)
        z: target z (meters)
        rotz: target rotation about z (radians). Use 0.0 for gripper facing down.
    Returns:
        A short status string.
    """

    # Could clamp bounds for safety

    for _ in range(800):
        panda.move_to_pose(ee_position=[x, y, z], ee_rotz=rotz, positionGain=0.01)
        p.stepSimulation()
        time.sleep(control_dt)

    return f"moved_to_pose({x:.3f}, {y:.3f}, {z:.3f}, {rotz:.3f})"

def open_gripper() -> str:
    """Open the robot gripper."""
    for _ in range(300):
        panda.open_gripper()
        p.stepSimulation()
        time.sleep(control_dt)
    return "gripper_opened"

def close_gripper() -> str:
    """Close the robot gripper."""
    for _ in range(300):
        panda.close_gripper()
        p.stepSimulation()
        time.sleep(control_dt)
    return "gripper_closed"

def done(reason: str = "") -> str:
    return f"done: {reason}"

available_functions = {
    "move_to_pose": move_to_pose,
    "open_gripper": open_gripper,
    "close_gripper": close_gripper,
    "done": done
}

def describe_state() -> str:
    s = panda.get_state()
    ee = s["ee-position"]
    return (
        f"End-Effector Position: ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}); "
        f"Gripper State: {s['gripper_state']}."
    )

def describe_env() -> str:
    cube1_pos, _ = p.getBasePositionAndOrientation(cube1)
    cube2_pos, _ = p.getBasePositionAndOrientation(cube2)
    return (
        f"Cube1 Position: ({cube1_pos[0]:.4f}, {cube1_pos[1]:.4f}, {cube1_pos[2]:.4f}); "
        f"Cube2 Position: ({cube2_pos[0]:.4f}, {cube2_pos[1]:.4f}, {cube2_pos[2]:.4f}); "
        f"Cube size: 0.05m."
    )


# System prompt for tool use; generated via Chat
SYSTEM = """
You control a Panda robot arm with a gripper in simulation by calling tools.

You MUST act by calling tools. Do not output plans, JSON, or explanations.
- If you need multiple steps, call multiple tools.

Available tools:
- move_to_pose(x, y, z, rotz)
- open_gripper()
- close_gripper()
- done()

To place a cube on top of another, you should move the first cube at least 10cm above the second cube before lowering it down.

Use rotz=0.0 unless the user explicitly requests otherwise.

When the task is complete, call done() exactly once and stop.
"""

MODEL = "qwen3:8b"

# let the scene initialize
for i in range (200):
    p.stepSimulation()
    time.sleep(control_dt)

terminate = False

while not terminate: # Execute commands for robot
    user_command = input("Enter your command for the robot (or 'exit' to quit): ")
    if user_command.lower() == 'exit':
        terminate = True
        break

    state = panda.get_state()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Observations:\nState: {describe_state()}\nEnv: {describe_env()}\nCommand: {user_command}"}
    ]

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=[move_to_pose, open_gripper, close_gripper, done],
        options={
            "temperature": 0.0,
        },
        keep_alive="30m",
    )

    # print("Thinking:", getattr(response.message, "thinking", None))
    # print("Content :", repr(getattr(response.message, "content", None)))
    # print("Tool calls:", getattr(response.message, "tool_calls", None))

    messages.append(response.message)

    tool_calls = getattr(response.message, "tool_calls", None) or []
    if not tool_calls:
        # Done (model chose not to call tools further)
        break

    for call in tool_calls:
        fn_name = call.function.name
        fn_args = call.function.arguments or {}
        fn = available_functions.get(fn_name)

        if fn is None:
            messages.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": f"ERROR: unknown tool {fn_name}",
            })
            continue

        try:
            result = fn(**fn_args)
        except Exception as e:
            result = f"ERROR executing {fn_name}({fn_args}): {e}"

        messages.append({
            "role": "tool",
            "tool_name": fn_name,
            "content": str(result),
        })

        messages.append({
        "role": "user",
        "content": "Updated observations:\n"
                f"State: {describe_state()}\n"
                f"Env: {describe_env()}"
        })


# Notes from class:

# Control robot with LLM commands via Ollama
# Ignore joints 8 and 9
# Joints 10 and 11 are the gripper fingers
# Revolute = radians and prismatic = meters
# We're using this Panda robot this year

# LLM should have access to state of arm and ball. It should be given command from user and append the state
# Have some code here that can call robot functions

# p.quaternionFromEuler([0, 0, 0]) # Uses quaternion by default. Can convert from Euler angle since easier