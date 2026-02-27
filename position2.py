import os
import time
import numpy as np
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

# Setup (Use your actual paths/ports)
PORT = "/dev/ttyACM0"
ROBOT_ID = "follower1"

# ALWAYS use absolute paths with placo!
URDF_PATH = os.path.abspath("so101_new_calib.urdf")

def main():
    if not os.path.exists(URDF_PATH):
        print(f"ERROR: Cannot find {URDF_PATH}")
        return

    # 1. Boot up the math solver (Expecting 5 joints) - optional here, but keeping for consistency
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    
    # 2. Connect to the robot
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, calibration_dir=Path("."), use_degrees=True)
    robot = SOFollower(config)
    robot.connect()
    
    # 3. Make the arm limp so you can move it by hand!
    print("Disabling torque...")
    for motor in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
        robot.bus.write("Torque_Enable", motor, 0)
    
    print("\n--- ARM IS LIMP! Move each joint to its min/max limits to observe angles ---")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # 4. Read the joint angles
            obs = robot.get_observation()
            
            # Grab the 5 arm joints
            kin_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            current_angles = np.array([obs[f"{n}.pos"] for n in kin_joint_names], dtype=float)
            
            # Print on a single continuously updating line
            print(f"\rPan: {current_angles[0]:+7.1f}° | Lift: {current_angles[1]:+7.1f}° | Elbow: {current_angles[2]:+7.1f}° | Wrist Flex: {current_angles[3]:+7.1f}° | Wrist Roll: {current_angles[4]:+7.1f}°", end="")
            time.sleep(0.05)  # 20Hz update
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()