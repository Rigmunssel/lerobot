import os
import time
import numpy as np
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

# Setup (Use your actual paths/ports)
PORT = "/dev/ttyACM3"
ROBOT_ID = "follower1"
# ALWAYS use absolute paths with placo!
URDF_PATH = os.path.abspath("so101_new_calib.urdf")

def main():
    if not os.path.exists(URDF_PATH):
        print(f"ERROR: Cannot find {URDF_PATH}")
        return

    # 1. Boot up the math solver (Expecting 5 joints)
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
    
    print("\n--- ARM IS LIMP! Move it around to see coordinates ---")
    
    try:
        while True:
            # 4. Read the joint angles
            obs = robot.get_observation()
            
            # CRITICAL FIX: Only grab the 5 arm joints for the math solver!
            kin_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            current_angles = np.array([obs[f"{n}.pos"] for n in kin_joint_names], dtype=float)
            
            # 5. Use Forward Kinematics to calculate X, Y, Z
            ee_pose = kin.forward_kinematics(current_angles)
            current_x = ee_pose[0, 3]
            current_y = ee_pose[1, 3]
            current_z = ee_pose[2, 3]
            
            # Print on a single continuously updating line to avoid spamming the console
            print(f"\rX: {current_x:+.4f}m | Y: {current_y:+.4f}m | Z (Height): {current_z:+.4f}m", end="")
            time.sleep(0.05) # 20Hz update is plenty fast and smooth
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()