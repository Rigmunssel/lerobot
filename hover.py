import os
import math
import numpy as np
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics
import math

# ── Configuration ────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"
ROBOT_ID = "follower"
URDF_PATH = os.path.abspath("so101_new_calib.urdf")

TARGET_X = 0.20        
TARGET_Y = 0.05        
TARGET_Z = 0.15    
MIN_Z = 0.15    

STEP_DISTANCE = 0.02   
XYZ_THRESHOLD = 0.01   

angle_offset = 0
x_pan = 0
y_pan = 0

offset_angle_pan = 0
offset_angle_shoulder = 0
offset_angle_elbow = 0

elbow_lenght = 0.10
shoulder_length = 0.10


def get_target_angle(x_target,y_target):
    x_diff = x_target-x_pan
    y_diff = y_target-y_pan
    if y_diff < 0:
        return 0
    elif y_diff > 0:
        return math.atan2(y_diff, x_diff)
    else:
        return -math.atan2(abs(y_diff), x_diff)

import math

# Measure this on your robot! (Center of elbow joint to tip of gripper in meters)
L_FOREARM = 0.135 
Z_MIN = 0.10       # Safe floor height (10cm)

def get_elbow_change(angle_elbow_rad, angle_lift_rad, z_current):
    """
    Computes how many radians to rotate the elbow to lift the gripper to Z_MIN.
    """
    # 1. How much vertical height are we missing?
    z_diff = Z_MIN - z_current
    
    # If we are already above the safe zone, no change needed!
    if z_diff <= 0:
        return 0.0
        
    # 2. Calculate the forearm's current vertical contribution
    # Z_contribution = length * sin(total_angle)
    current_forearm_z = L_FOREARM * math.sin(angle_lift_rad + angle_elbow_rad)
    
    # 3. What does the forearm's Z contribution NEED to be to hit Z_MIN?
    target_forearm_z = current_forearm_z + z_diff
    
    # 4. Math safety check (sine cannot be greater than 1.0)
    sin_target = target_forearm_z / L_FOREARM
    if sin_target > 1.0:
        # The arm physically cannot reach Z_MIN just by bending the elbow!
        # E.g., The upper arm is pointing too far down.
        # We cap it to 1.0 to get as high as physically possible.
        sin_target = 1.0 
        
    # 5. Reverse the sine wave (arcsin) to find the new required total angle
    target_total_angle = math.asin(sin_target)
    
    # 6. Isolate the new elbow angle and find the difference
    new_elbow_rad = target_total_angle - angle_lift_rad
    elbow_change_rad = new_elbow_rad - angle_elbow_rad
    
    return elbow_change_rad





def main():
    # 1. THE FIX: Only give the math solver the 3 heavy joints!
    ik_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names 
    )
    
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, calibration_dir=Path("."), use_degrees=True)
    robot = SOFollower(config)
    robot.connect()
    
    final_target = np.array([TARGET_X, TARGET_Y, TARGET_Z])
    
    print(f"\nConnected! Laying breadcrumbs to (X:{TARGET_X}m, Y:{TARGET_Y}m, Z:{TARGET_Z}m)...")

    try:
        while True:
            obs = robot.get_observation()
            
            # Grab ONLY the current Pan, Lift, and Elbow
            current_degrees = np.array([obs[f"{n}.pos"] for n in ik_joint_names], dtype=float)
            
            current_pose = kin.forward_kinematics(current_degrees)
            current_xyz = current_pose[:3, 3]
            
            vector_to_target = final_target - current_xyz
            total_distance = np.linalg.norm(vector_to_target)
            
            print(f"\n[CURRENT] X:{current_xyz[0]:.3f} Y:{current_xyz[1]:.3f} Z:{current_xyz[2]:.3f} | Dist left: {total_distance:.3f}m")
            
            if total_distance <= XYZ_THRESHOLD:
                print("🎉 Final target reached!")
                break
                
            move_distance = min(STEP_DISTANCE, total_distance)
            direction_unit_vector = vector_to_target / total_distance
            next_xyz = current_xyz + (direction_unit_vector * move_distance)
            
            # 2. THE FIX: Set orientation_weight to absolute 0.0 so it only cares about X, Y, Z
            target_pose = np.eye(4, dtype=float)
            target_pose[:3, 3] = next_xyz
            
            target_angles = kin.inverse_kinematics(
                current_joint_pos=current_degrees,
                desired_ee_pose=target_pose,
                position_weight=1.0,       
                orientation_weight=0.0    
            )
            
            print(f"📍 Next Breadcrumb -> X:{next_xyz[0]:.3f}, Y:{next_xyz[1]:.3f}, Z:{next_xyz[2]:.3f}")
            print(f"💡 Required Angles -> Pan:{target_angles[0]:.1f}°, Lift:{target_angles[1]:.1f}°, Elbow:{target_angles[2]:.1f}°")

            # ── INTERACTIVE PAUSE ──────────────────────────────────────────
            user_cmd = input("Press ENTER to take this step, or 'q' to quit: ")
            if user_cmd.strip().lower() == 'q':
                break
                
            action = {k: v for k, v in obs.items() if k.endswith(".pos")}
            
            # Apply the calculated angles to the big joints
            action["shoulder_pan.pos"] = target_angles[0]
            action["shoulder_lift.pos"] = target_angles[1]
            action["elbow_flex.pos"] = target_angles[2]
            
            # 3. THE FIX: Physically lock the wrist straight so the math matches reality
            action["wrist_flex.pos"] = 0.0
            action["wrist_roll.pos"] = 0.0
                
            robot.send_action(action)

    finally:
        robot.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()