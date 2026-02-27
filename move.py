import time
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"  # Your Linux port
ROBOT_ID = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(".")

def initialize_movement(robot, target_pan_angle: float):
    """
    Moves the arm into a safe 'crane' posture while rotating to face the target.
    """
    print("\n[Step 1] Initializing safe movement posture...")
    
    # 1. Grab the current state so we don't accidentally move the wrist/gripper
    obs = robot.get_observation()
    action = {k: v for k, v in obs.items() if k.endswith(".pos")}
    
    # 2. Define our "Safe" angles
    # NOTE: Based on your previous log where resting was -98°, 
    # you may need to tweak these exact numbers to get the perfect "straight up" posture!
    SAFE_SHOULDER_LIFT = 0  # Adjust this so the main arm points up/safely high
    SAFE_ELBOW_FLEX = -30       # Keeps the arm tucked so it doesn't fall forward
    
    print(f" -> Rotating base (pan) to: {target_pan_angle}°")
    print(f" -> Lifting shoulder to: {SAFE_SHOULDER_LIFT}°")
    print(f" -> Bending elbow to: {SAFE_ELBOW_FLEX}°")
    
    # 3. Update the action dictionary
    action["shoulder_pan.pos"] = target_pan_angle
    action["shoulder_lift.pos"] = SAFE_SHOULDER_LIFT
    action["elbow_flex.pos"] = SAFE_ELBOW_FLEX
    
    # 4. Send the command (they will all move simultaneously)
    robot.send_action(action)
    
    # Give the arm plenty of time to reach this position before doing anything else
    time.sleep(2.0) 
    print("Posture initialized safely.")

def main():
    config = SOFollowerRobotConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    )
    
    robot = SOFollower(config)
    robot.connect()
    print(f"\nConnected: {robot.is_connected}")

    try:
        # Test the initialization function!
        # Let's say our chess piece is at a pan angle of 45 degrees
        target_chess_piece_pan = 45.0 
        
        initialize_movement(robot, target_chess_piece_pan)
        
    finally:
        intput = input("press enter to close the robot, please hold the robot so it does not fall down")
        robot.disconnect()
        print("\nDisconnected.")

if __name__ == "__main__":
    main()