import time
import json
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM"
ROBOT_ID = "follower1"
CALIBRATION_DIR = Path(".")

# Your specific "Home" angles (Matches the recording script)
HOME_POS = {
    "shoulder_pan.pos": -4.7,
    "shoulder_lift.pos": -106.5,
    "elbow_flex.pos": 96.5,
    "wrist_flex.pos": -100.9,
    "wrist_roll.pos": 4.7,
    "gripper.pos": 50.0
}

def move_smoothly(robot, target_angles, duration_sec=3.0, steps=60):
    """Safely glides the arm to a specific posture."""
    obs = robot.get_observation()
    start_angles = {k: obs[k] for k in target_angles.keys()}
    current_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
    
    pause_time = duration_sec / steps
    for step in range(1, steps + 1):
        progress = step / steps
        for joint, final_angle in target_angles.items():
            current_action[joint] = start_angles[joint] + ((final_angle - start_angles[joint]) * progress)
        robot.send_action(current_action)
        time.sleep(pause_time)

def play_file(robot, filename):
    # 1. Load Recording
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found.")
        return

    print(f"Loaded {len(data)} frames from {filename}.")

    # 2. Sync to Home/Start Position
    print("\n[Step 1/3] Moving to HOME/START position...")
    move_smoothly(robot, HOME_POS, duration_sec=3.0)
    
    # 3. Playback
    input("\n[Step 2/3] Ready. Press ENTER to play the recording... ")
    print(" >> PLAYING... <<")
    
    for i in range(len(data)):
        frame = data[i]
        robot.send_action(frame["positions"])
        
        # Calculate timing based on recorded timestamps
        if i < len(data) - 1:
            wait = data[i+1]["timestamp"] - frame["timestamp"]
            # Cap wait time to 0.1s to prevent huge jumps if script lagged
            time.sleep(max(0, min(0.1, wait)))

    print(" >> PLAYBACK FINISHED <<")

    # 4. Return to Home
    print("\n[Step 3/3] Returning to HOME position for safety...")
    move_smoothly(robot, HOME_POS, duration_sec=2.0)
    print("Done.")

def main():
    config = SOFollowerRobotConfig(
        port=PORT, id=ROBOT_ID, calibration_dir=CALIBRATION_DIR, use_degrees=True
    )
    robot = SOFollower(config)
    robot.connect()

    try:
        while True:
            file_to_play = input("\nEnter the filename to replay (e.g. 'move1.json') or 'q' to quit: ").strip()
            if file_to_play.lower() == 'q':
                break
            
            play_file(robot, file_to_play)
            
    finally:
        print("\nShutting down. PLEASE HOLD THE ARM!")
        robot.disconnect()

if __name__ == "__main__":
    main()