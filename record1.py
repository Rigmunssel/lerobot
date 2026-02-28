import time
import json
import threading
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM5"
FOLLOWER_ID = "follower1"
LEADER_PORT = "/dev/ttyACM4"
LEADER_ID = "leader1"
CALIBRATION_DIR = Path(".")

# Your specific "Home" angles
HOME_POS = {
    "shoulder_pan.pos": -4.7,
    "shoulder_lift.pos": -106.5,
    "elbow_flex.pos": 96.5,
    "wrist_flex.pos": -100.9,
    "wrist_roll.pos": 4.7,
    "gripper.pos": 50.0  # Default open
}

# Global variables
is_running = True
is_recording = False
recorded_data = []
teleop_enabled = True # New flag to pause teleop during homing

def move_smoothly(robot, target_angles, duration_sec=3.0, steps=60):
    """Interpolates movement so the arm doesn't snap violently."""
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

def teleoperation_loop(leader, follower):
    global is_running, is_recording, recorded_data, teleop_enabled
    hz = 50
    cycle_time = 1.0 / hz
    
    while is_running:
        start_time = time.time()
        if teleop_enabled:
            leader_obs = leader.get_observation()
            action = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}
            follower.send_action(action)
            
            if is_recording:
                recorded_data.append({"timestamp": time.time(), "positions": action})
            
        elapsed = time.time() - start_time
        time.sleep(max(0.0, cycle_time - elapsed))

def set_leader_torque(leader, enabled=True):
    val = 1 if enabled else 0
    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for m in motors:
        leader.bus.write("Torque_Enable", m, val)

def main():
    global is_running, is_recording, recorded_data, teleop_enabled
    
    # 1. Init Robots
    f_config = SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, calibration_dir=CALIBRATION_DIR, use_degrees=True)
    follower = SOFollower(f_config)
    follower.connect()

    l_config = SOFollowerRobotConfig(port=LEADER_PORT, id=LEADER_ID, calibration_dir=CALIBRATION_DIR, use_degrees=True)
    leader = SOFollower(l_config)
    leader.connect()

    # 2. Start teleop thread
    set_leader_torque(leader, enabled=False) # Start limp
    teleop_thread = threading.Thread(target=teleoperation_loop, args=(leader, follower))
    teleop_thread.daemon = True
    teleop_thread.start()

    try:
        while True:
            print("\n" + "="*50)
            name = input("Enter name to record (or 'q' to quit): ").strip()
            if name.lower() == 'q': break
            
            # --- HOMING SEQUENCE ---
            print("\n[Homing] Moving arms to default position...")
            teleop_enabled = False  # Stop background mirroring
            set_leader_torque(leader, enabled=True) # Give leader "muscles" to move itself
            
            # Move both simultaneously
            h1 = threading.Thread(target=move_smoothly, args=(leader, HOME_POS))
            h2 = threading.Thread(target=move_smoothly, args=(follower, HOME_POS))
            h1.start(); h2.start()
            h1.join(); h2.join()
            
            set_leader_torque(leader, enabled=False) # Make leader limp again
            teleop_enabled = True # Resume teleop
            # -----------------------

            input(f"\nArms are Homed. Press ENTER to START recording '{name}'...")
            recorded_data = []
            is_recording = True
            print(" >> RECORDING LIVE... <<")
            
            input("Press ENTER to STOP recording...")
            is_recording = False
            
            with open(f"{name}.json", "w") as f:
                json.dump(recorded_data, f, indent=2)
            print(f"Saved {len(recorded_data)} frames to {name}.json")
            
    finally:
        is_running = False
        teleop_thread.join()
        follower.disconnect(); leader.disconnect()

if __name__ == "__main__":
    main()