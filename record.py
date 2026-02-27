import time
import json
import threading
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "follower"

LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "leader"

CALIBRATION_DIR = Path(".")

# Global variables for the background teleoperation thread
is_running = True
is_recording = False
recorded_data = []

# ── Background Teleoperation Thread ──────────────────────────────────────────
def teleoperation_loop(leader, follower):
    global is_running, is_recording, recorded_data
    
    # Target 50 frames per second
    hz = 50
    cycle_time = 1.0 / hz
    print("[Teleop] Background thread active. Teleoperation is live.")
    
    while is_running:
        start_time = time.time()
        
        # 1. Read the Leader arm's current position
        leader_obs = leader.get_observation()
        
        # 2. Extract only the position commands (e.g., 'shoulder_pan.pos')
        action = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}
        
        # 3. Send those exact positions to the Follower arm
        follower.send_action(action)
        
        # 4. If the user triggered a recording, save this exact frame
        if is_recording:
            recorded_data.append({
                "timestamp": time.time(),
                "positions": action
            })
            
        # 5. Sleep just enough to maintain our 50Hz frequency
        elapsed = time.time() - start_time
        time.sleep(max(0.0, cycle_time - elapsed))

# ── Main Application ─────────────────────────────────────────────────────────
def main():
    global is_running, is_recording, recorded_data
    
    # 1. Initialize Follower Arm
    print("\nConnecting to Follower...")
    f_config = SOFollowerRobotConfig(
        port=FOLLOWER_PORT, 
        id=FOLLOWER_ID, 
        calibration_dir=CALIBRATION_DIR, 
        use_degrees=True
    )
    follower = SOFollower(f_config)
    follower.connect()
    print(" -> Follower connected.")

    # 2. Initialize Leader Arm
    print("\nConnecting to Leader...")
    l_config = SOFollowerRobotConfig(
        port=LEADER_PORT, 
        id=LEADER_ID, 
        calibration_dir=CALIBRATION_DIR, 
        use_degrees=True
    )
    leader = SOFollower(l_config)
    leader.connect()
    print(" -> Leader connected.")

    # 3. Disable Torque on Leader (Make it limp)
    print("\nTurning off torque on Leader...")
    my_motors = [
        "shoulder_pan", 
        "shoulder_lift", 
        "elbow_flex", 
        "wrist_flex", 
        "wrist_roll", 
        "gripper"
    ]
    
    # Send a low-level command to each motor to turn off its holding torque
    for motor in my_motors:
            leader.bus.write("Torque_Enable", motor, 0)

    # 4. Start the Background Teleoperation Thread
    teleop_thread = threading.Thread(target=teleoperation_loop, args=(leader, follower))
    teleop_thread.daemon = True  # Ensures thread dies if the main script crashes
    teleop_thread.start()
    
    print("\n" + "!"*50)
    print("TELEOP ACTIVE: Move the Leader to move the Follower!")
    print("!"*50)

    # 5. Interactive Recording Loop
    try:
        while True:
            print("\n" + "="*50)
            name = input("Enter a name for the recording (or 'q' to quit): ").strip()
            
            if name.lower() == 'q': 
                break
            if not name: 
                continue
                
            input(f"\nReady? Press ENTER to START recording '{name}'...")
            recorded_data = []      # Clear previous data
            is_recording = True     # Signal the background thread to start saving
            print(" >> RECORDING LIVE... <<")
            
            input("Press ENTER to STOP recording...")
            is_recording = False    # Signal the background thread to stop saving
            print(" >> RECORDING STOPPED <<")
            
            # Save the captured data to a JSON file
            filename = f"{name}.json"
            with open(filename, "w") as f:
                json.dump(recorded_data, f, indent=2)
            print(f"Saved {len(recorded_data)} frames of data to {filename}.")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        
    finally:
        # 6. Clean Shutdown Sequence
        is_running = False  # Stops the background loop
        print("\nShutting down. PLEASE HOLD THE ARMS so they don't fall!")
        time.sleep(0.5)     # Give the thread a moment to finish its last loop
        
        # Safely disconnect both robots
        try:
            follower.disconnect()
            leader.disconnect()
        except Exception as e:
            print(f"Note during disconnect: {e}")
            
        print("Disconnected safely. Goodbye!")

if __name__ == "__main__":
    main()