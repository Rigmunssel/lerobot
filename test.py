"""
Example: Programmatic control of a LeRobot SO101 follower arm (Feetech STS3215 motors).

This script demonstrates how to:
  1. Connect to the arm with a pre-existing calibration file
  2. Read current joint positions
  3. Send position commands (e.g. move gripper)
  4. Disconnect cleanly
"""

import time
from pathlib import Path

# ── Imports ──────────────────────────────────────────────────────────────────
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
# 1. Find your serial port:
#      macOS:  ls /dev/tty.usbmodem*
#      Linux:  ls /dev/ttyUSB*  or  ls /dev/ttyACM*
#    Or run:  python -m lerobot.scripts.find_port   (alias: lerobot-find-port)
PORT = "/dev/ttyACM0"  # ← change to your port

# 2. The `id` must match the calibration filename (without .json).
#    The `calibration_dir` must point to the folder containing that file.
#    Here: ./my_awesome_follower_arm.json
ROBOT_ID = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(".")  # directory that contains <id>.json

config = SOFollowerRobotConfig(
    port=PORT,
    id=ROBOT_ID,
    calibration_dir=CALIBRATION_DIR,
    use_degrees=True,           # positions returned/sent in degrees
    # type is inferred automatically from the registry decorator
    # ("so101_follower" or "so100_follower" – they share the same class)
)


def main():
    # ── Instantiate ──────────────────────────────────────────────────────────
    robot = SOFollower(config)

    # ── Connect (loads calibration, configures PID, sets operating mode) ─────
    robot.connect()
    print(f"Connected: {robot.is_connected}")

    try:
        # ── Read current positions ───────────────────────────────────────────
        obs = robot.get_observation()
        print("\nCurrent joint positions:")
        for key, value in obs.items():
            if key.endswith(".pos"):
                print(f"  {key}: {value:.2f}°")

        # ── Send a position command ──────────────────────────────────────────
        # Actions are dicts of "<motor_name>.pos" → float (degrees if use_degrees=True).
        # Motor names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        #
        # Gripper uses MotorNormMode.RANGE_0_100  →  0 = fully closed, 100 = fully open
        # Body joints use degrees (with use_degrees=True).

        # Example 1: Close the gripper (keep other joints at current position)
        action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        action["gripper.pos"] = 20.0  # mostly closed
        print("\nSending action: close gripper …")
        robot.send_action(action)
        time.sleep(3.0)

        # Example 2: Open the gripper
        action["gripper.pos"] = 80.0  # mostly open
        print("Sending action: open gripper …")
        robot.send_action(action)
        time.sleep(3.0)

        # Example 3: Lift the entire arm up via shoulder_lift
        lift_offset = 30.0  # degrees upward
        action["shoulder_lift.pos"] = obs["shoulder_lift.pos"] + lift_offset
        print(f"Sending action: shoulder_lift +{lift_offset}° (lifting arm up) …")
        robot.send_action(action)
        time.sleep(2.0)

        # Bring it back down
        action["shoulder_lift.pos"] = obs["shoulder_lift.pos"]
        print("Sending action: shoulder_lift back to original …")
        robot.send_action(action)
        time.sleep(2.0)

        # ── Read positions again ─────────────────────────────────────────────
        obs2 = robot.get_observation()
        print("\nFinal joint positions:")
        for key, value in obs2.items():
            if key.endswith(".pos"):
                print(f"  {key}: {value:.2f}°")

    finally:
        # ── Disconnect (disables torque by default) ──────────────────────────
        robot.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
