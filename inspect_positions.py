"""
inspect_positions.py — live Cartesian position display

Releases torque so the arm can be moved by hand.
Prints X Y Z (from forward kinematics) continuously so you can verify
the coordinate system matches what hover2.py expects.

Press Ctrl+C to quit (torque is re-enabled on exit).
"""

import math
import time
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Robot config — keep in sync with hover2.py ───────────────────────────────
PORT     = "/dev/ttyACM3"
ROBOT_ID = "follower1"

# Hardware geometry — keep in sync with hover2.py
Z_OFFSET    = -0.0078
L_UPPER     =  0.13
L_FOREARM   =  0.14
PAN_X       =  0.08
PAN_Y       =  0.0016
SH_VERTICAL =  1.5
EL_VERTICAL = -85.2
PAN_OFFSET  = -4.1

POLL_HZ = 10   # how many readings per second to print


def get_xyz(pan, sh, el):
    alpha  = math.radians(sh  - SH_VERTICAL)
    gamma  = alpha + math.radians(el - EL_VERTICAL)
    R      = L_UPPER * math.sin(alpha) + L_FOREARM * math.sin(gamma)
    Z      = Z_OFFSET + L_UPPER * math.cos(alpha) + L_FOREARM * math.cos(gamma)
    pan_r  = math.radians(-(pan - PAN_OFFSET))
    return PAN_X + R * math.cos(pan_r), PAN_Y + R * math.sin(pan_r), Z


def main():
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID,
                                   calibration_dir=Path("."), use_degrees=True)
    robot = SOFollower(config)
    robot.connect()

    print("Torque OFF — move the arm freely.  Ctrl+C to quit.\n")
    robot.bus.disable_torque()

    try:
        while True:
            obs = robot.get_observation()
            pan = float(obs["shoulder_pan.pos"])
            sh  = float(obs["shoulder_lift.pos"])
            el  = float(obs["elbow_flex.pos"])
            wf  = float(obs["wrist_flex.pos"])
            wr  = float(obs["wrist_roll.pos"])
            gr  = float(obs["gripper.pos"])
            x, y, z = get_xyz(pan, sh, el)
            print(
                f"\r  X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f}"
                f"   pan={pan:+6.1f}°  sh={sh:+6.1f}°  el={el:+6.1f}°"
                f"  wf={wf:+6.1f}°  wr={wr:+6.1f}°  gr={gr:+5.1f}°",
                end="", flush=True
            )
            time.sleep(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        print("\nQuitting...")

    finally:
        robot.bus.enable_torque()
        robot.disconnect()
        print("Torque re-enabled.  Disconnected.")


if __name__ == "__main__":
    main()
