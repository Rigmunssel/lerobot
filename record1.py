import time
import json
import threading
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM4"
FOLLOWER_ID = "follower2"
LEADER_PORT = "/dev/ttyACM5"
LEADER_ID = "leader1"
CALIBRATION_DIR = Path(".")

