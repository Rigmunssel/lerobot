import time
import json
import sys
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration (same as record1.py) ───────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "follower2"
CALIBRATION_DIR = Path(".")
FPS = 30

# ── Recordings to play back-to-back ─────────────────────────────────────────
RECORDINGS = [
    Path("e4.json"),
    Path("kh5.json"),

    Path("Qxg5T.json"),
    Path("Qxg5M.json"),

    Path("Bb5.json"),

    Path("Qxg7T.json"),
    Path("Qxg7M.json"),

    Path("d4.json"),

    Path("Bxc6T.json"),
    Path("Bxc6M.json"),

    Path("Bg5.json"),
    Path("Nf3.json"),
    Path("Be7.json"),

    Path("Nxe5T.json"),
    Path("Nxe5T.json"),

    Path("Ke2.json"),

    Path("Kxf3T.json"),
    Path("Kxf3M.json"),

    Path("dxe5T.json"),
    Path("dxe5M.json"),

    Path("Kxg3T.json"),
    Path("Kxg3M.json"),

    Path("f4.json"),
    Path("Kh4.json"),

    Path("f4.json"),
    Path("Kh5.json"),
    Path("h3.json"),
    Path("h4.json"),   
]
SPEED = 2.0  # playback speed multiplier (2.0 = twice as fast)
HOME_FILE = Path("home.json")  # arm returns here at the end


def make_follower():
    cfg = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    return SOFollower(cfg)


def go_to_position(follower, target, steps=60, duration=1.5):
    """Smoothly move the follower to *target* over *duration* seconds."""
    obs = follower.get_observation()
    start = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    dt = duration / steps
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = {k: start[k] + alpha * (target[k] - start[k]) for k in target}
        follower.send_action(interp)
        time.sleep(dt)


def replay_file(follower, path: Path):
    """Play one recording file on the already-connected follower."""
    if not path.exists():
        print(f"  ❌  File not found: {path}")
        return

    frames = json.loads(path.read_text())
    if not frames:
        print(f"  ⚠️   {path} is empty, skipping")
        return

    dt = 1.0 / FPS

    # smoothly move to the first frame before replaying
    first_action = {k: v for k, v in frames[0].items() if not k.startswith("_")}
    print(f"  ↳ moving to start position...")
    go_to_position(follower, first_action)

    print(f"  ▶️   Playing {len(frames)} frames...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()

        action = {k: v for k, v in frame.items() if not k.startswith("_")}
        follower.send_action(action)

        # use recorded timestamps for timing when available
        if i + 1 < len(frames) and "_t" in frame and "_t" in frames[i + 1]:
            wait = (frames[i + 1]["_t"] - frame["_t"]) / SPEED
        else:
            wait = dt / SPEED

        elapsed = time.perf_counter() - t0
        if elapsed < wait:
            time.sleep(wait - elapsed)


def main():
    # allow overriding files from CLI:  python replay.py file1.json file2.json ...
    recordings = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else RECORDINGS

    follower = make_follower()
    follower.connect()

    print(f"\n▶️   Replaying {len(recordings)} recording(s) back-to-back")
    print(f"   (Ctrl+C to stop early)\n")

    try:
        for idx, rec in enumerate(recordings, 1):
            print(f"[{idx}/{len(recordings)}] {rec}")
            replay_file(follower, rec)
            input("  ⏎  Press Enter to continue...")
            print()

            # return to home position between moves
            if HOME_FILE.exists():
                print(f"  🏠  Playing home.json...")
                replay_file(follower, HOME_FILE)
                input("  ⏎  Press Enter to continue...")
                print()
            else:
                print(f"  ⚠️   {HOME_FILE} not found, skipping home position")
    except KeyboardInterrupt:
        print("\n⏹  Stopped early")

    try:
        follower.disconnect()
    except RuntimeError as e:
        print(f"⚠️   Could not cleanly disconnect follower: {e}")

    print("✅  Done")


if __name__ == "__main__":
    main()
