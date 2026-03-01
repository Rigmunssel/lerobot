import time
import json
import sys
import select
import tty
import termios
import threading
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

# ── Configuration ────────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM1"
FOLLOWER_ID = "follower3"
LEADER_PORT = "/dev/ttyACM0"
LEADER_ID = "leader1"
RECORDING_FILE = Path("recording.json")
FPS = 30  # how many positions per second to record / replay


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_follower():
    cfg = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
    )
    return SOFollower(cfg)


def make_leader():
    cfg = SOLeaderTeleopConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
    )
    return SOLeader(cfg)


# ── Key listener (non-blocking) ──────────────────────────────────────────────
def _key_pressed():
    """Return the key if one is waiting on stdin, else None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _go_to_position(follower, target, steps=60, duration=1.5):
    """
    Smoothly move the follower to *target* over *duration* seconds.
    """
    # read where the arm is right now
    obs = follower.get_observation()
    start = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

    dt = duration / steps
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = {k: start[k] + alpha * (target[k] - start[k]) for k in target}
        follower.send_action(interp)
        time.sleep(dt)


# ── Record ───────────────────────────────────────────────────────────────────
def record(output_path: Path = RECORDING_FILE):
    """
    Teleop loop: move the leader arm and the follower mirrors it.
    Joint positions are saved every frame.
      R       → restart: move back to start position & record from scratch
      Ctrl+C  → stop and save
    """
    follower = make_follower()
    leader = make_leader()

    follower.connect()
    leader.connect()

    dt = 1.0 / FPS

    # put terminal in raw/cbreak mode so we can read single keys
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    frames = []
    start_pos = None  # captured from the first frame of each take

    try:
        while True:  # outer loop: each "take"
            frames = []
            start_pos = None
            print(f"\n🔴  RECORDING  –  move the leader arm")
            print(f"   [R] restart  |  [Ctrl+C] stop & save")
            print(f"   saving to: {output_path.resolve()}\n")

            restart = False
            while True:  # inner loop: frame-by-frame
                t0 = time.perf_counter()

                # ── check for keypress ────────────────────────────
                key = _key_pressed()
                if key and key.lower() == "r":
                    restart = True
                    break

                # ── teleop ────────────────────────────────────────
                action = leader.get_action()
                follower.send_action(action)

                frame = {k: float(v) for k, v in action.items()}
                frame["_t"] = time.time()
                frames.append(frame)

                # remember very first position for resets
                if start_pos is None:
                    start_pos = {k: v for k, v in frame.items() if not k.startswith("_")}

                # ── keep constant rate ────────────────────────────
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            if restart:
                print(f"\n🔄  RESTARTING – moving back to start position...")
                if start_pos:
                    _go_to_position(follower, start_pos)
                print("   ...ready!")
                continue  # back to outer loop → new take
            break  # shouldn't reach here, but just in case

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
    finally:
        # restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # save FIRST so the recording is never lost
    if frames:
        output_path.write_text(json.dumps(frames, indent=2))
        print(f"✅  Saved {len(frames)} frames to {output_path}")
    else:
        print("⚠️   No frames recorded")

    # disconnect gracefully – motors may be in overload after Ctrl+C
    for name, dev in [("leader", leader), ("follower", follower)]:
        try:
            dev.disconnect()
        except RuntimeError as e:
            print(f"⚠️   Could not cleanly disconnect {name}: {e}")


# ── Replay ───────────────────────────────────────────────────────────────────
def replay(input_path: Path = RECORDING_FILE):
    """
    Replay a previously recorded JSON file on the follower arm.
    Press Ctrl+C to stop early.
    """
    if not input_path.exists():
        print(f"❌  Recording not found: {input_path}")
        return

    frames = json.loads(input_path.read_text())
    if not frames:
        print("❌  Recording is empty")
        return

    follower = make_follower()
    follower.connect()

    # figure out per-frame delay from timestamps (fall back to FPS)
    dt = 1.0 / FPS

    print(f"\n▶️   REPLAYING {len(frames)} frames from {input_path}")
    print(f"   (Ctrl+C to stop early)\n")

    try:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()

            # strip metadata key before sending
            action = {k: v for k, v in frame.items() if not k.startswith("_")}
            follower.send_action(action)

            # use recorded timestamps for timing when available
            if i + 1 < len(frames) and "_t" in frame and "_t" in frames[i + 1]:
                wait = frames[i + 1]["_t"] - frame["_t"]
            else:
                wait = dt

            elapsed = time.perf_counter() - t0
            if elapsed < wait:
                time.sleep(wait - elapsed)

    except KeyboardInterrupt:
        print("\n⏹  Stopped early")

    try:
        follower.disconnect()
    except RuntimeError as e:
        print(f"⚠️   Could not cleanly disconnect follower: {e}")
    print(f"✅  Replay finished")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    usage = (
        "Usage:\n"
        "  python record1.py record  [output.json]   – teleoperate & record\n"
        "  python record1.py replay  [input.json]    – replay a recording\n"
    )

    if len(sys.argv) < 2:
        print(usage)
        return

    cmd = sys.argv[1].lower()
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else RECORDING_FILE

    if cmd == "record":
        record(path)
    elif cmd == "replay":
        replay(path)
    else:
        print(f"Unknown command: {cmd}\n")
        print(usage)


if __name__ == "__main__":
    main()

