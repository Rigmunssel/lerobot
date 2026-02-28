"""
get_positions.py — chess board position calibration

Torque is released so you can physically hand-guide the arm to the centre
of each reference square.  Press Enter to record; the script bilinearly
interpolates (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) for all
64 squares and prints a BOARD_POSITIONS dict ready to paste into hover2.py.

Reference point order (position arm at the CENTRE of each square):
  A1 — column A (left),   rank 1  (near / White back rank)
  E1 — column E (centre), rank 1
  A8 — column A (left),   rank 8  (far  / Black back rank)
  E8 — column E (centre), rank 8

Columns F/G/H are extrapolated linearly beyond E.
"""

from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Robot config — keep in sync with hover2.py ───────────────────────────────
PORT     = "/dev/ttyACM3"
ROBOT_ID = "follower1"

# Reference points — two reachable columns (A and E) at both near and far ranks
CORNERS = [
    ("a1", "A1  —  column A (left),   rank 1  (near / White back rank)"),
    ("e1", "E1  —  column E (centre), rank 1"),
    ("a8", "A8  —  column A (left),   rank 8  (far  / Black back rank)"),
    ("e8", "E8  —  column E (centre), rank 8"),
]


# ── Read joint angles ─────────────────────────────────────────────────────────

def get_angles(robot):
    """Return (pan, sh, el, wf) from current joint positions."""
    obs = robot.get_observation()
    return (
        round(float(obs["shoulder_pan.pos"]),  2),
        round(float(obs["shoulder_lift.pos"]), 2),
        round(float(obs["elbow_flex.pos"]),    2),
        round(float(obs["wrist_flex.pos"]),    2),
    )


# ── Bilinear interpolation ────────────────────────────────────────────────────

def bilinear(corners, col, row):
    """
    corners: dict 'a1','e1','a8','e8' → (pan, sh, el, wf)
    col: 0 = A … 7 = H  (E is at index 4)
    row: 0 = rank 1 … 7 = rank 8

    t = col / 4  so that t=0 → column A, t=1 → column E.
    Columns F/G/H (col 5/6/7) have t > 1 and are linearly extrapolated.
    """
    t   = col / 4.0
    s   = row / 7.0
    p00 = corners['a1']
    p10 = corners['e1']
    p01 = corners['a8']
    p11 = corners['e8']
    return tuple(
        round((1-t)*(1-s)*p00[i] + t*(1-s)*p10[i] +
              (1-t)*s    *p01[i] + t*s    *p11[i], 2)
        for i in range(4)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID,
                                   calibration_dir=Path("."), use_degrees=True)
    robot = SOFollower(config)
    robot.connect()

    try:
        corners = {}
        for key, label in CORNERS:
            print(f"\n{'='*60}")
            print(f"  Next corner: {label}")
            print(f"  Position the arm at the CENTRE of the square.")
            print(f"  Torque is OFF — move the arm by hand.")
            print(f"  Press Enter when in position (or 'q' to quit).")

            robot.bus.disable_torque()
            cmd = input("  > ").strip().lower()
            robot.bus.enable_torque()

            if cmd == 'q':
                print("Aborted.")
                return

            pan, sh, el, wf = get_angles(robot)
            corners[key] = (pan, sh, el, wf)
            print(f"  Recorded {key.upper()}: pan={pan}  sh={sh}  el={el}  wf={wf}")

        # ── Interpolate all 64 squares ────────────────────────────────────────
        files     = 'abcdefgh'
        positions = {}
        for ci, f in enumerate(files):
            for ri in range(8):
                sq = f"{f}{ri + 1}"
                positions[sq] = bilinear(corners, ci, ri)

        # ── Print dict ────────────────────────────────────────────────────────
        print("\n\n# ── Paste this into hover2.py ──────────────────────────────")
        print("BOARD_POSITIONS = {")
        for rank in range(8, 0, -1):
            for f in files:
                sq             = f"{f}{rank}"
                pan, sh, el, wf = positions[sq]
                print(f'    "{sq}": ({pan}, {sh}, {el}, {wf}),')
            print()
        print("}")

    finally:
        try:
            robot.bus.enable_torque()
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
