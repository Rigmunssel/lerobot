"""
TWO-HAND CHESS ROBOT — HOVER2.PY

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION GUIDE
═══════════════════════════════════════════════════════════════════════════════

WHAT YOU NEED TO CONFIGURE FOR EACH HAND:

1. CONNECTION SETTINGS (required)
   ─────────────────────────────────
   • port         — USB port (e.g. "/dev/ttyACM4")
                    Find with: ls /dev/ttyACM*
   • robot_id     — Config file name without .json (e.g. "follower1")

2. HARDWARE GEOMETRY (measure or use existing calibration)
   ─────────────────────────────────────────────────────────
   • z_offset     — Vertical offset in metres (typically ~-0.008 m)
   • l_upper      — Upper arm length in metres (typically 0.13 m)
   • l_forearm    — Forearm length in metres (typically 0.14 m)
   • pan_x        — Pan axis X offset in metres (typically 0.08 m)
   • pan_y        — Pan axis Y offset in metres (typically 0.0016 m)
   
   → Leave these at hand1's values unless you measure differences

3. ANGLE CALIBRATION (critical — determines accuracy)
   ───────────────────────────────────────────────────
   Move arm STRAIGHT UP (vertical), then read angles with position2.py:
   
   • sh_vertical  — shoulder_lift angle when arm points straight up
   • el_vertical  — elbow_flex angle when forearm is collinear with upper arm
   • pan_offset   — calibration offset added to computed facing angle
   
   → Use position2.py: make arm limp, point it straight up, record angles

4. MOVEMENT POSES (degrees — use hand1 values as starting point)
   ─────────────────────────────────────────────────────────────────
   Safe pose (arm moves here before wrist rotation):
   • safe_lift_deg, safe_elbow_deg
   
   Approach pose (height before descending to board):
   • approach_lift_deg, approach_elbow_deg, approach_wrist_flex_deg, approach_wrist_roll_deg
   
   Base/home pose (resting position):
   • grasp_pan_deg, grasp_lift_deg, grasp_elbow_deg, grasp_wrist_flex_deg, grasp_wrist_roll_deg
   
   → Start with hand1's values, adjust if needed

5. GRIPPER (degrees)
   ───────────────────
   • gripper_default_deg  — Open/resting position
   • gripper_closed_deg   — Closed/grasping position
   
   → Use position2.py to find gripper angles

6. Z-DEPTHS (metres — how low to descend when picking)
   ─────────────────────────────────────────────────────
   • target_z_down       — Normal pick/place height (typically 0.065 m)
   • target_z_down_edge  — Extra-low for edge columns (typically 0.055 m)
   • edge_columns        — Tuple of columns using lower height (e.g. ('a', 'h'))
   
   → Measure board height, subtract piece height

7. BOARD POSITIONS (pan, lift, elbow, wrist_flex in degrees)
   ────────────────────────────────────────────────────────────
   • board_positions     — Dictionary mapping square names to (pan, lift, elbow, wrist) tuples
                          e.g. "a1": (45.0, -5.6, 17.1, 104.9)
   
   → Run get_positions.py to generate these systematically
   → OR use interactive adjustment mode ('y' when prompted during moves)

8. OUT POSITION (for captured pieces)
   ───────────────────────────────────
   • out_position        — Key in board_positions dict (default "out")
                          This is where captured pieces are dropped
   
   → Calibrate an off-board position reachable by the hand

═══════════════════════════════════════════════════════════════════════════════
HOW TO CALIBRATE:

Step 1: Connect hand and make it limp
        python position2.py  (edit PORT and ROBOT_ID first)

Step 2: Move arm straight up, record angles → update sh_vertical, el_vertical

Step 3: Test pan facing angles → adjust pan_offset if needed

Step 4: Generate board positions
        python get_positions.py  (generates grid of positions)
        → Copy output into board_positions dict below

Step 5: Fine-tune positions during first moves
        → Answer 'y' when prompted to adjust
        → Use p/o (pan), l/k (lift), e/w (elbow), j/h (wrist)
        → Press Enter to save, positions auto-save to coordinates_handX.txt

═══════════════════════════════════════════════════════════════════════════════
"""

import ast
import math
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


# ══════════════════════════════════════════════════════════════════════════════
# SHARED movement / timing / tolerance parameters (identical for both hands)
# ══════════════════════════════════════════════════════════════════════════════

STEP_DEG               =  2.0    # step size (°) for shoulder / elbow / wrist moves
STEP_DEG_DOWN          =  1.0    # step size (°) when lowering arm
STEP_DEG_UP            =  1.0    # step size (°) when raising arm
MIN_STEP_DEG           =  0.5    # minimum step (°) — prevents sub-resolution stalls
STEP_WAIT_SEC          =  0.01   # pause between steps for general movements
STEP_WAIT_DOWN_SEC     =  0.05   # pause between steps when lowering arm
STEP_WAIT_UP_SEC       =  0.05   # pause between steps when raising arm
STEP_DEG_GRIPPER       =  1.0    # step size (°) for gripper open/close
STEP_WAIT_GRIPPER_SEC  =  0.05   # pause between steps for gripper
PAN_STEP_DEG           =  10.0   # step size (°) for pan rotation
FOREARM_ANGLE_DEG      =  -15    # shoulder_lift + elbow_flex sum kept constant during sweep
TOLERANCE_PAN          =  0.55   # pan "close enough" threshold (°)
TOLERANCE_DEG          =  0.1    # "at target" threshold for vertical moves (°)
APPROACH_TOLERANCE_DEG =  0.55   # good-enough tolerance for approach / grasp poses
GRIPPER_TOLERANCE_DEG  =  1.0    # tight tolerance for gripper open/close
ADJUST_STEP_DEG        =  1.0    # nudge size (°) per keypress during interactive adjustment


# ══════════════════════════════════════════════════════════════════════════════
# HAND-SPECIFIC CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HandConfig:
    """All parameters specific to one robotic arm."""
    name: str
    port: str
    robot_id: str

    # Hardware geometry (metres)
    z_offset: float
    l_upper: float
    l_forearm: float
    pan_x: float
    pan_y: float

    # Angle references (degrees) — "straight up" calibration offsets
    sh_vertical: float
    el_vertical: float
    pan_offset: float

    # Safe initial pose — lift+elbow moved here before wrist rotation
    safe_lift_deg: float
    safe_elbow_deg: float

    # Fixed approach pose (pan is computed per-target)
    approach_lift_deg: float
    approach_elbow_deg: float
    approach_wrist_flex_deg: float
    approach_wrist_roll_deg: float

    # Base / home pose
    grasp_pan_deg: float
    grasp_lift_deg: float
    grasp_elbow_deg: float
    grasp_wrist_flex_deg: float
    grasp_wrist_roll_deg: float

    # Gripper
    gripper_default_deg: float
    gripper_closed_deg: float

    # Pick / place depth
    target_z_down: float
    target_z_down_edge: float
    edge_columns: tuple

    # Board positions  {square: (pan, lift, elbow, wrist_flex)}
    board_positions: dict = field(default_factory=dict)

    # Per-hand coordinates file for persistence
    coordinates_file: str = ""

    # Key in board_positions for the off-board drop spot (captures)
    out_position: str = "out"


# ══════════════════════════════════════════════════════════════════════════════
# HAND RUNTIME CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Hand:
    """Runtime wrapper: config + robot connection + commanded-state tracking."""

    def __init__(self, config: HandConfig):
        self.cfg   = config
        self.robot = None
        self._cmd  = {}

    # ── connection ────────────────────────────────────────────────────────

    def connect(self):
        rc = SOFollowerRobotConfig(
            port=self.cfg.port, id=self.cfg.robot_id,
            calibration_dir=Path("."), use_degrees=True,
        )
        self.robot = SOFollower(rc)
        self.robot.connect()
        self._init_cmd()

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()

    # ── commanded-state tracking ──────────────────────────────────────────

    def _init_cmd(self):
        """Seed _cmd from current obs on the first call."""
        if not self._cmd and self.robot:
            obs = self.robot.get_observation()
            self._cmd = {k[:-4]: float(v) for k, v in obs.items()
                         if k.endswith(".pos")}

    def send_joints(self, **updates):
        """Command joints; non-specified joints hold their last commanded value."""
        self._cmd.update(updates)
        self.robot.send_action({f"{k}.pos": v for k, v in self._cmd.items()})

    def get_obs(self):
        return self.robot.get_observation()

    # ── forward kinematics (uses hand-specific geometry) ──────────────────

    def _raw_to_rz(self, raw_sh, raw_el):
        c     = self.cfg
        alpha = math.radians(raw_sh - c.sh_vertical)
        beta  = math.radians(raw_el - c.el_vertical)
        gamma = alpha + beta
        R = c.l_upper * math.sin(alpha) + c.l_forearm * math.sin(gamma)
        Z = c.z_offset + c.l_upper * math.cos(alpha) + c.l_forearm * math.cos(gamma)
        return R, Z

    def get_xyz(self, raw_pan, raw_sh, raw_el):
        c       = self.cfg
        pan_rad = math.radians(-(raw_pan - c.pan_offset))
        R, Z    = self._raw_to_rz(raw_sh, raw_el)
        return (c.pan_x + R * math.cos(pan_rad),
                c.pan_y + R * math.sin(pan_rad),
                Z)

    def solve_elbow_for_z(self, sh_deg, target_z):
        """Find elbow angle (°) that places end-effector at target_z."""
        c     = self.cfg
        alpha = math.radians(sh_deg - c.sh_vertical)
        cos_g = (target_z - c.z_offset - c.l_upper * math.cos(alpha)) / c.l_forearm
        if abs(cos_g) > 1.0:
            return None
        gamma = math.acos(max(-1.0, min(1.0, cos_g)))
        return math.degrees(gamma - alpha) + c.el_vertical

    # ── coordinate persistence ────────────────────────────────────────────

    def save_coordinates(self):
        """Write board_positions to this hand's coordinates file."""
        bp   = self.cfg.board_positions
        rows = {}
        for key in bp:
            rows.setdefault(key[0], []).append(key)

        lines = [f"# {self.cfg.name} board positions\n",
                 "BOARD_POSITIONS = {\n"]
        for grp in sorted(rows):
            lines.append(f"    # Row {grp.upper()}\n")
            for key in sorted(rows[grp]):
                pan, sh, el, wf = bp[key]
                lines.append(f'    "{key}": ({pan:.1f}, {sh:.1f}, {el:.1f}, {wf:.1f}),\n')
            lines.append("\n")
        lines.append("}\n")

        with open(self.cfg.coordinates_file, "w") as f:
            f.writelines(lines)

    def load_coordinates(self):
        """Read coordinates file and merge into board_positions."""
        try:
            text  = Path(self.cfg.coordinates_file).read_text()
            start = text.index("{")
            end   = text.rindex("}") + 1
            loaded = ast.literal_eval(text[start:end])
            self.cfg.board_positions.update(loaded)
        except (FileNotFoundError, ValueError, SyntaxError):
            pass


# ══════════════════════════════════════════════════════════════════════════════
# HAND 1  —  columns a, b, c, d  +  e3, e4, e5, e6
# ══════════════════════════════════════════════════════════════════════════════

HAND1_BOARD_POSITIONS = {
    # Row A
    "a1": (45.0, -5.6, 17.1, 104.9),
    "a2": (31.5, -7.0, 18.6, 104.5),
    "a3": (18.0, -8.5, 20.1, 104.0),
    "a4": (4.5, -10.0, 21.6, 103.6),
    "a5": (-9.0, -11.5, 23.1, 103.3),
    "a6": (-22.5, -13.0, 24.6, 103.1),
    "a7": (-37.5, -14.2, 26.0, 102.9),
    "a8": (-59.0, -20.3, 14.3, 110.8),

    # Row B
    "b1": (38.6, 1.2, 7.5, 104.3),
    "b2": (26.5, -0.3, 8.7, 103.9),
    "b3": (14.5, -1.8, 10.0, 103.5),
    "b4": (2.5, -3.3, 11.3, 103.1),
    "b5": (-10.0, -4.8, 12.6, 102.9),
    "b6": (-22.5, -6.3, 13.8, 102.8),
    "b7": (-35.5, -7.8, 15.0, 102.8),
    "b8": (-48.0, -9.0, 16.2, 102.8),

    # Row C
    "c1": (32.0, 7.0, -2.0, 103.8),
    "c2": (20.0, 6.5, -1.5, 103.4),
    "c3": (8.0, 6.0, -1.0, 103.2),
    "c4": (-4.0, 5.5, -0.5, 103.0),
    "c5": (-16.0, 5.0, 0.0, 102.9),
    "c6": (-20.5, 4.5, 0.5, 102.8),
    "c7": (-26.5, 4.0, 1.0, 102.8),
    "c8": (-30.0, 3.5, 1.5, 102.8),

    # Row D
    "d1": (25.8, 13.5, -10.9, 103.4),
    "d2": (17.0, 13.5, -10.9, 103.2),
    "d3": (8.0, 13.5, -10.9, 103.0),
    "d4": (-1.0, 13.5, -10.9, 102.9),
    "d5": (-9.5, 13.5, -10.9, 102.8),
    "d6": (-17.0, 13.5, -10.9, 102.8),
    "d7": (-24.5, 13.5, -10.9, 102.8),
    "d8": (-32.7, 13.5, -10.9, 102.8),

    # Row E (only e3–e6 reachable by hand1)
    "e3": (1.6, 19.9, -20.2, 102.5),
    "e4": (-7.4, 19.9, -20.2, 102.4),
    "e5": (-15.9, 19.9, -20.2, 102.3),
    "e6": (-23.4, 19.9, -20.2, 102.3),

    # Off-board drop position for captured pieces
    "out": (60.0, -5.0, 17.0, 105.0),
}

HAND1_CONFIG = HandConfig(
    name="hand1",
    port="/dev/ttyACM4",
    robot_id="follower1",
    # Hardware geometry
    z_offset=-0.0078,
    l_upper=0.13,
    l_forearm=0.14,
    pan_x=0.08,
    pan_y=0.0016,
    # Angle references
    sh_vertical=1.5,
    el_vertical=-85.2,
    pan_offset=-4.1,
    # Safe pose
    safe_lift_deg=-22.0,
    safe_elbow_deg=22.5,
    # Approach pose
    approach_lift_deg=-38.0,
    approach_elbow_deg=23.0,
    approach_wrist_flex_deg=117.0,
    approach_wrist_roll_deg=-1.0,
    # Base / home pose
    grasp_pan_deg=-4.1,
    grasp_lift_deg=-105.5,
    grasp_elbow_deg=96.9,
    grasp_wrist_flex_deg=-102.0,
    grasp_wrist_roll_deg=-1.0,
    # Gripper
    gripper_default_deg=19.0,
    gripper_closed_deg=3.5,
    # Pick / place depth
    target_z_down=0.065,
    target_z_down_edge=0.055,
    edge_columns=('a', 'h'),
    # Positions & persistence
    board_positions=HAND1_BOARD_POSITIONS,
    coordinates_file="coordinates_hand1.txt",
    out_position="out",
)


# ══════════════════════════════════════════════════════════════════════════════
# HAND 2  —  columns e, f, g, h  +  d3, d4, d5, d6
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: All board positions below are PLACEHOLDERS (0,0,0,0).
#       Run get_positions.py with hand2 connected to calibrate them,
#       or use the interactive adjust mode.

HAND2_BOARD_POSITIONS = {
    # Row D (only d3–d6 reachable by hand2)
    "d3": (0.0, 0.0, 0.0, 0.0),
    "d4": (0.0, 0.0, 0.0, 0.0),
    "d5": (0.0, 0.0, 0.0, 0.0),
    "d6": (0.0, 0.0, 0.0, 0.0),

    # Row E
    "e1": (0.0, 0.0, 0.0, 0.0),
    "e2": (0.0, 0.0, 0.0, 0.0),
    "e3": (0.0, 0.0, 0.0, 0.0),
    "e4": (0.0, 0.0, 0.0, 0.0),
    "e5": (0.0, 0.0, 0.0, 0.0),
    "e6": (0.0, 0.0, 0.0, 0.0),
    "e7": (0.0, 0.0, 0.0, 0.0),
    "e8": (0.0, 0.0, 0.0, 0.0),

    # Row F
    "f1": (0.0, 0.0, 0.0, 0.0),
    "f2": (0.0, 0.0, 0.0, 0.0),
    "f3": (0.0, 0.0, 0.0, 0.0),
    "f4": (0.0, 0.0, 0.0, 0.0),
    "f5": (0.0, 0.0, 0.0, 0.0),
    "f6": (0.0, 0.0, 0.0, 0.0),
    "f7": (0.0, 0.0, 0.0, 0.0),
    "f8": (0.0, 0.0, 0.0, 0.0),

    # Row G
    "g1": (0.0, 0.0, 0.0, 0.0),
    "g2": (0.0, 0.0, 0.0, 0.0),
    "g3": (0.0, 0.0, 0.0, 0.0),
    "g4": (0.0, 0.0, 0.0, 0.0),
    "g5": (0.0, 0.0, 0.0, 0.0),
    "g6": (0.0, 0.0, 0.0, 0.0),
    "g7": (0.0, 0.0, 0.0, 0.0),
    "g8": (0.0, 0.0, 0.0, 0.0),

    # Row H
    "h1": (0.0, 0.0, 0.0, 0.0),
    "h2": (0.0, 0.0, 0.0, 0.0),
    "h3": (0.0, 0.0, 0.0, 0.0),
    "h4": (0.0, 0.0, 0.0, 0.0),
    "h5": (0.0, 0.0, 0.0, 0.0),
    "h6": (0.0, 0.0, 0.0, 0.0),
    "h7": (0.0, 0.0, 0.0, 0.0),
    "h8": (0.0, 0.0, 0.0, 0.0),

    # Off-board drop position for captured pieces
    "out": (0.0, 0.0, 0.0, 0.0),
}

HAND2_CONFIG = HandConfig(
    name="hand2",
    port="/dev/ttyACM5",
    robot_id="follower2",
    # Same default values as hand1 — update after calibrating hand2
    z_offset=-0.0078,
    l_upper=0.13,
    l_forearm=0.14,
    pan_x=0.08,
    pan_y=0.0016,
    sh_vertical=1.5,
    el_vertical=-85.2,
    pan_offset=-8.7,
    safe_lift_deg=-22.0,
    safe_elbow_deg=22.5,
    approach_lift_deg=-38.0,
    approach_elbow_deg=23.0,
    approach_wrist_flex_deg=117.0,
    approach_wrist_roll_deg=-1.0,
    grasp_pan_deg=-4.1,
    grasp_lift_deg=-104.3,
    grasp_elbow_deg=95.6,
    grasp_wrist_flex_deg=-102.0,
    grasp_wrist_roll_deg=-1.0,
    gripper_default_deg=19.0,
    gripper_closed_deg=3.5,
    target_z_down=0.065,
    target_z_down_edge=0.055,
    edge_columns=('a', 'h'),
    board_positions=HAND2_BOARD_POSITIONS,
    coordinates_file="coordinates_hand2.txt",
    out_position="out",
)


# ══════════════════════════════════════════════════════════════════════════════
# SQUARE-TO-HAND ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def _squares_for_columns(cols):
    return {f"{c}{r}" for c in cols for r in "12345678"}

HAND1_SQUARES = _squares_for_columns("abcd") | {"e3", "e4", "e5", "e6"}
HAND2_SQUARES = _squares_for_columns("efgh") | {"d3", "d4", "d5", "d6"}
TRANSFER_SQUARES = sorted(HAND1_SQUARES & HAND2_SQUARES)
# → ['d3','d4','d5','d6','e3','e4','e5','e6']


# ══════════════════════════════════════════════════════════════════════════════
# BOARD STATE TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def initial_board_state():
    """Return a set of occupied squares for the standard chess starting position."""
    occ = set()
    for col in "abcdefgh":
        for row in "12":
            occ.add(f"{col}{row}")
        for row in "78":
            occ.add(f"{col}{row}")
    return occ


def find_free_transfer_square(occupied, src_hand, dst_hand):
    """Return the first transfer square that is empty AND calibrated in both hands."""
    for sq in TRANSFER_SQUARES:
        if sq not in occupied:
            if sq in src_hand.cfg.board_positions and sq in dst_hand.cfg.board_positions:
                return sq
    return None


def print_board(occupied):
    """Print a simple text representation of the board state."""
    print("\n     a  b  c  d  e  f  g  h")
    print("   +------------------------+")
    for row in "87654321":
        pieces = []
        for col in "abcdefgh":
            sq = f"{col}{row}"
            pieces.append(" ■ " if sq in occupied else " · ")
        print(f" {row} |{''.join(pieces)}|")
    print("   +------------------------+")
    print(f"   Occupied: {len(occupied)} squares\n")


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_step_table(joints, extra=None):
    """Pretty-print a step status table for one or more joints."""
    w = 9
    print(f"\n  {'Joint':<20} {'Actual':>{w}} {'Commanded':>{w+1}} "
          f"{'Target':>{w}} {'Remaining':>{w+1}} {'Step':>{w+1}}")
    print(f"  {'-'*74}")
    for j in joints:
        rem  = j['target'] - j['commanded']
        step = j.get('step', 0.0)
        print(f"  {j['name']:<20} {j['actual']:>+{w}.2f}°"
              f" {j['commanded']:>+{w+1}.2f}°"
              f" {j['target']:>+{w}.2f}°"
              f" {rem:>+{w+1}.2f}°"
              f" {step:>+{w+1}.2f}°")
    if extra:
        print(f"  {extra}")


def _getch():
    """Read a single character from stdin without requiring Enter."""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ══════════════════════════════════════════════════════════════════════════════
# MOVEMENT PRIMITIVES — all take a Hand instance as first argument
# ══════════════════════════════════════════════════════════════════════════════

def step_to_pose(hand, targets, label="Move", tol=APPROACH_TOLERANCE_DEG,
                 step_deg=STEP_DEG, wait_sec=STEP_WAIT_SEC):
    """
    Move joints in `targets` {name: deg} step-by-step with timed pauses.
    Lead joint (largest Δ) takes step_deg; others scale proportionally.
    Only the joints listed in targets are moved; all others hold via _cmd.
    """
    hand._init_cmd()
    obs       = hand.get_obs()
    commanded = {n: hand._cmd.get(n, float(obs[f"{n}.pos"])) for n in targets}
    print(f"\n--- [{hand.cfg.name}] {label} ---")
    while True:
        remaining = {n: targets[n] - commanded[n] for n in targets}
        max_rem   = max(abs(v) for v in remaining.values())
        if max_rem <= tol:
            print("  All joints at target.")
            return True
        scale = min(1.0, step_deg / max_rem)
        steps = {}
        for n in targets:
            s     = remaining[n] * scale
            abs_s = max(MIN_STEP_DEG, abs(s))
            abs_s = min(abs_s, abs(remaining[n]))
            steps[n] = math.copysign(abs_s, remaining[n])
        obs    = hand.get_obs()
        actual = {n: float(obs[f"{n}.pos"]) for n in targets}
        _print_step_table([
            dict(name=n, actual=actual[n], commanded=commanded[n],
                 target=targets[n], step=steps[n])
            for n in targets
        ])
        time.sleep(wait_sec)
        new_vals  = {n: commanded[n] + steps[n] for n in targets}
        hand.send_joints(**new_vals)
        commanded = new_vals


def go_vertical(hand, pan_deg, sh_deg, target_el, label="Vertical"):
    """
    Lower or raise arm by stepping elbow toward target_el.
    wrist_flex compensates by -Δelbow each step (gripper keeps orientation).
    Returns history list for go_vertical_reverse.
    """
    hand._init_cmd()
    obs          = hand.get_obs()
    commanded_el = float(obs["elbow_flex.pos"])
    commanded_wf = float(obs["wrist_flex.pos"])
    target_wf    = commanded_wf - (target_el - commanded_el)
    history      = [(commanded_el, commanded_wf)]
    print(f"\n--- [{hand.cfg.name}] {label} ---")
    while True:
        remaining_el = target_el - commanded_el
        if abs(remaining_el) < TOLERANCE_DEG:
            obs = hand.get_obs()
            cz  = hand.get_xyz(pan_deg, sh_deg, float(obs["elbow_flex.pos"]))[2]
            print(f"  Done.  Z = {cz:.3f} m")
            return history
        abs_step = min(STEP_DEG_DOWN, abs(remaining_el))
        abs_step = max(MIN_STEP_DEG, abs_step)
        abs_step = min(abs_step, abs(remaining_el))
        step_el  = math.copysign(abs_step, remaining_el)
        new_el   = commanded_el + step_el
        new_wf   = commanded_wf - step_el
        obs      = hand.get_obs()
        act_el   = float(obs["elbow_flex.pos"])
        act_wf   = float(obs["wrist_flex.pos"])
        curr_z   = hand.get_xyz(pan_deg, sh_deg, act_el)[2]
        after_z  = hand.get_xyz(pan_deg, sh_deg, new_el)[2]
        _print_step_table([
            dict(name="elbow_flex", actual=act_el, commanded=commanded_el,
                 target=target_el, step=step_el),
            dict(name="wrist_flex", actual=act_wf, commanded=commanded_wf,
                 target=target_wf, step=-step_el),
        ], extra=(f"Z: {curr_z:+.3f} m  →  {after_z:+.3f} m  "
                  f"(target {hand.get_xyz(pan_deg, sh_deg, target_el)[2]:+.3f} m)"))
        time.sleep(STEP_WAIT_DOWN_SEC)
        hand.send_joints(elbow_flex=new_el, wrist_flex=new_wf)
        commanded_el = new_el
        commanded_wf = new_wf
        history.append((commanded_el, commanded_wf))


def go_vertical_reverse(hand, pan_deg, sh_deg, history, label="Vertical (reversed)"):
    """Raise arm back to the position recorded before descent (history[0])."""
    if len(history) < 2:
        return
    start_el, start_wf = history[0]
    hand._init_cmd()
    commanded_el = hand._cmd.get("elbow_flex", start_el)
    commanded_wf = hand._cmd.get("wrist_flex", start_wf)
    print(f"\n--- [{hand.cfg.name}] {label} ---")
    while True:
        remaining_el = start_el - commanded_el
        if abs(remaining_el) < TOLERANCE_DEG:
            obs = hand.get_obs()
            cz  = hand.get_xyz(pan_deg, sh_deg, float(obs["elbow_flex.pos"]))[2]
            print(f"  Done.  Z = {cz:.3f} m")
            return
        abs_step = min(STEP_DEG_UP, abs(remaining_el))
        abs_step = max(MIN_STEP_DEG, abs_step)
        abs_step = min(abs_step, abs(remaining_el))
        step_el  = math.copysign(abs_step, remaining_el)
        new_el   = commanded_el + step_el
        new_wf   = start_wf - (new_el - start_el)
        obs      = hand.get_obs()
        act_el   = float(obs["elbow_flex.pos"])
        act_wf   = float(obs["wrist_flex.pos"])
        curr_z   = hand.get_xyz(pan_deg, sh_deg, act_el)[2]
        after_z  = hand.get_xyz(pan_deg, sh_deg, new_el)[2]
        _print_step_table([
            dict(name="elbow_flex", actual=act_el, commanded=commanded_el,
                 target=start_el, step=step_el),
            dict(name="wrist_flex", actual=act_wf, commanded=commanded_wf,
                 target=start_wf, step=-step_el),
        ], extra=f"Z: {curr_z:+.3f} m  →  {after_z:+.3f} m")
        time.sleep(STEP_WAIT_UP_SEC)
        hand.send_joints(elbow_flex=new_el, wrist_flex=new_wf)
        commanded_el = new_el
        commanded_wf = new_wf


# ── Phases 1-3 helper ─────────────────────────────────────────────────────────

def move_to_angles(hand, pan_deg, sh_deg, el_deg, wf_deg, prefix=""):
    """
    Move arm to a board square specified by direct joint angles.
    Phase 1 — pan to pan_deg.
    Phase 2 — sweep shoulder to sh_deg, elbow tracks sh+el=FOREARM_ANGLE_DEG.
    Phase 3 — fine-tune elbow and wrist_flex to exact target angles.
    """
    hand._init_cmd()
    print(f"\n{'='*60}")
    print(f"  [{hand.cfg.name}] {prefix} Target angles:"
          f"  pan={pan_deg:.1f}°  sh={sh_deg:.1f}°  el={el_deg:.1f}°  wf={wf_deg:.1f}°")

    # ── Phase 1: align pan ────────────────────────────────────────────────
    print(f"\n--- [{hand.cfg.name}] {prefix} Phase 1: Align pan → {pan_deg:.1f}° ---")
    while True:
        obs      = hand.get_obs()
        curr_pan = float(obs["shoulder_pan.pos"])
        diff_pan = pan_deg - curr_pan
        if abs(diff_pan) <= TOLERANCE_PAN:
            print("  Pan aligned.")
            break
        step_pan = math.copysign(min(PAN_STEP_DEG, abs(diff_pan)), diff_pan)
        _print_step_table([
            dict(name="shoulder_pan", actual=curr_pan,
                 commanded=hand._cmd.get("shoulder_pan", curr_pan),
                 target=pan_deg, step=step_pan),
        ])
        time.sleep(STEP_WAIT_SEC)
        hand.send_joints(shoulder_pan=curr_pan + step_pan)

    # ── Phase 2: sweep shoulder (elbow maintains sum constraint) ──────────
    obs          = hand.get_obs()
    commanded_sh = hand._cmd.get("shoulder_lift", float(obs["shoulder_lift.pos"]))
    commanded_el = hand._cmd.get("elbow_flex",    float(obs["elbow_flex.pos"]))
    el_at_target = FOREARM_ANGLE_DEG - sh_deg
    print(f"\n--- [{hand.cfg.name}] {prefix} Phase 2: Sweep shoulder "
          f"{commanded_sh:.1f}° → {sh_deg:.1f}° ---")
    while True:
        remaining = sh_deg - commanded_sh
        if abs(remaining) <= TOLERANCE_DEG:
            print("  Shoulder at target.")
            break
        abs_step = min(STEP_DEG, abs(remaining))
        abs_step = max(MIN_STEP_DEG, abs_step)
        abs_step = min(abs_step, abs(remaining))
        step_sh  = math.copysign(abs_step, remaining)
        new_sh   = commanded_sh + step_sh
        new_el   = FOREARM_ANGLE_DEG - new_sh
        step_el  = new_el - commanded_el
        obs      = hand.get_obs()
        act_sh   = float(obs["shoulder_lift.pos"])
        act_el   = float(obs["elbow_flex.pos"])
        _print_step_table([
            dict(name="shoulder_lift", actual=act_sh, commanded=commanded_sh,
                 target=sh_deg, step=step_sh),
            dict(name="elbow_flex",    actual=act_el, commanded=commanded_el,
                 target=el_at_target,  step=step_el),
        ], extra=(f"sum after step: {new_sh:.2f}° + {new_el:.2f}° = "
                  f"{new_sh + new_el:.2f}°  (constraint {FOREARM_ANGLE_DEG}°)"))
        time.sleep(STEP_WAIT_SEC)
        hand.send_joints(shoulder_lift=new_sh, elbow_flex=new_el)
        commanded_sh = new_sh
        commanded_el = new_el

    # ── Phase 3: fine-tune elbow and wrist ────────────────────────────────
    step_to_pose(hand, {"elbow_flex": el_deg, "wrist_flex": wf_deg},
                 label=f"{prefix} Phase 3: Adjust elbow + wrist")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

def adjust_position_interactive(hand, square, pan_deg, sh_deg, el_deg, wf_deg):
    """
    Interactive fine-tuning of a hover position.

    Keys (each nudges ADJUST_STEP_DEG):
      p / o  →  pan  +/-
      l / k  →  lift +/-
      e / w  →  elbow +/-
      j / h  →  wrist +/-
    Enter  →  confirm & save.   Ctrl+C → cancel & revert.
    """
    pan, sh, el, wf = pan_deg, sh_deg, el_deg, wf_deg

    KEY_DELTAS = {
        'p': ('shoulder_pan',  +ADJUST_STEP_DEG),
        'o': ('shoulder_pan',  -ADJUST_STEP_DEG),
        'l': ('shoulder_lift', +ADJUST_STEP_DEG),
        'k': ('shoulder_lift', -ADJUST_STEP_DEG),
        'e': ('elbow_flex',    +ADJUST_STEP_DEG),
        'w': ('elbow_flex',    -ADJUST_STEP_DEG),
        'j': ('wrist_flex',    +ADJUST_STEP_DEG),
        'h': ('wrist_flex',    -ADJUST_STEP_DEG),
    }

    print(f"\n  [{hand.cfg.name}] Adjust '{square}':"
          f"  p/o=pan  l/k=lift  e/w=elbow  j/h=wrist"
          f"  (Enter=save  Ctrl+C=cancel)")

    def _show():
        print(f"\r    pan={pan:+.1f}°  lift={sh:+.1f}°  elbow={el:+.1f}°"
              f"  wrist={wf:+.1f}°   ", end='', flush=True)

    _show()

    while True:
        ch = _getch()

        if ch in ('\r', '\n'):
            print()
            print(f"  '{square}' confirmed:"
                  f" pan={pan:.1f}  lift={sh:.1f}  elbow={el:.1f}  wrist={wf:.1f}")
            return pan, sh, el, wf

        if ch == '\x03':
            print("\n  Adjustment cancelled — reverting to original angles.")
            hand.cfg.board_positions[square] = (pan_deg, sh_deg, el_deg, wf_deg)
            hand.save_coordinates()
            return pan_deg, sh_deg, el_deg, wf_deg

        if ch not in KEY_DELTAS:
            continue

        joint, delta = KEY_DELTAS[ch]
        if   joint == 'shoulder_pan':  pan += delta
        elif joint == 'shoulder_lift': sh  += delta
        elif joint == 'elbow_flex':    el  += delta
        elif joint == 'wrist_flex':    wf  += delta

        hand.cfg.board_positions[square] = (pan, sh, el, wf)
        hand.save_coordinates()

        hand.send_joints(shoulder_pan=pan, shoulder_lift=sh,
                         elbow_flex=el, wrist_flex=wf)
        _show()


# ══════════════════════════════════════════════════════════════════════════════
# PICK-AND-PLACE — full sequence for one hand
# ══════════════════════════════════════════════════════════════════════════════

def pick_and_place(hand, src_sq, dst_sq, adjust_pick=False, adjust_drop=False):
    """
    Full pick-and-place cycle:
      safe → approach → pick → lift → approach → drop → lift → base.
    Returns True on success, False if a position is unreachable.
    """
    bp  = hand.cfg.board_positions
    cfg = hand.cfg

    pan1, sh1, el1, wf1 = bp[src_sq]
    pan2, sh2, el2, wf2 = bp[dst_sq]

    # ── 1. Safe pose + open gripper ───────────────────────────────────────
    step_to_pose(hand, {
        "shoulder_lift": cfg.safe_lift_deg,
        "elbow_flex":    cfg.safe_elbow_deg,
        "gripper":       cfg.gripper_default_deg,
    }, label="Safe pose + gripper open", tol=GRIPPER_TOLERANCE_DEG)

    # ── 2. Rotate wrist down ─────────────────────────────────────────────
    step_to_pose(hand, {"wrist_flex": cfg.approach_wrist_flex_deg},
                 label="Wrist down")

    # ── 3. Raise to full approach height ─────────────────────────────────
    step_to_pose(hand, {
        "shoulder_lift": cfg.approach_lift_deg,
        "elbow_flex":    cfg.approach_elbow_deg,
    }, label="Raise to approach height")

    # ── 4. Move to pick position ─────────────────────────────────────────
    move_to_angles(hand, pan1, sh1, el1, wf1, prefix="[PICK]")

    # ── Optional: fine-tune pick position ─────────────────────────────────
    if adjust_pick:
        ans = input(f"\nAdjust pick position at '{src_sq}'? (y/n): ").strip().lower()
        if ans == 'y':
            pan1, sh1, el1, wf1 = adjust_position_interactive(
                hand, src_sq, pan1, sh1, el1, wf1)

    # ── 5. Go down at pick position ──────────────────────────────────────
    z_down1  = cfg.target_z_down_edge if src_sq[0] in cfg.edge_columns else cfg.target_z_down
    el_down1 = hand.solve_elbow_for_z(sh1, z_down1)
    if el_down1 is None:
        print(f"  Z={z_down1} m unreachable from sh={sh1:.1f}° — skip.")
        return False
    down_history1 = go_vertical(hand, pan1, sh1, el_down1, label="Go down (pick)")

    # ── 6. Close gripper ─────────────────────────────────────────────────
    step_to_pose(hand, {"gripper": cfg.gripper_closed_deg},
                 label="Close gripper", tol=GRIPPER_TOLERANCE_DEG,
                 step_deg=STEP_DEG_GRIPPER, wait_sec=STEP_WAIT_GRIPPER_SEC)

    # ── 7. Go back up ────────────────────────────────────────────────────
    go_vertical_reverse(hand, pan1, sh1, down_history1, label="Go up (pick)")

    # ── 8. Raise to approach height before panning to drop ───────────────
    step_to_pose(hand, {
        "shoulder_lift": cfg.approach_lift_deg,
        "elbow_flex":    cfg.approach_elbow_deg,
    }, label="Raise to approach height")

    # ── 9. Move to drop position ─────────────────────────────────────────
    move_to_angles(hand, pan2, sh2, el2, wf2, prefix="[DROP]")

    # ── Optional: fine-tune drop position ─────────────────────────────────
    if adjust_drop:
        ans = input(f"\nAdjust drop position at '{dst_sq}'? (y/n): ").strip().lower()
        if ans == 'y':
            pan2, sh2, el2, wf2 = adjust_position_interactive(
                hand, dst_sq, pan2, sh2, el2, wf2)

    # ── 10. Go down at drop position ─────────────────────────────────────
    z_down2  = cfg.target_z_down_edge if dst_sq[0] in cfg.edge_columns else cfg.target_z_down
    el_down2 = hand.solve_elbow_for_z(sh2, z_down2)
    if el_down2 is None:
        print(f"  Z={z_down2} m unreachable from sh={sh2:.1f}° — skip.")
        return False
    down_history2 = go_vertical(hand, pan2, sh2, el_down2, label="Go down (drop)")

    # ── 11. Open gripper ─────────────────────────────────────────────────
    step_to_pose(hand, {"gripper": cfg.gripper_default_deg},
                 label="Open gripper", tol=GRIPPER_TOLERANCE_DEG,
                 step_deg=STEP_DEG_GRIPPER, wait_sec=STEP_WAIT_GRIPPER_SEC)

    # ── 12. Go back up ───────────────────────────────────────────────────
    go_vertical_reverse(hand, pan2, sh2, down_history2, label="Go up (drop)")

    # ── 13. Return to base pose ──────────────────────────────────────────
    step_to_pose(hand, {
        "shoulder_pan":  cfg.grasp_pan_deg,
        "shoulder_lift": cfg.grasp_lift_deg,
        "elbow_flex":    cfg.grasp_elbow_deg,
        "wrist_flex":    cfg.grasp_wrist_flex_deg,
        "wrist_roll":    cfg.grasp_wrist_roll_deg,
        "gripper":       cfg.gripper_default_deg,
    }, label="Return to base")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Create hand objects ───────────────────────────────────────────────
    hand1 = Hand(HAND1_CONFIG)
    hand2 = Hand(HAND2_CONFIG)

    # Load saved calibrations (overrides the hardcoded defaults above)
    hand1.load_coordinates()
    hand2.load_coordinates()

    # Connect both arms
    hand1.connect()
    hand2.connect()

    # Board state — standard chess starting position
    occupied = initial_board_state()

    try:
        while True:
            # ── Prompt ────────────────────────────────────────────────────
            print("\n" + "=" * 60)
            raw = input("Enter move (e.g. 'd2 d4'), 'b'=board, 'r'=reset, 'q'=quit: ").strip().lower()

            if raw == 'q':
                break
            if raw == 'b':
                print_board(occupied)
                continue
            if raw == 'r':
                occupied = initial_board_state()
                print("  Board reset to starting position.")
                continue

            parts = raw.split()
            if len(parts) != 2:
                print("  Need two squares, e.g. 'd2 d4'")
                continue

            src, dst = parts

            # ── Validate squares ──────────────────────────────────────────
            all_squares = HAND1_SQUARES | HAND2_SQUARES
            missing = [s for s in (src, dst) if s not in all_squares]
            if missing:
                print(f"  Unknown square(s): {missing}")
                continue

            is_capture = dst in occupied

            # ── Determine hand assignment ─────────────────────────────────
            h1_src = src in HAND1_SQUARES
            h1_dst = dst in HAND1_SQUARES
            h2_src = src in HAND2_SQUARES
            h2_dst = dst in HAND2_SQUARES

            # Prefer a single hand that can reach both src and dst
            if h1_src and h1_dst:
                move_type = "direct"
                move_hand = hand1
            elif h2_src and h2_dst:
                move_type = "direct"
                move_hand = hand2
            else:
                move_type = "cross"
                # The hand that can reach src picks; the other delivers to dst
                if h1_src:
                    src_hand, dst_hand = hand1, hand2
                else:
                    src_hand, dst_hand = hand2, hand1

            # ── Handle capture (remove piece at dst first) ────────────────
            if is_capture:
                # Use the hand that will also do the final drop at dst
                if move_type == "direct":
                    cap_hand = move_hand
                else:
                    cap_hand = dst_hand

                out_sq = cap_hand.cfg.out_position
                if out_sq not in cap_hand.cfg.board_positions:
                    print(f"  No 'out' position calibrated for {cap_hand.cfg.name}!")
                    continue
                if dst not in cap_hand.cfg.board_positions:
                    print(f"  {cap_hand.cfg.name} has no position for '{dst}'!")
                    continue

                print(f"\n>>> CAPTURE: {cap_hand.cfg.name} removes piece from {dst}")
                if not pick_and_place(cap_hand, dst, out_sq):
                    continue
                occupied.discard(dst)

            # ── Execute the move ──────────────────────────────────────────
            if move_type == "direct":
                # Validate positions exist in the chosen hand's dict
                if src not in move_hand.cfg.board_positions:
                    print(f"  {move_hand.cfg.name} has no position for '{src}'")
                    continue
                if dst not in move_hand.cfg.board_positions:
                    print(f"  {move_hand.cfg.name} has no position for '{dst}'")
                    continue

                print(f"\n>>> DIRECT MOVE: {move_hand.cfg.name}  {src} → {dst}")
                if not pick_and_place(move_hand, src, dst,
                                      adjust_pick=True, adjust_drop=True):
                    continue

            else:
                # ── Cross-hand transfer via a common square ───────────────
                transfer_sq = find_free_transfer_square(occupied, src_hand, dst_hand)
                if transfer_sq is None:
                    print("  No free transfer square available!")
                    continue

                # Validate all needed positions exist
                for h, sq in [(src_hand, src), (src_hand, transfer_sq),
                              (dst_hand, transfer_sq), (dst_hand, dst)]:
                    if sq not in h.cfg.board_positions:
                        print(f"  {h.cfg.name} has no position for '{sq}'")
                        break
                else:
                    # No missing positions — proceed
                    print(f"\n>>> CROSS-HAND TRANSFER via {transfer_sq}")
                    print(f"    Step 1: {src_hand.cfg.name}  {src} → {transfer_sq}")
                    print(f"    Step 2: {dst_hand.cfg.name}  {transfer_sq} → {dst}")

                    if not pick_and_place(src_hand, src, transfer_sq,
                                          adjust_pick=True):
                        continue
                    if not pick_and_place(dst_hand, transfer_sq, dst,
                                          adjust_drop=True):
                        continue

            # ── Update board state ────────────────────────────────────────
            occupied.discard(src)
            occupied.add(dst)
            print(f"\n  Board updated: {src} → {dst}")

    finally:
        hand1.disconnect()
        hand2.disconnect()
        print("Both hands disconnected.")


if __name__ == "__main__":
    main()
