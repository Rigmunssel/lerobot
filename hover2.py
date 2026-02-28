import math
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ─────────────────────────────────────────────────────────────
PORT     = "/dev/ttyACM2"
ROBOT_ID = "follower1"

# Hardware geometry (metres)
Z_OFFSET  = -0.0078
L_UPPER   =  0.13
L_FOREARM =  0.14
PAN_X     =  0.08
PAN_Y     =  0.0016

# Angle references (degrees) — "straight up" calibration offsets
SH_VERTICAL = 1.5    # shoulder raw angle when arm points straight up
EL_VERTICAL = -85.2  # elbow raw angle when forearm is collinear with upper arm

# Pan calibration offset added to the computed facing angle
PAN_OFFSET = -4.1

# ── Movement parameters ───────────────────────────────────────────────────────
STEP_DEG               =  2.0  # step size (°) for shoulder / elbow / wrist moves
PAN_STEP_DEG           =  5.0  # step size (°) for pan rotation (faster, less precision needed)
FOREARM_ANGLE_DEG      =  -15  # shoulder_lift + elbow_flex sum kept constant during sweep
TOLERANCE_PAN          =  1.0  # pan "close enough" threshold (°)
TOLERANCE_DEG          =  0.1  # "at target" threshold for vertical moves (°)
APPROACH_TOLERANCE_DEG =  1.0  # good-enough tolerance for approach / grasp poses
GRIPPER_TOLERANCE_DEG  =  1.0  # tight tolerance for gripper open/close

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIPPER_DEFAULT_DEG = 15.3   # open / resting
GRIPPER_CLOSED_DEG  =  3.5   # closed / grasping

# ── Pick / place depth ────────────────────────────────────────────────────────
TARGET_Z_DOWN = 0.07         # Z height (metres) to lower to for pick / place

# ── Fixed approach pose (pan is computed per-target) ─────────────────────────
APPROACH_LIFT_DEG       = -38.0
APPROACH_ELBOW_DEG      =  23.0
APPROACH_WRIST_FLEX_DEG =  117
APPROACH_WRIST_ROLL_DEG =  -1.0

# ── Base / home pose ──────────────────────────────────────────────────────────
GRASP_PAN_DEG        =  -4.1
GRASP_LIFT_DEG       = -105.5
GRASP_ELBOW_DEG      =   96.9
GRASP_WRIST_FLEX_DEG = -100.1
GRASP_WRIST_ROLL_DEG =   -1.0

# ── Board position map ────────────────────────────────────────────────────────
# Run get_positions.py to generate this dict and paste the output here.
# Keys are lowercase square names ("a1"…"h8"), values are (x, y, z) in metres.
BOARD_POSITIONS = {}


# ── Forward kinematics ────────────────────────────────────────────────────────

def _raw_to_rz(raw_sh, raw_el):
    alpha = math.radians(raw_sh - SH_VERTICAL)
    beta  = math.radians(raw_el - EL_VERTICAL)
    gamma = alpha + beta
    R = L_UPPER * math.sin(alpha) + L_FOREARM * math.sin(gamma)
    Z = Z_OFFSET + L_UPPER * math.cos(alpha) + L_FOREARM * math.cos(gamma)
    return R, Z

def get_xyz(raw_pan, raw_sh, raw_el):
    pan_rad = math.radians(-(raw_pan - PAN_OFFSET))
    R, Z    = _raw_to_rz(raw_sh, raw_el)
    return PAN_X + R * math.cos(pan_rad), PAN_Y + R * math.sin(pan_rad), Z


# ── Target solvers ────────────────────────────────────────────────────────────

def compute_pan(tx, ty):
    return -math.degrees(math.atan2(ty - PAN_Y, tx - PAN_X)) + PAN_OFFSET


def solve_shoulder_target(tx, ty, tz):
    """Analytically find shoulder_lift for (tx,ty,tz) with sum constraint."""
    gamma_c  = math.radians(FOREARM_ANGLE_DEG - SH_VERTICAL - EL_VERTICAL)
    R_fore   = L_FOREARM * math.sin(gamma_c)
    Z_fore   = L_FOREARM * math.cos(gamma_c)
    R_target = math.sqrt((tx - PAN_X)**2 + (ty - PAN_Y)**2)
    alpha    = math.atan2(R_target - R_fore, tz - Z_OFFSET - Z_fore)
    sh       = math.degrees(alpha) + SH_VERTICAL
    el       = FOREARM_ANGLE_DEG - sh
    pan      = compute_pan(tx, ty)
    px, py, pz = get_xyz(pan, sh, el)
    residual = math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)
    return sh, el, residual


def solve_elbow_for_z(sh_deg, target_z):
    """
    Find elbow_flex angle (°) that places the end-effector at target_z,
    with shoulder_lift fixed.  Returns None if geometrically unreachable.
    """
    alpha  = math.radians(sh_deg - SH_VERTICAL)
    cos_g  = (target_z - Z_OFFSET - L_UPPER * math.cos(alpha)) / L_FOREARM
    if abs(cos_g) > 1.0:
        return None
    gamma  = math.acos(max(-1.0, min(1.0, cos_g)))
    return math.degrees(gamma - alpha) + EL_VERTICAL


# ── Commanded-state tracking ───────────────────────────────────────────────────
# _cmd stores the last value we sent for every joint.
# send_joints() uses _cmd for all joints not explicitly updated, so partial
# commands (e.g. gripper only) truly hold every other joint in place.

_cmd = {}

def _init_cmd(robot):
    """Seed _cmd from current obs on the first call."""
    global _cmd
    if not _cmd:
        obs  = robot.get_observation()
        _cmd = {k[:-4]: float(v) for k, v in obs.items() if k.endswith(".pos")}

def send_joints(robot, **updates):
    """
    Command specific joints; all others hold their last commanded value.
    Non-specified joints are NOT re-commanded from obs — they hold exactly
    what was last sent, with no obs noise creeping in.
    """
    _cmd.update(updates)
    robot.send_action({f"{k}.pos": v for k, v in _cmd.items()})


# ── Unified step display ───────────────────────────────────────────────────────

def _print_step_table(joints, extra=None):
    """
    joints: list of dicts — name, actual, commanded, target, step
    extra:  optional extra line printed below the table (e.g. Z prediction)
    """
    w = 9
    print(f"\n  {'Joint':<20} {'Actual':>{w}} {'Commanded':>{w+1}} {'Target':>{w}} {'Remaining':>{w+1}} {'Step':>{w+1}}")
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


# ── Step-to-pose (any set of joints) ──────────────────────────────────────────

def step_to_pose(robot, targets, label="Move", tol=APPROACH_TOLERANCE_DEG):
    """
    Move joints in `targets` {name: deg} step-by-step, Enter-confirmed.
    Lead joint (largest Δ) takes STEP_DEG; others scale proportionally.
    Only the joints listed in targets are moved; all others hold via _cmd.
    Returns False on cancel.
    """
    _init_cmd(robot)
    obs       = robot.get_observation()
    # Seed from last commanded value (not obs) so the step calculation is
    # based on what we actually sent, not the potentially noisy sensor reading.
    commanded = {n: _cmd.get(n, float(obs[f"{n}.pos"])) for n in targets}
    print(f"\n--- {label} ---")
    while True:
        remaining = {n: targets[n] - commanded[n] for n in targets}
        max_rem   = max(abs(v) for v in remaining.values())
        if max_rem <= tol:
            print("  All joints at target.")
            return True
        scale  = min(1.0, STEP_DEG / max_rem)
        steps  = {n: remaining[n] * scale for n in targets}
        obs    = robot.get_observation()
        actual = {n: float(obs[f"{n}.pos"]) for n in targets}
        _print_step_table([
            dict(name=n, actual=actual[n], commanded=commanded[n],
                 target=targets[n], step=steps[n])
            for n in targets
        ])
        if input("  Press Enter to take this step (or 'q' to cancel): ").strip().lower() == 'q':
            print("  Cancelled.")
            return False
        new_vals  = {n: commanded[n] + steps[n] for n in targets}
        send_joints(robot, **new_vals)
        commanded = new_vals


# ── Vertical movement (elbow + wrist_flex only) ────────────────────────────────

def go_vertical(robot, pan_deg, sh_deg, target_el, label="Vertical"):
    """
    Lower or raise arm by stepping elbow toward target_el.
    wrist_flex compensates by -Δelbow each step (gripper keeps orientation).
    Only elbow_flex and wrist_flex are moved; all other joints hold via _cmd.
    Returns history: list of (el, wf) starting from initial position,
    suitable for passing to go_vertical_reverse.
    """
    _init_cmd(robot)
    obs          = robot.get_observation()
    commanded_el = float(obs["elbow_flex.pos"])
    commanded_wf = float(obs["wrist_flex.pos"])
    target_wf    = commanded_wf - (target_el - commanded_el)
    history      = [(commanded_el, commanded_wf)]
    print(f"\n--- {label} ---")
    while True:
        remaining_el = target_el - commanded_el
        if abs(remaining_el) < TOLERANCE_DEG:
            obs = robot.get_observation()
            cz  = get_xyz(pan_deg, sh_deg, float(obs["elbow_flex.pos"]))[2]
            print(f"  Done.  Z = {cz:.3f} m")
            return history
        step_el  = math.copysign(min(STEP_DEG, abs(remaining_el)), remaining_el)
        new_el   = commanded_el + step_el
        new_wf   = commanded_wf - step_el
        obs      = robot.get_observation()
        act_el   = float(obs["elbow_flex.pos"])
        act_wf   = float(obs["wrist_flex.pos"])
        curr_z   = get_xyz(pan_deg, sh_deg, act_el)[2]
        after_z  = get_xyz(pan_deg, sh_deg, new_el)[2]
        _print_step_table([
            dict(name="elbow_flex", actual=act_el, commanded=commanded_el,
                 target=target_el, step=step_el),
            dict(name="wrist_flex", actual=act_wf, commanded=commanded_wf,
                 target=target_wf, step=-step_el),
        ], extra=f"Z: {curr_z:+.3f} m  →  {after_z:+.3f} m  (target {TARGET_Z_DOWN:+.3f} m)")
        if input("  Press Enter to take this step (or 'q' to cancel): ").strip().lower() == 'q':
            return history
        send_joints(robot, elbow_flex=new_el, wrist_flex=new_wf)
        commanded_el = new_el
        commanded_wf = new_wf
        history.append((commanded_el, commanded_wf))


def go_vertical_reverse(robot, pan_deg, sh_deg, history, label="Vertical (reversed)"):
    """
    Retrace a go_vertical path in reverse: replay each saved (el, wf)
    position in reverse order.  Exact mirror of the descent — no re-solving.
    Only elbow_flex and wrist_flex move; all other joints hold via _cmd.
    """
    # history[0] = position before descent, history[-1] = bottom position
    # Reverse: play history[-2], ..., history[0]
    targets = list(reversed(history[:-1]))
    if not targets:
        return
    _init_cmd(robot)
    print(f"\n--- {label} ---")
    for (target_el, target_wf) in targets:
        obs     = robot.get_observation()
        act_el  = float(obs["elbow_flex.pos"])
        act_wf  = float(obs["wrist_flex.pos"])
        cmd_el  = _cmd.get("elbow_flex", act_el)
        cmd_wf  = _cmd.get("wrist_flex", act_wf)
        curr_z  = get_xyz(pan_deg, sh_deg, act_el)[2]
        after_z = get_xyz(pan_deg, sh_deg, target_el)[2]
        _print_step_table([
            dict(name="elbow_flex", actual=act_el, commanded=cmd_el,
                 target=target_el, step=target_el - cmd_el),
            dict(name="wrist_flex", actual=act_wf, commanded=cmd_wf,
                 target=target_wf, step=target_wf - cmd_wf),
        ], extra=f"Z: {curr_z:+.3f} m  →  {after_z:+.3f} m")
        if input("  Press Enter to take this step (or 'q' to cancel): ").strip().lower() == 'q':
            return
        send_joints(robot, elbow_flex=target_el, wrist_flex=target_wf)
    obs = robot.get_observation()
    cz  = get_xyz(pan_deg, sh_deg, float(obs["elbow_flex.pos"]))[2]
    print(f"  Done.  Z = {cz:.3f} m")


# ── Phases 1-3 helper ─────────────────────────────────────────────────────────

def move_to_xyz(robot, tx, ty, tz, prefix="", skip_approach=False):
    """
    Run phases 1-3 (pan align → approach pose → shoulder sweep) to reach (tx,ty,tz).
    skip_approach=True skips Phase 2 and sweeps directly from the current shoulder
    position — use this when the arm is already in approach configuration (e.g. after
    pick, before drop).  All non-pan/shoulder joints hold their current commanded value.
    Returns (pan_deg, commanded_sh) on success, None if user cancelled.
    """
    _init_cmd(robot)
    pan_deg = compute_pan(tx, ty)
    shoulder_target, elbow_target, ik_err = solve_shoulder_target(tx, ty, tz)

    print(f"\n{'='*60}")
    print(f"  {prefix} Target XYZ   : X={tx:.3f}  Y={ty:.3f}  Z={tz:.3f}")
    print(f"  Computed pan   : {pan_deg:.1f}°  (offset {PAN_OFFSET:+.1f}°)")
    print(f"  Target shoulder: {shoulder_target:.1f}°   elbow: {elbow_target:.1f}°"
          f"  (sum={shoulder_target+elbow_target:.1f}°)")
    print(f"  IK residual    : {ik_err*1000:.1f} mm")

    # ── Phase 1: align pan ────────────────────────────────────────────────────
    print(f"\n--- {prefix} Phase 1: Align pan → {pan_deg:.1f}° ---")
    while True:
        obs      = robot.get_observation()
        curr_pan = float(obs["shoulder_pan.pos"])
        curr_sh  = float(obs["shoulder_lift.pos"])
        curr_el  = float(obs["elbow_flex.pos"])
        cx, cy, cz = get_xyz(curr_pan, curr_sh, curr_el)
        diff_pan   = pan_deg - curr_pan
        print(f"\n  [TARGET  POS]  X:{tx:+.3f}  Y:{ty:+.3f}  Z:{tz:+.3f}")
        print(f"  [CURRENT POS]  X:{cx:+.3f}  Y:{cy:+.3f}  Z:{cz:+.3f}")
        if abs(diff_pan) <= TOLERANCE_PAN:
            print("  Pan aligned.")
            break
        step_pan = math.copysign(min(PAN_STEP_DEG, abs(diff_pan)), diff_pan)
        _print_step_table([
            dict(name="shoulder_pan", actual=curr_pan,
                 commanded=_cmd.get("shoulder_pan", curr_pan),
                 target=pan_deg, step=step_pan),
        ])
        if input("  Press Enter to take this step (or 'q' to cancel): ").strip().lower() == 'q':
            return None
        send_joints(robot, shoulder_pan=curr_pan + step_pan)

    # ── Phase 2: approach pose (skipped when already in approach config) ──────
    if not skip_approach:
        if not step_to_pose(robot, {
                "shoulder_pan":  pan_deg,
                "shoulder_lift": APPROACH_LIFT_DEG,
                "elbow_flex":    APPROACH_ELBOW_DEG,
                "wrist_flex":    APPROACH_WRIST_FLEX_DEG,
                "wrist_roll":    APPROACH_WRIST_ROLL_DEG,
                "gripper":       GRIPPER_DEFAULT_DEG,
            }, label=f"{prefix} Phase 2: Approach pose"):
            return None

    # ── Phase 3: sweep shoulder/elbow to target ───────────────────────────────
    # Start from current commanded shoulder (either APPROACH_LIFT_DEG after Phase 2,
    # or wherever the arm currently is when skipping Phase 2).
    obs = robot.get_observation()
    commanded_sh = _cmd.get("shoulder_lift", float(obs["shoulder_lift.pos"]))
    commanded_el = _cmd.get("elbow_flex",    float(obs["elbow_flex.pos"]))
    print(f"\n--- {prefix} Phase 3: Sweep shoulder {commanded_sh:.1f}° → {shoulder_target:.1f}° ---")
    while True:
        remaining = shoulder_target - commanded_sh
        if abs(remaining) <= TOLERANCE_DEG:
            obs = robot.get_observation()
            cx, cy, cz = get_xyz(pan_deg, float(obs["shoulder_lift.pos"]),
                                 float(obs["elbow_flex.pos"]))
            err_mm = math.sqrt((cx-tx)**2+(cy-ty)**2+(cz-tz)**2)*1000.0
            print(f"\n  At target.  [FINAL POS]  X:{cx:+.3f}  Y:{cy:+.3f}  Z:{cz:+.3f}  (err {err_mm:.1f} mm)")
            break
        step_sh  = math.copysign(min(STEP_DEG, abs(remaining)), remaining)
        new_sh   = commanded_sh + step_sh
        new_el   = FOREARM_ANGLE_DEG - new_sh
        step_el  = new_el - commanded_el
        obs      = robot.get_observation()
        act_sh   = float(obs["shoulder_lift.pos"])
        act_el   = float(obs["elbow_flex.pos"])
        cx, cy, cz = get_xyz(pan_deg, act_sh, act_el)
        err_mm   = math.sqrt((cx-tx)**2+(cy-ty)**2+(cz-tz)**2)*1000.0
        print(f"\n  [TARGET  POS]  X:{tx:+.3f}  Y:{ty:+.3f}  Z:{tz:+.3f}")
        print(f"  [CURRENT POS]  X:{cx:+.3f}  Y:{cy:+.3f}  Z:{cz:+.3f}  (err {err_mm:.1f} mm)")
        _print_step_table([
            dict(name="shoulder_lift", actual=act_sh, commanded=commanded_sh,
                 target=shoulder_target, step=step_sh),
            dict(name="elbow_flex",    actual=act_el, commanded=commanded_el,
                 target=elbow_target,   step=step_el),
        ], extra=f"sum after step: {new_sh:.2f}° + {new_el:.2f}° = {new_sh+new_el:.2f}°"
                 f"  (target {FOREARM_ANGLE_DEG}°)")
        if input("  Press Enter to take this step (or 'q' to cancel): ").strip().lower() == 'q':
            return None
        send_joints(robot, shoulder_lift=new_sh, elbow_flex=new_el)
        commanded_sh = new_sh
        commanded_el = new_el

    return pan_deg, commanded_sh


def parse_xyz(prompt):
    """Ask user for X Y Z; returns (x,y,z) or None on quit/bad input."""
    raw = input(prompt).strip()
    if raw.lower() == 'q':
        return None
    parts = raw.split()
    if len(parts) != 3:
        print("  Need exactly 3 values.")
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        print("  Invalid numbers.")
        return None


def parse_move(prompt):
    """
    Accept a chess move like 'd2 d4'; looks up coordinates in BOARD_POSITIONS.
    Returns ((x1,y1,z1), (x2,y2,z2)) or None on quit / unknown square.
    """
    raw = input(prompt).strip().lower()
    if raw == 'q':
        return None
    parts = raw.split()
    if len(parts) != 2:
        print("  Need two squares, e.g. 'd2 d4'")
        return None
    src, dst = parts
    missing = [s for s in (src, dst) if s not in BOARD_POSITIONS]
    if missing:
        print(f"  Unknown square(s): {missing}  — run get_positions.py first")
        return None
    return BOARD_POSITIONS[src], BOARD_POSITIONS[dst]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, calibration_dir=Path("."), use_degrees=True)
    robot  = SOFollower(config)
    robot.connect()
    _init_cmd(robot)

    try:
        while True:
            # ── Get move from chess notation ──────────────────────────────────
            print("\n" + "="*60)
            move = parse_move("Enter move (e.g. 'd2 d4', or 'q' to quit): ")
            if move is None:
                break
            (tx1, ty1, tz1), (tx2, ty2, tz2) = move

            # ── Move to pick position ─────────────────────────────────────────
            result = move_to_xyz(robot, tx1, ty1, tz1, prefix="[PICK]")
            if result is None:
                continue
            pan1, sh1 = result

            # ── Go down at pick position ──────────────────────────────────────
            el_down1 = solve_elbow_for_z(sh1, TARGET_Z_DOWN)
            if el_down1 is None:
                print(f"  Z={TARGET_Z_DOWN} m unreachable from this shoulder angle — skip.")
                continue
            down_history1 = go_vertical(robot, pan1, sh1, el_down1, label="Go down (pick)")

            # ── Close gripper ─────────────────────────────────────────────────
            step_to_pose(robot, {"gripper": GRIPPER_CLOSED_DEG},
                         label="Close gripper", tol=GRIPPER_TOLERANCE_DEG)

            # ── Go back up — exact reverse of the descent ─────────────────────
            go_vertical_reverse(robot, pan1, sh1, down_history1, label="Go up (pick)")

            # ── Raise to approach height before panning (safe for rotation) ─────
            # This ensures FOREARM_ANGLE_DEG condition and safe height regardless
            # of where sh1 landed.
            step_to_pose(robot, {
                "shoulder_lift": APPROACH_LIFT_DEG,
                "elbow_flex":    APPROACH_ELBOW_DEG,
            }, label="Raise to approach height")

            # ── Move to drop position (arm already in approach config) ──────────
            result2 = move_to_xyz(robot, tx2, ty2, tz2, prefix="[DROP]", skip_approach=True)
            if result2 is None:
                continue
            pan2, sh2 = result2

            # ── Go down at drop position ──────────────────────────────────────
            el_down2 = solve_elbow_for_z(sh2, TARGET_Z_DOWN)
            if el_down2 is None:
                print(f"  Z={TARGET_Z_DOWN} m unreachable from this shoulder angle — skip.")
                continue
            down_history2 = go_vertical(robot, pan2, sh2, el_down2, label="Go down (drop)")

            # ── Open gripper ──────────────────────────────────────────────────
            step_to_pose(robot, {"gripper": GRIPPER_DEFAULT_DEG},
                         label="Open gripper", tol=GRIPPER_TOLERANCE_DEG)

            # ── Go back up — exact reverse of the descent ─────────────────────
            go_vertical_reverse(robot, pan2, sh2, down_history2, label="Go up (drop)")

            # ── Return to base pose ───────────────────────────────────────────
            step_to_pose(robot, {
                "shoulder_pan":  GRASP_PAN_DEG,
                "shoulder_lift": GRASP_LIFT_DEG,
                "elbow_flex":    GRASP_ELBOW_DEG,
                "wrist_flex":    GRASP_WRIST_FLEX_DEG,
                "wrist_roll":    GRASP_WRIST_ROLL_DEG,
                "gripper":       GRIPPER_DEFAULT_DEG,
            }, label="Return to base")

    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
