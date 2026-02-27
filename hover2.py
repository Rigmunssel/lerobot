import os
import math
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

# ── Configuration ────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"
ROBOT_ID = "follower1"

TARGET_X = 0.18
TARGET_Y = -0.15
Z_MIN = 0.2           # Minimum safe height (10cm)

# Hardware geometry (meters)
Z_OFFSET = -0.0015
L_UPPER = 0.13
L_FOREARM = 0.14
PAN_X = 0.087
PAN_Y = 0.0006
PAN_OFFSET = -0.1

# "Straight up" reference angles
SH_VERTICAL = 1.5
EL_VERTICAL = -85.2

SHOULDER_MAX = 108.0
ELBOW_MAX = 96

STEP_DEGREES = 5.0
MAX_ELBOW_STEP = 5.0
TOLERANCE_PAN = 1.0
TOLERANCE_DEG = 1.0    # angular convergence for IK stepping

# ── Math Solvers ──────────────────────────────────────────────────────────────

def raw_to_kinematics(raw_sh, raw_el):
    """Converts raw hardware degrees into standard math angles (radians)."""
    alpha = math.radians(raw_sh - SH_VERTICAL)
    beta = math.radians(raw_el - EL_VERTICAL)

    gamma = alpha + beta
    R = L_UPPER * math.sin(alpha) + L_FOREARM * math.sin(gamma)
    Z = Z_OFFSET + L_UPPER * math.cos(alpha) + L_FOREARM * math.cos(gamma)
    return alpha, beta, R, Z

def get_current_xyz(raw_pan, raw_sh, raw_el):
    """Calculates the current 3D position based purely on motor angles."""
    math_pan_rad = math.radians(-(raw_pan - PAN_OFFSET))
    _, _, R, Z = raw_to_kinematics(raw_sh, raw_el)
    X = PAN_X + R * math.cos(math_pan_rad)
    Y = PAN_Y + R * math.sin(math_pan_rad)
    return X, Y, Z

def solve_ik(R_target, Z_target):
    """
    Solve 2-link IK for (R, Z) target using the cosine rule.
    Returns (raw_sh, raw_el) or None if the target is unreachable.
    Uses elbow-down (positive beta) configuration.
    """
    v = Z_target - Z_OFFSET
    cos_beta = (R_target**2 + v**2 - L_UPPER**2 - L_FOREARM**2) / (2 * L_UPPER * L_FOREARM)
    if abs(cos_beta) > 1.0:
        return None
    beta = math.acos(max(-1.0, min(1.0, cos_beta)))
    k1 = L_UPPER + L_FOREARM * math.cos(beta)
    k2 = L_FOREARM * math.sin(beta)
    alpha = math.atan2(R_target, v) - math.atan2(k2, k1)
    raw_sh = math.degrees(alpha) + SH_VERTICAL
    raw_el = math.degrees(beta) + EL_VERTICAL
    return raw_sh, raw_el

# ── Main Robot Application ───────────────────────────────────────────────────
def main():
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, calibration_dir=Path("."), use_degrees=True)
    robot = SOFollower(config)
    robot.connect()
    
    # Pre-calculate absolute target values
    raw_pan_target = math.degrees(math.atan2(TARGET_Y - PAN_Y, TARGET_X - PAN_X))
    target_pan = -raw_pan_target + PAN_OFFSET
    R_target = math.sqrt((TARGET_X - PAN_X)**2 + (TARGET_Y - PAN_Y)**2)

    # Solve IK up front — fail fast if target is geometrically unreachable
    ik_result = solve_ik(R_target, Z_MIN)
    if ik_result is None:
        print(f"ERROR: Target (R={R_target:.3f}m, Z={Z_MIN:.3f}m) is geometrically unreachable!")
        robot.disconnect()
        return
    target_raw_sh, target_raw_el = ik_result
    print(f"IK target: shoulder={target_raw_sh:.1f}°, elbow={target_raw_el:.1f}°")

    # =========================================================================
    # PHASE 1: PAN ALIGNMENT
    # =========================================================================
    print("\n" + "="*50)
    print("  PHASE 1: ALIGNING BASE")
    print("="*50)
    
    while True:
        obs = robot.get_observation()
        curr_pan = float(obs["shoulder_pan.pos"])
        curr_sh = float(obs["shoulder_lift.pos"])
        curr_el = float(obs["elbow_flex.pos"])
        
        curr_X, curr_Y, curr_Z = get_current_xyz(curr_pan, curr_sh, curr_el)
        diff_pan = target_pan - curr_pan
        
        if abs(diff_pan) <= TOLERANCE_PAN:
            print("\n🎉 Base aligned! Transitioning to Phase 2.")
            break
            
        step_pan = diff_pan if abs(diff_pan) <= STEP_DEGREES else math.copysign(STEP_DEGREES, diff_pan)
        
        print("\n" + "-"*40)
        print(f"[TARGET POS]  X: {TARGET_X:.3f}m | Y: {TARGET_Y:.3f}m | Z-Floor: {Z_MIN:.3f}m")
        print(f"[CURRENT POS] X: {curr_X:.3f}m | Y: {curr_Y:.3f}m | Z: {curr_Z:.3f}m")
        print(f"[CURRENT ANG] Pan: {curr_pan:+.1f}° | Lift: {curr_sh:+.1f}° | Elbow: {curr_el:+.1f}°")
        print(f"[TARGET PAN]  {target_pan:+.1f}° (Distance left: {abs(diff_pan):.1f}°)")
        print(f"[NEXT ACTION] dPan: {step_pan:+.2f}° | dLift: +0.00° | dElbow: +0.00°")
        print("-"*40)
        
        if input("Press ENTER to step (or 'q' to quit): ").strip().lower() == 'q': return
        
        action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        action["shoulder_pan.pos"] += step_pan
        robot.send_action(action)


    # =========================================================================
    # PHASE 2: REACH TARGET (IK-based)
    # =========================================================================
    print("\n" + "="*50)
    print(f"  PHASE 2: MOVING TO TARGET (IK)")
    print(f"  Target: shoulder={target_raw_sh:.1f}°, elbow={target_raw_el:.1f}°")
    print("="*50)

    while True:
        obs = robot.get_observation()
        curr_pan = float(obs["shoulder_pan.pos"])
        curr_sh = float(obs["shoulder_lift.pos"])
        curr_el = float(obs["elbow_flex.pos"])

        curr_X, curr_Y, curr_Z = get_current_xyz(curr_pan, curr_sh, curr_el)

        err_sh = target_raw_sh - curr_sh
        err_el = target_raw_el - curr_el

        if abs(err_sh) <= TOLERANCE_DEG and abs(err_el) <= TOLERANCE_DEG:
            print("\nTarget Reached!")
            break

        step_sh = max(-STEP_DEGREES, min(STEP_DEGREES, err_sh))
        step_el = max(-MAX_ELBOW_STEP, min(MAX_ELBOW_STEP, err_el))

        print("\n" + "-"*40)
        print(f"[TARGET POS]  X: {TARGET_X:.3f}m | Y: {TARGET_Y:.3f}m | Z-Floor: {Z_MIN:.3f}m")
        print(f"[CURRENT POS] X: {curr_X:.3f}m | Y: {curr_Y:.3f}m | Z: {curr_Z:.3f}m")
        print(f"[CURRENT ANG] Pan: {curr_pan:+.1f}° | Lift: {curr_sh:+.1f}° | Elbow: {curr_el:+.1f}°")
        print(f"[TARGET ANG]  Lift: {target_raw_sh:+.1f}° | Elbow: {target_raw_el:+.1f}°")
        print(f"[ERROR]       dLift: {err_sh:+.2f}° | dElbow: {err_el:+.2f}°")
        print(f"[NEXT ACTION] dPan: +0.00° | dLift: {step_sh:+.2f}° | dElbow: {step_el:+.2f}°")
        print("-"*40)

        if input("Press ENTER to step (or 'q' to quit): ").strip().lower() == 'q': return

        action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        action["shoulder_lift.pos"] += step_sh
        action["elbow_flex.pos"] += step_el
        robot.send_action(action)

    robot.disconnect()
    print("Disconnected.")

if __name__ == "__main__":
    main()