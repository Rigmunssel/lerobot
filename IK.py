import time
import numpy as np
import os
from pathlib import Path
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

# ── Configuration ────────────────────────────────────────────────────────────
URDF_PATH = "./SO101/so101_new_calib.urdf"
PORT = "/dev/ttyACM0"
ROBOT_ID = "follower1"
CALIBRATION_DIR = Path(".")

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

MIN_STEP_DEG  = 0.2   # minimum angular step (degrees) — slow joints won't go below this
LEAD_STEP_DEG = 2     # step size for the fastest-moving (lead) joint (degrees)
IK_MAX_ITER   = 8     # max IK refinement iterations per seed
IK_TOL_M      = 5e-4  # IK convergence tolerance in metres (0.5 mm)
IK_WARN_M     = 0.010 # warn (and ask) if best solution is still above this (10 mm)

# Joints the IK is allowed to move.  Set to None to allow all joints.
# Example: ACTIVE_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
ACTIVE_JOINTS = None

# Orientation constraint applied at every IK iteration.
#   None         – fully relax orientation (best raw position accuracy)
#   "downward"   – keep GRIPPER_APPROACH_AXIS pointing straight down (-Z world)
#                  while allowing free roll around that axis.
ORIENTATION_CONSTRAINT = None

# Which local axis of the end-effector frame is the tool/approach direction.
# "z" is the most common URDF convention.  Check your gripper_frame_link in
# the URDF if you are unsure (look at the axis that points "out" of the gripper).
GRIPPER_APPROACH_AXIS = "z"


def _make_downward_orientation(current_rot):
    """
    Build a rotation matrix where the gripper approach axis (GRIPPER_APPROACH_AXIS)
    points straight down in world frame.

    The roll around the approach axis is preserved from current_rot so the
    gripper doesn't spin as the arm moves.

    Maths
    -----
    Let ax / ref / perp = the three column indices (approach, roll-ref, third).
    (ax, ref, perp) = (ax, (ax+1)%3, (ax+2)%3) is always a cyclic permutation
    of (0,1,2), so the right-hand rule col_ax × col_ref = col_perp holds and
    det(R) = +1 by construction.

    Steps:
      1. R[:, ax]   = world_down                    (approach points down)
      2. curr_ref   = project current R[:,ref] onto plane ⊥ world_down
                      (preserves roll; handles degenerate case)
      3. R[:, perp] = cross(world_down, curr_ref)   (orthogonal third axis)
      4. R[:, ref]  = cross(R[:,perp], world_down)  (re-orthogonalised ref)
    """
    world_down = np.array([0.0, 0.0, -1.0])
    ax   = {"x": 0, "y": 1, "z": 2}[GRIPPER_APPROACH_AXIS]
    ref  = (ax + 1) % 3   # roll-reference column
    perp = (ax + 2) % 3   # third column

    # Project roll-reference axis onto the plane ⊥ world_down
    curr_ref = current_rot[:, ref].copy()
    curr_ref -= np.dot(curr_ref, world_down) * world_down
    n = np.linalg.norm(curr_ref)
    if n < 1e-6:
        # Degenerate: ref axis is parallel to world_down — choose any fallback
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, world_down)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        curr_ref = fallback
    else:
        curr_ref /= n

    R = np.zeros((3, 3))
    R[:, ax]   = world_down
    R[:, perp] = np.cross(world_down, curr_ref)   # unit vec, no normalise needed
    R[:, ref]  = np.cross(R[:, perp], world_down)  # re-orthogonalise
    return R


def _active_mask():
    """Return a boolean array: True = joint may move, False = frozen."""
    if ACTIVE_JOINTS is None:
        return np.ones(len(MOTOR_NAMES), dtype=bool)
    mask = np.zeros(len(MOTOR_NAMES), dtype=bool)
    for name in ACTIVE_JOINTS:
        if name not in MOTOR_NAMES:
            raise ValueError(f"Unknown joint in ACTIVE_JOINTS: {name!r}. "
                             f"Valid names: {MOTOR_NAMES}")
        mask[MOTOR_NAMES.index(name)] = True
    return mask


# ── IK solver with iterative refinement ──────────────────────────────────────

def _refine_from_seed(kin, q_seed, desired_pose, frozen_vals):
    """
    Run IK from one seed with iterative re-seeding.
    Orientation is relaxed each iteration so the solver only minimises
    position error, not a combined position+orientation objective.

    frozen_vals: array same length as q where non-NaN entries are locked to
                 that value after every IK call (the joint cannot move).

    Returns (q, err_metres).
    """
    target_xyz  = desired_pose[:3, 3]
    frozen_mask = ~np.isnan(frozen_vals)   # True where joint is locked
    q   = q_seed.copy()
    err = float("inf")
    for _ in range(IK_MAX_ITER):
        fk_current = kin.forward_kinematics(q)
        pose_iter  = desired_pose.copy()

        if ORIENTATION_CONSTRAINT is None:
            # Fully relax orientation — best raw position accuracy
            pose_iter[:3, :3] = fk_current[:3, :3]
        elif ORIENTATION_CONSTRAINT == "downward":
            # Compute the orientation that points the approach axis down,
            # derived from the current FK so it updates each iteration.
            pose_iter[:3, :3] = _make_downward_orientation(fk_current[:3, :3])
        else:
            raise ValueError(f"Unknown ORIENTATION_CONSTRAINT: {ORIENTATION_CONSTRAINT!r}")

        q = kin.inverse_kinematics(current_joint_pos=q, desired_ee_pose=pose_iter)

        # Snap frozen joints back — forces solver to compensate with active ones
        q[frozen_mask] = frozen_vals[frozen_mask]

        fk  = kin.forward_kinematics(q)
        err = float(np.linalg.norm(fk[:3, 3] - target_xyz))
        if err < IK_TOL_M:
            break
    return q, err


def solve_ik_best(kin, q_curr, desired_pose):
    """
    Try several diverse seeds and return the solution with the lowest residual.

    Respects ACTIVE_JOINTS: frozen joints are locked to their current values
    across all seeds and all refinement iterations.

    Seeds:
      1. Current joint angles  (good when target is close)
      2. Current angles but shoulder_pan rotated to atan2(Y,X) — rescues the IK
         when the arm starts on the wrong side of the target.
      3/4. Neutral elbow-up / elbow-down configs seeded with the geometric pan.

    Prints the residual for every seed so you can see which one won.
    Returns (best_q, best_err, seed_label).
    """
    target_xyz  = desired_pose[:3, 3]
    geo_pan_deg = float(np.degrees(np.arctan2(target_xyz[1], target_xyz[0])))

    # Build frozen_vals: NaN = free, value = locked to current angle
    active = _active_mask()
    frozen_vals = np.where(active, np.nan, q_curr)   # lock inactive joints

    if not active.all():
        frozen_names  = [n for n, a in zip(MOTOR_NAMES, active) if not a]
        active_names  = [n for n, a in zip(MOTOR_NAMES, active) if a]
        print(f"   Joint filter : active={active_names}  frozen={frozen_names}")

    # Seeds — frozen joints will be overwritten by _refine_from_seed anyway,
    # so we only need to vary the active-joint values between seeds.
    q_geo_pan      = q_curr.copy();  q_geo_pan[0]  = geo_pan_deg
    q_neutral_up   = np.array([geo_pan_deg, -60.0,  90.0, -90.0,  0.0])
    q_neutral_down = np.array([geo_pan_deg, -120.0, 45.0, -90.0,  0.0])
    # Initialise all seeds with current frozen values so they start valid
    for qs in (q_geo_pan, q_neutral_up, q_neutral_down):
        qs[~active] = q_curr[~active]

    seeds = [
        ("current      ", q_curr),
        ("geo-pan      ", q_geo_pan),
        ("neutral-up   ", q_neutral_up),
        ("neutral-down ", q_neutral_down),
    ]

    best_q, best_err, best_label = None, float("inf"), ""
    for label, seed in seeds:
        q, err = _refine_from_seed(kin, seed, desired_pose, frozen_vals)
        print(f"     seed {label}: residual={err*1000:7.2f} mm")
        if err < best_err:
            best_q, best_err, best_label = q, err, label

    return best_q, best_err, best_label


# ── Frequency-based joint-space trajectory planner ───────────────────────────

def plan_joint_steps(q_start, q_target, min_step=MIN_STEP_DEG, lead_step=LEAD_STEP_DEG):
    """
    Plan a smooth joint-space trajectory from q_start to q_target.

    Algorithm
    ---------
    1. The joint with the largest movement is the 'lead' joint.
       It moves in steps of `lead_step` degrees, giving n_steps total.
    2. Every other joint is scaled proportionally.
       - If the scaled step >= min_step  →  move every step, step = d/n_steps.
       - If the scaled step <  min_step  →  move at min_step but every `freq`
         lead-steps, where freq = round(n_steps / (d / min_step)).
         This keeps the total movement correct.
    3. The final step always flushes all remaining movement so we land exactly
       on q_target (absorbs any floating-point rounding).
    4. Edge case: if a joint needs *more* moves than n_steps (its total
       movement / min_step > n_steps), fall back to proportional scaling for
       that joint so we never overshoot.

    Returns list of (q_waypoint: ndarray, step_delta: ndarray) tuples.
    """
    delta     = q_target - q_start
    abs_delta = np.abs(delta)
    max_delta = float(np.max(abs_delta))

    if max_delta < 1e-6:
        return []

    n_steps = max(1, int(np.ceil(max_delta / lead_step)))

    plans = []
    for i in range(len(delta)):
        d    = float(abs_delta[i])
        sign = float(np.sign(delta[i])) if d > 1e-9 else 0.0

        if d < 1e-9:
            plans.append({"sign": 0.0, "step_size": 0.0, "freq": 1, "remaining": 0.0})
            continue

        ideal_step = d / n_steps  # proportionally scaled step size

        if ideal_step < min_step:
            n_moves = max(1, round(d / min_step))  # how many min-step moves this joint needs
            if n_moves >= n_steps:
                # Can't use reduced-frequency trick — fall back to proportional
                freq      = 1
                step_size = ideal_step
            else:
                freq      = max(1, round(n_steps / n_moves))  # lead steps between each move
                step_size = min_step
        else:
            freq      = 1
            step_size = ideal_step

        plans.append({"sign": sign, "step_size": step_size, "freq": freq, "remaining": d})

    waypoints = []
    q_curr    = q_start.copy()

    for step_i in range(1, n_steps + 1):
        q_next     = q_curr.copy()
        step_delta = np.zeros(len(plans))
        is_last    = (step_i == n_steps)

        for j, plan in enumerate(plans):
            if plan["remaining"] <= 1e-9:
                continue

            should_move = is_last or (step_i % plan["freq"] == 0)

            if should_move:
                # Last step: flush all remaining to land exactly on q_target
                move = plan["remaining"] if is_last else min(plan["step_size"], plan["remaining"])
                q_next[j]         += plan["sign"] * move
                plan["remaining"] -= move
                step_delta[j]      = plan["sign"] * move

        waypoints.append((q_next.copy(), step_delta.copy()))
        q_curr = q_next

    return waypoints


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(URDF_PATH):
        print(f"❌ URDF not found at {URDF_PATH}")
        return

    print("🧠 Loading URDF kinematics...")
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=MOTOR_NAMES,
    )

    print("🤖 Connecting to arm...")
    config = SOFollowerRobotConfig(
        port=PORT, id=ROBOT_ID, calibration_dir=CALIBRATION_DIR, use_degrees=True
    )
    robot = SOFollower(config)
    robot.connect()

    try:
        while True:
            # ── Read current state ──────────────────────────────────────────
            obs    = robot.get_observation()
            q_curr = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)

            current_pose = kin.forward_kinematics(q_curr)
            curr_xyz     = current_pose[:3, 3]

            print(f"\n📍 CURRENT POSITION: X={curr_xyz[0]:.4f}, Y={curr_xyz[1]:.4f}, Z={curr_xyz[2]:.4f}")
            print("   Current joints:")
            for name, angle in zip(MOTOR_NAMES, q_curr):
                print(f"     {name:15s}: {angle:8.3f}°")

            inp = input("\nEnter target 'X Y Z' in meters (e.g., '0.2 0.0 0.1') or 'q': ")
            if inp.lower() == "q":
                break

            try:
                target_xyz = np.array([float(v) for v in inp.split()])
                if len(target_xyz) != 3:
                    raise ValueError
            except Exception:
                print("Invalid input. Format: 0.15 0.0 0.1")
                continue

            # ── Solve IK with iterative refinement ─────────────────────────
            desired_pose              = np.eye(4)
            desired_pose[:3, :3]     = current_pose[:3, :3]   # keep current orientation
            desired_pose[:3, 3]      = target_xyz

            print(f"\n🔍 Solving IK for target "
                  f"X={target_xyz[0]:.4f}, Y={target_xyz[1]:.4f}, Z={target_xyz[2]:.4f} ...")
            q_target, ik_err, winning_seed = solve_ik_best(kin, q_curr, desired_pose)

            fk_check = kin.forward_kinematics(q_target)
            fk_xyz   = fk_check[:3, 3]
            print(f"   Best seed    : {winning_seed.strip()}")
            print(f"   IK FK check  : X={fk_xyz[0]:.4f}, Y={fk_xyz[1]:.4f}, Z={fk_xyz[2]:.4f}"
                  f"  (residual={ik_err*1000:.2f} mm)")

            if ik_err > IK_WARN_M:
                print(f"\n⚠️  IK residual {ik_err*1000:.1f} mm exceeds {IK_WARN_M*1000:.0f} mm — "
                      f"target may be unreachable or near a singularity.")
                ans = input("   Proceed anyway? [y/N]: ").strip().lower()
                if ans != "y":
                    print("   Skipped.")
                    continue

            delta_q = q_target - q_curr
            print(f"\n   {'Joint':15s}  {'Current':>10s}  {'Target':>10s}  {'Delta':>10s}")
            for name, cur, tgt, d in zip(MOTOR_NAMES, q_curr, q_target, delta_q):
                print(f"   {name:15s}  {cur:10.3f}°  {tgt:10.3f}°  {d:+10.3f}°")

            # ── Plan smooth joint-space trajectory ─────────────────────────
            waypoints   = plan_joint_steps(q_curr, q_target)
            total_steps = len(waypoints)

            if total_steps == 0:
                print("   Already at target — no movement needed.")
                continue

            print(f"\n   Trajectory: {total_steps} steps "
                  f"(lead_step={LEAD_STEP_DEG}°, min_step={MIN_STEP_DEG}°)")

            # ── Print full step plan ────────────────────────────────────────
            short = [n.replace("shoulder_", "sh_").replace("elbow_", "el_")
                      .replace("wrist_", "wr_") for n in MOTOR_NAMES]
            col = 10
            header = f"   {'Step':>5s}  " + "  ".join(f"{s:>{col}s}" for s in short)
            print(header)
            print("   " + "-" * (len(header) - 3))
            for si, (_, sd) in enumerate(waypoints, 1):
                row = f"   {si:>5d}  " + "  ".join(
                    f"{sd[j]:>+{col}.2f}°" if abs(sd[j]) > 1e-9 else f"{'—':>{col}s}"
                    for j in range(len(MOTOR_NAMES))
                )
                print(row)
            print()

            for step_i, (q_next, step_delta) in enumerate(waypoints, 1):
                input(f"\n >>> [Step {step_i}/{total_steps}] Press ENTER to move...")

                # Read actual robot state for display (may differ from planned)
                obs_now  = robot.get_observation()
                q_actual = np.array([obs_now[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)
                fk_now   = kin.forward_kinematics(q_actual)
                actual_xyz = fk_now[:3, 3]

                print(f"\n   XYZ  actual : X={actual_xyz[0]:.4f}, Y={actual_xyz[1]:.4f}, Z={actual_xyz[2]:.4f}")
                print(f"   XYZ  target : X={target_xyz[0]:.4f}, Y={target_xyz[1]:.4f}, Z={target_xyz[2]:.4f}")
                print(f"   XYZ  error  : {np.linalg.norm(actual_xyz - target_xyz)*1000:.2f} mm")

                print(f"\n   {'Joint':15s}  {'Actual':>10s}  {'Target':>10s}  {'Remaining':>10s}  {'Next step':>10s}")
                for j, name in enumerate(MOTOR_NAMES):
                    remaining = q_target[j] - q_actual[j]
                    print(f"   {name:15s}  {q_actual[j]:10.3f}°  {q_target[j]:10.3f}°"
                          f"  {remaining:+10.3f}°  {step_delta[j]:+10.3f}°")

                # Send action
                action = {f"{name}.pos": float(q_next[j]) for j, name in enumerate(MOTOR_NAMES)}
                action["gripper.pos"] = obs_now["gripper.pos"]
                robot.send_action(action)
                time.sleep(0.05)  # brief settle before next ENTER

            # ── Final readback ─────────────────────────────────────────────
            time.sleep(0.3)
            obs_final = robot.get_observation()
            q_final   = np.array([obs_final[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)
            fk_final  = kin.forward_kinematics(q_final)
            final_xyz = fk_final[:3, 3]

            print(f"\n✅ Sequence finished.")
            print(f"   Target  XYZ: X={target_xyz[0]:.4f}, Y={target_xyz[1]:.4f}, Z={target_xyz[2]:.4f}")
            print(f"   Reached XYZ: X={final_xyz[0]:.4f}, Y={final_xyz[1]:.4f}, Z={final_xyz[2]:.4f}")
            print(f"   Final error: {np.linalg.norm(final_xyz - target_xyz)*1000:.2f} mm")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("🔌 Disconnected.")


if __name__ == "__main__":
    main()
