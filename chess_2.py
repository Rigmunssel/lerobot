import cv2
import numpy as np
import time

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_ID = 1          # Change to your overhead webcam ID
BOARD_SIZE = 800       # We will stretch the board to be a perfect 800x800 square
SQUARE_SIZE = 100      # 800 / 8 = 100 pixels per square
THRESHOLD = 30         # Sensitivity for pixel changes (0-255). Lower = more sensitive
STABILITY_SECONDS = 1  # Seconds of no motion required before checking for a move
MOTION_PIXELS = 800     # Frame-to-frame changed pixels that count as "motion" (hand/piece moving).
                         # Raise if spurious triggers, lower if moves go undetected.
COLOR_THRESHOLD = 5     # % change in red/blue presence to count as a piece arrival/departure.

# Chess grid mapping (Assuming you look at the board from White's perspective)
COLS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ROWS = ['8', '7', '6', '5', '4', '3', '2', '1']

# ── Game State ────────────────────────────────────────────────────────────────
# Piece convention: uppercase = White, lowercase = Black. Absent key = empty.
# K/k=King, Q/q=Queen, R/r=Rook, B/b=Bishop, N/n=Knight, P/p=Pawn
state_standard = {
    'a8': 'r', 'b8': 'n', 'c8': 'b', 'd8': 'q', 'e8': 'k', 'f8': 'b', 'g8': 'n', 'h8': 'r',
    'a7': 'p', 'b7': 'p', 'c7': 'p', 'd7': 'p', 'e7': 'p', 'f7': 'p', 'g7': 'p', 'h7': 'p',
    'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P', 'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P',
    'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
}

# Castling moves 4 squares at once; check top-4 changed squares against these patterns
CASTLING_PATTERNS = [
    {'name': 'O-O',   'squares': {'e1', 'g1', 'h1', 'f1'}, 'king_sq': 'e1', 'moves': [('e1', 'g1'), ('h1', 'f1')]},
    {'name': 'O-O-O', 'squares': {'e1', 'c1', 'a1', 'd1'}, 'king_sq': 'e1', 'moves': [('e1', 'c1'), ('a1', 'd1')]},
    {'name': 'O-O',   'squares': {'e8', 'g8', 'h8', 'f8'}, 'king_sq': 'e8', 'moves': [('e8', 'g8'), ('h8', 'f8')]},
    {'name': 'O-O-O', 'squares': {'e8', 'c8', 'a8', 'd8'}, 'king_sq': 'e8', 'moves': [('e8', 'c8'), ('a8', 'd8')]},
]

# ── Color presence detection ─────────────────────────────────────────────────
# HSV ranges for "bright red" and "bright blue".
# Tweak S_MIN / V_MIN if your lighting is dim or the colours look washed out.
RED_HSV_LOWER1 = np.array([  0, 120, 120], dtype=np.uint8)   # red wraps around hue=0
RED_HSV_UPPER1 = np.array([ 10, 255, 255], dtype=np.uint8)
RED_HSV_LOWER2 = np.array([165, 120, 120], dtype=np.uint8)
RED_HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)
BLUE_HSV_LOWER = np.array([ 85,  50,  80], dtype=np.uint8)   # wider hue, lower sat for light blues
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Visualization cell size (pixels) for the color-presence window
COLOR_CELL = 75


def compute_color_presence(warped_bgr):
    """
    For each of the 64 squares compute what fraction of pixels (0-100 %)
    match bright red and bright blue.

    Returns two (8, 8) float arrays: red_pct, blue_pct.
    """
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    mask_red  = cv2.bitwise_or(cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1),
                               cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2))
    mask_blue = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)

    red_pct  = np.zeros((8, 8), dtype=float)
    blue_pct = np.zeros((8, 8), dtype=float)
    total    = SQUARE_SIZE * SQUARE_SIZE

    for row in range(8):
        for col in range(8):
            y0, x0 = row * SQUARE_SIZE, col * SQUARE_SIZE
            y1, x1 = y0 + SQUARE_SIZE, x0 + SQUARE_SIZE
            red_pct[row, col]  = cv2.countNonZero(mask_red [y0:y1, x0:x1]) / total * 100
            blue_pct[row, col] = cv2.countNonZero(mask_blue[y0:y1, x0:x1]) / total * 100

    return red_pct, blue_pct


def draw_color_presence_window(red_pct, blue_pct):
    """
    Build and return a BGR image that shows two 8×8 grids side-by-side:
    left = red presence, right = blue presence.
    Each cell is tinted by its presence fraction and shows the numeric value.
    """
    pad    = 6                              # gap between the two grids
    label  = 22                            # top label bar height
    W      = 8 * COLOR_CELL * 2 + pad
    H      = 8 * COLOR_CELL + label

    img = np.full((H, W, 3), 40, dtype=np.uint8)   # dark background

    for side, pct_grid, base_color in [
        (0, red_pct,  (40,  40, 180)),   # BGR: reddish tint base
        (1, blue_pct, (160, 80,  40)),   # BGR: bluish tint base
    ]:
        x_offset = side * (8 * COLOR_CELL + pad)

        # Title
        title = "RED presence %" if side == 0 else "BLUE presence %"
        cv2.putText(img, title,
                    (x_offset + 4, label - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)

        for row in range(8):
            for col in range(8):
                v   = float(pct_grid[row, col])
                t   = min(v / 30.0, 1.0)      # saturate at 30 % for full colour

                # Blend from dark-grey to the base color
                bg = tuple(int(40 + t * (c - 40)) for c in base_color)

                x0 = x_offset + col * COLOR_CELL
                y0 = label    + row * COLOR_CELL
                x1, y1 = x0 + COLOR_CELL, y0 + COLOR_CELL

                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), bg, -1)
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (80, 80, 80), 1)

                # Square label (top-left corner)
                sq = get_square_name(col, row)
                cv2.putText(img, sq,
                            (x0 + 3, y0 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

                # Numeric value (centred)
                val_str = f"{v:.1f}"
                (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.putText(img, val_str,
                            (x0 + (COLOR_CELL - tw) // 2, y0 + (COLOR_CELL + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)

    return img


def detect_move_from_color(baseline_r, baseline_b, curr_r, curr_b):
    """
    Compare current vs. baseline color presence grids.

    For each colour (red, blue):
      - Squares where presence dropped  > COLOR_THRESHOLD → piece departed  (from_sq)
      - Squares where presence rose     > COLOR_THRESHOLD → piece arrived   (to_sq)

    Returns a list of (color_name, from_sq, to_sq) for every colour that shows
    exactly one departure and one arrival.  Returns [] if detection is ambiguous
    or if no colour has moved.
    """
    moves = []
    for color_name, baseline_pct, curr_pct in [
        ("red",  baseline_r, curr_r),
        ("blue", baseline_b, curr_b),
    ]:
        delta = curr_pct.astype(float) - baseline_pct.astype(float)
        from_cells = [(r, c) for r in range(8) for c in range(8) if delta[r, c] < -COLOR_THRESHOLD]
        to_cells   = [(r, c) for r in range(8) for c in range(8) if delta[r, c] >  COLOR_THRESHOLD]
        if len(from_cells) == 1 and len(to_cells) == 1:
            from_sq = get_square_name(from_cells[0][1], from_cells[0][0])
            to_sq   = get_square_name(to_cells[0][1],   to_cells[0][0])
            moves.append((color_name, from_sq, to_sq))
    return moves


PIECE_NAMES = {
    'K': 'King', 'Q': 'Queen', 'R': 'Rook', 'B': 'Bishop', 'N': 'Knight', 'P': 'Pawn',
    'k': 'King', 'q': 'Queen', 'r': 'Rook', 'b': 'Bishop', 'n': 'Knight', 'p': 'Pawn',
}

def is_own(piece, turn):
    if piece is None: return False
    return piece.isupper() if turn == 'w' else piece.islower()

def infer_move(sq1, sq2, board_state, turn):
    """Returns (from_sq, to_sq) using board state + whose turn it is, or None if ambiguous."""
    p1 = board_state.get(sq1)
    p2 = board_state.get(sq2)
    if is_own(p1, turn) and not is_own(p2, turn):
        return sq1, sq2
    if is_own(p2, turn) and not is_own(p1, turn):
        return sq2, sq1
    return None

def apply_move(from_sq, to_sq, board_state):
    """Moves a piece in board_state, handling pawn promotion (auto-queen)."""
    piece = board_state.pop(from_sq)
    board_state[to_sq] = piece
    if piece == 'P' and to_sq[1] == '8':
        board_state[to_sq] = 'Q'
    elif piece == 'p' and to_sq[1] == '1':
        board_state[to_sq] = 'q'

def print_board(board_state, turn):
    print("\n  a b c d e f g h")
    for row_char in ROWS:
        line = f"{row_char} "
        for col_char in COLS:
            line += f"{board_state.get(col_char + row_char, '.')} "
        print(line)
    print(f"  {'White' if turn == 'w' else 'Black'} to move\n")

# Globals for the mouse clicker
corners = []

def click_event(event, x, y, flags, param):
    """Captures the mouse clicks for calibration."""
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            print(f"📍 Corner {len(corners)} recorded at ({x}, {y})")

def get_square_name(x_index, y_index):
    """Converts a grid coordinate (e.g., 0,0) to chess notation (e.g., a8)."""
    return f"{COLS[x_index]}{ROWS[y_index]}"

def main():
    global corners
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # ── PHASE 1: CALIBRATION ────────────────────────────────────────────────
    print("\n--- PHASE 1: CALIBRATION ---")
    print("Click the 4 corners of the chessboard in this EXACT order:")
    print("1. Top-Left (a8)")
    print("2. Top-Right (h8)")
    print("3. Bottom-Right (h1)")
    print("4. Bottom-Left (a1)")
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", click_event)
    
    while len(corners) < 4:
        ret, frame = cap.read()
        if not ret: continue
        
        # Draw circles where you clicked
        for pt in corners:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)
        
    cv2.destroyWindow("Calibration")
    print("\n✅ Calibration Complete!")

    # Calculate the perspective warp matrix
    pts_src = np.array(corners, dtype="float32")
    pts_dst = np.array([
        [0, 0], 
        [BOARD_SIZE - 1, 0], 
        [BOARD_SIZE - 1, BOARD_SIZE - 1], 
        [0, BOARD_SIZE - 1]
    ], dtype="float32")
    
    warp_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Grab the baseline "before" image
    ret, baseline_frame = cap.read()
    prev_warped = cv2.warpPerspective(baseline_frame, warp_matrix, (BOARD_SIZE, BOARD_SIZE))
    prev_gray = cv2.cvtColor(prev_warped, cv2.COLOR_BGR2GRAY)
    # Blur to ignore tiny lighting noise/camera static
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0) 

    # ── PHASE 2: DIFFERENCE TRACKING ────────────────────────────────────────
    print("\n--- PHASE 2: PLAY CHESS ---")
    print(f"Make your move and remove your hand. Move is registered after {STABILITY_SECONDS:.0f}s of no motion.")
    print("Press 'q' to quit.")

    board_state = dict(state_standard)
    turn = 'w'
    print_board(board_state, turn)

    # baseline_gray / baseline_*_pct: last stable snapshot for move detection
    # rolling_gray: previous frame for per-frame motion detection
    baseline_gray = prev_gray.copy()
    rolling_gray  = prev_gray.copy()
    baseline_red_pct, baseline_blue_pct = compute_color_presence(prev_warped)
    last_motion_time = time.time()
    was_stable = True  # start as stable so the initial idle period doesn't trigger a false move

    while True:
        ret, frame = cap.read()
        if not ret: continue

        live_warped = cv2.warpPerspective(frame, warp_matrix, (BOARD_SIZE, BOARD_SIZE))
        curr_gray = cv2.cvtColor(live_warped, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        # ── Motion detection (consecutive frames) ────────────────────────────
        rolling_diff = cv2.absdiff(curr_gray, rolling_gray)
        _, rolling_thresh = cv2.threshold(rolling_diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(rolling_thresh)

        if motion_pixels > MOTION_PIXELS:
            last_motion_time = time.time()
            was_stable = False

        rolling_gray = curr_gray

        # ── Stability check ───────────────────────────────────────────────────
        time_since_motion = time.time() - last_motion_time
        is_stable = time_since_motion >= STABILITY_SECONDS

        # ── Color presence — computed once, used for both detection and display ─
        curr_red_pct, curr_blue_pct = compute_color_presence(live_warped)

        if is_stable and not was_stable:
            # Board just settled after disturbance — check for a chess move
            was_stable = True
            print("\n🔍 Board stable — analyzing move...")

            # ── Primary: colour-presence diff ────────────────────────────────
            color_moves = detect_move_from_color(
                baseline_red_pct, baseline_blue_pct,
                curr_red_pct,     curr_blue_pct,
            )

            move_detected = False
            if color_moves:
                for color_name, from_sq, to_sq in color_moves:
                    piece    = board_state.get(from_sq)
                    captured = board_state.get(to_sq)
                    if piece:
                        apply_move(from_sq, to_sq, board_state)
                        if piece in ('P', 'p') and from_sq[0] != to_sq[0] and not captured:
                            ep_sq = to_sq[0] + from_sq[1]
                            if board_state.pop(ep_sq, None):
                                print(f"  (En passant — captured piece removed from {ep_sq})")
                        capture_str = f" x {PIECE_NAMES[captured]}" if captured else ""
                        print(f"🎨 {color_name.capitalize()} piece: {from_sq} → {to_sq}"
                              f"{capture_str}  [{PIECE_NAMES.get(piece, piece)}]")
                        turn = 'b' if turn == 'w' else 'w'
                        move_detected = True
                if move_detected:
                    print_board(board_state, turn)

            # ── Fallback: grayscale diff (non-coloured pieces / ambiguous colour) ─
            if not move_detected:
                diff = cv2.absdiff(baseline_gray, curr_gray)
                _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
                cv2.imshow("Difference Math", thresh)

                square_changes = []
                for row in range(8):
                    for col in range(8):
                        y_start, x_start = row * SQUARE_SIZE, col * SQUARE_SIZE
                        square_crop = thresh[y_start:y_start + SQUARE_SIZE, x_start:x_start + SQUARE_SIZE]
                        change_amount = cv2.countNonZero(square_crop)
                        if change_amount > 100:
                            square_changes.append({"name": get_square_name(col, row), "amount": change_amount})

                square_changes.sort(key=lambda x: x["amount"], reverse=True)

                if len(square_changes) >= 2:
                    sq1 = square_changes[0]['name']
                    sq2 = square_changes[1]['name']
                    changed_set = {s['name'] for s in square_changes[:4]}

                    castled = False
                    for pat in CASTLING_PATTERNS:
                        if pat['squares'].issubset(changed_set) and board_state.get(pat['king_sq']) in ('K', 'k'):
                            for from_sq, to_sq in pat['moves']:
                                apply_move(from_sq, to_sq, board_state)
                            turn = 'b' if turn == 'w' else 'w'
                            print(f"Move: {pat['name']}")
                            castled = True
                            break

                    if not castled:
                        move = infer_move(sq1, sq2, board_state, turn)
                        if move:
                            from_sq, to_sq = move
                            piece = board_state[from_sq]
                            captured = board_state.get(to_sq)
                            apply_move(from_sq, to_sq, board_state)
                            if piece in ('P', 'p') and from_sq[0] != to_sq[0] and not captured:
                                ep_sq = to_sq[0] + from_sq[1]
                                if board_state.pop(ep_sq, None):
                                    print(f"  (En passant — captured piece removed from {ep_sq})")
                            capture_str = f" x {PIECE_NAMES[captured]}" if captured else ""
                            print(f"Move: {from_sq} -> {to_sq}{capture_str}  [{PIECE_NAMES[piece]}]")
                            turn = 'b' if turn == 'w' else 'w'
                        else:
                            print(f"Squares changed: {sq1} and {sq2} — could not infer direction (wrong turn?)")

                    print_board(board_state, turn)
                else:
                    print("⚠️ Not enough change detected vs. last stable snapshot.")

            # Update both baselines together
            baseline_gray     = curr_gray.copy()
            baseline_red_pct  = curr_red_pct.copy()
            baseline_blue_pct = curr_blue_pct.copy()

        # ── Color presence window (continuous) ───────────────────────────────
        cv2.imshow("Color Presence", draw_color_presence_window(curr_red_pct, curr_blue_pct))

        # ── Draw grid + stability status overlay ─────────────────────────────
        display = live_warped.copy()
        for row in range(8):
            for col in range(8):
                x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                cv2.rectangle(display, (x, y), (x + SQUARE_SIZE, y + SQUARE_SIZE), (0, 255, 0), 1)
                cv2.putText(display, get_square_name(col, row),
                            (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        if motion_pixels > MOTION_PIXELS:
            status_text = "MOTION"
            status_color = (0, 0, 255)
        elif not is_stable:
            status_text = f"Settling  {time_since_motion:.1f} / {STABILITY_SECONDS:.0f}s"
            status_color = (0, 165, 255)
        else:
            status_text = "STABLE"
            status_color = (0, 255, 0)

        cv2.putText(display, status_text, (5, BOARD_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Live Flattened Board", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()