import cv2
import numpy as np
import time

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_ID = 1          # Change to your overhead webcam ID
BOARD_SIZE = 800       # We will stretch the board to be a perfect 800x800 square
SQUARE_SIZE = 100      # 800 / 8 = 100 pixels per square
THRESHOLD = 30         # Sensitivity for pixel changes (0-255). Lower = more sensitive
STABILITY_SECONDS = 1  # Seconds of no motion required before checking for a move
MOTION_PIXELS = 800    # Frame-to-frame changed pixels that count as "motion" (hand/piece moving).
                        # Raise if spurious triggers, lower if moves go undetected.
RED_THRESHOLD  = 3     # % change in red  presence to count as a piece arrival/departure.
BLUE_THRESHOLD = 10     # % change in blue presence to count as a piece arrival/departure.

# ── Game mode ─────────────────────────────────────────────────────────────────
GAME_MODE = 2          # 1 = Human vs Human   (both sides detected from colour)
                       # 2 = Human vs Robot   (human detected; robot move entered as text)
HUMAN_COLOR = 'red'    # 'red' or 'blue' — which physical colour the human plays  (mode 2)
RED_PLAYS_WHITE = False # True  → red pieces = White (uppercase), blue = Black (lowercase)
                        # False → red = Black, blue = White  (default: black/red on top)

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


def detect_move_from_color(baseline_r, baseline_b, curr_r, curr_b, only_color=None):
    """
    Compare current vs. baseline color presence grids.

    For each colour in {red, blue} (filtered by only_color if given):
      - Squares where presence dropped  > threshold → piece departed  (from_sq)
      - Squares where presence rose     > threshold → piece arrived   (to_sq)

    Returns a list of (color_name, from_sq, to_sq) for every colour that shows
    exactly one departure and one arrival.  Returns [] if ambiguous or no move.
    """
    thresholds = {'red': RED_THRESHOLD, 'blue': BLUE_THRESHOLD}
    candidates = [
        ("red",  baseline_r, curr_r),
        ("blue", baseline_b, curr_b),
    ]
    if only_color is not None:
        candidates = [(n, b, c) for n, b, c in candidates if n == only_color]

    moves = []
    for color_name, baseline_pct, curr_pct in candidates:
        thr   = thresholds[color_name]
        delta = curr_pct.astype(float) - baseline_pct.astype(float)
        from_cells = [(r, c) for r in range(8) for c in range(8) if delta[r, c] < -thr]
        to_cells   = [(r, c) for r in range(8) for c in range(8) if delta[r, c] >  thr]
        if len(from_cells) == 1 and len(to_cells) == 1:
            from_sq = get_square_name(from_cells[0][1], from_cells[0][0])
            to_sq   = get_square_name(to_cells[0][1],   to_cells[0][0])
            moves.append((color_name, from_sq, to_sq))
        elif len(from_cells) == 2 and len(to_cells) == 2:
            # Castling moves both king and rook — check the 4 changed squares
            changed = {get_square_name(c, r) for r, c in from_cells + to_cells}
            for pat in CASTLING_PATTERNS:
                if pat['squares'] == changed:
                    # Return king move only; _apply_move_full handles the rook
                    moves.append((color_name, pat['moves'][0][0], pat['moves'][0][1]))
                    break
    return moves


# ── Chess coordinate helpers ──────────────────────────────────────────────────

def sq_to_cr(sq):
    """'e4' → (col=4, row=6)  where col 0='a', row 0='8' (top of board)."""
    return COLS.index(sq[0]), ROWS.index(sq[1])

def cr_to_sq(col, row):
    return COLS[col] + ROWS[row]

# ── Legal-move engine ─────────────────────────────────────────────────────────

def _pseudo_moves(sq, board, ep_sq, castling):
    """
    All squares a piece on sq could reach ignoring whether the move leaves
    the king in check.  Handles castling and en-passant.
    Returns list of to_sq strings.
    """
    piece = board.get(sq)
    if piece is None:
        return []
    col, row = sq_to_cr(sq)
    is_white  = piece.isupper()
    pt        = piece.lower()
    moves     = []

    def try_sq(c, r, capture_ok=True):
        """Add (c,r) if in range and not blocked by own piece; return True if clear."""
        if not (0 <= c < 8 and 0 <= r < 8):
            return False
        tsq = cr_to_sq(c, r)
        occ = board.get(tsq)
        if occ is None:
            moves.append(tsq); return True
        if capture_ok and occ.isupper() != is_white:
            moves.append(tsq)
        return False

    def slide(dc, dr):
        c, r = col + dc, row + dr
        while try_sq(c, r):
            c += dc; r += dr
        # If blocked by enemy, try_sq already added it and returned False — stop.

    if pt == 'p':
        d = -1 if is_white else 1          # direction: white moves toward row 0
        sr = 6  if is_white else 1         # starting rank index
        fwd = cr_to_sq(col, row + d)
        if board.get(fwd) is None:         # one forward
            moves.append(fwd)
            if row == sr:                  # two forward from start
                fwd2 = cr_to_sq(col, row + 2*d)
                if board.get(fwd2) is None:
                    moves.append(fwd2)
        for dc in (-1, 1):                 # diagonal captures
            nc, nr = col + dc, row + d
            if 0 <= nc < 8 and 0 <= nr < 8:
                csq = cr_to_sq(nc, nr)
                occ = board.get(csq)
                if (occ and occ.isupper() != is_white) or csq == ep_sq:
                    moves.append(csq)

    elif pt == 'n':
        for dc, dr in ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)):
            try_sq(col+dc, row+dr)

    elif pt == 'b':
        for dc, dr in ((-1,-1),(-1,1),(1,-1),(1,1)): slide(dc, dr)

    elif pt == 'r':
        for dc, dr in ((-1,0),(1,0),(0,-1),(0,1)): slide(dc, dr)

    elif pt == 'q':
        for dc, dr in ((-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)): slide(dc, dr)

    elif pt == 'k':
        for dc, dr in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            try_sq(col+dc, row+dr)
        # Castling
        rights_k = 'K' if is_white else 'k'
        rights_q = 'Q' if is_white else 'q'
        base_row  = 7 if is_white else 0
        if row == base_row and col == 4:
            # Kingside
            if castling.get(rights_k) and \
               board.get(cr_to_sq(5, base_row)) is None and \
               board.get(cr_to_sq(6, base_row)) is None:
                moves.append(cr_to_sq(6, base_row))
            # Queenside
            if castling.get(rights_q) and \
               board.get(cr_to_sq(3, base_row)) is None and \
               board.get(cr_to_sq(2, base_row)) is None and \
               board.get(cr_to_sq(1, base_row)) is None:
                moves.append(cr_to_sq(2, base_row))

    return moves


def _is_attacked(sq, board, by_white):
    """Return True if sq is attacked by any piece of the given colour."""
    for from_sq in list(board.keys()):
        p = board[from_sq]
        if p.isupper() == by_white:
            if sq in _pseudo_moves(from_sq, board, None, {}):
                return True
    return False


def _find_king(board, is_white):
    k = 'K' if is_white else 'k'
    return next((sq for sq, p in board.items() if p == k), None)


def _in_check(board, is_white):
    ksq = _find_king(board, is_white)
    return ksq is not None and _is_attacked(ksq, board, not is_white)


def is_legal_move(from_sq, to_sq, board, turn, ep_sq, castling):
    """
    Full legal-move check including check-exposure.
    Returns (True, '') or (False, reason_string).
    """
    piece = board.get(from_sq)
    if piece is None:
        return False, f"No piece on {from_sq}"
    is_white = (turn == 'w')
    if piece.isupper() != is_white:
        return False, f"{from_sq} is not a {turn} piece"

    pseudo = _pseudo_moves(from_sq, board, ep_sq, castling)
    if to_sq not in pseudo:
        return False, f"{piece.upper()} cannot move from {from_sq} to {to_sq}"

    # Simulate and check king safety
    test_board = dict(board)
    _apply_move_full(from_sq, to_sq, test_board, ep_sq, castling.copy())
    if _in_check(test_board, is_white):
        return False, "That move leaves your king in check"

    # Castling: king must not pass through or be in check
    col, row = sq_to_cr(from_sq)
    tc,  _   = sq_to_cr(to_sq)
    if piece.lower() == 'k' and abs(tc - col) == 2:
        mid_col = (col + tc) // 2
        mid_sq  = cr_to_sq(mid_col, row)
        if _is_attacked(from_sq, board, not is_white):
            return False, "Cannot castle while in check"
        if _is_attacked(mid_sq, board, not is_white):
            return False, "Cannot castle through check"

    return True, ''


def _apply_move_full(from_sq, to_sq, board, ep_sq, castling):
    """
    Apply a move in-place and return the new en-passant square (or None).
    Also updates castling dict.  Used both for the real game and simulations.
    """
    piece = board.pop(from_sq)
    captured = board.get(to_sq)
    is_white  = piece.isupper()

    # En-passant capture
    col_f, row_f = sq_to_cr(from_sq)
    col_t, row_t = sq_to_cr(to_sq)
    if piece.lower() == 'p' and to_sq == ep_sq:
        ep_cap_row = row_t + (1 if is_white else -1)
        board.pop(cr_to_sq(col_t, ep_cap_row), None)

    board[to_sq] = piece

    # Pawn promotion (auto-queen)
    if piece == 'P' and row_t == 0: board[to_sq] = 'Q'
    if piece == 'p' and row_t == 7: board[to_sq] = 'q'

    # Castling rook move
    if piece.lower() == 'k' and abs(col_t - col_f) == 2:
        base_row = 7 if is_white else 0
        if col_t == 6:  # kingside
            board[cr_to_sq(5, base_row)] = board.pop(cr_to_sq(7, base_row))
        else:           # queenside
            board[cr_to_sq(3, base_row)] = board.pop(cr_to_sq(0, base_row))

    # Update castling rights
    revoke = {
        'e1': ('K', 'Q'), 'e8': ('k', 'q'),
        'h1': ('K',),     'a1': ('Q',),
        'h8': ('k',),     'a8': ('q',),
    }
    for sq in (from_sq, to_sq):
        for r in revoke.get(sq, ()):
            castling[r] = False

    # Compute new en-passant square
    new_ep = None
    if piece.lower() == 'p' and abs(row_t - row_f) == 2:
        new_ep = cr_to_sq(col_f, (row_f + row_t) // 2)
    return new_ep


def color_to_turn(physical_color):
    """Map 'red'/'blue' to 'w'/'b' respecting RED_PLAYS_WHITE."""
    if RED_PLAYS_WHITE:
        return 'w' if physical_color == 'red' else 'b'
    else:
        return 'b' if physical_color == 'red' else 'w'


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

    board_state     = dict(state_standard)
    turn            = 'w'
    ep_square       = None                                  # en-passant target square
    castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}

    mode_str = "Human vs Human" if GAME_MODE == 1 else f"Human vs Robot  (human plays {HUMAN_COLOR})"
    print(f"  Game mode: {mode_str}")
    print_board(board_state, turn)

    # baseline_gray / baseline_*_pct: last stable snapshot for move detection
    # rolling_gray: previous frame for per-frame motion detection
    baseline_gray = prev_gray.copy()
    rolling_gray  = prev_gray.copy()
    baseline_red_pct, baseline_blue_pct = compute_color_presence(prev_warped)
    last_motion_time = time.time()
    was_stable = True

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
            was_stable = True
            print("\nBoard stable — analyzing...")

            human_turn_color = color_to_turn(HUMAN_COLOR)
            is_human_turn    = (GAME_MODE == 1) or (turn == human_turn_color)

            if is_human_turn:
                # Detect color-presence delta for the relevant side
                only_color = HUMAN_COLOR if GAME_MODE == 2 else None
                color_moves = detect_move_from_color(
                    baseline_red_pct, baseline_blue_pct,
                    curr_red_pct,     curr_blue_pct,
                    only_color=only_color,
                )

                # Mode 1: if both colors changed, keep only the one matching current turn
                if GAME_MODE == 1 and len(color_moves) > 1:
                    turn_phys = 'red' if (turn == 'w') == RED_PLAYS_WHITE else 'blue'
                    filtered  = [m for m in color_moves if m[0] == turn_phys]
                    if filtered:
                        color_moves = filtered

                if not color_moves:
                    print("  No color change detected — board may not have changed.")
                else:
                    color_name, from_sq, to_sq = color_moves[0]
                    if len(color_moves) > 1:
                        print(f"  Multiple candidates detected — using {from_sq} -> {to_sq}")

                    legal, reason = is_legal_move(
                        from_sq, to_sq, board_state, turn, ep_square, castling_rights
                    )
                    if not legal:
                        print(f"  ILLEGAL MOVE: {from_sq} -> {to_sq}  ({reason})")
                        print("  Please move the piece back and try again.")
                        # Do NOT update baselines — piece moves back, re-triggers stability
                    else:
                        piece       = board_state.get(from_sq)
                        captured    = board_state.get(to_sq)
                        ep_square   = _apply_move_full(
                            from_sq, to_sq, board_state, ep_square, castling_rights
                        )
                        capture_str = f" x {PIECE_NAMES[captured]}" if captured else ""
                        print(f"  {color_name.capitalize()}: {from_sq} -> {to_sq}"
                              f"{capture_str}  [{PIECE_NAMES.get(piece, piece)}]")
                        turn = 'b' if turn == 'w' else 'w'
                        print_board(board_state, turn)
                        baseline_red_pct  = curr_red_pct.copy()
                        baseline_blue_pct = curr_blue_pct.copy()

            else:
                # Robot's turn (Mode 2 only) — get move via text input
                side = 'White' if turn == 'w' else 'Black'
                print(f"  Robot's turn ({side}). Enter move (e.g. 'e2 e4' or 'O-O'): ",
                      end='', flush=True)
                while True:
                    raw = input().strip()
                    if not raw:
                        continue
                    parts = raw.lower().split()
                    if len(parts) == 2:
                        r_from, r_to = parts[0], parts[1]
                    elif raw.upper() in ('O-O', 'O-O-O'):
                        r_from = 'e1' if turn == 'w' else 'e8'
                        r_to   = ('g1' if turn == 'w' else 'g8') \
                                 if raw.upper() == 'O-O' \
                                 else ('c1' if turn == 'w' else 'c8')
                    else:
                        print("  Invalid format. Use 'e2 e4' or 'O-O'/'O-O-O': ",
                              end='', flush=True)
                        continue

                    legal, reason = is_legal_move(
                        r_from, r_to, board_state, turn, ep_square, castling_rights
                    )
                    if not legal:
                        print(f"  Illegal ({reason}). Try again: ", end='', flush=True)
                        continue

                    piece    = board_state.get(r_from)
                    captured = board_state.get(r_to)
                    ep_square = _apply_move_full(
                        r_from, r_to, board_state, ep_square, castling_rights
                    )
                    capture_str = f" x {PIECE_NAMES[captured]}" if captured else ""
                    print(f"  Robot: {r_from} -> {r_to}{capture_str}"
                          f"  [{PIECE_NAMES.get(piece, piece)}]")
                    turn = 'b' if turn == 'w' else 'w'
                    print_board(board_state, turn)

                    # The baseline for the next human-move comparison stays as-is —
                    # robot piece positions are not tracked in the human color grid.
                    # Only exception: if the robot captured a human piece, zero out
                    # that square so the delta correctly shows nothing left there.
                    if captured:
                        cap_col, cap_row = sq_to_cr(r_to)
                        if HUMAN_COLOR == 'red':
                            baseline_red_pct[cap_row, cap_col] = 0.0
                        else:
                            baseline_blue_pct[cap_row, cap_col] = 0.0

                    # Reset rolling_gray so the stale pre-input() frame doesn't
                    # cause a false motion spike on the first post-input() frame.
                    rolling_gray = curr_gray.copy()
                    break

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