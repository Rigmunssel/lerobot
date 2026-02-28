from stockfish import Stockfish

# 1. Initialize Stockfish with the path to your downloaded executable
# Replace this path with the actual path on your machine!
stockfish_path = "/home/rigmunssel/stockfish/stockfish-ubuntu-x86-64-avx2"

try:
    # You can pass parameters upon initialization
    stockfish = Stockfish(path=stockfish_path, depth=15, parameters={"Threads": 2, "Minimum Thinking Time": 30})
except FileNotFoundError:
    print(f"Error: Could not find Stockfish executable at {stockfish_path}")
    exit()

# 2. Set the Difficulty (Skill Level)
# Stockfish Skill Level ranges from 0 (easiest) to 20 (hardest).
difficulty_level = 10
stockfish.set_skill_level(difficulty_level)
print(f"Skill level set to: {difficulty_level}")

# 3. Set the Game State
# The standard way to represent a board state is using a FEN string.
# Here is the starting position, but you can paste any valid FEN here.
fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
stockfish.set_fen_position(fen_string)

# Alternatively, you can set the state by passing a list of moves from the start:
# stockfish.set_position(["e2e4", "e7e5", "g1f3"])

# 4. Get the Best Move
# This returns the best move in UCI format (e.g., 'e2e4')
best_move = stockfish.get_best_move()
print(f"Best move for the given position: {best_move}")

# (Optional) You can also ask for the top N moves with evaluations
top_moves = stockfish.get_top_moves(3)
print("\nTop 3 moves and their evaluations:")
for i, move in enumerate(top_moves):
    # Centipawn (cp) evaluation: positive favors White, negative favors Black
    print(f"{i+1}. Move: {move['Move']}, Centipawns: {move['Centipawn']}, Mate in: {move['Mate']}")