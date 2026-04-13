"""
teacher.py — Classical Chess Teacher Bot

Uses material evaluation, Piece-Square Tables (PST), and Minimax with Alpha-Beta
pruning to play at an intermediate level (approx 1400-1700 ELO at depth 4).

This engine generates high-quality training data for the Neural Student (ChessCNN)
via Imitation Learning / Supervised Pre-Training.
"""

import chess
import random
import math

# ─── Piece Values (centipawns) ────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# ─── Piece-Square Tables (from White's perspective, a1=0, h8=63) ─────────────
# Source: H. Muller's Simplified Evaluation Function (chessprogramming.org)

PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MIDGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST_KING_ENDGAME = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST_MAP = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
    chess.KING:   PST_KING_MIDGAME, # We switch to endgame dynamically below
}

def _mirror_sq(sq: int) -> int:
    """Mirror a square for Black (so PSTs are from piece's own perspective)."""
    rank = sq // 8
    file = sq % 8
    return (7 - rank) * 8 + file

def _is_endgame(board: chess.Board) -> bool:
    """Simple endgame detector: queens gone or very low material."""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    if queens == 0:
        return True
    minor = (
        len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) +
        len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    )
    return queens <= 2 and minor <= 2

def evaluate(board: chess.Board, position_counts: dict = None) -> float:
    """
    Returns centipawn evaluation from White's perspective.
    Applies a repetition penalty to strongly discourage draws by repetition.
    """
    if board.is_checkmate():
        return -30000 if board.turn == chess.WHITE else 30000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    endgame = _is_endgame(board)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        pval = PIECE_VALUES[piece.piece_type]

        if piece.piece_type == chess.KING:
            pst = PST_KING_ENDGAME if endgame else PST_KING_MIDGAME
        else:
            pst = PST_MAP[piece.piece_type]

        pst_val = pst[sq] if piece.color == chess.WHITE else pst[_mirror_sq(sq)]

        if piece.color == chess.WHITE:
            score += pval + pst_val
        else:
            score -= pval + pst_val

    # Mobility bonus
    mobility_weight = 5
    score += mobility_weight * (len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves)))

    # Repetition penalty: strongly discourage visiting the same position again.
    # This is the key fix for avoiding draw loops in self-play.
    if position_counts:
        key = board.fen().split(' ')[0]  # board hash without clock info
        count = position_counts.get(key, 0)
        if count >= 1:
            penalty = 300 * count  # 300cp per repeat — very significant
            # The side to move should want to avoid this position
            score += -penalty if board.turn == chess.WHITE else penalty

    return score


def _score_move(board: chess.Board, move: chess.Move) -> int:
    """Move ordering heuristic: MVV-LVA (Most Valuable Victim, Least Valuable Attacker)."""
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score = 10 * PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)
    return score


def _order_moves(board: chess.Board):
    """Return legal moves ordered by priority (captures first, then quiet moves)."""
    return sorted(board.legal_moves, key=lambda m: _score_move(board, m), reverse=True)


def _alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool, position_counts: dict = None) -> float:
    """Standard Alpha-Beta pruning minimax with optional repetition awareness."""
    if depth == 0 or board.is_game_over():
        return evaluate(board, position_counts)

    if maximizing:
        max_eval = -math.inf
        for move in _order_moves(board):
            board.push(move)
            val = _alpha_beta(board, depth - 1, alpha, beta, False, position_counts)
            board.pop()
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in _order_moves(board):
            board.push(move)
            val = _alpha_beta(board, depth - 1, alpha, beta, True, position_counts)
            board.pop()
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval


def get_best_move_and_policy(board_or_fen, depth: int = 3, temperature: float = 1.0, position_counts: dict = None):
    """
    Single-pass function: runs alpha-beta ONCE per legal move and returns
    both the best (sampled) move UCI and the 4096-dim softmax policy vector.

    temperature > 1.0: more random (opening variety)
    temperature = 1.0: standard softmax
    temperature -> 0: deterministic best move
    """
    if isinstance(board_or_fen, str):
        board = chess.Board(board_or_fen)
    else:
        board = board_or_fen

    if board.is_game_over():
        return None, [0.0] * 4096

    is_white = board.turn == chess.WHITE
    move_scores = {}
    alpha, beta = -math.inf, math.inf

    for move in _order_moves(board):
        board.push(move)
        score = _alpha_beta(board, depth - 1, alpha, beta, board.turn == chess.WHITE, position_counts)
        board.pop()
        move_scores[move] = score if is_white else -score

        if is_white:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

    if not move_scores:
        return None, [0.0] * 4096

    # Policy softmax (used as training target - always at fixed temperature)
    policy_temp = 50.0
    max_s = max(move_scores.values())
    exp_policy = {m: math.exp((s - max_s) / policy_temp) for m, s in move_scores.items()}
    sum_policy = sum(exp_policy.values())
    policy = [0.0] * 4096
    for move, exp_s in exp_policy.items():
        uci = move.uci()
        from_sq = (ord(uci[0]) - 97) + (int(uci[1]) - 1) * 8
        to_sq   = (ord(uci[2]) - 97) + (int(uci[3]) - 1) * 8
        policy[from_sq * 64 + to_sq] = exp_s / sum_policy

    # Move selection: sample stochastically by temperature for game variety
    # Higher temperature = explore more; lower = play the best move
    sample_temp = max(temperature, 0.01)
    exp_sample = {m: math.exp((s - max_s) / (policy_temp * sample_temp)) for m, s in move_scores.items()}
    sum_sample = sum(exp_sample.values())
    probs = [exp_sample[m] / sum_sample for m in move_scores]
    moves_list = list(move_scores.keys())
    chosen = random.choices(moves_list, weights=probs, k=1)[0]

    return chosen.uci(), policy


def play_teacher_game(depth: int = 3, max_moves: int = 200) -> dict:
    """
    Plays a varied Teacher-vs-Teacher self-play game.

    Anti-draw techniques:
    - Stochastic move sampling (high temperature in opening, low in endgame)
    - Position repetition tracking with eval penalty
    - Draws by repetition/50-move rule are treated as losses for the repeating side
    """
    board = chess.Board()
    game_fens = []
    game_pis = []
    moves_uci = []
    position_counts = {}  # track position visits by board hash

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        # Temperature schedule:
        # Opening (first 8 moves): high temp = lots of variety
        # Midgame: medium temp = mostly good moves
        # Endgame: low temp = precise play
        if move_num < 8:
            temp = 2.5   # lots of randomness in opening
        elif move_num < 20:
            temp = 1.2
        else:
            temp = 0.4   # near-deterministic in endgame

        fen = board.fen()
        # Track position visits (strip clock info for comparison)
        pos_key = fen.split(' ')[0]
        position_counts[pos_key] = position_counts.get(pos_key, 0) + 1

        move_uci, policy = get_best_move_and_policy(board, depth, temperature=temp, position_counts=position_counts)

        if move_uci is None:
            break

        game_fens.append(fen)
        game_pis.append(policy)
        moves_uci.append(move_uci)
        board.push(chess.Move.from_uci(move_uci))

    # Determine result
    result = 0.0
    outcome = board.outcome()
    if outcome:
        result = 1.0 if outcome.winner == chess.WHITE else (-1.0 if outcome.winner == chess.BLACK else 0.0)

    # Penalise draws caused by repetition — these are undesirable for training
    if result == 0.0 and board.is_repetition(2):
        # Whoever had the last move caused the repetition — penalise them
        result = 0.2 if board.turn == chess.WHITE else -0.2  # soft penalty instead of neutral 0

    return {
        "fens": game_fens,
        "pis": game_pis,
        "moves_uci": moves_uci,
        "result": result,
        "moves": len(game_fens)
    }
