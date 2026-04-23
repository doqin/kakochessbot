import chess
import torch
import torch.nn.functional as F
from model import model_manager
from typing import Optional

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Converts a FEN string to a (1, 13, 8, 8) tensor.
    13 planes: 6 White pieces, 6 Black pieces, 1 side-to-move.
    """
    board = chess.Board(fen)
    tensor = torch.zeros((1, 13, 8, 8), dtype=torch.float32)
    
    piece_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            color_offset = 0 if piece.color == chess.WHITE else 6
            idx_feat = piece_to_index[piece.piece_type] + color_offset
            
            row = sq // 8
            col = sq % 8
            tensor[0, idx_feat, row, col] = 1.0
            
    # Plane 12: Side to move (1.0 for White, 0.0 for Black)
    if board.turn == chess.WHITE:
        tensor[0, 12, :, :] = 1.0
            
    return tensor

def get_material_balance(board: chess.Board) -> float:
    """Returns material balance from MUST perspective (Current Side)."""
    current_turn = board.turn
    
    def count_val(color):
        p = len(board.pieces(chess.PAWN, color))
        n = len(board.pieces(chess.KNIGHT, color))
        b = len(board.pieces(chess.BISHOP, color))
        r = len(board.pieces(chess.ROOK, color))
        q = len(board.pieces(chess.QUEEN, color))
        return p*1 + n*3 + b*3 + r*5 + q*9

    my_score = count_val(current_turn)
    opp_score = count_val(not current_turn)
    
    balance = my_score - opp_score
    return min(max(balance * 0.05, -1.0), 1.0)

def evaluate_board(board: chess.Board, history: list = None) -> float:
    """Evaluates the board from the CURRENT player's perspective."""
    
    def get_pos_key(f):
        parts = f.split(" ")
        return " ".join(parts[:4]) if len(parts) >= 4 else f

    current_pos = get_pos_key(board.fen())
    repetition_count = 0
    if history:
        for f in history:
            if get_pos_key(f) == current_pos:
                repetition_count += 1

    if board.is_game_over() or repetition_count >= 2:
        if board.is_checkmate():
            return -1.0 # Current side to move just got mated
        return 0.0 # Standardize draw evaluation

    # 3. Mid-game evaluation (Perspective-aware)
    tensor = fen_to_tensor(board.fen())
    model_manager.model.eval()
    with torch.no_grad():
        nn_val = model_manager.model(tensor).item()
        
    material_val = get_material_balance(board)
    combined_val = (nn_val * 0.6) + (material_val * 0.4)

    if repetition_count == 1:
        combined_val -= 0.15 # Penalty for repeating

    return combined_val

def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, history: list = None) -> float:
    """Negamax implementation (Implicitly perspective-aware)."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, history)
        
    value = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        new_history = (history + [board.fen()]) if history else [board.fen()]
        # In Negamax, value = -search()
        child_val = -alphabeta(board, depth - 1, -beta, -alpha, new_history)
        board.pop()
        
        value = max(value, child_val)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value

def get_best_move(
    fen: str,
    history: list = None,
    depth: int = 2,
    is_training: bool = False,
    epsilon: Optional[float] = None,
) -> str:
    import random
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""
        
    if fen == chess.STARTING_FEN:
        return random.choice(["e2e4", "d2d4", "c2c4", "g1f3"])
        
    if is_training:
        eps = epsilon if epsilon is not None else (0.25 if board.fullmove_number < 10 else 0.05)
        eps = min(max(eps, 0.0), 1.0)
        if random.random() < eps:
            return random.choice(legal_moves).uci()
            
    best_move = None
    best_value = -float('inf')
    
    for move in legal_moves:
        board.push(move)
        # Pass current history to verify repetitions deep in search
        val = -alphabeta(board, depth - 1, -float('inf'), float('inf'), history)
        board.pop()
        
        val += random.uniform(-0.01, 0.01)
        
        if val > best_value:
            best_value = val
            best_move = move
                
    return best_move.uci() if best_move else ""

def train_bot(game_fens: list, result_val: float, lr: float = 0.0001, discount_factor: float = 0.95):
    """
    Trains on Perspective-Aware targets using BATCH updates.
    result_val: 1.0 (White won), -1.0 (Black won), 0.0 (Draw)
    """
    if not game_fens:
        return 0
        
    model_manager.model.train()
    for param_group in model_manager.optimizer.param_groups:
        param_group['lr'] = lr
        
    model_manager.optimizer.zero_grad()
    total_loss = 0.0
    valid_loss_count = 0
    
    # 1. Detection of game-ending state (e.g. repetition)
    is_repetition = False
    if result_val == 0.0 and len(game_fens) > 1:
        final_board = chess.Board(game_fens[-1])
        if final_board.is_repetition(3):
            is_repetition = True

    # 2. Build training data and calculate rewards
    for i, fen in enumerate(game_fens):
        board = chess.Board(fen)
        
        # Base Reward from final outcome
        if result_val != 0.0:
            side_to_move_winner = (result_val == 1.0 and board.turn == chess.WHITE) or \
                                 (result_val == -1.0 and board.turn == chess.BLACK)
            base_target = 1.0 if side_to_move_winner else -1.0
        else:
            # Draw handling with Repetition Penalty
            material = get_material_balance(board)
            if is_repetition and material > 0.2:
                # Penalize for forcing a draw when winning
                base_target = -0.1
            else:
                base_target = material

        # Reliability-first weighting: keep late/terminal plies strongest.
        remaining_plies = (len(game_fens) - 1) - i
        discounted_target = base_target * (discount_factor ** remaining_plies)
        
        # Tactical Shaping: Simple check for hanging pieces in current state
        # (This gives dense rewards to avoid blunders)
        tactical_penalty = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                # Is my piece attacked and undefended?
                if board.is_attacked_by(not board.turn, sq) and not board.is_attacked_by(board.turn, sq):
                    val_map = {1:0.05, 2:0.1, 3:0.1, 4:0.15, 5:0.25, 6:0.0}
                    tactical_penalty += val_map.get(piece.piece_type, 0)
        
        target = discounted_target - min(tactical_penalty, 0.5)
        target = max(-1.0, min(1.0, target))

        # Forward pass
        tensor = fen_to_tensor(fen)
        prediction = model_manager.model(tensor)
        target_tensor = torch.tensor([[target]], dtype=torch.float32)
        
        loss = model_manager.criterion(prediction, target_tensor)
        if not torch.isfinite(loss):
            print(f"[Training] Skipping non-finite loss at ply {i}.")
            continue

        loss.backward()
        total_loss += loss.item()
        valid_loss_count += 1

    # 3. Batch Update (Normalize by move count)
    # Scale gradients manually before step if needed, but standard optimizer.step() 
    # after multiple backward() calls behaves like batch summation.
    if valid_loss_count == 0:
        model_manager.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Training aborted: no finite loss values were produced.")

    torch.nn.utils.clip_grad_norm_(model_manager.model.parameters(), max_norm=1.0)
    model_manager.optimizer.step()

    avg_loss = total_loss / valid_loss_count
    print(f"Training Batch complete. Game Result: {result_val}, Avg Loss: {avg_loss:.6f}")
    return avg_loss
