import chess
import torch
import torch.nn.functional as F
from model import model_manager
from typing import Optional

def move_to_index(move: chess.Move) -> int:
    """Maps a chess move to an index 0-4095."""
    return move.from_square * 64 + move.to_square

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

def evaluate_board(board: chess.Board, history: list = None) -> tuple[torch.Tensor, float]:
    """
    Evaluates the board from the CURRENT player's perspective.
    Returns (policy_logits, scalar_value).
    """
    
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
        # Mock policy for terminal states
        policy_mock = torch.zeros((1, 4096))
        if board.is_checkmate():
            return policy_mock, -1.0
        return policy_mock, 0.0

    tensor = fen_to_tensor(board.fen())
    model_manager.model.eval()
    with torch.no_grad():
        policy_logits, value = model_manager.model(tensor)
        
    material_val = get_material_balance(board)
    combined_val = (value.item() * 0.7) + (material_val * 0.3)

    if repetition_count == 1:
        combined_val -= 0.15 

    return policy_logits, combined_val

def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, history: list = None) -> float:
    """Negamax implementation with Policy-guided move ordering."""
    if depth == 0 or board.is_game_over():
        _, val = evaluate_board(board, history)
        return val
        
    policy_logits, _ = evaluate_board(board, history)
    probs = F.softmax(policy_logits, dim=1).flatten()
    
    # Order moves by policy probability for better pruning
    moves = list(board.legal_moves)
    move_scores = []
    for move in moves:
        score = probs[move_to_index(move)].item()
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    
    value = -float('inf')
    for _, move in move_scores:
        board.push(move)
        new_history = (history + [board.fen()]) if history else [board.fen()]
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
        return "c2c4" # English Opening
        
    # Sicilian Defense: Response to 1. e4
    if board.fullmove_number == 1 and board.turn == chess.BLACK:
        e4_pawn = board.piece_at(chess.E4)
        if e4_pawn and e4_pawn.piece_type == chess.PAWN and e4_pawn.color == chess.WHITE:
            # Check if c5 is legal (it should be)
            c5_move = chess.Move.from_uci("c7c5")
            if c5_move in board.legal_moves:
                return "c7c5"
        
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

def train_bot(game_fens: list, game_moves: list, result_val: float, lr: float = 0.0001, discount_factor: float = 0.95):
    """
    Trains on Dual Heads (Policy + Value).
    """
    if not game_fens or len(game_fens) != len(game_moves):
        return 0
        
    model_manager.model.train()
    for param_group in model_manager.optimizer.param_groups:
        param_group['lr'] = lr
        
    model_manager.optimizer.zero_grad()
    total_loss = 0.0
    valid_loss_count = 0
    
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    for i, fen in enumerate(game_fens):
        board = chess.Board(fen)
        
        # 1. Value Target (Discounted outcome)
        side_to_move_winner = (result_val == 1.0 and board.turn == chess.WHITE) or \
                             (result_val == -1.0 and board.turn == chess.BLACK)
        base_v_target = 1.0 if (result_val != 0 and side_to_move_winner) else (0.0 if result_val == 0 else -1.0)
        remaining_plies = (len(game_fens) - 1) - i
        v_target = base_v_target * (discount_factor ** remaining_plies)
        
        # 2. Policy Target (The move actually played)
        played_move = game_moves[i]
        if isinstance(played_move, str):
            move_obj = chess.Move.from_uci(played_move)
        else:
            move_obj = played_move
        p_target = torch.tensor([move_to_index(move_obj)], dtype=torch.long)

        # Forward pass
        tensor = fen_to_tensor(fen)
        policy_logits, value_pred = model_manager.model(tensor)
        
        v_loss = value_criterion(value_pred, torch.tensor([[v_target]], dtype=torch.float32))
        p_loss = policy_criterion(policy_logits, p_target)
        
        loss = v_loss + p_loss
        
        if not torch.isfinite(loss): continue

        loss.backward()
        total_loss += loss.item()
        valid_loss_count += 1

    if valid_loss_count > 0:
        torch.nn.utils.clip_grad_norm_(model_manager.model.parameters(), max_norm=1.0)
        model_manager.optimizer.step()
        avg_loss = total_loss / valid_loss_count
        print(f"Training Batch complete. Avg Loss: {avg_loss:.6f}")
        return avg_loss
    return 0
