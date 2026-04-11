import chess
import torch
import torch.nn.functional as F
from model import model_manager

def fen_to_tensor(fen: str) -> torch.Tensor:
    """Converts a FEN string to a 768-length tensor."""
    board = chess.Board(fen)
    tensor = torch.zeros(768, dtype=torch.float32)
    
    piece_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            # Color offset: White = 0, Black = 6
            cf = 0 if piece.color == chess.WHITE else 6
            idx = (piece_to_index[piece.piece_type] + cf) * 64 + sq
            tensor[idx] = 1.0
            
    return tensor

def get_material_balance(board: chess.Board) -> float:
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    balance = (wp - bp)*1 + (wn - bn)*3 + (wb - bb)*3 + (wr - br)*5 + (wq - bq)*9
    # Map a 9-point queen advantage to a massive 0.45 swing towards `1.0` or `-1.0`
    return min(max(balance * 0.05, -1.0), 1.0)

def evaluate_board(board: chess.Board) -> float:
    """Evaluates the board using the Neural Network."""
    if board.is_game_over():
        result = board.result()
        if result == '1-0': return 1.0
        if result == '0-1': return -1.0
        return 0.0 # Draw
        
    tensor = fen_to_tensor(board.fen())
    model_manager.model.eval()
    with torch.no_grad():
        nn_val = model_manager.model(tensor).item()
        
    material_val = get_material_balance(board)
    
    # 60% weight to pattern recognition, 40% explicitly to material survival
    combined_val = (nn_val * 0.6) + (material_val * 0.4)
    return combined_val

def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
        
    if maximizing_player:
        value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alphabeta(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alphabeta(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def get_best_move(fen: str, depth: int = 2, is_training: bool = False) -> str:
    """Returns the best move as a SAN string, e.g., 'e4' or standard UCI like 'e2e4'."""
    import random
    board = chess.Board(fen)
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""
        
    # 1. Opening Book for variety in new games
    if fen == chess.STARTING_FEN:
        return random.choice(["e2e4", "d2d4", "c2c4", "g1f3"])
        
    # 2. Epsilon-Greedy: Large random exploration when explicitly training
    if is_training:
        epsilon = 0.25 if board.fullmove_number < 10 else 0.05
        if random.random() < epsilon:
            return random.choice(legal_moves).uci()
            
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
    
    for move in legal_moves:
        board.push(move)
        board_val = alphabeta(board, depth - 1, -float('inf'), float('inf'), board.turn == chess.WHITE)
        board.pop()
        
        # 3. Add very tiny noise to board_val to prevent exactly identical games across sessions
        board_val += random.uniform(-0.01, 0.01)
        
        if board.turn == chess.WHITE:
            if board_val > best_value:
                best_value = board_val
                best_move = move
        else:
            if board_val < best_value:
                best_value = board_val
                best_move = move
                
    # Fallback to first move if evaluation failed to differentiate
    if best_move is None and list(board.legal_moves):
        best_move = list(board.legal_moves)[0]
        
    return best_move.uci() if best_move else ""

def train_bot(game_fens: list, result_val: float):
    """
    Trains the NN on a sequence of FENs from a completed game.
    result_val: 1.0 (White won), -1.0 (Black won), 0.0 (Draw)
    """
    model_manager.model.train()
    total_loss = 0.0
    
    # TD Discount Logic: Cascades the reward backwards.
    # The final move gets 100% of the result. Moves further back in time decay by 5%.
    # This specifically forces it to learn the *actual* late-game moves that caused the win/loss.
    targets = []
    current_target = result_val
    discount_factor = 0.95
    
    for _ in reversed(game_fens):
        targets.append(current_target)
        current_target *= discount_factor
    targets.reverse()
    
    for i, fen in enumerate(game_fens):
        tensor = fen_to_tensor(fen)
        prediction = model_manager.model(tensor)
        
        target = torch.tensor([targets[i]], dtype=torch.float32)
        
        loss = model_manager.criterion(prediction, target)
        model_manager.optimizer.zero_grad()
        loss.backward()
        model_manager.optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(game_fens) if game_fens else 0
    print(f"Training complete. Average Loss: {avg_loss}")
    
    # Save the updated model
    model_manager.save_model()
    return avg_loss
