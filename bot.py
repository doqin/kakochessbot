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
        val = model_manager.model(tensor).item()
    return val

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
    
    # We will compute MSE between each position's evaluation and the final game result.
    # In RL, we might use TD learning, but for simple training, this is an MC update.
    target = torch.tensor([result_val], dtype=torch.float32)
    
    for fen in game_fens:
        tensor = fen_to_tensor(fen)
        prediction = model_manager.model(tensor)
        
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
