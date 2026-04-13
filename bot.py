import chess
import torch
import torch.nn.functional as F
import collections
import random
from model import model_manager

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, key: tuple, target: float):
        """key = (fen, policy, material_target)"""
        self.buffer.append((key, target))
        
    def sample(self, batch_size: int):
        # Filter out any stale entries from old format (before material head)
        valid = [item for item in self.buffer if isinstance(item[0], tuple) and len(item[0]) == 3]
        return random.sample(valid, min(len(valid), batch_size))

    def valid_size(self) -> int:
        return sum(1 for item in self.buffer if isinstance(item[0], tuple) and len(item[0]) == 3)
        
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=10000)

def fen_to_tensor(fen: str) -> torch.Tensor:
    """Converts a FEN string to a (20, 8, 8) tactical tensor for the CNN."""
    board = chess.Board(fen)
    tensor = torch.zeros((20, 8, 8), dtype=torch.float32)
    
    piece_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # 0-5: White pieces, 6-11: Black pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            cf = 0 if piece.color == chess.WHITE else 6
            channel = piece_to_index[piece.piece_type] + cf
            row = sq // 8
            col = sq % 8
            tensor[channel, row, col] = 1.0
            
    # 12-15: Castling rights
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
    
    # 16: En Passant target square
    if board.ep_square:
        row = board.ep_square // 8
        col = board.ep_square % 8
        tensor[16, row, col] = 1.0
        
    # 17: Turn (1 = White, 0 = Black)
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0
        
    # 18: White Attack Map, 19: Black Attack Map
    for sq in chess.SQUARES:
        row, col = sq // 8, sq % 8
        if board.attackers(chess.WHITE, sq):
            tensor[18, row, col] = 1.0
        if board.attackers(chess.BLACK, sq):
            tensor[19, row, col] = 1.0
            
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
    return min(max(balance * 0.05, -1.0), 1.0)

def train_bot(game_fens: list, mcts_policies: list, result_val: float, lr: float = 0.0001, batch_size_requested: int = 128):
    """
    Trains the Tactical ChessCNN.
    """
    if not game_fens or len(game_fens) != len(mcts_policies):
        return 0.0

    model_manager.model.train()
    
    # Material-aware result for draws
    if result_val == 0.0:
        final_board = chess.Board(game_fens[-1])
        material_val = get_material_balance(final_board)
        base_contempt = 0.3
        penalty = (material_val * 0.5) + (base_contempt if final_board.turn == chess.WHITE else -base_contempt)
        result_val = -penalty

    targets = []
    material_targets = []
    discount_factor = 0.99
    game_length = len(game_fens)
    discount_factor = 0.995 if game_length < 20 else 0.99
    blunder_penalty = 0.6 # Aggressive penalty for tactical errors
    
    for i, fen in enumerate(game_fens):
        board = chess.Board(fen)
        perspective_result = result_val if board.turn == chess.WHITE else -result_val
        dist_from_end = len(game_fens) - 1 - i
        target = perspective_result * (discount_factor ** dist_from_end)
        
        # Tactical Check
        if i < len(game_fens) - 1:
            next_board = chess.Board(game_fens[i+1])
            m_before = get_material_balance(board)
            m_after = get_material_balance(next_board)
            delta = (m_after - m_before) if board.turn == chess.WHITE else (m_before - m_after)
            
            if delta < -0.4: # Significant loss detected (pawn or higher)
                target -= (blunder_penalty * abs(delta))
                # Heavy blunt for Queen/Rook hangs
                if delta < -4.0: target = -1.0 
                target = max(-1.0, min(1.0, target))
        
        targets.append(target)
        material_targets.append(get_material_balance(board))
    
    # Store with material target
    for fen, policy, target, m_target in zip(game_fens, mcts_policies, targets, material_targets):
        replay_buffer.push((fen, policy, m_target), target)

    num_epochs = 3 
    valid_count = replay_buffer.valid_size()
    batch_size = min(valid_count, batch_size_requested)
    avg_total_loss = 0
    
    if valid_count < 4:
        print(f"Replay buffer has too few valid entries ({valid_count}). Skipping training epochs.")
        return 0.0
    
    if len(replay_buffer) > batch_size * 2:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model_manager.optimizer, max_lr=lr*2, 
            steps_per_epoch=num_epochs, epochs=1,
            pct_start=0.3
        )
    else:
        scheduler = None
        for param_group in model_manager.optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(num_epochs):
        batch = replay_buffer.sample(batch_size)
        batch_fens = [item[0][0] for item in batch]
        batch_pis = [item[0][1] for item in batch]
        batch_m_targets = [item[0][2] for item in batch] # Added material target
        batch_targets = [item[1] for item in batch]
        
        tensor_inputs = torch.stack([fen_to_tensor(f) for f in batch_fens])
        tensor_pis = torch.tensor(batch_pis, dtype=torch.float32)
        tensor_targets = torch.tensor(batch_targets, dtype=torch.float32).unsqueeze(1)
        tensor_m_targets = torch.tensor(batch_m_targets, dtype=torch.float32).unsqueeze(1)
        
        model_manager.optimizer.zero_grad()
        p_logits, v_preds, m_preds = model_manager.model(tensor_inputs)
        
        # 1. Policy Loss
        log_p = F.log_softmax(p_logits, dim=1)
        policy_loss = -(tensor_pis * log_p).sum(dim=1).mean()
        
        # 2. Value Loss
        value_loss = F.smooth_l1_loss(v_preds, tensor_targets)
        
        # 3. Material Auxiliary Loss (forces tactical awareness)
        material_loss = F.mse_loss(m_preds, tensor_m_targets)
        
        # 4. Entropy Regularization
        entropy = -(F.softmax(p_logits, dim=1) * log_p).sum(dim=1).mean()
        entropy_loss = -0.01 * entropy
        
        total_loss = policy_loss + value_loss + material_loss + entropy_loss
        total_loss.backward()
        model_manager.optimizer.step()
        if scheduler: scheduler.step()
        
        avg_total_loss += total_loss.item()
    
    final_avg_loss = avg_total_loss / num_epochs
    print(f"Training complete. Material Loss: {material_loss.item():.4f}, Final Total Loss: {final_avg_loss:.4f}")
    
    return final_avg_loss
