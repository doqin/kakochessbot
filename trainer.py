import os
import requests
import zstandard as zstd
import chess.pgn
import io
import torch
from tqdm import tqdm
from model import model_manager
from bot import fen_to_tensor
import random
import asyncio

# Configuration
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-01.pgn.zst"
TARGET_ELO_MIN = 1800
TARGET_ELO_MAX = 2500
MAX_GAMES = 500000
BATCH_SIZE = 64

def download_file(url, path):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(path)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = f.write(data)
            bar.update(size)

def download_and_train():
    db_filename = os.path.basename(LICHESS_DB_URL)
    local_path = os.path.join(os.path.dirname(__file__), db_filename)
    
    print(f"Starting Lichess data acquisition from {LICHESS_DB_URL}...")
    
    try:
        if not os.path.exists(local_path):
            download_file(LICHESS_DB_URL, local_path)
        else:
            print(f"Using cached database: {local_path}")

        dctx = zstd.ZstdDecompressor()
        
        with open(local_path, 'rb') as f:
            with dctx.stream_reader(f) as stream_reader:
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

                games_processed = 0
                games_found = 0
                training_data = []

                pbar = tqdm(total=MAX_GAMES, desc="Filtering games")

                try:
                    while games_found < MAX_GAMES:
                        try:
                            game = chess.pgn.read_game(text_stream)
                        except Exception as e:
                            print(f"\nError reading game: {e}. Skipping...")
                            continue
                            
                        if game is None:
                            break
                            
                        games_processed += 1
                        
                        white_elo = game.headers.get("WhiteElo")
                        black_elo = game.headers.get("BlackElo")
                        
                        if white_elo and black_elo:
                            try:
                                w_elo = int(white_elo)
                                b_elo = int(black_elo)
                                
                                if TARGET_ELO_MIN <= w_elo <= TARGET_ELO_MAX and TARGET_ELO_MIN <= b_elo <= TARGET_ELO_MAX:
                                    # Found a match!
                                    result = game.headers.get("Result")
                                    if result == "1-0":
                                        res_val = 1.0
                                    elif result == "0-1":
                                        res_val = -1.0
                                    else:
                                        res_val = 0.0
                                        
                                    # Extract positions (sampled)
                                    board = game.board()
                                    moves = list(game.mainline_moves())
                                    if not moves:
                                        continue
                                        
                                    n_moves = len(moves)
                                    sample_indices = set(random.sample(range(n_moves), min(n_moves, 10)))
                                    
                                    for i, move in enumerate(moves):
                                        if i in sample_indices:
                                            training_data.append((board.fen(), res_val, move.uci()))
                                        board.push(move)
                                        
                                    games_found += 1
                                    pbar.update(1)
                                    
                                    # Periodic training
                                    if len(training_data) >= BATCH_SIZE * 10:
                                        run_training_batch(training_data)
                                        training_data = []
                            except ValueError:
                                continue
                finally:
                    pbar.close()
                    print(f"Finished. Processed {games_processed} games to find {games_found} matching Elo range.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Final save
        print("Saving model...")
        asyncio.run(model_manager.save_model())

def run_training_batch(data):
    model_manager.model.train()
    total_loss = 0
    
    # Shuffle
    random.shuffle(data)
    
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    # Simple training loop
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        if not batch: continue
        
        from bot import move_to_index
        model_manager.optimizer.zero_grad()
        
        inputs = torch.cat([fen_to_tensor(f) for f, r, m in batch])
        v_targets = torch.tensor([[r] for f, r, m in batch], dtype=torch.float32)
        p_targets = torch.tensor([move_to_index(chess.Move.from_uci(m)) for f, r, m in batch], dtype=torch.long)
        
        policy_logits, value_preds = model_manager.model(inputs)
        
        v_loss = value_criterion(value_preds, v_targets)
        p_loss = policy_criterion(policy_logits, p_targets)
        
        loss = v_loss + p_loss
        loss.backward()
        model_manager.optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Batch trained. Avg Loss: {total_loss / (len(data)/BATCH_SIZE):.4f}")

if __name__ == "__main__":
    download_and_train()

