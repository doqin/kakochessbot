import os
import requests
import zstandard as zstd
import chess.pgn
import io
import torch
from tqdm import tqdm
from model import model_manager
from bot import fen_to_tensor
import time

# Configuration
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst"
TARGET_ELO_MIN = 1000
TARGET_ELO_MAX = 1200
MAX_GAMES = 5000
BATCH_SIZE = 32

def download_and_train():
    print(f"Starting Lichess data acquisition from {LICHESS_DB_URL}...")
    
    response = requests.get(LICHESS_DB_URL, stream=True)
    if response.status_code != 200:
        print(f"Failed to download database: {response.status_code}")
        return

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

    games_processed = 0
    games_found = 0
    training_data = []

    pbar = tqdm(total=MAX_GAMES, desc="Filtering games")

    while games_found < MAX_GAMES:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
            
        games_processed += 1
        
        white_elo = game.headers.get("WhiteElo")
        black_elo = game.headers.get("BlackElo")
        
        try:
            if white_elo and black_elo:
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
                    # Take up to 10 positions per game to avoid overfitting to specific games
                    indices = [i for i in range(len(moves))]
                    import random
                    sample_indices = random.sample(indices, min(len(indices), 10))
                    
                    for i, move in enumerate(moves):
                        if i in sample_indices:
                            training_data.append((board.fen(), res_val))
                        board.push(move)
                        
                    games_found += 1
                    pbar.update(1)
                    
                    # Periodic training
                    if len(training_data) >= BATCH_SIZE * 10:
                        run_training_batch(training_data)
                        training_data = []
        except ValueError:
            continue

    pbar.close()
    print(f"Finished. Processed {games_processed} games to find {games_found} matching Elo range.")
    
    # Final save
    import asyncio
    asyncio.run(model_manager.save_model())

def run_training_batch(data):
    model_manager.model.train()
    total_loss = 0
    
    # Shuffle
    import random
    random.shuffle(data)
    
    # Simple training loop
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        if not batch: continue
        
        model_manager.optimizer.zero_grad()
        
        inputs = torch.cat([fen_to_tensor(f) for f, r in batch])
        targets = torch.tensor([[r] for f, r in batch], dtype=torch.float32)
        
        outputs = model_manager.model(inputs)
        loss = model_manager.criterion(outputs, targets)
        loss.backward()
        model_manager.optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Batch trained. Avg Loss: {total_loss / (len(data)/BATCH_SIZE):.4f}")

if __name__ == "__main__":
    download_and_train()
