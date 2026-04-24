# Kakochessbot (Backend) - V2

The advanced chess bot backend powered by a **Deep Residual Network (ResNet)**.

## Architecture
- **Neural Network**: 6-block Residual Network (ResNet) with Value and Policy heads.
- **Evaluation**: Backend-only evaluation (no ONNX offloading).
- **Training Strategy**: Trained on public Lichess datasets, specifically filtered for games between players in the **1000-1200 Elo range** to simulate realistic intermediate play.
- **Search**: Alpha-Beta pruning with move-ordering influenced by the neural network.

## Setup
1. Make sure you have `uv` installed (`pip install uv`).
2. Initialize dependencies:
   ```bash
   uv sync
   ```
3. Copy `.env.sample` to `.env` and configure your `BLOB_READ_WRITE_TOKEN`.

## Training
To retrain or update the model with fresh Lichess data:
```bash
python trainer.py
```

## Running Locally

To start the FastAPI development server:
```bash
uv run uvicorn main:app --port 8000 --reload
```

## API Endpoints
- `GET /` - Health check.
- `GET /api/model/status` - Returns model architecture and status.
- `POST /api/move` - Calculates the best move for a given FEN.
- `POST /api/train` - Admin-only endpoint for reinforcement learning from local play.

