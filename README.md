# Kakochessbot (Backend)

The Self-learning chess bot API using PyTorch and FastAPI. 

This service processes board states (FEN), evaluates positions using a neural net, and determines the best moves via Alpha-Beta Search. It persistently uses Vercel Blob to store and retrieve the trained model weights.

## Setup

1. Make sure you have `uv` installed (`pip install uv`).
2. Initialize dependencies:
   ```bash
   uv sync
   ```
3. Copy `.env.sample` to `.env` and configure your `BLOB_READ_WRITE_TOKEN`.

## Running Locally

To start the FastAPI development server:
```bash
uv run uvicorn main:app --port 8000 --reload
```

## API Endpoints

- `GET /` - Health check.
- `POST /api/move` - Calculates and returns the smartest move (JSON body requires `fen`).
- `POST /api/train` - Protected/Admin endpoint. Trains the model from a completed game history (requires `fens` array and `result` float).
