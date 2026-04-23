from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import asyncio
import httpx
import secrets
import os
import hashlib
import torch
from typing import Optional
import chess

from bot import get_best_move, train_bot
from model import model_manager, stats_manager

# ---------------------------------------------------------------------------
# Periodic save
# ---------------------------------------------------------------------------

SAVE_INTERVAL_SECONDS = 2 * 60 * 60  # 2 hours

async def _periodic_save():
    """Saves model every SAVE_INTERVAL_SECONDS."""
    while True:
        try:
            await asyncio.sleep(SAVE_INTERVAL_SECONDS)
            print("[Scheduler] Periodic save triggered.")
            await asyncio.gather(
                model_manager.save_model(),
                stats_manager.save_to_blob()
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Scheduler] Save failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start periodic background save task
    task = asyncio.create_task(_periodic_save())
    print(f"[Lifespan] Periodic model save scheduled every {SAVE_INTERVAL_SECONDS // 3600}h.")

    yield  # Server runs here

    # Shutdown
    print("[Lifespan] Server stopping — cancelling tasks.")
    task.cancel()
    # Final save on clean shutdown
    print("[Lifespan] Performing final model & stats save...")
    await asyncio.gather(
        model_manager.save_model(),
        stats_manager.save_to_blob()
    )
    print("[Lifespan] Save complete. Goodbye.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Kakochess Bot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str
    history: List[str] = []
    is_admin: bool = False
    depth: Optional[int] = None
    epsilon: Optional[float] = None

class TrainRequest(BaseModel):
    fens: List[str]
    result: float
    learning_rate: float = 0.0001
    discount_factor: float = 0.95


def _same_position_state(a: chess.Board, b: chess.Board) -> bool:
    """Compares stable position identity across consecutive FENs.

    We intentionally avoid strict en-passant/halfmove equality here because
    chess.js and python-chess can encode these fields differently while still
    representing the same legal next position.
    """
    return (
        a.board_fen() == b.board_fen()
        and a.turn == b.turn
        and a.castling_rights == b.castling_rights
        and a.fullmove_number == b.fullmove_number
    )


def validate_train_request(req: TrainRequest):
    if not req.fens:
        raise HTTPException(status_code=422, detail="Training payload must include at least one FEN.")

    if req.result not in (-1.0, 0.0, 1.0):
        raise HTTPException(status_code=422, detail="Result must be one of: -1.0, 0.0, 1.0.")

    if not (0.0 < req.learning_rate <= 0.1):
        raise HTTPException(status_code=422, detail="learning_rate must be in (0.0, 0.1].")

    if not (0.0 <= req.discount_factor <= 1.0):
        raise HTTPException(status_code=422, detail="discount_factor must be in [0.0, 1.0].")

    for idx, fen in enumerate(req.fens):
        try:
            chess.Board(fen)
        except Exception as fen_error:
            raise HTTPException(status_code=422, detail=f"Invalid FEN at index {idx}: {fen_error}")

    # Enforce that payload represents one legal game progression.
    for idx in range(1, len(req.fens)):
        prev = chess.Board(req.fens[idx - 1])
        nxt = chess.Board(req.fens[idx])

        if prev.turn == nxt.turn:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sequence at index {idx}: side-to-move did not alternate.",
            )

        if prev.is_game_over():
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sequence at index {idx}: previous position is already game-over.",
            )

        progressed = False
        for mv in prev.legal_moves:
            prev.push(mv)
            if _same_position_state(prev, nxt):
                progressed = True
                prev.pop()
                break
            prev.pop()

        if not progressed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid sequence at index {idx}: position is not reachable from prior FEN by one legal move."
                ),
            )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to Kakochess Bot API"}

@app.post("/api/move")
def play_move(req: MoveRequest):
    """Server-side alpha-beta move fallback."""
    try:
        depth = req.depth if req.depth is not None else 2
        best_move_uci = get_best_move(
            req.fen,
            history=req.history,
            depth=depth,
            is_training=req.is_admin,
            epsilon=req.epsilon,
        )
        return {"move": best_move_uci}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/model")
async def get_onnx_model():
    """
    Serve the latest chess_model.onnx.
    Prioritizes local file for speed. If missing, proxies from Vercel Blob
    (Proxying is required because Blob stores are often private, so redirecting fails).
    """
    onnx_filename = model_manager.onnx_filename

    # 1. Fallback: serve from local disk (Quickest)
    if os.path.exists(onnx_filename):
        print(f"[/api/model] Serving local file: {onnx_filename}")
        with open(onnx_filename, "rb") as f:
            data = f.read()
        version = hashlib.sha256(data).hexdigest()[:16]
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"x-model-version": version, "cache-control": "no-store"},
        )

    # 2. Proxy from Vercel Blob
    blob_token = model_manager.blob_token
    if blob_token:
        try:
            headers = {"Authorization": f"Bearer {blob_token}"}
            async with httpx.AsyncClient() as client:
                # Find the URL
                resp = await client.get(
                    "https://blob.vercel-storage.com",
                    headers=headers,
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    blobs = resp.json().get("blobs", [])
                    urls = [b["url"] for b in blobs if b["pathname"] == onnx_filename]
                    if urls:
                        # Proxy the binary data instead of redirecting
                        model_resp = await client.get(urls[0], headers=headers, timeout=30.0)
                        if model_resp.status_code == 200:
                            print(f"[/api/model] Served via Blob proxy: {urls[0]}")
                            payload = model_resp.content
                            version = hashlib.sha256(payload).hexdigest()[:16]
                            return Response(
                                content=payload,
                                media_type="application/octet-stream",
                                headers={"x-model-version": version, "cache-control": "no-store"},
                            )
        except Exception as e:
            print(f"[/api/model] Blob lookup failed: {e}")

    raise HTTPException(status_code=404, detail="ONNX model not found.")


@app.get("/api/model/metadata")
async def get_model_metadata():
    """Returns lightweight model metadata for version polling."""
    onnx_filename = model_manager.onnx_filename
    if os.path.exists(onnx_filename):
        with open(onnx_filename, "rb") as f:
            data = f.read()
        return {
            "available": True,
            "version": hashlib.sha256(data).hexdigest()[:16],
            "bytes": len(data),
            "source": "local",
        }

    blob_token = model_manager.blob_token
    if blob_token:
        try:
            headers = {"Authorization": f"Bearer {blob_token}"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://blob.vercel-storage.com",
                    headers=headers,
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    blobs = resp.json().get("blobs", [])
                    urls = [b["url"] for b in blobs if b["pathname"] == onnx_filename]
                    if urls:
                        model_resp = await client.get(urls[0], headers=headers, timeout=30.0)
                        if model_resp.status_code == 200:
                            payload = model_resp.content
                            return {
                                "available": True,
                                "version": hashlib.sha256(payload).hexdigest()[:16],
                                "bytes": len(payload),
                                "source": "blob",
                            }
        except Exception as e:
            return {"available": False, "error": f"metadata lookup failed: {e}"}

    return {"available": False}


@app.get("/api/contracts")
async def get_runtime_contracts():
    """Runtime smoke-check for model input/output contracts used by frontend."""
    try:
        model = model_manager.model
        model.eval()

        with torch.no_grad():
            dummy = torch.zeros((1, 64, 12), dtype=torch.float32)
            out = model(dummy)

        if not hasattr(out, "shape"):
            raise RuntimeError("Model output has no shape attribute.")

        shape = list(out.shape)
        finite = bool(torch.isfinite(out).all().item())

        # Keep payload compact but actionable for clients and dashboards.
        return {
            "tensor_contract": {
                "input_name": "board",
                "input_shape": [1, 64, 12],
                "output_name": "value",
                "expected_output_shape": [1, 1],
                "actual_output_shape": shape,
                "finite_output": finite,
            },
            "semantics": {
                "value_range": "[-1, 1]",
                "perspective": "side-to-move",
                "training_result_encoding": {
                    "white_win": 1.0,
                    "draw": 0.0,
                    "black_win": -1.0,
                },
            },
            "model": await get_model_metadata(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract check failed: {e}")

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, os.getenv("ADMIN_USERNAME", "admin"))
    correct_password = secrets.compare_digest(credentials.password, os.getenv("ADMIN_PASSWORD", "password"))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/api/stats")
async def get_stats():
    """Returns the training loss history."""
    return stats_manager.history

@app.post("/api/train")
async def train(req: TrainRequest, username: str = Depends(verify_credentials)):
    """Note: Train is a 'def' endpoint, so it runs in a threadpool and won't block the loop."""
    try:
        validate_train_request(req)

        # Run training in threadpool
        avg_loss = await asyncio.to_thread(
            train_bot,
            req.fens, 
            req.result, 
            lr=req.learning_rate, 
            discount_factor=req.discount_factor
        )
        # Record stats async
        await stats_manager.add_stat(avg_loss)
        # Refresh local ONNX immediately so live clients can sync.
        # Blob upload is intentionally deferred to periodic/shutdown save schedule.
        export_ok = await model_manager.export_to_onnx(upload_to_blob=False)
        if not export_ok:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Training completed but local ONNX export failed: "
                    f"{model_manager.last_export_error or 'unknown error'}"
                ),
            )
        return {
            "message": "Training successful (ONNX Blob upload deferred to scheduled save).",
            "loss": avg_loss,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset-engine")
async def reset_engine(username: str = Depends(verify_credentials)):
    """Resets the model to fresh weights and clears all statistics."""
    try:
        await model_manager.reset_model()
        await stats_manager.clear_stats()
        return {"message": "Engine brain reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
