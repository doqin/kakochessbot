from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import secrets
import os
from fastapi.responses import FileResponse

from bot import train_bot
from model import model_manager
from teacher import play_teacher_game
import asyncio
from contextlib import asynccontextmanager

async def periodic_save():
    while True:
        await asyncio.sleep(2 * 3600) # Every 2 hours
        print("Running scheduled model save...")
        model_manager.save_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    save_task = asyncio.create_task(periodic_save())
    yield
    # Shutdown
    save_task.cancel()
    print("Server stopping. Saving final model state...")
    model_manager.save_model()

app = FastAPI(title="Kakochess Bot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For Vercel hosting, we can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str
    is_admin: bool = False

class TrainRequest(BaseModel):
    fens: List[str]
    pis: List[List[float]]
    result: float # 1.0 for White win, -1.0 for Black win, 0.0 for Draw
    learning_rate: float = 0.0001
    batch_size: int = 128

class TeacherGameRequest(BaseModel):
    depth: int = 4             # Minimax depth (4 = ~1500 ELO, 5 = ~1700 ELO)
    learning_rate: float = 0.0001
    batch_size: int = 128

@app.get("/")
def read_root():
    return {"message": "Welcome to Kakochess Bot API"}

@app.get("/api/model.onnx")
def get_model_onnx():
    # Provide the locally exported ONNX model
    if os.path.exists(model_manager.onnx_filename):
        return FileResponse(model_manager.onnx_filename, media_type="application/octet-stream", filename="chess_model.onnx")
    raise HTTPException(status_code=404, detail="ONNX model not exported yet")

# Notice: /api/move has been REMOVED because the frontend handles inference via WebWorker!

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

@app.post("/api/train")
def train(req: TrainRequest, username: str = Depends(verify_credentials)):
    try:
        avg_loss = train_bot(req.fens, req.pis, req.result, req.learning_rate, req.batch_size)
        return {"message": "Training successful", "loss": avg_loss}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/teacher-game")
def run_teacher_game(req: TeacherGameRequest, username: str = Depends(verify_credentials)):
    """
    Streams a Teacher vs Teacher game move-by-move using Server-Sent Events (SSE).
    Each move is sent immediately as it's computed so the frontend can animate live.
    After the game ends, the neural student is trained and a final summary is sent.
    """
    import json
    from fastapi.responses import StreamingResponse
    from teacher import get_best_move_and_policy
    import chess as chess_lib

    def generate():
        board = chess_lib.Board()
        game_fens = []
        game_pis = []
        moves_uci = []
        move_num = 0

        while not board.is_game_over() and move_num < 200:
            fen = board.fen()
            move_uci, policy = get_best_move_and_policy(board, req.depth)

            if move_uci is None:
                break

            game_fens.append(fen)
            game_pis.append(policy)
            moves_uci.append(move_uci)
            move_num += 1

            # Stream this move immediately to the client
            payload = json.dumps({"type": "move", "move": move_uci, "move_num": move_num, "fen": board.fen()})
            yield f"data: {payload}\n\n"

            board.push(chess_lib.Move.from_uci(move_uci))

        # Determine result
        result = 0.0
        outcome = board.outcome()
        if outcome:
            result = 1.0 if outcome.winner == chess_lib.WHITE else (-1.0 if outcome.winner == chess_lib.BLACK else 0.0)

        # Train the neural student on this game
        avg_loss = 0.0
        if game_fens:
            try:
                avg_loss = train_bot(game_fens, game_pis, result, req.learning_rate, req.batch_size)
            except Exception as e:
                avg_loss = -1.0
                print(f"Training error after teacher game: {e}")

        # Final summary event
        result_str = "White wins" if result == 1.0 else ("Black wins" if result == -1.0 else "Draw")
        payload = json.dumps({
            "type": "done",
            "result": result,
            "result_str": result_str,
            "moves": move_num,
            "loss": avg_loss
        })
        yield f"data: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind proxy
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
