from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import secrets
import os

from bot import get_best_move, train_bot

app = FastAPI(title="Kakochess Bot API")

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
    result: float # 1.0 for White win, -1.0 for Black win, 0.0 for Draw

@app.get("/")
def read_root():
    return {"message": "Welcome to Kakochess Bot API"}

@app.post("/api/move")
def play_move(req: MoveRequest):
    try:
        best_move_uci = get_best_move(req.fen, depth=2, is_training=req.is_admin) # Keep depth low for speed
        return {"move": best_move_uci}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        avg_loss = train_bot(req.fens, req.result)
        return {"message": "Training successful", "loss": avg_loss}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
