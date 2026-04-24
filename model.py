import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import httpx
import asyncio
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None
from dotenv import load_dotenv

load_dotenv()

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=6, channels=64):
        super(ChessResNet, self).__init__()
        # Input: (Batch, 13, 8, 8)
        # Planes: 6 White, 6 Black, 1 Side-to-move
        self.start_conv = nn.Conv2d(13, channels, kernel_size=3, padding=1)
        self.start_bn = nn.BatchNorm2d(channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        # Value Head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 32)
        self.value_fc2 = nn.Linear(32, 1)
        
        # Policy Head (Future-proofing for better move selection)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, 4096) # Simplified policy: 64x64 possible move indices

    def forward(self, x):
        # x shape: (Batch, 13, 8, 8)
        x = F.relu(self.start_bn(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 64)
        policy = self.policy_fc(p) # Logits for 64*64 moves
            
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 64)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value

class ModelManager:
    def __init__(self):
        self.model = ChessResNet()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
        self.model_filename = "chess_model_alphazero.pt"
        self.last_checkpoint_load_error = None
        self._loaded_checkpoint = False
        
        self.load_model()
        
        if not self._loaded_checkpoint:
            self.seed_starter_knowledge()

    def load_model(self):
        """
        Loads the model synchronously on startup.
        (Kept synchronous as it only happens once during initialization).
        """
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN found. Using a fresh model locally.")
            return

        try:
            headers = {"Authorization": f"Bearer {self.blob_token}"}
            # List blobs
            response = httpx.get("https://blob.vercel-storage.com", headers=headers, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                urls = [blob["url"] for blob in data.get("blobs", []) if blob["pathname"] == self.model_filename]
                if urls:
                    model_url = urls[0]
                    response = httpx.get(model_url, headers=headers, timeout=60.0)
                    if response.status_code == 200:
                        try:
                            self.model.load_state_dict(torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=False))
                            print("Successfully loaded model from Vercel Blob.")
                            self._loaded_from_blob = True
                            self._loaded_checkpoint = True
                        except Exception as architecture_error:
                            print(f"Architecture mismatch: {architecture_error}. Re-initializing ResNet.")
                            self.reset_model_sync()
                    else:
                        print(f"Failed to download model. Status: {response.status_code}")
                else:
                    print("Model file not found in Blob.")
            else:
                print(f"Failed to list Blob contents. {response.status_code}")
        except Exception as e:
            print(f"Failed to load model from Vercel Blob: {e}")

    async def save_model(self):
        """Asynchronously saves PyTorch model."""
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN. Saving locally only.")
            torch.save(self.model.state_dict(), self.model_filename)
            return

        try:
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            content = buffer.getvalue()

            headers = {
                "Authorization": f"Bearer {self.blob_token}",
                "x-api-version": "7",
                "x-vercel-blob-access": "private",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1"
            }

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"https://blob.vercel-storage.com/{self.model_filename}",
                    headers=headers,
                    content=content,
                    timeout=60.0
                )

            if response.status_code == 200:
                print("Successfully saved PyTorch model to Vercel Blob.")
            else:
                print(f"Failed to save model to Blob. Status: {response.status_code}")
        except Exception as e:
            print(f"Error saving model to Vercel Blob: {e}")

    # Removed ONNX export methods as we now run purely on backend.

    def _load_checkpoint_compat(self, checkpoint: dict, source: str):
        """
        Loads only shape-compatible tensors and fails if compatibility is too low.
        This avoids silently accepting mostly-incompatible checkpoints.
        """
        model_state = self.model.state_dict()

        compatible = {}
        skipped = []

        for key, tensor in checkpoint.items():
            direct_key = key
            alt_key = key[6:] if key.startswith("model.") else None

            target_key = None
            if direct_key in model_state:
                target_key = direct_key
            elif alt_key and alt_key in model_state:
                target_key = alt_key

            if target_key is None:
                skipped.append(key)
                continue

            if model_state[target_key].shape != tensor.shape:
                skipped.append(key)
                continue

            compatible[target_key] = tensor

        total = len(model_state)
        matched = len(compatible)
        ratio = (matched / total) if total else 0.0

        if matched == 0:
            raise RuntimeError(f"No compatible parameters found in checkpoint from {source}.")

        if ratio < 0.7:
            raise RuntimeError(
                f"Checkpoint compatibility too low from {source}: matched {matched}/{total} ({ratio:.1%})."
            )

        load_res = self.model.load_state_dict(compatible, strict=False)
        missing = len(load_res.missing_keys)
        unexpected = len(load_res.unexpected_keys)
        print(
            f"[ModelManager] Loaded checkpoint from {source}: matched={matched}/{total}, "
            f"skipped={len(skipped)}, missing={missing}, unexpected={unexpected}."
        )

    async def reset_model(self):
        """Re-initializes the model and deletes all backups from Blob."""
        print("[ModelManager] Resetting model and deleting backups...")
        self.reset_model_sync()
        
        if self.blob_token:
            try:
                # 1. List Blobs to find full URLs
                headers = {"Authorization": f"Bearer {self.blob_token}"}
                response = httpx.get("https://blob.vercel-storage.com", headers=headers, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    # Filter for our files
                    targets = [self.model_filename, self.onnx_filename]
                    urls_to_delete = [b["url"] for b in data.get("blobs", []) if b["pathname"] in targets]
                    
                    if urls_to_delete:
                        # 2. Delete Blobs
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                "https://blob.vercel-storage.com/delete",
                                headers=headers,
                                json={"urls": urls_to_delete},
                                timeout=30.0
                            )
                        print(f"Successfully deleted {len(urls_to_delete)} model blobs.")
            except Exception as e:
                print(f"Error deleting model blobs: {e}")

        # Local cleanup
        if os.path.exists(self.model_filename): os.remove(self.model_filename)
        
        # Save fresh model locally
        torch.save(self.model.state_dict(), self.model_filename)

    def reset_model_sync(self):
        """Internal synchronous reset."""
        self.model = ChessResNet()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self._loaded_checkpoint = False
        self.seed_starter_knowledge()

    def seed_starter_knowledge(self):
        """
        Seeds the ResNet start conv with basic chess piece knowledge.
        We adjust the first layer weights to respond to piece values.
        """
        print("[ModelManager] Seeding starter knowledge into ResNet...")
        with torch.no_grad():
            # Initial conv layer: 13 input planes to 64 output channels
            # Piece indices: P=0, N=1, B=2, R=3, Q=4, K=5 (White), 6-11 (Black), 12 (Turn)
            vals = [1.0, 3.0, 3.2, 5.0, 9.0, 1.0]
            for i, v in enumerate(vals):
                # Positive for our pieces
                self.model.start_conv.weight[:, i, :, :] += v * 0.01
                # Negative for opponent pieces
                self.model.start_conv.weight[:, i+6, :, :] -= v * 0.01
        print("[ModelManager] Starter pack initialized.")

    def download_starter_weights(self):
        """
        Downloads a small set of starter weights from Hugging Face
        to give the distilled model an immediate IQ boost.
        """
        try:
            print(f"[ModelManager] Checking for starter weights from Hugging Face ({self.hf_repo_id})...")
            # We use local_files_only=False first time, then True
            path = hf_hub_download(repo_id=self.hf_repo_id, filename=self.hf_filename)
            
            if path:
                if path.endswith(".safetensors"):
                    if load_safetensors:
                        checkpoint = load_safetensors(path, device="cpu")
                    else:
                        raise ImportError("safetensors library missing but .safetensors file found.")
                else:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

                self._load_checkpoint_compat(checkpoint, source=f"huggingface:{self.hf_repo_id}/{self.hf_filename}")
                self._loaded_checkpoint = True
                self.last_checkpoint_load_error = None
                print(f"[ModelManager] SUCCESSFULLY LOADED starter weights from Hugging Face: {path}")
            else:
                print("[ModelManager] Hugging Face download returned no path.")
        except Exception as e:
            self.last_checkpoint_load_error = str(e)
            print(f"[ModelManager] Hugging Face download skipped or failed: {e}")
            print("[ModelManager] Falling back to Smart Initialization.")

import json
from datetime import datetime

class StatsManager:
    def __init__(self, blob_token, stats_filename="stats.json"):
        self.blob_token = blob_token
        self.stats_filename = stats_filename
        self.history = []
        self._load_initial_stats()

    def _load_initial_stats(self):
        """Synchronously loads stats on startup."""
        if os.path.exists(self.stats_filename):
            try:
                with open(self.stats_filename, "r") as f:
                    self.history = json.load(f)
                return
            except:
                pass

        if not self.blob_token:
            return

        try:
            headers = {"Authorization": f"Bearer {self.blob_token}"}
            response = httpx.get("https://blob.vercel-storage.com", headers=headers, timeout=10.0)
            if response.status_code == 200:
                blobs = response.json().get("blobs", [])
                urls = [b["url"] for b in blobs if b["pathname"] == self.stats_filename]
                if urls:
                    resp = httpx.get(urls[0], timeout=10.0)
                    if resp.status_code == 200:
                        self.history = resp.json()
                        with open(self.stats_filename, "w") as f:
                            json.dump(self.history, f)
        except Exception as e:
            print(f"Failed to load stats from Blob: {e}")

    async def add_stat(self, loss: float):
        """Adds a new loss point and saves locally."""
        stat = {
            "timestamp": datetime.now().isoformat(),
            "loss": round(loss, 6)
        }
        self.history.append(stat)
        
        # Keep only last 50 points to avoid huge blobs
        if len(self.history) > 50:
            self.history = self.history[-50:]

        try:
            # Save local first (Free and instant)
            def _save_local():
                with open(self.stats_filename, "w") as f:
                    json.dump(self.history, f)
            await asyncio.to_thread(_save_local)
        except Exception as e:
            print(f"Error saving stats locally: {e}")

    async def save_to_blob(self):
        """Batch upload stats to Vercel Blob (Scheduled)."""
        if not self.blob_token or not self.history:
            return

        try:
            headers = {
                "Authorization": f"Bearer {self.blob_token}",
                "x-api-version": "7",
                "x-vercel-blob-access": "private",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"https://blob.vercel-storage.com/{self.stats_filename}",
                    headers=headers,
                    content=json.dumps(self.history),
                    timeout=30.0
                )
            if response.status_code == 200:
                print(f"Successfully saved stats to Vercel Blob.")
            else:
                print(f"Failed to save stats to Blob. Status: {response.status_code}")
        except Exception as e:
            print(f"Error saving stats to Vercel Blob: {e}")

    async def clear_stats(self):
        """Clears all training stats and deletes backups from Blob."""
        print("[StatsManager] Clearing stats and deleting backups...")
        self.history = []
        
        if self.blob_token:
            try:
                headers = {"Authorization": f"Bearer {self.blob_token}"}
                response = httpx.get("https://blob.vercel-storage.com", headers=headers, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    urls_to_delete = [b["url"] for b in data.get("blobs", []) if b["pathname"] == self.stats_filename]
                    
                    if urls_to_delete:
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                "https://blob.vercel-storage.com/delete",
                                headers=headers,
                                json={"urls": urls_to_delete},
                                timeout=30.0
                            )
                        print(f"Successfully deleted stats blob.")
            except Exception as e:
                print(f"Error deleting stats blob: {e}")

        # Local cleanup
        if os.path.exists(self.stats_filename): 
            os.remove(self.stats_filename)

model_manager = ModelManager()
stats_manager = StatsManager(model_manager.blob_token)

