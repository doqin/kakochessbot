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

class ChessDistilledTransformer(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, num_layers=2):
        super(ChessDistilledTransformer, self).__init__()
        # 12 piece types as input features for each of the 64 squares
        self.embedding = nn.Linear(12, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (Batch, 64, 12)
        x = self.embedding(x) + self.pos_emb
        x = self.transformer(x)
        
        # Aggregate across sequence (Global Average Pooling)
        x = x.mean(dim=1)
        return self.value_head(x)

class ModelManager:
    def __init__(self):
        self.model = ChessDistilledTransformer()
        # Use AdamW for better fine-tuning stability
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
        self.model_filename = "chess_model.pt"
        self.onnx_filename = "chess_model.onnx"
        self.hf_repo_id = "Maxlegrec/ChessBot"
        self.hf_filename = "model.safetensors"
        self.last_export_error = None
        self.last_checkpoint_load_error = None
        self._loaded_checkpoint = False
        
        self.load_model()
        
        # If no local/blob model exists, download from Hugging Face
        if not os.path.exists(self.model_filename) and not hasattr(self, '_loaded_from_blob'):
            self.download_starter_weights()
            if not self._loaded_checkpoint:
                self.seed_starter_knowledge() # Ensure seeded weights are included in first ONNX snapshot
            # Immediately export to ONNX so the browser gets the fresh HF weights
            try:
                self.export_to_onnx_sync()
            except Exception as export_error:
                print(f"[ModelManager] Initial ONNX export failed: {export_error}")
        
        if not self._loaded_checkpoint:
            self.seed_starter_knowledge() # Layers basic piece values on top of fresh weights only.

        # Keep the inference endpoint available after startup even when only
        # PyTorch weights were loaded from Blob and local ONNX is missing.
        if not os.path.exists(self.onnx_filename):
            try:
                self.export_to_onnx_sync()
            except Exception as export_error:
                print(f"[ModelManager] Startup ONNX export failed: {export_error}")

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
                            self.model.load_state_dict(torch.load(io.BytesIO(response.content), weights_only=False))
                            print("Successfully loaded model from Vercel Blob.")
                            self._loaded_from_blob = True
                            self._loaded_checkpoint = True
                        except Exception as architecture_error:
                            print(f"Architecture mismatch: {architecture_error}. Re-initializing Distilled Transformer.")
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
        """Asynchronously saves PyTorch model and re-exports ONNX."""
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN. Saving locally only.")
            await self.export_to_onnx(upload_to_blob=False)
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

        await self.export_to_onnx(upload_to_blob=True)

    async def export_to_onnx(self, upload_to_blob=True):
        """Asynchronously exports ONNX and optionally uploads it to Blob."""
        try:
            onnx_bytes = await asyncio.to_thread(self.export_to_onnx_sync)
            if not onnx_bytes:
                self.last_export_error = "ONNX export produced empty artifact."
                return False
            self.last_export_error = None
            print(f"ONNX export complete ({len(onnx_bytes)} bytes). Local file updated.")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            self.last_export_error = str(e)
            return False

        if not upload_to_blob:
            return True

        if not self.blob_token:
            return True

        try:
            # Upload to Vercel Blob...
            headers = {
                "Authorization": f"Bearer {self.blob_token}",
                "x-api-version": "7",
                "x-vercel-blob-access": "public",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1",
                "Content-Type": "application/octet-stream"
            }
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"https://blob.vercel-storage.com/{self.onnx_filename}",
                    headers=headers,
                    content=onnx_bytes,
                    timeout=120.0
                )
                if response.status_code != 200:
                    headers["x-vercel-blob-access"] = "private"
                    fallback_response = await client.put(
                        f"https://blob.vercel-storage.com/{self.onnx_filename}",
                        headers=headers,
                        content=onnx_bytes,
                        timeout=120.0
                    )
                    if fallback_response.status_code != 200:
                        self.last_export_error = (
                            f"Blob upload failed. public={response.status_code}, private={fallback_response.status_code}"
                        )
                        return False
        except Exception as e:
            print(f"Error uploading ONNX: {e}")
            self.last_export_error = str(e)
            return False

        return True

    def export_to_onnx_sync(self):
        """Synchronous version of ONNX export."""
        self.model.eval()
        dummy_input = torch.zeros(1, 64, 12, dtype=torch.float32)

        # Export only if model honors the value-head contract used by frontend.
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        if dummy_output.ndim != 2 or dummy_output.shape[1] != 1:
            raise RuntimeError(
                f"Invalid model output shape for ONNX export: {tuple(dummy_output.shape)} (expected [batch, 1])."
            )
        if not torch.isfinite(dummy_output).all():
            raise RuntimeError("Model produced non-finite values on dummy input; aborting ONNX export.")

        buffer = io.BytesIO()
        try:
            torch.onnx.export(
                self.model, dummy_input, buffer,
                export_params=True, opset_version=18, do_constant_folding=True,
                input_names=["board"], output_names=["value"],
                dynamic_axes={"board": {0: "batch_size"}, "value": {0: "batch_size"}}
            )
        except ModuleNotFoundError as e:
            if e.name == "onnxscript":
                raise RuntimeError("Missing dependency 'onnxscript' required for ONNX export.")
            raise
        onnx_bytes = buffer.getvalue()
        with open(self.onnx_filename, "wb") as f:
            f.write(onnx_bytes)
        return onnx_bytes

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
        if os.path.exists(self.onnx_filename): os.remove(self.onnx_filename)
        
        # Save fresh model locally; Blob upload happens in scheduled save cycle.
        await self.export_to_onnx(upload_to_blob=False)

    def reset_model_sync(self):
        """Internal synchronous reset."""
        self.model = ChessDistilledTransformer()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self._loaded_checkpoint = False
        self.download_starter_weights()
        if not self._loaded_checkpoint:
            self.seed_starter_knowledge()

    def seed_starter_knowledge(self):
        """
        Seeds the Transformer embeddings with basic chess piece knowledge
        (P=1, N=3, B=3, R=5, Q=9) to avoid starting from total noise.
        This provides an 'Intermediate Base' instantly.
        """
        print("[ModelManager] Seeding starter knowledge into embeddings...")
        with torch.no_grad():
            # Initialize embedding weights based on piece values
            # Piece indices: P=0, N=1, B=2, R=3, Q=4, K=5
            # Offset for opponent: +6
            vals = [1.0, 3.0, 3.2, 5.0, 9.0, 1.0] # Piece values
            for i, v in enumerate(vals):
                # Our pieces (positive)
                self.model.embedding.weight[:, i] = v * 0.1
                # Opponent pieces (negative)
                self.model.embedding.weight[:, i+6] = -v * 0.1
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

