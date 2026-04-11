import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import httpx
from dotenv import load_dotenv

load_dotenv()

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        # Board representation: 64 squares * 12 piece types
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1) # Value network: evaluation between -1 (Black winning) and 1 (White winning)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.output(x))

class ModelManager:
    def __init__(self):
        self.model = ChessNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
        self.model_filename = "chess_model.pt"
        self.load_model()

    def load_model(self):
        """Loads the model from Vercel Blob or initializes a new one if not found."""
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN found. Using a fresh model locally.")
            return

        try:
            # Query Vercel Blob to get the download url
            headers = {"Authorization": f"Bearer {self.blob_token}"}
            # List blobs to find our model
            response = httpx.get("https://blob.vercel-storage.com", headers=headers, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                urls = [blob["url"] for blob in data.get("blobs", []) if blob["pathname"] == self.model_filename]
                if urls:
                    model_url = urls[0]
                    # Download the model with Authorization headers for private blobs
                    response = httpx.get(model_url, headers=headers, timeout=60.0)
                    if response.status_code == 200:
                        model_data = response.content
                        buffer = io.BytesIO(model_data)
                        state_dict = torch.load(buffer, weights_only=False)
                        self.model.load_state_dict(state_dict)
                        print("Successfully loaded model from Vercel Blob.")
                    else:
                        print(f"Failed to download model from Blob. Status: {response.status_code}, Body: {response.text[:100]}")
                else:
                    print("Model file not found in Blob. Using a fresh model.")
            else:
                print(f"Failed to list Blob contents. {response.status_code}, Body: {response.text[:100]}")
        except Exception as e:
            print(f"Failed to load model from Vercel Blob: {e}")

    def save_model(self):
        """Saves the model state to Vercel Blob."""
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN found. Cannot save model.")
            return

        try:
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)
            
            headers = {
                "Authorization": f"Bearer {self.blob_token}",
                "x-api-version": "7",
                "x-vercel-blob-access": "private",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1"
            }
            
            # Vercel Blob upload endpoint
            response = httpx.put(
                f"https://blob.vercel-storage.com/{self.model_filename}",
                headers=headers,
                content=buffer.read(),
                timeout=60.0
            )
            
            if response.status_code == 200:
                print("Successfully saved model to Vercel Blob.")
            else:
                print(f"Failed to save model. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Error saving model to Vercel Blob: {e}")

model_manager = ModelManager()
