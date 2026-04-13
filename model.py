import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import httpx
from dotenv import load_dotenv

load_dotenv()

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Input channel size increased to 20 for Tactical Vision
        # (12 pieces + 4 castling + 1 EP + 1 Turn + 2 Attack Maps)
        self.conv_input = nn.Conv2d(20, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        # 4 Residual Blocks with Squeeze-and-Excitation
        self.res_blocks = nn.Sequential(
            ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64)
        )
        
        # Policy head (predicts moves: 4096 possible transitions)
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value head (predicts game outcome: -1 to 1)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Auxiliary Material Head (predicts material balance: -1 to 1)
        self.material_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.material_bn = nn.BatchNorm2d(1)
        self.material_fc1 = nn.Linear(1 * 8 * 8, 32)
        self.material_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x is shape: (Batch, 20, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)
        
        # Policy prediction
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value prediction
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Material prediction
        m = F.relu(self.material_bn(self.material_conv(x)))
        m = m.reshape(m.size(0), -1)
        m = F.relu(self.material_fc1(m))
        m = torch.tanh(self.material_fc2(m))
        
        return p, v, m

class ModelManager:
    def __init__(self):
        self.model = ChessCNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
        self.model_filename = "chess_model.pt"
        self.onnx_filename = "chess_model.onnx"
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
                        try:
                            self.model.load_state_dict(state_dict)
                            print("Successfully loaded model from Vercel Blob.")
                        except RuntimeError as re:
                            print(f"Architecture mismatch (likely due to upgrade to 20-plane Tactical CNN): {re}. Starting fresh.")
                    else:
                        print(f"Failed to download model from Blob. Status: {response.status_code}")
                else:
                    print("Model file not found in Blob. Using a fresh model.")
            else:
                print(f"Failed to list Blob contents. {response.status_code}")
        except Exception as e:
            print(f"Failed to load model from Vercel Blob: {e}")

    def save_model(self):
        """Saves the model state to Vercel Blob."""
        if not self.blob_token:
            print("No BLOB_READ_WRITE_TOKEN found. Cannot save model.")
            return

        try:
            # 1. Save standard state dict (.pt)
            buffer_pt = io.BytesIO()
            torch.save(self.model.state_dict(), buffer_pt)
            buffer_pt.seek(0)
            
            # 2. Export ONNX (.onnx)
            self.model.eval()
            buffer_onnx = io.BytesIO()
            dummy_input = torch.randn(1, 20, 8, 8)
            torch.onnx.export(
                self.model, dummy_input, buffer_onnx, 
                input_names=['input'], output_names=['policy', 'value', 'material'],
                dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}, 'material': {0: 'batch_size'}}
            )
            self.model.train()
            buffer_onnx.seek(0)
            
            # Save locally so FastAPI can serve it directly to the frontend
            with open(self.onnx_filename, 'wb') as f:
                f.write(buffer_onnx.read())
            buffer_onnx.seek(0)
            
            headers = {
                "Authorization": f"Bearer {self.blob_token}",
                "x-api-version": "7",
                "x-vercel-blob-access": "private",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1"
            }
            
            # Upload basic .pt state dict
            httpx.put(
                f"https://blob.vercel-storage.com/{self.model_filename}",
                headers=headers,
                content=buffer_pt.read(),
                timeout=60.0
            )
            
            # Upload ONNX model file
            res_onnx = httpx.put(
                f"https://blob.vercel-storage.com/{self.onnx_filename}",
                headers=headers,
                content=buffer_onnx.read(),
                timeout=90.0
            )
            
            if res_onnx.status_code == 200:
                print("Successfully saved Tactical CNN and exported ONNX model to Blob.")
            else:
                print(f"Failed to save ONNX to Blob. Status: {res_onnx.status_code}, Response: {res_onnx.text}")
        except Exception as e:
            print(f"Error saving model to Vercel Blob: {e}")

model_manager = ModelManager()
