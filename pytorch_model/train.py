import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np

from export_to_onnx import SimpleModel, export_to_onnx

# ------------------------
# Config
# ------------------------
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
ONNX_PATH = "simple.onnx"

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Fake dataset (simple)
# ------------------------
def make_dataset(n=1024):
    X = np.random.randn(n, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

X, y = make_dataset()
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------
# Model
# ------------------------
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------
# MLflow
# ------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("ml-pipeline-triton-deploy")

with mlflow.start_run(run_name="simplemodel-train"):
    mlflow.log_params({
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "device": device,
    })

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", acc, step=epoch)

        print(f"Epoch {epoch} | loss={avg_loss:.4f} acc={acc:.4f}")

    # ------------------------
    # Export ONNX + log artifact
    # ------------------------
    export_to_onnx(model, ONNX_PATH)
    mlflow.log_artifact(ONNX_PATH, artifact_path="model")
