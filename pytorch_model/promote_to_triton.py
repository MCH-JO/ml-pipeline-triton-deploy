import os
import shutil
import requests
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ------------------------
# Env vars
# ------------------------
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-pipeline-triton-deploy")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TRITON_MODEL_DIR = Path(os.getenv("TRITON_MODEL_DIR", "/home/mch-jo/workspace/ml-pipeline-triton-deploy/serving/triton/model_repository/simple"))
MLFLOW_ARTIFACT_ONNX = "model/simple.onnx"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment not found: {EXPERIMENT_NAME}")
    # ------------------------
    # Determine best + most recent run in cas of equality (accuracy desc)
    # ------------------------
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC", "attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise SystemExit("No runs found in experiment.")

    run = runs[0]
    run_id = run.info.run_id
    print(f"Using run_id={run_id}")

    # ------------------------
    # DL ONNX artifact ONNX
    # ------------------------
    local_onnx_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=MLFLOW_ARTIFACT_ONNX)
    local_onnx_path = Path(local_onnx_path)
    print(f"Downloaded: {local_onnx_path}")

    # ------------------------
    # Triton next version number + promotion
    # ------------------------
    if not TRITON_MODEL_DIR.exists():
        raise SystemExit(f"Triton model dir not found: {TRITON_MODEL_DIR.resolve()}")

    existing_versions = [int(p.name) for p in TRITON_MODEL_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    next_version = (max(existing_versions) + 1) if existing_versions else 1

    dest_dir = TRITON_MODEL_DIR / str(next_version)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / "model.onnx"
    shutil.copyfile(local_onnx_path, dest_path)

    # ------------------------
    # Triton Server Helth Check
    # ------------------------
    url = "http://localhost:8000/v2/models/simple/versions/{next_version}/ready"
    r = requests.get(url)

    if r.status_code != 200:
        raise SystemExit("Model not READY in Triton → rollback")

    print(f"Promoted to Triton: {dest_path.resolve()}")
    print(f"Next Triton version: {next_version}")

    # ------------------------
    # Best run added in DEPLOYED_FROM_RUN.txt
    # ------------------------
    (dest_dir / "DEPLOYED_FROM_RUN.txt").write_text(
        f"experiment={EXPERIMENT_NAME}\nrun_id={run_id}\nartifact={MLFLOW_ARTIFACT_ONNX}\n"
    )

if __name__ == "__main__":
    main()
