# Runbook – ML Platform Operations

This document describes **operational procedures**
for running and maintaining the ML serving platform.

---

## Prerequisites

- Linux host
- NVIDIA GPU with compatible drivers
- Docker and NVIDIA Container Toolkit installed

Check GPU availability:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.4.3-base-ubuntu20.04 nvidia-smi
```

## Platform Startup

Start Triton server:

```bash
docker compose -f serving/triton/compose.yml up -d
```

Verify health:

```bash
curl localhost:8000/v2/health/ready
```

Expected result:

```text
OK
```

Platform Shutdown

```bash
docker compose -f serving/triton/compose.yml down
```

Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Model Promotion Workflow

1. Train a model using PyTorch.
2. Export the model to ONNX format.
3. Track the run and artifacts in MLflow.
4. Promote the selected model using the promotion script:

  - a new version directory is created in the Triton model repository
  - the ONNX model is copied
  - metadata is stored for traceability

After promotion:

- Triton automatically detects the new model version
- The model becomes available for inference

## Model Rollback

To rollback to a previous model version:

1. Identify the target version directory.
2. Remove or disable the latest version.
3. Restart Triton if required.

This allows fast recovery in case of performance or functional regressions.

## Monitoring & Health Checks

- Triton readiness endpoint:

```bash
curl localhost:8000/v2/health/ready
```

- Model list:

```bash
curl localhost:8000/v2/models
```

Triton'shealth is checked in promotion script

- GPU usage:

```bash
nvidia-smi
```

## Backup & Data Considerations

- MLflow artifacts should be considered persistent data.
- The Triton model repository should be backed up if used in production.
- In this lab setup, persistence is local to the host.

## Incident Response Checklist

- Is the Triton container running?
- Are GPUs visible inside the container?
- Is the model repository mounted correctly?
- Are models loading successfully?
