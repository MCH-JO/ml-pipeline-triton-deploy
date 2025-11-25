# ML Platform – PyTorch to Triton (GPU Serving)

This repository demonstrates a **production-oriented ML serving platform**
designed from an **infrastructure / platform engineering perspective**.

The goal is not model accuracy, but:

- reproducible environments
- GPU-enabled serving
- model versioning
- observability
- operational workflows

---

## Architecture

High-level workflow:

- Model training with PyTorch
- Experiment tracking and artifacts stored in MLflow
- Model export to ONNX
- Promotion of models to a Triton model repository
- GPU inference served by NVIDIA Triton Inference Server
- Client-side inference and benchmarking

The platform is designed to be:

- containerized
- GPU-aware
- close to real production constraints

See: `docs/architecture.md`

---

## Model Serving Pipeline

1. Train a simple PyTorch model
2. Export model to ONNX format
3. Track artifacts with MLflow
4. Promote selected model versions to Triton
5. Serve models via Triton Inference Server
6. Run inference from a client application

Model versions are explicitly managed inside the Triton model repository.

---

## Quickstart (Docker Compose)

Prerequisites:

- Linux host
- NVIDIA GPU + drivers + CUDA
- Docker + NVIDIA Container Toolkit

```bash
docker compose -f serving/triton/compose.yaml up -d

# ------------------------
# Verify Triton is running:
# ------------------------
curl localhost:8000/v2/health/ready

# ------------------------
# Activate python env and install client requirements
# ------------------------
source <env-path>/bin/activate
pip install -r client/requirements.txt
python3 client/infer.py --url localhost:8000 --model simple
```

This validates:

- model availability
- inference latency
- GPU execution

## Observability

The platform is designed with observability in mind.

Metrics and signals of interest:

- GPU utilization
- inference latency
- model load status
- Triton server health

## Runbook & Operations

Operational documentation is available:

- Start / stop procedures
- Model promotion workflow
- Common failure scenarios
- GPU and container troubleshooting

See:
`docs/runbook.md`
`docs/troubleshooting.md`

## Roadmap

- Kubernetes deployment (kind / Helm)
- GPU metrics integration (DCGM exporter)
- Load testing and autoscaling experiments
- CI checks for model repository consistency
