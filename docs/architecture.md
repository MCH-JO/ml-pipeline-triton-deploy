# Architecture – ML Platform (PyTorch → Triton)

## Overview

This platform provides a **GPU-enabled ML serving environment** designed from an **infrastructure and operations perspective**.

The objective is to expose a reproducible and observable inference platform rather than to optimize model accuracy.

---

## High-Level Components

### 1. Model Training & Export

- PyTorch is used to define and train a simple model.
- Models are exported to **ONNX** format for framework-agnostic serving.
- Artifacts are tracked using **MLflow**.

### 2. Model Registry & Promotion

- MLflow acts as an experiment tracker and artifact store.
- A promotion script selects a trained model and promotes it to Triton by:
  - copying the ONNX artifact
  - creating a new version directory in the Triton model repository
  - keeping traceability of the source run

### 3. Model Serving

- **NVIDIA Triton Inference Server** serves models on GPU.
- Models are loaded from a versioned model repository.
- The server exposes HTTP/gRPC endpoints for inference and health checks.

### 4. Client Inference

- A lightweight Python client sends inference requests.
- Used for functional validation and basic latency measurement.

---

## Deployment Architecture

- GPU access is provided via NVIDIA Container Toolkit.
- Triton runs as a standalone serving component.

```text
+-------------------+
|   Client (Python) |
+---------+---------+
          |
          v
+---------+---------+
| Triton Inference  |
| Server (GPU)      |
+---------+---------+
          |
          v
+---------+---------+
| Model Repository  |
| (versioned ONNX)  |
+-------------------+
```

## Model Repository Structure

Models are versioned following Triton conventions:

```text
model_repository/
└── simple/
    ├── 1/
    │   └── model.onnx
    ├── 2/
    │   └── model.onnx
    └── config.pbtxt

```

This allows:

- safe model upgrades
- rollback to previous versions
- reproducibility of inference behavior

## Observability (Design)

The platform is designed to expose operational signals such as:

- Triton server health
- model load status
- inference latency
- GPU usage

## Future Evolution

- Kubernetes deployment (kind / Helm)
- GPU metrics via NVIDIA DCGM exporter
- Horizontal scaling experiments
- Integration with CI pipelines
