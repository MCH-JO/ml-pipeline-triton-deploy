# Hardware & Software Compatibility Notes

This document records hardware constraints and compatibility decisions
encountered while building the ML serving platform.

## GPU Hardware

- Architecture: NVIDIA Pascal
- Compute Capability: 6.1
- Consumer-grade GPU (no MIG support)

These constraints impose limitations on supported drivers and software versions.

## Compatibility Constraints

Drivers and software versions had to be considered to ensure compatiblity and stability.
Drivers and stability dictated the choice of OS version.
Selected OS: Ubuntu 20.04

### NVIDIA Drivers

Because of choosen versions, drivers can't be installed with CUDA here.

- Recent drivers are required by newer CUDA versions.
- Older GPUs (like Pascal) limit the maximum supported driver version.
- Older drivers limit the maximum compatible OS version

### CUDA

- CUDA versions above a certain threshold are not supported on Pascal GPUs.
- The platform uses a CUDA version compatible with both:
  - the installed driver
  - the target GPU architecture

### PyTorch

- PyTorch GPU builds are tightly coupled to CUDA versions.
- The selected PyTorch version was chosen to:
  - support CUDA compatibility
  - allow ONNX export without runtime errors

### Triton Inference Server

- Triton versions are coupled to CUDA and driver versions.
- Not all Triton releases support older GPU architectures.

### Selected Software Versions

Component | Version | Reason
|:----|----|----|
NVIDIA Drivers | 470 | Compatible with Pascal GPU + provided by Ubuntu 20.04
CUDA | 11.4 | Maximum supported version
NVIDIA Container Toolkit | 1.17.8 | Compatible with Ubuntu 20.04

PyTorch | 1.12 | CUDA-compatible build
Triton | 2.15 | Supports Pascal + CUDA

(Exact versions documented in deployment files)

## Lessons Learned

- GPU architecture must be considered early in platform design.
- Upgrading a single component can break the entire stack.
- Version pinning and documentation are mandatory for stability.
- Production platforms should rely on validated compatibility matrices.

## Production Recommendations

- Maintain a compatibility matrix per hardware generation.
- Pin versions in container images.
- Test upgrades in isolated environments.
- Document all constraints and decisions.
