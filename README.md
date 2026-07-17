# qbminer

Minimal CUDA Stratum miner for Qbit PRISM pool SHA256d mining.

This is purpose-built for Qbit/PRISM and avoids the old ccminer dependency tree.

## Quick Start

On the VPS:

```bash
git clone https://github.com/rowanhere/qbit.git
cd qbit
./build.sh
```

`build.sh` installs common Ubuntu build packages, finds `nvcc`, detects the first
GPU's compute capability with `nvidia-smi`, and compiles for that architecture.

Manual build:

```bash
make NVCC=/usr/local/cuda/bin/nvcc ARCH=sm_89
```

## Run

By default, qbminer mines on **all CUDA GPUs** it can see. Each GPU opens its own
Stratum connection. On multi-GPU systems the worker suffix `.gpuN` is appended.

```bash
./qbminer \
  -o mine.prismpool.io:4335 \
  -u qb1zhqwu3s35yyrfsqlr42snrzx7xwgdqhx89vdaupdc4nuyt95y8v4qxttk86.4090vps \
  -p x
```

To force one GPU:

```bash
./qbminer -d 0 \
  -o mine.prismpool.io:4335 \
  -u qb1zhqwu3s35yyrfsqlr42snrzx7xwgdqhx89vdaupdc4nuyt95y8v4qxttk86.4090vps \
  -p x
```

Optional flags:

```text
  -d <device>        CUDA device id; omit to use all GPUs
  -b <blocks>        CUDA blocks per launch, default 131072
  -t <threads>       CUDA threads per block, default 256
```

## Architecture Notes

Common values:

```text
RTX 20xx / T4:        sm_75
RTX 30xx / A6000:     sm_86
RTX 40xx / Ada:       sm_89
RTX 50xx / Blackwell: sm_120, requires CUDA 12.8+
```

Prebuilt binaries are architecture-specific. Source builds are preferred on new
GPU generations because the installed CUDA toolkit must support that GPU.
