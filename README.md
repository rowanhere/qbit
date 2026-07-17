# qbminer

Minimal CUDA Stratum miner for Qbit PRISM pool SHA256d mining.

This is purpose-built for Qbit/PRISM and avoids the old ccminer dependency tree.

## Release Binary

Download the release asset:

```text
qbminer-linux-x86_64-cuda12.tar.gz
```

Use it:

```bash
tar -xzf qbminer-linux-x86_64-cuda12.tar.gz
cd qbminer-linux-x86_64-cuda12
chmod +x qbminer
./qbminer \
  -o mine.prismpool.io:4335 \
  -u qb1zhqwu3s35yyrfsqlr42snrzx7xwgdqhx89vdaupdc4nuyt95y8v4qxttk86.4090vps \
  -p x
```

The release binary is one executable with CUDA fatbin support for common NVIDIA
GPUs: Turing `sm_75`, Ampere `sm_86`, Ada `sm_89`, and Blackwell `sm_120`.

## Build From Source

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
The miner displays a live terminal dashboard with pool, address, uptime, total
hashrate, average hashrate, per-GPU speed, accepted/rejected/stale shares, and
the latest per-GPU event.

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
  --no-dashboard     plain log output instead of the live terminal dashboard
```

For log files or services, use:

```bash
./qbminer --no-dashboard -o mine.prismpool.io:4335 -u YOUR_QBIT_ADDRESS.worker -p x
```

Password `x` uses normal PRISM vardiff behavior. For short testing only, a pool
may accept a difficulty request like `-p d=64`, but normal mining should use
`-p x`.

## Architecture Notes

Common values:

```text
RTX 20xx / T4:        sm_75
RTX 30xx / A6000:     sm_86
RTX 40xx / Ada:       sm_89
RTX 50xx / Blackwell: sm_120, requires CUDA 12.8+
```

The release binary is multi-arch. Source builds are still useful if you want to
tune flags or compile for a GPU generation not included in the release.
