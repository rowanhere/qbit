# qbminer

Minimal CUDA Stratum miner for Qbit PRISM pool SHA256d mining.

This is purpose-built for Qbit/PRISM and avoids the old ccminer dependency tree.

## Build

On the VPS:

```bash
apt update
apt install -y build-essential cuda-toolkit-12-8
cd qbminer
make
```

For an RTX 4090, the default architecture is `sm_89`.

If your CUDA toolkit is elsewhere:

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

## Run

```bash
./qbminer \
  -o mine.prismpool.io:4335 \
  -u qb1zhqwu3s35yyrfsqlr42snrzx7xwgdqhx89vdaupdc4nuyt95y8v4qxttk86.4090vps \
  -p x
```

Optional flags:

```text
  -d <device>        CUDA device id, default 0
  -b <blocks>        CUDA blocks per launch, default 131072
  -t <threads>       CUDA threads per block, default 256
```

This first version is intentionally compact and conservative. It should be easier to build than ccminer, but it is not yet deeply optimized.
