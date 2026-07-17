#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

install_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  if [ "$(id -u)" -ne 0 ]; then
    SUDO=sudo
  else
    SUDO=
  fi

  $SUDO apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ca-certificates git make

  if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8 || \
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
  fi
}

find_nvcc() {
  if [ -n "${NVCC:-}" ] && [ -x "$NVCC" ]; then
    echo "$NVCC"
  elif command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    echo /usr/local/cuda/bin/nvcc
  else
    echo "nvcc not found. Install a CUDA toolkit that supports your GPU." >&2
    exit 1
  fi
}

install_packages
NVCC_BIN="$(find_nvcc)"
ARCH="${ARCH:-$(bash "$ROOT/scripts/detect-arch.sh")}"

echo "Using NVCC=$NVCC_BIN"
echo "Using ARCH=$ARCH"

make clean
make NVCC="$NVCC_BIN" ARCH="$ARCH"

echo
echo "Built: $ROOT/qbminer"
