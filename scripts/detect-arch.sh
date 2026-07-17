#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' .')"
  if [[ "$cap" =~ ^[0-9]+$ ]]; then
    echo "sm_${cap}"
    exit 0
  fi
fi

name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
case "$name" in
  *5090*|*5080*|*5070*|*Blackwell*|*"RTX PRO 6000"*) echo "sm_120" ;;
  *4090*|*4080*|*4070*|*4060*|*Ada*) echo "sm_89" ;;
  *3090*|*3080*|*3070*|*3060*|*A6000*|*A5000*|*Ampere*) echo "sm_86" ;;
  *2080*|*2070*|*2060*|*T4*|*Turing*) echo "sm_75" ;;
  *) echo "sm_89" ;;
esac
