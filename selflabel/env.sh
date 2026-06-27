#!/usr/bin/env bash
# Source this before running any selflabel script.
#   source selflabel/env.sh
# Sets up the dedicated venv + the CUDA13/cuDNN9 libs that onnxruntime-gpu 1.27 needs
# (borrowed from the adventure_ctrgcn venv's nvidia wheels — they are standard NVIDIA
# redistributables, ABI-stable). Without these, rtmlib silently falls back to slow CPU.
CONTEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV="$CONTEST_ROOT/.venv"
export PY="$VENV/bin/python"
_NV=/home/pairlab/adventure_ctrgcn/.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$_NV/cu13/lib:$_NV/cudnn/lib:$LD_LIBRARY_PATH"
# HF_TOKEN must be exported by you before running (never committed):
#   export HF_TOKEN=hf_xxx
echo "selflabel env ready: PY=$PY  (HF_TOKEN ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET})"
