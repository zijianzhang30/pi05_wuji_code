#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
export DEX_OPTIX_WEIGHTS="$DEX_PROJECT_ROOT/runtime/optix580/usr/share/nvidia/nvoptix.bin"
export LD_PRELOAD="$DEX_PROJECT_ROOT/runtime/liboptix_weights_redirect.so${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

export PYTHONPATH="$DEX_PROJECT_ROOT/runtime/python${PYTHONPATH:+:$PYTHONPATH}"
