#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUGSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

cd "${HUGSIM_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [ ! -f "${SOURCE_PATH}/points3d.ply" ]; then
    echo "[reconstruct] ERROR: missing ${SOURCE_PATH}/points3d.ply"
    echo "Run data preprocessing first: cd data && bash waymo/run.sh"
    exit 1
fi

mkdir -p "${MODEL_PATH}"

run_python() {
    if command -v pixi >/dev/null 2>&1; then
        pixi run python "$@"
    else
        python "$@"
    fi
}

echo "[reconstruct] seq=${SEQ} gpu=${CUDA_DEVICE}"
echo "[reconstruct] source=${SOURCE_PATH}"
echo "[reconstruct] model=${MODEL_PATH}"

if [ ! -f "${MODEL_PATH}/ckpts/ground_chkpnt${ITERATION}.pth" ]; then
    echo "[reconstruct] step 1/2: train_ground.py (${ITERATION} iters)"
    run_python -u train_ground.py \
        --data_cfg "${DATA_CFG}" \
        --source_path "${SOURCE_PATH}" \
        --model_path "${MODEL_PATH}"
else
    echo "[reconstruct] skip train_ground.py (checkpoint exists)"
fi

if [ ! -f "${MODEL_PATH}/ckpts/chkpnt${ITERATION}.pth" ]; then
    echo "[reconstruct] step 2/2: train.py (${ITERATION} iters)"
    run_python -u train.py \
        --data_cfg "${DATA_CFG}" \
        --source_path "${SOURCE_PATH}" \
        --model_path "${MODEL_PATH}"
else
    echo "[reconstruct] skip train.py (checkpoint exists)"
fi

echo "[reconstruct] done → ${MODEL_PATH}"
