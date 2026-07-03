#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUGSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

SCENARIO_PATH="${1:-./configs/benchmark/waymo/${SEQ}_easy_00.yaml}"
AD="${2:-${SIM_AD}}"
AD_CUDA="${3:-${SIM_AD_CUDA}}"

cd "${HUGSIM_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [ ! -f "${EXPORT_PATH}/scene.pth" ]; then
    echo "[simulate] ERROR: missing ${EXPORT_PATH}/scene.pth"
    echo "Run export first: bash scripts/waymo/export.sh"
    exit 1
fi

if [ ! -f "${SCENARIO_PATH}" ]; then
    echo "[simulate] ERROR: scenario not found: ${SCENARIO_PATH}"
    echo "Create one with GUI or copy configs/benchmark/waymo/1179959.yaml and set scene_name: \"${SEQ}\""
    exit 1
fi

run_python() {
    if command -v pixi >/dev/null 2>&1; then
        pixi run python "$@"
    else
        python "$@"
    fi
}

echo "[simulate] scenario=${SCENARIO_PATH}"
echo "[simulate] ad=${AD} sim_gpu=${CUDA_DEVICE} ad_gpu=${AD_CUDA}"

run_python closed_loop.py \
    --scenario_path "${SCENARIO_PATH}" \
    --base_path "${SIM_BASE_CFG}" \
    --camera_path "${SIM_CAMERA_CFG}" \
    --kinematic_path "${SIM_KINEMATIC_CFG}" \
    --ad "${AD}" \
    --ad_cuda "${AD_CUDA}"
