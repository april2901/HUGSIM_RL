#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUGSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

CONVERT_VEHICLES="${CONVERT_VEHICLES:-0}"

cd "${HUGSIM_ROOT}"

if [ ! -f "${MODEL_PATH}/ckpts/chkpnt${ITERATION}.pth" ]; then
    echo "[export] ERROR: missing ${MODEL_PATH}/ckpts/chkpnt${ITERATION}.pth"
    echo "Run reconstruction first: bash scripts/waymo/reconstruct.sh"
    exit 1
fi

run_python() {
    if command -v pixi >/dev/null 2>&1; then
        pixi run python "$@"
    else
        python "$@"
    fi
}

mkdir -p "${EXPORT_PATH}"

echo "[export] model=${MODEL_PATH}"
echo "[export] export=${EXPORT_PATH}"

run_python eval_render/export_scene.py \
    --model_path "${MODEL_PATH}" \
    --output_path "${EXPORT_PATH}" \
    --iteration "${ITERATION}"

run_python eval_render/convert_scene.py \
    --model_path "${EXPORT_PATH}"

if [ "${CONVERT_VEHICLES}" = "1" ]; then
    echo "[export] converting 3DRealCar assets for GUI"
    run_python eval_render/convert_vehicles.py \
        --vehicle_path "${REALCAR_PATH}"
fi

echo "[export] done → ${EXPORT_PATH}"
echo "[export] GUI: cd gui && python app.py --scene ${EXPORT_PATH} --car_folder ${REALCAR_PATH}/converted"
