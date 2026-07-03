#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/waymo/run_pipeline.sh reconstruct
#   bash scripts/waymo/run_pipeline.sh export
#   bash scripts/waymo/run_pipeline.sh export-with-vehicles
#   bash scripts/waymo/run_pipeline.sh simulate [scenario.yaml] [ad] [ad_cuda]
#   bash scripts/waymo/run_pipeline.sh all [scenario.yaml]
#
# Override paths:
#   SEQ=1023192 CUDA_DEVICE=0 bash scripts/waymo/run_pipeline.sh reconstruct

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP="${1:-help}"

case "${STEP}" in
    reconstruct)
        bash "${SCRIPT_DIR}/reconstruct.sh"
        ;;
    export)
        bash "${SCRIPT_DIR}/export.sh"
        ;;
    export-with-vehicles)
        CONVERT_VEHICLES=1 bash "${SCRIPT_DIR}/export.sh"
        ;;
    simulate)
        shift
        bash "${SCRIPT_DIR}/simulate.sh" "$@"
        ;;
    all)
        SCENARIO="${2:-}"
        bash "${SCRIPT_DIR}/reconstruct.sh"
        CONVERT_VEHICLES=1 bash "${SCRIPT_DIR}/export.sh"
        if [ -n "${SCENARIO}" ]; then
            bash "${SCRIPT_DIR}/simulate.sh" "${SCENARIO}"
        else
            echo "[pipeline] reconstruct + export done."
            echo "[pipeline] configure scenario in GUI, then run:"
            echo "  bash scripts/waymo/run_pipeline.sh simulate configs/benchmark/waymo/YOUR_SCENARIO.yaml"
        fi
        ;;
    help|*)
        echo "Usage: bash scripts/waymo/run_pipeline.sh {reconstruct|export|export-with-vehicles|simulate|all}"
        echo ""
        echo "Examples:"
        echo "  SEQ=1023192 bash scripts/waymo/run_pipeline.sh reconstruct"
        echo "  bash scripts/waymo/run_pipeline.sh export-with-vehicles"
        echo "  bash scripts/waymo/run_pipeline.sh simulate configs/benchmark/waymo/my_scene.yaml ltf 1"
        ;;
esac
