#!/bin/bash
# Waymo pipeline paths — edit SEQ and paths for your scene.

SEQ="${SEQ:-1023192}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ITERATION="${ITERATION:-30000}"

DATA_CFG="${DATA_CFG:-./configs/waymo.yaml}"

SOURCE_PATH="${SOURCE_PATH:-/mnt/sda/data/waymo_data_hugsim/processed/${SEQ}}"
MODEL_PATH="${MODEL_PATH:-/mnt/sda/projects/hugsim/HUGSIM/models/waymo/${SEQ}}"
EXPORT_PATH="${EXPORT_PATH:-/mnt/sda/projects/hugsim/HUGSIM/export/waymo/${SEQ}}"

REALCAR_PATH="${REALCAR_PATH:-/mnt/sda/data/3dRealCar}"

SIM_BASE_CFG="${SIM_BASE_CFG:-./configs/sim/waymo_base.yaml}"
SIM_CAMERA_CFG="${SIM_CAMERA_CFG:-./configs/sim/waymo_camera.yaml}"
SIM_KINEMATIC_CFG="${SIM_KINEMATIC_CFG:-./configs/sim/kinematic.yaml}"
SIM_AD="${SIM_AD:-ltf}"
SIM_AD_CUDA="${SIM_AD_CUDA:-1}"
