#!/bin/bash
set -euo pipefail

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda
export XFORMERS_DISABLE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# base_dir="/nas/datasets/Waymo_NOTR/static"
# segment="segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"

base_dir="/mnt/sdb/data/waymo_data_hugsim/raw"
segment="segment-10231929575853664160_1160_000_1180_000_with_camera_labels.tfrecord"
# Waymo Open Dataset GCS bucket (EmerNeRF NOTR와 동일)
gcs_source="gs://waymo_open_dataset_scene_flow/train"

seg_prefix=$(echo $segment| cut -c 9-15)
seq_name=${seg_prefix}
out=/mnt/sda/data/waymo_data_hugsim/processed/$seq_name
cameras="1 2 3"

# normalize segment filename
if [[ "${segment}" == *.tfrecord ]]; then
    segment_base="${segment%.tfrecord}"
else
    segment_base="${segment}"
    segment="${segment}.tfrecord"
fi

mkdir -p "${base_dir}" "${out}"

# download tfrecord if missing (requires: gcloud auth login)
tfrecord_path="${base_dir}/${segment}"
if [ ! -f "${tfrecord_path}" ]; then
    echo "[download] ${segment} not found locally, downloading from ${gcs_source} ..."
    if ! command -v gsutil >/dev/null 2>&1; then
        echo "[download] ERROR: gsutil not found. Install Google Cloud SDK and run 'gcloud auth login'."
        exit 1
    fi
    gsutil cp -n "${gcs_source}/${segment_base}.tfrecord" "${base_dir}/"
    if [ ! -f "${tfrecord_path}" ]; then
        echo "[download] ERROR: download finished but ${tfrecord_path} is still missing."
        exit 1
    fi
    echo "[download] saved to ${tfrecord_path}"
else
    echo "[download] using existing ${tfrecord_path}"
fi

# load images, camera pose, etc
python waymo/load.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type waymo
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype waymo
