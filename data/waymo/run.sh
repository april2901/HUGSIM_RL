#!/bin/bash

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda
export XFORMERS_DISABLE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# base_dir="/nas/datasets/Waymo_NOTR/static"
# segment="segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"

base_dir="/mnt/sdb/data/waymo_data_hugsim/raw"
segment="segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord"

seg_prefix=$(echo $segment| cut -c 9-15)
seq_name=${seg_prefix}
out=/mnt/sda/data/waymo_data_hugsim/processed/$seq_name
cameras="1 2 3"


mkdir -p $out

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