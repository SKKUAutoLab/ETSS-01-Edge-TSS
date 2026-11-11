#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1
export MKL_SERVICE_FORCE_INTEL=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")

DATASET="tsd_suwon_19_46_20"
DATASETSUBSET="clearsky_1764390903"

# NOTE: IMAGE EXTRACTION
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "19_SUWON.yaml"
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "20_SUWON.yaml"
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "46_SUWON.yaml"
#wait

echo "PROCESS DETECTION"

# NOTE: DETECTION
img_size=1280
conf_thres=0.1
gpu_id=0
# shellcheck disable=SC2068
CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.detectors.detect   \
  --name 19_SUWON  \
  --output data/${DATASET}/outputs/  \
  --source data/${DATASET}/outputs/images/19_SUWON  \
  --weights models_zoo/yolox/yolox_x_19_suwon/best_ckpt.pth  \
  --exp_file third_party/YOLOX/exps/example/custom/yolox_x_19_suwon.py  \
  --roi_region data/${DATASET}/${DATASETSUBSET}/19_SUWON/19_SUWON_roi.jpg \
  --conf ${conf_thres}  \
  --agnostic  \
  --save-txt  \
  --save-conf  \
  --img-size ${img_size}

CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.detectors.detect   \
  --name 20_SUWON  \
  --output data/${DATASET}/outputs/  \
  --source data/${DATASET}/outputs/images/20_SUWON  \
  --weights models_zoo/yolox/yolox_x_20_suwon/best_ckpt.pth  \
  --exp_file third_party/YOLOX/exps/example/custom/yolox_x_20_suwon.py  \
  --roi_region data/${DATASET}/${DATASETSUBSET}/20_SUWON/20_SUWON_roi.jpg \
  --conf ${conf_thres}  \
  --agnostic  \
  --save-txt  \
  --save-conf  \
  --img-size ${img_size}

CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.detectors.detect   \
  --name 46_SUWON  \
  --output data/${DATASET}/outputs/  \
  --source data/${DATASET}/outputs/images/46_SUWON  \
  --weights models_zoo/yolox/yolox_x_46_suwon/best_ckpt.pth  \
  --exp_file third_party/YOLOX/exps/example/custom/yolox_x_46_suwon.py  \
  --roi_region data/${DATASET}/${DATASETSUBSET}/46_SUWON/46_SUWON_roi.jpg \
  --conf ${conf_thres}  \
  --agnostic  \
  --save-txt  \
  --save-conf  \
  --img-size ${img_size}
wait

END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
