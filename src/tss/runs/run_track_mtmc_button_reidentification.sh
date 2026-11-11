#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1
export MKL_SERVICE_FORCE_INTEL=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
# shellcheck disable=SC2034
DIR_CURRENT=$(dirname "${THIS}")
 
DATASET="tsd_suwon_19_46_20"
DATASETSUBSET="clearsky_1764390903"

echo "PROCESS REIDENTIFICATION"

# NOTE: FEATURE EXTRACTION - 1 - resnet101_ibn_a_2
python runs/main_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/19_SUWON_resnet101_ibn_a_2.yaml" &
python runs/main_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/46_SUWON_resnet101_ibn_a_2.yaml" &
python runs/main_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/20_SUWON_resnet101_ibn_a_2.yaml"
wait

# NOTE: MERGE ALL FEATURES
python3 runs/main_mtmc.py  --dataset ${DATASET} \
    --config "default.yaml" --feature_merge --config_feat_merge "reid_merge.yaml"
wait


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
