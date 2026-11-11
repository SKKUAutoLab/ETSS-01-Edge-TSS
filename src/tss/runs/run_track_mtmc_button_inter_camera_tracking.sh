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

echo "PROCESS INTER CAMERA TRACKING"

# NOTE: SUB-CLUSTER
python runs/main_mtmc.py --dataset ${DATASET}  --matching_scene \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: WRITE RESULT
python runs/main_mtmc.py --dataset ${DATASET}  --write_result \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: DRAW FINAL RESULT
python runs/main_mtmc.py --dataset ${DATASET}  --save_mtmc_result \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: EVALUATION
python tests/mot_evaluation/evaluate_mtmc.py  \
  --gt data/tsd_suwon_19_46_20/groundtruths/gt.txt  \
  --re data/tsd_suwon_19_46_20/temps/mtmc_result.txt  \
  --ou data/tsd_suwon_19_46_20/outputs/track_mtmc.txt


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
