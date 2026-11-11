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

echo "PROCESS SINGLE CAMERA TRACKING"

# NOTE: TRACKING + DRAW MOTS + MATCHING ZONE + DRAW MOTS WITH ZONE
# One process per camera: MTMCTrackingCamera.run() runs the four tasks in this order.
python runs/main_mtmc.py --dataset ${DATASET}  --tracking --save_tracks_img \
   --matching_zone --use_mots_feat_raw --save_tracks_zone \
   --config "19_SUWON.yaml" --config_feat_merge "reid_merge.yaml"  &
python runs/main_mtmc.py --dataset ${DATASET}  --tracking --save_tracks_img \
   --matching_zone --use_mots_feat_raw --save_tracks_zone \
   --config "20_SUWON.yaml" --config_feat_merge "reid_merge.yaml"  &
python runs/main_mtmc.py --dataset ${DATASET}  --tracking --save_tracks_img \
   --matching_zone --use_mots_feat_raw --save_tracks_zone \
   --config "46_SUWON.yaml" --config_feat_merge "reid_merge.yaml"
wait


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
