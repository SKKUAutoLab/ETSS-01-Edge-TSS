#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")

DATASET="synthehicle_town05"
DATASETSUBSET="N-rain"
#DATASET="aic22_mtmc"
#DATASETSUBSET="dataset_a"

# NOTE: IMAGE EXTRACTION
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c01.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c02.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c03.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c04.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --images_extraction \
#--config "c05.yaml"
#wait

# NOTE: DETECTION
#bash "${DIR_CURRENT}/run_track_mtmc_det.sh"
#wait

# NOTE: FEATURE EXTRACTION
#bash "${DIR_CURRENT}/run_track_mtmc_reid.sh"
#wait

# NOTE: MERGE ALL FEATURES
python3 runs/main_aic21_track_mtmc.py  --dataset ${DATASET} \
    --config "default.yaml" --feature_merge --config_feat_merge "reid_merge.yaml"
wait

# NOTE: RUN DETECTIONS FILTER
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --detection_filter \
#--config "c01.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --detection_filter \
#--config "c02.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --detection_filter \
#--config "c03.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --detection_filter \
#--config "c04.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --detection_filter \
#--config "c05.yaml"
#wait

# NOTE: DRAW DETECTIONS FILTER
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_dets_img_filter \
#--config "c01.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_dets_img_filter \
#--config "c02.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_dets_img_filter \
#--config "c03.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_dets_img_filter \
#--config "c04.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_dets_img_filter \
#--config "c05.yaml"
#wait

# NOTE: RUN TRACKING
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking --use_dets_feat_filter \
   --config "c01.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking --use_dets_feat_filter \
   --config "c02.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking --use_dets_feat_filter \
   --config "c03.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking --use_dets_feat_filter \
   --config "c04.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --tracking --use_dets_feat_filter \
   --config "c05.yaml"
wait

# NOTE: DRAW MOTS
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c01.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c02.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c03.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c04.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_img \
#--config "c05.yaml"
#wait

# NOTE: MATCHING ZONE
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c01.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c02.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c03.yaml"
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c04.yaml" &
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_zone --use_mots_feat_raw \
   --config "c05.yaml"
wait

# NOTE: DRAW MOTS WITH ZONE
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c01.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c02.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c03.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c04.yaml" &
#python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_tracks_zone \
# --config "c05.yaml"
#wait

# NOTE: SUB-CLUSTER
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --matching_scene \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: WRITE RESULT
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --write_result \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: DRAW FINAL RESULT
python3 runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --save_mtmc_result \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
