#!/bin/bash


START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")

DATASET="tsd_suwon_19_46_20"
DATASETSUBSET="clearsky_1764390903"

echo "PROCESS EXTRACTION"

bash runs/run_track_mtmc_button_image_extraction.sh

echo "PROCESS DETECTION"

bash runs/run_track_mtmc_button_detection.sh
#bash runs/run_track_mtmc_button_detection.sh  2>/dev/null
wait

echo "PROCESS REIDENTIFICATION"

bash runs/run_track_mtmc_button_reidentification.sh
wait

echo "PROCESS SINGLE CAMERA TRACKING"

bash runs/run_track_mtmc_button_single_camera_tracking.sh
wait

echo "PROCESS INTER CAMERA TRACKING"

bash runs/run_track_mtmc_button_inter_camera_tracking.sh
wait

#python /media/sugarubuntu/DataSKKU4/4_Workspace/_MAIN_CODE/src/imagesorvideo/images_video.py


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
