#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # mon/project/aic-tss/script
export DIR_TSS=$(dirname $DIR_CURRENT)              # mon/project/aic-tss
export DIR_SOURCE=$DIR_TSS"/src"                    # mon/project/aic-tss/scr
export DIR_MON=$(dirname $(dirname $DIR_TSS))"/src" # mon/scr/

# Add data dir
export DIR_DATA="/usr/src/aic23-track_5/data"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD        # .
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE # mon/project/aic-tss/scr
export PYTHONPATH=$PYTHONPATH:$DIR_MON    # mon/scr/

export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

# NOTE: COPY FILE
echo "*******"
echo "COPYING"
echo "*******"
cp -f $DIR_TSS"/configs/class_labels_1cls.json" $DIR_TSS"/data/class_labels_1cls.json" 
cp -f $DIR_TSS"/configs/class_labels_7cls.json" $DIR_TSS"/data/class_labels_7cls.json" 

# NOTE: DETECTION
echo "**********"
echo "EXTRACTION"
echo "**********"
python $DIR_SOURCE/utils/extract_frame.py  \
    --source      $DIR_DATA"/videos/" \
    --destination $DIR_DATA"/images/" \
    --verbose

# NOTE: DETECTION
echo "*********"
echo "DETECTION"
echo "*********"
python $DIR_SOURCE/main_s2.py  \
    --detection  \
    --run_image  \
    --config $DIR_TSS"/configs/solution_motorbike_violation.yaml"

# NOTE: DRAW DETECTION RESULT
echo "*********************"
echo "DRAW DETECTION RESULT"
echo "*********************"
python $DIR_SOURCE/utils/drawing_result_image.py \
    --draw_pickle  \
    --path_pickle_in "${DIR_DATA}/outputs_s2_v8_det_v8_iden/dets_crop_pkl/yolov8x6/"  \
    --path_video_out "${DIR_DATA}/outputs_s2_v8_det_v8_iden/dets_crop_pkl_debug/"  \
    --path_video_in "${DIR_DATA}/images/"

# NOTE: IDENTIFICATION
echo "**************"
echo "IDENTIFICATION"
echo "**************"
python $DIR_SOURCE/main_s2.py  \
    --identification  \
    --config $DIR_TSS"/configs/solution_motorbike_violation.yaml"

# NOTE: HEURISTIC PROCESS
echo "*****************"
echo "HEURISTIC PROCESS"
echo "*****************"
python $DIR_SOURCE/main_s2.py  \
    --heuristic  \
    --config $DIR_TSS"/configs/solution_motorbike_violation.yaml"

# NOTE: WRITE FINAL RESULT
echo "*****************"
echo "WRITE FINAL RESULT"
echo "*****************"
python $DIR_SOURCE/main_s2.py  \
    --write_final  \
    --config $DIR_TSS"/configs/solution_motorbike_violation.yaml"

# NOTE: DRAW FINAL RESULT
echo "*****************"
echo "DRAW FINAL RESULT"
echo "*****************"
python $DIR_SOURCE/utils/drawing_result_image.py \
    --draw_final  \
    --path_final "${DIR_DATA}/outputs_s2_v8_det_v8_iden/final_result_s2.txt"  \
    --path_video_out "${DIR_DATA}/outputs_s2_v8_det_v8_iden/dets_final_debug/"  \
    --path_video_in "${DIR_DATA}/images/"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
