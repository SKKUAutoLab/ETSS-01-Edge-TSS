#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .
export DIR_SOURCE=$DIR_TSS"/motordriver"            # ./motordriver

# Add data dir
export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test"
export DIR_RESULT="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test"

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD                              # .
export PYTHONPATH=$PYTHONPATH:$DIR_SOURCE                       # ./motordriver

export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

echo "###########################"
echo "STARTING"
echo "###########################"

# NOTE: COPY FILE
cp -f $DIR_TSS"/configs/class_labels_1cls.json" $DIR_TSS"/data/class_labels_1cls.json"
cp -f $DIR_TSS"/configs/class_labels_9cls.json" $DIR_TSS"/data/class_labels_9cls.json"

# NOTE: EXTRACTION
#echo "**********"
#echo "EXTRACTION"
#echo "**********"
#python utilities/extract_frame.py  \
#    --source $DIR_DATA"/videos/" \
#    --destination $DIR_DATA"/images/" \
#    --verbose

# NOTE: DETECTION
echo "*********"
echo "DETECTION"
echo "*********"
python main.py  \
    --detection  \
    --identification  \
    --heuristic  \
    --run_image  \
    --config $DIR_TSS"/configs/aic24.yaml"

# NOTE: WRITE FINAL RESULT
echo "*****************"
echo "WRITE FINAL RESULT"
echo "*****************"
python main.py  \
    --write_final  \
    --config $DIR_TSS"/configs/aic24.yaml"


echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
