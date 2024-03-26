#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .
export DIR_SOURCE=$DIR_TSS"/motordriver"            # ./motordriver

# Add data dir
#export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train"
#export DIR_RESULT="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train"

export DIR_DATA="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test"
export DIR_RESULT="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test"

# DEBUG: print data dir
#echo "DIR_TSS: $DIR_TSS"
#echo "DIR_SOURCE: $DIR_SOURCE"
#echo "DIR_DATA: $DIR_DATA"
#echo "DIR_RESULT: $DIR_RESULT"

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
#echo "*********"
#echo "DETECTION"
#echo "*********"
#python main.py  \
#    --detection  \
#    --run_image  \
#    --config $DIR_TSS"/configs/aic24.yaml"

#python main.py  --detection  --run_image --config $DIR_TSS"/configs/aic24_001_025.yaml" &
#python main.py  --detection  --run_image --config $DIR_TSS"/configs/aic24_026_050.yaml" &
#python main.py  --detection  --run_image --config $DIR_TSS"/configs/aic24_051_075.yaml" &
#python main.py  --detection  --run_image --config $DIR_TSS"/configs/aic24_076_100.yaml"

#python main.py  \
#    --detection  \
#    --run_image  \
#    --drawing  \
#    --config $DIR_TSS"/configs/aic24.yaml"


# NOTE: IDENTIFICATION
#echo "**************"
#echo "IDENTIFICATION"
#echo "**************"
#python main.py  \
#    --identification  \
#    --config $DIR_TSS"/configs/aic24.yaml"

#python main.py  \
#    --identification  \
#    --drawing  \
#    --config $DIR_TSS"/configs/aic24_yolov9.yaml"

# NOTE: HEURISTIC PROCESS
#echo "*****************"
#echo "HEURISTIC PROCESS"
#echo "*****************"
#python main.py  \
#    --heuristic  \
#    --run_image  \
#    --config $DIR_TSS"/configs/aic24.yaml"

#python main.py  \
#    --heuristic  \
#    --run_image  \
#    --config $DIR_TSS"/configs/aic24_yolov9.yaml"


# NOTE: WRITE FINAL RESULT
#echo "*****************"
#echo "WRITE FINAL RESULT"
#echo "*****************"
#python main.py  \
#    --write_final  \
#    --config $DIR_TSS"/configs/aic24.yaml"

#python main.py  \
#    --write_final  \
#    --config $DIR_TSS"/configs/aic24_yolov9.yaml"


# NOTE: DRAW FINAL RESULT
echo "*****************"
echo "DRAW FINAL RESULT"
echo "*****************"
#python utilities/drawing_result_image.py \
#    --draw_final  \
#    --path_final "${DIR_DATA}/output_aic24/final_result.txt"  \
#    --path_video_out "${DIR_DATA}/output_aic24/final_result_debug/"  \
#    --path_video_in "${DIR_DATA}/images/"

python utilities/drawing_result_image.py \
    --draw_final  \
    --path_final "${DIR_DATA}/output_aic24/final_result_Submission_12_0.0001.txt"  \
    --path_video_out "${DIR_DATA}/output_aic24/final_result_debug_Submission_12/"  \
    --path_video_in "${DIR_DATA}/images/"

# NOTE: EVALUATION RESULT
#echo "*****************"
#echo "EVALUATION RESULT"
#echo "*****************"
#python utilities/evaluator.py  \
#    --anno_file_json_path /media/sugarubuntu/DataSKKU2/2_Dataset/COCO_dataset/example/gt_aic2024_train.json  \
#    --result_file_txt_path /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result.txt  \
#    --result_file_json_path /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result.json  \
#    --result_evaluation_txt_path /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result_evaluation_2024030601.txt

#python utilities/aicity24_eval_script_track5/aicityeval-helmet.py  \
#    --ground_truth_file /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/gt_clean.txt  \
#    --predictions_file /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result.txt

#python utilities/aicity24_eval_script_track5/aicityeval-helmet.py  \
#--ground_truth_file /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/gt_clean.txt  \
#--predictions_file /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/final_result_add_mini.txt

# NOTE: MERGE RESULT
#echo "************"
#echo "MERGE RESULT"
#echo "************"
#python utilities/merge_result.py  \
#  --result_list  \
#     "${DIR_DATA}/output_aic24/final_result_Submission_07.txt"  \
#     "${DIR_DATA}/output_aic24/final_result_Submission_09.txt"  \
#  --class_list  \
#     "1,2,3,4,5,6,7,8"  \
#     "9"  \
#  --result_merge  \
#     "${DIR_DATA}/output_aic24/final_result_Submission_07_09.txt"


# NOTE: SUMMARY RESULT
#echo "*****************"
#echo "SUMMARY RESULT"
#echo "*****************"
#python utilities/summary_gt.py  \
#  --result_file_txt_path "${DIR_DATA}/output_aic24/final_result.txt"  \
#  --summary_path_txt_path "${DIR_DATA}/output_aic24/final_result_summary.txt"

#python utilities/summary_gt.py  \
#  --result_file_txt_path "${DIR_DATA}/output_aic24/final_result.txt"  \
#  --summary_path_txt_path "${DIR_DATA}/output_aic24/final_result_summary.txt"

echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
