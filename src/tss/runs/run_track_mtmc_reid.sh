#!/bin/bash

DATASET="synthehicle_town05"
DATASETSUBSET="N-rain"


# NOTE: FEATURE EXTRACTION - 1 - resnet101_ibn_a_2
python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c01_resnet101_ibn_a_2.yaml" &
python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c02_resnet101_ibn_a_2.yaml" &
python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c03_resnet101_ibn_a_2.yaml" &
python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c04_resnet101_ibn_a_2.yaml" &
python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
--config "reid/c05_resnet101_ibn_a_2.yaml"
wait

# NOTE: FEATURE EXTRACTION - 2 - resnet101_ibn_a_3
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c01_resnet101_ibn_a_3.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c02_resnet101_ibn_a_3.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c03_resnet101_ibn_a_3.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c04_resnet101_ibn_a_3.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c05_resnet101_ibn_a_3.yaml"
#wait

# NOTE: FEATURE EXTRACTION - 3 - resnext101_ibn_a_2
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c01_resnext101_ibn_a_2.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c02_resnext101_ibn_a_2.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c03_resnext101_ibn_a_2.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c04_resnext101_ibn_a_2.yaml" &
#python runs/main_aic21_track_mtmc.py --dataset ${DATASET}  --feature_extraction \
#--config "reid/c05_resnext101_ibn_a_2.yaml"
#wait

