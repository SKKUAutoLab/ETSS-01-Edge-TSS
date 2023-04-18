#!/bin/bash

docker run  \
    --ipc=host  \
    --gpus all   \
    -v /media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test_docker:/usr/src/aic23-track_5/data   \
    -it supersugar/skku_automation_lab_aic23_track_5:latest

# -it skku_automation_lab_aic23_track_5
#-v /media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/mon/project/aic-tss:/usr/src/aic23-track_5  \
