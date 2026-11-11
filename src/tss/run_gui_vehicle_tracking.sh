#!/bin/bash


#export CUDA_LAUNCH_BLOCKING=1
export MKL_SERVICE_FORCE_INTEL=1

#cd /media/sugarubuntu/DataSKKU4/4_Workspace/traffic_surveillance_system/MLKit/

eval "$(conda shell.bash hook)"

conda activate etss-vehicereidentification

time python src/views/multicamera_multiobject_tracking_view.py

