#!/bin/bash

# Run each camera
python main.py  \
   --dataset aicity2021_final  \
   --config cam_1.yaml  \
   --write_video True

# Run all cameras
#python main_all.py  \
#   --dataset aicity2021_final \
#   --write_video True
