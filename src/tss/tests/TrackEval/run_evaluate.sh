#!/bin/bash

# Add python path for MLKit
export PYTHONPATH=$PYTHONPATH:$PWD

START_TIME="$(date -u +%s.%N)"

python scripts/run_mot_challenge.py  \
  --BENCHMARK MOT20  \
  --TRACKERS_TO_EVAL MyTracker  \
  --TRACKERS_FOLDER /media/sugarubuntu/DataSKKU4/4_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/synthehicle_town05/outputs/track_mtmc.txt  \
  --GT_FOLDER /media/sugarubuntu/DataSKKU4/4_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/synthehicle_town05/outputs/track_mtmc.txt  \
  --OUTPUT_FOLDER /media/sugarubuntu/DataSKKU4/4_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/synthehicle_town05/outputs


END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
