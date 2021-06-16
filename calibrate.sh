#!/bin/bash
# Usage:
#
#   ./calibrate.sh recording-folder
#
# where the folder should contain the files data.jsonl, data.???. data2.???,
# that represent a sequence where a calibration pattern is filmed appropriately.
# It is also assumed that, in the same folder with this script, there exists
# a folder allan/ and the file april_6x6_80x80cm.yaml, where the size of the
# ApriTag is correctly specified (please check this before running)
#
# In particular, set (example): tagSize: 0.03765
# where the number is the side length of the grid divided by 8.1
#
set -eux -o pipefail

CAM_MODEL=pinhole-radtan
tmp_dir=tmp
rm -rf tmp
mkdir tmp
cp -R "$1" $tmp_dir/camera_calibration_raw
cp april_6x6_80x80cm.yaml $tmp_dir
cp -R allan $tmp_dir/
python3 ../vio_benchmark/convert/jsonl-to-kalibr.py $tmp_dir/camera_calibration_raw -output $tmp_dir/converted/
DOCKER_RUN="docker run -v `pwd`/$tmp_dir:/kalibr -it stereolabs/kalibr:kinetic"
$DOCKER_RUN kalibr_bagcreater --folder /kalibr/converted --output-bag /kalibr/data.bag
set +e
$DOCKER_RUN bash -c "cd /kalibr && kalibr_calibrate_cameras --bag data.bag \
    --topics /cam0/image_raw /cam1/image_raw \
    --models $CAM_MODEL $CAM_MODEL \
    --target april_6x6_80x80cm.yaml \
    --dont-show-report"
$DOCKER_RUN bash -c "cd kalibr && kalibr_calibrate_imu_camera --bag data.bag \
    --cams camchain-data.yaml \
    --target april_6x6_80x80cm.yaml \
    --imu allan/imu.yaml  \
    --dont-show-report"
set -e
python ../stereo-vio-code/scripts/convert/kalibr-to-parameters.py $tmp_dir/camchain-imucam-data.yaml
mv parameters.txt $tmp_dir/camera_calibration_raw/
