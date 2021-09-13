#!/bin/bash
# Usage:
#
#   ./docker_calibrate.sh recording-folder apriltag-size-in-m
#
# where the folder should contain the files data.jsonl, data.???. data2.???,
# that represent a sequence where a calibration pattern is filmed appropriately.
#
# NOTE: the AprilTag size is the size of one tag in the grid (in meters), not
# the size of the entire grid

set -eu -o pipefail

RECORDING="$1"
CAM_MODEL=pinhole-radtan
# CAM_MODEL=pinhole-equi # Kannala-Brandt 4
tmp_dir=tmp
DOCKER_KALIBR_RUN="docker run --rm -v `pwd`/$tmp_dir:/kalibr -it stereolabs/kalibr:kinetic"
DOCKER_OURS_RUN="docker run --rm -v `pwd`/$tmp_dir:/kalibr -it ghcr.io/spectacularai/kalibr-conversion:1.0"
APRIL_TAG_SIZE=$2

# must clear using docker to avoid permission issues
docker run -v `pwd`:/cur -it stereolabs/kalibr:kinetic rm -rf /cur/tmp
mkdir -p tmp/allan
cp -R $RECORDING $tmp_dir/camera_calibration_raw

printf "target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: ${APRIL_TAG_SIZE}
tagSpacing: 0.3" >> $tmp_dir/april_6x6_80x80cm.yaml

# May have some (limited) effect on IMU-camera calibration
printf "accelerometer_noise_density: 0.00074562202949377
accelerometer_random_walk: 0.0011061605306550387
gyroscope_noise_density: 3.115084637301622e-05
gyroscope_random_walk: 1.5610557862757885e-05
rostopic: /imu0
update_rate: 400
" >> $tmp_dir/allan/imu.yaml

$DOCKER_OURS_RUN python3 /scripts/jsonl-to-kalibr.py /kalibr/camera_calibration_raw -output /kalibr/converted/
$DOCKER_KALIBR_RUN kalibr_bagcreater --folder /kalibr/converted --output-bag /kalibr/data.bag
set +e
if [ -d "$tmp_dir/converted/cam1" ]; then
  # stereo
  $DOCKER_KALIBR_RUN bash -c "cd /kalibr && kalibr_calibrate_cameras --bag data.bag \
      --topics /cam0/image_raw /cam1/image_raw \
      --models $CAM_MODEL $CAM_MODEL \
      --target april_6x6_80x80cm.yaml \
      --dont-show-report"
else
  # mono... with a really ugly hack for patching kalibr to support it
  $DOCKER_KALIBR_RUN bash -c "sed -i '201,205d' /kalibr_workspace/src/Kalibr/aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras \
    && cd /kalibr && kalibr_calibrate_cameras --bag data.bag \
      --topics /cam0/image_raw \
      --models $CAM_MODEL \
      --target april_6x6_80x80cm.yaml \
      --dont-show-report"
fi
$DOCKER_KALIBR_RUN bash -c "cd kalibr && kalibr_calibrate_imu_camera --bag data.bag \
  --cams camchain-data.yaml \
  --target april_6x6_80x80cm.yaml \
  --imu allan/imu.yaml  \
  --dont-show-report"
set -e
$DOCKER_OURS_RUN python3 /scripts/kalibr-to-calibration.py /kalibr/camchain-imucam-data.yaml -output /kalibr/camera_calibration_raw/

echo ""
echo "Calibration completed!"
echo "Calibration file: ./tmp/camera_calibration_raw/calibration.json"
