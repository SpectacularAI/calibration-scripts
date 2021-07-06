# Calibration scripts

_For stereo-camera-IMU calibration using Kalibr_

Usage: see `./calibrate.sh`.
Outputs are written to the `tmp` subfolder (`parameters.txt` in `camera_calibration_raw`)

## Dependencies

 * Python (virtualenv recommended): `pip install PyYAML allantools matplotlib`
 * Docker `sudo apt install docker.io` (for the Kalibr image)

## Calibration target

From [Kalibr](https://github.com/ethz-asl/kalibr/wiki/downloads), direct G-Drive link: https://drive.google.com/file/d/0B0T1sizOvRsUdjFJem9mQXdiMTQ/edit?usp=sharing
