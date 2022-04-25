# Calibration scripts

_For stereo-camera-IMU calibration using Kalibr_

Usage: see `./calibrate.sh`.
Outputs are written to the `tmp` subfolder (`parameters.txt` in `camera_calibration_raw`)

## Dependencies

 * Python (virtualenv recommended): `pip install PyYAML allantools matplotlib`
 * Docker `sudo apt install docker.io` (for the Kalibr image)

## Calibration target

From [Kalibr](https://github.com/ethz-asl/kalibr/wiki/downloads), direct link: https://github.com/ethz-asl/kalibr/files/8514447/april_6x6_80x80cm_A0.pdf
