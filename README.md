# Miscellaneous Spectacular AI data scripts

Intended for use with Spectacular AI SDK or, in general, with data stored in the Spectacular AI data format (see [here](https://spectacularai.github.io/docs/sdk/recording.html) for more info).

### Instructions: Kalibr-based calibration scripts

**_NOTE: these calibration tools are no-longer maintained, please use the [Spectacular AI Calibration Tool](https://spectacularai.github.io/docs/sdk/tools/calibration.html) instead_**

#### Calibration using Docker

    ./docker_calibrate.sh /path/to/vio_recording SIZE_OF_APRILTAG_IN_METERS

In the above, `/path/to/vio_recording` should be a folder that contains data in the format recorded by Spectacular AI SDK (for example [this program](https://github.com/SpectacularAI/sdk-examples/blob/main/python/oak/vio_record.py)). The second parameter, the AprilTag size, is easiest to calculate by measuring the size of the AprilGrid as instructed [here](https://github.com/SpectacularAI/oak-d-capture/blob/master/measuring_calibration_target.jpg) and **dividing the result by 8.1**.

#### Non-dockerized setup

Usage: see `./calibrate.sh`.
Outputs are written to the `tmp` subfolder (`parameters.txt` in `camera_calibration_raw`)

**Dependencies**

 * Python (virtualenv recommended): `pip install PyYAML allantools matplotlib`
 * Docker `sudo apt install docker.io` (for the Kalibr image)

#### Calibration target

From [Kalibr](https://github.com/ethz-asl/kalibr/wiki/downloads), direct link: https://github.com/ethz-asl/kalibr/files/8514447/april_6x6_80x80cm_A0.pdf
