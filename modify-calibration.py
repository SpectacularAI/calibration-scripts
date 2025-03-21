"""
Modify Spectacular AI calibration JSON files
"""

import json
import pathlib
import sys

import numpy as np

def set_imu_to_camera_matrix(calibration, imu_to_cam, second_camera=False):
    out = json.loads(json.dumps(calibration)) # deep copy
    out['cameras'][0]['imuToCamera'] = imu_to_cam

    if len(out['cameras']) > 1:
        itoc1, itoc2 = [
            np.array(c['imuToCamera'])
            for c in calibration['cameras']
        ]

        # imu -> second * first -> imu
        first_to_second = np.dot(itoc2, np.linalg.inv(itoc1))

        if second_camera:
            second_to_first = np.linalg.inv(first_to_second)
            new_imu_to_cam_first = np.dot(second_to_first, np.array(imu_to_cam))
            out['cameras'][0]['imuToCamera'] = new_imu_to_cam_first.tolist()
            out['cameras'][1]['imuToCamera'] = imu_to_cam
        else:
            new_imu_to_cam_second = np.dot(first_to_second, np.array(imu_to_cam))
            out['cameras'][1]['imuToCamera'] = new_imu_to_cam_second.tolist()

    return out

def set_first_to_second_matrix(calibration, first_to_second, second_camera=False):
    out = json.loads(json.dumps(calibration)) # deep copy
    assert(len(out['cameras']) == 2)

    itoc1, itoc2 = [
        np.array(c['imuToCamera'])
        for c in calibration['cameras']
    ]

    if second_camera:
        second_to_first = np.linalg.inv(first_to_second)
        itoc1 = np.dot(second_to_first, itoc2)
    else:
        second_to_first = np.linalg.inv(first_to_second)
        itoc2 = np.dot(first_to_second, itoc1)

    out['cameras'][0]['imuToCamera'] = itoc1.tolist()
    out['cameras'][1]['imuToCamera'] = itoc2.tolist()
    return out

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)

    p.add_argument('action',
        choices=['set_imu_to_camera', 'set_first_to_second', 'set_imu_to_camera_translation'],
        help='set IMU to camera matrix or stereo extrinsic matrix, keeping the other intact',
    )
    p.add_argument('matrix',
        help='Data used for modification. For example: "[[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]", or "path/to/calibration.json"',
    )
    p.add_argument('--matrixIndexInCalibration',
        type=int,
        default=0,
        help='If the `matrix` argument is a path, use its imuToCamera field from this camera index as input',
    )
    p.add_argument('--second',
        action='store_true',
        help='the input_calibration IMU-to-cam matrix (set or kept intact) concerns the second camera',
    )
    p.add_argument('--input_calibration',
        type=pathlib.Path,
        help='Calibration to be modified. If not given will be read from stdin.',
    )
    p.add_argument('--output_calibration',
        type=pathlib.Path,
        help='Output calibration. If not given will be written to stdout.',
    )

    args = p.parse_args()

    try:
        with open(args.matrix) as f:
            calibration = json.load(f)
        if args.action == 'set_imu_to_camera' or args.action == 'set_imu_to_camera_translation':
            matrix = calibration["cameras"][args.matrixIndexInCalibration]["imuToCamera"]
        else:
            assert(args.action == 'set_first_to_second')
            imuToCam0 = np.array(calibration["cameras"][0]["imuToCamera"])
            imuToCam1 = np.array(calibration["cameras"][1]["imuToCamera"])
            matrix = imuToCam1 @ np.linalg.inv(imuToCam0)
    except:
        matrix = json.loads(args.matrix)

    if args.input_calibration:
        with open(args.input_calibration) as f:
            calib_in = json.load(f)
    else:
        calib_in = json.load(sys.stdin)

    if args.action == 'set_imu_to_camera':
        calib_out = set_imu_to_camera_matrix(calib_in, matrix, args.second)
    elif args.action == 'set_imu_to_camera_translation':
        ind = 1 if args.second else 0
        imuToCamera = np.array(calib_in["cameras"][ind]["imuToCamera"])
        # Modify IMU position in given camera coordinates, keeping the rotation.
        imuToCamera[:3, 3] = np.array(matrix)[:3, 3]
        calib_out = set_imu_to_camera_matrix(calib_in, imuToCamera.tolist(), args.second)
    else:
        assert(args.action == 'set_first_to_second')
        calib_out = set_first_to_second_matrix(calib_in, matrix, args.second)

    calibration_string = json.dumps(calib_out, indent=4)
    if args.output_calibration:
        with open(args.output_calibration, "w") as f:
            f.write(calibration_string)
    else:
        print(calibration_string)
