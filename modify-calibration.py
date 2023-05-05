"""
Modify Spectacular AI calibration JSON files

Input JSON in read from stdin an output written to stdout
"""
import numpy as np
import json
import sys

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
    p.add_argument('action', choices=['set_imu_to_camera', 'set_first_to_second'],
                   help='set IMU to camera matrix or stereo extrinsic matrix, keeping the other intact')

    p.add_argument('matrix',
        help='for example: [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]')

    p.add_argument('--second', action='store_true',
        help='the IMU-to-cam matrix (set or kept intact) concerns the second camera')
    
    args = p.parse_args()

    calib_in = json.load(sys.stdin)
    matrix = json.loads(args.matrix)

    if args.action == 'set_imu_to_camera':
        calib_out = set_imu_to_camera_matrix(calib_in, matrix, args.second)

    elif args.action == 'set_first_to_second':
        calib_out = set_first_to_second_matrix(calib_in, matrix, args.second)
        
    print(json.dumps(calib_out, indent=2))
