"""
Modify Spectacular AI calibration JSON files
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

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    subs = p.add_subparsers()

    set_imu_to_cam = subs.add_parser('set_imu_to_camera',
        help='set IMU to camera matrix, keeping stereo camera calibration intact')
    set_imu_to_cam.add_argument('imu_to_camera_matrix',
        help='for example: [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]')
    set_imu_to_cam.add_argument('--second', action='store_true',
        help='the given IMU-to-cam matrix is for the second camera')

    args = p.parse_args()
    if hasattr(args, 'imu_to_camera_matrix'):
        print(json.dumps(set_imu_to_camera_matrix(
            json.load(sys.stdin),
            json.loads(args.imu_to_camera_matrix),
            args.second), indent=2))
