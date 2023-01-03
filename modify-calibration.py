"""
Modify Spectacular AI and Basalt calibration JSON files
"""
import numpy as np
import json
import sys

def to_quat_and_pos(invM):
    from scipy.spatial.transform import Rotation
    M = np.linalg.inv(invM)
    q = Rotation.from_matrix(M[0:3, 0:3]).as_quat()
    return { "px": M[0, 3], "py": M[1, 3], "pz": M[2, 3], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3] }

def to_rmat(qp):
    from scipy.spatial.transform import Rotation
    q = [qp["qx"], qp["qy"], qp["qz"], qp["qw"]]
    p = [qp["px"], qp["py"], qp["pz"]]
    M = np.identity(4)
    M[0:3, 0:3] = Rotation.from_quat(q).as_matrix()
    M[0:3, 3] = p
    return np.linalg.inv(M)

def set_imu_to_camera_matrix_basalt(calibration, imu_to_cam):
    out = json.loads(json.dumps(calibration)) # deep copy
    out["value0"]["T_imu_cam"][0] = to_quat_and_pos(np.array(imu_to_cam))

    assert(len(out["value0"]["T_imu_cam"]) == 2)
    itoc1, itoc2 = [
        np.array(to_rmat(c))
        for c in calibration["value0"]["T_imu_cam"]
    ]

    first_to_second = np.dot(itoc2, np.linalg.inv(itoc1))
    new_imu_to_cam_second = np.dot(first_to_second, np.array(imu_to_cam))
    out["value0"]["T_imu_cam"][1] = to_quat_and_pos(new_imu_to_cam_second)
    return out

def set_imu_to_camera_matrix_ours(calibration, imu_to_cam):
    out = json.loads(json.dumps(calibration)) # deep copy
    out['cameras'][0]['imuToCamera'] = imu_to_cam

    if len(out['cameras']) > 1:
        itoc1, itoc2 = [
            np.array(c['imuToCamera'])
            for c in calibration['cameras']
        ]

        # imu -> second * first -> imu
        first_to_second = np.dot(itoc2, np.linalg.inv(itoc1))
        new_imu_to_cam_second = np.dot(first_to_second, np.array(imu_to_cam))

        out['cameras'][1]['imuToCamera'] = new_imu_to_cam_second.tolist()

    return out

def set_imu_to_camera_matrix(args):
    inputCalibration = json.load(sys.stdin)
    imuToCamera = json.loads(args.imu_to_camera_matrix)
    if args.basalt:
        outputCalibration = set_imu_to_camera_matrix_basalt(inputCalibration, imuToCamera)
    else:
        outputCalibration = set_imu_to_camera_matrix_ours(inputCalibration, imuToCamera)
    print(json.dumps(outputCalibration, indent=4))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    subs = p.add_subparsers()

    set_imu_to_cam = subs.add_parser('set_imu_to_camera',
        help='set IMU to camera matrix, keeping stereo camera calibration intact')
    set_imu_to_cam.add_argument('imu_to_camera_matrix',
        help='for example: [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]')
    set_imu_to_cam.add_argument('--basalt', action="store_true")

    args = p.parse_args()
    if hasattr(args, 'imu_to_camera_matrix'):
        set_imu_to_camera_matrix(args)
