"""
Analyze Spectacular AI calibration JSON files
"""
import numpy as np
import json

def rad2deg(a):
    return a / np.pi * 180

def angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2))

def getReferenceImuToWorld(imuToFwdCamera, imuLeveled=True):
    """
    Guess a typical "reference pose" for the device.
    
    in the reference pose, either the IMU or camera coordinate system is
    rotated n*90 degrees w.r.t. the world coordinates. Furthermore, of the
    camera coordinate axes (x, y, and z), y is the one that most conicides
    with the direction of gravity and z points most towards the world y-axis.
    If we know which sensor is leveled (IMU or camera), these assumptions
    allows determining the IMU-to-world matrix in such a reference pose.
    """
    def snapTo90Deg(vec):
        ax = np.array([0, 0, 0])
        mainIdx = np.argmax(np.abs(vec))
        ax[mainIdx] = np.sign(vec[mainIdx])
        return ax


    imuToCamRot =  imuToFwdCamera[:3, :3]

    if imuLeveled:
        upAxis = -snapTo90Deg(imuToCamRot[1, :])
        fwdAxis = snapTo90Deg(imuToCamRot[2, :])
        leftAxis = np.cross(fwdAxis, upAxis)

        worldToImuRot = np.hstack([a[:, np.newaxis] for a in [leftAxis, fwdAxis, upAxis]])
        imuToWorldRot = np.transpose(worldToImuRot)
    else:
        camToWorldRot = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0,-1, 0]
        ])

        imuToWorldRot = np.dot(camToWorldRot, imuToCamRot)
    
    imuToWorld = np.eye(4)
    imuToWorld[:3, :3] = imuToWorldRot
    return imuToWorld

def getVergence(imuToCam1, imuToCam2):
    def getPrincipalAxisInImuCoords(imuToCam):
        camToImuRot = imuToCam[:3, :3].transpose()
        return camToImuRot[:, 2]

    return angle(
        getPrincipalAxisInImuCoords(imuToCam1),
        getPrincipalAxisInImuCoords(imuToCam2))

def getBaseline(imuToCam1, imuToCam2):
    cam1to2 = np.dot(imuToCam2, np.linalg.inv(imuToCam1))
    return np.linalg.norm(cam1to2[:3, 3])

def getPitch(imuToCam, imuToWorld):
    camToWorldRot = np.dot(imuToWorld[:3, :3], imuToCam[:3, :3].transpose())
    return np.arcsin(camToWorldRot[2, 2])

def analyze_calibration(calib, imuLeveled=True):
    cams = [np.array(c['imuToCamera']) for c in calib['cameras']]
    imuToWorld = getReferenceImuToWorld(cams[0], imuLeveled=imuLeveled)
    stats = {
        'imuToWorld': imuToWorld.tolist(),
        'baselineMillimeters': 1000 * getBaseline(cams[0], cams[1]), 
        'vergenceDegrees': rad2deg(getVergence(cams[0], cams[1]))
    }

    if len(cams) > 1:
        stats['baselineMillimeters'] = 1000 * getBaseline(cams[0], cams[1])
        stats['vergenceDegrees'] = rad2deg(getVergence(cams[0], cams[1]))

    if imuLeveled:
        stats['pitchPerCameraInDegrees'] = [rad2deg(getPitch(c, imuToWorld)) for c in cams]

    return stats

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('input_calibration_json_file', type=argparse.FileType())
    p.add_argument('--leveled_camera', action='store_true')

    args = p.parse_args()
    print(
        json.dumps(
            analyze_calibration(
                json.load(args.input_calibration_json_file),
                imuLeveled=not args.leveled_camera),
        indent=2))
