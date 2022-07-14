#!/usr/bin/python

import argparse
import os
import yaml
import json

parser = argparse.ArgumentParser()
parser.add_argument("yamlFile", help="Folder containing JSONL and video file")
parser.add_argument("-output", help="Output folder, if not current directory")
parser.add_argument("--imu_to_camera_matrix", default=None,
	help='Optional IMU-to-camera matrix (for cam0), for example: [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]')
args = parser.parse_args()

def parseCamera(results, imuToCam0=None):
    coeffs = results["distortion_coeffs"]
    if results['distortion_model'] == 'equidistant':
        coeffs = coeffs[:4]
        distortionModel = "kannala-brandt4"
    else:
        # TODO: Ignores 4th coefficient
        coeffs = coeffs[:3]
        # coeffs = coeffs[:4]
        distortionModel = "pinhole"

    if 'T_cam_imu' in results:
        imuToCam = results['T_cam_imu']
    else:
        assert(imuToCam0 is not None)
        if 'T_cn_cnm1' in results:
            import numpy as np
            T = results['T_cn_cnm1']
            imuToCam = np.dot(T, imuToCam0).tolist()
        else:
            imuToCam = imuToCam0

    return {
        "imuToCamera": imuToCam,
        "imageWidth": results["resolution"][0],
        "imageHeight": results["resolution"][1],
        "distortionCoefficients": coeffs,
        "focalLengthX": results["intrinsics"][0],
        "focalLengthY": results["intrinsics"][1],
        "model": distortionModel,
        "principalPointX": results["intrinsics"][2],
        "principalPointY": results["intrinsics"][3]
    }


def main(args):
    outputFolder = args.output if args.output else "."
    os.makedirs(outputFolder, exist_ok=True)

    outputDict = {
        "cameras": []
    }

    with open(args.yamlFile) as f:
        calibrationResults = yaml.load(f, Loader=yaml.FullLoader)
        imuToCam = args.imu_to_camera_matrix
        if imuToCam is not None:
            imuToCam = json.loads(imuToCam)
        outputDict["cameras"].append(parseCamera(calibrationResults["cam0"], imuToCam))
        if calibrationResults.get("cam1"):
            outputDict["cameras"].append(parseCamera(calibrationResults["cam1"], imuToCam))

    with open(outputFolder + "/calibration.json", "w") as f:
        outputString = json.dumps(outputDict, sort_keys=True, indent=4)
        print(outputString)
        f.write(outputString)


if __name__ == "__main__":
    main(args)
