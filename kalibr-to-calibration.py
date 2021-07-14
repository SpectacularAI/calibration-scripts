#!/usr/bin/python

import argparse
import os
import yaml
import json

parser = argparse.ArgumentParser()
parser.add_argument("yamlFile", help="Folder containing JSONL and video file")
parser.add_argument("-output", help="Output folder, if not current directory")
args = parser.parse_args()


def parseCamera(results):
    coeffs = results["distortion_coeffs"]
    if results['distortion_model'] == 'equidistant':
        coeffs = coeffs[:4]
        distortionModel = "kannala-brandt4"
    else:
        # TODO: Ignores 4th coefficient
        coeffs = coeffs[:3]
        distortionModel = "pinhole"

    return {
        "imuToCamera": results["T_cam_imu"],
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
        outputDict["cameras"].append(parseCamera(calibrationResults["cam0"]))
        if calibrationResults.get("cam1"):
            outputDict["cameras"].append(parseCamera(calibrationResults["cam1"]))

    with open(outputFolder + "/calibration.json", "w") as f:
        outputString = json.dumps(outputDict, sort_keys=True, indent=4)
        print(outputString)
        f.write(outputString)

    print("Done!")


if __name__ == "__main__":
    main(args)
