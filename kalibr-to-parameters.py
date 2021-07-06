#!/usr/bin/python

import argparse
import os
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("yamlFile", help="Folder containing JSONL and video file")
parser.add_argument("-output", help="Output folder, if not current directory")
args = parser.parse_args()


def arrayToString(arr):
    return ",".join([str(x) for x in arr])


def parseResults(results, names):
    imu = results["T_cam_imu"]
    columnMajor = []
    for i in range(4):
        for j in range(4):
            columnMajor.append(imu[j][i])
    output = []
    output.append("{} {}".format(names["imu"], arrayToString(columnMajor)))

    intrinsics = results["intrinsics"]
    output.append("{} {}".format(names["focalX"], intrinsics[0]))
    output.append("{} {}".format(names["focalY"], intrinsics[1]))
    output.append("{} {}".format(names["pX"], intrinsics[2]))
    output.append("{} {}".format(names["pY"], intrinsics[3]))

    coeffs = results["distortion_coeffs"]
    if results['distortion_model'] == 'equidistant':
        coeffs = coeffs[:4]
    else:
        # TODO: Ignores 4th coefficient
        coeffs = coeffs[:3]
    output.append("{} {}".format(names["coeffs"], arrayToString(coeffs)))

    return output


def main(args):
    outputFolder = args.output if args.output else "."
    os.makedirs(outputFolder, exist_ok=True)

    fields = []
    with open(args.yamlFile) as f:
        calibrationResults = yaml.load(f, Loader=yaml.FullLoader)

        fields.extend(parseResults(calibrationResults["cam0"], {
            "imu": "imuToCameraMatrix",
            "focalX": "focalLengthX",
            "focalY": "focalLengthY",
            "pX": "principalPointX",
            "pY": "principalPointY",
            "coeffs": "distortionCoeffs"
        }))
        if calibrationResults.get("cam1"):
            fields.extend(parseResults(calibrationResults["cam1"], {
                "imu": "secondImuToCameraMatrix",
                "focalX": "secondFocalLengthX",
                "focalY": "secondFocalLengthY",
                "pX": "secondPrincipalPointX",
                "pY": "secondPrincipalPointY",
                "coeffs": "secondDistortionCoeffs"
            }))

    cliParams = ""
    with open(outputFolder + "/parameters.txt", "w") as f:
        for line in fields:
            cliParams = cliParams + "-" + line.replace(" ", "=") + " "
            f.write(line + ";\n")
    print(cliParams)

    print("Done!")


if __name__ == "__main__":
    main(args)
