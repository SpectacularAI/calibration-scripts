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

    # see https://github.com/ethz-asl/kalibr/wiki/supported-models
    # and https://spectacularai.github.io/docs/pdf/calibration_manual.pdf

    coeffs = results["distortion_coeffs"]
    model = results['distortion_model']
    if model == 'equidistant':
        assert(len(coeffs) == 4)
        our_model = 'kannala-brandt4'
    elif mode == 'radtan':
        k1, k2, p1, p2 = coeffs
        coeffs = [k1, k2, p1, p2, 0, 0, 0, 0]
        our_model = 'brown-conrady'
    else:
        raise RuntimeError('unsupported model %s' % model)
    output.append("{} {}".format(names["coeffs"], arrayToString(coeffs)))
    output.append("{} {}".format('model', our_model)
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
