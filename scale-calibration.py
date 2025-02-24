#!/usr/bin/env python3

"""Scale Spectacular AI calibration.json file to a different resolution."""

import json
import pathlib

def readJson(filePath):
    with open(filePath) as f:
        return json.load(f)

def main(args):
    calibration = readJson(args.inputPath)
    assert("cameras" in calibration)
    newResolutions = args.newResolution.split(":")
    assert(len(newResolutions) == len(calibration["cameras"]))

    for i in range(len(calibration["cameras"])):
        camera = calibration["cameras"][i]
        if "imageWidth" not in camera or "imageHeight" not in camera:
            raise Exception(f"Calibration camera #{i} does not contain fields `imageWidth` and `imageHeight`.")
        xy = newResolutions[i].split("x")
        assert(camera["imageWidth"] > 0 and camera["imageHeight"] > 0)
        width = int(xy[0])
        height = int(xy[1])
        scaleX = width / camera["imageWidth"]
        scaleY = height / camera["imageHeight"]
        assert(scaleX > 0 and scaleY > 0)
        EPS = 1e-3
        if abs(scaleX - scaleY) > EPS:
            raise Exception("New resolution aspect ratio is not the same as the old one: {:.5f} -> {:.5f}".format(
                camera["imageWidth"] / camera["imageHeight"], float(xy[0]) / float (xy[1])))
        print(f"Scaling camera #{i} by factor of {scaleX}.")
        camera["imageWidth"] = width
        camera["imageHeight"] = height
        camera["focalLengthX"] *= scaleX
        camera["focalLengthY"] *= scaleX
        camera["principalPointX"] *= scaleX
        camera["principalPointY"] *= scaleX

    print("---")
    if args.outputPath:
        with open(args.outputPath, "w") as f:
            f.write(json.dumps(calibration, indent=4))
        print("Wrote calibration to `{}`.".format(args.outputPath))
    else:
        print(json.dumps(calibration, indent=4))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("inputPath", type=pathlib.Path, help="Path to input calibration.json.")
    p.add_argument("newResolution", help="New resolution, for example `1920x1080`. Split multiple cameras with `:`.")
    p.add_argument("--outputPath", type=pathlib.Path, help="Write output calibration.json to this file, otherwise print.")
    args = p.parse_args()
    main(args)
