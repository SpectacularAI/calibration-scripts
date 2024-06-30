#!/usr/bin/env python3
"""
Visualize the data in a Spectacular AI data.jsonl file
"""

import argparse
from matplotlib import pyplot
import json
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", help="Folder containing data.jsonl file (or path to data.jsonl)", nargs='?', default=None)
parser.add_argument("-dir", help="Directory containing benchmarks you want to plot")
parser.add_argument("-zero", help="Rescale time to start from zero", action='store_true')
parser.add_argument("-skip", help="Skip N seconds from the start", type=float)


def addSubplot(plot, x, ys, title, style=None, plottype='plot', **kwargs):
    if len(np.array(ys).shape) < 2:
        ys = [ys]
    plot.title.set_text(title)
    p = getattr(plot, plottype)
    for y in ys:
        if style is not None:
            p(x, y, style, **kwargs)
        else:
            p(x, y, **kwargs)


def plotDataset(folder, args):
    jsonlFile = folder if folder.endswith(".jsonl") else folder + "/data.jsonl"

    title = os.path.basename(os.path.normpath(folder))

    accelerometer = {"x": [], "y": [], "z": [], "t": [], "td": []}
    gyroscope = {"x": [], "y": [], "z": [], "t": [], "td": []}
    cameras = {}

    startTime = sys.maxsize
    with open(jsonlFile) as f:
        for line in f.readlines():
            measurement = json.loads(line)
            if measurement.get("time") is not None:
                if startTime > measurement["time"]:
                    startTime = measurement["time"]

    timeOffset = 0
    if args.zero:
        timeOffset = startTime

    minTime = None
    maxTime = None

    with open(jsonlFile) as f:
        for line in f.readlines():
            measurement = json.loads(line)
            if args.skip != None and measurement.get("time") != None and measurement.get("time") - startTime < args.skip:
                print("Skiping")
                continue

            if "time" in measurement:
                if minTime == None or minTime > measurement.get("time"): minTime = measurement.get("time")
                if maxTime == None or maxTime < measurement.get("time"): maxTime = measurement.get("time")

            if measurement.get("sensor") is not None:
                measurementType = measurement["sensor"]["type"]
                if measurementType == "accelerometer":
                    accelerometer["x"].append(measurement["sensor"]["values"][0])
                    accelerometer["y"].append(measurement["sensor"]["values"][1])
                    accelerometer["z"].append(measurement["sensor"]["values"][2])
                    if len(accelerometer["t"]) == 0:
                        diff = 0
                    else:
                        diff = measurement["time"] - timeOffset - accelerometer["t"][-1]
                    accelerometer["td"].append(diff * 1000.)
                    accelerometer["t"].append(measurement["time"] - timeOffset)
                if measurementType == "gyroscope":
                    gyroscope["x"].append(measurement["sensor"]["values"][0])
                    gyroscope["y"].append(measurement["sensor"]["values"][1])
                    gyroscope["z"].append(measurement["sensor"]["values"][2])
                    if len(gyroscope["t"]) == 0:
                        diff = 0
                    else:
                        diff = measurement["time"] - timeOffset - gyroscope["t"][-1]
                    gyroscope["td"].append(diff * 1000.)
                    gyroscope["t"].append(measurement["time"] - timeOffset)
            if measurement.get("frames") is not None:
                frames = measurement.get("frames")
                for f in frames:
                    ind = f["cameraInd"]
                    if cameras.get(ind) is None:
                        cameras[ind] = {"diff": [0], "t": [] }
                        if "features" in f:
                            cameras[ind]["features"] = []
                    else:
                        diff = measurement["time"] - cameras[ind]["t"][-1]
                        # print("Time {}, Diff {}".format(measurement["time"], diff))
                        cameras[ind]["diff"].append(diff * 1000.)

                    if "features" in f:
                        cameras[ind]["features"].append(len(f["features"]))
                    cameras[ind]["t"].append(measurement["time"] - timeOffset)


    camPlots = 0
    for ind in cameras.keys():
        c = cameras[ind]
        camPlots += 1
        if "features" in c:
            camPlots += 1

    fig, subplots = pyplot.subplots(5 + camPlots)
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    for subplot in subplots:
        subplot.set_xlim([minTime, maxTime])


    addSubplot(subplots[0], accelerometer["t"], [accelerometer[c] for c in 'xyz'], "acc (m/s)")
    addSubplot(subplots[1], accelerometer["t"], accelerometer["td"], "acc time diff (ms)", ".")
    addSubplot(subplots[2], accelerometer["t"][1:], np.diff(accelerometer["x"]), "Subsequent acc x diff")

    addSubplot(subplots[3], gyroscope["t"], [gyroscope[c] for c in 'xyz'], "gyro (m/s)")
    addSubplot(subplots[4], gyroscope["t"], gyroscope["td"], "gyro time diff (ms)", ".")

    i = 0
    for ind in cameras.keys():
        camera = cameras[ind]
        order = np.argsort(camera['diff'])[::-1]
        plotkwargs=dict(
            plottype='scatter',
            color=np.array([(1, 0, 0) if c <= 0 else (0.6, 0.6, 1) for c in camera["diff"]])[order],
            s=6
        )
        t = np.array(camera["t"])[order]
        y = np.array(camera["diff"])[order]
        addSubplot(subplots[5 + i], t, y, "frame time #{} (ms)".format(ind), **plotkwargs)
        i += 1
        if camera.get("features"):
            y = np.array(camera["features"])[order]
            addSubplot(subplots[5 + i], t, y, "features #{}".format(ind), **plotkwargs)
            i += 1

        if len(camera["t"]) > 0:
            print("camera ind {} rate:         {:.2f}FPS  (frame count {})".format(
              ind,
              len(camera["t"]) / (camera["t"][-1] - camera["t"][0]),
              len(camera["t"])))

    if len(accelerometer["t"]) > 0:
        print("accelerometer frequency:   {:.2f}Hz  (sample count {})".format(
          len(accelerometer["t"]) / (accelerometer["t"][-1] - accelerometer["t"][0]),
          len(accelerometer["t"])))
    if len(gyroscope["t"]) > 0:
        print("gyroscope frequency:       {:.2f}Hz  (sample count {})".format(
          len(gyroscope["t"]) / (gyroscope["t"][-1] - gyroscope["t"][0]),
          len(gyroscope["t"])))

    pyplot.show()


def main(args):
    if args.case:
        plotDataset(args.case, args)
    elif args.dir:
        groups = {}
        for x in os.walk(args.dir):
            for file in x[2]:
                if file == "data.jsonl":
                    plotDataset(x[0], args)
    else:
        print("Invalid arguments")
        exit(1)


if __name__ == "__main__":
    main(parser.parse_args())
