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
parser.add_argument("-max", help="Plot max N seconds from the start", type=float)
parser.add_argument('--plot_acc_x_diff', action='store_true')

def addSubplot(plot, x, ys, title, style=None, plottype='plot', ymin=None, ymax=None, **kwargs):
    if len(np.array(ys).shape) < 2:
        ys = [ys]
    plot.title.set_text(title)
    p = getattr(plot, plottype)
    if ymin is not None and ymax is not None:
        plot.axis(ymin=ymin, ymax=ymax)
    for y in ys:
        if style is not None:
            p(x, y, style, **kwargs)
        else:
            p(x, y, **kwargs)


def plotDataset(folder, args):
    jsonlFile = folder if folder.endswith(".jsonl") else folder + "/data.jsonl"

    title = os.path.basename(os.path.normpath(folder))

    empty3d = lambda: {"x": [], "y": [], "z": [], "t": [], "td": []}
    acc = empty3d()
    gyro = empty3d()
    mag = empty3d()
    altitude = {"v": [], "t": [] }
    cpu = {"v": [], "t": []}
    cameras = {}

    sensors3d = { 'accelerometer': acc, 'gyroscope': gyro, 'magnetometer': mag}

    startTime = None
    timeOffset = 0
    minTime = None
    maxTime = None

    with open(jsonlFile) as f:
        nSkipped = 0
        for line in f.readlines():
            try:
                measurement = json.loads(line)
            except:
                sys.stderr.write('ignoring non JSON line: %s' % line)
                continue
            sensor = measurement.get("sensor")
            frames = measurement.get("frames")
            metrics = measurement.get("systemMetrics")
            if frames is None and 'frame' in measurement:
                frames = [measurement['frame']]
                frames[0]['cameraInd'] = 0
            if sensor is None and frames is None and metrics is None: continue

            if "time" in measurement:
                if startTime is None:
                    startTime = measurement["time"]
                    if args.zero:
                        timeOffset = startTime

                t = measurement["time"]
                if (args.skip is not None and t - startTime < args.skip) or (args.max is not None and t - startTime > args.max):
                    nSkipped += 1
                    continue

                t_corr = measurement["time"] - timeOffset
                if minTime == None or minTime > t_corr: minTime = t_corr
                if maxTime == None or maxTime < t_corr: maxTime = t_corr

            if sensor is not None:
                measurementType = sensor["type"]

                if measurementType in sensors3d:
                    data = sensors3d[measurementType]
                    for i, c in enumerate('xyz'): data[c].append(sensor["values"][i])
                    if len(data["t"]) == 0:
                        diff = 0
                    else:
                        diff = t_corr - data["t"][-1]
                    data["td"].append(diff * 1000.)
                    data["t"].append(t_corr)
                elif measurementType == "altitude":
                    altitude["v"].append(sensor["values"][0])
                    altitude["t"].append(t_corr)
            elif frames is not None:
                for f in frames:
                    if f.get("missingBitmap", False): continue
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
                    cameras[ind]["t"].append(t_corr)
            elif metrics is not None and 'cpu' in metrics:
                cpu["t"].append(t_corr)
                cpu["v"].append(metrics['cpu'].get('systemTotalUsagePercent', 0))

        if nSkipped > 0:
            print('skipped %d lines' % nSkipped)

    plots = [
        lambda s: addSubplot(s, acc["t"], [acc[c] for c in 'xyz'], "acc (m/s^2)"),
        lambda s: addSubplot(s, acc["t"], acc["td"], "acc time diff (ms)", "."),
        lambda s: addSubplot(s, gyro["t"], [gyro[c] for c in 'xyz'], "gyro (rad/s)"),
        lambda s: addSubplot(s, gyro["t"], gyro["td"], "gyro time diff (ms)", ".")
    ]

    if len(mag['t']) > 0:
        plots.extend([
            lambda s: addSubplot(s, mag["t"], [mag[c] for c in 'xyz'], "mag (muT)"),
            lambda s: addSubplot(s, mag["t"], mag["td"], "mag time diff (ms)", ".")
        ])

    if len(altitude['t']) > 0:
        plots.append(lambda s: addSubplot(s, altitude["t"], altitude["v"], "altitude (m)"))

    if args.plot_acc_x_diff:
        plots.append(lambda s: addSubplot(s, gyro["t"][1:], np.diff(gyro["x"]), "Subsequent acc x diff"))

    for ind in cameras.keys():
        camera = cameras[ind]
        order = np.argsort(camera['diff'])[::-1]
        plotkwargs = dict(
            plottype='scatter',
            color=np.array([(1, 0, 0) if c <= 0 else (0.6, 0.6, 1) for c in camera["diff"]])[order],
            s=6
        )
        t = np.array(camera["t"])[order]
        y = np.array(camera["diff"])[order]
        plots.append(
            lambda s, t=t, y=y, ind=ind, plotkwargs=plotkwargs: addSubplot(s, t, y, "frame time diff #{} (ms)".format(ind), **plotkwargs)
        )
        if camera.get("features"):
            y = np.array(camera["features"])[order]
            plots.append(
                lambda s, t=t, y=y, ind=ind, plotkwargs=plotkwargs: addSubplot(s, t, y, "features #{}".format(ind), **plotkwargs)
            )

        if len(camera["t"]) > 0:
            print("{:<25} {:.2f}FPS  (frame count {})".format(
              'camera ind %d rate' % ind,
              len(camera["t"]) / (camera["t"][-1] - camera["t"][0]),
              len(camera["t"])))

    if len(cpu["t"]) > 0:
        plots.append(lambda s: addSubplot(s, cpu["t"], cpu["v"], "system load (%)", ymin=0, ymax=100))

    for name, data in sensors3d.items():
      if len(data["t"]) > 0:
        print("{:<25} {:.2f}Hz  (sample count {})".format(
          name + ' frequency:',
          len(data["t"]) / (data["t"][-1] - data["t"][0]),
          len(data["t"])))

    fig, subplots = pyplot.subplots(len(plots), sharex=True)
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    for subplot, plot_func in zip(subplots, plots): plot_func(subplot)

    for subplot in subplots:
        subplot.set_xlim([minTime, maxTime])

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
