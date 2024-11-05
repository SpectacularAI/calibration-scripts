#!/usr/bin/env python
#
# Compute rotation between two IMUs from time-synchronized signals.
# Assumes the biases are zero, i.e. small enough.

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

def interpNd(x, xp, fp):
    return np.hstack([np.interp(x, xp, fp[:, i])[:, np.newaxis] for i in range(fp.shape[1])])

def flatten(l):
    return [item for sublist in l for item in sublist]

def readImu(filePath, t0, t1, kind):
    assert(kind in ["accelerometer", "gyroscope"])
    v = []
    firstT = None
    with open(filePath) as f:
        for line in f:
            obj = json.loads(line)
            if "sensor" not in obj: continue
            if obj["sensor"]["type"] != kind: continue
            if firstT is None: firstT = obj["time"]
            t = obj["time"] - firstT
            if t0 is not None and t < t0: continue
            if t1 is not None and t > t1: break
            v.append(flatten([[obj["time"]], obj["sensor"]["values"]]))
    if len(v) <= 0:
        raise Exception(f"No IMU samples found in {filePath}.")
    return np.array(v)

def main(args):
    v0 = readImu(args.dataImu0, args.t0, args.t1, args.kind)
    v1 = readImu(args.dataImu1, args.t0, args.t1, args.kind)
    t0 = np.maximum(v0[0, 0], v1[0, 0])
    t1 = np.minimum(v0[-1, 0], v1[-1, 0])
    tGrid = np.arange(t0, t1, args.step)
    g0 = interpNd(tGrid, v0[:, 0], v0[:, 1:])
    g1 = interpNd(tGrid, v1[:, 0], v1[:, 1:])

    colors = "rgb"
    for i in range(3):
        plt.plot(tGrid, g0[:, i], label=f"imu0-{i}", color=colors[i], linestyle="--")
        plt.plot(tGrid, g1[:, i], label=f"imu1-{i}", color=colors[i])
    plt.xlabel("t [s]")
    plt.title("Before alignment")
    plt.show()

    B = g0.transpose() @ g1
    U, S, Vt = np.linalg.svd(B)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0.0:
        flip = np.diag([1, 1, -1])
        R = np.dot(U, np.dot(flip, Vt))

    imu0toImu1 = np.eye(4)
    imu0toImu1[:3, :3] = R
    print(imu0toImu1.tolist())
    aligned0 = np.dot(R, g0.transpose()).transpose()

    for i in range(3):
        plt.plot(tGrid, aligned0[:, i], label=f"imu0-{i}", color=colors[i], linestyle="--")
        plt.plot(tGrid, g1[:, i], label=f"imu1-{i}", color=colors[i])
    plt.xlabel("t [s]")
    plt.title("After alignment")
    plt.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("dataImu0", help="data.jsonl file for 'imu0'")
    p.add_argument("dataImu1", help="data.jsonl file for 'imu1'")
    p.add_argument("--kind", default="accelerometer", help="'accelerometer' or 'gyroscope'")
    p.add_argument("--t0", type=float, help="Skip data before this many seconds from the beginning.")
    p.add_argument("--t1", type=float, help="Skip data after this many seconds from the beginning.")
    p.add_argument("--step", type=float, default=0.1, help="Sample IMU at this interval (seconds)")
    args = p.parse_args()
    main(args)
