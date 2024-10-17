#!/usr/bin/env python3
"""
Computes timeshift needed to match gyroscope and video (optical flow) rotations.
Plots the data before and after the synchronization.
"""

import json
import pathlib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy import interpolate


class OpticalFlowComputer:
    def __init__(self, fn, args):
        self.cap = cv.VideoCapture(fn)
        self.args = args

        self.frame_no = 0
        self.src = self._next()
        self.hsv = None

    def next_avg_speed_flow(self):
        target = self._next()
        if target is None: return None
        self.flow = cv.calcOpticalFlowFarneback(self.src, target, None, 0.5, 3, self.args.flow_winsize, 3, 5, 1.2, 0)
        #flow_scalar = np.mean(np.mean(self.flow, axis=0), axis=0)
        vector_lengths = np.linalg.norm(self.flow, axis=1)
        average_length = np.mean(vector_lengths)

        self.src = target
        return average_length

    def _next(self):
        while True:
            self.frame_no += 1
            frame = self._grab_frame()

            if frame is None: return None
            if self.frame_no <= args.skip_first_n_frames: continue

            return self._convert_frame(frame)

    def _grab_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return frame

    def _convert_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.args.resize_width > 0:
            h, w = frame.shape
            w1 = self.args.resize_width
            h1 = h * w1 // w
            frame = cv.resize(frame, (w1, h1), interpolation=cv.INTER_AREA)
        return frame

    def show_preview(self, wnd_tag=''):
        if not self.args.preview: return

        cv.imshow('input ' + wnd_tag, self.src)

        if self.hsv is None:
            self.hsv = np.zeros((self.src.shape[0], self.src.shape[1], 3), dtype=self.src.dtype)
            self.hsv[..., 1] = 255

        mag, ang = cv.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)
        cv.imshow('optical flow ' + wnd_tag, bgr)
        cv.waitKey(1)

def minimize_error(flow1, flow2, max_offset):

    result_x = np.zeros(max_offset*2+1, dtype=int)
    result_y = np.zeros(max_offset*2+1, dtype=float)

    def shift(offset):
        if offset >= 0:
            shifted1 = flow1[offset:,:]
            shifted2 = flow2[:shifted1.shape[0],:]
        else:
            shifted2 = flow2[(-offset):,:]
            shifted1 = flow1[:shifted2.shape[0],:]
        return shifted1, shifted2

    for i in range(len(result_x)):
        offset = i - max_offset
        result_x[i] = offset

        shifted1, shifted2 = shift(offset)
        result_y[i] = np.mean(np.abs(shifted1 - shifted2))

        # plt.plot(np.mean(shifted1, axis=1))
        # plt.plot(np.mean(shifted2, axis=1))
        # plt.show()
        # print(result_y[i])

    offset = int(result_x[np.argmin(result_y)])
    print(offset)

    return offset, (result_x, result_y), shift(offset)


def slurpJson(dataJsonl):
    data = []
    with open(dataJsonl) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def meanError(times, values, fn, fn_range, offset):
    newTimes = times - offset
    startIdx = None
    endIdx = None
    for idx, t in enumerate(newTimes):
        if startIdx == None and t >= fn_range[0]:
            startIdx = idx
        if endIdx == None and t > fn_range[1]:
            endIdx = idx
            break
    croppedTime = newTimes[startIdx:endIdx]
    croppedValues = values[startIdx:endIdx]
    if croppedTime.size < 2: return math.nan
    try:
        fnValues = fn(croppedTime)
    except ValueError as e:
        print(e)
        return math.nan
    return np.mean(np.abs(croppedValues - fnValues))


def findMinimumError(gyro_time, gyro_angular_speed, fn, fn_range, offsets):
    errors = []
    for offset in offsets:
        errors.append(meanError(gyro_time, gyro_angular_speed, fn, fn_range, offset))
    errors = np.array(errors)
    smallestIdx = np.nanargmin(errors)
    return offsets, errors, offsets[smallestIdx], errors[smallestIdx]

def main(args):
    if not pathlib.Path(args.data).exists():
        raise Exception("Data file `{}` does not exist.".format(args.data))
    if not pathlib.Path(args.video).exists():
        raise Exception("Video file `{}` does not exist.".format(args.video))

    leader_flow = OpticalFlowComputer(args.video, args)

    gyroTimes = []
    gyroSpeed = []
    frameTimes = []
    frameSpeed = []

    data = slurpJson(args.data)
    prevFrameTime = None
    for entry in data:
        if "sensor" in entry and entry["sensor"]["type"] == "gyroscope":
            gyroTimes.append(entry["time"])
            gyroSpeed.append(np.linalg.norm(entry["sensor"]["values"]))
        elif "frames" in entry:
            if prevFrameTime != None:
                frameTimes.append((prevFrameTime + entry["time"]) / 2.0)
            prevFrameTime = entry["time"]

    initialOffset = 0.0
    if args.sameStart:
        initialOffset = gyroTimes[0] - frameTimes[0]
        print(f"Asuming same start time, offseting frames by {initialOffset}")
        for i in range(len(frameTimes)):
            frameTimes[i] += initialOffset
    if args.sameEnd:
        initialOffset = gyroTimes[-1] - frameTimes[-1]
        print(f"Asuming same end time, offseting frames by {initialOffset}")
        for i in range(len(frameTimes)):
            frameTimes[i] += initialOffset

    if args.max_frames > 0:
        frameTimes = frameTimes[args.skip_first_n_frames : args.skip_first_n_frames + args.max_frames]
        gyroSpeed = [gyroSpeed[i] for i, t in enumerate(gyroTimes) if t >= frameTimes[0] and t <= frameTimes[-1]]
        gyroTimes = [t for t in gyroTimes if t >= frameTimes[0] and t <= frameTimes[-1]]
    elif args.skip_first_n_frames > 0:
        frameTimes = frameTimes[args.skip_first_n_frames:]
        gyroSpeed = [gyroSpeed[i] for i, t in enumerate(gyroTimes) if t >= frameTimes[0] and t <= frameTimes[-1]]
        gyroTimes = [t for t in gyroTimes if t >= frameTimes[0] and t <= frameTimes[-1]]

    for i in range(len(frameTimes)):
        avg_speed = leader_flow.next_avg_speed_flow()
        if avg_speed == None:
            avg_speed = 0.0
            print(f"Failed to get speed for frame {i}")
        frameSpeed.append(avg_speed)
        leader_flow.show_preview('leader')

    if len(gyroSpeed) == 0:
        print("No gyroscope data to plot.")
        return
    if len(frameSpeed) == 0:
        print("No frame data to plot.")
        return

    gyroSpeed = np.array(gyroSpeed)
    frameSpeed = np.array(frameSpeed)
    gyroSpeed /= np.mean(gyroSpeed)
    frameSpeed /= np.mean(frameSpeed)

    testedOffsets = []
    estimatedErrors = []

    if args.frameTimeOffsetSeconds:
        timeOffset = args.frameTimeOffsetSeconds
    else:
        fn = interpolate.interp1d(frameTimes, frameSpeed, assume_sorted=True)
        fn_range = [frameTimes[0], frameTimes[-1]]

        offset = 0
        maxOffset = args.maxOffset
        ITERATIONS=3
        for i in range(ITERATIONS):
            step = maxOffset * 2 / 100
            offsets, errors, offset, error = findMinimumError(gyroTimes, gyroSpeed, fn, fn_range, np.arange(-maxOffset, maxOffset, step) + offset)
            maxOffset /= 20
            estimatedErrors.append(errors)
            testedOffsets.append(offsets)
        timeOffset = offset

    totalOffset = initialOffset + timeOffset
    print("Estimated time offset: {:.4f}s".format(totalOffset))

    if args.rawPlotOnly:
        plt.plot(gyroTimes, gyroSpeed, label='Gyroscope speed')
        plt.plot(frameTimes, frameSpeed, label='Optical flow speed')
        plt.title("Original data")
        plt.legend()
        plt.xlabel('t (s)')
        plt.ylabel('normalized speed')
        plt.show()
    elif not args.noPlot:
        _, subplots = plt.subplots(2 + len(estimatedErrors))
        subplots[0].plot(gyroTimes, gyroSpeed, label='Gyroscope speed')
        subplots[0].plot(frameTimes, frameSpeed, label='Optical flow speed')
        subplots[0].title.set_text("Original data")
        subplots[0].set_xlabel('t (s)')
        subplots[0].set_ylabel('normalized speed')
        subplots[0].legend()

        frameTimes = np.array(frameTimes) + timeOffset
        subplots[1].plot(gyroTimes, gyroSpeed, label='Gyroscope speed')
        subplots[1].plot(frameTimes, frameSpeed, label=f'Optical flow speed')
        subplots[1].title.set_text(f"After correction by {timeOffset:.4f}s added to frame timetamps")
        subplots[1].set_xlabel('t (s)')
        subplots[1].set_ylabel('normalized speed')
        subplots[1].legend()

        for i in range(len(estimatedErrors)):
            frameTimes = np.array(frameTimes) + timeOffset
            subplots[2+i].plot(testedOffsets[i], estimatedErrors[i], label='Error as function of time offset')
            subplots[2+i].legend()

        plt.legend()
        plt.show()

    if args.output:
        print(f"All frame timestamps corrected in {args.output}")
        for entry in data:
            if "frames" in entry:
                entry["time"] += totalOffset
        data_sorted = sorted(data, key=lambda x: x.get("time", 0.0))
        with open(args.output, "w") as f:
            for entry in data_sorted:
                f.write(json.dumps(entry) + "\n")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('video', help="path to video file")
    p.add_argument('data', help="path to data.jsonl file")
    p.add_argument('--flow_winsize', type=int, default=15)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('-skip', '--skip_first_n_frames', type=int, default=0, help='Skip first N frames')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--noPlot', action='store_true')
    p.add_argument('--rawPlotOnly', action='store_true')
    p.add_argument('--sameStart', action='store_true', help="Assume video and gyroscope start roughly at same time")
    p.add_argument('--sameEnd', action='store_true', help="Assume video and gyroscope end roughly at same time")
    p.add_argument('--frameTimeOffsetSeconds', type=float)
    p.add_argument('--resize_width', type=int, default=200)
    p.add_argument('--output', help="data.jsonl with frame timestamp shifted to match gyroscope timestamps")
    p.add_argument('--maxOffset', help="Maximum offset between gyroscope and frame times in seconds", type=float, default=5.0)

    args = p.parse_args()
    main(args)
