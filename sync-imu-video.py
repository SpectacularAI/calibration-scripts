"""
Plot video and gyroscope speed to help synchronizing them
"""
import cv2 as cv
import numpy as np
import json

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
            # print('frame %d' % frame_no)
            # if self.frame_no <= args.skip_first_n_frames: continue

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

def normalize_array(array):
    np_array = np.array(array)
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    normalized_array = (np_array - min_value) / (max_value - min_value)
    return normalized_array

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('video')
    p.add_argument('data')
    p.add_argument('--flow_winsize', type=int, default=15)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--frameTimeOffsetSeconds', type=float, default=-0.1)
    p.add_argument('--resize_width', type=int, default=200)

    args = p.parse_args()

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

    for i in range(len(frameTimes)):
        avg_speed = leader_flow.next_avg_speed_flow()
        frameSpeed.append(avg_speed)
        leader_flow.show_preview('leader')


    gyroSpeed = normalize_array(gyroSpeed)
    frameSpeed = normalize_array(frameSpeed)

    import matplotlib.pyplot as plt

    _, subplots = plt.subplots(2)


    subplots[0].plot(gyroTimes, gyroSpeed, label='Gyro speed')
    subplots[0].plot(frameTimes, frameSpeed, label='Optical flow speed')
    subplots[0].title.set_text("Normalized gyro speed vs. optical flow speed")

    frameTimes = np.array(frameTimes) + args.frameTimeOffsetSeconds

    subplots[1].plot(gyroTimes, gyroSpeed, label='Gyro speed')
    subplots[1].plot(frameTimes, frameSpeed, label=f'Optical flow speed')
    subplots[1].title.set_text(f"Normalized gyro speed vs. optical flow speed with {args.frameTimeOffsetSeconds} seconds added to frame timetamps")
    plt.legend()
    plt.show()
