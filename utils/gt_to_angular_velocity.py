import json
import numpy as np

def ori_to_quat(ori):
    return np.array([ori["w"], ori["x"], ori["y"], ori["z"]])

def conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]

def mult(q1, q2):
    w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    x = q1[1] * q2[0] + q1[0] * q2[1] - q1[3] * q2[2] + q1[2] * q2[3]
    y = q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2] - q1[1] * q2[3]
    z = q1[3] * q2[0] - q1[2] * q2[1] + q1[1] * q2[2] + q1[0] * q2[3]
    return [w, x, y, z]

def dot(q1, q2):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

def angular_velocity(q1, q2, dt):
    if dot(q1, q2) < -.9: q2 = -q2
    qd = mult(q1, conjugate(q2))
    qd /= np.linalg.norm(qd)
    angle = 2 * np.arccos(qd[0])
    div = np.sqrt(1 - qd[0] ** 2)
    if div < 0.0000001: return ([0,0,0], 0)
    axis = qd[1:] / div
    return (angle * axis / dt, angle / dt)

def compute_angular_velocities(gt, key):
    prev = gt[0]
    angular_velocities = []
    angular_speed = []
    euler_ori = []
    time = []
    orientationKey = "orientation" if "orientation" in prev[key] else "enuOrientation"
    for index, cur in enumerate(gt):
        if index == 0: continue
        dt = cur["time"] - prev["time"]
        if dt == 0.0: continue
        angular, speed = angular_velocity(
            ori_to_quat(prev[key][orientationKey]),
            ori_to_quat(cur[key][orientationKey]),
            dt
        )
        time.append((prev["time"] + cur["time"]) * .5)
        angular_velocities.append(angular)
        angular_speed.append(speed)
        euler = cur[key].get("euler")
        if euler: euler_ori.append([euler["x"], euler["y"], euler["z"]])
        prev = cur


    return (np.array(time).T, np.array(angular_velocities), np.array(euler_ori), np.array(angular_speed))


def simulate_angular_velocity(input, gtName=None):
    data = []
    validGtNames = ["groundTruth", "gps"]
    with open(input) as f:
        for line in f.readlines():
            if gtName == None:
                for n in validGtNames:
                    if n in line:
                        gtName = n
                        break
            if not gtName: continue
            if not gtName in line: continue
            data.append(json.loads(line))
    gt_time, gt_angular, gt_euler, gt_angular_speed = compute_angular_velocities(data, gtName)

    return np.hstack((gt_time[:, np.newaxis], gt_angular))
