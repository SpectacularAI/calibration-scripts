"""
Finds time offset between two IMU signals using cross-correlation:
t_imu2 = t_imu1 + offset

Optionally, can also estimate time scale (linear, quadratic) using non-linear least squares, recommended with longer datasets:
(linear) t_imu2 = a * t_imu1 + b + t_imu1 = (1 + a) * t_imu1 + b
(quadratic) t_imu2 = a * t_imu1^2 + b * t_imu1 + c + t_imu1 = a * t_imu1^2 + (1 + b) * t_imu1 + c

Requirements: numpy, scipy and matplotlib
"""

import json
import argparse
from scipy import signal, optimize
import numpy as np
import matplotlib.pyplot as plt

def read_imu_data(filename, timestampRange, sensor):
    if timestampRange:
        start, end = timestampRange.split(":")
        start = float(start)
        end = float(end)

    imuData = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            if "sensor" not in data: continue
            if data["sensor"]["type"] == sensor:
                values = data["sensor"]["values"]
                time = data["time"]
                if timestampRange and (time < start or time > end): continue
                imuData.append([time, values[0], values[1], values[2]])

    return np.array(imuData)

def read_imu1_to_imu2(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        if "imu1ToImu2" in data:
            return np.array(data["imu1ToImu2"])
        raise KeyError("{} does not contain key 'imu1ToImu2'".format(filename))

def write_output_jsonl(inputFilename, outputFilename, model, modelParams):
    with open(inputFilename, 'r') as input, open(outputFilename, 'w') as output:
        for line in input:
            data = json.loads(line)
            if "time" in data:
                data["time"] = model(modelParams, data["time"])
            output.write(json.dumps(data) + '\n')

def compute_imu_frequency(timestamps):
    avgTimeDiff = (timestamps[-1] - timestamps[0]) / len(timestamps)
    return 1.0 / avgTimeDiff

def plot_synchronized_signals(imu1, imu2, lag):
    plt.plot(np.arange(0, len(imu1)) + lag, imu1, label='imu1')
    plt.plot(imu2, label='imu2')
    plt.xlabel('Timestamp')
    plt.ylabel('IMU value')
    plt.title('Synchronized IMU Signals')
    plt.legend()
    plt.show()

def compute_lag_cross_correlation(imu1, imu2, plot, mode):
    crossCorr = signal.correlate(imu2, imu1, mode=mode)
    lags = signal.correlation_lags(len(imu2), len(imu1), mode=mode)
    lag = lags[np.argmax(crossCorr)] # best lag

    if plot:
        # Plot the cross-correlation
        plt.plot(lags, crossCorr)
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.title('Cross-Correlation of IMU Signals')
        plt.show()
        plot_synchronized_signals(imu1, imu2, lag)

    return lag

# Computes sum((imu2-imu1)^2) at each lag value (minLag, maxLag) and returns the best lag.
# The returned lag aligns the two signals as: t_imu2 = t_imu1 + lag_to_time_offset(lag)
# Note: len(imu2) >= len(imu1) + (maxLag - minLag)
def compute_lag_euclidian(imu1, imu2, minLag, maxLag, plot):
    minDistance = float('inf')
    bestLag = 0

    # Iterate over different time offsets
    totalLag = maxLag - minLag
    for lag in range(totalLag):
        start = lag
        end = start + len(imu1)
        if end >= len(imu2):
            raise IndexError("Tried to access index {0} in array with size {1}".format(end, len(imu2)))

        distance = np.sum(np.square(imu2[start:end] - imu1))

        if distance < minDistance:
            minDistance = distance
            bestLag = lag

    if plot: plot_synchronized_signals(imu1, imu2, bestLag)

    return bestLag + minLag

def lag_to_time_offset(lag, timestamps1, timestamps2):
    timeOffset = lag / compute_imu_frequency(timestamps1)
    timeOffset += timestamps2[0] - timestamps1[0]
    return timeOffset

def linear_model(params, x):
    a, b = params
    return a * x + b

def quadratic_model(params, x):
    a, b, c = params
    return a * x**2 + b * x + c

def estimate_time_scale(dataImu1, dataImu2, stepSeconds):
    timestamps1 = dataImu1[:, 0]
    timestamps2 = dataImu2[:, 0]
    imu1 = dataImu1[:, 1]
    imu2 = dataImu2[:, 1]
    n1 = len(timestamps1)
    n2 = len(timestamps2)

    # Compute rough time offset; should be pretty accurate even with long datasets.
    # t_imu2 = t_imu1 + lag_to_offset(estimateLag)
    estimateLag = compute_lag_cross_correlation(imu1, imu2, True, 'full')

    times = []
    lags = []
    frequency = compute_imu_frequency(timestamps1)
    maxLag = round(0.2 * frequency)
    minLag = -maxLag
    step = round(stepSeconds * frequency)
    for idx1 in range(0, n1 - step, step):
        start1 = idx1
        end1 = idx1+step

        idx2 = idx1 + estimateLag # t_imu2 = t_imu1 + t_offset
        start2 = idx2+minLag
        end2 = idx2+step+maxLag
        if start2 < 0: continue
        if end2 >= n2: break

        part1 = imu1[start1:end1]
        part2 = imu2[start2:end2]

        # Skip "flat" signals, cannot be registered reliably
        if np.var(part1) < 1e-4: continue

        lag = compute_lag_euclidian(part1, part2, minLag, maxLag, False) + estimateLag
        lags.append(lag)
        times.append(timestamps1[idx1])

    plt.plot(times, lags, linestyle='None', marker='.')
    plt.ylabel('Lag (index)')
    plt.xlabel('Time (seconds)')
    plt.title('Time offset computed from {0} second sequences'.format(stepSeconds))
    plt.show()

    # Fit linear and quadratic models to estimate how time offset changes over time
    x = np.asarray(times)
    y = np.asarray(lag_to_time_offset(lags, timestamps1, timestamps2))

    def objective_linear(params, x, y):
        return y - linear_model(params, x)

    def objective_quadratic(params, x, y):
        return y - quadratic_model(params, x)

    # Fit linear model with soft_l1 loss using least_squares
    resultLinear = optimize.least_squares(objective_linear, [0.0, lag_to_time_offset(estimateLag, timestamps1, timestamps2)], loss='soft_l1', args=(x, y))
    paramsLinear = resultLinear.x
    linearFit = linear_model(paramsLinear, x)
    rmseLinear = np.sqrt(np.mean((y - linearFit)**2))

    # Fit quadratic model with soft_l1 loss using least_squares
    resultQuadratic = optimize.least_squares(objective_quadratic, [0.0, 0.0, lag_to_time_offset(estimateLag, timestamps1, timestamps2)], loss='soft_l1', args=(x, y))
    paramsQuadratic = resultQuadratic.x
    quadratic_fit = quadratic_model(paramsQuadratic, x)
    rmseQuadratic = np.sqrt(np.mean((y - quadratic_fit)**2))

    # Plot the original data and the fitted models
    plt.scatter(x, y, label='Data')
    plt.plot(x, linearFit, label=f'Linear Fit (RMSE={rmseLinear:.2f})', color='red')
    plt.plot(x, quadratic_fit, label=f'Quadratic Fit (RMSE={rmseQuadratic:.2f})', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time offset (seconds)')
    plt.title('Time offset over time fits with soft L1 loss')
    plt.legend()
    plt.show()

    timestamps1Linear = timestamps1 + linear_model(paramsLinear, timestamps1)
    plt.plot(timestamps1Linear, imu1, label='imu1')
    plt.plot(timestamps2, imu2, label='imu2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Imu value')
    plt.title('Synchronized IMU Signals (linear model applied to imu1 timestamps)')
    plt.legend()
    plt.show()

    timestamps1Quadratic = timestamps1 + quadratic_model(paramsQuadratic, timestamps1)
    plt.plot(timestamps1Quadratic, imu1, label='imu1')
    plt.plot(timestamps2, imu2, label='imu2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Imu value')
    plt.title('Synchronized IMU Signals (quadratic model applied to imu1 timestamps)')
    plt.legend()
    plt.show()

    # (linear) t_imu2 = a * t_imu1 + b + t_imu1 = (1 + a) * t_imu1 + b
    # (quadratic) t_imu2 = a * t_imu1^2 + b * t_imu1 + c + t_imu1 = a * t_imu1^2 + (1 + b) * t_imu1 + c
    paramsLinear[0] += 1
    paramsQuadratic[1] += 1

    return paramsLinear, paramsQuadratic

# Resample the lower frequency IMU signal (the time sync code assumes that both signals have same frequency)
def resample_IMU_data(dataImu1, dataImu2):
    timestamps1 = dataImu1[:, 0]
    timestamps2 = dataImu2[:, 0]
    freq1 = compute_imu_frequency(timestamps1)
    freq2 = compute_imu_frequency(timestamps2)

    if freq1 > freq2:
        ratio = freq1 / freq2
        n = round(ratio * len(timestamps2))
        timestamps2Resampled = np.linspace(timestamps2[0], timestamps2[-1], n)
        imuValues2Resampled = np.interp(timestamps2Resampled, timestamps2, dataImu2[:, 1])
        return dataImu1, np.column_stack((timestamps2Resampled, imuValues2Resampled))
    else:
        ratio = freq2 / freq1
        n = round(ratio * len(timestamps1))
        timestamps1Resampled = np.linspace(timestamps1[0], timestamps1[-1], n)
        imuValues1Resampled = np.interp(timestamps1Resampled, timestamps1, dataImu1[:, 1])
        return np.column_stack((timestamps1Resampled, imuValues1Resampled)), dataImu2

if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("imu1", help="Path to first data.jsonl")
    p.add_argument("imu2", help="Path to second data.jsonl")
    p.add_argument("--output", help="Output directory (copy of imu1 data.jsonl with timestamps adjusted will be saved there)")
    p.add_argument("--axis", choices=['x', 'y', 'z'], default='x', help="Axis (x, y, or z)")
    p.add_argument("--accelerometer", help="Use accelerometer instead of gyroscape", action="store_true")
    p.add_argument("--timestamp_range", help="Compute the offset using subsample of the original data.jsonl. Format is start:end in seconds")
    p.add_argument("--time_scale", help="Estimate time scale; Specifically, t_imu2 = t_imu1 * t_scale + t_offset", action="store_true")
    p.add_argument("--step", type=float, default=5.0, help="In time scale estimation, the dataset is divided into parts using this step (seconds)")
    p.add_argument("--imu1_to_imu2", help="Path to json file that contains 3x3 rotation matrix 'imu1ToImu2' to align the imu signals")
    args = p.parse_args()

    sensor = "accelerometer" if args.accelerometer else "gyroscope"
    dataImu1 = read_imu_data(args.imu1, args.timestamp_range, sensor)
    dataImu2 = read_imu_data(args.imu2, args.timestamp_range, sensor)

    if args.imu1_to_imu2:
        imu1ToImu2 = read_imu1_to_imu2(args.imu1_to_imu2)
        dataImu1[:, 1:] = np.matmul(imu1ToImu2, dataImu1[:, 1:].T).T

    # Plot original data (with imu1 optionally rotated)
    _, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, sharex=True)
    ax1.plot(dataImu1[:, 0] - dataImu1[0, 0], dataImu1[:, 1])
    ax2.plot(dataImu1[:, 0] - dataImu1[0, 0], dataImu1[:, 2])
    ax3.plot(dataImu1[:, 0] - dataImu1[0, 0], dataImu1[:, 3])
    ax4.plot(dataImu2[:, 0] - dataImu2[0, 0], dataImu2[:, 1])
    ax5.plot(dataImu2[:, 0] - dataImu2[0, 0], dataImu2[:, 2])
    ax6.plot(dataImu2[:, 0] - dataImu2[0, 0], dataImu2[:, 3])
    ax1.set_ylabel('{0} (1) x'.format(sensor))
    ax2.set_ylabel('{0} (1) y'.format(sensor))
    ax3.set_ylabel('{0} (1) z'.format(sensor))
    ax4.set_ylabel('{0} (2) x'.format(sensor))
    ax5.set_ylabel('{0} (2) y'.format(sensor))
    ax6.set_ylabel('{0} (2) z'.format(sensor))
    ax6.set_xlabel('Timestamp')
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    if args.axis == 'x':
        axis = 1
    elif args.axis == 'y':
        axis = 2
    else:
        axis = 3

    dataImu1 = dataImu1[:, [0, axis]]
    dataImu2 = dataImu2[:, [0, axis]]
    dataImu1, dataImu2 = resample_IMU_data(dataImu1, dataImu2)

    if args.time_scale:
        paramsLinear, paramsQuadratic = estimate_time_scale(dataImu1, dataImu2, args.step)
        print('(linear) t_imu2 = {0} * t_imu1 + {1}'.format(paramsLinear[0], paramsLinear[1]))
        print('(quadratic) t_imu2 = {0} * t_imu1^2 + {1} * t_imu1 + {2}'.format(paramsQuadratic[0], paramsQuadratic[1], paramsQuadratic[2]))
        if args.output:
            output = "{}/imu1_linear_data.jsonl".format(args.output)
            write_output_jsonl(args.imu1, output, linear_model, paramsLinear)
            output = "{}/imu1_quadratic_data.jsonl".format(args.output)
            write_output_jsonl(args.imu1, output, quadratic_model, paramsQuadratic)
    else:
        lag = compute_lag_cross_correlation(dataImu1[:, 1], dataImu2[:, 1], True, 'full')
        timeOffset = lag_to_time_offset(lag, dataImu1[:, 0], dataImu2[:, 0])
        print('t_imu2 = t_imu1 + t_offset, where t_offset = {0}'.format(timeOffset))
        if args.output:
            output = "{}/imu1_offset_data.jsonl".format(args.output)
            write_output_jsonl(args.imu1, output, linear_model, (1.0, timeOffset))
