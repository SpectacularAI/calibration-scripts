"""
Convert a recording to "ground truth" format and add it to another recording for benchmarking.

For instance, you can get relatively accurate poses with Orbbec Femto Mega/Bolt device and then compare those with OAK-D poses.
"""

import json
import os
import numpy as np

def synchronizeRecordings(args):
    import sys
    import importlib

    sys.argv = [
        'sync-imus.py',
        f'{os.path.join(args.ground_truth, "data.jsonl")}',
        f'{os.path.join(args.data, "data.jsonl")}',
        '--no_plot'
    ]
    offset = importlib.import_module("sync-imus").synchronizeImus()
    return offset

def computeGroundTruth(args):
    import spectacularAI

    with open(os.path.join(args.ground_truth, "calibration.json")) as f:
        calibration = json.load(f)
        imuToPrimaryCamera = np.array(calibration["cameras"][0]["imuToCamera"])

    def poseToJson(position, orientation):
        poseJson = {}
        poseJson["position"] = {
            "x": position.x,
            "y": position.y,
            "z": position.z
        }
        if orientation is not None:
            poseJson["orientation"] = {
                "x": orientation.x,
                "y": orientation.y,
                "z": orientation.z,
                "w": orientation.w
            }
        return poseJson

    def isAlreadyRectified(input_dir):
        vioConfigYaml = f"{input_dir}/vio_config.yaml"
        if os.path.exists(vioConfigYaml):
            with open(vioConfigYaml) as file:
                for line in file:
                    if "alreadyRectified" in line:
                        _, value = line.split(":")
                        return value.lower().strip() == "true"
        return False

    visualizer = None
    groundTruthLines = []

    def onVioOutput(output):
        if visualizer is not None:
            visualizer.onVioOutput(output.getCameraPose(0), status=output.status)

    def onMappingOutput(output):
        if visualizer is not None:
            visualizer.onMappingOutput(output)

        if output.finalMap:
            for frameId in output.map.keyFrames:
                keyFrame = output.map.keyFrames.get(frameId)
                primary = keyFrame.frameSet.primaryFrame
                pose = primary.cameraPose.pose
                imuToWorld = pose.asMatrix() @ imuToPrimaryCamera # imu->world = imu->camera->world = camera->world * imu->camera
                pose = spectacularAI.Pose.fromMatrix(pose.time, imuToWorld)
                gtJson = {
                    "time": pose.time,
                    "groundTruth": poseToJson(pose.position, None if args.no_orientation else pose.orientation)
                }
                groundTruthLines.append(gtJson)

    def parseInputDir(input_dir):
        device = None
        metadataJson = f"{input_dir}/metadata.json"
        if os.path.exists(metadataJson):
            with open(metadataJson) as f:
                metadata = json.load(f)
                if metadata.get("platform") == "ios":
                    device = "ios-tof"
        if device == None:
            vioConfigYaml = f"{input_dir}/vio_config.yaml"
            if os.path.exists(vioConfigYaml):
                with open(vioConfigYaml) as file:
                    supported = ['oak-d', 'k4a', 'realsense', 'orbbec-astra2', 'orbbec-femto', 'android', 'android-tof']
                    for line in file:
                        if "parameterSets" in line:
                            for d in supported:
                                if d in line:
                                    device = d
                                    break
                        if device: break
        return device

    config = {
        "maxMapSize": 0,
        "useSlam": True,
        "passthroughColorImages": True,
        "keyframeDecisionDistanceThreshold": args.key_frame_distance,
        "icpVoxelSize": min(args.key_frame_distance, 0.1)
    }

    parameter_sets = ['wrapper-base']
    device_preset = parseInputDir(args.ground_truth)

    if not args.fast:
        parameter_sets.append('offline-base')
        # remove these to further trade off speed for quality
        mid_q = {
            'maxKeypoints': 1000,
            'optimizerMaxIterations': 30
        }
        for k, v in mid_q.items(): config[k] = v

    if args.device_preset:
        device_preset = args.device_preset

    if args.internal is not None:
        for param in args.internal:
            k, _, v = param.partition(':')
            config[k] = v

    if device_preset: print(f"Selected device type: {device_preset}", flush=True)
    else: print("Warning! Couldn't automatically detect device preset, to ensure best results suply one via --device_preset argument", flush=True)

    if device_preset:
        parameter_sets.append(device_preset)

    if device_preset == 'k4a':
        parameter_sets.extend(['icp'])
        if not args.fast: parameter_sets.append('offline-icp')
    elif device_preset == 'realsense':
        parameter_sets.extend(['icp', 'realsense-icp'])
        if not args.fast: parameter_sets.append('offline-icp')
    elif device_preset == 'oak-d':
        config['stereoPointCloudMinDepth'] = 0.5
        config['alreadyRectified'] = isAlreadyRectified(args.ground_truth) # rectification required for stereo point cloud
    elif device_preset is not None and "orbbec" in device_preset:
        parameter_sets.extend(['icp'])
        if not args.fast: parameter_sets.append('offline-icp')
    config['parameterSets'] = parameter_sets

    print(f"configuration={config}")
    replay = spectacularAI.Replay(args.ground_truth, mapperCallback=onMappingOutput, configuration=config, ignoreFolderConfiguration=True)

    if args.no_preview:
        replay.runReplay()
    else:
        from spectacularAI.cli.visualization.visualizer import Visualizer, VisualizerArgs, ColorMode
        visArgs = VisualizerArgs()
        visArgs.targetFps = 30
        visArgs.colorMode = ColorMode.NORMAL
        visualizer = Visualizer(visArgs)
        replay.setOutputCallback(onVioOutput)
        replay.startReplay()
        visualizer.run()
        replay.close()

    replay = None
    if len(groundTruthLines) == 0:
        print("Failed to compute ground truth")
    else:
        print("Computed ground truth")
    return groundTruthLines

def addGroundTruthLinesToRecording(args, gt):
    output = []
    dataJsonl = os.path.join(args.data, "data.jsonl")
    with open(dataJsonl) as inputJsonFile:
        for line in inputJsonFile:
            obj = json.loads(line)
            if not "time" in obj: continue
            if "groundTruth" in obj: continue # drop existing gt lines
            output.append(obj)
    output.extend(gt)
    output = sorted(output, key=lambda row: row["time"])
    with open(dataJsonl, "w") as outputJsonFile:
        for filtered in output:
            outputJsonFile.write(json.dumps(filtered, separators=(',', ':')))
            outputJsonFile.write("\n")

if __name__ == '__main__':
    def parseArgs():
        import argparse
        p = argparse.ArgumentParser(__doc__)
        p.add_argument("ground_truth", help="Ground truth is computed using this recording", default="data")
        p.add_argument("data", help="Recording to which the ground truth is added", default="output")
        p.add_argument("--no_orientation", help="Don't add orientation to ground truth", action="store_true")
        p.add_argument("--no_preview", help="Disable visualization", action="store_true")
        p.add_argument('--fast', action='store_true', help='Fast but lower quality settings')
        p.add_argument("--key_frame_distance", help="Minimum distance between keyframes (meters)", type=float, default=0.05)
        p.add_argument('--device_preset', choices=['none', 'oak-d', 'k4a', 'realsense', 'android', 'android-tof', 'ios-tof', 'orbbec-astra2', 'orbbec-femto'], help="Automatically detected in most cases")
        p.add_argument('--internal', action='append', type=str, help='Internal override parameters in the form --internal=name:value')
        p.add_argument('--imu_to_output', help='data_imu->ground_truth_imu transformation, for example: "[[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]"')
        return p.parse_args()

    args = parseArgs()
    if args.imu_to_output:
        imuToOutput = json.loads(args.imu_to_output)
        filename = os.path.join(args.data, "calibration.json")
        with open(filename, 'r') as f:
            calibration = json.load(f)
            calibration["imuToOutput"] = imuToOutput
        with open(filename, 'w') as f:
            json.dump(calibration, f, indent=4)

    offset = synchronizeRecordings(args)
    gt = computeGroundTruth(args)
    for line in gt: line["time"] += offset
    addGroundTruthLinesToRecording(args, gt)
