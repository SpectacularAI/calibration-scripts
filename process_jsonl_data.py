#!/usr/bin/env python
#
# Process "JSONL format" data precisely so that the frame indices don't get out of sync.
#
# For example, the following crops the 10s segment starting from the 15s mark:
#   python process_jsonl_data.py data/benchmark/euroc-v1-01-easy cropped-euroc --t0 15 --t1 25

import argparse
import json
import os
import pathlib
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to JSONL data folder.")
parser.add_argument("output", help="Path to folder to be created.")
parser.add_argument("--t0", type=float, help="Skip data before this many seconds from beginning")
parser.add_argument("--t1", type=float, help="Skip data after this many seconds from beginning.")
parser.add_argument("--subsample", type=int, help="Keep every nth frame.")
parser.add_argument("--downscale", help="Factor to downscale videos by.")
parser.add_argument("--crf", type=int, default=15, help="h264 encoding quality value (0=lossless, 17=visually lossless)")

def slurpJsonl(path):
    jsonls = []
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if not "time" in j: continue
            jsonls.append(j)
    return jsonls

def findVideos(folder):
    FORMATS = ['avi', 'mp4', 'mov', 'mkv']
    return [os.path.join(folder, x) for x in os.listdir(folder) if x.split('.')[-1] in FORMATS]

def probeFps(inputFile):
    fpsDiv = subprocess.check_output("ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=avg_frame_rate {}".format(inputFile), shell=True).decode('utf-8').strip()
    return eval(fpsDiv) # ffprobe out has format like "60/1", evaluate the division.

def probeCodec(inputFile):
    codec = subprocess.check_output("ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {}".format(inputFile), shell=True).decode('utf-8').strip()
    return codec

def handleVideo(args, inputVideo, outputFolder, n0, n1):
    if n0 and n1: assert(n1 > n0)
    n0value = n0 if n0 else 0

    container = "mkv"
    filters = []
    if probeCodec(inputVideo) == "ffv1":
        # This is the depth format for eg OAK-D recordings.
        container = "avi"
        codecArgs = "-vcodec ffv1"
    else:
        preset = " -preset veryfast"
        codecArgs = "-c:v libx264 -crf {}{}".format(args.crf, preset)
        filters.append("format=yuv420p")

    if "data2" in inputVideo:
        output = "data2.{}".format(container)
    elif "data3" in inputVideo:
        output = "data3.{}".format(container)
    else:
        assert("data." in inputVideo)
        output = "data.{}".format(container)

    output = "{}/{}".format(outputFolder, output)
    # Tested these two filter to remove exactly correct number of frames, counting them
    # from the resulting video using `ffmpeg`. VIO also works well so the start index should be correct.
    if n0 and n0 >= 1: filters.append("select=gt(n\, {}),setpts=PTS-STARTPTS".format(n0 - 1))
    if n1: filters.append("select=lt(n\, {})".format(n1 - n0value + 1))

    fps = probeFps(inputVideo)
    assert(fps > 0)
    fpsSub = fps
    if args.subsample:
        filters.append("select=not(mod(n\,{}))".format(args.subsample))
        fpsSub = fps / args.subsample
    if args.downscale: filters.append("scale=iw/{}:ih/{}".format(args.downscale, args.downscale))

    cmd = "ffmpeg -r {fps} -i {inputVideo} -r {fpsSub} -start_number 0 {codecArgs} -vf \"{filters}\" {output}".format(
        inputVideo=inputVideo,
        fps=fps,
        fpsSub=fpsSub,
        codecArgs=codecArgs,
        output=output,
        filters=",".join(filters)
    )
    print("Running command:", cmd)
    subprocess.run(cmd, shell=True)

def crop(args, jsonls):
    tDataStart = None
    lastFrameInd = None

    skipStartFramesInd = None
    skipStartTime = None
    searchingStart = args.t0 is not None

    skipEndFramesInd = None
    skipEndTime = None
    searchingEnd = args.t1 is not None

    for j in jsonls:
        # With this, the relative timestamps are video timestamps which are
        # easy to find using a video player.
        if not "frames" in j:
            continue

        frameInd = int(j["number"])
        if lastFrameInd:
            # The cropping logic probably breaks if this doesn't hold for the time sorted rows.
            assert(frameInd > lastFrameInd)
        lastFrameInd = frameInd

        t = j["time"]
        if not tDataStart:
            tDataStart = t
        td = t - tDataStart

        if searchingStart and td >= args.t0:
            searchingStart = False
            skipStartFramesInd = int(j["number"])
            skipStartTime = float(j["time"])

        if searchingEnd and td >= args.t1:
            searchingEnd = False
            skipEndFramesInd = int(j["number"])
            skipEndTime = float(j["time"])

    # Remove triggers.
    jsonls = [j for j in jsonls if not "trigger" in j]

    if skipStartTime:
        jsonls = [j for j in jsonls if j["time"] >= skipStartTime]
        for j in jsonls:
            if not "frames" in j: continue
            j["number"] -= skipStartFramesInd
            # These nested copies are not required.
            for f in j["frames"]:
                if "number" in f: f["number"] -= skipStartFramesInd
    if skipEndTime:
        jsonls = [j for j in jsonls if j["time"] <= skipEndTime]

    return jsonls, skipStartFramesInd, skipEndFramesInd

def subsample(args, jsonls):
    assert(args.subsample > 1)
    n = 0
    nOut = 0
    output = []
    for j in jsonls:
        if "frames" in j:
            if n % args.subsample == 0:
                j["number"] = nOut
                nOut += 1
                output.append(j)
            n += 1
        else:
            output.append(j)
    return output # Already sorted.

def main(args):
    jsonls = slurpJsonl("{}/data.jsonl".format(args.input))
    jsonls = sorted(jsonls, key=lambda row: row["time"])

    skipStartFramesInd = None
    skipEndFramesInd = None
    # These are difficult to implement together, support only one at a time. You
    # can always run the script again.
    if args.t0 or args.t1:
        assert(not args.subsample)
        jsonls, skipStartFramesInd, skipEndFramesInd = crop(args, jsonls)
    elif args.subsample:
        jsonls = subsample(args, jsonls)

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    with open("{}/data.jsonl".format(args.output), "w") as f:
        for j in jsonls:
            f.write(json.dumps(j, separators=(',', ':')))
            f.write("\n")

    for video in findVideos(args.input):
        handleVideo(args, video, args.output, skipStartFramesInd, skipEndFramesInd)

    for file in ["calibration.json", "vio_config.yaml"]:
        inputPath = pathlib.Path(args.input) / file
        outputPath = pathlib.Path(args.output) / file
        if not inputPath.exists(): continue
        shutil.copyfile(inputPath, outputPath)

if __name__ == '__main__':
    main(parser.parse_args())
