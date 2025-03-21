#!/usr/bin/env python
#
# Process "JSONL format" data precisely so that the frame indices don't get out of sync.
#
# For example, the following crops the 10s segment starting from the 15s mark:
#   python process_jsonl_data.py data/benchmark/euroc-v1-01-easy cropped-euroc --t0 15 --t1 25

import json
import os
import pathlib
import shutil
import subprocess

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
    return [x for x in os.listdir(folder) if x.split('.')[-1] in FORMATS]

def probeFps(inputFile):
    fpsDiv = subprocess.check_output("ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=avg_frame_rate {}".format(inputFile), shell=True).decode('utf-8').strip()
    return eval(fpsDiv) # ffprobe out has format like "60/1", evaluate the division.

def probeCodec(inputFile):
    codec = subprocess.check_output("ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {}".format(inputFile), shell=True).decode('utf-8').strip()
    return codec

def handleVideo(args, inputVideo, outputFolder, videoInd):
    # Note: these are set in crop() if using --t0 or --t1.
    n0 = args.skipStartFramesInd
    n1 = args.skipEndFramesInd

    if args.skipFramesInVideo:
        tokens = args.skipFramesInVideo.split(",")
        extra = int(tokens[videoInd])
        print("Skipping {} extra frames in {}".format(extra, inputVideo))
        if n0 is None: n0 = 0
        n0 += extra

    if n0 and n1: assert(n1 > n0)
    n0value = n0 if n0 else 0

    container = "mkv"
    filters = []
    if probeCodec(inputVideo) == "ffv1":
        # Lossless 16bit codec.
        codecArgs = "-vcodec ffv1"
    else:
        preset = " -preset veryfast"
        codecArgs = "-c:v libx264 -crf {}{}".format(args.crf, preset)
        filters.append("format=yuv420p")

    if "data2" in inputVideo:
        output = "data2.{}".format(container)
    elif "data3" in inputVideo:
        output = "data3.{}".format(container)
    elif not "data." in inputVideo:
        raise Exception(f"Unexpected video name: {inputVideo}")
    else:
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

    skipStartTime = None
    searchingStart = args.t0 is not None
    searchingStartInd = args.skipStartFramesInd is not None

    skipEndTime = None
    searchingEnd = args.t1 is not None
    searchingEndInd = args.skipEndFramesInd is not None

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
            args.skipStartFramesInd = int(j["number"])
            skipStartTime = float(j["time"])
        elif searchingStartInd and frameInd == args.skipStartFramesInd:
            searchingStartInd = False
            skipStartTime = float(j["time"])

        if searchingEnd and td >= args.t1:
            searchingEnd = False
            args.skipEndFramesInd = int(j["number"])
            skipEndTime = float(j["time"])
        elif searchingEndInd and frameInd == args.skipEndFramesInd:
            searchingEndInd = False
            skipEndTime = float(j["time"])

    # Remove triggers.
    jsonls = [j for j in jsonls if not "trigger" in j]

    if skipStartTime:
        jsonls = [j for j in jsonls if j["time"] >= skipStartTime]
        for j in jsonls:
            if not "frames" in j: continue
            j["number"] -= args.skipStartFramesInd
            # These nested copies are not required.
            for f in j["frames"]:
                if "number" in f: f["number"] -= args.skipStartFramesInd
    if skipEndTime:
        jsonls = [j for j in jsonls if j["time"] <= skipEndTime]

    return jsonls

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
    jsonls = slurpJsonl(args.input / "data.jsonl")
    jsonls = sorted(jsonls, key=lambda row: row["time"])

    # These are difficult to implement together, support only one at a time. You
    # can always run the script again.
    if args.t0 or args.t1 or args.skipStartFramesInd or args.skipEndFramesInd:
        assert(not args.subsample)
        jsonls = crop(args, jsonls)
    elif args.subsample:
        jsonls = subsample(args, jsonls)

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    with open("{}/data.jsonl".format(args.output), "w") as f:
        for j in jsonls:
            f.write(json.dumps(j, separators=(',', ':')))
            f.write("\n")

    if args.videos is not None:
        videos = args.videos.split(",")
    else:
        videos = findVideos(args.input)

    for i, fileName in enumerate(videos):
        handleVideo(args, str(args.input / fileName), args.output, i)

    for file in ["calibration.json", "vio_config.yaml"]:
        inputPath = args.input / file
        outputPath = args.output / file
        if not inputPath.exists(): continue
        shutil.copyfile(inputPath, outputPath)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path, help="Path to JSONL data folder.")
    p.add_argument("output", type=pathlib.Path, help="Path to folder to be created.")
    p.add_argument("--t0", type=float, help="Skip data before this many seconds from beginning")
    p.add_argument("--t1", type=float, help="Skip data after this many seconds from beginning.")
    p.add_argument("--skipStartFramesInd", type=int, help="Like t0 but exact number of frames")
    p.add_argument("--skipEndFramesInd", type=int, help="Like t1 but exact number of frames")
    p.add_argument("--subsample", type=int, help="Keep every nth frame.")
    p.add_argument("--downscale", help="Factor to downscale videos by.")
    p.add_argument("--crf", type=int, default=15, help="h264 encoding quality value (0=lossless, 17=visually lossless)")
    p.add_argument("--videos", help="List videos to convert comma-separated, otherwise will use all from the input folder")
    p.add_argument("--skipFramesInVideo", help="Comma-separated extra frames to skip per input video. Can be used to adjust stereo sync.")
    args = p.parse_args()

    assert(args.t0 is None or args.skipStartFramesInd is None)
    assert(args.t1 is None or args.skipEndFramesInd is None)

    main(args)
