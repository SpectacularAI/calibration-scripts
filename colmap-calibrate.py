#!/usr/bin/env python3
#
# For sensible running speed, you need the GPU (CUDA) support enabled in COLMAP.
#
# Usage:
#   * Install COLMAP and FFmpeg. Then run:
#   python colmap-calibrate.py path/to/recording-folder
#
#   * You may need to tweak `--mapperParameters` for best results / faster convergence.

"""Use COLMAP to calibrate a Spectacular AI SDK recording. Also requires FFmpeg."""

import json
import os
import pathlib
import re
import shutil
import subprocess

def findVideos(folder):
    FORMATS = ['.avi', '.mp4', '.mov', '.mkv']
    return [(folder / x) for x in os.listdir(folder) if pathlib.Path(x).suffix in FORMATS]

def getVideoInd(videoPath):
    m = re.match("data(\d*)", videoPath.stem)
    if m.group(1) == "": return 1 # data.mp4 is same as data1.mp4.
    return int(m.group(1))

def countFrames(videoPath):
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {videoPath}"
    n = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return int(n)

def runWithLogging(cmd, name, path):
    process = subprocess.run(cmd, shell=True, capture_output=True)
    with open(path / f"{name}-stderr", "w") as f:
        err = process.stderr.decode('utf-8').strip()
        f.write(err)
    with open(path / f"{name}-stdout", "w") as f:
        f.write(process.stdout.decode('utf-8').strip())
    if err == "": return None
    return err

def calibrateVideo(args, videoPath, videoWorkPath):
    imagesPath = videoWorkPath / "images"
    if imagesPath.exists():
        print("Skipping video-to-image conversion.")
    else:
        # Take every n:th frame to get approximately args.frameCount frames.
        print("Counting frames.")
        n = countFrames(videoPath)

        subsample = int(n / args.frameCount)
        if subsample == 0: subsample = 1
        imagesPath.mkdir(parents=True, exist_ok=True)
        cmd = f"ffmpeg -i {videoPath} -vf \"select=not(mod(n\\,{subsample}))\" -vsync 0 {imagesPath}/%08d.png"

        print("Converting video to images ({} frames).".format(int(n / subsample)))
        runWithLogging(cmd, "ffmpeg", videoWorkPath)

    # May fix crash with COLMAP.
    env = "QT_QPA_PLATFORM=offscreen"

    databasePath = videoWorkPath / "database.db"
    if databasePath.exists():
        print("Skipping feature extraction.")
    else:
        print("Running feature extraction.")
        cmd = f"{env} colmap feature_extractor --database_path {databasePath}"
        cmd += f" --image_path {imagesPath}"
        cmd += " --ImageReader.single_camera 1"
        cmd += f" --ImageReader.camera_model {args.model.upper()}"
        runWithLogging(cmd, "colmap-feature-extractor", videoWorkPath)

    matchingMethod = "sequential_matcher" # "vocab_tree_matcher" might be better.
    matchingDonePath = videoWorkPath / "matching_done"
    if matchingDonePath.exists():
        print("Skipping feature matching.")
    else:
        print("Running feature matching.")
        cmd = f"{env} colmap {matchingMethod} --database_path {databasePath}"
        runWithLogging(cmd, "colmap-matching", videoWorkPath)
        with open(matchingDonePath, "w") as f:
            f.write(matchingMethod)

    mapperPath = videoWorkPath / "mapper"
    if mapperPath.exists():
        print("Skipping mapping.")
    else:
        print("Running mapping.")
        mapperPath.mkdir(parents=True, exist_ok=True)
        cmd = f"{env} colmap mapper --database_path {databasePath}"
        cmd += f" --image_path {imagesPath}"
        cmd += f" --output_path {mapperPath}"
        cmd += f" {args.mapperParameters}"
        runWithLogging(cmd, "colmap-mapper", videoWorkPath)

        # The principal point in all camera models is by default the exact middle of the image,
        # and COLMAP documentation says estimating it "unstable" (although there is an option to do so).
        print("Refining principal points.")
        cmd = f"{env} colmap bundle_adjuster"
        cmd += f" --input_path {mapperPath}/0"
        cmd += f" --output_path {mapperPath}/0"
        cmd += " --BundleAdjustment.refine_principal_point 1"
        runWithLogging(cmd, "colmap-ba-principal-point-refinement", videoWorkPath)

    # Never skip this phase.
    print("Converting outputs to text format.")
    textModelPath = videoWorkPath / "text-model"
    textModelPath.mkdir(parents=True, exist_ok=True)
    cmd = f"{env} colmap model_converter --output_type TXT"
    cmd += f" --input_path \"{mapperPath}/0\""
    cmd += f" --output_path \"{textModelPath}\""
    runWithLogging(cmd, "colmap-model-converter", videoWorkPath)

    print("Video ok.")
    return None

def main(args):
    if not shutil.which("colmap"):
        print("Could not find `colmap`. Install COLMAP and setup paths so that the commandline tool works.")
        return
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Could not find `ffmpeg`/`ffprobe`. Install FFmpeg and setup paths so that the commandline tool works.")
        return

    videoPaths = findVideos(args.datasetPath)
    if len(videoPaths) == 0:
        print("No video files found.")
        return
    videoPaths.sort()

    calibrationPath = args.datasetPath / "colmap-calibration"
    if not args.dirty and calibrationPath.exists():
        print(f"Folder {calibrationPath} exists. It will be removed. Continue? [y/N]")
        if input().lower() != "y": return
        shutil.rmtree(calibrationPath)

    workPath = calibrationPath / "work"
    workPath.mkdir(parents=True, exist_ok=True)

    for videoPath in videoPaths:
        videoInd = getVideoInd(videoPath)
        if videoInd is None:
            print(f"Skipping {videoPath}")
            continue
        print(f"---\nProcessing video {videoInd}/{len(videoPaths)}\n---")
        videoWorkPath = workPath / f"data{videoInd}"

        err = calibrateVideo(args, videoPath, videoWorkPath)
        if err is not None:
            print("\nCalibration failed:")
            print(err)
            return

    # Convert to Spectacular AI calibration.json format.
    calibration = { "cameras": [] }
    for videoPath in videoPaths:
        videoInd = getVideoInd(videoPath)
        videoWorkPath = workPath / f"data{videoInd}"
        cameraPath = videoWorkPath / "text-model" / "cameras.txt"
        with open(cameraPath) as f:
            for line in f:
                if line.startswith("#"): continue
                tokens = line.split(" ")
                break
        # TODO Support other camera models.
        # <https://colmap.github.io/cameras.html>
        if tokens[1] == "RADIAL":
            calibration["cameras"].append({
                "imageWidth": int(tokens[2]),
                "imageHeight": int(tokens[3]),
                "focalLengthX": float(tokens[4]), # Note that focal length x and y are not separate.
                "focalLengthY": float(tokens[4]),
                "principalPointX": float(tokens[5]),
                "principalPointY": float(tokens[6]),
                "model": "pinhole",
                "distortionCoefficients": [float(tokens[7]), float(tokens[8]), 0.],
            })
        elif tokens[1] == "OPENCV_FISHEYE":
            calibration["cameras"].append({
                "imageWidth": int(tokens[2]),
                "imageHeight": int(tokens[3]),
                "focalLengthX": float(tokens[4]),
                "focalLengthY": float(tokens[5]),
                "principalPointX": float(tokens[6]),
                "principalPointY": float(tokens[7]),
                "model": "kannala-brandt4",
                "distortionCoefficients": [float(tokens[8]), float(tokens[9]), float(tokens[10]), float(tokens[11]) ],
            })
        else:
            print("Unsupported conversion, raw output:", tokens)
        # TODO Add option to copy imuToCamera from an existing calibration.

    with open(calibrationPath / "calibration.json", "w") as f:
        f.write(json.dumps(calibration, indent=4))

    print("Finished successfully.")
    print("Remove work directory? [y/N]")
    if input().lower() == "y": shutil.rmtree(workPath)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("datasetPath", type=pathlib.Path, help="Recording folder with data.jsonl and video files.")
    p.add_argument("--frameCount", type=int, default=300, help="Target number of frames per video. Smaller is faster but may cause the calibration to fail.")
    p.add_argument("--model", default="radial", help="COLMAP camera model to use.")
    p.add_argument("--dirty", action="store_true", help="Use existing intermediary outputs when found. (Not recommended)")
    p.add_argument("--mapperParameters", default="--Mapper.ba_global_function_tolerance=1e-6", help="COLMAP mapper parameters")
    args = p.parse_args()
    main(args)
