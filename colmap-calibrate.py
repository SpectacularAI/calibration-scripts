#!/usr/bin/env python3

"""Use COLMAP to calibrate a Spectacular AI SDK recording. Also requires FFmpeg."""

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
        err = runWithLogging(cmd, "colmap-feature-extractor", videoWorkPath)
        if err is not None: return err

    matchingMethod = "sequential_matcher" # "vocab_tree_matcher" might be better.
    matchingDonePath = videoWorkPath / "matching_done"
    if matchingDonePath.exists():
        print("Skipping feature matching.")
    else:
        print("Running feature matching.")
        cmd = f"{env} colmap {matchingMethod} --database_path {databasePath}"
        err = runWithLogging(cmd, "colmap-matching", videoWorkPath)
        if err is not None: return err
        with open(matchingDonePath, "w") as f:
            f.write(matchingMethod)

    mapperPath = videoWorkPath / "mapper"
    if mapperPath.exists():
        print("Skipping mapping.")
    else:
        print("Running mapping.")
        cmd = f"{env} colmap mapper --database_path {databasePath}"
        cmd += f" --image_path {imagesPath}"
        cmd += f" --output_path {mapperPath}"
        cmd += " --Mapper.ba_global_function_tolerance=1e-6"
        err = runWithLogging(cmd, "colmap-mapper", videoWorkPath)
        if err is not None: return err

    # Never skip this phase.
    print("Converting outputs to text format.")
    textModelPath = videoWorkPath / "text-model"
    textModelPath .mkdir(parents=True, exist_ok=True)
    cmd = f"{env} colmap model_converter --output_type TXT"
    cmd += f" --input_path \"{mapperPath}/0\""
    cmd += f" --output_path \"{textModelPath}\""
    err = runWithLogging(cmd, "colmap-model-converter", videoWorkPath)
    if err is not None: return err

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

    # TODO Convert to Spectacular AI calibration.json format.

    print("Finished successfully.")
    # TODO Enable.
    # print("Remove work directory? [y/N]")
    # if input().lower() == "y": shutil.rmtree(workPath)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("datasetPath", type=pathlib.Path, help="Recording folder with data.jsonl and video files.")
    p.add_argument("--frameCount", type=int, default=300, help="Target number of frames per video. Smaller is faster but may cause the calibration to fail.")
    p.add_argument("--model", default="radial", help="COLMAP camera model to use.")
    p.add_argument("--dirty", action="store_true", help="Use existing intermediary outputs when found. (Not recommended)")
    args = p.parse_args()
    main(args)
