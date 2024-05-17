"""Interactive tool to set fixed rectangular masks on video files"""
# Run with a list of video paths as arguments. Select regions by left-clicking at two
# points of the same video window. Undo the last region by clicking the right or middle
# button. The output is shown on the console.
#
# The output format is: Series of comma-separated numbers, each a set of 4 numbers
# defining a rectangle x0, y0, x1, y1 such that 0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1.
# Series split by `_` apply to different cameras, starting from the first.
# If there is no `_` in the string, then the masks apply to all cameras.
#
# Examples:
#   masks: 0,0,1,1                  (all cameras whole frame)
#   masks: 0,0,0.5,0.5              (all cameras top-left quadrant)
#   masks: 0,0,0.5,0.5_             (first camera top-left)
#   masks: _0,0,0.5,0.5             (second camera top-left)
#   masks: 0,0,0.5,0.5_0.5,0,1,0.5  (first camera top-left, and second camera top-right)
#   masks: 0,0,0.5,0.5,0.5,0,1,0.5  (top-left and top-right in all cameras)

import os
import pathlib

import cv2

shouldPrint = False

def printMasks(label, videos):
    s = f"{label}: "
    for videoInd, video in enumerate(videos):
        if videoInd > 0: s += "_"
        for pointInd in range(0, len(video.points), 2):
            if pointInd + 1 >= len(video.points): break # Incomplete mask.
            if pointInd > 0: s += ","
            x = sorted([video.points[pointInd][0] / video.width, video.points[pointInd + 1][0] / video.width])
            y = sorted([video.points[pointInd][1] / video.height, video.points[pointInd + 1][1] / video.height])
            s += f"{x[0]:.3f},{y[0]:.3f},{x[1]:.3f},{y[1]:.3f}"
    print(s)

class ImageWithMask():
    def __init__(self, filePath):
        self.filePath = filePath
        self.name = os.path.basename(filePath)
        self.image = None
        self.mousePos = None
        self.points = []
        self.restart()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(self.name, flags=cv2.WINDOW_GUI_NORMAL) # Do not show toolbar.

        def mouseCallback(event, x, y, flags, param):
            global shouldPrint
            self.mousePos = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                if len(self.points) % 2 == 0:
                    shouldPrint = True
            elif event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                # Undo.
                if len(self.points) >= 2 and len(self.points) % 2 == 0:
                    self.points = self.points[:-2]
                    shouldPrint = True
                elif len(self.points) >= 1 and len(self.points) % 2 == 1:
                    self.points = self.points[:-1]

        cv2.setMouseCallback(self.name, mouseCallback)

    def restart(self):
        self.cap = cv2.VideoCapture(str(self.filePath))

    def next(self):
        ret, frame = self.cap.read()
        if not ret: return False
        self.image = frame.copy()
        return True

    def show(self):
        image = self.image.copy()
        for pointInd in range(0, len(self.points), 2):
            if pointInd + 1 >= len(self.points): break # Incomplete mask.
            alpha = 0.5
            overlay = image.copy()
            cv2.rectangle(overlay, self.points[pointInd], self.points[pointInd + 1], (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        if len(self.points) % 2 == 1:
            alpha = 0.5
            overlay = image.copy()
            cv2.rectangle(overlay, self.points[pointInd], self.mousePos, (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.imshow(self.name, image)

    def close(self):
        self.cap.release()

def main(args):
    videos = [ImageWithMask(p) for p in args.videoFiles]

    paused = False
    shouldRestart = False
    while True:
        global shouldPrint
        if shouldPrint:
            printMasks(args.label, videos)
            shouldPrint = False

        if not paused:
            for video in videos:
                if not video.next(): shouldRestart = True

        for video in videos:
            video.show()

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r') or shouldRestart:
            for video in videos: video.restart()
            shouldRestart = False
        elif key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break

    for video in videos:
        video.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "videoFiles",
        nargs='+',
        type=pathlib.Path,
        help="List of videos. For example '/path/to/data.mp4 /path/to/data2.mp4'"
    )
    p.add_argument("--label", default="masks", help="Label that will show in the output.")

    args = p.parse_args()
    main(args)
