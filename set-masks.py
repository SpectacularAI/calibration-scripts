import cv2

WIDTH = None
HEIGHT = None

def print_rectangular_mask(points):
    global WIDTH, HEIGHT
    x0 = points[0][0] / WIDTH
    x1 = points[1][0] / WIDTH
    y0 = points[0][1] / HEIGHT
    y1 = points[1][1] / HEIGHT
    print(f"rectangularMasks: {x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}")

points = []
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2: points = []
        points.append((x, y))
        if len(points) == 2: print_rectangular_mask(points)

class ImageWithMask():
    def __init__(self, filename, name):
        global WIDTH, HEIGHT
        self.cap = cv2.VideoCapture(filename)
        self.name = name
        self.image = None
        WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, click_event)

    def next(self):
        ret, frame = self.cap.read()
        if not ret: return False
        self.image = frame.copy()
        return True

    def show(self, points):
        image = self.image.copy()
        if len(points) == 2:
            alpha = 0.5
            overlay = image.copy()
            cv2.rectangle(overlay, points[0], points[1], (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.imshow(self.name, image)

    def close(self):
        self.cap.release()

def parseArgs():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("first", help="Path to first video")
    p.add_argument("second", help="Path to second video")
    return p.parse_args()

if __name__ == '__main__':
    args = parseArgs()

    first = ImageWithMask(args.first, "first")
    second = ImageWithMask(args.second, "second")

    paused = False
    while True:
        if not paused:
            if not first.next(): break
            if not second.next(): break

        first.show(points)
        second.show(points)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break

    first.close()
    second.close()
    cv2.destroyAllWindows()
