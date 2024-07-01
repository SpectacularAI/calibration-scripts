import cv2
import json
import argparse
import numpy as np

DELETE_LAST_N_RESULTS_WHEN_D_PRESSED = 10

class SaddlePoint:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

# TODO: use cv2.cornerSubPix
class SaddlePointCornerDetector:
    def __init__(self, ksize=3, threshold=100, nms_enabled=True, nms_radius=30):
        self.ksize = ksize # Sobel kernel size
        self.threshold = threshold # Key point response function threshold
        self.nms_enabled = nms_enabled # Enable non-maximum supression (NMS)
        self.nms_radius = nms_radius # NMS radius

    def detect(self, image, plot=False):
        def convert_to_gray_scale_image(image, min_val=None, max_val=None):
            if min_val is None:
                min_val = np.min(image)
            if max_val is None:
                max_val = np.max(image)
            image = np.maximum(0, np.minimum(max_val, image) - min_val) / (max_val - min_val) * 255
            image = image.astype(np.uint8)
            return image

        def apply_nms(keypoints, responses, nms_radius):
            indices = np.argsort(-responses)
            nms_keypoints = []
            for i in indices:
                kp = keypoints[i]
                response = responses[i]
                is_new_key_point = True
                for kp2 in nms_keypoints:
                    kp2 = np.array([kp2.pt[1], kp2.pt[0]])
                    if np.linalg.norm(kp - kp2) < nms_radius:
                        is_new_key_point = False
                        break
                if is_new_key_point:
                    nms_keypoints.append(cv2.KeyPoint(kp[1], kp[0], response))
            return nms_keypoints

        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
        else:
            gray = image

        I_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        I_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=self.ksize)
        I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=self.ksize)
        I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=self.ksize)
        # I_yx = cv2.Sobel(I_y, cv2.CV_64F, 1, 0, ksize=self.ksize)

        p = I_xx * I_yy - I_xy**2
        m = 0.5*(I_xx + I_yy)
        l1 = m + np.sqrt(m**2 - p)
        l2 = m - np.sqrt(m**2 - p)
        response = -np.sign(l1*l2) * np.minimum(np.abs(l1), np.abs(l2))
        corners = response

        keypoints = np.argwhere(corners > self.threshold)
        keypoints = keypoints.astype(np.float32)
        responses = corners[corners > self.threshold]
        if self.nms_enabled:
            keypoints = apply_nms(keypoints, responses, self.nms_radius)
        else:
            keypoints = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in keypoints]

        if plot:
            image = cv2.drawKeypoints(image.copy(), keypoints, None, color=(255, 0, 0), flags=0)
            cv2.imshow('Response', convert_to_gray_scale_image(corners, min_val=0, max_val=500))
            cv2.imshow('Corners', cv2.convertScaleAbs(image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return keypoints

def draw_keypoints(image, keypoints, radius=3, color=(255, 0, 0), thickness=-1):
    for kp in keypoints:
        image = cv2.circle(image, (int(kp.x), int(kp.y)), radius, color, thickness)
    return image

def serialize_checkerboard_corners(frame_id, corners):
    def serialize_corner(corner):
        corner_json = {
            "id" : corner.id,
            "pixel": [corner.x, corner.y]
        }
        return corner_json

    corners_json = []
    for c in corners:
        corners_json.append(serialize_corner(c))

    image_json = {
        "id" : frame_id,
        "points2d": corners_json
    }
    return image_json

def checkerboard_corner_to_id(row, col, rows):
    # NOTE: assumes only tracking inner corners
    return  col * (rows - 1) + row

def checkerboard_id_to_corner(id, rows):
    # NOTE: assumes only tracking inner corners
    row = id % (rows - 1)
    col = int(id / (rows - 1))
    return row, col

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('video', type=str, help='Path to the video file.')
    p.add_argument('--output', type=str, help='Save detected corners to this file (.jsonl)')
    p.add_argument('--start', type=int, default=0, help='Start tracking on this frame')
    p.add_argument('--bottom', type=int, default=5, help='Skip N pixels from bottom (issue where the IR images have some artefacts)')
    p.add_argument("--rows", type=int, default=5, help="Number of rows in the checkerboard")
    p.add_argument("--cols", type=int, default=8, help="Number of columns in the checkerboard")
    p.add_argument('--nms_radius', type=int, default=20, help="Non-maximum supression radius")
    p.add_argument('--corner_threshold', type=float, default=100, help="Corner-detection threshold")
    return p.parse_args()

def fix_frame(image, bottom):
    image[-bottom:, :] = (0, 0, 0)

def predict_checkerboard_corners(bottom_left, bottom_right, top_right, top_left, rows, cols):
    delta_up_left = (top_left - bottom_left) / (rows - 2)
    delta_up_right = (top_right - bottom_right) / (rows - 2)
    delta_right_bottom = (bottom_right - bottom_left) / (cols - 2)

    def lerp(start, stop, w):
        return w * start + (1.0 - w) * stop

    # Detect all checkerboard corners based on the 4 corner points
    corners = []
    for j in range(cols - 1):
        for i in range(rows - 1):
            w = 1.0 - j / (cols - 2)
            id = checkerboard_corner_to_id(i, j, rows)
            xy = bottom_left + delta_right_bottom * j + lerp(delta_up_left, delta_up_right, w) * i
            corners.append(SaddlePoint(id, xy[0], xy[1]))
    return corners

def detect_checkerboard_corners(detector, image, rows, cols):
    keypoints = detector.detect(image)
    keypoints = np.array([SaddlePoint(id, kp.pt[0], kp.pt[1]) for id, kp in enumerate(keypoints)])

    def select_closest_keypoint(keypoints, point, radius=20):
        closest_kp = None
        closest_dist = None

        for kp in keypoints:
            dist = np.sqrt((kp.x - point[0]) ** 2 + (kp.y - point[1]) ** 2)
            if dist > radius: continue
            if closest_kp is None or dist < closest_dist:
                closest_kp = kp
                closest_dist = dist
        return closest_kp

    title = 'Select checkerboard corners in order: bottom-left, bottom-right, top-right, top-left'
    selected_kps = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            kp = select_closest_keypoint(keypoints, (x, y))
            if kp is None: return
            selected_kps.append(kp)
        else: return

        if len(selected_kps) == 0: print("Select BOTTOM LEFT corner")
        if len(selected_kps) == 1: print("Select BOTTOM RIGHT corner")
        if len(selected_kps) == 2: print("Select TOP RIGHT corner")
        if len(selected_kps) == 3: print("Select TOP LEFT corner")

        cv2.imshow(title, draw_keypoints(image_all_keypoints.copy(), selected_kps, color=(0, 255, 0)))

    print("Select BOTTOM LEFT corner")
    image_all_keypoints = draw_keypoints(image.copy(), keypoints, color=(255, 0, 0))
    cv2.imshow(title, image_all_keypoints)
    cv2.setMouseCallback(title, click_event)

    while len(selected_kps) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    bottom_left = np.array([selected_kps[0].x, selected_kps[0].y])
    bottom_right = np.array([selected_kps[1].x, selected_kps[1].y])
    top_right = np.array([selected_kps[2].x, selected_kps[2].y])
    top_left = np.array([selected_kps[3].x, selected_kps[3].y])

    predicted_corners = predict_checkerboard_corners(bottom_left, bottom_right, top_right, top_left, rows, cols)
    corners = []
    used_kp_ids = []
    for predicted in predicted_corners:
        kp = select_closest_keypoint(keypoints, (predicted.x, predicted.y))
        if kp is None: continue
        if kp.id in used_kp_ids: continue # TODO: Handle duplicate matches better...
        used_kp_ids.append(kp.id)
        corners.append(SaddlePoint(predicted.id, kp.x, kp.y))

    title = 'Checkerboard corners [SPACE]=continue, [LEFT-CLICK]=remove closest corner, [RIGHT-CLICK]=remove all)'
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            kp = select_closest_keypoint(corners, (x, y))
            if kp is None: return
            corners.remove(kp)
        elif event == cv2.EVENT_RBUTTONDOWN:
            corners.clear()

        image_checkerboard = image.copy()
        draw_keypoints(image_checkerboard, predicted_corners, 3, (255, 0, 0), 1)
        draw_keypoints(image_checkerboard, corners, 3, (0, 255, 0), -1)
        cv2.imshow(title, image_checkerboard)

    image_checkerboard = image.copy()
    draw_keypoints(image_checkerboard, predicted_corners, 3, (255, 0, 0), 1)
    draw_keypoints(image_checkerboard, corners, 3, (0, 255, 0), -1)
    cv2.imshow(title, image_checkerboard)
    cv2.setMouseCallback(title, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(corners)

def read_frame(capture, frame_number, bottom):
    success, frame = capture.read()
    if success: fix_frame(frame, bottom)
    frame_number += 1
    return success, frame, frame_number

def filter_points_by_border(points, image_shape, border_margin):
    """Filter out points that are within the border margin of the image."""
    if len(points) == 0: return np.array([])
    h, w = image_shape[:2]
    return np.array([1 if border_margin <= p[0] <= w - border_margin and border_margin <= p[1] <= h - border_margin else 0 for p in points])

def filter_points_by_motion(points, new_points, max_deviation):
    """Filter out points that deviate significantly from the median motion."""
    if len(points) == 0: return np.array([])
    motions = new_points - points
    median_motion = np.median(motions, axis=0)
    distances = np.linalg.norm(motions - median_motion, axis=1)
    return np.array([1 if d <= max_deviation else 0 for d in distances])

if __name__ == '__main__':
    args = parse_args()
    capture = cv2.VideoCapture(args.video)

    # Skip frames at start
    frame_number = -1
    for _ in range(args.start):
        success, frame, frame_number = read_frame(capture, frame_number, args.bottom)
        if not success:
            print(f"Failed to read frame number {frame_number}")
            capture.release()
            exit(0)

    # Read the first frame
    success, first_frame, frame_number = read_frame(capture, frame_number, args.bottom)
    if not success:
        print("Failed to capture the first frame")
        capture.release()
        exit(0)

    # Detect saddle point corners in the first frame
    detector = SaddlePointCornerDetector(nms_radius=args.nms_radius, threshold=args.corner_threshold)
    corners = detect_checkerboard_corners(detector, cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), args.rows, args.cols)
    prev_points = np.array([[kp.x, kp.y] for kp in corners], dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    serialized_corners = [serialize_checkerboard_corners(frame_number, corners)]

    should_quit = not capture.isOpened()
    while not should_quit:
        success, frame, frame_number = read_frame(capture, frame_number, args.bottom)
        if not success: break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.shape(prev_points)[0] > 0:
            # Calculate optical flow
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)

            # Filter points by
            # 1) Tracking status
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]
            status = np.squeeze(status) if len(status) > 1 else status[0]
            corners = corners[status == 1]

            # 2) Proximity to image borders
            status = filter_points_by_border(good_new, gray_frame.shape, border_margin=20)
            good_new = good_new[status == 1]
            good_old = good_old[status == 1]
            corners = corners[status == 1]

            # 3) Motion, all points should move roughly in the same direction
            status = filter_points_by_motion(good_new, good_old, max_deviation=5.0)
            good_new = good_new[status == 1]
            good_old = good_old[status == 1]
            corners = corners[status == 1]

            # Update corner positions
            for i in range(len(corners)):
                corners[i].x = float(good_new[i][0])
                corners[i].y = float(good_new[i][1])

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                frame = cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)
                frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 2)
        else:
            good_new = np.array([])
            goold_old = np.array([])
            corners = np.array([])

        serialized_corners.append(serialize_checkerboard_corners(frame_number, corners))

        # Update the previous frame and previous points
        prev_gray = gray_frame.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        should_quit = not capture.isOpened()

        while True:
            cv2.imshow('Tracking result [SPACE]=next, [R]=redetect corners, [D]=delete last N results, [Q]=quit', frame)
            key = cv2.waitKey(0)

            if key == 32: # space (next frame)
                break
            elif key == 114: # 'R' (re-detect corners)
                corners = detect_checkerboard_corners(detector, prev_gray, args.rows, args.cols)
                prev_points = np.array([[kp.x, kp.y] for kp in corners], dtype=np.float32).reshape(-1, 1, 2)
                break
            elif key == 100: # 'D' (delete last N results)
                for _ in range(DELETE_LAST_N_RESULTS_WHEN_D_PRESSED):
                    if len(serialized_corners) > 0: serialized_corners.pop()
                print(f"Deleted last {DELETE_LAST_N_RESULTS_WHEN_D_PRESSED} results")
            elif key == 113: # 'Q' (quit)
                should_quit = True
                break

    capture.release()
    cv2.destroyAllWindows()

    if args.output:
        with open(args.output, 'w') as file:
            for image_corners in serialized_corners:
                json_line = json.dumps(image_corners)
                file.write(json_line + '\n')
