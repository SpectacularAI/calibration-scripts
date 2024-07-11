import json
import pathlib

import cv2
import numpy as np

DELETE_LAST_N_RESULTS_WHEN_D_PRESSED = 10

MAIN_WINDOW = "Guided checkerboard detection"

class SaddlePoint:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

def scaled_imshow(args, name, image):
    image = cv2.resize(image, None, fx=args.scale_view, fy=args.scale_view)
    cv2.imshow(name, image)

def set_scaled_mouse_callback(args, name, click_event):
    def wrapped_click_event(event, x, y, flags, param):
        x = x / args.scale_view
        y = y / args.scale_view
        click_event(event, x, y, flags, param)
    cv2.setMouseCallback(name, wrapped_click_event)

def refine_corners(args, image, corners, radius=2, refine_itr=2, plot=False):
    h, w = image.shape

    def do_refine(c):
        x0 = int(c.x - radius)
        y0 = int(c.y - radius)
        ww = radius*2 + 1
        x1 = x0 + ww
        y1 = y0 + ww
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None

        wnd = image[y0:y1, x0:x1]

        rng = np.arange(0, ww)
        xx, yy = [np.ravel(c) for c in list(np.meshgrid(rng, rng))]

        A = np.vstack([np.ones_like(xx), xx, yy, xx*yy, xx**2, yy**2]).T
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, np.ravel(wnd), rcond=None)

        rhs = [-coeffs[1], -coeffs[2]]
        lhs = [
            [2*coeffs[4], coeffs[3]],
            [coeffs[3], 2*coeffs[5]]
        ]

        MAX_COND = 1e4
        if np.linalg.cond(lhs) > MAX_COND:
            return None

        # print(lhs, rhs)
        sol = np.linalg.solve(lhs, rhs)

        x, y = [p - radius for p in sol]
        #print(x,y)

        # grad_x = coeffs[1] + coeffs[3]*yy + 2*coeffs[4]*xx == 0
        # grad_y = coeffs[2] + coeffs[3]*xx + 2*coeffs[5]*yy == 0
        # --> yy = -(coeffs[1] - 2*coeffs[4]*xx)/coeffs[3]
        # --> coeffs[2] + coeffs[3]*xx - 2*coeffs[5]*(coeffs[1] - 2*coeffs[4]*xx)

        if plot:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(rgb_image, (x0,y0), (x1-1, y1-1), (0xff, 0, 0), 1)
            scaled_imshow(args, 'window location', rgb_image)

            zoom = 30
            wnd_img = cv2.resize(cv2.cvtColor(wnd, cv2.COLOR_GRAY2RGB), (ww*zoom, ww*zoom), interpolation=cv2.INTER_NEAREST)
            cx = int((x + radius + 0.5)*zoom)
            cy = int((y + radius + 0.5)*zoom)

            cx0 = int((radius + 0.5)*zoom)
            cy0 = int((radius + 0.5)*zoom)

            wnd_img = cv2.circle(wnd_img, (cx0, cy0), 2, (0, 0, 0xff), 1)
            wnd_img = cv2.circle(wnd_img, (cx, cy), 3, (0xff, 0, 0), 1)
            scaled_imshow(args, 'window', wnd_img)

            fitted_image = (coeffs[0] + coeffs[1]*xx + coeffs[2]*yy + coeffs[3]*xx*yy + coeffs[4]*xx**2 + coeffs[5]*yy**2).reshape(wnd.shape)
            error_img = fitted_image - wnd
            scaled_imshow(args, 'fit', fitted_image.astype(np.uint8))
            scaled_imshow(args, 'err', cv2.applyColorMap((np.arctan(error_img)*10 + 128).astype(np.uint8), cv2.COLORMAP_JET))

            cv2.waitKey(0)

        return SaddlePoint(c.id, c.x + x, c.y + y)

    for c in corners:
        c_orig = c
        for _ in range(refine_itr):
            c1 = do_refine(c)
            if c1 is None: break
            c = c1

        MAX_CHANGE = 2.5
        if max(abs(c.x - c_orig.x), abs(c.y - c_orig.y)) > MAX_CHANGE:
            yield(c_orig)
        else:
            yield(c)

    if plot: cv2.destroyAllWindows()

class SaddlePointCornerDetector:
    def __init__(self, detector="sobel", ksize=3, threshold=100, nms_enabled=True, nms_radius=30, debug=False):
        assert ksize % 2 == 1, "kernel size must be odd"
        self.detector = detector
        self.ksize = ksize # Sobel kernel size
        self.threshold = threshold # Key point response function threshold
        self.nms_enabled = nms_enabled # Enable non-maximum supression (NMS)
        self.nms_radius = nms_radius # NMS radius
        self.kernel = self.__create_kernel_custom(ksize) if detector == "custom" else None
        self.debug = debug

    """
    Returns kernel of this type:
        [-1,  0,  1]
        [ 0,  0,  0]
        [ 1,  0, -1]
    """
    def __create_kernel_custom(self, N):
        kernel = np.zeros((N, N), dtype=int)
        center = N // 2

        for i in range(N):
            for j in range(N):
                if i < center and j < center:
                    kernel[i, j] = -1
                elif i > center and j > center:
                    kernel[i, j] = -1
                elif i < center and j > center:
                    kernel[i, j] = 1
                elif i > center and j < center:
                    kernel[i, j] = 1
        return kernel

    def detect(self, image):
        def convert_response_to_gray_scale_image(response, min_val=None, max_val=None):
            response[response < self.threshold] = 0
            if min_val is None:
                min_val = np.min(response)
            if max_val is None:
                max_val = np.max(response)
            print(min_val, max_val)
            response = np.maximum(0, np.minimum(max_val, response) - min_val) / (max_val - min_val) * 255
            return response.astype(np.uint8)

        def apply_nms(keypoints, responses, nms_radius):
            if len(keypoints) == 0: return []

            # Sort keypoints by response in descending order
            indices = np.argsort(-responses)
            sorted_keypoints = keypoints[indices]
            sorted_responses = responses[indices]

            # To keep track of suppressed keypoints
            suppressed = np.zeros(len(keypoints), dtype=bool)
            nms_keypoints = []

            for i in range(len(sorted_keypoints)):
                if suppressed[i]:
                    continue

                kp = sorted_keypoints[i]
                nms_keypoints.append(cv2.KeyPoint(kp[1], kp[0], sorted_responses[i]))

                # Calculate distances to remaining keypoints
                dists = np.sqrt(np.sum((sorted_keypoints[i+1:] - kp) ** 2, axis=1))

                # Suppress all keypoints within the nms_radius
                suppressed[i+1:] = suppressed[i+1:] | (dists < nms_radius)

            return nms_keypoints

        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
        else:
            gray = image

        def detect_corners_sobel():
            I_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
            I_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
            I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=self.ksize)
            I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=self.ksize)
            I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=self.ksize)

            p = I_xx * I_yy - I_xy**2
            m = 0.5*(I_xx + I_yy)
            l1 = m + np.sqrt(m**2 - p)
            l2 = m - np.sqrt(m**2 - p)

            response = -np.sign(l1*l2) * np.minimum(np.abs(l1), np.abs(l2))
            keypoints = np.argwhere(response > self.threshold)
            keypoints = keypoints.astype(np.float32)
            keypoint_responses = response[response > self.threshold]
            return response, keypoints, keypoint_responses

        def detect_corners_sobel_simple():
            I_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
            I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=self.ksize)

            response = np.abs(I_xy)
            keypoints = np.argwhere(response > self.threshold)
            keypoints = keypoints.astype(np.float32)
            keypoint_responses = response[response > self.threshold]
            return response, keypoints, keypoint_responses

        def detect_corners_harris():
            response = cv2.cornerHarris(np.float32(gray), blockSize=7, ksize=self.ksize, k=0.04)
            keypoints = np.argwhere(response > self.threshold)
            keypoints = keypoints.astype(np.float32)
            keypoint_responses = response[response > self.threshold]
            return response, keypoints, keypoint_responses

        def detect_corners_custom():
            response = cv2.filter2D(gray, cv2.CV_64F, self.kernel)
            response = np.abs(response)
            keypoints = np.argwhere(response > self.threshold)
            keypoints = keypoints.astype(np.float32)
            keypoint_responses = response[response > self.threshold]
            return response, keypoints, keypoint_responses

        if self.detector == "sobel":
            response, keypoints, keypoint_responses = detect_corners_sobel()
        elif self.detector == "sobel_simple":
            response, keypoints, keypoint_responses = detect_corners_sobel_simple()
        elif self.detector == "harris":
            response, keypoints, keypoint_responses = detect_corners_harris()
        elif self.detector == "custom":
            response, keypoints, keypoint_responses = detect_corners_custom()
        else:
            print(f"Invalid detector: {self.detector}")
            exit(0)

        if self.nms_enabled:
            keypoints = apply_nms(keypoints, keypoint_responses, self.nms_radius)
        else:
            keypoints = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in keypoints]

        if self.debug:
            image_with_keypoints = cv2.drawKeypoints(image.copy(), keypoints, None, color=(255, 0, 0), flags=0)
            scaled_imshow(args, 'Response', convert_response_to_gray_scale_image(response, min_val=0, max_val=500))
            scaled_imshow(args, 'Image with keypoints', cv2.convertScaleAbs(image_with_keypoints))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return keypoints

def draw_keypoints(image, keypoints, radius=2, color=(255, 0, 0), thickness=-1):
    for kp in keypoints:
        image = cv2.circle(image, (int(kp.x), int(kp.y)), radius, color, thickness)
    return image

def draw_tracks(image, good_new, good_old):
    for _, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        image = cv2.circle(image, (a, b), 2, (0, 0, 255), -1)
        image = cv2.line(image, (a, b), (c, d), (255, 0, 0), 1)
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

def fix_frame(image, margin):
    if margin <= 0: return
    image[:margin, :] = (0, 0, 0)
    image[-margin:, :] = (0, 0, 0)
    image[:, :margin] = (0, 0, 0)
    image[:, -margin:] = (0, 0, 0)

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

def detect_checkerboard_corners(args, detector, image):
    refine = not args.no_refine
    rows = args.rows
    cols = args.cols
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray_image)
    keypoints = np.array([SaddlePoint(id, kp.pt[0], kp.pt[1]) for id, kp in enumerate(keypoints)])

    title = 'Select checkerboard corners in order: bottom-left, bottom-right, top-right, top-left. [SPACE]=skip image'
    print(title)
    selected_kps = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            kp = select_closest_keypoint(keypoints, (x, y))
            if kp is None: return
            selected_kps.append(kp)
        else: return
        scaled_imshow(args, MAIN_WINDOW, draw_keypoints(image_all_keypoints.copy(), selected_kps, color=(0, 0, 255)))

    image_all_keypoints = draw_keypoints(image.copy(), keypoints, color=(255, 255, 0))
    scaled_imshow(args, MAIN_WINDOW, image_all_keypoints)
    set_scaled_mouse_callback(args, MAIN_WINDOW, click_event)
    cv2.setWindowTitle(MAIN_WINDOW, title)

    while len(selected_kps) < 4:
        key = cv2.waitKey(1)
        if key == 32: # space (skip frame)
            return np.array([])

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

    title = '[SPACE]=continue, [LEFT-CLICK]=remove closest, [RIGHT-CLICK]=remove all'
    print(title)
    cv2.setWindowTitle(MAIN_WINDOW, title)
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            kp = select_closest_keypoint(corners, (x, y))
            if kp is None: return
            corners.remove(kp)
        elif event == cv2.EVENT_RBUTTONDOWN:
            corners.clear()
        else: return

        image_checkerboard = image.copy()
        draw_keypoints(image_checkerboard, predicted_corners, 3, (255, 0, 0), 1)
        draw_keypoints(image_checkerboard, corners, 3, (0, 255, 0), -1)
        scaled_imshow(args, MAIN_WINDOW, image_checkerboard)

    unrefined_corners = None
    if refine:
        unrefined_corners = corners[:]
        for i, c in enumerate(refine_corners(args, gray_image, corners)):
            corners[i] = c

    image_checkerboard = image.copy()
    draw_keypoints(image_checkerboard, predicted_corners, 3, (255, 0, 0), 1)
    if unrefined_corners is not None:
        draw_keypoints(image_checkerboard, unrefined_corners, 3, (0, 0, 255), 1)
    draw_keypoints(image_checkerboard, corners, 3, (0, 255, 0), -1)
    scaled_imshow(args, MAIN_WINDOW, image_checkerboard)
    set_scaled_mouse_callback(args, MAIN_WINDOW, click_event)
    cv2.waitKey(0)

    return np.array(corners)

def read_frame(capture, frame_number, margin):
    success, frame = capture.read()
    if success: fix_frame(frame, margin)
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

def main(args):
    if not pathlib.Path(args.video).exists():
        print("Video file does not exist:", args.video)
        exit(0)
    capture = cv2.VideoCapture(args.video)

    # Skip frames at start
    frame_number = -1
    for _ in range(args.start):
        success, frame, frame_number = read_frame(capture, frame_number, args.margin)
        if not success:
            print(f"Failed to read frame number {frame_number}")
            capture.release()
            exit(0)

    # Read the first frame
    success, first_frame, frame_number = read_frame(capture, frame_number, args.margin)
    if not success:
        print("Failed to capture the first frame")
        capture.release()
        exit(0)

    # Detect saddle point corners in the first frame
    detector = SaddlePointCornerDetector(
        detector=args.detector,
        ksize=args.detector_ksize,
        nms_radius=args.detector_nms_radius,
        threshold=args.detector_threshold,
        debug=args.detector_debug)
    corners = detect_checkerboard_corners(args, detector, first_frame)
    prev_points = np.array([[kp.x, kp.y] for kp in corners], dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    serialized_corners = [serialize_checkerboard_corners(frame_number, corners)]

    should_quit = not capture.isOpened()
    while not should_quit:
        success, frame, frame_number = read_frame(capture, frame_number, args.margin)
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

            if not args.no_refine_after_track:
                for i, c in enumerate(refine_corners(args, gray_frame, corners)):
                    corners[i] = c
        else:
            good_new = np.array([])
            good_old = np.array([])
            corners = np.array([])

        prev_gray = gray_frame.copy()
        prev_points = good_new.reshape(-1, 1, 2)

        title = '[SPACE]=next, [R]=redetect corners, [D]=delete last N results, [Q]=quit'
        cv2.setWindowTitle(MAIN_WINDOW, title)
        def click_event(event, x, y, flags, param):
            nonlocal corners, prev_points, good_new, good_old
            if event == cv2.EVENT_LBUTTONDOWN:
                kp = select_closest_keypoint(corners, (x, y))
                if kp is None: return
                idx = np.where(corners == kp)[0][0]
                corners = np.delete(corners, idx)
                good_new = np.delete(good_new, idx, axis=0)
                good_old = np.delete(good_old, idx, axis=0)
            elif event == cv2.EVENT_RBUTTONDOWN:
                good_new = np.array([])
                good_old = np.array([])
                corners = np.array([])
            else: return
            prev_points = good_new.reshape(-1, 1, 2)
            scaled_imshow(args, MAIN_WINDOW, draw_tracks(frame.copy(), good_new, good_old))

        scaled_imshow(args, MAIN_WINDOW, draw_tracks(frame.copy(), good_new, good_old))
        set_scaled_mouse_callback(args, MAIN_WINDOW, click_event)
        while True:
            key = cv2.waitKey(0)

            if key == 32: # space, next frame
                break
            elif key == ord('r'): # re-detect corners
                corners = detect_checkerboard_corners(args, detector, frame)
                prev_points = np.array([[kp.x, kp.y] for kp in corners], dtype=np.float32).reshape(-1, 1, 2)
                break
            elif key == ord('d'): # delete last N results
                for _ in range(DELETE_LAST_N_RESULTS_WHEN_D_PRESSED):
                    if len(serialized_corners) > 0: serialized_corners.pop()
                good_new = np.array([])
                good_old = np.array([])
                corners = np.array([])
                prev_points = np.array([])
                scaled_imshow(args, MAIN_WINDOW, draw_tracks(frame.copy(), good_new, good_old))
                print(f"Deleted last {DELETE_LAST_N_RESULTS_WHEN_D_PRESSED} results")
            elif key == ord('q'): # quit
                should_quit = True
                break

        serialized_corners.append(serialize_checkerboard_corners(frame_number, corners))

    capture.release()
    cv2.destroyAllWindows()

    if args.output:
        with open(args.output, 'w') as file:
            for image_corners in serialized_corners:
                json_line = json.dumps(image_corners)
                file.write(json_line + '\n')

if __name__ == '__main__':
    def parse_args():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('video', type=str, help='Path to the video file.')
        p.add_argument('--output', type=str, help='Save detected corners to this file (.jsonl)')
        p.add_argument('--start', type=int, default=0, help='Start tracking on this frame')
        p.add_argument('--margin', type=int, default=5, help='Mask N pixels from edges of the images (issue where the IR images have some artefacts)')
        p.add_argument("--rows", type=int, default=5, help="Number of rows in the checkerboard")
        p.add_argument("--cols", type=int, default=8, help="Number of columns in the checkerboard")
        p.add_argument('--detector', choices=['sobel', 'sobel_simple', 'harris', 'custom'], default='sobel', help='Corner detector type')
        p.add_argument('--detector_threshold', type=float, default=100, help="Corner-detection threshold")
        p.add_argument('--detector_ksize', type=int, default=3, help="Corner detector kernel size (must be odd)")
        p.add_argument('--detector_nms_radius', type=int, default=20, help="Detector non-maximum supression radius (pixels)")
        p.add_argument('--detector_debug', action="store_true", help="Enable additional detector plots")
        p.add_argument('--no_refine', action='store_true', help="Do not refine corner points after detection")
        p.add_argument('--no_refine_after_track', action='store_true', help="Do not refine corner points after tracking")
        p.add_argument('--scale_view', type=float, default=2.0, help="Images larger on screen, does not affect eg corner detection and tracking")
        return p.parse_args()
    main(parse_args())
