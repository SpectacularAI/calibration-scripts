import json
import pathlib

import cv2
import numpy as np

def scaled_imshow(args, name, image):
    image = cv2.resize(image, None, fx=args.scale_view, fy=args.scale_view)
    cv2.imshow(name, image)

def refine_corners(image, corners, radius=20, max_change=5, plot=False):
    h, w = image.shape

    def do_refine(c):
        x0 = int(c['pixel'][0] - radius)
        y0 = int(c['pixel'][1] - radius)
        ww = radius*2 + 1
        x1 = x0 + ww
        y1 = y0 + ww
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None

        wnd = image[y0:y1, x0:x1]

        rng = np.arange(0, ww) - radius
        xx, yy = [np.ravel(c) for c in list(np.meshgrid(rng, rng))]

        C = np.sqrt(0.5)
        def checker_image(x, y, gradient_width=4):
            h = gradient_width/2
            vx = np.clip(x / h, -1, 1) * C
            vy = np.clip(y / h, -1, 1) * C
            gx = (np.abs(x) < h) / h * C
            gy = (np.abs(y) < h) / h * C
            return vx * vy + 0.5, gx*vy, gy*vx
        
        def func(x, y, params):
            cx, cy, skew, rot = params
            s, c = np.sin(rot), np.cos(rot)
            skew_mat = np.array([[1, skew], [0, 1]])
            rot_mat = np.array([[c, -s], [s, c]])

            d_skew_mat = np.array([[0, 1], [0, 0]])
            d_rot_mat = np.array([[-s, -c], [c, -s]])

            xy = np.hstack([x[:, None], y[:, None]])
            xy_centered = xy - np.array([[cx, cy]])
            mat_T = skew_mat.T @ rot_mat.T
            xy_transformed = xy_centered @ mat_T
            d_cx = xy*0
            d_cx[:, 0] = -1
            d_cy = xy*0
            d_cy[:, 1] = -1

            d_trans_d_cx = d_cx @ mat_T
            d_trans_d_cy = d_cy @ mat_T
            d_trans_d_skew = xy_centered @ d_skew_mat.T @ rot_mat.T
            d_trans_d_rot = xy_centered @ skew_mat.T @ d_rot_mat.T

            img, d_img_x, d_img_y = checker_image(xy_transformed[:, 0], xy_transformed[:, 1])
            derivatives = [d[:, 0] * d_img_x + d[:, 1] * d_img_y for d in [
                d_trans_d_cx,
                d_trans_d_cy,
                d_trans_d_skew,
                d_trans_d_rot
            ]]

            return img, derivatives

        # remove mean and linear gradient
        normalized_wnd = wnd - np.mean(wnd)
        xx_wnd = xx.reshape(wnd.shape)
        yy_wnd = yy.reshape(wnd.shape)
        normalized_wnd = normalized_wnd - xx_wnd * np.sum(xx_wnd * normalized_wnd) / np.sum(xx_wnd**2)
        normalized_wnd = normalized_wnd - yy_wnd * np.sum(yy_wnd * normalized_wnd) / np.sum(yy_wnd**2)

        # normalize intensity and clip
        normalized_wnd = np.clip(normalized_wnd / np.median(np.abs(normalized_wnd)), -1, 1)
        normalized_wnd = normalized_wnd * 0.5 + 0.5

        wnd = normalized_wnd

        params = np.array([0, 0, 0, 0]).astype(np.float64)
        if np.mean(normalized_wnd[:radius, :radius]) < 0.5: params[3] = np.pi/2

        n_itr = 10
        b = wnd.ravel()
        for _ in range(n_itr):
            img, derivatives = func(xx, yy, params)
            error = img - b

            J = np.vstack(derivatives).T
            JtJ = J.T @ J
            JtE = J.T @ error
            d_params = np.linalg.solve(JtJ, -JtE)
            params += d_params
        
        x, y = params[:2]

        if plot:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(rgb_image, (x0,y0), (x1-1, y1-1), (0xff, 0, 0), 1)
            cv2.imshow('window location', rgb_image)

            zoom = 30
            wnd_img = cv2.resize(cv2.cvtColor((normalized_wnd * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB), (ww*zoom, ww*zoom), interpolation=cv2.INTER_NEAREST)

            cx0 = int((radius + 0.5)*zoom)
            cy0 = int((radius + 0.5)*zoom)
            cx = int((x + radius + 0.5)*zoom)
            cy = int((y + radius + 0.5)*zoom)
            wnd_img = cv2.circle(wnd_img, (cx0, cy0), 2, (0, 0, 0xff), 1)
            wnd_img = cv2.circle(wnd_img, (cx, cy), 3, (0xff, 0, 0), 1)
            cv2.imshow('window', wnd_img)

            fitted_image = func(xx, yy, params)[0].reshape(wnd.shape)
            error_img = fitted_image - wnd

            cv2.imshow('fit', (fitted_image * 255).astype(np.uint8))
            cv2.imshow('err', cv2.applyColorMap((error_img*128 + 128).astype(np.uint8), cv2.COLORMAP_JET))

            cv2.waitKey(0)

        return { 'id': c['id'], 'pixel': [c['pixel'][0]+ x, c['pixel'][1] + y] }

    for c in corners:
        c_orig = c

        if c is not None:
            c_orig = c
            c = do_refine(c)

            if c is None or np.max(np.abs(np.array(c['pixel']) - np.array(c_orig['pixel']))) > max_change:
                c = c_orig

        yield(c)

    if plot: cv2.destroyAllWindows()

def draw_keypoints(image, keypoints, radius=2, color=(255, 0, 0), thickness=-1):
    for kp in keypoints:
        image = cv2.circle(image, (int(kp['pixel'][0]), int(kp['pixel'][1])), radius, color, thickness)
    return image

def read_checkerboard_corners(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [json.loads(line) for line in lines]

def checkerboard_corner_to_id(row, col, rows):
    # NOTE: assumes only tracking inner corners
    return  col * (rows - 1) + row

def checkerboard_id_to_corner(id, rows):
    # NOTE: assumes only tracking inner corners
    row = id % (rows - 1)
    col = int(id / (rows - 1))
    return row, col

def read_frame(capture, frame_number):
    success, frame = capture.read()
    frame_number += 1
    return success, frame, frame_number

def filter_points_by_border(points, image_shape, border_margin):
    """Filter out points that are within the border margin of the image."""
    if len(points) == 0: return np.array([])
    h, w = image_shape[:2]
    return np.array([1 if border_margin <= p[0] <= w - border_margin and border_margin <= p[1] <= h - border_margin else 0 for p in points])

def find_approximate_square_size_pixels(corners):
    prev_corner = None
    min_consec_dist2 = None

    for c in sorted(corners, key=lambda c: c['id']):
        if prev_corner is not None and prev_corner['id'] + 1 == c['id']:
            d2 = np.sum((np.array(c['pixel']) - np.array(prev_corner['pixel']))**2)
            if min_consec_dist2 is None or d2 < min_consec_dist2:
                min_consec_dist2 = d2
        prev_corner = c
    
    if min_consec_dist2 is None:
        return None
    
    return np.sqrt(min_consec_dist2)

MAIN_WINDOW_TITLE = 'Corner refinement'

def main(args):
    if not pathlib.Path(args.video).exists():
        raise RuntimeError("Video file does not exist:" + args.video)
    
    capture = cv2.VideoCapture(args.video)

    if not capture.isOpened():
        raise RuntimeError("Could not read video:" + args.video)

    if args.output and args.output.exists() and not args.overwrite_without_prompt:
        print(f"Output file `{args.output.name}` already exists and will be over-written. Continue? [y/N]")
        if input().lower() != "y": return

    # Skip frames at start
    frame_number = -1
    input_corners = read_checkerboard_corners(args.corners)
    output_corners = []

    should_quit = False
    debug_next = False
    refine_radius = args.max_refine_radius
    while not should_quit:
        success, frame, frame_number = read_frame(capture, frame_number)
        if not success: break

        while len(input_corners) > 0 and input_corners[0]["id"] < frame_number:
            input_corners.pop(0)
        
        corners = input_corners[0]

        if corners["id"] > frame_number or len(corners['points2d']) == 0:
            continue

        coarse_corners = corners['points2d']
        sq_size = find_approximate_square_size_pixels(coarse_corners)
        if sq_size is None or sq_size < 2:
            print(f'frame {frame_number}: could not determine square size')
            if refine_radius is None: continue
        else:
            refine_radius = min(int(sq_size/2), args.max_refine_radius)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_corners = list(refine_corners(gray_frame, coarse_corners, radius=refine_radius, plot=args.plot_debug or debug_next))

        debug_next = False
        if args.plot:
            im = frame.copy()
            #im = draw_keypoints(im, coarse_corners, color=(255, 0, 0))
            im = draw_keypoints(im, new_corners, color=(0, 255, 0))
            scaled_imshow(args, MAIN_WINDOW_TITLE, im)
            while True:
                key = cv2.waitKey(0)
                if key == 32: # space, next frame
                    break
                elif key == ord('q'): # quit
                    should_quit = True
                    break
                elif key == ord('d'):
                    debug_next = True
                    break
        
        corners['points2d'] = new_corners
        output_corners.append(corners)

    capture.release()
    if args.plot:
        cv2.destroyAllWindows()

    if args.output:
        with open(args.output, 'w') as file:
            for image_corners in output_corners:
                json_line = json.dumps(image_corners)
                file.write(json_line + '\n')
        print("Saved corners.")

if __name__ == '__main__':
    def parse_args():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('video', type=str, help='Path to the video file.')
        p.add_argument('corners', type=pathlib.Path, help='Load coarse detected corners from this file (.jsonl)')
        p.add_argument('--max_refine_radius', default=10, type=int, help='Maximum refinement window radius')
        p.add_argument('--output', type=pathlib.Path, help='Save detected corners to this file (.jsonl)')
        p.add_argument('--plot', action='store_true', help='Plot the detected corners.')
        p.add_argument('--plot_debug', action='store_true', help='Plot each refinement window')
        p.add_argument('-y', '--overwrite_without_prompt', action='store_true', help='Overwrite output without prompting')
        p.add_argument('--scale_view', type=float, default=2.0,
            help="Images larger on screen, does not affect eg corner detection and tracking")

        return p.parse_args()
    main(parse_args())
