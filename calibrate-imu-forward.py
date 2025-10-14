"""
Calibrate imuForward by replaying an SDK recording and prompting the user to
click on a point that horizontally corresponds to the forward direction."
"""
import spectacularAI
import cv2
import numpy as np
import argparse
import sys
import os
import json

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        zoom_factor = param['zoom_factor']
        
        original_x = int(x / zoom_factor)
        original_y = int(y / zoom_factor)
        
        clicked_point = (original_x, original_y)
        
        print(f"Clicked at ({x}, {y}) on zoomed image.")
        print(f"   -> Corresponding pixel on original image: ({original_x}, {original_y})")
        
        if param['confirm']:
            image = param['image']
            window_name = param['window_name']
            cv2.circle(image, (x, y), 5, (0, 255, 0), 1)
            cv2.imshow(window_name, image)
        else:
            cv2.destroyAllWindows()

class RayApp:
    def __init__(self, args):
        self.args = args
        self.vio_output_counter = 0
        self.ray_computed = False
        self.replay = spectacularAI.Replay(
            args.sdk_recording_path,
            ignoreFolderConfiguration=True,
            configuration={'useStereo': False})
        
        with open(os.path.join(args.sdk_recording_path, 'calibration.json')) as f:
            self.calibration_json = json.load(f)
        
        assert len(self.calibration_json['cameras']) == 1
        self.imuToCam = np.array(self.calibration_json['cameras'][0]['imuToCamera'])
        
        self.replay.setExtendedOutputCallback(self.on_vio_output)
        self.replay.setPlaybackSpeed(-1)

    def on_vio_output(self, vio_output, frames):
        """
        Callback function that gets called for each VIO output from the replay.
        """
        if self.ray_computed:
            return

        self.vio_output_counter += 1

        # Skip outputs if requested
        if self.vio_output_counter <= self.args.skip_outputs:
            if self.vio_output_counter % 100 == 0:
                 print(f"Skipped {self.vio_output_counter} outputs...")
            return

        primary_frame = None
        for frame in frames:
            if frame.image is not None:
                primary_frame = frame
                break
        if primary_frame is None:
            print("No frame")
            return
                
        image = primary_frame.image.toArray()

        # --- Get Pixel from User Click (with zoom) ---
        zoom_factor = self.args.zoom
        if zoom_factor < 0.1:
            print("Warning: Zoom factor is very small, clamping to 0.1.")
            zoom_factor = 0.1

        if zoom_factor != 1.0:
            display_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        else:
            display_image = image

        window_name = "Select a point, then press any key"
        cv2.namedWindow(window_name)
        callback_params = {
            'image': display_image,
            'window_name': window_name,
            'zoom_factor': zoom_factor,
            'confirm': self.args.show_and_confirm_point,
        }
        cv2.setMouseCallback(window_name, mouse_callback, callback_params)

        print("Please click on a point in the image to select it.")
        print("Press any key in the image window to continue...")
        cv2.imshow(window_name, display_image)
        cv2.waitKey(0)
        if self.args.show_and_confirm_point:
            cv2.destroyAllWindows()
        
        self.ray_computed = True # Mark as done to prevent re-triggering

        if clicked_point is None:
            print("No point was selected. Exiting.")
            return
            
        main_camera = frame.cameraPose.camera
        ray = main_camera.pixelToRay(spectacularAI.PixelCoordinates(*clicked_point))
        if ray is None:
            print("pixelToRay failed (outside valid FoV?)")
            return

        camToImu = self.imuToCam[:3,:3].transpose()
        self.rayImu = camToImu @ [ray.x, ray.y, ray.z]

        if self.args.use_gravity or self.args.store_imu_to_output_frd:
            camToWorld = frame.cameraPose.getCameraToWorldMatrix()
            imuToWorld = camToWorld[:3, :3] @ self.imuToCam[:3,:3]
            worldToImu = imuToWorld[:3, :3].transpose()
            downVectorWorld = [0,0,-1]
            downVectorImu = worldToImu @ downVectorWorld
            rightVectorImu = np.cross(downVectorImu, self.rayImu)
            forwardVectorImu = np.cross(rightVectorImu, downVectorImu)

            self.frd = np.eye(4)
            self.frd[:3,:3] = np.hstack([v[:, np.newaxis] / np.linalg.norm(v) for v in [forwardVectorImu, rightVectorImu, downVectorImu]])
            if self.args.use_gravity:
                self.rayImu = self.frd[:3, 0]

    def displayResults(self):
        print("\n" + "="*45)
        print("Camera Ray in World Coordinates")
        print("="*45)
        print(f"Original Pixel Coords: {clicked_point}")
        print(f"Ray Direction (IMU):   {np.array2string(self.rayImu, precision=4)}")
        print("="*45)
        print('Updated calibration.json:\n')
        self.calibration_json['imuForward'] = self.rayImu.tolist()
        if self.args.store_imu_to_output_frd:
            self.calibration_json['imuToOutput'] = self.frd.tolist()
        print(json.dumps(self.calibration_json, indent=2))

    def run(self):
        self.replay.runReplay()
        print("\nReplay finished.")
        self.displayResults()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("sdk_recording_path", help="Path to the Spectacular AI SDK recording directory.")
    parser.add_argument(
        "--skip-outputs",
        type=int,
        default=0,
        help="Optional: Number of VIO outputs to skip before displaying an image. Default is 0."
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Optional: Zoom factor for the image selection window. E.g., 2.0 for 2x zoom. Default is 1.0."
    )
    parser.add_argument(
        '-c', '--show_and_confirm_point',
        action='store_true',
        help='Show selected point on the image and confirm it with SPACE')
    parser.add_argument(
        '-g', '--use_gravity',
        action='store_true',
        help='Use gravity direction to refine the result')
    parser.add_argument(
        '-frd', '--store_imu_to_output_frd',
        action='store_true',
        help='Compute approximate IMU-to-output in the Front-Right-Down convention using gravity')

    args = parser.parse_args()

    app = RayApp(args)
    app.run()

if __name__ == "__main__":
    main()
