import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt
from calibpy.Camera import Camera
from calibpy.Settings import Settings
from calibpy.Aruco import ArucoTarget, get_aruco_corners, create_aruco_board
from calibpy.Stream import Stream


class Calibration:
    """Calibration class handling intrinsic 
    and extrinsic calibrations of aruco targets.
    """

    def __init__(self, settings: Settings = None):
        """
        :param settings: Settings instance, defaults to None
        :type settings: Settings, optional
        """
        self._settings = None       # Settings instance
        self._aruco_target = None   # Aruco target descriptor
        self._board = None          # Aruco board
        self._board_pts = None      # Board 3D Points
        self._board_pts_ids = None  # Board Ids
        self._visualize = False     # En-/Disables visualization

        if settings is not None:
            self.setup(settings)

    @property
    def visualize(self):
        return self._visualize

    @visualize.setter
    def visualize(self, value: bool):
        assert isinstance(value, bool)
        self._visualize = value

    @staticmethod
    def undistort_image(
            img: np.ndarray,
            intrinsics: np.ndarray,
            distortion: np.ndarray):
        h, w = img.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (w, h), 1, (w, h))
        undistorted = cv2.undistort(
            img, intrinsics, distortion, None, newcameramatrix)
        roi_x, roi_y, roi_w, roi_h = roi
        undistorted = undistorted[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        undistorted = cv2.resize(
            undistorted, (w, h), interpolation=cv2.INTER_AREA)
        return undistorted

    @staticmethod
    def show_image(img: np.ndarray,
                   text: str = "",
                   proportion: int = 1000,
                   duration: int = 0):
        """Shows an image using cv2 imshow

        :param img: Input image
        :type img: np.ndarray
        :param text: Label, defaults to ""
        :type text: str, optional
        :param proportion: Display width, defaults to 1000
        :type proportion: int, optional
        :param duration: show duration, defaults to 0
        :type duration: int, optional
        """
        proportion = max(img.shape) / proportion
        out = cv2.resize(img,
                         (int(img.shape[1] / proportion),
                          int(img.shape[0] / proportion)))
        out = cv2.putText(out, f"{text}",
                          (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('img', out)
        cv2.waitKey(duration)

    def setup(self, settings: Settings):
        """Setting up calibration instance from settings object

        :param settings: Settings instance
        :type settings: Settings
        """
        assert isinstance(settings, Settings)
        self._settings = settings
        self._settings.ensure("aruco_dict", str)
        self._settings.ensure("cols", int)
        self._settings.ensure("rows", int)
        self._settings.ensure("square_size", float)
        self._settings.ensure("marker_size", float)

        self._aruco_target = ArucoTarget.get(self._settings.aruco_dict,
                                             self._settings.cols,
                                             self._settings.rows,
                                             self._settings.square_size,
                                             self._settings.marker_size)

        self._board, self._board_pts, self._board_pts_ids = \
            create_aruco_board(
                self._settings.aruco_dict,
                self._settings.cols,
                self._settings.rows,
                self._settings.square_size,
                self._settings.marker_size)

        self._criteria = (cv2.TERM_CRITERIA_EPS +
                          cv2.TERM_CRITERIA_MAX_ITER,
                          self._settings.max_count,
                          self._settings.epsilon)

    def calibrate_extrinsics(self, stream: Stream, cam: Camera) -> list:
        # self._settings.ensure("min_number_of_corners", int)
        # self._settings.ensure("min_number_of_calibration_images", int)
        cams = []

        image_size = None
        while True:
            # get next image
            img = stream.next()
            if img is None:
                break

            name = Path(stream.current_filename()).name.split(".")[0]

            if image_size is None:
                image_size = img.shape[::-1]

            assert img is not None, "Failed to read image(s)!"

            if cam.intrinsics is None or cam.distortion is None:
                print("Missing internal calibration")
                raise RuntimeError("Calibration Failed!")

            img = Calibration.undistort_image(
                img,
                cam.intrinsics,
                cam.distortion
            )

            # get targets aruco corners
            response, charuco_corners, charuco_ids, corners = \
                get_aruco_corners(img, self._aruco_target, self._criteria)

            if self.visualize:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                vis = cv2.aruco.drawDetectedMarkers(image=vis, corners=corners)
                if response >= self._settings.min_number_of_corners:
                    vis = cv2.aruco.drawDetectedCornersCharuco(
                        image=vis,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids)
                    Calibration.show_image(
                        vis,
                        text=f"{name}",
                        proportion=1000)

            p3d = []
            for id in charuco_ids:
                p3d.append(self._board_pts[id[0]])
            p3d = np.array(p3d)

            success, rvec, tvec = cv2.solvePnP(
                p3d,
                charuco_corners,
                cam.intrinsics,
                cam.distortion,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_ITERATIVE)

            R = cv2.Rodrigues(rvec)[0]
            Rt = np.zeros((4, 4), dtype=np.float32)
            Rt[0:3, 0:3] = R
            Rt[0:3, 3] = np.array([x[0] for x in tvec])
            Rt[3, 3] = 1

            cam_n = Camera.from_cam(cam)
            cam_n.name = name
            cam_n.RT = Rt
            cams.append(cam_n)

        return cams

    def calibrate_intrinsics(self, stream: Stream) -> Camera:
        """calibrate instrinsics from input stream 

        :param stream: Stream instance
        :type stream: Stream
        :raises RuntimeError: if calibration fails a RuntimeError is thrown
        :return: Camera instance
        :rtype: Camera
        """
        frame = 0               # current frame
        accepted_images = 0     # number of images accepted for calibration
        corners_all = []        # corners discovered in all images processed
        ids_all = []            # aruco ids corresponding to corners discovered

        self._settings.ensure("min_number_of_corners", int)
        self._settings.ensure("min_number_of_calibration_images", int)

        min_N = self._settings.min_number_of_calibration_images
        image_size = None

        if stream.length < min_N:
            print(
                f"Cannot apply internal calibration on \
                    less than {min_N} images!")
            raise RuntimeError("Internal Calibration Failed!")
        while True:
            # get next image
            img = stream.next()
            if img is None:
                break

            if image_size is None:
                image_size = img.shape[::-1]

            # get targets aruco corners
            response, charuco_corners, charuco_ids, corners = \
                get_aruco_corners(img, self._aruco_target, self._criteria)

            if self.visualize:
                # outline the aruco markers found in our query image
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                vis = cv2.aruco.drawDetectedMarkers(image=vis, corners=corners)

            # if a Charuco board was found, collect image/corner
            # points requires at least min_response squares for a
            # valid calibration image
            if response >= self._settings.min_number_of_corners:
                accepted_images += 1

                # add these corners and ids to our calibration arrays
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)

                if self.visualize:
                    # draw the Charuco board we've detected to show our
                    # calibrator the board was properly detected
                    vis = cv2.aruco.drawDetectedCornersCharuco(
                        image=vis,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids)
                    Calibration.show_image(
                        vis,
                        text=f"Frame: {str(frame).zfill(5)}",
                        proportion=800,
                        duration=1)

            frame += 1

        if accepted_images < min_N:
            print("Calibration Failed! Found less than {min_N} images")
            raise RuntimeError("Internal Calibration Failed")
        else:
            print(f"{accepted_images} valid captures")

        # calibrate
        results = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=self._board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None)

        rpe = results[0]
        intrinsics = results[1]
        distortion = results[2]
        stdDeviationsIntrinsics = results[5]
        stdDeviationsExtrinsics = results[6]
        perViewErrors = results[7]

        print("Reprojection Error:", rpe)

        if self.visualize:
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            ax0.plot(range(1, len(stdDeviationsIntrinsics)+1),
                     [x[0] for x in stdDeviationsIntrinsics],
                     label="stdDeviationsIntrinsics")
            ax1.plot(range(1, len(stdDeviationsExtrinsics)+1),
                     [x[0] for x in stdDeviationsExtrinsics],
                     label="stdDeviationsExtrinsics")
            ax2.plot(range(1, len(perViewErrors)+1),
                     [x[0] for x in perViewErrors], label="perViewErrors")
            ax0.set_ylabel("stdDeviationsIntrinsics")
            ax1.set_ylabel("stdDeviationsExtrinsics")
            ax2.set_ylabel("perViewErrors")
            ax2.set_xlabel("Frame")
            fig.suptitle(f"Reprojection Error:{rpe}", fontsize=12)
            plt.grid(True)
            plt.show()

        cam = Camera()
        cam.intrinsics = intrinsics
        cam.distortion = distortion
        if "sensor_width_mm" in self._settings \
                and "sensor_height_mm" in self._settings:
            h = self._settings.sensor_height_mm
            w = self._settings.sensor_width_mm
            cam.sensor_size_mm = (h, w)
            cam.image_size = (image_size[1], image_size[0])
            cam.compute_f_mm()
        if "f_mm" in self._settings:
            print(f"Focal Length: {self._settings.f_mm} mm")
            print(f"Focal Length estimated: {cam.f_mm} mm")
            print(
                f"Focal Length Error: {abs(cam.f_mm - self._settings.f_mm)} mm")
        return cam
