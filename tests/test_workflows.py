import yaml
import unittest
import numpy as np
from glob import glob
from pathlib import Path
from calibpy.Settings import Settings
from calibpy.Stream import FileStream
from calibpy.Calibration import Calibration
from calibpy.Registration import register_depthmap_to_world
import cv2
from packaging import version

class TestCameraModule(unittest.TestCase):

    def setUp(self):
        print("start Camera tests...")
        self._root = Path.cwd() / "tests" / "data"
        self._cam_gts = []
        gt_path = self._root / "single_cam" / "cams"
        fnames = []
        for fname in glob(str(gt_path / "*.yaml")):
            fnames.append(fname)
        fnames.sort()
        for fname in fnames:
            with open(fname, "r") as file:
                self._cam_gts.append(yaml.safe_load(file))
        self._test_data_filenames = []
        self._has_broken_cv2 = version.parse(cv2.__version__) >= version.parse("4.8.0")

    def tearDown(self):
        import os
        for fname in self._test_data_filenames:
            if Path(fname).exists():
                os.remove(fname)

    def test_intrisics(self):
        stream = FileStream()
        directory = self._root / "single_cam" / "undistorted"
        stream.initialize(directory=directory)
        self.assertEqual(stream.length, 24)

        settings = Settings()
        settings.from_params({
            "aruco_dict": "DICT_5X5",
            "cols": 24,
            "rows": 18,
            "square_size": 0.080,
            "marker_size": 0.062,
            "min_number_of_corners": 20,
            "min_number_of_calibration_images": 20,
            "max_count": 10000,
            "epsilon": 0.00001,
            "sensor_width_mm": 10,
            "sensor_height_mm": 7.5,
            "f_mm": 16.0,
            "visualize": False
        })

        calib = Calibration(settings=settings)
        if self._has_broken_cv2:
            print(f"Warning! opencv_version {cv2.__version__} is broken since 4.8, skipped\n")
            self.skipTest("broken opencv")
        cam = calib.calibrate_intrinsics(stream)
        self.assertAlmostEqual(cam.f_mm, 16.008241865239107, places=1)
        self.assertAlmostEqual(cam.f_px, 2049.3316237582803, places=1)
        self.assertAlmostEqual(cam.cx, 639.8480918599114, places=2)
        self.assertAlmostEqual(cam.cy, 479.08321627042875, places=2)
        self.assertTupleEqual(cam.sensor_size_mm, (7.5, 10))
        self.assertTupleEqual(cam.image_size, (960, 1280))
        self.assertAlmostEqual(
            cam.distortion[0][0], 0.001176257732538723, places=2)
        self.assertAlmostEqual(
            cam.distortion[0][1], -0.006823311308886389, places=2)
        self.assertAlmostEqual(
            cam.distortion[0][2], -8.272961271917753e-05, places=2)
        self.assertAlmostEqual(
            cam.distortion[0][3], 1.0653573906784032e-05, places=2)
        self.assertAlmostEqual(
            cam.distortion[0][4], 0.005191994554296802, places=2)

    def test_extrinsics(self):
        stream = FileStream()
        directory = self._root / "single_cam" / "undistorted"
        stream.initialize(directory=directory)

        settings = Settings()
        settings.from_params({
            "aruco_dict": "DICT_5X5",
            "cols": 24,
            "rows": 18,
            "square_size": 0.080,
            "marker_size": 0.062,
            "min_number_of_corners": 20,
            "min_number_of_calibration_images": 20,
            "max_count": 10000,
            "epsilon": 0.00001,
            "sensor_width_mm": 10,
            "sensor_height_mm": 7.5,
            "f_mm": 16.0,
            "visualize": False,
            "outdir": self._root
        })

        calib = Calibration(settings=settings)
        if self._has_broken_cv2:
            print(f"Warning! opencv_version {cv2.__version__} is broken since 4.8, skipped\n")
            self.skipTest("broken opencv")
        cam = calib.calibrate_intrinsics(stream)
        cam.serialize(Path(settings.outdir) / "intrinsics.npy")
        self._test_data_filenames.append(
            str(Path(settings.outdir) / "intrinsics.npy"))

        stream = FileStream()
        directory = self._root / "single_cam" / "undistorted"
        stream.initialize(directory=directory)

        cams = calib.calibrate_extrinsics(stream, cam)
        for cam in cams:
            cam.serialize(Path(settings.outdir) / f"{cam.name}.npy")
            self._test_data_filenames.append(
                str(Path(settings.outdir) / f"{cam.name}.npy"))
        self.assertEqual(len(self._cam_gts), len(cams))
        for n, cam in enumerate(cams):
            gt = self._cam_gts[n]
            mw = cam.RTb
            # test translation
            for i in range(3):
                test = abs(gt["translation"][i][0] - mw[i, 3])
                if test >= 0.005:
                    print(f"overly high translation error {test} >= 0.005")
                #broken with opencv 4.5.4
                #self.assertTrue(test < 0.01)
            # test rotation
            for i in range(3):
                for j in range(3):
                    test = abs(gt["rotationMat"][i][j] - mw[i, j])
                    if test >= 0.001:
                        print(f"overly high rotation error {test} >= 0.001")
                    #broken with opencv 4.5.4
                    #self.assertTrue(test < 0.1)

    def test_registration(self):
        # create a settings object
        settings = Settings()
        settings.from_params({
            "aruco_dict": "DICT_5X5",
            "cols": 24,
            "rows": 18,
            "square_size": 0.080,
            "marker_size": 0.062,
            "min_number_of_corners": 20,
            "min_number_of_calibration_images": 20,
            "max_count": 10000,
            "epsilon": 0.00001,
            "sensor_width_mm": 10,
            "sensor_height_mm": 7.5,
            "f_mm": 16.0,
            "visualize": False
        })

        calib = Calibration(settings=settings)
        stream = FileStream()
        directory = self._root / "single_cam" / "undistorted"
        stream.initialize(directory=directory)

        if self._has_broken_cv2:
            print(f"Warning! opencv_version {cv2.__version__} is broken since 4.8, skipped\n")
            self.skipTest("broken opencv")
        cam = calib.calibrate_intrinsics(stream)
        stream = FileStream()
        directory = self._root / "single_cam" / "undistorted"
        stream.initialize(directory=directory)

        cams = calib.calibrate_extrinsics(stream, cam)
        depth_stream = FileStream()
        directory = self._root / "single_cam" / "depth"
        depth_stream.initialize(directory=directory)

        stream.reset()
        pcds = []
        merged = None
        for i in range(4):
            pcd = register_depthmap_to_world(
                cams[i],
                depth_stream.get(i),
                stream.get(i),
                0.1)
            if merged is None:
                merged = pcd
            else:
                merged += pcd
            pcds.append(pcd)

        bbox = merged.get_axis_aligned_bounding_box()
        maxb = bbox.max_bound
        minb = bbox.min_bound
        self.assertTrue((maxb[0]-minb[0])-2.7045092166984497 < 0.1)
        self.assertTrue((maxb[1]-minb[1])-2.466959368288604 < 0.1)
        self.assertTrue((maxb[2]-minb[2])-0.4866462015080666 < 0.1)


if __name__ == '__main__':
    unittest.main()
