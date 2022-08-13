import yaml
import unittest
import numpy as np
from glob import glob
from pathlib import Path
from calibpy.Settings import Settings
from calibpy.Stream import FileStream
from calibpy.Calibration import Calibration
from calibpy.Registration import register_depthmap_to_world


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
        cam = calib.calibrate_intrinsics(stream)
        self.assertAlmostEqual(cam.f_mm, 16.008241865239107, places=1)
        self.assertAlmostEqual(cam.f_px, 2.04907412e+03, places=1)
        self.assertAlmostEqual(cam.cx, 6.40063878e+02, places=2)
        self.assertAlmostEqual(cam.cy, 4.79229659e+02, places=2)
        self.assertTupleEqual(cam.sensor_size_mm, (7.5, 10))
        self.assertTupleEqual(cam.image_size, (960, 1280))
        self.assertAlmostEqual(cam.distortion[0][0], -4.40496973e-04, places=2)
        self.assertAlmostEqual(cam.distortion[0][1], 1.95103766e-02, places=2)
        self.assertAlmostEqual(cam.distortion[0][2], -1.00665794e-04, places=2)
        self.assertAlmostEqual(cam.distortion[0][3], 6.85122003e-05, places=2)
        self.assertAlmostEqual(cam.distortion[0][4], -1.16823432e-01, places=2)

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
                self.assertTrue(test < 0.002)
            # test rotation
            for i in range(3):
                for j in range(3):
                    test = abs(gt["rotationMat"][i][j] - mw[i, j])
                    self.assertTrue(test < 0.001)

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
        means = np.zeros((4, 3), dtype=np.float32)
        for i in range(4):
            pcd = register_depthmap_to_world(
                cams[i],
                depth_stream.get(i),
                stream.get(i),
                0.1)
            mean, _ = pcd.compute_mean_and_covariance()
            means[i, :] = mean
            pcds.append(pcd)

        std_x = np.std(means[:, 0])
        std_y = np.std(means[:, 1])
        std_z = np.std(means[:, 2])
        mean_z = np.mean(means[:, 2])
        self.assertTrue(std_x < 0.1)
        self.assertTrue(std_y < 0.1)
        self.assertTrue(std_z < 0.001)
        self.assertTrue(abs(mean_z-0.072) < 0.001)


if __name__ == '__main__':
    unittest.main()
