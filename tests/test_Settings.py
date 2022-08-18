import unittest
from pathlib import Path
from calibpy.Settings import Settings


class TestSettingsModule(unittest.TestCase):

    def setUp(self):
        print("start Settings tests...")
        self._root = Path.cwd() / "tests" / "data"

    def test_eq_op(self):
        settings1 = Settings()
        settings2 = Settings()
        settings3 = Settings()
        settings4 = Settings()
        settings5 = Settings()

        settings1.from_params({
            "a": "str",
            "b": 1,
            "c": 0.1
        })
        settings2.from_params({
            "a": "str",
            "b": 1,
            "c": 0.1
        })
        settings3.from_params({
            "a": "str",
            "b": 1,
            "c": 1.1
        })
        settings4.from_params({
            "a": "str",
            "b": 1,
            "c": "1.1"
        })
        settings5.from_params({
            "a": "str",
            "b": 1,
            "d": 0.1
        })

        self.assertTrue(settings1 == settings2)
        self.assertTrue(settings1 == settings3)
        self.assertFalse(settings1 == settings4)
        self.assertFalse(settings1 == settings5)

    def test_contains(self):
        settings = Settings()
        settings.from_params({"a": 1, "b": 2})
        self.assertTrue("a" in settings)
        self.assertTrue("b" in settings)
        self.assertFalse("c" in settings)

    def test_serialization(self):
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
        settings.save(
            save_dir=str(self._root),
            filename="test_settings")
        self.assertTrue(Path(self._root / "test_settings.yaml").is_file())
        settings2 = Settings()
        settings2.from_config(str(self._root / "test_settings.yaml"))
        self.assertTrue(settings == settings2)
        import os
        os.remove(str(self._root / "test_settings.yaml"))

    def test_creation_from_dict(self):
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

        self.assertEqual(settings.aruco_dict, "DICT_5X5")
        self.assertEqual(settings.cols, 24)
        self.assertEqual(settings.rows, 18)
        self.assertEqual(settings.square_size, 0.080)
        self.assertEqual(settings.marker_size, 0.062)
        self.assertEqual(settings.min_number_of_corners, 20)
        self.assertEqual(settings.min_number_of_calibration_images, 20)
        self.assertEqual(settings.max_count, 10000)
        self.assertEqual(settings.epsilon, 0.00001)
        self.assertEqual(settings.sensor_width_mm, 10)
        self.assertEqual(settings.sensor_height_mm, 7.5)
        self.assertEqual(settings.f_mm, 16.0)
        self.assertEqual(settings.visualize, False)
