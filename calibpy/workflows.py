import yaml
import unittest
import numpy as np
from glob import glob
from pathlib import Path
from calibpy.Stream import Stream, FileStream
from calibpy.Settings import Settings
from calibpy.Calibration import Calibration
from calibpy.Registration import register_depthmap_to_world


ROOT = Path.cwd() / "..", "tests" / "data"


def instric_calibration(stream: Stream, settings: Settings):
    pass


def main_instric_calibration():
    # To get access to data in a form the calibration objects
    # understands we always fir create a Stream object. In this
    # case we create an FileStream instance that can read images
    # from disc
    fs = FileStream()
    # We read all files from a directory by passing a directory str
    fs.initialize(directory=str(ROOT / "undistorted"))

    # To control the processing we need a Seetings object we can
    # initialize via a dictionary or from a .yaml file
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

    instric_calibration(fs, settings)
