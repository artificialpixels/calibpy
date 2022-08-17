from pathlib import Path

from calibpy.Settings import Settings
from calibpy.Stream import FileStream
from workflows import instric_calibration

ROOT = Path.cwd() / "tests" / "data"


def main_instric_calibration():
    # To get access to data in a form the calibration objects
    # understands we always fir create a Stream object. In this
    # case we create an FileStream instance that can read images
    # from disc
    fs = FileStream()
    # We read all files from a directory by passing a directory str
    fs.initialize(directory=str(ROOT / "single_cam" / "undistorted"))

    # To control the processing we need a Seetings object we can
    # initialize via a dictionary or from a .yaml file
    settings = Settings()
    settings.from_file(str(ROOT / "demo_settings_intrinsic_calibration.yaml"))

    instric_calibration(fs, settings)


if __name__ == "__main__":
    main_instric_calibration()
