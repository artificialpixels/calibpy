from pathlib import Path

from calibpy.Stream import Stream
from calibpy.Settings import Settings
from calibpy.Calibration import Calibration

from calibpy.Camera import Camera

if __name__ == "__main__":

    cam = Camera()
    cam.quick_init()

    stream = Stream(Path.cwd() / "tests" / "data" /
                    "single_cam" / "undistorted")
    print("Number of frames:", stream.length)

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

    stream = Stream(Path.cwd() / "tests" / "data" /
                    "single_cam" / "undistorted")
    print("Number of frames:", stream.length)

    cams = calib.calibrate_extrinsics(stream, cam)
    for cam in cams:
        print(cam.get_blender_mw())
