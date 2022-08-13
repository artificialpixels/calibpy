import open3d as o3d
import numpy as np
from pathlib import Path

from calibpy.Stream import Stream
from calibpy.Settings import Settings
from calibpy.Calibration import Calibration

from calibpy.Registration import load_as_rgbd, register_depthmap_to_world, show_registration


if __name__ == "__main__":
    root = Path.cwd() / "tests" / "data"

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
        "outdir": "C:\\Users\\svenw\\OneDrive\\Desktop\\results"
    })

    calib = Calibration(settings=settings)
    stream = Stream(root / "single_cam" / "undistorted")
    cam = calib.calibrate_intrinsics(stream)
    stream = Stream(root / "single_cam" / "undistorted")
    cams = calib.calibrate_extrinsics(stream, cam)
    depth_stream = Stream(root / "single_cam" / "depth")

    stream.reset()
    pcds = []
    for i in [0, 1, 2, 3]:
        pcd = register_depthmap_to_world(
            cams[i],
            depth_stream.get(i),
            stream.get(i),
            0.1)
        pcds.append(pcd)

    show_registration(pcds, True)
