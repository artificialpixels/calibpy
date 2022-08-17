from genericpath import isfile
import os
import pickle
import numpy as np
import open3d as o3d
from glob import glob
from pathlib import Path
from calibpy.Camera import Camera
from calibpy.Stream import FileStream
from calibpy.Settings import Settings
from calibpy.Calibration import Calibration
from calibpy.Registration import register_depthmap_to_world


def get_savename_pattern(
        save_dir: Path = None,
        name: str = "",
        pattern: object = None,
        ftype: str = "npy"):
    fname = None
    if save_dir is None:
        return None
    if not ftype.startswith("."):
        ftype = "." + ftype
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    entry = ""
    if isinstance(pattern, int):
        entry = f"_{pattern:06d}"
    elif isinstance(pattern, str):
        entry = f"_{pattern}"
    if save_dir.is_dir():
        fname = save_dir / f"{name}{entry}{ftype}"
        fname.parent.mkdir(parents=True, exist_ok=True)
    return fname


def instric_calibration(
        image_directory: str,
        settings_filename: str,
        out_dir: Path = None,
        is_lazy: bool = True):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    # To control the processing we need a Seetings object we
    # can initialize via a dictionary or from a .yaml file
    settings = Settings()
    settings.from_file(settings_filename)

    # generate load/save filename
    fname = get_savename_pattern(save_dir=out_dir, name="intrinsics")

    # We load the calibration if is_lazy is True and
    # if no existing calibration file can be found.
    cam = None
    if isinstance(fname, Path) and fname.is_file() and is_lazy:
        cam = Camera()
        cam.load(fname)
        return cam

    # To get access to data in a form the calibration objects
    # understands we always fir create a Stream object. In this
    # case we create an FileStream instance that can read images
    # from disc
    fs = FileStream()
    # We read all files from a directory by passing a directory str
    fs.initialize(directory=image_directory)

    # create a Calibration instance and pass the settings object
    calib = Calibration(settings=settings)
    # exectute intrinsic calibration
    cam = calib.calibrate_intrinsics(fs)

    # save cam
    if fname is not None:
        cam.serialize(fname)

    return cam


def extrinsic_calibration_image(
        cam: Camera,
        image_filename: str,
        settings_filename: str,
        out_dir: Path = None):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    stream = FileStream()
    stream.initialize(filename=image_filename)

    settings = Settings()
    settings.from_file(settings_filename)

    calib = Calibration(settings=settings)
    cams = calib.calibrate_extrinsics(stream, cam)

    # generate load/save filename
    fname = get_savename_pattern(save_dir=out_dir, name="extrinsics")
    # save cam
    if fname is not None:
        cams[0].serialize(fname)
    return cams[0]


def extrinsic_calibration_sequence(
        cam: Camera,
        image_directory: str,
        settings_filename: str,
        out_dir: Path = None,
        register_from_frame: int = 0,
        register_to_frame: int = 0):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    stream = FileStream()
    stream.initialize(
        directory=image_directory,
        from_frame=register_from_frame,
        to_frame=register_to_frame)

    settings = Settings()
    settings.from_file(settings_filename)

    calib = Calibration(settings=settings)
    cams = calib.calibrate_extrinsics(stream, cam)

    for n, cam in enumerate(cams):
        # generate load/save filename
        fname = get_savename_pattern(
            save_dir=out_dir,
            name="extrinsics",
            pattern=n)
        # save cam
        if fname is not None:
            cam.serialize(fname)
    return cams


def register_view(
        image_filename: str,
        depth_filename: str,
        extrinsics: Camera,
        settings_filename: str,
        out_dir: Path,
        register_from_frame: int = 0,
        register_to_frame: int = 0):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    # To control the processing we need a Seetings object we
    # can initialize via a dictionary or from a .yaml file
    settings = Settings()
    settings.from_file(settings_filename)

    # Use a FileStream instance to read
    # depth and color images from disc
    fs_imgs = FileStream()
    fs_depths = FileStream()
    # We read just a subset of the files from a directory
    # by using from_frame, to_frame parameters
    fs_imgs.initialize(
        filename=image_filename,
        from_frame=register_from_frame,
        to_frame=register_to_frame)
    fs_depths.initialize(
        filename=depth_filename,
        from_frame=register_from_frame,
        to_frame=register_to_frame)

    pcd = register_depthmap_to_world(
        extrinsics,
        fs_depths.get(0),
        fs_imgs.get(0),
        0.1)

    # generate load/save filename
    fname = get_savename_pattern(
        save_dir=out_dir,
        name="pcl",
        ftype="ply")

    if fname is not None:
        o3d.io.write_point_cloud(str(fname), pcd)

    return [pcd]


def register_stream(
        image_directory: str,
        depth_directory: str,
        extrinsics: list,
        settings_filename: str,
        out_dir: Path,
        register_from_frame: int = 0,
        register_to_frame: int = 0):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    # To control the processing we need a Seetings object we
    # can initialize via a dictionary or from a .yaml file
    settings = Settings()
    settings.from_file(settings_filename)

    # Use a FileStream instance to read
    # depth and color images from disc
    fs_imgs = FileStream()
    fs_depths = FileStream()
    # We read just a subset of the files from a directory
    # by using from_frame, to_frame parameters
    fs_imgs.initialize(
        directory=image_directory,
        from_frame=register_from_frame,
        to_frame=register_to_frame)
    fs_depths.initialize(
        directory=depth_directory,
        from_frame=register_from_frame,
        to_frame=register_to_frame)

    pcds = []
    for i in range(4):
        pcd = register_depthmap_to_world(
            extrinsics[i],
            fs_depths.get(i),
            fs_imgs.get(i),
            0.1)
        pcds.append(pcd)

        # generate load/save filename
        fname = get_savename_pattern(
            save_dir=out_dir,
            name="pcl",
            pattern=i,
            ftype="ply")

        if fname is not None:
            o3d.io.write_point_cloud(str(fname), pcd)

    return pcds


def single_camera_workflow(
        project_dir: str,
        project_name: str,
        intrinsic_calibration_input_dir: str,
        calibration_config_file: str,
        extrinsic_calibration_input: str = None,
        depth_registration_input: str = None,
        color_registration_input: str = None,
        register_from_frame: int = 0,
        register_to_frame: int = 1,
        lazy_intrinsics: bool = True):
    intr = None
    extr = None
    extrs = None
    pcds = None
    single_file_mode = False

    #root = Path.cwd() / "tests" / "data"
    out_root = Path(project_dir) / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    #image_directory = root / "single_cam" / "distorted"
    #settings_fname = root / "demo_calibration_settings.yaml"
    settings_fname = calibration_config_file

    # ***** Intrinsic calibration *****
    intr = instric_calibration(
        image_directory=intrinsic_calibration_input_dir,
        settings_filename=settings_fname,
        out_dir=out_root,
        is_lazy=lazy_intrinsics)

    # ***** If no extrinsic calibration desired we're done here
    if extrinsic_calibration_input is None:
        return intr, None, None

    # ***** Check if single view or stream mode is needed
    assert isinstance(extrinsic_calibration_input, str)
    if Path(extrinsic_calibration_input).is_file():
        single_file_mode = True
    else:
        assert Path(extrinsic_calibration_input).is_dir()

    # ***** Extrinsic calibration of a single view *****
    extrs = None
    if single_file_mode:
        extr = extrinsic_calibration_image(
            cam=intr,
            image_filename=extrinsic_calibration_input,
            settings_filename=settings_fname,
            out_dir=out_root)
        extrs = [extr]
    # ***** Extrinsic calibration of a camera stream *****
    else:
        extrs = extrinsic_calibration_sequence(
            cam=intr,
            image_directory=extrinsic_calibration_input,
            settings_filename=settings_fname,
            out_dir=out_root,
            register_from_frame=register_from_frame,
            register_to_frame=register_to_frame)

    # ***** If no depth map registration desired we're done here
    if depth_registration_input is None:
        return intr, extrs, None

    # ***** Check if single view or stream mode is needed
    single_file_mode = False
    assert isinstance(depth_registration_input, str)
    if Path(depth_registration_input).is_file():
        single_file_mode = True
    else:
        assert Path(depth_registration_input).is_dir()

    # ***** Registration of a single depth view *****
    if single_file_mode:
        if not Path(color_registration_input).is_file():
            color_registration_input = ""
        pcds = register_view(
            image_filename=color_registration_input,
            depth_filename=depth_registration_input,
            extrinsics=extrs[0],
            settings_filename=settings_fname,
            out_dir=out_root,
            register_from_frame=register_from_frame,
            register_to_frame=register_to_frame)
    # ***** Registration of a depth stream *****
    else:
        if not Path(color_registration_input).is_dir():
            color_registration_input = ""
        pcds = register_stream(
            image_directory=color_registration_input,
            depth_directory=depth_registration_input,
            extrinsics=extrs,
            settings_filename=settings_fname,
            out_dir=out_root,
            register_from_frame=register_from_frame,
            register_to_frame=register_to_frame)

    return intr, extrs, pcds


if __name__ == "__main__":
    PROJECT_DIR = "D:\\Tmp"
    PROJECT_NAME = "scw_test"
    DATA_ROOT = Path.cwd() / "tests" / "data"
    IMG_ROOT = DATA_ROOT / "single_cam" / "distorted"
    DEPTH_ROOT = DATA_ROOT / "single_cam" / "depth"
    SETTINGS_FILE = DATA_ROOT / "demo_calibration_settings.yaml"
    REGISTER_RANGE = (0, 4)

    intr, extrs, pcds = single_camera_workflow(
        project_dir=PROJECT_DIR,
        project_name=PROJECT_NAME,
        intrinsic_calibration_input_dir=str(IMG_ROOT),
        calibration_config_file=str(SETTINGS_FILE),
        extrinsic_calibration_input=str(IMG_ROOT),
        depth_registration_input=str(DEPTH_ROOT),
        color_registration_input=str(IMG_ROOT),
        register_from_frame=REGISTER_RANGE[0],
        register_to_frame=REGISTER_RANGE[1],
        lazy_intrinsics=True)
    a = 0
