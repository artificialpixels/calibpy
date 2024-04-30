"""
:Copyrights: Artificial Pixels
:Author: Sven Wanner (artificial.pixels@gmail.com)
:Sponsor: SpexAI GmbH
"""

from pathlib import Path
import open3d as o3d
from calibpy.Camera import Camera
from calibpy.Settings import Settings
from calibpy.Stream import FileStream
from calibpy.Calibration import Calibration
from calibpy.Registration import register_depthmap_to_world, show_registration


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


def show_pcl_set(pcds: list):
    show_registration(pcds)


def instric_calibration(
        calib: Calibration,
        image_directory: str,
        out_dir: Path = None,
        is_lazy: bool = True):

    # generate load/save filename
    fname = get_savename_pattern(save_dir=out_dir, name="intrinsics")

    # If is_lazy is True and an existing calibration file
    # can be found, we load the calibration data from file.
    cam = None
    if isinstance(fname, Path) and fname.is_file() and is_lazy:
        cam = Camera()
        cam.load(fname)
        return cam

    # We use FileStream with directory to read all files from a directory
    fs = FileStream()
    fs.initialize(directory=image_directory)

    # run intrinsic calibration on the images loaded
    cam = calib.calibrate_intrinsics(fs)

    # save cam if out_dir wasn't None
    if fname is not None:
        cam.serialize(fname)

    return cam


def extrinsic_calibration_image(
        calib: Calibration,
        intrinsics: Camera,
        image_filename: str,
        out_dir: Path = None):

    # We use FileStream with filename to load a single image
    fs = FileStream()
    fs.initialize(filename=image_filename)

    # run extrinsic calibration on the image loaded
    cams = calib.calibrate_extrinsics(fs, intrinsics)

    # save cam if out_dir wasn't None
    fname = get_savename_pattern(
        save_dir=out_dir, name=f"{intrinsics.name}_extrinsics")
    if fname is not None:
        cams[0].serialize(fname)
    return cams[0]


def extrinsic_calibration_sequence(
        calib: Calibration,
        intrinsics: Camera,
        image_directory: str,
        out_dir: Path = None,
        register_from_frame: int = 0,
        register_to_frame: int = 0):

    # We use FileStream with directory to read all files from a directory
    # from_frame/to_frame chooses a frame subset of the folder content
    fs = FileStream()
    fs.initialize(
        directory=image_directory,
        from_frame=register_from_frame,
        to_frame=register_to_frame)

    # run extrinsic calibration on the images loaded
    cams = calib.calibrate_extrinsics(fs, intrinsics)

    # save each frame cam if out_dir wasn't None
    for n, cam in enumerate(cams):
        fname = get_savename_pattern(
            save_dir=out_dir,
            name="extrinsics",
            pattern=n)
        if fname is not None:
            cam.serialize(fname)
    return cams


def register_view(
        image_filename: str,
        depth_filename: str,
        extrinsics: Camera,
        out_dir: Path,
        register_from_frame: int = 0,
        register_to_frame: int = 0,
        blender_conform: bool = True):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    # We use FileStreams with directory to read all files from a directory
    # from_frame/to_frame chooses a frame subset of the folder content
    fs_imgs = FileStream()
    fs_depths = FileStream()
    fs_imgs.initialize(
        filename=image_filename,
        from_frame=register_from_frame,
        to_frame=register_to_frame)
    fs_depths.initialize(
        filename=depth_filename,
        from_frame=register_from_frame,
        to_frame=register_to_frame)

    # register the depth and color images as pointclouds to
    # the global coordinate system of the extrnal calibration
    pcd = register_depthmap_to_world(
        extrinsics,
        fs_depths.get(0),
        fs_imgs.get(0),
        0.1,
        blender_conform)

    # save pointcloud if out_dir wasn't None
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
        out_dir: Path,
        register_from_frame: int = 0,
        register_to_frame: int = 0,
        blender_conform: bool = True):

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    # We use FileStreams with directory to read all files from a directory
    # from_frame/to_frame chooses a frame subset of the folder content
    fs_imgs = FileStream()
    fs_depths = FileStream()
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
        # register the depth and color images as pointclouds to
        # the global coordinate system of the external calibration
        pcd = register_depthmap_to_world(
            extrinsics[i],
            fs_depths.get(i),
            fs_imgs.get(i),
            0.1,
            blender_conform)
        pcds.append(pcd)

        # save each pointcloud if out_dir wasn't None
        fname = get_savename_pattern(
            save_dir=out_dir,
            name="pcl",
            pattern=i,
            ftype="ply")
        if fname is not None:
            o3d.io.write_point_cloud(str(fname), pcd)

    return pcds


def single_cam_workflow(
        project_dir: str,
        project_name: str,
        intrinsic_calibration_input_dir: str,
        calibration_config_file: str,
        extrinsic_calibration_input: str = None,
        depth_registration_input: str = None, # file or dir
        color_registration_input: str = None,
        blender_conform: bool = True,
        register_from_frame: int = 0,
        register_to_frame: int = 1,
        lazy_intrinsics: bool = True,
        visualize: bool = True):
    intr = None
    extr = None
    extrs = None
    pcds = None
    single_file_mode = False

    out_root = Path(project_dir) / project_name
    out_root.mkdir(parents=True, exist_ok=True)

    # To control the processing we need a Settings object we
    # can initialize via a dictionary or from a .yaml file
    settings = Settings()
    settings.from_config(calibration_config_file)

    # create a Calibration instance and pass the settings object
    calib = Calibration(settings=settings)
    calib.visualize = visualize

    # ***** Intrinsic calibration *****
    intr = instric_calibration(
        calib=calib,
        image_directory=intrinsic_calibration_input_dir,
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
            calib=calib,
            intrinsics=intr,
            image_filename=extrinsic_calibration_input,
            out_dir=out_root)
        extrs = [extr]
    # ***** Extrinsic calibration of a camera stream *****
    else:
        extrs = extrinsic_calibration_sequence(
            calib=calib,
            intrinsics=intr,
            image_directory=extrinsic_calibration_input,
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
            out_dir=out_root,
            register_from_frame=register_from_frame,
            register_to_frame=register_to_frame,
            blender_conform=blender_conform)
    # ***** Registration of a depth stream *****
    else:
        if not Path(color_registration_input).is_dir():
            color_registration_input = ""
        pcds = register_stream(
            image_directory=color_registration_input,
            depth_directory=depth_registration_input,
            extrinsics=extrs,
            out_dir=out_root,
            register_from_frame=register_from_frame,
            register_to_frame=register_to_frame,
            blender_conform=blender_conform)

    return intr, extrs, pcds
