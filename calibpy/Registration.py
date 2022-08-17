import numpy as np
import open3d as o3d

from calibpy.Camera import Camera
from calibpy.Calibration import Calibration


def load_as_rgbd(camera, depth_map, color_img=None):
    if color_img is None:
        color_img = np.ones(list(depth_map.shape)+[3], dtype=np.uint8)*200
    color_img = Calibration.undistort_image(
        color_img,
        camera.intrinsics,
        camera.distortion)
    color_raw = o3d.geometry.Image(color_img.astype(np.uint8))
    dmap = np.copy(depth_map)
    dmap *= 1000
    depth_raw = o3d.geometry.Image(dmap)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000,
        depth_trunc=100)
    return rgbd_image


def register_depthmap_to_world(
        camera: Camera,
        depth_map: np.ndarray,
        color_img: np.ndarray = None,
        downsample_factor: float = 0.1):

    assert camera.image_size is not None
    assert camera.fx is not None
    assert camera.fy is not None
    assert camera.cx is not None
    assert camera.cy is not None
    assert camera.RT is not None

    o3d_cam = o3d.camera.PinholeCameraIntrinsic(
        camera.image_size[1],
        camera.image_size[0],
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy)

    rgbd = load_as_rgbd(camera, depth_map, color_img)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d_cam,
        camera.RT)
    pcd = pcd.random_down_sample(downsample_factor)
    return pcd


def show_registration(pointclouds, with_origin: bool = True):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if with_origin:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    for pcd in pointclouds:
        vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(100000)
    vis.run()
    vis.destroy_window()
