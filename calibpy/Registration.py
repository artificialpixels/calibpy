from calibpy.Calibration import Calibration
from calibpy.Camera import Camera
import open3d as o3d
import numpy as np


BLENDER_CONFORM = True


def project_3d_blender_conform(depth_map: np.ndarray, color_img: np.ndarray, camera: Camera):
    if color_img is None:
        color_img = np.ones(list(depth_map.shape)+[3], dtype=np.uint8)*200
    color_img = Calibration.undistort_image(
        color_img,
        camera.intrinsics,
        camera.distortion)
    if np.amax(color_img > 1):
        color_img = color_img.astype(np.float32)
        color_img /= 255
    if len(color_img.shape) == 2:
        tmp = np.zeros(list(color_img.shape) + [3])
        for i in range(3):
            tmp[:, :, i] = color_img
        color_img = tmp

    fx = camera.intrinsics[0, 0]
    fy = camera.intrinsics[1, 1]
    cx = camera.intrinsics[0, 2]
    cy = camera.intrinsics[1, 2]
    w = depth_map.shape[1]
    h = depth_map.shape[0]
    _x = np.arange(w)
    _y = np.arange(h)
    u, v = np.meshgrid(_x, _y)
    x = (u - cx)
    x = np.multiply(x, depth_map)
    x /= fx
    y = -(v - cy)
    y = np.multiply(y, depth_map)
    y /= fy
    z = -depth_map
    xyz = np.zeros((4, w*h))
    rgb = np.zeros((w*h, 3))
    pcl = np.zeros((w*h, 3))
    xyz[0, :] = np.reshape(x, -1)
    xyz[1, :] = np.reshape(y, -1)
    xyz[2, :] = np.reshape(z, -1)
    xyz[3, :] = 1

    xyz = np.matmul(camera.RTb, xyz)
    for i in range(3):
        pcl[:, i] = xyz[i, :]

    rgb[:, 0] = np.reshape(color_img[:, :, 0], -1)
    rgb[:, 1] = np.reshape(color_img[:, :, 1], -1)
    rgb[:, 2] = np.reshape(color_img[:, :, 2], -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # pcd.transform(np.linalg.inv(camera.RTb))
    return pcd


def load_as_rgbd(camera, depth_map, color_img=None):
    if color_img is None:
        color_img = np.ones(list(depth_map.shape)+[3], dtype=np.uint8)*200
    color_img = Calibration.undistort_image(
        color_img,
        camera.intrinsics,
        camera.distortion)
    color_raw = o3d.geometry.Image(color_img.astype(np.uint8))
    dmap = np.copy(depth_map)
    depth_raw = o3d.geometry.Image(dmap)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1,
        depth_trunc=100)
    return rgbd_image


def register_depthmap_to_world(
        camera: Camera,
        depth_map: np.ndarray,
        color_img: np.ndarray = None,
        downsample_factor: float = 0.1,
        blender_conform: bool = True):

    assert camera.image_size is not None
    assert camera.fx is not None
    assert camera.fy is not None
    assert camera.cx is not None
    assert camera.cy is not None
    assert camera.RT is not None

    if blender_conform:
        pcd = project_3d_blender_conform(depth_map, color_img, camera)
    else:
        o3d_cam = o3d.camera.PinholeCameraIntrinsic(
            width=camera.image_size[1],
            height=camera.image_size[0],
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy)
        rgbd = load_as_rgbd(camera, depth_map, color_img)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d_cam,
            camera.RT)

    pcd = pcd.random_down_sample(downsample_factor)
    return pcd


def show_registration(pointclouds, with_origin: bool = True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    if with_origin:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    for pcd in pointclouds:
        vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(100000)
    vis.run()
    vis.destroy_window()
