#!/usr/bin/env python3

import numpy as np
try:
    from Serializer import Serializer
except (ModuleNotFoundError, ImportError):
    from .Serializer import Serializer


class Camera(Serializer):
    """
    Camera class modelling all aspects of a pinhole camera.
    It derives from a Serializer class to ensure all protected 
    class members can be serialized and deserialized.
    """

    def __init__(self):
        self._f_mm = None           # f in mm
        self._sensor_size = None    # size in mm, (y, x)
        self._image_size = None     # size in px, (y, x)
        self._intrinsics = None     # intrinsic camera matrix
        self._distortion = None     # (k1, k2, p1, p2, k3)
        self._RT = None             # 4x4 transformation matrix
        self._RTb = None            # 4x4 Blender matrix_world
        print("Camera initialized!")

    def quick_init(self,
                   f_mm: float = 50,
                   sensor_size: tuple = (20.25, 36.0),
                   image_size: tuple = (1080, 1920)):
        """Automatically initializes the camera instance as Blender default camera

        Args:
            f_mm (float, optional): Focal length in mm. Defaults to 50.
            sensor_size (tuple, optional): Sensor size in mm (sy, sx). Defaults to (20.25, 36.0).
            image_size (tuple, optional): Image size in px (y, x). Defaults to (1080, 1920).
        """
        self._f_mm = f_mm
        self._sensor_size = sensor_size
        self._image_size = image_size
        f_x = self._f_mm / self._sensor_size[1] * self._image_size[1]
        f_y = self._f_mm / self._sensor_size[0] * self._image_size[0]
        s_x = self._image_size[1] / 2
        s_y = self._image_size[0] / 2
        self._intrinsics = np.array([[f_x, 0, s_x], [0, f_y, s_y], [0, 0, 1]])
        self._distortion = np.zeros((1, 5))
        self.RT = np.identity(4)

    def compute_intrinsics(self,
                           f_mm: float = None,
                           sensor_size: tuple = None,
                           image_size: tuple = None):
        """Computing intrinsic camera matrix. To successfully compute the
        matrix f_mm, sensor_size and image_size are needed, thus each must
        be passed as argument when calling this function or must have been
        set in advance.

        Args:
            f_mm (float, optional): focal length in mm. Defaults to None.
            sensor_size (tuple, optional): Sensor size in mm (sy, sx). Defaults to None.
            image_size (tuple, optional): Image size in px (y, x). Defaults to None.
        """
        if f_mm is not None:
            self._f_mm = f_mm
        if sensor_size is not None:
            self._sensor_size = sensor_size
        if image_size is not None:
            self._image_size = image_size
        assert isinstance(self._f_mm, float)
        assert isinstance(self._sensor_size, tuple) and len(
            self._sensor_size) == 2
        assert isinstance(self._image_size, tuple) and len(
            self._image_size) == 2
        f_x = self._f_mm / self._sensor_size[1] * self._image_size[1]
        f_y = self._f_mm / self._sensor_size[0] * self._image_size[0]
        s_x = self._image_size[1] / 2
        s_y = self._image_size[0] / 2
        self._intrinsics = np.array([[f_x, 0, s_x], [0, f_y, s_y], [0, 0, 1]])

    @property
    def f_mm(self):
        return self._f_mm

    @f_mm.setter
    def f_mm(self, value: float):
        assert value > 0
        self._f_mm = value

    @property
    def sensor_size_mm(self):
        return self._sensor_size

    @sensor_size_mm.setter
    def sensor_size(self, value: tuple):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self._sensor_size = value

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value: tuple):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self._image_size = value

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value: np.ndarray):
        assert isinstance(value, np.ndarray)
        assert value.shape == (3, 3)
        self._intrinsics = value

    def set_intrinsics(self,
                       fx: float,
                       fy: float,
                       cx: float,
                       cy: float,
                       s: float = 0):
        """set intrinsics, fx/fy focal length in px, cx/cy optical center in
        px, s skew factor

        Args:
            fx (float): focal length in px
            fy (float): focal length in px
            cx (float): optical center in px
            cy (float): optical center in px
            s (float, optional): skew factor. Defaults to 0.
        """
        self.intrinsics = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, value: np.ndarray):
        assert isinstance(value, np.ndarray)
        assert value.shape == (1, 5)
        self._distortion = value

    def set_distortion(self,
                       k1: float,
                       k2: float,
                       p1: float,
                       p2: float,
                       k3: float):
        """Set distortion coefficients. The coefficients are the
        opencv 5 parameter model:
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        Args:
            k1 (float):
            k2 (float):
            p1 (float):
            p2 (float):
            k3 (float):
        """
        self.distortion(np.array([[k1, k2, p1, p2, k3]]))

    @property
    def RT(self):
        return self._RT

    @RT.setter
    def RT(self, value: np.ndarray):
        assert value.shape == (4, 4)
        self._RT = value

        # from opencv to blender convention
        T = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
        self._RTb = np.linalg.inv(self._RT) @ T

    @property
    def RTb(self):
        return self._RTb


if __name__ == "__main__":
    cam = Camera()
    cam.quick_init()
    cam.serialize("C:\\Users\\svenw\\OneDrive\\Desktop\\test.npy")
    cam2 = Camera()
    cam2.load("C:\\Users\\svenw\\OneDrive\\Desktop\\test.npy")
