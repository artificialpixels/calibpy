#!/usr/bin/env python3

import numpy as np
try:
    from Serializer import Serializer
except (ModuleNotFoundError, ImportError):
    from .Serializer import Serializer


class Camera(Serializer):
    """
    Camera class models a pinhole camera and keeps track of all
    relevant data like intrinsics and transformations.
    It inherits from class Serializer, which allows to serialize
    and deserialize all protected class members.

    :inherit Serializer: Serializer base class enables serialization
    """

    def __init__(self):
        super().__init__()

        self._name = None
        self._f_mm = None           # f in mm
        self._sensor_size = None    # size in mm, (y, x)
        self._image_size = None     # size in px, (y, x)
        self._intrinsics = None     # intrinsic camera matrix
        self._distortion = None     # (k1, k2, p1, p2, k3)
        self._RT = None             # 4x4 transformation matrix
        self._RTb = None            # 4x4 Blender matrix_world
        print("Camera initialized!")

    @staticmethod
    def from_cam(cam):
        """Copy Camera object

        :param cam: Camera instance
        :type cam: Camera
        :return: Camera instance copy
        :rtype: Camera
        """
        from copy import deepcopy
        return deepcopy(cam)

    def quick_init(self,
                   f_mm: float = 50,
                   sensor_size: tuple = (20.25, 36.0),
                   image_size: tuple = (1080, 1920)):
        """
        Automatically initializes the camera
        instance as Blender default camera

        :param f_mm: Focal length in mm, defaults to 50
        :type f_mm: float, optional
        :param sensor_size: Sensor size in mm (y, x), defaults to (20.25, 36.0)
        :type sensor_size: tuple, optional
        :param image_size: Image size in px (y, x), defaults to (1080, 1920)
        :type image_size: tuple, optional
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
        """
        Computing intrinsic camera matrix. To successfully compute the
        matrix f_mm, sensor_size and image_size are needed, thus each must
        be passed as argument when calling this function or must have been
        set in advance.

        :param f_mm: focal length in mm, defaults to None
        :type f_mm: float, optional
        :param sensor_size: Sensor size in mm (y, x), defaults to None
        :type sensor_size: tuple, optional
        :param image_size: Image size in px (y, x), defaults to None
        :type image_size: tuple, optional
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
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        assert isinstance(value, str)
        self._name = value

    @property
    def f_px(self):
        if self.intrinsics is not None:
            return (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2
        return None

    @property
    def fx(self):
        if self.intrinsics is not None:
            return self.intrinsics[0, 0]
        return None

    @property
    def fy(self):
        if self.intrinsics is not None:
            return self.intrinsics[1, 1]
        return None

    @property
    def cx(self):
        if self.intrinsics is not None:
            return self.intrinsics[0, 2]
        return None

    @property
    def cy(self):
        if self.intrinsics is not None:
            return self.intrinsics[1, 2]
        return None

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
    def sensor_size_mm(self, value: tuple):
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
        """
        set intrinsics, fx/fy focal length in px, cx/cy
        optical center in px, s skew factor

        :param fx: focal length in px
        :type fx: float
        :param fy: focal length in px
        :type fy: float
        :param cx: optical center in px
        :type cx: float
        :param cy: optical center in px
        :type cy: float
        :param s: skew factor, defaults to 0
        :type s: float, optional
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
        """
        Set distortion coefficients. The coefficients are the
        opencv 5 parameter model:
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        :param k1:
        :type k1: float
        :param k2:
        :type k2: float
        :param p1:
        :type p1: float
        :param p2:
        :type p2: float
        :param k3:
        :type k3: float
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
        T1 = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
        self._RTb = self._RT @ T1
        self._RTb = np.linalg.inv(self._RTb)
        self._RTb = self._RTb @ T1

    @property
    def RTb(self):
        return self._RTb

    def set_rotation_and_translation(
            self,
            rot_3x3: np.ndarray,
            translation: np.ndarray):
        """Set 4x4 transformation matrix from rotation
        matrix and translation vector

        :param rot_3x3: 3x3 Rotation matrix
        :type rot_3x3: np.ndarray
        :param translation: translation vector
        :type translation: np.ndarray
        """
        Rt = np.zeros((4, 4), dtype=np.float32)
        Rt[0:3, 0:3] = rot_3x3
        Rt[0:3, 3] = translation.ravel()
        self.RT = Rt

    def compute_f_mm(self):
        """ Compute focal length in mm if image size and sensor size are available
        """
        if self.image_size is None or self.sensor_size_mm is None:
            return
        self.f_mm = self.f_px / self.image_size[1] * self.sensor_size_mm[1]
