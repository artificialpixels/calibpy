"""
:Copyrights: Artificial Pixels
:Author: Sven Wanner (artificial.pixels@gmail.com)
:Sponsor: SpexAI GmbH
"""

import cv2
import numpy as np
from packaging import version

def get_aruco_dict(dict_key: str) -> int:
    """returns the opencv aruco dict identifier from settings string

    :param dict_key: identifier string DICT_NXN N=[4,5,6,7]
    :type dict_key: str
    :raises IOError: if identifier not found
    :return: aruco dict identifier
    :rtype: int
    """
    opencv_version = version.parse(cv2.__version__)
    version_4_7 = version.parse("4.7.0")
    if opencv_version < version_4_7:
        if dict_key == "DICT_4X4":
            return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        elif dict_key == "DICT_5X5":
            return cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        elif dict_key == "DICT_6X6":
            return cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
        elif dict_key == "DICT_7X7":
            return cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
        else:
            raise IOError(
                f"Unknown ARUCO_DICT {dict_key}, \
                supported are [DICT_4X4, DICT_5X5, DICT_6X6, DICT_7X7]")
    else:
        if dict_key == "DICT_4X4":
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        elif dict_key == "DICT_5X5":
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        elif dict_key == "DICT_6X6":
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        elif dict_key == "DICT_7X7":
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        else:
            raise IOError(
                f"Unknown ARUCO_DICT {dict_key}, \
                supported are [DICT_4X4, DICT_5X5, DICT_6X6, DICT_7X7]")


def create_aruco_board(
        dict_key: str,
        cols: int,
        rows: int,
        square_size: float,
        marker_size: float) -> tuple:
    """create aruco board descriptors: cv2.aruco.CharucoBoard instance,
    board 3D point and id list

    :param dict_key: identifier string DICT_NXN N=[4,5,6,7]
    :type dict_key: str
    :param cols: number of calibration target columns
    :type cols: int
    :param rows: number of calibration target rows
    :type rows: int
    :param square_size: size of a checkerboard square in m
    :type square_size: float
    :param marker_size: size of a aruco target in m
    :type marker_size: float
    :return: target board descriptors
    :rtype: cv2.aruco.CharucoBoard instance, points_3d, ids
    """
    opencv_version = version.parse(cv2.__version__)
    version_4_7 = version.parse("4.7.0")
    if opencv_version < version_4_7:
        board = cv2.aruco.CharucoBoard_create(
            squaresX=cols,
            squaresY=rows,
            squareLength=square_size,
            markerLength=marker_size,
            dictionary=get_aruco_dict(dict_key))
    else:
        board = cv2.aruco.CharucoBoard(
            (cols, rows),
            squareLength=square_size,
            markerLength=marker_size,
            dictionary=get_aruco_dict(dict_key))

    N = (cols-1) * (rows-1)
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols-1, 0:rows-1].T.reshape(-1, 2)
    objp[:] *= square_size
    objp_ids = np.arange(0, N, 1, dtype=np.int32)
    return board, objp, objp_ids


class ArucoTarget:
    """Class providing all necessary aruco target data,
    dict key, aruco board instance, 3d points and ids
    """

    def __init__(self,
                 dict_key: str,
                 cols: int,
                 rows: int,
                 square_size: float,
                 marker_size: float):
        """
        :param dict_key: identifier string DICT_NXN N=[4,5,6,7]
        :type dict_key: str
        :param cols: number of calibration target columns
        :type cols: int
        :param rows: number of calibration target rows
        :type rows: int
        :param square_size: size of a checkerboard square in m
        :type square_size: float
        :param marker_size: size of a aruco target in m
        :type marker_size: float
        """
        self._dict_key = dict_key
        self._dict = get_aruco_dict(dict_key)
        self._board, self._points, self._ids = create_aruco_board(
            dict_key, cols, rows, square_size, marker_size)

    @property
    def dict(self):
        return self._dict

    @property
    def board(self):
        return self._board

    @staticmethod
    def get(dict_key: str,
            cols: int,
            rows: int,
            square_size: float,
            marker_size: float) -> 'ArucoTarget':
        """_summary_

        :param dict_key: identifier string DICT_NXN N=[4,5,6,7]
        :type dict_key: str
        :param cols: number of calibration target columns
        :type cols: int
        :param rows: number of calibration target rows
        :type rows: int
        :param square_size: size of a checkerboard square in m
        :type square_size: float
        :param marker_size: size of a aruco target in m
        :type marker_size: float
        :return: ArucoTarget instance
        :rtype: ArucoTarget
        """
        target = ArucoTarget(dict_key, cols, rows, square_size, marker_size)
        return target


def get_aruco_corners(img: np.ndarray,
                      aruco_target: ArucoTarget,
                      criteria: tuple) -> tuple:
    """Finds corners on aruco board

    :param img: input image
    :type img: np.ndarray
    :param aruco_target: ArucoTarget instance
    :type aruco_target: ArucoTarget
    :param criteria: search criteria
    :type criteria: tuple
    :return: response, charuco_corners, charuco_ids, corners
    :rtype: tuple
    """
    # find aruco markers in the query image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_target.dict)

    # if none found, take next image
    if ids is None:
        return None

    # apply subpix optimization
    for corner in corners:
        cv2.cornerSubPix(img, corner, winSize=(
            3, 3), zeroZone=(-1, -1), criteria=criteria)

    # get charuco corners and ids from detected aruco markers
    num_corners, charuco_corners, charuco_ids = \
        cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img,
            board=aruco_target.board)

    return num_corners, charuco_corners, charuco_ids, corners
