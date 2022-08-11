import cv2
import numpy as np


def get_aruco_dict(dict_key):
    if dict_key == "DICT_4X4":
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    elif dict_key == "DICT_5X5":
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    elif dict_key == "DICT_6X6":
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    elif dict_key == "DICT_7X7":
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
    else:
        raise IOError(
            f"Unknown ARUCO_DICT {dict_key}, \
                supported are [DICT_4X4, DICT_5X5, DICT_6X6, DICT_7X7]")


def create_aruco_board(dict_key, cols, rows, square_size, marker_size):
    board = cv2.aruco.CharucoBoard_create(
        squaresX=cols,
        squaresY=rows,
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
    def __init__(self,
                 dict_key: str,
                 cols: int,
                 rows: int,
                 square_size: float,
                 marker_size: float):
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
            marker_size: float) -> object:
        target = ArucoTarget(dict_key, cols, rows, square_size, marker_size)
        return target


def get_aruco_corners(img: np.ndarray,
                      aruco_target: ArucoTarget,
                      criteria: tuple) -> tuple:
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
    response, charuco_corners, charuco_ids = \
        cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img,
            board=aruco_target.board)

    return response, charuco_corners, charuco_ids, corners
