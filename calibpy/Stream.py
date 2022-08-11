import os
import cv2
import Imath
import numpy as np
import OpenEXR as exr
from pathlib import Path


STREAM_FILETYPES = ["png", "jpg", "jpeg", "tif", "tiff", "exr"]


class Stream:
    """
    This class implements as lazy loading Image Streamer. It can
    read a subset of images within a directory, depending on file-
    name pre- and suffixes. If pre- and/or suffixes are specified,
    it reminds all the filenames within a directory fullfilling the
    corresponding filename structure. If neither pre- nor suffixes
    are specified, all image filenames are loaded. Using the method
    next, the next image is physically loaded and returned.
    """

    def __init__(self,
                 dir: str = None,
                 prefix: str = None,
                 suffix: str = None):
        """
        :param dir: directory with image files, defaults to None
        :type dir: str, optional
        :param prefix: filename prefix [pre_]000x..., defaults to None
        :type prefix: str, optional
        :param suffix: filename suffix ...000x_[suf], defaults to None
        :type suffix: str, optional
        """
        self._dir = None
        self._filenames = []
        self._current_frame = -1

        if dir is not None:
            self.load(dir=dir, prefix=prefix, suffix=suffix)

    def __str__(self):
        return f"Stream:\n{self._dir}\nframes: {self.length}"

    @staticmethod
    def load_image(filename: str,
                   flag: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        """Loading a single image using opencv flags 
        https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html)

        :param filename: image filename
        :type filename: str
        :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
        :type flag: int, optional
        :return: numpy image
        :rtype: np.ndarray
        """
        assert Path(filename).is_file
        if filename.split(".")[-1] == "exr":
            return Stream.exrchannel2numpy(filename)
        else:
            return cv2.imread(filename, flag)

    @staticmethod
    def exrchannel2numpy(filename: str, channel_name="R") -> np.ndarray:
        """Loading a single channel from a .ext file.

        :param filename: filename
        :type filename: str
        :param channel_name: channel name, defaults to "R"
        :type channel_name: str, optional
        :return: numpy single channel image
        :rtype: np.ndarray
        """
        assert Path(filename).is_file
        file = exr.InputFile(filename)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_str = file.channel(channel_name, Float_Type)
        channel = np.fromstring(
            channel_str, dtype=np.float32).reshape(size[1], -1)
        return (channel)

    @property
    def length(self):
        return len(self._filenames)

    @property
    def filenames(self):
        return self._filenames

    def reset(self):
        """Reset the frame counter to the first frame
        """
        self._current_frame = -1

    def set_frame(self, frame: int = 0):
        """Set the frame pointer to a specific frame

        :param frame: frame number [0, length[, defaults to 0
        :type frame: int, optional
        """
        assert frame < 0 or frame >= self.length
        self._current_frame = frame-1

    def current_filename(self) -> str:
        """Get the filename of the current frame

        :return: filename
        :rtype: str
        """
        if self._current_frame >= 0 and self._current_frame < self.length:
            return self._filenames[self._current_frame]
        return None

    def _sort_filenames(self):
        """Do numeric sorting of the _filename list, precondition is that 
        filenames follow the pattern [prefix_]000x[_suffix]
        """
        def filename_splitter(filename):
            name = Path(filename).name
            name = str(name).split(".")[0]
            if "_" not in name:
                return int(name)
            name_split = name.split("_")
            if len(name_split) == 3:
                return name_split[1]
            elif len(name_split) == 2:
                name = name_split[1]
                try:
                    num = int(name_split[1])
                    return num
                except ValueError:
                    try:
                        num = int(name_split[0])
                        return num
                    except ValueError:
                        raise IOError(
                            "Unknown naming convention! Expecting: 000x,\
                            suffix_000x, 000x_prefix or suffix_000x_prefix")

        self._filenames.sort(key=lambda x: filename_splitter(x))

    def load(self,
             dir: str,
             prefix: str = None,
             suffix: str = None):
        """Loading a stream from a directory, if pre- and suffix
        is None, all filenames are read, otherwise just filenames
        with the respective pre- and/or suffix are loaded.

        :param dir: directory with image files
        :type dir: str
        :param prefix: filename prefix [pre_]000x..., defaults to None
        :type prefix: str, optional
        :param suffix: filename suffix ...000x_[suf], defaults to None
        :type suffix: str, optional
        """
        assert Path(dir).is_dir()
        self._dir = str(dir)

        from glob import glob
        for fname in glob(self._dir + os.sep + "*"):
            if fname.split(".")[-1].lower() in STREAM_FILETYPES:
                if suffix is None and prefix is None:
                    self._filenames.append(fname)
                    continue
                if suffix is not None and prefix is not None:
                    name = Path(fname).name
                    name = name.split(".")[0]
                    split = name.split("_")
                    if len(split) == 3 and \
                            split[0] == prefix and \
                            split[-1] == suffix:
                        self._filenames.append(fname)
                elif suffix is not None:
                    name = Path(fname).name
                    name = str(name).split(".")[0]
                    split = name.split("_")
                    if len(split) > 1 and split[-1] == suffix:
                        self._filenames.append(fname)
                elif prefix is not None:
                    name = Path(fname).name
                    name = str(name).split(".")[0]
                    split = name.split("_")
                    if len(split) > 1 and split[0] == prefix:
                        self._filenames.append(fname)
        assert len(self._filenames) > 0, f"No images found in {self._dir}"
        self._sort_filenames()

    def next(self, flag: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        """Read the next frame

        :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
        :type flag: int, optional
        :return: image
        :rtype: np.ndarray
        """
        self._current_frame += 1
        if self._current_frame >= self.length:
            return None
        fname = self._filenames[self._current_frame]
        return Stream.load_image(fname, flag)
