import os
import cv2
import Imath
import numpy as np
import OpenEXR as exr
from pathlib import Path

from abc import abstractmethod


STREAM_FILETYPES = ["png", "jpg", "jpeg", "tif", "tiff", "exr"]


class Stream:

    def __init__(self, is_looping=True):

        self._current_frame = -1
        self._is_looping = is_looping

    @property
    def current_frame(self):
        return self._current_frame

    @property
    def is_looping(self):
        return self._is_looping

    @is_looping.setter
    def is_looping(self, value: bool):
        assert isinstance(value, bool)
        self._is_looping = value

    def reset(self):
        """Reset the frame counter to the first frame
        """
        self._current_frame = -1

    @abstractmethod
    def initialize(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "Please derive from this class and do not use it directly!")

    @abstractmethod
    def get(self, index: int, *args, **kwargs):
        raise NotImplementedError(
            "Please derive from this class and do not use it directly!")

    @abstractmethod
    def next(self, *args, **kwargs):
        raise NotImplementedError(
            "Please derive from this class and do not use it directly!")


class FileStream(Stream):
    """Implementation of a Stream class that handles loading from file tasks.
    The initialize method can handle loading a single specific filename, a list
    of filenames or loading files from a directory. See the doc strings of the 
    initialize method for more details.
    """

    def __init__(self, is_looping=False):
        super().__init__(is_looping)
        self._filenames = []

    @property
    def length(self):
        return len(self._filenames)

    @property
    def filenames(self):
        return self._filenames

    def current_filename(self) -> str:
        """Get the filename of the current frame

        :return: filename
        :rtype: str
        """
        if self._current_frame >= 0 and self._current_frame < self.length:
            return self._filenames[self._current_frame]
        return None

    @staticmethod
    def load_image(
            filename: str,
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
            return FileStream.exrchannel2numpy(filename)
        else:
            return cv2.imread(filename, flag)

    @staticmethod
    def exrchannel2numpy(
            filename: str,
            channel_name="R") -> np.ndarray:
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
        channel = np.frombuffer(
            channel_str, dtype=np.float32).reshape(size[1], -1)
        return (channel)

    @staticmethod
    def sort_filenames(filenames):
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

        filenames.sort(key=lambda x: filename_splitter(x))
        return filenames

    @staticmethod
    def filenames_from_directory(
            dir: str,
            prefix: str = None,
            suffix: str = None):
        assert Path(dir).is_dir()

        from glob import glob
        filenames = []
        for fname in glob(dir + os.sep + "*"):
            if fname.split(".")[-1].lower() in STREAM_FILETYPES:
                if suffix is None and prefix is None:
                    filenames.append(fname)
                    continue
                if suffix is not None and prefix is not None:
                    name = Path(fname).name
                    name = name.split(".")[0]
                    split = name.split("_")
                    if len(split) == 3 and \
                            split[0] == prefix and \
                            split[-1] == suffix:
                        filenames.append(fname)
                elif suffix is not None:
                    name = Path(fname).name
                    name = str(name).split(".")[0]
                    split = name.split("_")
                    if len(split) > 1 and split[-1] == suffix:
                        filenames.append(fname)
                elif prefix is not None:
                    name = Path(fname).name
                    name = str(name).split(".")[0]
                    split = name.split("_")
                    if len(split) > 1 and split[0] == prefix:
                        filenames.append(fname)
        assert len(filenames) > 0, f"No images found in {dir}"
        return FileStream.sort_filenames(filenames)

    def _from_list(self, filenames: list):
        assert isinstance(filenames, list)
        assert len(filenames) > 0
        self._filenames = filenames

    def _from_dir(
            self,
            directory: str,
            from_frame: int,
            to_frame: int,
            prefix: str,
            suffix: str):
        if isinstance(directory, Path):
            directory = str(directory)
        assert isinstance(directory, str)
        assert Path(directory).is_dir()
        print("Load from dir: ", directory)
        filenames = FileStream.filenames_from_directory(
            directory, prefix=prefix, suffix=suffix)
        if from_frame > 0:
            filenames = filenames[from_frame:]
        if to_frame > from_frame:
            filenames = filenames[:to_frame-from_frame]
        self._from_list(filenames)

    def _from_filename(self, filename):
        if isinstance(filename, Path):
            filename = str(filename)
        assert isinstance(filename, str)
        assert Path(filename).is_file()
        print("Load image from file: ", filename)
        self._from_list([filename])

    def initialize(self, *args, **kwargs) -> bool:
        print("Initialize FileStream:")
        if "directory" in kwargs.keys():
            from_frame = 0
            to_frame = 0
            prefix = None
            suffix = None
            if "from_frame" in kwargs.keys():
                from_frame = kwargs["from_frame"]
            if "to_frame" in kwargs.keys():
                to_frame = kwargs["to_frame"]
            if "prefix" in kwargs.keys():
                prefix = kwargs["prefix"]
            if "suffix" in kwargs.keys():
                suffix = kwargs["suffix"]
            self._from_dir(
                directory=kwargs["directory"],
                from_frame=from_frame,
                to_frame=to_frame,
                prefix=prefix,
                suffix=suffix)
            if len(self._filenames) > 0:
                return True
        elif "filename" in kwargs.keys():
            self._from_filename(kwargs["filename"])
        elif "filenames" in kwargs.keys():
            self._from_list(kwargs["filenames"])

        if len(self._filenames) > 0:
            return True
        return False

    def get(self, index: int = None, *args, **kwargs) -> np.ndarray:
        """Access an arbitrary frame of the stream. If the index
        passed is out of range, None is returned. If the index is None,
        the image at the current_frame pointer is returned. An additional
        parameter is flag to specify the opencv imread flag. Default
        value is cv2.IMREAD_GRAYSCALE

        :param index: frame pointer
        :type index: int
        :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
        :type index: int
        :return: image
        :rtype: np.ndarray
        """
        if index is None:
            index = self._current_frame
        flag = cv2.IMREAD_GRAYSCALE
        if "flag" in kwargs.keys():
            flag = kwargs["flag"]
        if 0 <= index < self.length:
            fname = self._filenames[index]
            self._current_frame = index
            return FileStream.load_image(fname, flag)
        return None

    def next(self, *args, **kwargs) -> np.ndarray:
        """Function returns the next image from the streams buffer.
        If the buffer is empty and is_looping is set to True, the buffer 
        is automtaically resetted, otherwise None is returned. The buffer 
        can aslo be resetted manually using the method reset.

        :return: image
        :rtype: np.ndarray
        """
        flag = cv2.IMREAD_GRAYSCALE
        if "flag" in kwargs.keys():
            flag = kwargs["flag"]
        self._current_frame += 1
        if self.current_frame >= self.length:
            if self._is_looping:
                self._current_frame = 0
            else:
                return None
        fname = self._filenames[self._current_frame]
        return FileStream.load_image(fname, flag)


# class Stream:
#     """
#     This class implements as lazy loading Image Streamer. It can
#     read a subset of images within a directory, depending on file-
#     name pre- and suffixes. If pre- and/or suffixes are specified,
#     it reminds all the filenames within a directory fullfilling the
#     corresponding filename structure. If neither pre- nor suffixes
#     are specified, all image filenames are loaded. Using the method
#     next, the next image is physically loaded and returned.
#     """

#     def __init__(self,
#                  dir: str = None,
#                  prefix: str = None,
#                  suffix: str = None):
#         """
#         :param dir: directory with image files, defaults to None
#         :type dir: str, optional
#         :param prefix: filename prefix [pre_]000x..., defaults to None
#         :type prefix: str, optional
#         :param suffix: filename suffix ...000x_[suf], defaults to None
#         :type suffix: str, optional
#         """
#         self._dir = None
#         self._filenames = []
#         self._current_frame = -1

#         if dir is not None:
#             self.load(dir=dir, prefix=prefix, suffix=suffix)

#     def __str__(self):
#         return f"Stream:\n{self._dir}\nframes: {self.length}"

#     @staticmethod
#     def load_image(filename: str,
#                    flag: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
#         """Loading a single image using opencv flags
#         https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html)

#         :param filename: image filename
#         :type filename: str
#         :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
#         :type flag: int, optional
#         :return: numpy image
#         :rtype: np.ndarray
#         """
#         assert Path(filename).is_file
#         if filename.split(".")[-1] == "exr":
#             return Stream.exrchannel2numpy(filename)
#         else:
#             return cv2.imread(filename, flag)

#     @staticmethod
#     def exrchannel2numpy(filename: str, channel_name="R") -> np.ndarray:
#         """Loading a single channel from a .ext file.

#         :param filename: filename
#         :type filename: str
#         :param channel_name: channel name, defaults to "R"
#         :type channel_name: str, optional
#         :return: numpy single channel image
#         :rtype: np.ndarray
#         """
#         assert Path(filename).is_file
#         file = exr.InputFile(filename)
#         dw = file.header()['dataWindow']
#         size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#         Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
#         channel_str = file.channel(channel_name, Float_Type)
#         channel = np.fromstring(
#             channel_str, dtype=np.float32).reshape(size[1], -1)
#         return (channel)

#     @property
#     def length(self):
#         return len(self._filenames)

#     @property
#     def filenames(self):
#         return self._filenames

#     def reset(self):
#         """Reset the frame counter to the first frame
#         """
#         self._current_frame = -1

#     def set_frame(self, frame: int = 0):
#         """Set the frame pointer to a specific frame

#         :param frame: frame number [0, length[, defaults to 0
#         :type frame: int, optional
#         """
#         assert frame < 0 or frame >= self.length
#         self._current_frame = frame-1

#     def current_filename(self) -> str:
#         """Get the filename of the current frame

#         :return: filename
#         :rtype: str
#         """
#         if self._current_frame >= 0 and self._current_frame < self.length:
#             return self._filenames[self._current_frame]
#         return None

#     def _sort_filenames(self):
#         """Do numeric sorting of the _filename list,
#         precondition is that filenames follow the
#         pattern [prefix_]000x[_suffix]
#         """
#         def filename_splitter(filename):
#             name = Path(filename).name
#             name = str(name).split(".")[0]
#             if "_" not in name:
#                 return int(name)
#             name_split = name.split("_")
#             if len(name_split) == 3:
#                 return name_split[1]
#             elif len(name_split) == 2:
#                 name = name_split[1]
#                 try:
#                     num = int(name_split[1])
#                     return num
#                 except ValueError:
#                     try:
#                         num = int(name_split[0])
#                         return num
#                     except ValueError:
#                         raise IOError(
#                             "Unknown naming convention! Expecting: 000x,\
#                             suffix_000x, 000x_prefix or suffix_000x_prefix")

#         self._filenames.sort(key=lambda x: filename_splitter(x))

#     def load(self,
#              dir: str,
#              prefix: str = None,
#              suffix: str = None):
#         """Loading a stream from a directory, if pre- and suffix
#         is None, all filenames are read, otherwise just filenames
#         with the respective pre- and/or suffix are loaded.

#         :param dir: directory with image files
#         :type dir: str
#         :param prefix: filename prefix prefix_000x..., defaults to None
#         :type prefix: str, optional
#         :param suffix: filename suffix ...000x_suffix, defaults to None
#         :type suffix: str, optional
#         """
#         assert Path(dir).is_dir()
#         self._dir = str(dir)

#         from glob import glob
#         for fname in glob(self._dir + os.sep + "*"):
#             if fname.split(".")[-1].lower() in STREAM_FILETYPES:
#                 if suffix is None and prefix is None:
#                     self._filenames.append(fname)
#                     continue
#                 if suffix is not None and prefix is not None:
#                     name = Path(fname).name
#                     name = name.split(".")[0]
#                     split = name.split("_")
#                     if len(split) == 3 and \
#                             split[0] == prefix and \
#                             split[-1] == suffix:
#                         self._filenames.append(fname)
#                 elif suffix is not None:
#                     name = Path(fname).name
#                     name = str(name).split(".")[0]
#                     split = name.split("_")
#                     if len(split) > 1 and split[-1] == suffix:
#                         self._filenames.append(fname)
#                 elif prefix is not None:
#                     name = Path(fname).name
#                     name = str(name).split(".")[0]
#                     split = name.split("_")
#                     if len(split) > 1 and split[0] == prefix:
#                         self._filenames.append(fname)
#         assert len(self._filenames) > 0, f"No images found in {self._dir}"
#         self._sort_filenames()

#     def get(self, frame_number: int, flag: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
#         """Read the frame with the the frame_number passed

#         :param frame_number: frame_number [0, stream.length-1]
#         :type frame_number: int
#         :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
#         :type flag: int, optional
#         :return: image
#         :rtype: np.ndarray
#         """
#         if 0 <= frame_number < self.length:
#             fname = self._filenames[frame_number]
#             return Stream.load_image(fname, flag)
#         return None

#     def next(self, flag: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
#         """Read the next frame

#         :param flag: opencv imread flag, defaults to cv2.IMREAD_GRAYSCALE
#         :type flag: int, optional
#         :return: image
#         :rtype: np.ndarray
#         """
#         self._current_frame += 1
#         if self._current_frame >= self.length:
#             return None
#         fname = self._filenames[self._current_frame]
#         return Stream.load_image(fname, flag)


# if __name__ == "__main__":
#     fs = FileStream()
#     fs.initialize(file_dir="bla")
#     fs.next()
#     fs.get(0)
