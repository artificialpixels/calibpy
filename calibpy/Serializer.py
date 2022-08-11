#!/usr/bin/env python3

import pickle
import tempfile
from pathlib import Path


class Serializer:
    """
    Base class for objects that needs to be serializable. When
    deriving from this class, the child class inherits the functions
    serialize and load, which allow to serialize all protected
    class attributes as dictionary and as .npy file. The latter
    can be used to instantiate the class from file.
    """

    def __init__(self):
        print("Serializer initialized!")

    def __str__(self):
        string = ""
        data = self.serialize()
        for key, value in data.items():
            string += f"{key} | {value} | {type(value)}\n"
        return string

    def serialize(self, filename: str = None) -> dict:
        """The function serializes all protected class attributes and
        returns a dictionary. When passing a filename argument, the
        serialized object is dumped to a .npy file.

        :param filename: dump filename., defaults to None
        :type filename: str, optional
        :return: serialized dict keeping all protected class attributes
        :rtype: dict
        """
        if filename is not None:
            if isinstance(filename, Path):
                filename = str(filename)
            assert isinstance(filename, str)

        data = {}
        for key in self.__dict__.keys():
            if not key.startswith('__') and not callable(key):
                if key.startswith('_'):
                    print(key, type(key))
                    data[key[1:]] = self.__dict__[key]
        if filename is None:
            self.root = tempfile.gettempdir()
            name = self.__class__.__name__
            filename = str(Path(self.root) / name)
        if isinstance(filename, str):
            if not filename.endswith(".npy"):
                filename += ".npy"
            self.location = filename
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        return data

    def load(self, filename: str):
        """Loads a .npy file and creates a class attribute
        for each entry in the dictionary loaded.

        :param filename: dump file filename
        :type filename: str
        """
        assert Path(filename).is_file()
        assert Path(filename).suffix == ".npy"
        with open(filename, "rb") as f:
            data = pickle.load(f)
        for key, value in data.items():
            setattr(self, "_"+key, value)
