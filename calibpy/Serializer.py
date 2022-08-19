"""
:Copyrights: Artificial Pixels
:Author: Sven Wanner (artificial.pixels@gmail.com)
:Sponsor: SpexAI GmbH
"""
import json
import yaml
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

    def __str__(self):
        string = f"{self.__class__.__name__}:\n"
        string += "-----------------------------------\n"
        data = self.serialize()
        for key, value in data.items():
            string += f" - {key[1:]} | {type(value)}:\n"
            string += f"\t{value}\n"
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
            if isinstance(filename, str):
                filename = Path(filename)
            assert isinstance(filename, Path)

        data = {}
        for key in self.__dict__.keys():
            if not key.startswith('__') and not callable(key):
                if key.startswith('_'):
                    data[key[1:]] = self.__dict__[key]
        if filename is not None:
            self.write(filename, data)
        return data

    def load(self, filename: str):
        """
        load attributes from file, supported are 
        pickle .npy files, .yaml and .json files

        :param filename: filename
        :type filename: str
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert isinstance(filename, Path)
        assert filename.is_file()
        if filename.suffix == ".npy":
            self._load_npy(filename)
        elif filename.suffix == ".yaml":
            self._load_yaml(filename)
        elif filename.suffix == ".json":
            self._load_json(filename)

    def write(self, filename: str, data: dict):
        """
        write attributes to file, supported are 
        pickle .npy files, .yaml and .json files

        :param filename: filename
        :type filename: str
        """
        assert isinstance(data, dict)
        if isinstance(filename, str):
            filename = Path(filename)
        assert isinstance(filename, Path)
        if filename.suffix == ".npy":
            self._write_npy(filename, data)
        if filename.suffix == ".yaml":
            self._write_yaml(filename, data)
        if filename.suffix == ".json":
            self._write_json(filename, data)

    def from_dict(self, data: dict):
        """Load attributes from a dictionary

        :param data: input data dict
        :type data: dict
        """
        assert isinstance(data, dict)
        for key, value in data.items():
            if not key.startswith("_"):
                key = "_"+key
            setattr(self, key, value)

    def _write_npy(self, filename: Path, data: dict):
        """Write pickle .npy file

        :param filename: Filename
        :type filename: Path
        :param data: input data dict
        :type data: dict
        """
        with filename.open(mode="wb") as f:
            pickle.dump(data, f)
            print(f"File {str(filename)} saved")

    def _write_yaml(self, filename: Path, data: dict):
        """Write .yaml file

        :param filename: Filename
        :type filename: Path
        :param data: input data dict
        :type data: dict
        """
        with filename.open(mode="w") as f:
            yaml.dump(data, f)
            print(f"File {str(filename)} saved")

    def _write_json(self, filename: Path, data: dict):
        """Write .json file

        :param filename: Filename
        :type filename: Path
        :param data: input data dict
        :type data: dict
        """
        with filename.open(mode="w") as f:
            json.dump(data, f)
            print(f"File {str(filename)} saved")

    def _load_npy(self, filename: Path):
        """Loads a .npy file and creates a class attribute
        for each entry in the dictionary loaded.

        :param filename: dump file filename
        :type filename: Path
        """
        with filename.open(mode="rb") as f:
            data = pickle.load(f)
        self.from_dict(data)

    def _load_yaml(self, filename: Path):
        """Loads a .yaml file and creates a class attribute
        for each entry in the dictionary loaded.

        :param filename: dump file filename
        :type filename: Path
        """
        with filename.open(mode="rb") as f:
            data = yaml.safe_load(f)
        self.from_dict(data)

    def _load_json(self, filename: Path):
        """Loads a .json file and creates a class attribute
        for each entry in the dictionary loaded.

        :param filename: dump file filename
        :type filename: Path
        """
        with filename.open(mode="rb") as f:
            data = json.load(f)
        self.from_dict(data)
