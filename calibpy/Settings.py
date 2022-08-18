import yaml
from pathlib import Path


class Settings:
    """General Settings Handler class. The idea is to have a class
    that can have arbitrary attributes to be set from a dict or a
    .yaml file.
    """

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def __eq__(self, other: object) -> bool:
        def get_dict(obj):
            out = {}
            for key in obj.__dict__.keys():
                if not key.startswith('__') and not callable(key):
                    if not key.startswith('_'):
                        print(key, type(key))
                        out[key] = type(obj.__dict__[key])
            return out

        obj1 = get_dict(self)
        obj2 = get_dict(other)
        for key in obj1.keys():
            if key in obj2.keys():
                if obj1[key] != obj2[key]:
                    return False
            else:
                return False
        return True

    def ensure(self, key: str, dtype: type):
        """Check if a desired attribute actually exist and has the
        expected type. If not an exception is thrown.

        :param key: attribute name
        :type key: str
        :param dtype: attribute type
        :type dtype: type
        :raises RuntimeError: Raises an exception if the key
            does not exist ot the type is wrong
        """
        if key not in self:
            raise RuntimeError("Missing Settings Entry [{key}] Exception!")
        if not isinstance(self.__dict__[key], dtype):
            raise RuntimeError(
                "Invalid Settings Type: {key} expected to be type {type}!")

    def save(self, save_dir: str, filename: str, with_timestamp=False):
        """Serializes the object and saves all protected attributes to
        a .yaml file.

        :param save_dir: output directory
        :type save_dir: str
        :param filename: output name
        :type filename: str
        :param with_timestamp: if True a timestamp is attached,
            defaults to False
        :type with_timestamp: bool, optional
        """
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            Path.mkdir(save_dir)
        if Path(filename).suffix != '':
            assert Path(filename).suffix == "yaml"
            filename = filename.split(".")[0]
        if with_timestamp:
            import time
            filename += "_" + time.strftime('%Y%m%d-%H%M%S')
        filename += ".yaml"
        filename = save_dir / filename
        with filename.open(mode="w") as file:
            import yaml
            data = {}
            for key in self.__dict__.keys():
                if not key.startswith('__') and not callable(key):
                    data[key] = self.__dict__[key]
            yaml.dump(data, file)

    def from_params(self, params: dict):
        """set attributes via dictionary. Each dict entry is
        added as class attribute and can be accessed as such
        afterwards.

        :param params: dict with attributes to be added
        :type params: dict
        """
        for key in params.keys():
            setattr(self, key, params[key])

    def from_config(self, config_filename: str):
        """Set atttributes via .yaml config files. Each entry 
        is added as class attribute and can be accessed as such
        afterwards.

        :param config_filename: Config filename, .yaml files expected
        :type config_filename: str
        """
        assert isinstance(config_filename, str) or isinstance(
            config_filename, Path)
        fname = Path(config_filename)
        assert fname.is_file()
        assert fname.suffix == ".yaml"
        with fname.open() as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        self.from_params(cfg)
