import yaml
from pathlib import Path


class Settings:

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
        if key not in self:
            raise RuntimeError("Missing Settings Entry [{key}] Exception!")
        if not isinstance(self.__dict__[key], dtype):
            raise RuntimeError(
                "Invalid Settings Type: {key} expected to be type {type}!")

    def save(self, save_dir: str, filename: str, with_timestamp=False):
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

    def from_file(self, filename: str):
        if isinstance(filename, Path):
            filename = str(filename)
        assert (isinstance(filename, str))
        assert Path(filename).is_file()

        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        self.from_params(data)

    def from_params(self, params):
        for key in params.keys():
            setattr(self, key, params[key])

    def from_config(self, config_filename: str):
        fname = Path(config_filename)
        assert fname.is_file()
        assert fname.suffix == ".yaml"
        with fname.open() as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        self.from_params(cfg)
