import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def ensure(self, key: str, dtype: type):
        if key not in self:
            raise RuntimeError("Missing Settings Entry [{key}] Exception!")
        if not isinstance(self.__dict__[key], dtype):
            raise RuntimeError(
                "Invalid Settings Type: {key} expected to be type {type}!")

    def has_key(self, key: str) -> bool:
        return key in self.__dict__.keys()

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
