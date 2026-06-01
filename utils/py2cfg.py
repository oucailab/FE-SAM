import sys
import pydoc
from addict import Dict
from typing import Union
from pathlib import Path
from importlib import import_module


class config_dict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def py2dict(file_path: Union[str, Path]) -> dict:
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)

    return {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}


def py2cfg(file_path: Union[str, Path]) -> config_dict:

    return config_dict(py2dict(file_path))


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
        
    return pydoc.locate(object_type)(**kwargs)


