from pathlib import Path
from pymodaq.utils.logger import set_logger  # to be imported by other modules.


from importlib import metadata
from importlib.metadata import PackageNotFoundError


from .utils import Config
config = Config()

try:
    __version__ = metadata.version(__package__)
except PackageNotFoundError:
    __version__ = '0.0.0dev'
