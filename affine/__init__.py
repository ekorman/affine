from importlib import metadata

try:
    __version__ = metadata.version("affine")
except Exception:
    __version__ = "0.0.0-dev"
del metadata
