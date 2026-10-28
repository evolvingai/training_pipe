__version__ = "0.1.0"
SOURCE_REVISION = "f4914b0"


def __getattr__(name: str):
    if name == "Model":
        from trainer.detection.api import Model
        return Model
    raise AttributeError(f"module 'trainer' has no attribute {name!r}")


__all__ = ["Model", "SOURCE_REVISION", "__version__"]
