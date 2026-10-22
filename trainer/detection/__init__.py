__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "Model":
        from trainer.detection.api import Model
        return Model
    raise AttributeError(f"module 'trainer.detection' has no attribute {name!r}")


__all__ = ["Model", "__version__"]
