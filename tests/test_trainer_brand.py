from trainer import Model, TrainerDetector
from trainer.detection.nn.model import resolve_model_toml


def test_trainer_is_primary_detector_namespace() -> None:
    assert TrainerDetector is Model
    path, scale = resolve_model_toml("trainer-csp-s")
    assert path.name == "trainer-csp.toml"
    assert scale == "s"
    assert Model("trainer-n").net.model_name == "trainer-n"
