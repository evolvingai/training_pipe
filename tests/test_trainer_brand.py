from trainer import Model, TrainerDetector
from trainer.detection import SOURCE_REVISION
from trainer.detection.nn.model import resolve_model_toml


def test_trainer_is_primary_detector_namespace() -> None:
    assert TrainerDetector is Model
    path, scale = resolve_model_toml("trainer-csp-s")
    assert path.name == "trainer-csp.toml"
    assert scale == "s"
    path, scale = resolve_model_toml("trainer-p2-n")
    assert path.name == "trainer-p2.toml"
    assert scale == "n"
    assert SOURCE_REVISION == "f4914b0"
    assert Model("trainer-n").net.model_name == "trainer-n"
