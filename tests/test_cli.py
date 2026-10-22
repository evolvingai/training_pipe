import pytest

from trainer.detection.cli import main, parse_kv


def test_parse_kv_types():
    out = parse_kv(["epochs=100", "lr0=0.01", "amp=false", "name=exp2", "imgsz=640"])
    assert out == {"epochs": 100, "lr0": 0.01, "amp": False, "name": "exp2", "imgsz": 640}


def test_parse_kv_rejects_positional():
    with pytest.raises(SystemExit):
        parse_kv(["notakv"])


def test_help_and_unknown():
    assert main([]) == 0
    assert main(["help"]) == 0
    assert main(["frobnicate"]) == 2


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_train_requires_data():
    with pytest.raises(SystemExit):
        main(["train", "model=trainer-n"])


def test_cli_train_and_val(tiny_dataset, tmp_path):
    code = main(["train", f"data={tiny_dataset}", "model=trainer-n", "epochs=1",
                 "imgsz=96", "batch=4", "workers=0", "device=cpu", "amp=false",
                 "mosaic=0.0", f"project={tmp_path}", "name=cli", "warmup_epochs=0.5"])
    assert code == 0
    best = tmp_path / "cli" / "best.pt"
    assert best.exists()
    assert main(["val", f"model={best}", f"data={tiny_dataset}", "imgsz=96",
                 "batch=4", "workers=0", "device=cpu"]) == 0


def test_cli_viz_augment(tiny_dataset, tmp_path):
    out = tmp_path / "aug.png"
    code = main(["viz-augment", f"data={tiny_dataset}", "n=4", f"out={out}",
                 "imgsz=96", "workers=0", "mosaic=0.5"])
    assert code == 0
    assert out.exists()
