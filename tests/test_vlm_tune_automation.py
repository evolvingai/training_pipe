from __future__ import annotations

import json
from pathlib import Path

import pytest

from automation.vlm_tune import load_records, parse_record


def test_vlm_records_accept_flat_and_chat_formats(tmp_path: Path) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"test")
    flat = parse_record({"image": "images/one.jpg", "prompt": "What?", "answer": "A thing."}, tmp_path, 1)
    assert flat.image == image.resolve()
    chat = parse_record({"messages": [
        {"role": "user", "content": [{"type": "image", "path": "images/one.jpg"}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A thing."}]},
    ]}, tmp_path, 2)
    assert (chat.prompt, chat.answer) == ("Describe.", "A thing.")


def test_vlm_loader_validates_local_images(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"image": "missing.jpg", "prompt": "What?", "answer": "Nothing."}) + "\n")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_records(data)
    with pytest.raises(ValueError, match="remote images"):
        parse_record({"image": "https://example.com/a.jpg", "prompt": "x", "answer": "y"}, tmp_path, 1)
