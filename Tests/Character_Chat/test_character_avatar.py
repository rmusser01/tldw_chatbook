"""Tests for the UI-free avatar resolution helper (TASK-32954, Task 2)."""

import base64
from pathlib import Path

from tldw_chatbook.Character_Chat import character_avatar as ca

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
CARD = {"name": "Aria", "description": "A silver-haired pilot", "personality": "wry"}


def test_file_avatar_reads_valid_image(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG)
    outcome = ca.resolve_avatar({"source": "file", "path": str(path)}, CARD)
    assert outcome.kind == "image" and outcome.image == PNG


def test_file_avatar_rejects_non_image_bytes(tmp_path):
    path = tmp_path / "fake.png"
    path.write_text("not an image")
    outcome = ca.resolve_avatar({"source": "file", "path": str(path)}, CARD)
    assert outcome.kind == "failed" and "not an image" in outcome.reason


def test_file_avatar_rejects_bad_suffix_and_size(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_bytes(PNG)
    assert ca.resolve_avatar({"source": "file", "path": str(txt)}, CARD).kind == "failed"
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"\0" * ca.AVATAR_MAX_BYTES)
    assert "5 MB" in ca.resolve_avatar({"source": "file", "path": str(big)}, CARD).reason


def test_generate_uses_card_prompt_when_none_given():
    prompts = []
    outcome = ca.resolve_avatar(
        {"source": "generate"}, CARD, generate=lambda p: prompts.append(p) or PNG
    )
    assert outcome.kind == "image" and "silver-haired" in prompts[0]


def test_generate_failure_is_reported():
    def boom(_prompt):
        raise RuntimeError("backend down")

    outcome = ca.resolve_avatar({"source": "generate", "prompt": "x"}, CARD, generate=boom)
    assert outcome.kind == "failed" and "generation failed" in outcome.reason


def test_remove():
    assert ca.resolve_avatar({"source": "remove"}, CARD).kind == "remove"


def test_generate_without_description_or_prompt_is_reported():
    # Never raises (fix round 1, Important finding #1): a blank card
    # description with no explicit prompt used to let ValueError from
    # compose_expression_prompt escape resolve_avatar.
    blank_card = {"name": "Aria", "description": "", "personality": "wry"}
    outcome = ca.resolve_avatar({"source": "generate"}, blank_card, generate=lambda p: PNG)
    assert outcome.kind == "failed"
    assert outcome.reason == "add a description or give an avatar prompt"


def test_file_avatar_read_error_is_reported(tmp_path, monkeypatch):
    # Never raises (fix round 1, Important finding #2): an OSError raised
    # between the is_file() check and the actual read used to escape
    # resolve_avatar instead of becoming a failed outcome.
    path = tmp_path / "a.png"
    path.write_bytes(PNG)

    def boom(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", boom)
    outcome = ca.resolve_avatar({"source": "file", "path": str(path)}, CARD)
    assert outcome.kind == "failed"
