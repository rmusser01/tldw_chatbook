"""Pinned local model closure works on native directory handles and refuses aliases."""

import os

import pytest

from tldw_chatbook.RAG_Search.model_recovery import _closure


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"synthetic weights")
    return root


def test_native_closure_enumerates_pinned_directory(model):
    entries = _closure(model)
    assert any(
        name == "/model.safetensors" and len(digest) == 64 for name, digest in entries
    )
    assert {name for name, _ in entries} >= {
        "",
        "/config.json",
        "/tokenizer.json",
        "/model.safetensors",
    }


def test_native_closure_refuses_symlink_alias(model):
    alias = model / "alias.json"
    try:
        alias.symlink_to(model / "config.json")
    except OSError as error:
        pytest.skip(f"file symlink unavailable: {type(error).__name__}")
    with pytest.raises((OSError, ValueError)):
        _closure(model)


def test_native_closure_refuses_hardlinked_file(model):
    os.link(model / "config.json", model / "linked.json")
    with pytest.raises((OSError, ValueError)):
        _closure(model)


def test_native_closure_refuses_manifest_outside_path(model):
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"../outside.safetensors"}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="local_model_reference_outside_closure"):
        _closure(model)
