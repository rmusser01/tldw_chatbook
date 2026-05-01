"""Regression coverage for first-run startup polish issues."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest
from loguru import logger


def test_splash_effect_modules_import_without_annotation_name_errors() -> None:
    """Splash effects should not fail import due to runtime-only typing names."""
    modules = [
        "tldw_chatbook.Utils.Splash_Screens.classic.scrolling_credits",
        "tldw_chatbook.Utils.Splash_Screens.tech.terminal_boot",
    ]

    for module_name in modules:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


def test_nltk_download_false_is_not_logged_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_chatbook.Chunking import Chunk_Lib

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(lambda message: messages.append((message.record["level"].name, message.record["message"])))
    try:
        fake_nltk = SimpleNamespace(
            data=SimpleNamespace(find=lambda _path: (_ for _ in ()).throw(LookupError("missing"))),
            download=lambda _package: False,
        )

        monkeypatch.setattr(Chunk_Lib, "NLTK_AVAILABLE", True)
        monkeypatch.setattr(Chunk_Lib, "nltk", fake_nltk)

        Chunk_Lib.ensure_nltk_data()
    finally:
        logger.remove(sink_id)

    assert not any("downloaded successfully" in message for _level, message in messages)
    assert any(level in {"WARNING", "ERROR"} and "punkt" in message for level, message in messages)


def test_missing_openai_tts_mapping_falls_back_without_error_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import resources as importlib_resources
    from tldw_chatbook import config

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(lambda message: messages.append((message.record["level"].name, message.record["message"])))
    original_open = builtins.open
    try:
        monkeypatch.setattr(
            importlib_resources,
            "files",
            lambda _package: (_ for _ in ()).throw(FileNotFoundError("packaged mapping missing")),
        )

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("openai_tts_mappings.json"):
                raise FileNotFoundError("filesystem mapping missing")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)

        mappings = config.load_openai_mappings()
    finally:
        logger.remove(sink_id)

    assert mappings["models"]["tts-1"] == "openai_official_tts-1"
    assert mappings["voices"]["alloy"] == "alloy"
    assert not any(level == "ERROR" for level, _message in messages)
