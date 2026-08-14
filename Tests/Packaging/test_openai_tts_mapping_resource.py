from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from loguru import logger


def test_built_wheel_contains_openai_mapping_resource(tmp_path):
    from tldw_chatbook import config

    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(repository),
        ],
        check=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        resource_name = "tldw_chatbook/Config_Files/openai_tts_mappings.json"
        assert resource_name in archive.namelist()
        payload = archive.read(resource_name)
    assert json.loads(payload) == config._BUILT_IN_OPENAI_TTS_MAPPINGS


def test_mapping_fallback_is_informational_and_does_not_expose_exception(
    monkeypatch,
):
    from importlib import resources as importlib_resources

    from tldw_chatbook import config

    sentinel = "private-resource-detail"
    monkeypatch.setattr(
        importlib_resources,
        "files",
        lambda _package: (_ for _ in ()).throw(FileNotFoundError(sentinel)),
    )
    messages: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: messages.append(
            (message.record["level"].name, message.record["message"])
        )
    )
    try:
        mappings = config.load_openai_mappings()
    finally:
        logger.remove(sink_id)

    assert mappings["models"]["tts-1"] == "openai_official_tts-1"
    assert mappings["voices"]["alloy"] == "alloy"
    assert any(
        level == "INFO" and "built-in defaults" in message
        for level, message in messages
    )
    assert all(sentinel not in message for _level, message in messages)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"models": [], "voices": {}},
        {"models": {}, "voices": "not-a-map"},
        {"models": {"tts-1": 7}, "voices": {}},
        {"models": {7: "model"}, "voices": {}},
        {"models": {}, "voices": {"alloy": None}},
    ),
)
def test_structurally_invalid_mapping_uses_sanitized_builtin_fallback(
    monkeypatch,
    tmp_path,
    payload,
):
    from importlib import resources as importlib_resources

    from tldw_chatbook import config

    resource = tmp_path / "openai_tts_mappings.json"
    resource.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(importlib_resources, "files", lambda _package: tmp_path)
    monkeypatch.setattr(config.json, "load", lambda _stream: payload)
    messages: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: messages.append(
            (message.record["level"].name, message.record["message"])
        )
    )
    try:
        mappings = config.load_openai_mappings()
    finally:
        logger.remove(sink_id)

    assert mappings == config._BUILT_IN_OPENAI_TTS_MAPPINGS
    assert mappings is not config._BUILT_IN_OPENAI_TTS_MAPPINGS
    assert any(
        level == "INFO" and message.endswith("using built-in defaults")
        for level, message in messages
    )
    assert all("not-a-map" not in message for _level, message in messages)
