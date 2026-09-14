"""App-only publishing must preserve the native voice release boundary."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from Packaging.check_voice_aec_version_sync import check_version_sync

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "tldw_chatbook/Audio/voice_qualification_manifest.json"


@pytest.fixture
def release_root(tmp_path):
    for relative in (
        "pyproject.toml",
        "native/voice_aec/pyproject.toml",
        "native/voice_aec/PYBIND11_LICENSE.txt",
        "tldw_chatbook/__init__.py",
        MANIFEST,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        "\n".join(
            line
            for line in project.read_text().splitlines()
            if "tldw-voice-aec==" not in line
        )
        + "\n"
    )
    return tmp_path


def test_app_only_accepts_absent_companion_and_unqualified_rollout(release_root):
    assert check_version_sync(release_root, app_only=True) == []
    assert any(
        "exact companion pin" in error for error in check_version_sync(release_root)
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "tldw_voice_aec==0.2.1",
        "TLDW-VOICE-AEC==0.2.1",
        "tldw-voice-aec@https://example.invalid/voice.whl",
        " tldw.voice.aec @ https://example.invalid/voice.whl",
    ],
)
@pytest.mark.parametrize("section", ["dependencies", "speech_recording", "realtime"])
def test_app_only_rejects_companion_in_any_dependency_group(
    release_root, section, requirement
):
    project = release_root / "pyproject.toml"
    source = project.read_text().replace(
        f"{section} = [", f'{section} = [\n    "{requirement}",', 1
    )
    project.write_text(source)
    assert any(
        "must omit" in error
        for error in check_version_sync(release_root, app_only=True)
    )


@pytest.mark.parametrize("state", ["qualified", "missing", "malformed", "empty"])
def test_app_only_rejects_missing_or_qualified_authority(release_root, state):
    path = release_root / MANIFEST
    document = json.loads(path.read_text())
    if state == "missing":
        path.unlink()
    elif state == "malformed":
        path.write_text("{")
    else:
        if state == "qualified":
            document["platforms"]["linux-x86_64"]["qualified"] = True
        else:
            document["platforms"] = {}
        path.write_text(json.dumps(document))
    assert any(
        "unqualified" in error
        for error in check_version_sync(release_root, app_only=True)
    )


def test_strict_companion_release_still_requires_exact_pin(release_root):
    project = release_root / "pyproject.toml"
    source = project.read_text()
    version = tomllib.loads(source)["project"]["version"]
    exact_pin = f"tldw-voice-aec=={version}"
    project.write_text(
        source.replace(
            "speech_recording = [",
            f'speech_recording = [\n    "{exact_pin}",',
            1,
        )
    )
    assert check_version_sync(release_root) == []
    project.write_text(
        project.read_text().replace(exact_pin, f"tldw-voice-aec>={version}")
    )
    assert any(
        "exact companion pin" in error for error in check_version_sync(release_root)
    )
