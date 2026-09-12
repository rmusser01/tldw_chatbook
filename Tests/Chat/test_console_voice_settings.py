"""Bounded settings contracts for speculative Console voice."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook import __version__
from tldw_chatbook.Chat.console_voice_settings import (
    RESPONSE_EAGERNESS_DEFAULT_MS,
    RESPONSE_EAGERNESS_MAX_MS,
    RESPONSE_EAGERNESS_MIN_MS,
    RESPONSE_EAGERNESS_PRESETS,
    _speculative_voice_qualified,
    pipeline_aec_enabled,
    response_eagerness_ms,
    speculative_voice_qualified,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    _PipelineVoiceSettingsDraft,
)


WARNING_TEXT = "Invalid speculative voice response eagerness; using 700 ms."
_UPSTREAM = "109e23c9cec3a44e67c08774874a409741b1e58a"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _runtime_authority(
    tmp_path: Path,
    *,
    qualified: bool = True,
    history_gate_passed: bool = True,
) -> tuple[Path, Path, Path, Path]:
    extension = tmp_path / "_native.fixture.so"
    extension.write_bytes(b"qualified-native-extension")
    extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
    platform_entry = {
        "qualified": qualified,
        "python_tag": "cp312",
        "wheel_sha256": "1" * 64,
        "extension_sha256": extension_sha,
        "automated_report_sha256": "2" * 64,
        "physical_report_sha256": "3" * 64,
        "history_gate_sha256": "4" * 64,
        "history_gate_passed": history_gate_passed,
    }
    manifest = {
        "schema_version": 1,
        "pipeline_version": 1,
        "app_version": __version__,
        "source_tree_digest": "5" * 64,
        "aec_package": {
            "name": "tldw-voice-aec",
            "version": __version__,
            "upstream_commit": _UPSTREAM,
        },
        "platforms": {
            key: dict(platform_entry)
            for key in (
                "macos-arm64",
                "macos-x86_64",
                "windows-x86_64",
                "linux-x86_64",
                "linux-aarch64",
            )
        },
    }
    manifest_path = tmp_path / "voice_qualification_manifest.json"
    manifest_bytes = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    identity = {
        "schema_version": 1,
        "pipeline_version": 1,
        "app_version": __version__,
        "source_tree_digest": "5" * 64,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    identity_path = tmp_path / "voice_build_identity.json"
    identity_path.write_bytes(_canonical_bytes(identity))
    provenance = tmp_path / "UPSTREAM.json"
    provenance.write_text(json.dumps({"commit": _UPSTREAM}), encoding="utf-8")
    return manifest_path, identity_path, extension, provenance


class _Distribution:
    version = __version__

    def __init__(self, provenance: Path) -> None:
        self._provenance = provenance

    def locate_file(self, _relative: str) -> Path:
        return self._provenance


def _runtime_qualified(
    tmp_path: Path,
    *,
    qualified: bool = True,
    history_gate_passed: bool = True,
    platform_key: str = "macos-arm64",
    python_tag: str = "cp312",
    module_loader=lambda _name: SimpleNamespace(AecProcessor=object),
    version: str = __version__,
    commit: str = _UPSTREAM,
    extension_bytes: bytes | None = None,
) -> tuple[bool, list[str]]:
    manifest, identity, extension, provenance = _runtime_authority(
        tmp_path,
        qualified=qualified,
        history_gate_passed=history_gate_passed,
    )
    if extension_bytes is not None:
        extension.write_bytes(extension_bytes)
    provenance.write_text(json.dumps({"commit": commit}), encoding="utf-8")
    distribution = _Distribution(provenance)
    distribution.version = version
    warnings: list[str] = []
    result = _speculative_voice_qualified(
        manifest_path=manifest,
        build_identity_path=identity,
        platform_key=platform_key,
        python_tag=python_tag,
        module_loader=module_loader,
        distribution_reader=lambda _name: distribution,
        extension_resolver=lambda: extension,
        warn=warnings.append,
    )
    return result, warnings


def test_response_eagerness_defaults_and_presets_are_bounded() -> None:
    assert RESPONSE_EAGERNESS_MIN_MS == 500
    assert RESPONSE_EAGERNESS_MAX_MS == 3000
    assert RESPONSE_EAGERNESS_DEFAULT_MS == 700
    assert RESPONSE_EAGERNESS_PRESETS == {
        "fast": 700,
        "balanced": 1200,
        "deliberate": 2000,
    }


@pytest.mark.parametrize("value", [500, 700, 3000])
def test_response_eagerness_accepts_inclusive_integer_range(value: int) -> None:
    assert (
        response_eagerness_ms({"dictation": {"response_eagerness_ms": value}}) == value
    )


@pytest.mark.parametrize("config", [{}, {"dictation": {}}])
def test_missing_response_eagerness_defaults_without_warning(
    config: dict[str, object],
) -> None:
    warnings: list[str] = []

    assert response_eagerness_ms(config, warn=warnings.append) == 700
    assert warnings == []


@pytest.mark.parametrize(
    "value",
    [499, 3001, None, True, 700.0, "700", [], {"value": 700}],
)
def test_invalid_response_eagerness_warns_once_and_falls_back(value: object) -> None:
    warnings: list[str] = []

    assert (
        response_eagerness_ms(
            {"dictation": {"response_eagerness_ms": value}},
            warn=warnings.append,
        )
        == 700
    )
    assert warnings == [WARNING_TEXT]


def test_malformed_dictation_section_warns_once_and_falls_back() -> None:
    warnings: list[str] = []

    assert (
        response_eagerness_ms({"dictation": "not-a-table"}, warn=warnings.append) == 700
    )
    assert warnings == [WARNING_TEXT]


def test_qualification_is_hard_off_even_with_development_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLDW_DEV_SPECULATIVE_VOICE", "1")

    assert speculative_voice_qualified() is False


def test_runtime_qualification_accepts_exact_packaged_and_installed_identity(
    tmp_path: Path,
) -> None:
    result, warnings = _runtime_qualified(tmp_path)

    assert result is True
    assert warnings == []


def test_runtime_authority_fixture_defaults_follow_current_app_version(
    tmp_path: Path,
) -> None:
    manifest_path, identity_path, _extension, _provenance = _runtime_authority(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))

    assert manifest["app_version"] == __version__
    assert manifest["aec_package"]["version"] == __version__
    assert identity["app_version"] == __version__


@pytest.mark.parametrize("stale_field", ["manifest_app", "companion", "build_identity"])
def test_runtime_qualification_rejects_old_matching_version_authority(
    tmp_path: Path,
    stale_field: str,
) -> None:
    manifest_path, identity_path, extension, provenance = _runtime_authority(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    stale_version = "0.1.8.0"
    if stale_field == "manifest_app":
        manifest["app_version"] = stale_version
        identity["app_version"] = stale_version
    elif stale_field == "companion":
        manifest["aec_package"]["version"] = stale_version
    else:
        identity["app_version"] = stale_version
    manifest_raw = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    identity["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    identity_path.write_bytes(_canonical_bytes(identity))
    provenance.write_text(json.dumps({"commit": _UPSTREAM}), encoding="utf-8")
    distribution = _Distribution(provenance)
    distribution.version = stale_version if stale_field == "companion" else __version__
    warnings: list[str] = []

    result = _speculative_voice_qualified(
        manifest_path=manifest_path,
        build_identity_path=identity_path,
        platform_key="macos-arm64",
        python_tag="cp312",
        module_loader=lambda _name: SimpleNamespace(AecProcessor=object),
        distribution_reader=lambda _name: distribution,
        extension_resolver=lambda: extension,
        warn=warnings.append,
    )

    assert result is False
    assert warnings == [
        "Speculative voice qualification unavailable; using legacy mode."
    ]


@pytest.mark.parametrize(
    "change",
    [
        "qualified_false",
        "history_false",
        "unknown_platform",
        "version_mismatch",
        "commit_mismatch",
        "extension_mismatch",
        "python_abi_mismatch",
        "import_failure",
    ],
)
def test_runtime_qualification_fails_closed_for_installed_or_platform_mismatch(
    tmp_path: Path,
    change: str,
) -> None:
    kwargs: dict[str, object] = {}
    if change == "qualified_false":
        kwargs["qualified"] = False
    elif change == "history_false":
        kwargs["history_gate_passed"] = False
    elif change == "unknown_platform":
        kwargs["platform_key"] = "freebsd-x86_64"
    elif change == "version_mismatch":
        kwargs["version"] = "0.1.7.0"
    elif change == "commit_mismatch":
        kwargs["commit"] = "a" * 40
    elif change == "extension_mismatch":
        kwargs["extension_bytes"] = b"tampered-extension"
    elif change == "python_abi_mismatch":
        kwargs["python_tag"] = "cp313"
    else:
        kwargs["module_loader"] = lambda _name: (_ for _ in ()).throw(
            ImportError("content-bearing import detail")
        )

    result, warnings = _runtime_qualified(tmp_path, **kwargs)

    assert result is False
    assert warnings == [
        "Speculative voice qualification unavailable; using legacy mode."
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest, identity: manifest.__setitem__("unknown", True),
        lambda manifest, identity: manifest.__setitem__("schema_version", 2),
        lambda manifest, identity: manifest["platforms"]["macos-arm64"].pop(
            "history_gate_sha256"
        ),
        lambda manifest, identity: identity.__setitem__("source_tree_digest", "6" * 64),
        lambda manifest, identity: identity.__setitem__("manifest_sha256", "7" * 64),
    ],
)
def test_runtime_qualification_fails_closed_for_malformed_or_mixed_authority(
    tmp_path: Path,
    mutation,
) -> None:
    manifest_path, identity_path, extension, provenance = _runtime_authority(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    mutation(manifest, identity)
    manifest_path.write_bytes(_canonical_bytes(manifest))
    identity_path.write_bytes(_canonical_bytes(identity))
    warnings: list[str] = []

    result = _speculative_voice_qualified(
        manifest_path=manifest_path,
        build_identity_path=identity_path,
        platform_key="macos-arm64",
        python_tag="cp312",
        module_loader=lambda _name: SimpleNamespace(AecProcessor=object),
        distribution_reader=lambda _name: _Distribution(provenance),
        extension_resolver=lambda: extension,
        warn=warnings.append,
    )

    assert result is False
    assert warnings == [
        "Speculative voice qualification unavailable; using legacy mode."
    ]


@pytest.mark.parametrize("value", [True, False])
def test_pipeline_aec_accepts_explicit_toml_boolean(value: bool) -> None:
    assert pipeline_aec_enabled({"dictation": {"pipeline_aec_enabled": value}}) is value


def test_pipeline_aec_defaults_enabled_without_reusing_legacy_keys() -> None:
    legacy_only = {
        "dictation": {
            "acoustic_barge_in": False,
            "handsfree_send_delay_seconds": 42,
        }
    }

    assert pipeline_aec_enabled({}) is True
    assert pipeline_aec_enabled(legacy_only) is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_pipeline_aec_fails_closed_for_non_toml_boolean(value: object) -> None:
    assert pipeline_aec_enabled({"dictation": {"pipeline_aec_enabled": value}}) is False


def test_pipeline_aec_fails_closed_for_malformed_dictation_section() -> None:
    assert pipeline_aec_enabled({"dictation": "not-a-table"}) is False


def test_pipeline_voice_draft_is_separate_and_owns_only_pipeline_fields() -> None:
    draft = _PipelineVoiceSettingsDraft(
        response_eagerness_ms="1200",
        pipeline_aec_enabled=False,
    )

    assert [field.name for field in fields(draft)] == [
        "response_eagerness_ms",
        "pipeline_aec_enabled",
    ]
    assert draft.snapshot() == ("1200", False)
