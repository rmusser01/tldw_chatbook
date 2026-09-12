"""Pure settings readers for the speculative Console voice pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from importlib import import_module, metadata
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import platform
import re
import sys
from typing import Any

from tldw_chatbook import __version__
from tldw_chatbook.Audio.voice_process_types import (
    RESPONSE_EAGERNESS_DEFAULT_MS,
    RESPONSE_EAGERNESS_MAX_MS,
    RESPONSE_EAGERNESS_MIN_MS,
)


RESPONSE_EAGERNESS_PRESETS = {
    "fast": 700,
    "balanced": 1200,
    "deliberate": 2000,
}

_INVALID_EAGERNESS_WARNING = (
    "Invalid speculative voice response eagerness; using 700 ms."
)
_MISSING = object()
_LOG = logging.getLogger(__name__)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _PACKAGE_ROOT / "Audio/voice_qualification_manifest.json"
_BUILD_IDENTITY_PATH = _PACKAGE_ROOT / "Audio/voice_build_identity.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PLATFORM_KEYS = {
    "macos-arm64",
    "macos-x86_64",
    "windows-x86_64",
    "linux-x86_64",
    "linux-aarch64",
}
_QUALIFICATION_WARNING = (
    "Speculative voice qualification unavailable; using legacy mode."
)


class _VoiceAuthorityError(RuntimeError):
    pass


def response_eagerness_ms(
    config: Mapping[str, Any],
    *,
    warn: Callable[[str], None] = _LOG.warning,
) -> int:
    """Return the bounded silence-to-dispatch delay for speculative voice."""

    section = config.get("dictation", _MISSING)
    if section is _MISSING:
        return RESPONSE_EAGERNESS_DEFAULT_MS
    if not isinstance(section, Mapping):
        warn(_INVALID_EAGERNESS_WARNING)
        return RESPONSE_EAGERNESS_DEFAULT_MS

    raw = section.get("response_eagerness_ms", _MISSING)
    if raw is _MISSING:
        return RESPONSE_EAGERNESS_DEFAULT_MS
    if (
        type(raw) is not int
        or raw < RESPONSE_EAGERNESS_MIN_MS
        or raw > RESPONSE_EAGERNESS_MAX_MS
    ):
        warn(_INVALID_EAGERNESS_WARNING)
        return RESPONSE_EAGERNESS_DEFAULT_MS
    return raw


def pipeline_aec_enabled(config: Mapping[str, Any]) -> bool:
    """Return the speculative pipeline's strict TOML-boolean AEC setting."""

    section = config.get("dictation", _MISSING)
    if section is _MISSING:
        return True
    if not isinstance(section, Mapping):
        return False

    raw = section.get("pipeline_aec_enabled", _MISSING)
    if raw is _MISSING:
        return True
    if type(raw) is not bool:
        return False
    return raw


def speculative_voice_qualified() -> bool:
    """Return whether this installed build qualifies the speculative pipeline."""

    return _speculative_voice_qualified(
        manifest_path=_MANIFEST_PATH,
        build_identity_path=_BUILD_IDENTITY_PATH,
        platform_key=_current_voice_platform_key(),
        python_tag=_current_python_tag(),
        module_loader=import_module,
        distribution_reader=metadata.distribution,
        extension_resolver=_installed_voice_extension,
        warn=_LOG.warning,
    )


def _current_voice_platform_key() -> str:
    system = platform.system().casefold()
    machine = platform.machine().casefold()
    architecture = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "arm64" if system == "darwin" else "aarch64",
    }.get(machine, machine)
    prefix = {"darwin": "macos", "windows": "windows", "linux": "linux"}.get(system)
    return f"{prefix}-{architecture}" if prefix is not None else "unknown"


def _current_python_tag() -> str:
    if sys.implementation.name != "cpython" or sys.version_info[:2] not in {
        (3, 11),
        (3, 12),
        (3, 13),
    }:
        return "unknown"
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _installed_voice_extension() -> Path:
    spec = find_spec("tldw_voice_aec._native")
    if spec is None or spec.origin is None:
        raise _VoiceAuthorityError("native extension is unavailable")
    return Path(spec.origin)


def _read_authority(path: Path) -> tuple[dict[str, object], bytes]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise _VoiceAuthorityError("packaged authority is unavailable")
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _VoiceAuthorityError("packaged authority is malformed") from exc
    if not isinstance(value, dict):
        raise _VoiceAuthorityError("packaged authority is malformed")
    return value, raw


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_manifest_shape(manifest: Mapping[str, object]) -> None:
    if set(manifest) != {
        "schema_version",
        "pipeline_version",
        "app_version",
        "source_tree_digest",
        "aec_package",
        "platforms",
    }:
        raise _VoiceAuthorityError("manifest shape is invalid")
    if (
        manifest["schema_version"] != 1
        or manifest["pipeline_version"] != 1
        or manifest["app_version"] != __version__
        or not _is_hash(manifest["source_tree_digest"])
    ):
        raise _VoiceAuthorityError("manifest identity is invalid")
    companion = manifest["aec_package"]
    if (
        not isinstance(companion, Mapping)
        or set(companion) != {"name", "version", "upstream_commit"}
        or companion["name"] != "tldw-voice-aec"
        or companion["version"] != __version__
        or not isinstance(companion["upstream_commit"], str)
        or _COMMIT.fullmatch(companion["upstream_commit"]) is None
    ):
        raise _VoiceAuthorityError("manifest companion identity is invalid")
    platforms = manifest["platforms"]
    if not isinstance(platforms, Mapping) or set(platforms) != _PLATFORM_KEYS:
        raise _VoiceAuthorityError("manifest platform matrix is invalid")
    for entry in platforms.values():
        required = {
            "qualified",
            "wheel_sha256",
            "extension_sha256",
            "automated_report_sha256",
            "physical_report_sha256",
            "history_gate_sha256",
            "history_gate_passed",
        }
        if (
            not isinstance(entry, Mapping)
            or not required <= set(entry)
            or set(entry) - (required | {"python_tag"})
        ):
            raise _VoiceAuthorityError("manifest platform entry is invalid")
        if (
            type(entry["qualified"]) is not bool
            or type(entry["history_gate_passed"]) is not bool
            or any(
                not _is_hash(entry[key])
                for key in (
                    "wheel_sha256",
                    "extension_sha256",
                    "automated_report_sha256",
                    "physical_report_sha256",
                    "history_gate_sha256",
                )
            )
            or (entry["qualified"] is True and entry["history_gate_passed"] is not True)
            or (
                "python_tag" in entry
                and (
                    not isinstance(entry["python_tag"], str)
                    or re.fullmatch(r"cp3(?:11|12|13)", entry["python_tag"]) is None
                )
            )
            or (entry["qualified"] is True and "python_tag" not in entry)
        ):
            raise _VoiceAuthorityError("manifest platform evidence is invalid")


def _validate_build_identity(
    identity: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_raw: bytes,
) -> None:
    if set(identity) != {
        "schema_version",
        "pipeline_version",
        "app_version",
        "source_tree_digest",
        "manifest_sha256",
    }:
        raise _VoiceAuthorityError("build identity shape is invalid")
    if (
        identity["schema_version"] != 1
        or identity["pipeline_version"] != 1
        or identity["app_version"] != __version__
        or identity["source_tree_digest"] != manifest["source_tree_digest"]
        or identity["manifest_sha256"] != hashlib.sha256(manifest_raw).hexdigest()
    ):
        raise _VoiceAuthorityError("build identity does not match manifest")


def _speculative_voice_qualified(
    *,
    manifest_path: Path,
    build_identity_path: Path,
    platform_key: str,
    python_tag: str,
    module_loader: Callable[[str], Any],
    distribution_reader: Callable[[str], Any],
    extension_resolver: Callable[[], Path],
    warn: Callable[[str], None],
) -> bool:
    try:
        manifest, manifest_raw = _read_authority(manifest_path)
        identity, _identity_raw = _read_authority(build_identity_path)
        _validate_manifest_shape(manifest)
        _validate_build_identity(
            identity,
            manifest=manifest,
            manifest_raw=manifest_raw,
        )
        platforms = manifest["platforms"]
        companion = manifest["aec_package"]
        assert isinstance(platforms, Mapping) and isinstance(companion, Mapping)
        entry = platforms.get(platform_key)
        if (
            not isinstance(entry, Mapping)
            or entry["qualified"] is not True
            or entry["history_gate_passed"] is not True
            or entry.get("python_tag") != python_tag
        ):
            raise _VoiceAuthorityError("current platform is not qualified")
        module = module_loader("tldw_voice_aec")
        if getattr(module, "AecProcessor", None) is None:
            raise _VoiceAuthorityError("companion import is invalid")
        distribution = distribution_reader("tldw-voice-aec")
        if distribution.version != companion["version"]:
            raise _VoiceAuthorityError("companion version does not match")
        provenance_path = Path(
            distribution.locate_file("tldw_voice_aec/provenance/UPSTREAM.json")
        )
        if not provenance_path.is_file() or provenance_path.is_symlink():
            raise _VoiceAuthorityError("companion provenance is unavailable")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("commit") != companion["upstream_commit"]
        ):
            raise _VoiceAuthorityError("companion provenance does not match")
        extension = Path(extension_resolver())
        if (
            not extension.is_file()
            or extension.is_symlink()
            or extension.suffix.casefold() not in {".so", ".pyd", ".dylib"}
            or hashlib.sha256(extension.read_bytes()).hexdigest()
            != entry["extension_sha256"]
        ):
            raise _VoiceAuthorityError("native extension does not match")
        return True
    except Exception:
        warn(_QUALIFICATION_WARNING)
        return False
