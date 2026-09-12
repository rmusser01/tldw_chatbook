"""Qualify an installed native AEC wheel with content-free evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from importlib import import_module
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from Packaging.voice_aec_corpus import (
    evaluate_voice_aec_corpus,
    load_voice_aec_corpus,
)


class InstalledVoiceAecError(RuntimeError):
    """Raised when the installed companion cannot qualify full duplex."""


def qualify_installed_voice_aec(
    manifest_path: Path,
    *,
    module_loader: Callable[[str], Any] = import_module,
    version_reader: Callable[[str], str] = metadata.version,
    extension_resolver: Callable[[], Path] | None = None,
    corpus_evaluator: Callable[
        [Mapping[str, Any], Callable[[], Any]], Mapping[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Require a native extension and a passing synthetic corpus report."""

    try:
        module = module_loader("tldw_voice_aec")
        processor_type = module.AecProcessor
        version = version_reader("tldw-voice-aec")
    except Exception as exc:
        raise InstalledVoiceAecError("installed AEC companion is unavailable") from exc
    if not isinstance(version, str) or not version:
        raise InstalledVoiceAecError("installed AEC companion version is invalid")

    resolver = extension_resolver or _installed_extension_path
    extension = resolver()
    if (
        not extension.is_file()
        or extension.is_symlink()
        or extension.suffix.lower() not in {".so", ".pyd", ".dylib"}
    ):
        raise InstalledVoiceAecError("installed AEC native extension is missing")

    manifest = load_voice_aec_corpus(manifest_path)
    processor_factory = lambda: processor_type(  # noqa: E731 - passed as factory
        sample_rate=48_000,
        channels=1,
    )
    if corpus_evaluator is None:
        report = evaluate_voice_aec_corpus(
            manifest,
            processor_factory=processor_factory,
        )
    else:
        report = dict(corpus_evaluator(manifest, processor_factory))

    if report.get("unqualified_case_ids"):
        raise InstalledVoiceAecError("installed AEC corpus has unqualified cases")
    if report.get("effective_mode") != "full-duplex":
        raise InstalledVoiceAecError("installed AEC did not qualify full duplex")
    if report.get("passed") is not True:
        raise InstalledVoiceAecError("installed AEC corpus did not pass")

    return {
        "schema_version": 1,
        "companion_version": version,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        **report,
    }


def _installed_extension_path() -> Path:
    spec = find_spec("tldw_voice_aec._native")
    if spec is None or spec.origin is None:
        raise InstalledVoiceAecError("installed AEC native extension is missing")
    return Path(spec.origin)
