"""Content-safe automated speculative-voice qualification reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Packaging import qualify_speculative_voice
from Packaging.qualify_speculative_voice import (
    VoiceQualificationError,
    assemble_automated_report,
    installed_companion_identity,
    write_qualification_report,
)


_DIGEST = "d" * 64


def _passing_sections() -> dict[str, object]:
    return {
        "interpreter": {
            "implementation": "cpython",
            "version": "3.12.11",
            "system": "Darwin",
            "machine": "arm64",
        },
        "companion": {
            "version": "0.1.8.0",
            "upstream_commit": "1" * 40,
            "wheel_sha256": "2" * 64,
            "extension_sha256": "3" * 64,
            "passed": True,
        },
        "corpus": {"thresholds": {"median_erle_db_min": 20.0}, "passed": True},
        "latency_distributions": {
            "trial_count": 40,
            "eos_to_dispatch": {"p95_ms": 700.0},
            "passed": True,
        },
        "completed_pair_history_gate": {
            "inventory": {"count": 18, "sha256": "4" * 64},
            "passed": True,
        },
        "lifecycle": {"suite_sha256": "5" * 64, "passed": True},
        "durable_owners": {
            "inventory_sha256": "6" * 64,
            "report_sha256": "7" * 64,
            "owner_count": 14,
            "passed": True,
        },
    }


def test_automated_report_contains_complete_content_free_evidence() -> None:
    report = assemble_automated_report(
        source_tree_digest=_DIGEST,
        git_revision="a" * 40,
        **_passing_sections(),
    )

    assert report["schema_version"] == 1
    assert report["scenario"] == "automated"
    assert report["source_tree_digest"] == _DIGEST
    assert report["git_revision"] == "a" * 40
    assert report["passed"] is True
    assert set(report) == {
        "schema_version",
        "scenario",
        "source_tree_digest",
        "git_revision",
        "interpreter",
        "companion",
        "corpus",
        "latency_distributions",
        "completed_pair_history_gate",
        "lifecycle",
        "durable_owners",
        "passed",
    }
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        "transcript",
        "response_text",
        "pcm",
        "device_name",
        "credential",
        "request_body",
        "capture_body",
    ):
        assert forbidden not in serialized


def test_automated_report_rejects_invalid_digest_or_content_bearing_fields() -> None:
    sections = _passing_sections()
    with pytest.raises(VoiceQualificationError, match="source-tree digest"):
        assemble_automated_report(source_tree_digest="not-a-digest", **sections)

    sections["lifecycle"] = {
        "passed": True,
        "transcript": "VOICE-POISON",
    }
    with pytest.raises(VoiceQualificationError, match="content-bearing"):
        assemble_automated_report(source_tree_digest=_DIGEST, **sections)


def test_automated_report_is_failed_when_any_gate_fails() -> None:
    sections = _passing_sections()
    sections["lifecycle"] = {"suite_sha256": "5" * 64, "passed": False}

    report = assemble_automated_report(source_tree_digest=_DIGEST, **sections)

    assert report["passed"] is False


def test_installed_companion_identity_uses_installed_provenance_and_wheel_hash(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "tldw_voice_aec-0.1.8.0.whl"
    wheel.write_bytes(b"qualified-wheel")
    provenance = tmp_path / "UPSTREAM.json"
    provenance.write_text(json.dumps({"commit": "1" * 40}), encoding="utf-8")
    distribution = SimpleNamespace(
        version="0.1.8.0",
        locate_file=lambda _path: provenance,
        read_text=lambda _name: None,
    )

    identity = installed_companion_identity(
        aec_report={
            "companion_version": "0.1.8.0",
            "extension_sha256": "3" * 64,
        },
        wheel_path=wheel,
        distribution_reader=lambda _name: distribution,
    )

    assert identity == {
        "version": "0.1.8.0",
        "upstream_commit": "1" * 40,
        "wheel_sha256": hashlib.sha256(b"qualified-wheel").hexdigest(),
        "extension_sha256": "3" * 64,
        "passed": True,
    }


def test_installed_companion_identity_can_use_pep610_wheel_digest(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "UPSTREAM.json"
    provenance.write_text(json.dumps({"commit": "1" * 40}), encoding="utf-8")
    distribution = SimpleNamespace(
        version="0.1.8.0",
        locate_file=lambda _path: provenance,
        read_text=lambda name: (
            json.dumps({"archive_info": {"hash": f"sha256={'2' * 64}"}})
            if name == "direct_url.json"
            else None
        ),
    )

    identity = installed_companion_identity(
        aec_report={
            "companion_version": "0.1.8.0",
            "extension_sha256": "3" * 64,
        },
        distribution_reader=lambda _name: distribution,
    )

    assert identity["wheel_sha256"] == "2" * 64


def test_report_writer_emits_sorted_canonical_json(tmp_path: Path) -> None:
    report = assemble_automated_report(
        source_tree_digest=_DIGEST,
        **_passing_sections(),
    )
    output = tmp_path / "reports" / "automated.json"

    write_qualification_report(output, report)

    expected = (
        json.dumps(
            report,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert output.read_text(encoding="utf-8") == expected


def test_cli_routes_soak_and_computes_source_digest_internally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run_soak(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "schema_version": 1,
            "scenario": "duplex-soak",
            "source_tree_digest": _DIGEST,
            "soak": {"passed": True},
            "passed": True,
        }

    monkeypatch.setattr(qualify_speculative_voice, "run_soak_qualification", run_soak)
    output = tmp_path / "duplex-soak.json"

    result = qualify_speculative_voice.main(
        [
            "--scenario",
            "duplex-soak",
            "--minutes",
            "30",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "scenario": "duplex-soak",
            "minutes": 30,
            "expected_source_tree_digest": None,
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
