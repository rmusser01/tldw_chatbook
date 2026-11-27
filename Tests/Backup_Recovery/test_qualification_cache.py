"""Parsed evidence reuse must never cache native authorization or stale bytes."""

import json
import os
import time
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery import qualification
from tldw_chatbook.Backup_Recovery import storage_admission as storage


IDENTITY = {
    "os": "Linux",
    "release": "test",
    "arch": "test",
    "python": "3.12.0",
    "filesystem": "ext4",
    "flags": 0,
}


@pytest.fixture(autouse=True)
def _no_observation_window(monkeypatch):
    """These tests are about the check made on a call; the window has its own below.

    On Windows one positive look at the evidence file covers a second
    (TASK-40608, owner decisions O1 and O3), so a change made right after a
    call is seen up to a second later. Every other platform looks every time.
    """
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    monkeypatch.setattr(qualification, "_EVIDENCE_OBSERVED", float("-inf"))


@pytest.fixture
def evidence_file(tmp_path, monkeypatch):
    raw = (
        Path(qualification.__file__).with_name("native_qualification.json").read_bytes()
    )
    path = tmp_path / "native_qualification.json"
    path.write_bytes(raw)
    monkeypatch.setattr(qualification, "__file__", str(tmp_path / "qualification.py"))
    return path


def _invalidate(path, *, replace):
    """Change the document while preserving its byte count and modification time."""
    before = path.stat()
    raw = path.read_bytes()
    record = json.loads(raw)
    record["schema_version"] = 9
    changed = json.dumps(record).encode()
    # Whitespace padding preserves the original fixture's exact length.
    assert len(changed) <= len(raw)
    changed += b" " * (len(raw) - len(changed))
    if replace:
        replacement = path.with_name("replacement.json")
        replacement.write_bytes(changed)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)
    else:
        path.write_bytes(changed)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_reuses_parsing_for_unchanged_evidence(evidence_file, monkeypatch):
    original = qualification._Evidence.model_validate_json
    parsed = []

    def parse(raw):
        parsed.append(raw)
        return original(raw)

    monkeypatch.setattr(qualification._Evidence, "model_validate_json", parse)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    assert qualification._qualified_identity("publish_file", IDENTITY)[0]
    assert len(parsed) == 1


@pytest.mark.parametrize("replace", [False, True], ids=["in-place", "replacement"])
def test_rejects_changed_evidence_with_same_size_and_mtime(evidence_file, replace):
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    _invalidate(evidence_file, replace=replace)
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_evidence_invalid",
    )


def test_deleted_evidence_refuses_and_recovers(evidence_file):
    raw = evidence_file.read_bytes()
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    evidence_file.unlink()
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
    evidence_file.write_bytes(b"{invalid")
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_evidence_invalid",
    )
    evidence_file.write_bytes(raw)
    assert qualification._qualified_identity("admission", IDENTITY)[0]


def test_transient_evidence_open_failure_is_not_cached(evidence_file, monkeypatch):
    original = qualification.os.open
    attempts = []

    def open_once(path, *args, **kwargs):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("transient_evidence_failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "open", open_once)
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
    assert qualification._qualified_identity("admission", IDENTITY)[0]


def test_replacement_between_stat_and_open_refuses(evidence_file, monkeypatch):
    original = qualification.os.open
    replaced = False

    def replace_before_open(path, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            _invalidate(evidence_file, replace=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "open", replace_before_open)
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_evidence_invalid",
    )


@pytest.mark.parametrize("replace", [False, True], ids=["in-place", "replacement"])
def test_mutation_during_parse_refuses_without_caching_old_bytes(
    evidence_file, monkeypatch, replace
):
    original = qualification._Evidence.model_validate_json
    replaced = False

    def replace_after_read(raw):
        nonlocal replaced
        parsed = original(raw)
        if not replaced:
            replaced = True
            _invalidate(evidence_file, replace=replace)
        return parsed

    monkeypatch.setattr(
        qualification._Evidence, "model_validate_json", replace_after_read
    )
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_evidence_invalid",
    )


def test_qualification_observes_native_identity_on_every_call(
    evidence_file, monkeypatch
):
    identities = iter((IDENTITY, dict(IDENTITY, filesystem="unsupported"), IDENTITY))
    monkeypatch.setattr(qualification, "native_identity", lambda fd: next(identities))
    assert qualification.qualified_for("admission", evidence_file.parent)[0]
    assert qualification.qualified_for("admission", evidence_file.parent) == (
        False,
        "operation_not_qualified",
    )
    assert qualification.qualified_for("admission", evidence_file.parent)[0]


# --- the observation window (Windows; TASK-40608) ----------------------------


def _count_looks(monkeypatch):
    """Count the looks at the evidence file: each one takes one or more stamps."""
    looks = []
    original = qualification._evidence_stamp

    def stamp(info):
        looks.append(info)
        return original(info)

    monkeypatch.setattr(qualification, "_evidence_stamp", stamp)
    return looks


def test_inside_the_window_the_evidence_file_is_not_looked_at_again(
    evidence_file, monkeypatch
):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 30.0)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    looks = _count_looks(monkeypatch)

    for _ in range(5):
        assert qualification._qualified_identity("admission", IDENTITY)[0]
        assert qualification._qualified_identity("publish_file", IDENTITY)[0]

    assert not looks


def test_with_the_window_off_every_call_looks(evidence_file, monkeypatch):
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    looks = _count_looks(monkeypatch)

    for _ in range(3):
        assert qualification._qualified_identity("admission", IDENTITY)[0]

    assert len(looks) == 3


@pytest.mark.parametrize("replace", [False, True], ids=["in-place", "replacement"])
def test_a_change_is_refused_once_the_window_ends(evidence_file, monkeypatch, replace):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.2)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    _invalidate(evidence_file, replace=replace)
    # Inside the window the look that confirmed the evidence still stands.
    assert qualification._qualified_identity("admission", IDENTITY)[0]

    time.sleep(0.25)

    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_evidence_invalid",
    )


def test_the_window_starts_when_the_confirming_look_began(evidence_file, monkeypatch):
    """A slow look must not extend the time its result is believed."""
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.2)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    time.sleep(0.25)
    original = qualification.os.stat

    def slow_stat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        time.sleep(0.25)  # longer than the window
        return result

    monkeypatch.setattr(qualification.os, "stat", slow_stat)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    monkeypatch.setattr(qualification.os, "stat", original)
    looks = _count_looks(monkeypatch)

    assert qualification._qualified_identity("admission", IDENTITY)[0]

    assert looks, "a look that began before the window is still believed"


def test_a_fresh_observation_looks_whatever_the_window(evidence_file, monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 30.0)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    _invalidate(evidence_file, replace=False)

    with storage.fresh_observation():
        refused = qualification._qualified_identity("admission", IDENTITY)

    assert refused == (False, "qualification_evidence_invalid")


def test_a_failed_look_is_not_remembered_as_a_confirmation(evidence_file, monkeypatch):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.2)
    assert qualification._qualified_identity("admission", IDENTITY)[0]
    time.sleep(0.25)
    evidence_file.unlink()

    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
    assert qualification._qualified_identity("admission", IDENTITY) == (
        False,
        "qualification_unavailable",
    )
