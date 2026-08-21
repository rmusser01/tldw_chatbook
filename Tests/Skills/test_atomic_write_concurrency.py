"""Tests/Skills/test_atomic_write_concurrency.py

TASK-17963: the skills-subsystem atomic write-then-`Path.replace` helpers
used a FIXED temp filename, so two concurrent writers to the same target
(two app instances, or two threads/async callers in one) could race on the
same temp path -- one writer's `replace()` consuming the other's
still-being-written temp file, raising `FileNotFoundError`.

This file exercises the shared fix (`Skills_Interop/atomic_write.py`) at
three levels:
  * the shared helper module itself (uniqueness, cleanup-on-failure),
  * a concurrency race reproduction against the two converted call sites
    (`LocalSkillsService`'s index/text/bytes writers,
    `skill_trust_store`'s `_atomic_write_json`/`_atomic_write_bytes`),
  * format-preservation checks (index sort_keys/trailing newline, trust
    store's dot-prefixed temp convention) so the fix didn't change on-disk
    shape, only temp-name uniqueness.

Before `local_skills_service.py`/`skill_trust_store.py` were converted to
call the shared helper, the race tests below reliably failed with
`FileNotFoundError` against the old fixed `<name>.tmp` / `.{name}.tmp`
temp names (captured as the task's RED evidence) -- see TASK-17963.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop import atomic_write as aw
from tldw_chatbook.Skills_Interop import skill_trust_store as trust_store_module
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

N_THREADS = 6
N_ITERATIONS = 30

#: Tests/conftest.py's autouse XDG-isolation fixture creates `tmp_path /
#: "test_data"` for every test in the suite (even ones that never touch
#: RAG/config) -- it is not a stray temp file left by anything under test
#: here, so the "no leftover temp file" assertions below must ignore it.
_KNOWN_FIXTURE_ENTRIES = {"test_data"}


def _stray_entries(tmp_path: Path, *exclude: Path) -> list[Path]:
    """Return unexpected entries directly under ``tmp_path``.

    Ignores the autouse XDG-isolation ``test_data`` directory and any
    explicitly-expected paths (e.g. the write target itself).
    """
    excluded = set(exclude)
    return [
        p
        for p in tmp_path.iterdir()
        if p.name not in _KNOWN_FIXTURE_ENTRIES and p not in excluded
    ]


def _run_concurrent(fn, *, n_threads: int = N_THREADS, n_iterations: int = N_ITERATIONS):
    """Run ``fn(worker_id, i)`` from ``n_threads`` threads, ``n_iterations`` times
    each, and return every exception any call raised (empty list == clean run).
    """
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        for i in range(n_iterations):
            try:
                fn(worker_id, i)
            except BaseException as exc:  # noqa: BLE001 -- captured for assertion
                with errors_lock:
                    errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


# ---------------------------------------------------------------------------
# Uniqueness: unique_temp_path derives from the target name and from the
# writer's (pid, thread id).
# ---------------------------------------------------------------------------


class TestUniqueTempPath:
    def test_derives_from_target_name_and_writer_identity(self, monkeypatch):
        monkeypatch.setattr(aw.os, "getpid", lambda: 4242)
        monkeypatch.setattr(aw.threading, "get_ident", lambda: 9999)

        path = Path("/some/dir/my_target.json")
        temp = aw.unique_temp_path(path)

        assert temp.parent == path.parent
        assert temp.name == "my_target.json.4242.9999.tmp"

    def test_hidden_dot_prefixes_while_keeping_the_rest(self, monkeypatch):
        monkeypatch.setattr(aw.os, "getpid", lambda: 4242)
        monkeypatch.setattr(aw.threading, "get_ident", lambda: 9999)

        path = Path("/some/dir/skill_trust_manifest.json")
        temp = aw.unique_temp_path(path, hidden=True)

        assert temp.name == ".skill_trust_manifest.json.4242.9999.tmp"

    def test_two_different_pid_tid_contexts_never_collide(self, monkeypatch):
        path = Path("/some/dir/store.json")

        monkeypatch.setattr(aw.os, "getpid", lambda: 111)
        monkeypatch.setattr(aw.threading, "get_ident", lambda: 222)
        first = aw.unique_temp_path(path)

        monkeypatch.setattr(aw.os, "getpid", lambda: 111)
        monkeypatch.setattr(aw.threading, "get_ident", lambda: 333)
        second = aw.unique_temp_path(path)

        monkeypatch.setattr(aw.os, "getpid", lambda: 444)
        monkeypatch.setattr(aw.threading, "get_ident", lambda: 222)
        third = aw.unique_temp_path(path)

        assert len({first, second, third}) == 3


# ---------------------------------------------------------------------------
# Cleanup on failure: no stray temp file survives a write or replace failure.
# ---------------------------------------------------------------------------


class TestCleanupOnFailure:
    def test_replace_atomically_unlinks_temp_when_write_fn_raises_after_writing(
        self, tmp_path
    ):
        target = tmp_path / "target.txt"
        temp = aw.unique_temp_path(target)

        def write_then_fail(path: Path) -> None:
            path.write_text("partial content", encoding="utf-8")
            raise RuntimeError("simulated failure after the temp file was written")

        with pytest.raises(RuntimeError, match="simulated failure"):
            aw.replace_atomically(temp, target, write_then_fail)

        assert not temp.exists()
        assert not target.exists()
        assert _stray_entries(tmp_path) == []

    def test_write_text_atomic_cleans_up_on_replace_failure_and_still_raises(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "target.json"

        def boom_replace(self, other):  # noqa: ANN001 -- Path.replace signature
            raise OSError("simulated replace failure")

        monkeypatch.setattr(Path, "replace", boom_replace)

        with pytest.raises(OSError, match="simulated replace failure"):
            aw.write_text_atomic(target, "hello")

        # The genuine failure propagated (asserted above); it must not have
        # left a stray temp file behind either.
        assert _stray_entries(tmp_path) == []

    def test_write_bytes_atomic_cleans_up_on_write_failure_and_still_raises(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "target.bin"

        def boom_write_bytes(self, data):  # noqa: ANN001 -- Path.write_bytes signature
            raise OSError("simulated disk-full failure")

        monkeypatch.setattr(Path, "write_bytes", boom_write_bytes)

        with pytest.raises(OSError, match="simulated disk-full failure"):
            aw.write_bytes_atomic(target, b"payload")

        assert _stray_entries(tmp_path) == []


# ---------------------------------------------------------------------------
# Race reproduction: N threads hammering the SAME target path through the
# real (converted) call sites must never raise, and the final file must be
# one complete, valid writer's content.
#
# Pre-conversion, these reliably raised FileNotFoundError against the old
# fixed `<name>.tmp` (local_skills_service.py) / `.{name}.tmp`
# (skill_trust_store.py) temp names -- that failure was captured as this
# task's RED evidence before the sites below were converted to
# atomic_write.py's writer-unique naming.
# ---------------------------------------------------------------------------


class TestRaceLocalSkillsServiceWriteTextAtomic:
    def test_concurrent_writers_same_target_never_raise(self, tmp_path):
        target = tmp_path / "shared_text.txt"

        def write(worker_id: int, i: int) -> None:
            LocalSkillsService._write_text_atomic(
                target, json.dumps({"worker": worker_id, "i": i})
            )

        errors = _run_concurrent(write)
        assert errors == []
        data = json.loads(target.read_text(encoding="utf-8"))
        assert set(data.keys()) == {"worker", "i"}


class TestRaceLocalSkillsServiceWriteBytesAtomic:
    def test_concurrent_writers_same_target_never_raise(self, tmp_path):
        target = tmp_path / "shared_bytes.bin"

        def write(worker_id: int, i: int) -> None:
            LocalSkillsService._write_bytes_atomic(
                target, json.dumps({"worker": worker_id, "i": i}).encode("utf-8")
            )

        errors = _run_concurrent(write)
        assert errors == []
        data = json.loads(target.read_bytes())
        assert set(data.keys()) == {"worker", "i"}


class TestRaceLocalSkillsServiceSaveIndex:
    def test_concurrent_save_index_never_raises(self, tmp_path):
        service = LocalSkillsService(store_dir=tmp_path)

        def write(worker_id: int, i: int) -> None:
            service._save_index({f"skill-{worker_id}-{i}": {"name": "x"}})

        errors = _run_concurrent(write)
        assert errors == []
        loaded = service._load_index()
        assert len(loaded) == 1  # one writer's complete record, whichever won


class TestRaceSkillTrustStoreAtomicWriteJson:
    def test_concurrent_writers_same_target_never_raise(self, tmp_path):
        target = tmp_path / "skill_trust_manifest.json"

        def write(worker_id: int, i: int) -> None:
            trust_store_module._atomic_write_json(
                target,
                {"worker": worker_id, "i": i},
                indent=2,
                base_dir=tmp_path,
            )

        errors = _run_concurrent(write)
        assert errors == []
        data = json.loads(target.read_text(encoding="utf-8"))
        assert set(data.keys()) == {"worker", "i"}


class TestRaceSkillTrustStoreAtomicWriteBytes:
    def test_concurrent_writers_same_target_never_raise(self, tmp_path):
        target = tmp_path / "skill_trust_snapshot.bin"

        def write(worker_id: int, i: int) -> None:
            trust_store_module._atomic_write_bytes(
                target,
                json.dumps({"worker": worker_id, "i": i}).encode("utf-8"),
                base_dir=tmp_path,
            )

        errors = _run_concurrent(write)
        assert errors == []
        data = json.loads(target.read_bytes())
        assert set(data.keys()) == {"worker", "i"}


# ---------------------------------------------------------------------------
# Semantics preservation: the fix must not change on-disk shape, only
# temp-name uniqueness.
# ---------------------------------------------------------------------------


class TestSemanticsPreservation:
    def test_save_index_round_trips_with_sorted_keys_and_trailing_newline(
        self, tmp_path
    ):
        service = LocalSkillsService(store_dir=tmp_path)
        records = {
            "zebra_skill": {"name": "zebra_skill"},
            "alpha_skill": {"name": "alpha_skill"},
        }

        service._save_index(records)

        raw = service.index_path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        parsed = json.loads(raw)
        assert parsed["version"] == 1
        assert list(parsed["skills"].keys()) == ["alpha_skill", "zebra_skill"]

        loaded = service._load_index()
        assert loaded == records

        # No stray temp file left beside the index after a clean write.
        assert _stray_entries(tmp_path, service.index_path) == []

    def test_write_text_atomic_writes_exact_content_no_stray_temp(self, tmp_path):
        target = tmp_path / "SKILL.md"
        LocalSkillsService._write_text_atomic(target, "# Title\n\nBody text.\n")

        assert target.read_text(encoding="utf-8") == "# Title\n\nBody text.\n"
        assert _stray_entries(tmp_path, target) == []

    def test_write_bytes_atomic_writes_exact_content_no_stray_temp(self, tmp_path):
        target = tmp_path / "bundle.zip"
        payload = b"\x50\x4b\x03\x04binary-ish-payload"
        LocalSkillsService._write_bytes_atomic(target, payload)

        assert target.read_bytes() == payload
        assert _stray_entries(tmp_path, target) == []

    def test_trust_store_atomic_write_json_preserves_indent_and_sort_keys(
        self, tmp_path
    ):
        target = tmp_path / "skill_trust_manifest.json"
        payload = {"zeta": 1, "alpha": 2, "nested": {"z": 1, "a": 2}}

        trust_store_module._atomic_write_json(
            target, payload, indent=2, base_dir=tmp_path
        )

        raw = target.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        # sort_keys=True + indent=2, matching the pre-existing format exactly.
        expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        assert raw == expected

        # Nothing else (in particular no leftover dot-prefixed temp file)
        # remains in the directory after a clean write.
        assert _stray_entries(tmp_path, target) == []

    def test_trust_store_atomic_write_bytes_no_stray_temp(self, tmp_path):
        target = tmp_path / "skill_trust_manifest.json"
        payload = b"encrypted-bytes-not-really"

        trust_store_module._atomic_write_bytes(target, payload, base_dir=tmp_path)

        assert target.read_bytes() == payload
        assert _stray_entries(tmp_path, target) == []

    def test_trust_store_temp_name_is_dot_prefixed_hidden_convention(
        self, tmp_path, monkeypatch
    ):
        """The trust store's temp file, while in flight, keeps its
        pre-existing dot-prefixed ("hidden") convention -- only the
        writer-unique suffix is new."""
        target = tmp_path / "skill_trust_manifest.json"
        observed_temp_names: list[str] = []

        original_replace_atomically = aw.replace_atomically

        def spy_replace_atomically(temp_path, target_path, write_fn):
            observed_temp_names.append(temp_path.name)
            return original_replace_atomically(temp_path, target_path, write_fn)

        monkeypatch.setattr(
            trust_store_module, "replace_atomically", spy_replace_atomically
        )

        trust_store_module._atomic_write_json(
            target, {"a": 1}, indent=2, base_dir=tmp_path
        )

        assert len(observed_temp_names) == 1
        assert observed_temp_names[0].startswith(".")
        assert observed_temp_names[0].startswith(f".{target.name}.")
        assert observed_temp_names[0].endswith(".tmp")
