"""TASK-40608 D1: Windows reuses admission evidence inside a bounded window.

ADR-126's 2026-11-21 amendment (owner decisions O1 and O3) turns the PERF-07/08
evidence reuse on for Windows. Every stamp there is a native handle open per
path component, so a positive re-observation of confirmed evidence is kept for
``_EVIDENCE_OBSERVATION_SECONDS`` instead of being repeated on every call:

* with the window at 0, reuse and the full derivation agree after every
  mutation, as on POSIX (the oracle below runs the mutations Windows supports);
* inside the window a warm acquisition touches no file at all;
* a change another process makes is seen once the window ends;
* an in-process admission write and a local pause end the window at once.
"""

from __future__ import annotations

import os
import time

import pytest

from Tests.Backup_Recovery.test_admission_evidence_reuse import (
    MUTATIONS,
    REFUSED_BY_THE_DERIVATION,
    _count_derivations,
    _verdict,
)
from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import bind_profile

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the window exists on Windows only")

#: The catalog entries that need no POSIX mode bits or symbolic links.
WINDOWS_MUTATIONS = (
    "control-no-change",
    "pending-record-from-subprocess",
    "unrelated-pending-from-subprocess",
    "registry-replaced-from-subprocess",
    "registry-intent-file",
    "profile-record-edited-in-place",
    "enrollment-marker-replaced",
    "config-selector-edited",
    "data-dir-renamed-and-recreated",
    "bootstrap-root-removed",
)


@pytest.fixture
def settled(monkeypatch):
    """Evidence recorded milliseconds ago is reusable (the worst case)."""
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", True)


def test_the_window_is_on_for_windows_only():
    assert storage._EVIDENCE_OBSERVATION_SECONDS == 1.0


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
@pytest.mark.parametrize("mutation", WINDOWS_MUTATIONS)
def test_with_the_window_off_reuse_matches_the_full_derivation(
    local_scope, settled, monkeypatch, tmp_path, mutation, bound  # noqa: F811
):
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.0)
    root, config, data, _ = local_scope
    if mutation == "profile-record-edited-in-place" and not bound:
        pytest.skip("an unbound selection has no profile record to edit")
    if bound:
        bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):  # record evidence, confirm it, then serve from it
            assert _verdict(target)[0] == "allowed"
        MUTATIONS[mutation](root, config, data, tmp_path)
        reused = _verdict(target)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        derived = _verdict(target)
    finally:
        startup.close()

    assert reused == derived, f"{mutation}: reuse gave {reused}, derivation {derived}"
    if mutation in REFUSED_BY_THE_DERIVATION:
        assert derived[0] == "refused", f"{mutation} was expected to refuse: {derived}"


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
def test_a_warm_acquisition_neither_derives_nor_observes(
    local_scope, settled, monkeypatch, bound  # noqa: F811
):
    root, config, data, _ = local_scope
    if bound:
        bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        calls = _count_derivations(monkeypatch)
        observed = []
        observe = storage._Evidence.observe

        def counted(self):
            observed.append(self)
            return observe(self)

        monkeypatch.setattr(storage._Evidence, "observe", counted)
        warm = [_verdict(target) for _ in range(5)]
    finally:
        startup.close()

    assert all(verdict[0] == "allowed" for verdict in warm)
    assert calls == {"permission": 0, "scope": 0}, calls
    assert observed == [], "a warm acquisition inside the window re-read stamps"


def test_a_change_from_outside_is_refused_once_the_window_ends(
    local_scope, settled, monkeypatch, tmp_path  # noqa: F811
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        # No in-process hook sees this write: only the stamps can.
        MUTATIONS["registry-intent-file"](root, config, data, tmp_path)
        inside = _verdict(target)
        monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", 0.05)
        time.sleep(0.1)
        after = _verdict(target)
    finally:
        startup.close()

    assert inside[0] == "allowed", "the bounded window is the approved trade (O3)"
    assert after[0] == "refused", after


def test_a_fresh_observation_sees_a_change_from_outside_at_once(
    local_scope, settled, tmp_path  # noqa: F811
):
    """The OpenAI reconnect gates do not use the window.

    ``LLM_Calls.recovery_review._history`` reads inside ``fresh_observation``:
    with the window, a review, its approval and a request were accepted
    against an activation record deleted a moment earlier
    (``Tests/Backup_Recovery/test_activation_openai_reconnect.py``).
    """
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        MUTATIONS["registry-intent-file"](root, config, data, tmp_path)
        assert storage._observation_window() == 1.0
        with storage.fresh_observation():
            assert storage._observation_window() == 0.0
            with storage.fresh_observation():
                fresh = _verdict(target)
            assert storage._observation_window() == 0.0
        assert storage._observation_window() == 1.0
    finally:
        startup.close()

    assert fresh[0] == "refused", fresh


def test_an_in_process_admission_write_ends_the_window_at_once(
    local_scope, settled, monkeypatch  # noqa: F811
):
    from tldw_chatbook.Backup_Recovery import bootstrap

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        bootstrap.advance_admission_epoch()
        calls = _count_derivations(monkeypatch)
        assert _verdict(target)[0] == "allowed"
    finally:
        startup.close()

    assert calls["scope"] >= 1, "an epoch change must run the derivation again"


def test_a_local_pause_forgets_every_observation(local_scope, settled):  # noqa: F811
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        with storage._lock:
            hold = storage._holds[(os.getpid(), str(root))]
            entries = [*hold.evidence.values(), *hold.path_evidence.values()]
            assert entries and any(entry.verified_at for entry in entries)
        pause = storage._begin_local_pause()
        try:
            with storage._lock:
                assert all(entry.verified_at == 0.0 for entry in entries)
                assert storage._writable_volumes == {}
        finally:
            # The startup lease stays in place, as in the running app.
            with storage._changed:
                storage._pause = None
                storage._changed.notify_all()
            del pause
    finally:
        startup.close()
