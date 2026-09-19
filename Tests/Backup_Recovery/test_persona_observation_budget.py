"""The Linux observation borrows setup headroom without losing final checks."""

# Only instantiate TimeoutExpired; no process is started.
import subprocess  # nosec B404
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_created_persona_subtree_rollback as workflow


def _observe(monkeypatch, platform, elapsed, *, missing=None, failure=None):
    clock = SimpleNamespace(now=100.0)
    calls = []
    original_prepare = workflow._prepare_native
    markers = {
        "persona-later": "NATIVE_NESTED_PERSONA_LATER_COMPLETE",
        "persona-restored-reopen": "NATIVE_PERSONA_ORDINARY_RESTORED_REOPEN_COMPLETE",
        "persona-restored-capture": "NATIVE_PERSONA_CAPTURE_COMPLETE",
    }

    def child(home, script, label, *, timeout):
        calls.append((label, timeout))
        if label == "persona-later" and failure is not None:
            raise failure
        return "" if label == missing else markers.get(label, "")

    def prepare(root):
        home = original_prepare(root)
        clock.now += elapsed
        return home

    monkeypatch.setattr(workflow, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        workflow, "time", SimpleNamespace(monotonic=lambda: clock.now), raising=False
    )
    monkeypatch.setattr(workflow, "_child", child)
    monkeypatch.setattr(workflow, "_prepare_native", prepare)
    return calls


@pytest.mark.parametrize("platform,elapsed,later", [
    ("linux", 0, 650), ("linux", 50, 600), ("linux", 500, 150),
    ("linux", 610, 40), ("darwin", 50, 150), ("win32", 50, 900),
])
def test_later_observation_spends_only_remaining_setup_headroom(
    tmp_path, monkeypatch, platform, elapsed, later
):
    calls = _observe(monkeypatch, platform, elapsed)
    workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path)
    assert calls == [
        ("seed-persona", 480 if platform == "win32" else 70),
        ("seed-persona", 480 if platform == "win32" else 70),
        ("persona-capture", 480 if platform == "win32" else 120),
        ("persona-replacement", 600 if platform == "win32" else 150),
        ("persona-reopen", 300 if platform == "win32" else 90),
        ("persona-later", later),
        ("persona-restored-reopen", 300 if platform == "win32" else 90),
        ("persona-restored-capture", 480 if platform == "win32" else 120),
    ]


@pytest.mark.parametrize("elapsed", [650, 651])
def test_exhausted_observation_budget_starts_no_later_child(tmp_path, monkeypatch, elapsed):
    calls = _observe(monkeypatch, "linux", elapsed)
    with pytest.raises(TimeoutError, match="persona_rollback_test_budget_exhausted"):
        workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path)
    assert len(calls) == 5


@pytest.mark.parametrize("missing", [
    "persona-later", "persona-restored-reopen", "persona-restored-capture",
])
def test_redistribution_still_requires_every_native_completion_marker(
    tmp_path, monkeypatch, missing
):
    calls = _observe(monkeypatch, "linux", 50, missing=missing)
    with pytest.raises(AssertionError):
        workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path)
    assert calls[-1][0] == missing


def test_later_child_timeout_is_preserved_without_running_final_checks(tmp_path, monkeypatch):
    original = subprocess.TimeoutExpired("native-fixture", 600)
    calls = _observe(monkeypatch, "linux", 50, failure=original)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path)
    assert caught.value is original
    assert calls[-1] == ("persona-later", 600)
