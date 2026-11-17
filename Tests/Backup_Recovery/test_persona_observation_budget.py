"""Every persona rollback child keeps its own idle check and backstop
(TASK-41115): a slow setup no longer shrinks or cancels the later
observation, and the final checks still run."""

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

    def child(home, script, label, *, timeout, idle=None):
        calls.append((label, timeout, idle))
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


@pytest.mark.parametrize("platform,elapsed", [
    ("linux", 0), ("linux", 610), ("linux", 3000), ("darwin", 50), ("win32", 50),
])
def test_each_child_keeps_its_idle_check_and_backstop(
    tmp_path, monkeypatch, platform, elapsed
):
    """However long setup took, on every platform (elapsed 610 s left the old
    shared 860 s budget 40 s for the later child; 3000 s refused it)."""
    calls = _observe(monkeypatch, platform, elapsed)
    workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path)
    idle = workflow._HANG_SECONDS
    assert calls == [
        ("seed-persona", 600, idle),
        ("seed-persona", 600, idle),
        ("persona-capture", 600, idle),
        ("persona-replacement", 600, idle),
        ("persona-reopen", 600, idle),
        ("persona-later", 900, idle),
        ("persona-restored-reopen", 300, idle),
        ("persona-restored-capture", 480, idle),
    ]


def test_the_test_timeout_covers_every_child_backstop():
    """The pytest mark must outlast the children's summed backstops, or it
    would cut a slow but working run short instead of the idle checks."""
    mark = next(
        mark
        for mark in workflow.test_native_nested_persona_pack_survives_reopen_and_later_rollback.pytestmark
        if mark.name == "timeout"
    )
    backstops = 5 * 600 + (
        workflow._LATER_BACKSTOP
        + workflow._RESTORED_REOPEN_BACKSTOP
        + workflow._RESTORED_CAPTURE_BACKSTOP
    )
    assert mark.args[0] > backstops


@pytest.mark.parametrize("missing", [
    "persona-later", "persona-restored-reopen", "persona-restored-capture",
])
def test_every_native_completion_marker_is_still_required(
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
    assert calls[-1] == ("persona-later", 900, workflow._HANG_SECONDS)
