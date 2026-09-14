"""A failed later child retains bounded metadata before the suite can time out."""

import ast
import asyncio
import json
# Synthetic subprocess results/errors only; the launch is monkeypatched.
import subprocess  # nosec B404

import pytest


@pytest.mark.parametrize("kind", ["timeout", "launch", "exit"])
def test_parent_records_immediate_later_child_failure(tmp_path, monkeypatch, kind):
    from Tests.Backup_Recovery import test_later_rollback_credential_ui as workflow

    monkeypatch.setattr(workflow, "_earn_replacement", lambda *a, **k: None)
    error = (
        subprocess.TimeoutExpired("private-command", 900, output=b"private-output")
        if kind == "timeout"
        else OSError(5, "private-launch-message")
    )

    def run(*args, **kwargs):
        assert kwargs["timeout"] == (900 if workflow.sys.platform == "win32" else 300)
        if kind == "exit":
            return subprocess.CompletedProcess("private-command", 17)
        raise error

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with pytest.raises(AssertionError if kind == "exit" else type(error)) as caught:
        workflow.test_f9_later_rollback_requires_explicit_credential_review(
            tmp_path, tmp_path / "installed", default_profile=True
        )
    if kind != "exit":
        assert caught.value is error
    data = json.loads((tmp_path / "later-parent-failure.json.log").read_text())
    if kind == "exit":
        assert data == {"event": "child_exit", "returncode": 17}
    else:
        assert data["error"]["error_class"] == type(error).__name__
    assert "private-" not in json.dumps(data)


@pytest.mark.parametrize("write_fails", [False, True])
def test_actual_child_entry_guard_retains_original_exception(
    tmp_path, monkeypatch, write_fails
):
    from Tests.Backup_Recovery import thread_diagnostics
    from Tests.Backup_Recovery.test_later_rollback_credential_ui import _LATER

    error = RuntimeError("private-child-message")

    async def main():
        raise error

    if write_fails:

        def broken(*args):
            raise OSError("private-write-message")

        monkeypatch.setattr(thread_diagnostics, "_write", broken)
    blocks = [node for node in ast.parse(_LATER).body if isinstance(node, ast.Try)]
    assert len(blocks) == 1, "the actual child entry needs an immediate exception guard"
    namespace = {"asyncio": asyncio, "main": main, "home": tmp_path}
    with pytest.raises(RuntimeError) as caught:
        # Fixed test-driver AST; no external code.
        exec(  # noqa: S102  # nosec B102
            compile(
                ast.Module(body=blocks, type_ignores=[]), "later-child-entry", "exec"
            ),
            namespace,
        )
    assert caught.value is error

    path = tmp_path / "later-child-failure.json.log"
    if not write_fails:
        data = json.loads(path.read_text())
        assert data["error"]["error_class"] == "RuntimeError"
        assert "private-" not in json.dumps(data)


def test_failure_record_is_collected_before_pytest_finalization(tmp_path):
    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
    from Tests.Backup_Recovery.run_platform_product import _collect_safe_logs

    private = tmp_path / "private"
    child = private / "product-pytest" / "later" / "home"
    child.mkdir(parents=True)
    record_failure(
        child / "later-child-failure.json.log",
        error=RuntimeError("private-child-message"),
    )
    artifacts = tmp_path / "artifacts"
    assert _collect_safe_logs(private, artifacts) == 1
    recorded = next(artifacts.rglob("later-child-failure.json.log")).read_text()
    assert json.loads(recorded)["error"]["error_class"] == "RuntimeError"
    assert "private-child-message" not in recorded


@pytest.mark.parametrize("metadata_error", [RuntimeError, asyncio.CancelledError])
def test_diagnostic_metadata_failure_does_not_replace_parent_error(
    tmp_path, monkeypatch, metadata_error
):
    from Tests.Backup_Recovery import test_later_rollback_credential_ui as workflow
    from Tests.Backup_Recovery import thread_diagnostics

    monkeypatch.setattr(workflow, "_earn_replacement", lambda *a, **k: None)
    error = subprocess.TimeoutExpired("private-command", 900)

    def fail_run(*args, **kwargs):
        raise error

    def fail_metadata(*args):
        raise metadata_error("private-metadata-message")

    monkeypatch.setattr(workflow.subprocess, "run", fail_run)
    monkeypatch.setattr(thread_diagnostics, "_error_metadata", fail_metadata)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        workflow.test_f9_later_rollback_requires_explicit_credential_review(
            tmp_path, tmp_path / "installed"
        )
    assert caught.value is error
