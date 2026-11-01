"""New raw-proof descriptors retain native exclusion after a reached close fault."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


def _raw_fault(tmp_path, monkeypatch, phase):
    from Tests.Backup_Recovery.test_mobile_captured_authority import (
        test_captured_shared_raw_owners_prove_exact_references_without_writes,
    )
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        mobile_captured_authority,
        storage_admission,
    )

    reached = {}
    native_close = archive_reader.os.close
    native_bytes = mobile_captured_authority._checked_bytes
    native_payload = mobile_captured_authority._payload
    native_selected = mobile_captured_authority._selected_source
    native_fdopen = archive_reader.os.fdopen

    def arm(scope, path):
        info = path.stat()
        reached.update(scope=scope, selected=(info.st_dev, info.st_ino))

    def selected_source(scope, purpose, owner):
        if phase == "selector":
            selector = (
                scope.captured.preview.source_selector
                if purpose == "legacy"
                else scope.captured.preview.target_selector
            )
            arm(scope, selector)
        return native_selected(scope, purpose, owner)

    def fdopen(fd, *args, **kwargs):
        stream = native_fdopen(fd, *args, **kwargs)
        if (
            phase == "selector"
            and "scope" in reached
            and "fd" not in reached
            and kwargs.get("closefd", True)
        ):
            info = archive_reader.os.fstat(fd)
            if (info.st_dev, info.st_ino) == reached["selected"]:

                @contextmanager
                def unowned_selector():
                    try:
                        yield stream
                    finally:
                        # Keep the real stream alive to model its failed physical close.
                        reached.update(fd=fd, unowned_stream=stream)
                        raise OSError("injected reached selector close")

                return unowned_selector()
        return stream

    def checked_bytes(scope, path):
        if phase in {"raw", "parent"} and path.name == "tldw_chatbook_personas.json":
            arm(scope, path.parent if phase == "parent" else path)
        return native_bytes(scope, path)

    def payload(scope, sealed, member):
        if phase == "member":
            arm(scope, sealed.path)
        return native_payload(scope, sealed, member)

    def close(fd):
        scope = reached.get("scope")
        if scope is not None and fd in scope.custody.fds and "fd" not in reached:
            info = archive_reader.os.fstat(fd)
            if (info.st_dev, info.st_ino) == reached["selected"]:
                reached["fd"] = fd
                raise OSError("injected reached raw descriptor close")
        return native_close(fd)

    monkeypatch.setattr(mobile_captured_authority, "_selected_source", selected_source)
    monkeypatch.setattr(archive_reader.os, "fdopen", fdopen)
    monkeypatch.setattr(mobile_captured_authority, "_checked_bytes", checked_bytes)
    monkeypatch.setattr(mobile_captured_authority, "_payload", payload)
    monkeypatch.setattr(archive_reader.os, "close", close)
    with pytest.raises(RuntimeError, match="capture_resources_not_retired"):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    scope = reached["scope"]
    assert scope.custody.close_failed
    assert reached["fd"] in scope.custody.fds
    actual = archive_reader.os.fstat(reached["fd"])
    assert (actual.st_dev, actual.st_ino) == reached["selected"]
    assert storage_admission._failed_capture_holds
    assert scope.resources and all(path.exists() for path, _ in scope.roots)
    root = scope.session._control.parent
    probe = """from pathlib import Path
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
import sys
try:
 with admission_authority(Path(sys.argv[1])).maintenance(('bootstrap.unbound','mobile.profile'),0.1): pass
except AdmissionError:
 sys.exit(23)
raise AssertionError('raw descriptor fault released native maintenance')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(root)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    Path(os.environ["TLDW_I_RAW_FAULT_PROOF"]).write_text(
        json.dumps(
            {
                "phase": phase,
                "control_root": str(root),
                "fd": reached["fd"],
                "device": actual.st_dev,
                "inode": actual.st_ino,
            }
        )
    )


@pytest.mark.parametrize("phase", ("raw", "parent", "member", "selector"))
def test_raw_resources_keep_exclusion_until_actual_process_retirement(tmp_path, phase):
    child = tmp_path / "child"
    child.mkdir(mode=0o700)
    proof_path = child / "proof.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "--no-cov",
            "--basetemp",
            str(child / "pytest"),
            __file__ + "::test_raw_fault_child",
        ],
        env=dict(
            os.environ,
            TLDW_I_RAW_FAULT_PHASE=phase,
            TLDW_I_RAW_FAULT_PROOF=str(proof_path),
        ),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(proof_path.read_text())
    assert proof["phase"] == phase
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    with admission_authority(Path(proof["control_root"])).maintenance(
        ("bootstrap.unbound", "mobile.profile"), 2
    ):
        pass


def test_raw_fault_child(tmp_path, monkeypatch):
    phase = os.environ.get("TLDW_I_RAW_FAULT_PHASE")
    if phase is None:
        pytest.skip("private native quarantine child only")
    _raw_fault(tmp_path, monkeypatch, phase)
    # Physical process retirement follows the retained-FD and exclusion proof.
    os._exit(0)
