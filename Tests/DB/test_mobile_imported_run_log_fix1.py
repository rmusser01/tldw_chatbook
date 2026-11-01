"""Actual ordinary reader iterator custody through process retirement."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.DB.test_mobile_imported_run_log import _history


@pytest.mark.parametrize("mode", ["failed_close", "healthy_close", "exhausted"])
def test_runtime_iterator_custody(tmp_path, monkeypatch, mode):
    marker = os.environ.get("TLDW_RUNTIME_ITERATOR_PROOF")
    if marker is None:
        proof = tmp_path / "proof.json"
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--show-capture=no",
                "--no-cov",
                "--basetemp",
                str(tmp_path / "child"),
                __file__ + "::test_runtime_iterator_custody[" + mode + "]",
            ],
            env={**os.environ, "TLDW_RUNTIME_ITERATOR_PROOF": str(proof)},
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        evidence = json.loads(proof.read_text())
        assert evidence["mode"] == mode and evidence["guards_empty"]
        from tldw_chatbook.Backup_Recovery.control_records import admission_authority

        with admission_authority(Path(evidence["control"])).maintenance(
            ("profile",), 2
        ):
            pass
        return
    from Tests import network_guard, real_profile_guard
    from tldw_chatbook.DB import mobile_imported_run_log as runtime

    reached = {}
    census, scandir = runtime._PinnedHistory._census, runtime.os.scandir

    class Iterator:
        def __init__(self, native):
            self.native, self.close_calls = native, 0

        def __iter__(self):
            return self

        def __next__(self):
            if mode != "exhausted":
                raise OSError("interrupted before native exhaustion")
            return next(self.native)

        def close(self):
            self.close_calls += 1
            if mode == "failed_close":
                raise OSError("close refused before underlying closure")
            self.native.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def inspect_census(pins, relative):
        reached["pins"] = pins

        def opened(fd):
            iterator = Iterator(scandir(fd))
            reached["iterator"] = iterator
            return iterator

        with monkeypatch.context() as fault:
            fault.setattr(runtime.os, "scandir", opened)
            return census(pins, relative)

    with _history(tmp_path, monkeypatch) as values:
        db, _root, child, _historical, operation, source, _namespace, _log_dir = values
        with monkeypatch.context() as fault:
            fault.setattr(runtime._PinnedHistory, "_census", inspect_census)
            if mode == "exhausted":
                page = db.read_imported_run_log_page(child, operation, source)
                assert [part.record.run_id for part in page.slices] == ["old-child"]
            else:
                with pytest.raises((OSError, ValueError)):
                    db.read_imported_run_log_page(child, operation, source)
        pins, iterator = reached["pins"], reached["iterator"]
        assert iterator.close_calls == 1
        if mode == "failed_close":
            assert pins in runtime._ACTIVE and pins in runtime._QUARANTINED
            assert pins.close_failed and pins.directory_iterators == [iterator]
            assert pins.fds and all(os.fstat(fd) for fd in pins.fds)
            assert next(iterator.native).name
            for path, lease in pins.leases:
                assert lease.execution_context(path) == pins.context
            with pytest.raises(ValueError, match="retirement_uncertain"):
                pins.retire()
            assert iterator.close_calls == 1
        else:
            assert pins not in runtime._ACTIVE and pins not in runtime._QUARANTINED
            assert not pins.fds and not pins.directory_iterators
            assert list(iterator.native) == []
    control = tmp_path / "bootstrap"
    if mode == "failed_close":
        probe = """from Tests.real_profile_guard import install
install()
from Tests.network_guard import install
install()
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
import sys
try:
 with admission_authority(Path(sys.argv[1])).maintenance(('profile',),0.1): pass
except AdmissionError:
 sys.exit(23)
raise AssertionError('iterator custody released file admission')
"""
        contender = subprocess.run(
            [sys.executable, "-c", probe, str(control)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert contender.returncode == 23, contender.stdout + contender.stderr
    assert not real_profile_guard.take_violations()
    assert not network_guard.blocked_attempts()
    Path(marker).write_text(
        json.dumps({"mode": mode, "control": str(control), "guards_empty": True})
    )
    os._exit(0)
