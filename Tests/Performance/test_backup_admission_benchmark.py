"""Private probe contracts: isolation, provenance, complete timing and errors."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


def probe():
    path = (
        Path(__file__).parents[2]
        / "Helper_Scripts/Benchmarks/backup_admission_benchmark.py"
    )
    assert path.is_file(), "paired private admission probe is missing"
    spec = importlib.util.spec_from_file_location("admission_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(module, root):
    (root / "Tests").mkdir(parents=True)
    guard = Path(__file__).parents[1] / "network_guard.py"
    (root / "Tests/network_guard.py").write_bytes(guard.read_bytes())
    package = root / "tldw_chatbook/DB"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    backup = package.parent / "Backup_Recovery"
    backup.mkdir()
    (backup / "__init__.py").write_text("")
    (backup / "participants.py").write_text(
        "class _RepositoryParticipant:\n"
        " def operation(self):\n  raise AssertionError('unused in failing import')\n"
    )
    (package / "ChaChaNotes_DB.py").write_text(
        "import os, sys, keyring\n"
        "from pathlib import Path\n"
        "from keyring.backends.null import Keyring\n"
        "assert isinstance(keyring.get_keyring(), Keyring)\n"
        "assert 'Tests.network_guard' in sys.modules\n"
        "assert Path(os.environ['HOME']).is_relative_to(Path.cwd().parent)\n"
        "assert os.environ['HOME'] == os.environ['USERPROFILE']\n"
        "assert Path(os.environ['TLDW_CONFIG_PATH']).is_file()\n"
        "assert str(Path(__file__).parents[2]) == os.environ['TLDW_PROBE_SOURCE']\n"
        "raise RuntimeError('private-error-content-must-not-escape')\n"
    )
    manifest = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "content_sha256": module.source_digest(root),
    }
    (root / module.MANIFEST).write_text(json.dumps(manifest))
    return root


def test_private_environment_replaces_ambient_selections(tmp_path, monkeypatch):
    module = probe()
    monkeypatch.setenv("HOME", "/ambient")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "real.backend")
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/ambient/config")
    environment = module.private_environment(tmp_path / "profile")
    assert environment["HOME"] == environment["USERPROFILE"]
    assert environment["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert all(
        Path(environment[name]).is_relative_to(tmp_path)
        for name in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TLDW_CONFIG_PATH")
    )
    assert "PYTHONPATH" not in environment


def test_source_selection_refuses_modified_snapshot(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    assert module.select_source(selected)["commit"] == "a" * 40
    (selected / "tldw_chatbook/__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="source_digest_changed"):
        module.select_source(selected)


def test_child_isolates_before_import_and_preserves_failure_without_content(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    assert result["exit_code"] == 1
    assert result["error_type"] == "RuntimeError"
    assert "private-error-content" not in json.dumps(result)
    assert result["network_attempts"] == 0
    assert result["source_sha256"] == module.source_digest(selected)


def test_admission_timing_counts_entry_and_retirement_and_preserves_suppression():
    module = probe()
    ticks = iter((0, 30, 100, 140))
    counts = module.counters()

    class Scope:
        def __enter__(self):
            return 42

        def __exit__(self, *error):
            return error[0] is ValueError

    with module.TimedScope(Scope, counts, lambda: next(ticks)) as value:
        assert value == 42
        raise ValueError("expected")
    assert counts["entry_ns"] == 30
    assert counts["retirement_ns"] == 40
    assert counts["entries"] == counts["retirements"] == 1


def test_admission_retirement_error_is_not_hidden():
    module = probe()
    counts = module.counters()

    class Scope:
        def __enter__(self):
            return None

        def __exit__(self, *error):
            raise OSError("private-error")

    with pytest.raises(OSError), module.TimedScope(Scope, counts):
        pass
    assert counts["retirements"] == 0
    assert counts["retirement_attempts"] == 1


def test_open_audit_counts_retirement_without_replacing_os_open(tmp_path):
    module = probe()
    original = os.open
    counts = module.counters()
    module.install_open_audit(counts)
    counts["counting"] = True
    descriptor = os.open(tmp_path, os.O_RDONLY)
    os.close(descriptor)
    counts["counting"] = False
    assert counts["os_opens"] == 1
    assert os.open is original


def test_network_attempt_fails_receipt_even_if_application_swallows_it(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    target = selected / "tldw_chatbook/DB/ChaChaNotes_DB.py"
    target.write_text(
        "import socket\n"
        "try:\n socket.create_connection(('127.0.0.1', 9))\n"
        "except OSError:\n pass\n"
        "raise ValueError('private-error')\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    assert result["exit_code"] == 1
    assert result["network_attempts"] == 1
    assert result["error_type"] == "ValueError"


def test_child_preserves_explicit_nonzero_exit_code(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    (selected / "tldw_chatbook/DB/ChaChaNotes_DB.py").write_text(
        "raise SystemExit(23)\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    assert result["exit_code"] == 23


def test_abrupt_zero_exit_cannot_reuse_an_old_success_receipt(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    (selected / "tldw_chatbook/DB/ChaChaNotes_DB.py").write_text(
        "import os\nos._exit(0)\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    profile = tmp_path / "profile"
    module.private_environment(profile)
    (profile / "transaction-child.json").write_text(
        json.dumps({"exit_code": 0, "retired": True, "stale": True})
    )
    result = module.run_child(selected, profile, "transaction", 1)
    assert result["exit_code"] != 0
    assert result["error_type"] == "MissingReceipt" and "stale" not in result


@pytest.mark.parametrize("measured_status", [0, -15, 23])
def test_main_reports_any_measured_failure_and_keeps_raw_status(
    tmp_path, monkeypatch, capsys, measured_status
):
    module = probe()
    selected = source(module, tmp_path / "source")
    receipt_file = tmp_path / "receipt.json"
    statuses = iter((0, measured_status))
    monkeypatch.setattr(
        module, "run_child", lambda *args: {"exit_code": next(statuses)}
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "probe",
            "--source",
            str(selected),
            "--phase",
            "transaction",
            "--iterations",
            "1",
            "--receipt",
            str(receipt_file),
        ],
    )
    status = module.main()
    receipt = json.loads(receipt_file.read_text())
    assert status == receipt["exit_code"] == int(measured_status != 0)
    assert receipt["seed"]["exit_code"] == 0
    assert receipt["runs"][0]["exit_code"] == measured_status
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.skipif(os.name == "nt", reason="POSIX child group ownership")
def test_timeout_terminates_the_owned_child_group(tmp_path, monkeypatch):
    import signal

    module = probe()
    selected = source(module, tmp_path / "source")
    (selected / "tldw_chatbook/DB/ChaChaNotes_DB.py").write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "Path(os.environ['HOME']).parent.joinpath('group').write_text(str(os.getpgrp()))\n"
        "time.sleep(30)\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "CHILD_TIMEOUT", 2)
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    group = int((tmp_path / "profile/group").read_text())
    assert group != os.getpgrp()
    with pytest.raises(ProcessLookupError):
        os.killpg(group, signal.SIGCONT)
    assert result["exit_code"] != 0 and result["retired"] is False


def test_cleanup_uses_existing_fenced_owner_closure_before_retirement(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    backup = selected / "tldw_chatbook/Backup_Recovery"
    (backup / "storage_admission.py").write_text(
        "import threading\n"
        "_lock = threading.RLock()\n_live_leases = {1}\n_fenced = False\n"
        "class Pause:\n def resume(self):\n  assert not _live_leases\n"
        "def _begin_local_pause():\n global _fenced\n _fenced = True\n return Pause()\n"
        "def _shutdown():\n pass\n"
    )
    with (backup / "participants.py").open("a") as output:
        output.write(
            "def _retire_current_thread_caches(pause):\n"
            " from . import storage_admission as storage\n"
            " assert storage._fenced\n storage._live_leases.clear()\n"
        )
    (selected / "tldw_chatbook/DB/ChaChaNotes_DB.py").write_text(
        "from tldw_chatbook.Backup_Recovery import storage_admission\n"
        "raise RuntimeError('original failure')\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    assert result["retired"] is True
    assert result["error_type"] == "RuntimeError" and result["exit_code"] == 1


def test_cleanup_uses_public_quiescence_for_registered_foreign_thread_handles(tmp_path):
    module = probe()
    selected = source(module, tmp_path / "source")
    backup = selected / "tldw_chatbook/Backup_Recovery"
    (backup / "storage_admission.py").write_text(
        "import threading\n_lock = threading.RLock()\n_live_leases = {1}\n_fenced = False\n"
        "class Pause:\n def resume(self):\n  pass\n"
        "def _begin_local_pause():\n global _fenced\n _fenced = True\n return Pause()\n"
        "def _shutdown():\n pass\n"
    )
    with (backup / "participants.py").open("a") as output:
        output.write(
            "_installed_repositories = []\n"
            "def _retire_current_thread_caches(pause):\n pass\n"
        )
    (selected / "tldw_chatbook/config.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def get_chachanotes_db_path():\n return Path(os.environ['XDG_DATA_HOME']) / 'probe.db'\n"
    )
    (selected / "tldw_chatbook/DB/ChaChaNotes_DB.py").write_text(
        "import contextlib\nfrom types import SimpleNamespace\n"
        "from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage\n"
        "class CharactersRAGDB:\n"
        " def __init__(self, *args):\n  raise RuntimeError('preserved failure')\n"
        " @contextlib.contextmanager\n"
        " def quiesce_connections(self, *, timeout_seconds):\n"
        "  assert timeout_seconds == 0 and storage._fenced\n"
        "  storage._live_leases.clear()\n  yield\n"
        "owned = object.__new__(CharactersRAGDB)\n"
        "participants._installed_repositories.append(SimpleNamespace(\n"
        " owner_id='db.chachanotes.primary', repository=lambda: owned))\n"
    )
    manifest = json.loads((selected / module.MANIFEST).read_text())
    manifest["content_sha256"] = module.source_digest(selected)
    (selected / module.MANIFEST).write_text(json.dumps(manifest))
    result = module.run_child(selected, tmp_path / "profile", "transaction", 1)
    assert result["retired"] is True
    assert result["error_type"] == "RuntimeError" and result["exit_code"] == 1
