"""Actual standalone interruption after native child retirement, without an App."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from Tests.subprocess_pipes import popen_with_captured_stderr

_SCRIPT = r"""
import json
import os
import pwd
import signal
import sys
from pathlib import Path

from Tests import network_guard, real_profile_guard

entry = {key: os.environ.get(key) for key in ('HOME', 'USERPROFILE')}
assert entry == {'HOME': pwd.getpwuid(os.getuid()).pw_dir, 'USERPROFILE': None}
real_profile_guard.install()
network_guard.install()
assert os.environ['TLDW_TEST_REAL_HOME'] == entry['HOME']
assert str(Path(entry['HOME']) / '.config/tldw_cli').lower() in real_profile_guard.protected_roots()
assert not real_profile_guard.take_violations() and not network_guard.blocked_attempts()
from Tests.Backup_Recovery import test_mobile_mounted_manual as mounted

result_path = Path(sys.argv[1])
interruption = KeyboardInterrupt('owned child already reaped')
native_popen, native_killpg = mounted.subprocess.Popen, os.killpg
children, waits, signals = [], [], []
caught = None

def start_owned_child(argv, **options):
    assert not children, 'the App child must not be launched'
    assert Path(argv[-1]).name == 'native-seed.json'
    assert options['start_new_session'] is True
    assert {key: options['env'].get(key) for key in entry} == entry
    # Only this launch is replaced. The real standalone owns the Popen object,
    # session, log, wait, signal, exception path and interrupted receipt.
    child = native_popen([sys.executable, '-c', 'pass'], **options)
    native_wait = child.wait
    children.append((child, native_wait))

    def interrupted_wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            assert timeout == 300
            assert native_wait(timeout=5) == 0
            assert child.returncode == 0
            raise interruption
        return native_wait(timeout=timeout)

    child.wait = interrupted_wait
    return child

def signal_owned_group(pid, signum):
    assert len(children) == 1 and pid == children[0][0].pid
    assert pid != os.getpgrp()
    try:
        native_killpg(pid, signum)
    except ProcessLookupError:
        signals.append((pid, signum))
        raise
    raise AssertionError('already-reaped child unexpectedly retained a group')

mounted.subprocess.Popen = start_owned_child
os.killpg = signal_owned_group
try:
    try:
        mounted.run_standalone(result_path)
    except BaseException as error:
        caught = error
finally:
    mounted.subprocess.Popen = native_popen
    os.killpg = native_killpg
    for child, native_wait in children:
        child.wait = native_wait
        if child.poll() is None:
            child.kill()
        native_wait(timeout=5)
    facts = {
        'pids': [child.pid for child, _ in children],
        'returncodes': [child.returncode for child, _ in children],
        'wait_timeouts': waits,
        'native_esrch': signals,
        'caught_type': type(caught).__name__,
        'original_interruption': caught is interruption,
        'entry_home': entry,
        'exit_home': {key: os.environ.get(key) for key in entry},
        'profile_guard': real_profile_guard.take_violations(),
        'network_guard': network_guard.blocked_attempts(),
    }
    result_path.with_name('interruption-facts.json').write_text(json.dumps(facts, indent=2))

assert caught is interruption, facts
assert len(children) == 1 and facts['returncodes'] == [0]
assert waits == [300, 5]
assert signals == [(children[0][0].pid, signal.SIGTERM)]
report = json.loads(result_path.read_text())
assert report.get('completed') is not True
assert len(report['children']) == 1
record = report['children'][0]
assert record['name'] == 'seed' and record['pid'] == children[0][0].pid
assert record['returncode'] == 0 and record['interrupted'] is True
assert not Path(record['output']).exists()
assert not (Path(report['fixture']) / 'profile').exists()
assert 'tldw_chatbook.app' not in sys.modules and 'tldw_chatbook.config' not in sys.modules
assert facts['entry_home'] == facts['exit_home']
assert report['entry_home'] == report['exit_home'] == entry
assert not facts['profile_guard'] and not facts['network_guard']
assert not report['profile_guard'] and not report['network_guard']
"""


@pytest.mark.skipif(os.name != "posix", reason="standalone owns POSIX process groups")
def test_standalone_records_already_reaped_child_and_preserves_interrupt(tmp_path):
    import pwd

    account_home = pwd.getpwuid(os.getuid()).pw_dir
    assert os.environ["TLDW_TEST_REAL_HOME"] == account_home
    scratch = Path(tempfile.mkdtemp(prefix="mounted-cleanup-", dir="/tmp")).resolve(
        strict=True
    )
    assert not scratch.is_relative_to(Path(account_home))
    environment = os.environ.copy()
    # The fresh standalone parent starts with the actual UID home; pytest's
    # isolated HOME remains unchanged in this test process.
    environment.update(HOME=account_home, TMPDIR=str(scratch))
    environment.pop("USERPROFILE", None)
    repository = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(repository)
    stderr_path = tmp_path / "standalone-parent.stderr"
    parent = popen_with_captured_stderr(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            _SCRIPT,
            str(tmp_path / "standalone-result.json"),
        ],
        stderr_path,
        cwd=repository,
        env=environment,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert parent.wait(timeout=30) == 0, stderr_path.read_text()[-16000:]
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        parent.stderr.close()


@pytest.mark.skipif(os.name != "posix", reason="standalone owns POSIX process groups")
@pytest.mark.parametrize(
    "mutation",
    ("missing_record", "incomplete", "seeded_id", "config", "source", "target"),
)
def test_seed_evidence_rejects_missing_or_changed_fixture_before_producer(
    tmp_path, mutation
):
    import hashlib
    import json

    from Tests.Backup_Recovery.test_mobile_mounted_manual import (
        _closed_seed_files,
        _read_seed_record,
    )

    fixture = tmp_path / "fixture"
    fixture.mkdir(mode=0o700)
    selector = fixture / "profile/config.toml"
    source = fixture / "profile/data/shared/mobile"
    targets = fixture / "profile/new/nested"
    source.mkdir(mode=0o700, parents=True)
    targets.mkdir(mode=0o700, parents=True)
    selector.write_bytes(b"selected-config")
    leaves = [selector]
    for root, names in (
        (source, ("chat.db", "workspaces.db", "runs.db")),
        (targets, ("chat.db", "workspaces.db", "agent_runs.db")),
    ):
        for name in names:
            path = root / name
            path.write_bytes(name.encode())
            leaves.append(path)
    for path in leaves:
        path.chmod(0o600)
    files = _closed_seed_files(fixture)
    record = {
        "complete": True,
        "fixture": str(fixture),
        "seed_calls": [True, False],
        "seeded": {
            "legacy": {
                "conversation": "actual-seeded-id",
                "fingerprints": {
                    name: facts["sha256"] for name, facts in files["source"].items()
                },
            },
            "canonical": {"conversation": "actual-canonical-id"},
        },
        "files": files,
        "selected_config": selector.read_bytes().hex(),
    }
    path = fixture / "native-seed.json"
    path.write_text(json.dumps(record))
    path.with_name("native-seed-exit.json").write_text(
        json.dumps(
            {
                "entry_home": {"HOME": "unchanged", "USERPROFILE": None},
                "exit_home": {"HOME": "unchanged", "USERPROFILE": None},
                "profile_guard": [],
                "network_guard": [],
                "target_files": files["targets"],
            }
        )
    )
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _read_seed_record(fixture, expected_digest) == (record, expected_digest)
    if mutation == "missing_record":
        path.unlink()
    elif mutation == "incomplete":
        record["complete"] = False
        path.write_text(json.dumps(record))
    elif mutation == "seeded_id":
        record["seeded"]["legacy"]["conversation"] = "different-id"
        path.write_text(json.dumps(record))
    else:
        changed = {
            "config": selector,
            "source": source / "chat.db",
            "target": targets / "chat.db",
        }[mutation]
        changed.write_bytes(b"changed")
    with pytest.raises((AssertionError, FileNotFoundError)):
        _read_seed_record(
            fixture, None if mutation == "incomplete" else expected_digest
        )


@pytest.mark.skipif(os.name != "posix", reason="standalone owns POSIX process groups")
@pytest.mark.parametrize("posture", ("missing", "linked", "public"))
def test_prepared_seed_reuse_refuses_invalid_profile_before_seed(
    tmp_path, monkeypatch, posture
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared

    def forbidden_seed(*args, **kwargs):
        pytest.fail("invalid reused profile reached native fixture seeding")

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", forbidden_seed)
    profile = tmp_path / "profile"
    if posture == "linked":
        actual = tmp_path / "actual"
        actual.mkdir(mode=0o700)
        profile.symlink_to(actual, target_is_directory=True)
    elif posture == "public":
        profile.mkdir(mode=0o700)
        profile.chmod(0o755)
    with (
        pytest.raises((AssertionError, FileNotFoundError)),
        prepared(tmp_path, monkeypatch, reuse_seeded_profile=True),
    ):
        pytest.fail("invalid reused profile reached admission")
