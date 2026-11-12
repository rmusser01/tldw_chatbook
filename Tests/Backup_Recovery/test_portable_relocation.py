"""Portable selection hints cannot substitute for committed recovery authority."""

import json
import os
import subprocess
import sys

import pytest


def test_selection_hint_with_escaping_control_is_refused(tmp_path):
    from tldw_chatbook.Backup_Recovery.portable_relocation import read_selection

    data = tmp_path / "Data"
    work = data / ".portable-relocation"
    work.mkdir(parents=True)
    (work / "selection-000001.json").write_text(
        json.dumps({"version": 1, "control": "../../outside"})
    )
    with pytest.raises(ValueError, match="portable_selection_invalid"):
        read_selection(data)


def test_no_selection_does_not_initialize_or_open_a_profile(tmp_path):
    from tldw_chatbook.Backup_Recovery.portable_relocation import read_selection

    data = tmp_path / "Data"
    data.mkdir()
    assert read_selection(data) is None
    assert list(data.iterdir()) == []


def test_concurrent_relocation_lock_refuses_second_contender(tmp_path):
    from tldw_chatbook.Backup_Recovery.portable_relocation import relocation_lock

    data = tmp_path / "Data"
    data.mkdir()
    with (
        relocation_lock(data),
        pytest.raises(ValueError, match="portable_relocation_busy"),
        relocation_lock(data),
    ):
        pytest.fail("second contender acquired relocation")


def test_ambiguous_started_publication_never_allocates_another_attempt(tmp_path):
    from tldw_chatbook.Backup_Recovery.portable_relocation import pending_attempt

    data = tmp_path / "Data"
    work = data / ".portable-relocation"
    attempt = work / ("attempt-" + "1" * 32)
    attempt.mkdir(parents=True)
    (attempt / "publication-started.json").write_text('{"version":1}')
    with pytest.raises(ValueError, match="portable_relocation_recovery_required"):
        pending_attempt(data)
    assert len(list(work.glob("attempt-*"))) == 1


def test_commit_before_selection_is_resumed_instead_of_republished(tmp_path):
    from tldw_chatbook.Backup_Recovery.portable_relocation import (
        _write,
        pending_attempt,
    )

    data = tmp_path / "Data"
    attempt = data / ".portable-relocation" / ("attempt-" + "1" * 32)
    attempt.mkdir(parents=True)
    _write(
        attempt,
        "publication-started.json",
        {
            "version": 1,
            "location": str(data.parent),
            "source_digest": "2" * 64,
        },
    )
    _write(attempt, "committed.json", {"version": 1})
    assert pending_attempt(data) == attempt


def test_repeated_move_acquires_confined_selected_payload_without_old_location(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.portable_relocation import (
        PortableSelection,
        acquire_selected_source,
    )

    data = tmp_path / "moved again ü" / "Data"
    attempt = "attempt-" + "1" * 32
    base = data / ".portable-relocation" / attempt
    config = base / "profiles" / "restored" / "config"
    payload = base / "profiles" / "restored" / "data"
    config.mkdir(parents=True)
    payload.mkdir()
    (config / "config.toml").write_bytes(b"settings")
    (payload / "note").write_bytes(b"selected note")
    (base / "home" / ".config" / "tldw_cli" / "recovery-bootstrap").mkdir(parents=True)
    (base / "control").mkdir()
    old = tmp_path / "original unavailable"
    selection = PortableSelection(
        1,
        str(old),
        attempt,
        "2" * 32,
        (config / "config.toml").relative_to(data).as_posix(),
        payload.relative_to(data).as_posix(),
        "3" * 64,
    )
    view, reviews = acquire_selected_source(data, selection, tmp_path / "view")
    assert (
        view / "home" / ".config" / "tldw_cli" / "config.toml"
    ).read_bytes() == b"settings"
    assert (view / selection.data / "note").read_bytes() == b"selected note"
    assert json.loads((view / "portable-location.json").read_text())["root"] == str(old)
    assert len(reviews) == 4
    assert not old.exists()


def test_fresh_host_drops_provider_environment_and_uses_private_temp(tmp_path):
    data = tmp_path / "Data"
    attempt = "attempt-" + "1" * 32
    (data / ".portable-relocation" / attempt / "home").mkdir(parents=True)
    code = """
import json,os,sys,tempfile
from pathlib import Path
from tldw_chatbook.Backup_Recovery.portable_relocation import host_environment
host_environment(Path(sys.argv[1]),sys.argv[2])
print(json.dumps({'secret':os.getenv('OPENAI_API_KEY'),'tmp':tempfile.gettempdir()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(data), attempt],
        env={**os.environ, "OPENAI_API_KEY": "synthetic-secret"},
        text=True,
        capture_output=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    assert actual["secret"] is None
    assert actual["tmp"] == str(data / ".portable-relocation" / attempt / "tmp")


def test_fresh_profile_child_keeps_explicit_terminal_roots_only(monkeypatch):
    from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_environment

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("CHATBOOK_WEZTERM_HOME", "portable-home")
    monkeypatch.setenv("CHATBOOK_WEZTERM_DATA_DIR", "portable-data")
    monkeypatch.setenv("CHATBOOK_WEZTERM_RUNTIME_DIR", "portable-runtime")
    result = _launch_environment()
    assert "OPENAI_API_KEY" not in result
    assert result["CHATBOOK_WEZTERM_HOME"] == "portable-home"
    assert result["CHATBOOK_WEZTERM_DATA_DIR"] == "portable-data"
    assert result["CHATBOOK_WEZTERM_RUNTIME_DIR"] == "portable-runtime"


@pytest.mark.parametrize("forged_commit", [None, "malformed", "selected"])
def test_move_again_cannot_hide_an_unsettled_previous_publication(
    tmp_path, forged_commit
):
    from dataclasses import asdict

    from tldw_chatbook.Backup_Recovery.portable_relocation import (
        PortableSelection,
        _write,
        pending_attempt,
    )

    data = tmp_path / "third location" / "Data"
    work = data / ".portable-relocation"
    pending = work / ("attempt-" + "2" * 32)
    pending.mkdir(parents=True)
    selected = PortableSelection(
        1,
        str(tmp_path / "first location"),
        "attempt-" + "1" * 32,
        "3" * 32,
        ".portable-relocation/attempt-"
        + "1" * 32
        + "/profiles/restored/config/config.toml",
        ".portable-relocation/attempt-" + "1" * 32 + "/profiles/restored/data",
        "4" * 64,
    )
    _write(work, "selection-000001.json", asdict(selected))
    _write(
        pending,
        "publication-started.json",
        {
            "version": 1,
            "location": str(tmp_path / "second location"),
            "source_digest": "5" * 64,
        },
    )
    if forged_commit == "selected":
        completed = PortableSelection(
            1,
            str(tmp_path / "second location"),
            pending.name,
            "6" * 32,
            f".portable-relocation/{pending.name}/profiles/restored/config/config.toml",
            f".portable-relocation/{pending.name}/profiles/restored/data",
            "5" * 64,
        )
        _write(pending, "committed.json", asdict(completed))
        _write(work, "selection-000002.json", asdict(completed))
    elif forged_commit:
        _write(pending, "committed.json", {"version": 1})
    with pytest.raises(ValueError, match="portable_relocation_recovery_required"):
        pending_attempt(data)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("external_edit_after_seal", [False, True])
def test_reviewed_import_commits_real_notes_and_keeps_activation_review(
    tmp_path, external_edit_after_seal
):
    code = r"""
import hashlib,json,sqlite3,sys,tomllib
from pathlib import Path
from Tests.network_guard import install
install()
from Tests.Backup_Recovery.test_detached_layout import _base_profile
from tldw_chatbook.Backup_Recovery.portable_relocation import relocate
base=Path(sys.argv[1]);base.mkdir()
source,_,old=_base_profile(base)
assert not old.exists()
before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob('*') if path.is_file()}
if sys.argv[2]=='edit':
    import tldw_chatbook.Backup_Recovery.isolated_restore as executor
    original=executor.restore_isolated
    def external_edit(*args,**kwargs):
        note=source/'home/.local/share/tldw_cli/fixture/tldw_chatbook_ChaChaNotes.db'
        note.write_bytes(b'external actor edit after sealed review')
        before[note]=hashlib.sha256(note.read_bytes()).hexdigest()
        return original(*args,**kwargs)
    executor.restore_isolated=external_edit
def confirm(lines,digest):
    assert 'tldw_chatbook.config' not in sys.modules
    assert len(digest)==64
    assert any(line.startswith('Included saved data:') for line in lines)
    assert any(line.startswith('Fresh destination:') for line in lines)
    assert any(line.startswith('Estimated additional free space required:') for line in lines)
    assert not list(source.glob('.portable-relocation/attempt-*/publication-started.json'))
    return True
selected=relocate(source,confirm)
assert selected is not None
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in before.items())
from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_state,profile_requirements
control=source/'.portable-relocation'/selected.attempt/'control'
entry,_=_launch_state(selected.profile_id,control)
assert entry.installation_id!='tldw_cli_local_instance_v1'
cfg=tomllib.loads((source/selected.config).read_text())
assert not cfg['api_settings']['openai'].get('api_key')
db=Path(cfg['database']['chachanotes_db_path'])
assert db.is_relative_to(source)
with sqlite3.connect(f'{db.as_uri()}?mode=ro&immutable=1',uri=True) as connection:
    assert connection.execute("SELECT content FROM notes WHERE id='kept'").fetchone()==('Copied note',)
    assert connection.execute("SELECT title FROM conversations WHERE id='chat'").fetchone()==('Copied chat',)
requirements=profile_requirements(selected.profile_id,control)
assert requirements['required_owners']
assert requirements['pending_owners']==requirements['required_owners']
print('SUCCESS '+json.dumps({'profile':selected.profile_id,'data':str(source)}),flush=True)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "fixture"),
            "edit" if external_edit_after_seal else "unchanged",
        ],
        text=True,
        capture_output=True,
        timeout=220,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-3000:]
    assert "SUCCESS " in result.stdout
