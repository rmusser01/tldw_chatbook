"""Actual disposable APFS image tests; no native qualification overrides.

Set TLDW_TEST_APFS_MOUNT to the mounted private test image. The surrounding
fixture owns image attachment/detachment; these tests create only private children.
Raw primitive success does not qualify destination publication.
"""

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import pytest


def _run_image_case(tmp_path, program, receipt):
    supplied = os.environ.get("TLDW_TEST_APFS_MOUNT")
    if not supplied:
        pytest.skip("requires an explicitly mounted disposable APFS test image")
    mount = Path(supplied).resolve(strict=True)
    assert mount.is_dir() and mount.stat().st_dev != tmp_path.stat().st_dev
    image_source = Path(tempfile.mkdtemp(prefix="chatbook-test-", dir=mount))
    for name in ("home", "config", "data", "cache", "host", "logs", "tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
    }
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        TMPDIR=str(tmp_path / "tmp"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        VOLUME_FIXTURE=str(tmp_path),
        IMAGE_SOURCE=str(image_source),
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    (tmp_path / "fixture.json").write_text(
        json.dumps(
            {
                "image_source": str(image_source),
                "source_device": image_source.stat().st_dev,
                "host_device": tmp_path.stat().st_dev,
            },
            indent=2,
        )
    )
    with (tmp_path / "child-output.log").open("w") as output:
        # The interpreter and child program are fixed, owned fixture code.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", program],
            cwd=environment["PYTHONPATH"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
    assert result.returncode == 0, (tmp_path / "child-output.log").read_text()[-7000:]
    return json.loads((tmp_path / "logs" / receipt).read_text())


def test_actual_image_primitives_and_normal_qualification_boundary(tmp_path):
    evidence = _run_image_case(tmp_path, _RAW, "raw-evidence.json")
    assert evidence["host_authority_image_source"] == "normal_and_maintenance_passed"


def test_actual_image_database_and_files_capture_to_host_archive(tmp_path):
    evidence = _run_image_case(tmp_path, _CAPTURE, "capture-evidence.json")
    assert evidence["source_device"] != evidence["archive_device"]
    assert evidence["empty_directory_retained"]
    assert evidence["blocked_network_attempts"] == 0


_RAW = r"""
from Tests.network_guard import install, blocked_attempts
install()
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory, native_identity, create_private_directory, create_private_file, flush_directory, _rename_new, publish_new
from tldw_chatbook.Backup_Recovery.qualification import qualified_for
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionError

root = Path(os.environ['VOLUME_FIXTURE'])
host = root / 'host'
image = Path(os.environ['IMAGE_SOURCE'])
facts = {'identities': {}, 'raw': [], 'qualification': {}}
for label, path in (('host', host), ('image', image)):
    with pinned_directory(path) as fd:
        facts['identities'][label] = {'identity': native_identity(fd), 'device': os.fstat(fd).st_dev}
    facts['qualification'][label] = {operation: qualified_for(operation, path) for operation in ('publish_new','publish_file','publish_directory','admission')}
assert facts['identities']['host']['device'] != facts['identities']['image']['device']
raw = image / 'raw'
create_private_directory(raw)
for kind in ('file', 'empty_directory', 'populated_directory'):
    case = raw / kind
    create_private_directory(case)
    source, existing = case/'source', case/'existing'
    if kind == 'file':
        source.write_bytes(b'candidate'); existing.write_bytes(b'previous')
    else:
        source.mkdir(mode=0o700); existing.mkdir(mode=0o700)
        if kind == 'populated_directory':
            (source/'payload').write_bytes(b'candidate'); (existing/'payload').write_bytes(b'previous')
    with pinned_directory(case) as fd:
        try: _rename_new(fd, 'source', fd, 'existing')
        except FileExistsError: pass
        else: raise AssertionError('no-replace overwritten')
        assert source.exists() and existing.exists()
        _rename_new(fd, 'source', fd, 'published')
        flush_directory(fd)
    assert not source.exists() and (case/'published').exists()
    if kind == 'file': assert existing.read_bytes()==b'previous' and (case/'published').read_bytes()==b'candidate'
    if kind == 'populated_directory': assert (existing/'payload').read_bytes()==b'previous' and (case/'published/payload').read_bytes()==b'candidate'
    facts['raw'].append(kind)
with create_private_file(raw/'private-file') as fd: os.write(fd,b'private')
assert (raw/'private-file').stat().st_mode & 0o777 == 0o600
facts['raw'].append('private_creation_and_full_flush')
cross = host/'cross-source'
cross.write_bytes(b'cross-device retained')
with pinned_directory(host) as a, pinned_directory(raw) as b:
    try: _rename_new(a,'cross-source',b,'cross-destination')
    except OSError as error: assert error.errno==errno.EXDEV,error
    else: raise AssertionError('cross-device rename unexpectedly succeeded')
assert cross.read_bytes()==b'cross-device retained' and not (raw/'cross-destination').exists()
facts['raw'].append('EXDEV_preserved_source')
death = raw/'death'
create_private_directory(death)
(death/'source').write_bytes(b'abrupt native evidence')
child = "from Tests.network_guard import install; install(); import os,sys; from pathlib import Path; from tldw_chatbook.Backup_Recovery.native_files import pinned_directory,_rename_new;\nwith pinned_directory(Path(sys.argv[1])) as fd:\n _rename_new(fd,'source',fd,'published'); os._exit(23)"
run = subprocess.run([sys.executable,'-c',child,str(death)],timeout=10)
assert run.returncode==23 and not (death/'source').exists() and (death/'published').read_bytes()==b'abrupt native evidence'
facts['raw'].append('process_exit_after_raw_rename')
candidate = raw/'gated-candidate'
candidate.write_bytes(b'unqualified candidate');candidate.chmod(0o600)
if not facts['qualification']['image']['publish_new'][0]:
    try: publish_new(candidate,raw/'gated-target')
    except OSError as error: assert str(error)==facts['qualification']['image']['publish_new'][1],error
    else: raise AssertionError('unqualified publication accepted')
    assert candidate.read_bytes()==b'unqualified candidate' and not (raw/'gated-target').exists()
    facts['gated_publication']='refused_before_mutation'
if not facts['qualification']['image']['admission'][0]:
    try: Admission(raw/'control')
    except AdmissionError as error: assert str(error)==facts['qualification']['image']['admission'][1],error
    else: raise AssertionError('unqualified admission accepted')
    assert not (raw/'control').exists()
    facts['image_authority']='refused_before_creation'
authority = Admission(host/'control')
authority.register('image.source',(raw/'private-file',))
with authority.normal(('image.source',)): assert (raw/'private-file').read_bytes()==b'private'
with authority.maintenance(('image.source',),3): assert (raw/'private-file').read_bytes()==b'private'
facts['host_authority_image_source']='normal_and_maintenance_passed'
assert not blocked_attempts(),blocked_attempts()
(root/'logs/raw-evidence.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts),flush=True)
"""


_CAPTURE = r"""
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import threading
import zipfile
from contextlib import closing
from pathlib import Path

from Tests.network_guard import blocked_attempts, install

install()
for name in ("sounddevice", "pyaudio"):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring

keyring.set_keyring(Keyring())
root = Path(os.environ["VOLUME_FIXTURE"])
selector = Path(os.environ["TLDW_CONFIG_PATH"])
image_source = Path(os.environ["IMAGE_SOURCE"])
external = image_source / "documents"
external.mkdir(mode=0o700)
(external / "empty").mkdir(mode=0o700)
document = external / "retained.txt"
document.write_bytes(b"Actual second-volume document\n")
document.chmod(0o600)
research_path = image_source / "research.db"
(root / "data" / "profile").mkdir(mode=0o700)
selector.write_text(
    '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
    '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
    '[paths]\ndata_dir=' + json.dumps(str(root / "data" / "profile")) + '\n'
    '[database]\nresearch_db_path=' + json.dumps(str(research_path)) + '\n'
)
selector.chmod(0o600)

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app


async def main():
    app = TldwCli()
    monitoring = None
    cancel = threading.Event()
    watchdog = None
    try:
        research = app.local_research_service
        assert research.db_path == research_path
        saved = research.create_session(title="Image research", query="Retained on image")
        assert research.get_session(saved["id"])["query"] == "Retained on image"
        assert research_path.stat().st_dev != selector.stat().st_dev
        options = {"allow_partial": True, "staging_parent": root / "host", "external_roots": (external,)}
        preview = preview_capture((selector,), options=options)
        rows = [item for item in preview.items if item.path is not None and (item.path == research_path or external == item.path or external in item.path.parents)]
        assert any(item.owner == "research.local" and item.path == research_path and item.status == "included" for item in rows)
        assert any(item.path == document and item.status == "included" for item in rows)
        assert any(item.path == external / "empty" and item.status == "included_directory" for item in rows)
        (root / "logs" / "capture-preview.json").write_text(json.dumps({"complete": preview.complete, "issues": preview.issues, "selected_image_rows": [{"id": item.logical_id, "owner": item.owner, "path": str(item.path), "status": item.status} for item in rows]}, indent=2))
        print("PREVIEW", preview.complete, preview.issues, flush=True)
        destination = root / "host" / "image-source.tldw-backup.zip"
        monitoring = asyncio.create_task(monitor_app(app))
        watchdog = asyncio.get_running_loop().call_later(55, cancel.set)
        result = await asyncio.to_thread(capture, (selector,), preview.scope_digest, destination, options=options, cancel=cancel)
        print("CAPTURED", flush=True)
        for _ in range(500):
            if storage_admission._pause is None and app._backup_runtime_maintenance is None:
                break
            await asyncio.sleep(.01)
        assert storage_admission._pause is None and app._backup_runtime_maintenance is None
        resumed = research.create_session(title="After capture", query="Resumed original writer")
        assert research.get_session(resumed["id"])["query"] == "Resumed original writer"
        manifest = json.loads(result.manifest_bytes)
        (root / "logs" / "capture-manifest.json").write_bytes(result.manifest_bytes)
        logical_paths = {item.logical_id: item.path for item in result.inventory.items}
        research_member = next(member for member in manifest["files"] if logical_paths[member["logical_id"]] == research_path)
        document_member = next(member for member in manifest["files"] if logical_paths[member["logical_id"]] == document)
        assert (result.root / document_member["payload"]).read_bytes() == document.read_bytes()
        assert result.root.stat().st_dev == selector.stat().st_dev
        assert bootstrap.default_bootstrap_root().stat().st_dev == selector.stat().st_dev
        sealed_output = await asyncio.to_thread(write_archive, result, destination, password=None, cancel=cancel)
        sealed = await asyncio.to_thread(acquire, destination, root / "host" / "readback", ArchiveLimits(), None, cancel)
        verify_sealed(sealed, cancel)
        final_manifest = json.loads(sealed.manifest_bytes)
        assert final_manifest["consistency"] == "partial"  # Explicit external files.
        assert final_manifest["files"] == manifest["files"]
        with zipfile.ZipFile(sealed.path) as archive:
            assert archive.read(document_member["payload"]) == document.read_bytes()
            readback = root / "host" / "research-readback.db"
            readback.write_bytes(archive.read(research_member["payload"]))
        with closing(sqlite3.connect(readback.as_uri() + "?mode=ro", uri=True)) as connection:
            assert connection.execute("SELECT id,title,query FROM research_sessions").fetchall() == [(saved["id"], "Image research", "Retained on image")]
        assert research.get_session(saved["id"])["query"] == "Retained on image"
        assert not blocked_attempts()
        evidence = {"consistency": final_manifest["consistency"], "inventory_issues": list(preview.issues), "source_device": research_path.stat().st_dev, "control_device": bootstrap.default_bootstrap_root().stat().st_dev, "stage_device": result.root.stat().st_dev, "archive_device": destination.stat().st_dev, "archive": str(destination), "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "research_id": saved["id"], "resumed_id": resumed["id"], "archive_files": len(final_manifest["files"]), "empty_directory_retained": any(logical_paths[entry["logical_id"]] == external / "empty" for entry in final_manifest["directories"]), "blocked_network_attempts": len(blocked_attempts())}
        (root / "logs" / "capture-evidence.json").write_text(json.dumps(evidence, indent=2))
        print("SUCCESS", json.dumps(evidence), flush=True)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        cancel.set()
        if monitoring is not None:
            monitoring.cancel()
            try:
                await monitoring
            except asyncio.CancelledError:
                pass
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()


asyncio.run(main())
"""
