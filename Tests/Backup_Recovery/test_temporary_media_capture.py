"""Temporary media becomes private retained owner data without live TTL changes."""

import hashlib
import json
import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Backup_Recovery.test_core_owners import application_authority


def test_private_materialization_retains_video_and_unreferenced_gallery(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import recovered_media as media
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    live = tmp_path / "live-video"
    live.write_bytes(b"actual temporary video")
    live.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    video = stage / "video-copy"
    video.write_bytes(live.read_bytes())
    video.chmod(0o600)
    image = stage / "image-copy"
    image.write_bytes(b"unreferenced gallery image")
    image.chmod(0o600)
    authority = application_authority(tmp_path, live, monkeypatch)
    reference = ("profile", "message", "clip", "video/mp4")
    limits = ArchiveLimits()
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session._capture_bound_sources((), stage, limits, limits.expanded_bytes),
    ):
        result = media.materialize_temporary_media(
            session,
            stage / "recovered",
            (
                media.TemporaryMediaSource(
                    "video-source", video, "video/mp4", reference
                ),
                media.TemporaryMediaSource("image-source", image, "image/png"),
            ),
            cancel=Event(),
            limits=limits,
            byte_budget=limits.expanded_bytes,
        )
        assert (
            media._RecoveredAdapter().validate(stage / "recovered/catalog.sqlite3")
            == ()
        )
    root = stage / "recovered"
    assert (
        root / (result["video-source"] + ".payload")
    ).read_bytes() == live.read_bytes()
    assert (
        root / (result["image-source"] + ".payload")
    ).read_bytes() == image.read_bytes()
    with closing(sqlite3.connect(root / "catalog.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM refs").fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT profile,message,slug,media_type FROM refs"
            ).fetchone()
            == reference
        )
    assert live.read_bytes() == b"actual temporary video"
    assert not (live.parent / "recovered_media").exists()


def test_native_transcript_identity_reader_preserves_real_video_keys(tmp_path):
    from Tests.Backup_Recovery.test_core_owners import adapter_for
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    owner = CharactersRAGDB(tmp_path / "chat.db", "fixture")
    try:
        conversation = owner.add_conversation({"title": "Captured videos"})
        metadata = VideoGenerationMetadata(
            name="original-clip", prompt="test", backend="comfyui", container="webm"
        )
        message = owner.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "video",
                "metadata_json": metadata.to_json(),
            }
        )
        owner.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "image",
                "image_data": b"stored DB image",
                "image_mime_type": "image/png",
            }
        )
        owner.close()
        references = adapter_for("chachanotes").temporary_video_references(
            owner.db_path, cancel=Event()
        )
        assert references == ((message, "original-clip", "video/webm"),)
    finally:
        owner.close()


@pytest.mark.parametrize(
    "message,slug,container",
    [
        ("../outside", "clip", "mp4"),
        ("message", "../outside", "mp4"),
        ("message", ".video-stage-private", "mp4"),
        ("message", "clip", "avi"),
    ],
)
def test_video_reference_path_rejects_non_owned_names(message, slug, container):
    from tldw_chatbook.Video_Generation import video_metadata

    with pytest.raises(ValueError):
        video_metadata.video_relative_path(message, slug, container)


@pytest.fixture
def private_case(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    source = tmp_path / "live"
    source.write_bytes(b"live temporary source")
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    copied = stage / "copied"
    copied.write_bytes(source.read_bytes())
    copied.chmod(0o600)
    authority = application_authority(tmp_path, source, monkeypatch)
    return authority, stage, copied, ArchiveLimits()


@pytest.mark.parametrize(
    "damage", ["outside", "symlink", "hardlink", "cancel", "budget", "duplicate"]
)
def test_private_materialization_refuses_unapproved_or_unbounded_input(
    private_case, tmp_path, damage
):
    from tldw_chatbook.Backup_Recovery import recovered_media as media
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    authority, stage, copied, limits = private_case
    root = stage / "recovered"
    cancel = Event()
    budget = limits.expanded_bytes
    if damage == "outside":
        copied = tmp_path / "live"
    elif damage == "symlink":
        copied = stage / "alias"
        copied.symlink_to(stage / "copied")
    elif damage == "hardlink":
        import os

        os.link(copied, stage / "second-name")
    elif damage == "cancel":
        cancel.set()
    elif damage == "budget":
        budget = 1
    sources = (media.TemporaryMediaSource("source", copied, "image/png"),)
    if damage == "duplicate":
        sources *= 2
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session._capture_bound_sources((), stage, limits, limits.expanded_bytes),
        pytest.raises((ValueError, InterruptedError, RecoveryRequired)),
    ):
        media.materialize_temporary_media(
            session, root, sources, cancel=cancel, limits=limits, byte_budget=budget
        )
    assert not root.exists()


def test_materialization_without_actual_held_scope_creates_nothing(tmp_path):
    from tldw_chatbook.Backup_Recovery import recovered_media as media
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="temporary_capture_scope_required"):
        media.materialize_temporary_media(
            None, root, (), cancel=Event(), limits=ArchiveLimits(), byte_budget=1024
        )
    assert not root.exists()


@pytest.mark.parametrize("state", ["ready", "deleted", "missing", "collision"])
def test_existing_recovered_identity_and_tombstone_are_never_reassigned(
    private_case, state
):
    from tldw_chatbook.Backup_Recovery import recovered_media as media
    from tldw_chatbook.Backup_Recovery.recovered_media_schema import migrate

    authority, stage, copied, limits = private_case
    root = stage / "recovered"
    root.mkdir(mode=0o700)
    asset = "a" * 32
    reference = ("profile", "message", "clip", "video/mp4")
    payload = b"other content" if state == "collision" else copied.read_bytes()
    with closing(sqlite3.connect(root / "catalog.sqlite3")) as connection:
        migrate(connection)
        with connection:
            connection.execute(
                "INSERT INTO assets VALUES (?,?,?,?,?)",
                (
                    asset,
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                    "video/mp4",
                    "deleted" if state == "deleted" else "ready",
                ),
            )
            connection.execute(
                "INSERT INTO refs VALUES (?,?,?,?,?)", reference + (asset,)
            )
            if state == "deleted":
                connection.execute(
                    "INSERT INTO tombstones VALUES (?,1,?)",
                    (asset, json.dumps([reference])),
                )
        before = tuple(connection.iterdump())
    (root / "catalog.sqlite3").chmod(0o600)
    if state not in {"deleted", "missing"}:
        (root / (asset + ".payload")).write_bytes(payload)
        (root / (asset + ".payload")).chmod(0o600)
    sources = (media.TemporaryMediaSource("source", copied, "video/mp4", reference),)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session._capture_bound_sources((), stage, limits, limits.expanded_bytes),
    ):
        if state in {"missing", "collision"}:
            with pytest.raises((ValueError, FileNotFoundError)):
                media.materialize_temporary_media(
                    session,
                    root,
                    sources,
                    cancel=Event(),
                    limits=limits,
                    byte_budget=limits.expanded_bytes,
                )
        else:
            assert media.materialize_temporary_media(
                session,
                root,
                sources,
                cancel=Event(),
                limits=limits,
                byte_budget=limits.expanded_bytes,
            ) == {"source": asset}
    with closing(sqlite3.connect(root / "catalog.sqlite3")) as connection:
        assert tuple(connection.iterdump()) == before
    assert (root / (asset + ".payload")).exists() == (
        state not in {"deleted", "missing"}
    )


@pytest.mark.parametrize("kind", ["legacy", "malformed", "unsafe", "cancel", "limit"])
def test_native_video_identity_reader_bounds_and_legacy_container(tmp_path, kind):
    from Tests.Backup_Recovery.test_core_owners import adapter_for
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    owner = CharactersRAGDB(tmp_path / "chat.db", "fixture")
    try:
        conversation = owner.add_conversation({"title": "Retained metadata"})
        metadata = {
            "video_generation": {
                "name": "clip",
                "backend": "comfyui",
                "prompt": "legacy",
            }
        }
        if kind == "malformed":
            metadata["video_generation"] = []
        elif kind == "unsafe":
            metadata["video_generation"]["name"] = "../outside"
        message = owner.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "video",
                "metadata_json": json.dumps(metadata),
            }
        )
        owner.close()
        cancel = Event()
        if kind == "cancel":
            cancel.set()
        if kind == "legacy":
            assert adapter_for("chachanotes").temporary_video_references(
                owner.db_path, cancel=cancel
            ) == ((message, "clip", "video/mp4"),)
        else:
            with pytest.raises((ValueError, InterruptedError)):
                adapter_for("chachanotes").temporary_video_references(
                    owner.db_path,
                    cancel=cancel,
                    max_metadata_bytes=1 if kind == "limit" else 16 * 1024**2,
                )
    finally:
        owner.close()


_PUBLIC = r"""
import asyncio, json, os, sqlite3, sys, threading
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home = Path.home()
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.recovered_media import current_profile_id
from tldw_chatbook.Media_Creation.image_generation_service import ImageGenerationService
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

async def main():
    app = TldwCli()
    images = ImageGenerationService()
    gallery = images.output_dir / 'temp' / 'generated.png'
    gallery.write_bytes(b'available temporary gallery')
    store = app.console_runtime.ensure_chat_store()
    session = store.ensure_session(title='Retained temporary media')
    videos = app.generated_video_store
    orphan = videos.save('orphan-message', 'orphan-clip', b'orphan video bytes', extension='mp4')
    originals = {}
    for container in ('mp4', 'webm'):
        message = 'message-' + container
        metadata = VideoGenerationMetadata(name='clip', prompt='temporary proof', backend='comfyui', container=container)
        originals[container] = videos.save(message, 'clip', ('available ' + container).encode(), extension=container)
        store.append_video_message(session.id, video_metadata=metadata, persist=True, message_id=message)
    store.append_video_message(session.id, video_metadata=VideoGenerationMetadata(name='expired', prompt='expired', backend='comfyui'), persist=True, message_id='expired-message')
    options = {'staging_parent': home, 'temporary_media': True}
    preview = preview_capture((selector,), options=options)
    assert preview.complete, (preview.issues, [(i.owner, i.status) for i in preview.items if i.status == 'unsupported'])
    destination = home / 'retained-temporary.tldw-backup.zip'
    monitoring = asyncio.create_task(monitor_app(app))
    cancel = threading.Event()
    watchdog = asyncio.get_running_loop().call_later(30, cancel.set)
    try:
        captured = await asyncio.to_thread(capture, (selector,), preview.scope_digest, destination, options=options, cancel=cancel)
        for _ in range(500):
            if storage._pause is None and app._backup_runtime_maintenance is None:
                break
            await asyncio.sleep(.01)
        assert storage._pause is None and app._backup_runtime_maintenance is None
        # Ordinary source writes resume before packaging; TTL sources stay in place.
        videos.save('after-capture', 'clip', b'resumed native video', extension='mp4')
        assert gallery.read_bytes() == b'available temporary gallery'
        assert all(path.read_bytes() == ('available ' + kind).encode() for kind, path in originals.items())
        assert not (videos.root.parent / 'recovered_media').exists()
        manifest = json.loads(captured.manifest_bytes)
        assert captured.inventory.complete and manifest['consistency'] == 'coherent'
        assert any(i.owner == 'generation.assets' and i.status == 'included' for i in captured.inventory.items)
        assert not [f for f in manifest['files'] if f['owner_id'] == 'generation.assets']
        members = [f for f in manifest['files'] if f['owner_id'] == 'recovered.media']
        catalog = next(f for f in members if f['relative_path'] == 'catalog.sqlite3')
        with sqlite3.connect(captured.root / catalog['payload']) as connection:
            assets = connection.execute('SELECT asset_id,media_type FROM assets ORDER BY media_type').fetchall()
            refs = connection.execute('SELECT profile,message,slug,media_type,asset_id FROM refs ORDER BY media_type').fetchall()
        assert len(assets) == 4 and len(refs) == 2
        assert [(p,m,s,t) for p,m,s,t,a in refs] == [(current_profile_id(),'message-'+kind,'clip','video/'+kind) for kind in ('mp4','webm')]
        payloads = {f['relative_path']: captured.root / f['payload'] for f in members}
        referenced_assets = {ref[-1] for ref in refs}
        for asset, kind in assets:
            expected = b'available temporary gallery' if kind == 'image/png' else ('available ' + kind.split('/')[1]).encode()
            if kind.startswith('video/') and asset not in referenced_assets:
                expected = b'orphan video bytes'
            assert payloads[asset+'.payload'].read_bytes() == expected
        assert orphan.read_bytes() == b'orphan video bytes'
        assert 'Unavailable temporary references: 1' in manifest['report']['lines']
        assert not destination.exists()
        await asyncio.to_thread(write_archive, captured, destination, password=None, cancel=threading.Event())
        assert destination.is_file()
        assert not blocked_attempts()
    finally:
        watchdog.cancel()
        cancel.set()
        monitoring.cancel()
        try:
            await monitoring
        except asyncio.CancelledError:
            pass
        try:
            await app._shutdown_app_owned_lifecycles()
        except asyncio.CancelledError:
            pass
        try:
            await app.tts_service.close()
        except asyncio.CancelledError:
            pass
asyncio.run(main())
print('retired and reopened')
"""


def test_public_capture_materializes_available_temporary_bytes_and_resumes(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "temporary", "complete", script=_PUBLIC)


_RESTORE = r"""
import json, os, sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
home = Path.home()
source = Path(sys.argv[1])
sealed = acquire(source / 'retained-temporary.tldw-backup.zip', home / 'acquired', ArchiveLimits(), None, Event())
doc = verify_sealed(sealed)
profile = doc.profile_ids[0]
destination = home / 'destination'
destination.mkdir(mode=0o700)
data = destination / 'data' / 'restored'
producer = {row.logical_id: row for row in doc.producer_inventory}
leaves = {'agents.history':'tool_sandbox', 'chat.dictionaries':'chat_dicts', 'chatbooks.archives':'chatbooks', 'runtime.chatbook_scratch':'temp', 'skills':'skills', 'persona.visual_identity_builtin':'persona-assets', 'generation.assets':'generated_images', 'recovered.media':'recovered_media'}
mapping = {}
for row in doc.directories:
    if row.parent_id is not None:
        continue
    if row.synthetic:
        owners = {f.owner_id for f in doc.files if f.root_id == row.logical_id}
        assert len(owners) == 1, owners
        owner = next(iter(owners))
        mapping[row.logical_id] = destination / 'config' if owner in {'config','runtime.source_state'} else data
    else:
        owner = producer[row.logical_id].owner_id
        mapping[row.logical_id] = destination / 'persona-assets' if owner == 'persona.visual_identity_builtin' else data / leaves[owner]
mapping[f'profile:{profile}:paths.data_dir'] = destination / 'data'
plan = plan_restore(sealed, mode='isolated', destinations=mapping, target=None, profile_names={profile:'restored'})
restored = restore_isolated(sealed, plan, home / 'control', Event())
source.rename(source.with_name('source-home-removed'))
(home / 'restored.json').write_text(json.dumps({'profile':restored,'source_profile':profile,'data':str(data)}))
assert not blocked_attempts()
print('retired and reopened')
"""

_REOPEN = r"""
import asyncio, json, os, sqlite3, sys, threading
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice','pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home = Path.home()
receipt = json.loads((home / 'restored.json').read_text())
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(receipt['profile'], home / 'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia, current_profile_id
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture, capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission as storage
async def main():
    app = TldwCli()
    data = Path(receipt['data'])
    videos = app.generated_video_store
    for kind in ('mp4','webm'):
        status, path = videos.resolve_state('message-'+kind,'clip',extension=kind)
        assert status == 'ready' and path.read_bytes() == ('available '+kind).encode(), (status,path)
        assert data / 'recovered_media' in path.parents
    assert videos.resolve_state('expired-message','expired',extension='mp4')[0] == 'expired'
    recovered = RecoveredMedia(data / 'recovered_media')
    with sqlite3.connect(recovered.db_path) as connection:
        assets = connection.execute('SELECT asset_id FROM assets ORDER BY asset_id').fetchall()
        profiles = {row[0] for row in connection.execute('SELECT profile FROM refs')}
    assert len(assets) == 4 and profiles == {receipt['source_profile'],current_profile_id()}
    unreferenced = []
    with sqlite3.connect(recovered.db_path) as connection:
        unreferenced = connection.execute('SELECT asset_id FROM assets WHERE asset_id NOT IN (SELECT asset_id FROM refs)').fetchall()
    assert {recovered.resolve(asset)[1].read_bytes() for (asset,) in unreferenced} == {b'available temporary gallery',b'orphan video bytes'}
    assert not list(videos.root.glob('*/*.mp4'))
    selector = Path(os.environ['TLDW_CONFIG_PATH'])
    options = {'staging_parent':home, 'temporary_media':False}
    preview = preview_capture((selector,), options=options)
    assert preview.complete, (preview.issues, [(i.owner,i.status,i.path) for i in preview.items if i.status == 'unsupported'])
    from tldw_chatbook.Evals import _default_config_path
    definitions = [i for i in preview.items if i.owner == 'eval.definitions' and i.status == 'included']
    assert {i.path for i in definitions} == {_default_config_path(), data/'eval_config.yaml'}
    assert _default_config_path() != data/'eval_config.yaml'
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Backup_Recovery import bootstrap
    _, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
    witness = next(row['activation'] for row in profiles if row['selector'] == str(selector))
    activation = ActivationStore(Path(witness['store_root']))
    assert not activation.allowed(witness['generation'], 'eval.definitions')
    monitoring = asyncio.create_task(monitor_app(app))
    cancel = threading.Event()
    watchdog = asyncio.get_running_loop().call_later(30,cancel.set)
    try:
        captured = await asyncio.to_thread(capture,(selector,),preview.scope_digest,home/'second.tldw-backup.zip',options=options,cancel=cancel)
        manifest = json.loads(captured.manifest_bytes)
        members = [f for f in manifest['files'] if f['owner_id']=='recovered.media']
        assert len(members)==5
        catalog = next(f for f in members if f['relative_path']=='catalog.sqlite3')
        with sqlite3.connect(captured.root/catalog['payload']) as connection:
            assert connection.execute('SELECT asset_id FROM assets ORDER BY asset_id').fetchall()==assets
        assert captured.inventory.complete and manifest['consistency']=='coherent'
        retained = [f for f in manifest['files'] if f['owner_id']=='eval.definitions']
        assert len(retained) == 2
        for f in retained:
            assert (captured.root/f['payload']).read_bytes() == (data/'eval_config.yaml').read_bytes()
        from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
        for _ in range(500):
            if storage._pause is None and app._backup_runtime_maintenance is None:
                break
            await asyncio.sleep(.01)
        assert storage._pause is None and app._backup_runtime_maintenance is None
        videos.save('after-rebackup','clip',b'ordinary resumed video',extension='mp4')
        await asyncio.to_thread(write_archive,captured,home/'second.tldw-backup.zip',password=None,cancel=threading.Event())
        assert (home/'second.tldw-backup.zip').is_file()
        assert not activation.allowed(witness['generation'], 'eval.definitions')
    finally:
        watchdog.cancel(); cancel.set(); monitoring.cancel()
        try:
            await monitoring
        except asyncio.CancelledError:
            pass
        try:
            await app._shutdown_app_owned_lifecycles()
        except asyncio.CancelledError:
            pass
        try:
            await app.tts_service.close()
        except asyncio.CancelledError:
            pass
    assert not blocked_attempts()
asyncio.run(main())
print('retired and reopened')
"""


def test_temporary_capture_isolated_restore_reopens_without_ttl_sources(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    original = tmp_path / "original"
    original.mkdir()
    _run(original, "temporary", "complete", script=_PUBLIC)
    restored = tmp_path / "restored"
    restored.mkdir()
    _run(restored, str(original / "home"), "isolated", script=_RESTORE)
    environment = os.environ.copy()
    environment.update(
        HOME=str(restored / "home"),
        XDG_CONFIG_HOME=str(restored / "config"),
        XDG_DATA_HOME=str(restored / "data"),
        TLDW_CONFIG_PATH=str(restored / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _REOPEN],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "retired and reopened" in result.stdout


@pytest.mark.parametrize(
    "statement",
    (
        "UPDATE assets SET state='deleted'",
        "DELETE FROM refs",
        "CREATE TABLE unrelated(value TEXT)",
        "ATTACH DATABASE ':memory:' AS unrelated",
    ),
)
def test_private_alias_policy_refuses_unrelated_sql(tmp_path, statement):
    from tldw_chatbook.Backup_Recovery.recovered_media import (
        _recovered_alias_authorizer,
    )
    from tldw_chatbook.Backup_Recovery.recovered_media_schema import migrate
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    candidate = tmp_path / "candidate.sqlite3"
    with closing(sqlite3.connect(candidate)) as connection:
        migrate(connection)
    candidate.chmod(0o600)
    with open_recovery_validation(
        "recovered.media", candidate, writable=True
    ) as connection:
        restrictions = _Restrictions(connection)
        connection.set_authorizer(_recovered_alias_authorizer(restrictions.authorize))
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)


def test_private_alias_policy_refuses_trigger_owned_reference_insert(tmp_path):
    from tldw_chatbook.Backup_Recovery.recovered_media import (
        _recovered_alias_authorizer,
    )
    from tldw_chatbook.Backup_Recovery.recovered_media_schema import migrate
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    candidate = tmp_path / "candidate.sqlite3"
    with closing(sqlite3.connect(candidate)) as connection:
        migrate(connection)
        connection.execute(
            "CREATE TRIGGER imported BEFORE INSERT ON refs BEGIN INSERT INTO refs VALUES ('other','message','slug','video/mp4','asset'); END"
        )
        connection.commit()
    candidate.chmod(0o600)
    with open_recovery_validation(
        "recovered.media", candidate, writable=True
    ) as connection:
        restrictions = _Restrictions(connection)
        connection.set_authorizer(_recovered_alias_authorizer(restrictions.authorize))
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "INSERT INTO refs VALUES ('profile','message','slug','video/mp4','asset')"
            )


@pytest.mark.parametrize(
    "member",
    (
        "generated_videos/clip.mp4",
        "generated_videos/message/clip.avi",
        "generated_videos/message/.video-stage-pending.mp4",
        "generated_images/temp/unknown.bin",
    ),
)
def test_temporary_selection_refuses_non_owner_member_layout(tmp_path, member):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        DiscoverySelections,
    )

    data = tmp_path / "data" / "Ada"
    path = data / member
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unknown temporary file")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            tmp_path / "config.toml",
            "profile",
            DiscoverySelections(temporary_media=True),
        ),
    }
    owner = next(
        owner for owner in recovery_adapters() if owner.owner_id == "generation.assets"
    )
    assert (
        next(item for item in owner.discover(config) if item.path == path).status
        == "unsupported"
    )


_DELETED_PUBLIC = (
    _PUBLIC.replace(
        "    options = {'staging_parent': home, 'temporary_media': True}",
        """    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia
    deleted_source = videos.save('deleted-message','deleted',b'deleted temporary bytes',extension='mp4')
    store.append_video_message(session.id, video_metadata=VideoGenerationMetadata(name='deleted',prompt='deleted',backend='comfyui'),persist=True,message_id='deleted-message')
    owned = RecoveredMedia(videos.root.parent/'recovered_media')
    deleted_asset = owned.retain(deleted_source,profile=current_profile_id(),message='deleted-message',slug='deleted',media_type='video/mp4')
    owned.delete(deleted_asset)
    with sqlite3.connect(owned.db_path) as connection:
        original_catalog = tuple(connection.iterdump())
    options = {'staging_parent': home, 'temporary_media': True}""",
    )
    .replace(
        "assert not (videos.root.parent / 'recovered_media').exists()",
        "assert deleted_source.read_bytes()==b'deleted temporary bytes'",
    )
    .replace(
        "'SELECT asset_id,media_type FROM assets ORDER BY media_type'",
        "\"SELECT asset_id,media_type FROM assets WHERE state='ready' ORDER BY media_type\"",
    )
    .replace(
        "'SELECT profile,message,slug,media_type,asset_id FROM refs ORDER BY media_type'",
        "\"SELECT profile,message,slug,media_type,asset_id FROM refs WHERE asset_id IN (SELECT asset_id FROM assets WHERE state='ready') ORDER BY media_type\"",
    )
    .replace(
        "        assert not destination.exists()",
        """        assert 'Temporary files kept deleted: 1' in manifest['report']['lines']
        assert deleted_asset+'.payload' not in payloads
        with sqlite3.connect(captured.root/catalog['payload']) as connection:
            assert connection.execute('SELECT state FROM assets WHERE asset_id=?',(deleted_asset,)).fetchone()==('deleted',)
            assert connection.execute('SELECT references_json FROM tombstones WHERE asset_id=?',(deleted_asset,)).fetchone() is not None
        with sqlite3.connect(owned.db_path) as connection:
            assert tuple(connection.iterdump())==original_catalog
        assert not destination.exists()""",
    )
)


def test_public_temporary_capture_keeps_existing_deleted_reference_deleted(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "deleted", "complete", script=_DELETED_PUBLIC)
