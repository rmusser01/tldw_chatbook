"""Destination-profile aliases retain recovered-media source/deletion evidence."""

import hashlib
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia


def profile(selector):
    return hashlib.sha256(str(selector).encode()).hexdigest()[:24]


def declared(store, selector):
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile(selector))}
    entries = store.recovery_adapter().discover(config)
    assert not any(
        i.status in {"unavailable", "unsupported", "missing_required"} for i in entries
    )
    return next(i for i in entries if i.path == store.db_path)


def rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


@pytest.fixture
def source(tmp_path, local_root):
    selector = tmp_path / "original.toml"
    selector.write_text("")
    payload = tmp_path / "original.png"
    payload.write_bytes(b"retained exact recovered-media bytes")
    store = RecoveredMedia(tmp_path / "original")
    identity = {
        "profile": profile(selector),
        "message": "message",
        "slug": "same",
        "media_type": "image/png",
    }
    asset = store.retain(payload, **identity)
    return store, selector, identity, asset, payload


@pytest.mark.parametrize("deleted", [False, True])
def test_relocation_preserves_source_and_aliases_only_captured_profile(
    source, tmp_path, deleted
):
    store, selector, identity, asset, payload = source
    unrelated = identity | {"profile": "unrelated", "message": "other"}
    store.add_reference(asset, **unrelated)
    if deleted:
        store.delete(asset)
    original = rows(store.db_path)
    target = tmp_path / "relocated"
    shutil.copytree(store.root, target)
    destination = tmp_path / "new-config.toml"
    item = declared(store, selector)
    mapping = {f"profile:{profile(selector)}:config": destination}
    adapter = store.recovery_adapter()
    adapter.relocate_restore(item, target / "catalog.sqlite3", mapping)
    restored = RecoveredMedia(target)
    expected = "deleted" if deleted else "ready"
    assert (
        restored.resolve_reference(**(identity | {"profile": profile(destination)}))[0]
        == expected
    )
    assert restored.resolve_reference(**identity)[0] == expected
    assert restored.resolve_reference(**unrelated)[0] == expected
    assert (
        restored.resolve_reference(**(unrelated | {"profile": profile(destination)}))[0]
        == "unknown"
    )
    assert restored.resolve(asset)[0] == expected
    assert rows(store.db_path) == original
    assert adapter.validate(restored.db_path) == ()
    if deleted:
        assert not restored._path(asset).exists()
    else:
        assert restored.resolve(asset)[1].read_bytes() == payload.read_bytes()
    first = rows(restored.db_path)
    adapter.relocate_restore(item, restored.db_path, mapping)
    assert rows(restored.db_path) == first

    # The next backup discovers the new profile; relocate only those new aliases.
    second_item = declared(restored, destination)
    second_root = tmp_path / "twice-relocated"
    shutil.copytree(restored.root, second_root)
    second_selector = tmp_path / "second-config.toml"
    adapter.relocate_restore(
        second_item,
        second_root / "catalog.sqlite3",
        {
            f"profile:{profile(destination)}:config": second_selector,
        },
    )
    twice = RecoveredMedia(second_root)
    for selected in (selector, destination, second_selector):
        assert (
            twice.resolve_reference(**(identity | {"profile": profile(selected)}))[0]
            == expected
        )
    assert (
        twice.resolve_reference(**(unrelated | {"profile": profile(second_selector)}))[
            0
        ]
        == "unknown"
    )
    assert adapter.validate(twice.db_path) == ()
    declared(twice, second_selector)


@pytest.mark.parametrize("deleted", [False, True])
def test_destination_collision_rolls_back_all_aliases_and_tombstones(
    source, tmp_path, deleted
):
    store, selector, identity, asset, payload = source
    store.add_reference(asset, **(identity | {"message": "a-before-conflict"}))
    destination = tmp_path / "new-config.toml"
    conflict = store.retain(payload, **(identity | {"profile": profile(destination)}))
    assert conflict != asset  # Same bytes are not the same catalog identity.
    if deleted:
        store.delete(asset)
    before = rows(store.db_path)
    with pytest.raises(ValueError, match="recovered_reference_collision"):
        store.recovery_adapter().relocate_restore(
            declared(store, selector),
            store.db_path,
            {
                f"profile:{profile(selector)}:config": destination,
            },
        )
    assert rows(store.db_path) == before
    assert store.recovery_adapter().validate(store.db_path) == ()


@pytest.mark.parametrize("kind", ["missing", "other-profile", "missing-dependency"])
def test_selected_profile_config_is_required_before_catalog_mutation(
    source, tmp_path, kind
):
    store, selector, _identity, _asset, _payload = source
    item = declared(store, selector)
    mapping = {}
    if kind == "other-profile":
        mapping["profile:other:config"] = tmp_path / "new.toml"
    if kind == "missing-dependency":
        item = replace(item, dependencies=())
        mapping[f"profile:{profile(selector)}:config"] = tmp_path / "new.toml"
    before = rows(store.db_path)
    with pytest.raises(ValueError, match="recovered_profile_mapping_required"):
        store.recovery_adapter().relocate_restore(item, store.db_path, mapping)
    assert rows(store.db_path) == before


def test_payload_role_needs_no_config_alias_mapping(source, tmp_path):
    store, selector, _identity, asset, payload = source
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile(selector))}
    item = next(
        i
        for i in store.recovery_adapter().discover(config)
        if i.path == store._path(asset)
    )
    candidate = tmp_path / "payload-candidate"
    shutil.copyfile(item.path, candidate)
    store.recovery_adapter().relocate_restore(item, candidate, {})
    assert candidate.read_bytes() == payload.read_bytes()


@pytest.mark.parametrize("deleted", [False, True])
def test_relocated_catalog_native_backup_preserves_aliases(
    source, tmp_path, monkeypatch, deleted
):
    from Tests.Backup_Recovery.test_core_owners import application_authority

    store, selector, _identity, asset, _payload = source
    if deleted:
        store.delete(asset)
    destination = tmp_path / "new.toml"
    adapter = store.recovery_adapter()
    adapter.relocate_restore(
        declared(store, selector),
        store.db_path,
        {
            f"profile:{profile(selector)}:config": destination,
        },
    )
    item = declared(store, destination)
    control = tmp_path / "native-control"
    control.mkdir()
    authority = application_authority(control, store.db_path, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "catalog.sqlite3"
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((store.db_path,), stage),
    ):
        adapter.capture(item, candidate, Event())
        assert adapter.validate(candidate) == ()
    assert rows(candidate) == rows(store.db_path)


@pytest.mark.parametrize("corruption", ["tombstone", "pending"])
def test_invalid_catalog_is_refused_before_alias_mutation(source, tmp_path, corruption):
    store, selector, _identity, asset, _payload = source
    item = declared(store, selector)
    store.delete(asset)
    with closing(sqlite3.connect(store.db_path)) as connection, connection:
        if corruption == "tombstone":
            connection.execute("UPDATE tombstones SET references_json='[]'")
        else:
            connection.execute("INSERT INTO operations VALUES (?, 'delete')", (asset,))
    before = rows(store.db_path)
    with pytest.raises(
        ValueError, match="invalid_recovered_tombstone|recovered_operation_pending"
    ):
        store.recovery_adapter().relocate_restore(
            item,
            store.db_path,
            {
                f"profile:{profile(selector)}:config": tmp_path / "new.toml",
            },
        )
    assert rows(store.db_path) == before


_APP_PROBE = r"""
import asyncio, hashlib, json, os, shutil, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia,current_profile_id
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY,DiscoveryContext
from tldw_chatbook.Utils.paths import get_user_data_dir
async def main():
 app=TldwCli();data=get_user_data_dir();selector=Path(os.environ['TLDW_CONFIG_PATH'])
 source=data/'recovered_media';old=RecoveredMedia(source)
 payload=Path.home()/'original.png';payload.write_bytes(b'original retained image');payload.chmod(0o600)
 identity=dict(profile=current_profile_id(),message='message',slug='same',media_type='image/png')
 asset=old.retain(payload,**identity)
 if sys.argv[1]=='deleted':old.delete(asset)
 config={'paths':{'data_dir':str(data.parent)},'general':{'users_name':data.name},DISCOVERY_CONTEXT_KEY:DiscoveryContext(selector,identity['profile'])}
 adapter=old.recovery_adapter();entries=adapter.discover(config)
 item=next(i for i in entries if i.path==old.db_path)
 relocated=data/'probe-relocated';shutil.copytree(source,relocated)
 new_selector=relocated/'config.toml'
 mapping={f"profile:{identity['profile']}:config":new_selector,item.logical_id:relocated/'catalog.sqlite3'}
 adapter.relocate_restore(item,relocated/'catalog.sqlite3',mapping)
 assert adapter.validate(relocated/'catalog.sqlite3')==()
 restored=RecoveredMedia(relocated)
 destination_profile=hashlib.sha256(str(new_selector).encode()).hexdigest()[:24]
 observed=restored.resolve_reference(**(identity|{'profile':destination_profile}))[0]
 original=restored.resolve_reference(**identity)[0]
 assert observed==('deleted' if sys.argv[1]=='deleted' else 'ready'),observed
 assert original==('deleted' if sys.argv[1]=='deleted' else 'ready'),original
 assert restored.resolve(asset)[0]==original
 assert observed==original
 print('actual app-created catalog retains source and resolves selected destination profile')
 try:await app._shutdown_app_owned_lifecycles()
 except asyncio.CancelledError:pass
 try:await app.tts_service.close()
 except asyncio.CancelledError:pass
 assert not blocked_attempts()
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("state", ["ready", "deleted"])
def test_actual_app_created_catalog_resolves_destination_profile(tmp_path, state):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, state, "relocated", script=_APP_PROBE)
