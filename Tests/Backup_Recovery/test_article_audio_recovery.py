"""The subscriptions recovery owner binds both audio trees and selector partitions."""

from dataclasses import replace
from tldw_chatbook.Subscriptions.article_listening_store import ArticleListeningStore
from pathlib import Path
import shutil
import sys
from threading import Event

import pytest
import toml

from Tests.Subscriptions.test_article_listening_store import prepare, records as _records_fixture, setup_store
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.native_files import (
    create_private_directory,
    create_private_file,
)
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
from tldw_chatbook.DB.recovery_operations import recovery_adapters as database_adapters
from tldw_chatbook.Subscriptions.recovery import recovery_adapters as asset_adapters

records = _records_fixture


def owners(records):
    m, binding, store, voice = setup_store(records)
    selector = records.db_path.with_name("config.toml")
    context = DiscoveryContext(selector, binding[0].profile)
    base = records.db_path.parent / "data"
    root = base / "shared"
    create_private_directory(base)
    create_private_directory(root)
    config = {
        DISCOVERY_CONTEXT_KEY: context,
        "general": {"users_name": "shared"},
        "paths": {"data_dir": str(base)},
        "database": {"subscriptions_db_path": str(records.db_path)},
    }
    adapter = next(a for a in database_adapters() if a.owner_id == "db.subscriptions")
    return adapter, asset_adapters()[0], config, root, m, binding, store, voice


def test_capture_classifies_absence_and_exact_private_stages(records, monkeypatch):
    from tldw_chatbook.Utils import windows_job

    def unexpected_observation(*args):
        pytest.fail("Archive discovery queried a native process")

    monkeypatch.setattr(windows_job, "predecessor_retired", unexpected_observation)
    _, assets, config, root, _, _, _, _ = owners(records)
    briefing = root / "briefing_audio"
    article = root / "article_audio"
    lock = records.db_path.with_name(records.db_path.name + ".article-listening.lock")
    inventory = {row.path: row for row in assets.discover(config)}
    assert set(inventory) == {briefing, article, lock}
    assert inventory[briefing].status == inventory[article].status == "unused"
    assert inventory[briefing].logical_id != inventory[article].logical_id
    assert inventory[lock].status == "intentionally_excluded" and not lock.exists()
    create_private_directory(article)
    key = "a" * 32 + ".wav"
    for name in ("." + key + ".stage", key + ".stage", key.upper()):
        with create_private_file(article / name):
            pass
    inventory = {row.path.name: row for row in assets.discover(config)}
    assert inventory["." + key + ".stage"].status == "intentionally_excluded"
    assert (
        inventory[key + ".stage"].status
        == inventory[key.upper()].status
        == "unsupported"
    )
    from tldw_chatbook.Utils.platform_files import os

    with create_private_file(lock) as handle:
        from tldw_chatbook.Subscriptions.article_playback_receipt import (
            encode_predecessor,
        )

        os.write(handle, encode_predecessor(42, 987654321))
    assert (
        next(row.status for row in assets.discover(config) if row.path == lock)
        == "intentionally_excluded"
    )
    with lock.open("r+b") as handle:
        handle.write(b"ALP2")
    assert (
        next(row.status for row in assets.discover(config) if row.path == lock)
        == "unsupported"
    )


@pytest.mark.parametrize("status", ["pending", "ready", "missing", "pending_delete"])
def test_installed_db_only_validation_enforces_audio_timestamps(records, status):
    from contextlib import closing
    import sqlite3
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    adapter, _, config, _, m, binding, store, voice = owners(records)
    session = store.replace_session(
        prepare(store, binding[0], voice), expected_revision=None
    )
    job = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    if status != "pending":
        store.complete_audio(job, m.AudioMetadata("b" * 64, 48000, 12, 12012, 48092))
    if status == "missing":
        store.mark_audio_missing(job.generation)
    if status == "pending_delete":
        records.delete_article(job.item_id)
    item = adapter.discover(config)[0]
    assert item.status == "included"
    records.close()
    assert validate_candidate(adapter, records.db_path, Event(), migrate=False) == ()
    for field in ("last_used", "created_at"):
        with closing(sqlite3.connect(records.db_path)) as c, c:
            original = c.execute(f"SELECT {field} FROM article_audio").fetchone()[0]
            assert (original is None) == (status == "pending_delete")
            c.execute(
                f"UPDATE article_audio SET {field}=?",
                (1 if original is None else None,),
            )
        assert validate_candidate(adapter, records.db_path, Event(), migrate=False) == (
            "invalid_listening_state",
        )
        assert adapter.validate(records.db_path) == ("invalid_listening_state",)
        with closing(sqlite3.connect(records.db_path)) as c, c:
            c.execute(f"UPDATE article_audio SET {field}=?", (original,))


def test_mixed_current_stamp_validates_listening_in_discovery_restore_and_staging(
    records,
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    adapter, _, config, _, m, binding, store, voice = owners(records)
    session = store.replace_session(
        prepare(store, binding[0], voice), expected_revision=None
    )
    job = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    store.complete_audio(job, m.AudioMetadata("a" * 64, 48000, 12, 12012, 48092))
    item = adapter.discover(config)[0]
    with records.transaction() as c:
        c.execute("INSERT INTO schema_version VALUES(1)")
        c.execute("UPDATE article_audio SET sha256='invalid'")
    records.close()
    assert validate_candidate(adapter, records.db_path, Event(), migrate=False) == (
        "invalid_listening_state",
    )
    with _preview_reads():
        assert adapter.validate_restore(item, records.db_path) == (
            "invalid_listening_state",
        )
    assert adapter.discover(config)[0].status == "unsupported"


def test_live_discovery_qualifies_catalog_once_and_reports_connection_failure(
    records, monkeypatch
):
    from tldw_chatbook.DB import private_sqlite

    adapter, _, config, root, m, binding, _, _ = owners(records)
    _, audio = seed_audio(records, binding[0].profile, root)
    records.close()
    calls = []
    original = type(adapter)._validate_catalog

    def check(self, connection):
        calls.append(connection)
        return original(self, connection)

    monkeypatch.setattr(type(adapter), "_validate_catalog", check)
    (item,) = adapter.discover(config)
    assert item.status == "included"
    assert any(dep.endswith(audio.wav.path.name) for dep in item.dependencies)
    assert len(calls) == 1

    def unavailable(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(private_sqlite, "connect_private_sqlite", unavailable)
    assert adapter.discover(config)[0].status == "unavailable"


def test_candidate_rebind_keeps_bookmark_and_refuses_foreign_audio_partition(records):
    adapter, _, config, _, m, binding, store, voice = owners(records)
    authority = binding[0]
    session = store.replace_session(
        prepare(store, authority, voice), expected_revision=None
    )
    job = store.reserve_audio(
        session.foreground.entry_id, expected_revision=session.revision
    )
    session = store.complete_audio(
        job, m.AudioMetadata("b" * 64, 48000, 12, 12012, 48092)
    )
    item = adapter.discover(config)[0]
    selector = records.db_path.with_name("relocated.toml")
    config_id = f"profile:{authority.profile}:config"
    records.close()
    candidate = records.db_path.with_name("private-candidate.db")
    shutil.copyfile(records.db_path, candidate)
    with _preview_reads():
        adapter.rebind_listening_profile(item, candidate, {config_id: selector})
    binding[0] = replace(authority, profile=m.listening_profile_key(selector))
    from contextlib import closing
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    with closing(SubscriptionsDB(candidate)) as imported:
        target_store = m.ArticleListeningStore(
            imported, current_authority=lambda: binding[0]
        )
        moved = target_store.load_session()
        assert (
            moved.foreground == session.foreground
            and moved.revision == session.revision
        )
        target_store.clear(expected_revision=moved.revision)
        with imported.transaction() as cursor:
            cursor.execute(
                "DELETE FROM listening_session WHERE profile=?", (binding[0].profile,)
            )
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="listening_profile_collision"),
    ):
        adapter.rebind_listening_profile(item, candidate, {config_id: selector})


@pytest.mark.parametrize(
    "declaration_kind", ["member", "stage", "lock", "article_root", "changed_topology"]
)
def test_only_installed_subscription_root_absence_can_select_retirements(
    records, declaration_kind
):
    from types import SimpleNamespace
    from tldw_chatbook.Backup_Recovery.archive_models import ProducerItem
    from tldw_chatbook.Backup_Recovery.data_groups import ResolvedGroups
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_groups import selected_absent_target_ids

    _, owner, config, root, _, binding, _, _ = owners(records)
    selector = config[DISCOVERY_CONTEXT_KEY].config_path
    selector.write_text(
        toml.dumps({k: v for k, v in config.items() if k != DISCOVERY_CONTEXT_KEY})
    )
    selector.chmod(0o600)
    audio = root / "article_audio"
    create_private_directory(audio)
    member = audio / ("a" * 32 + ".wav")
    with create_private_file(member):
        pass
    observed = {item.path: item for item in owner.discover(config)}
    prefix = f"profile:{binding[0].profile}:"
    config_id = prefix + "config"
    root_id = observed[audio].logical_id
    claim = {
        "member": observed[member].logical_id,
        "stage": root_id + "-excluded-" + "b" * 64,
        "lock": prefix + "subscriptions.assets:listening_lock",
        "article_root": root_id,
        "changed_topology": root_id,
    }[declaration_kind]
    declaration = ProducerItem(
        logical_id=claim,
        owner_id=owner.owner_id,
        status="unused",
        dependencies=(config_id,),
    )
    config_row = StorageItem("config", config_id, selector, "included", ())
    doc = SimpleNamespace(
        producer_inventory=(
            ProducerItem(
                logical_id=config_id,
                owner_id="config",
                status="included",
                dependencies=(),
            ),
            declaration,
        ),
        files=(),
        directories=(),
    )
    scope = ResolvedGroups(("research",), ("research",), (), (claim,), ())
    target = Inventory((config_row, *observed.values()), True, "", ())
    if declaration_kind == "changed_topology":
        with create_private_file(audio / ("b" * 32 + ".wav")):
            pass
    with _preview_reads():
        if declaration_kind == "article_root":
            assert selected_absent_target_ids(
                doc, scope, target, ((config_id, selector),), ()
            ) == {root_id, observed[member].logical_id}
        else:
            with pytest.raises(ValueError, match="selected_absence_mapping_required"):
                selected_absent_target_ids(
                    doc, scope, target, ((config_id, selector),), ()
                )
    assert member.exists()


def seed_audio(db, profile, root, *, frames=8):
    import os
    from Tests.Subscriptions.test_article_audio_assets import metadata, publish, wav
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets
    from tldw_chatbook.Subscriptions.article_listening_models import (
        FrozenVoice,
        ListeningAuthority,
        ListeningCheckpoint,
    )
    from tldw_chatbook.TTS.controlled_wav_profile import WavBookmark

    authority = ListeningAuthority(profile, "local", 7)
    store = ArticleListeningStore(db, current_authority=lambda: authority)
    voice = FrozenVoice("openai", "tts-1", "alloy")
    brief = prepare(store, authority, voice, brief=True)
    foreground = prepare(store, authority, voice, item=2).foreground
    session = store.replace_session(
        replace(brief, foreground=foreground), expected_revision=None
    )
    if os.name == "nt":
        audio = publish(store, session, root, session.foreground, frames=frames)
    else:
        # Portable recovery accepts historical recordings from a qualified producer.
        claim = store.reserve_audio(
            session.foreground.entry_id, expected_revision=session.revision
        )
        if not (root / "article_audio").exists():
            create_private_directory(root / "article_audio")
        data = wav(frames)
        with create_private_file(root / "article_audio" / claim.file_key) as fd:
            with os.fdopen(os.dup(fd), "wb") as stream:
                stream.write(data)
        store.complete_audio(claim, metadata(data, frames))
        audio = ArticleAudioAssets(store, profile_data_root=root).read_verified(
            claim.generation
        )
    session = store.load_session()
    session = store.attach_audio(
        session.queue[0].entry_id, audio.generation, expected_revision=session.revision
    )
    points = tuple(
        (entry.entry_id, WavBookmark(audio.wav.sha256, 48000, frame))
        for entry, frame in ((session.foreground, 6), (session.queue[0], 2))
    )
    session = store.checkpoint(
        ListeningCheckpoint(
            authority,
            session.incarnation,
            session.playback_generation,
            0,
            True,
            "paused",
            points,
        ),
        expected_revision=session.revision,
    )
    return session, audio


def test_reviewed_digest_refuses_a_replaced_member_before_read(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.storage_admission import _digest_recovery_file
    from tldw_chatbook.Utils.platform_files import os

    candidate = tmp_path / "candidate.wav"
    with create_private_file(candidate) as fd:
        os.write(fd, b"reviewed")
        reviewed = os.fstat(fd)
    candidate.rename(tmp_path / "reviewed.wav")
    with create_private_file(candidate) as fd:
        replacement = os.fstat(fd)
    read = os.read

    def guard(fd, size):
        opened = os.fstat(fd)
        assert (opened.st_dev, opened.st_ino) != (
            replacement.st_dev,
            replacement.st_ino,
        )
        return read(fd, size)

    monkeypatch.setattr(os, "read", guard)
    with _preview_reads(), pytest.raises(ValueError, match="recovery_file_changed"):
        _digest_recovery_file(
            "subscriptions.assets", candidate, max_bytes=64, expected_stat=reviewed
        )


def selected_audio_inventory(selectors):
    """Keep the three installed owners from actual multi-selector discovery."""
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries, discover

    rows = discover(selectors).items
    return classify_entries(
        tuple(
            row
            for row in rows
            if row.owner in {"config", "db.subscriptions", "subscriptions.assets"}
            or row.owner == "sqlite.transient"
            and any(dep.endswith(":db.subscriptions") for dep in row.dependencies)
        )
    )


def capture_audio(root, config, adapter, assets, tmp_path, monkeypatch, *, peers=()):
    """Run the installed finite owners under real native admission and package the result."""
    from tldw_chatbook.Backup_Recovery import bootstrap, capture, owner_registry
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path

    configs = (config, *peers)
    selectors = tuple(c[DISCOVERY_CONTEXT_KEY].config_path for c in configs)
    for selected, selector in zip(configs, selectors, strict=False):
        selector.write_text(
            toml.dumps(
                {k: v for k, v in selected.items() if k != DISCOVERY_CONTEXT_KEY}
            ),
            encoding="utf8",
        )
        selector.chmod(0o600)

    def discover(*args, **kwargs):
        if peers:
            return selected_audio_inventory(selectors)
        profile = config[DISCOVERY_CONTEXT_KEY].profile_id
        return classify_entries(
            (
                StorageItem(
                    "config", f"profile:{profile}:config", selectors[0], "included", ()
                ),
                *adapter.discover(config),
                *assets.discover(config),
            )
        )

    controls = tmp_path / "native"
    authority = admission_authority(controls)
    authority.register(
        "subscriptions",
        tuple(
            dict.fromkeys(
                (
                    *selectors,
                    *(database_path(c, "subscriptions_db_path") for c in configs),
                    root.parent if peers else root,
                )
            )
        ),
    )
    with authority.maintenance(("subscriptions", "bootstrap.unbound"), 30) as session:
        for selector in selectors:
            bind_profile(
                controls,
                selector,
                ("subscriptions",),
                controls / "admission",
                session=session,
            )
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: controls)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selectors[0]))
    monkeypatch.setattr(capture, "discover", discover)
    declarations = (config_adapter(), adapter, assets)
    monkeypatch.setattr(
        owner_registry, "_adapters", {owner.owner_id: owner for owner in declarations}
    )
    dest = tmp_path / "source.tldw-backup.zip"
    preview = discover()
    assert preview.complete, preview.issues
    with authority.maintenance(("subscriptions", "bootstrap.unbound"), 4) as session:
        result = capture._capture_under_maintenance(
            session,
            selectors,
            preview.scope_digest,
            dest,
            options={"staging_parent": tmp_path},
            cancel=Event(),
        )
    write_archive(result, dest, password=None, cancel=Event())
    return acquire(dest, tmp_path / "verified", ArchiveLimits(), None, Event())


@pytest.mark.parametrize("asset_present", [True, False])
def test_packaged_audio_relocates_to_target_selector_with_original_removed(
    records, tmp_path, monkeypatch, asset_present
):
    from contextlib import closing
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.article_audio_assets import (
        ArticleAudioAssets,
        ArticleAudioError,
    )

    adapter, assets, config, root, m, binding, _, _ = owners(records)
    session, audio = seed_audio(records, binding[0].profile, root)
    key = audio.wav.path.name
    create_private_directory(root / "briefing_audio")
    brief = root / "briefing_audio" / key
    brief.write_bytes(b"independent briefing recording")
    brief.chmod(0o600)
    with records.transaction() as c:
        c.execute(
            "INSERT INTO briefing_scripts(id,briefing_id,preset_name,roster_snapshot_json) VALUES(1,1,'saved','[]')"
        )
        c.execute(
            "INSERT INTO briefing_audio(script_id,voice_snapshot_json,status,file_path) VALUES(1,'[]','complete',?)",
            (str(brief),),
        )
    records.close()
    if not asset_present:
        audio.wav.path.unlink()
        audio.wav.path.parent.rmdir()
    archive = capture_audio(root, config, adapter, assets, tmp_path, monkeypatch)
    doc = verify_sealed(archive)
    peers = [p for p in doc.files if p.owner_id == assets.owner_id]
    assert len(peers) == 1 + asset_present
    assert all(p.relative_path == key for p in peers)
    assert len({p.root_id for p in peers}) == len(peers)
    root.rename(root.with_name("retired-source"))
    original_selector = config[DISCOVERY_CONTEXT_KEY].config_path
    original_selector.rename(original_selector.with_name("retired-selector"))
    records.db_path.rename(records.db_path.with_name("retired-source-db"))
    destination_parent = tmp_path / "restore-destinations"
    create_private_directory(destination_parent)
    base = destination_parent / "relocated"
    from tldw_chatbook.Backup_Recovery import bootstrap

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "new-native"
    )
    plan = resolve_destinations(
        archive,
        mode="isolated",
        profile_bases={binding[0].profile: base},
        external_destinations={},
        target=None,
        profile_names={binding[0].profile: "Other"},
    )
    mapped = dict(plan.restore)
    prefix = f"profile:{binding[0].profile}:"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(mapped[prefix + "config"]))
    restore_isolated(archive, plan, tmp_path / "restore-control", Event())
    assert (
        not original_selector.exists()
        and not records.db_path.exists()
        and not root.exists()
    )
    target = replace(
        binding[0], profile=m.listening_profile_key(mapped[prefix + "config"])
    )
    assert target.profile != binding[0].profile
    with closing(SubscriptionsDB(mapped[prefix + "db.subscriptions"])) as db:
        restored = m.ArticleListeningStore(db, current_authority=lambda: target)
        state = restored.load_session()
        assert (state.foreground, state.queue, state.revision, state.incarnation) == (
            session.foreground,
            session.queue,
            session.revision,
            session.incarnation,
        )
        path = Path(db.get_briefing_audio(1)["file_path"])
        with db.transaction() as cursor:
            assert (
                cursor.execute("SELECT custody_witness FROM article_audio").fetchone()[
                    0
                ]
                is None
            )
        assert path != brief and path.read_bytes() == b"independent briefing recording"
        actual = ArticleAudioAssets(restored, profile_data_root=path.parent.parent)
        if asset_present:
            assert actual.read_verified(audio.generation).wav.sha256 == audio.wav.sha256
        else:
            with pytest.raises(ArticleAudioError):
                actual.read_verified(audio.generation)
            assert (
                restored.load_session().foreground.bookmark
                == session.foreground.bookmark
            )


@pytest.mark.parametrize(
    ("separate_database", "target_layout"),
    [(False, "split"), (True, "split"), (False, "common"), (False, "replace")],
    ids=["shared-db", "distinct-dbs", "shared-target", "shared-replacement"],
)
def test_shared_article_directory_archives_only_verified_partition_members(
    records,
    tmp_path,
    monkeypatch,
    separate_database,
    target_layout,
    helper_resource_root,
):
    from contextlib import closing
    from tldw_chatbook.Backup_Recovery import bootstrap, owner_registry
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets

    adapter, assets, config, _, m, binding, _, _ = owners(records)
    config.pop("general")
    config.pop("database")
    root = user_data_dir(config)
    create_private_directory(root)
    other = tmp_path / "peer.toml"
    peer = {
        **config,
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(other, m.listening_profile_key(other)),
    }
    if separate_database:
        peer["database"] = {"subscriptions_db_path": str(tmp_path / "second.db")}
    database = database_path(config, "subscriptions_db_path")
    other_database = database_path(peer, "subscriptions_db_path")
    records.close()
    for path in {database, other_database}:
        shutil.copyfile(records.db_path, path)
        path.chmod(0o600)
    expected = {}
    configs = (config, peer)
    for i, selected in enumerate(configs):
        context = selected[DISCOVERY_CONTEXT_KEY]
        context.config_path.write_text(
            toml.dumps(
                {k: v for k, v in selected.items() if k != DISCOVERY_CONTEXT_KEY}
            )
        )
        context.config_path.chmod(0o600)
        with closing(
            SubscriptionsDB(database_path(selected, "subscriptions_db_path"))
        ) as db:
            expected[context.profile_id] = seed_audio(
                db, context.profile_id, root, frames=8 + i
            )
    selectors = tuple(c[DISCOVERY_CONTEXT_KEY].config_path for c in configs)
    declarations = (config_adapter(), adapter, assets)
    monkeypatch.setattr(
        owner_registry, "_adapters", {o.owner_id: o for o in declarations}
    )
    unselected = selected_audio_inventory(selectors[:1])
    assert not unselected.complete
    sibling_audio = expected[peer[DISCOVERY_CONTEXT_KEY].profile_id][1]
    assert {r.status for r in unselected.items if r.path == sibling_audio.wav.path} == {
        "unsupported"
    }
    unknown = root / "article_audio" / ("f" * 32 + ".wav")
    with create_private_file(unknown):
        pass
    unexplained = selected_audio_inventory(selectors)
    assert not unexplained.complete and {
        r.status for r in unexplained.items if r.path == unknown
    } == {"unsupported"}
    unknown.unlink()
    if separate_database:
        import sqlite3

        first = expected[binding[0].profile][1].wav.path
        with closing(sqlite3.connect(other_database)) as db_copy, db_copy:
            db_copy.execute("UPDATE article_audio SET file_key=?", (first.name,))
        ambiguous = selected_audio_inventory(selectors)
        assert not ambiguous.complete and {
            r.status for r in ambiguous.items if r.path == first
        } == {"unsupported"}
        with closing(sqlite3.connect(other_database)) as db_copy, db_copy:
            db_copy.execute(
                "UPDATE article_audio SET file_key=?", (sibling_audio.wav.path.name,)
            )
    archive = capture_audio(
        root, config, adapter, assets, tmp_path, monkeypatch, peers=(peer,)
    )
    doc = verify_sealed(archive)
    assert {
        (row.logical_id.split(":")[1], row.relative_path)
        for row in doc.files
        if row.owner_id == assets.owner_id
    } == {(profile, audio.wav.path.name) for profile, (_, audio) in expected.items()}
    destination = tmp_path / "restored"
    create_private_directory(destination)
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "new-native"
    )
    if target_layout == "replace":
        assert_shared_replacement_preserves_unqualified_retirement(
            archive,
            records.db_path,
            destination,
            expected,
            tmp_path,
            monkeypatch,
            helper_resource_root,
        )
        return
    if target_layout == "common":
        from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
        from tldw_chatbook.Subscriptions.recovery import shared_article_roots

        root_rows = [
            r for r in doc.directories if r.logical_id.endswith(":article_audio")
        ]
        bad = root_rows[0]
        changed = bad.model_copy(
            update={
                "metadata": bad.metadata.model_copy(
                    update={"mtime_ns": bad.metadata.mtime_ns + 1}
                )
            }
        )
        with pytest.raises(ValueError, match="unverified_article_root_group"):
            shared_article_roots(
                doc.model_copy(
                    update={
                        "directories": tuple(
                            changed if r == bad else r for r in doc.directories
                        )
                    }
                )
            )
        member = next(p for p in doc.files if p.root_id == bad.logical_id)
        db_id = f"profile:{member.logical_id.split(':')[1]}:db.subscriptions"
        declarations = tuple(
            r.model_copy(
                update={
                    "dependencies": tuple(
                        d for d in r.dependencies if d != member.logical_id
                    )
                }
            )
            if r.logical_id == db_id
            else r
            for r in doc.producer_inventory
        )
        with pytest.raises(ValueError, match="unverified_article_root_group"):
            shared_article_roots(
                doc.model_copy(update={"producer_inventory": declarations})
            )

        data = destination / "common-data"
        user = data / "default_user"
        locations = {}
        for row in doc.files:
            if row.owner_id == "db.subscriptions":
                locations[row.root_id] = user
            elif row.owner_id == "subscriptions.assets":
                locations[row.root_id] = user / "article_audio"
            elif row.owner_id == "config":
                profile = row.logical_id.split(":")[1]
                locations[row.root_id] = destination / profile / "config"
                locations[f"profile:{profile}:paths.data_dir"] = data
        plan = plan_restore(
            archive,
            mode="isolated",
            destinations=locations,
            target=None,
            profile_names=dict.fromkeys(expected, "default_user"),
        )
    else:
        plan = resolve_destinations(
            archive,
            mode="isolated",
            profile_bases={k: destination / k for k in expected},
            external_destinations={},
            target=None,
            profile_names={k: k for k in expected},
        )
    mapping = dict(plan.restore)
    assert len({mapping[f"profile:{k}:db.subscriptions"] for k in expected}) == (
        2 if separate_database else 1
    )
    monkeypatch.setenv(
        "TLDW_CONFIG_PATH", str(mapping[f"profile:{binding[0].profile}:config"])
    )
    restore_isolated(archive, plan, tmp_path / "restore-control", Event())
    for profile, (session, audio) in expected.items():
        prefix = f"profile:{profile}:"
        target = replace(
            binding[0], profile=m.listening_profile_key(mapping[prefix + "config"])
        )
        with closing(SubscriptionsDB(mapping[prefix + "db.subscriptions"])) as restored:
            store = ArticleListeningStore(
                restored, current_authority=lambda target=target: target
            )
            state = store.load_session()
            assert (state.foreground, state.queue) == (
                session.foreground,
                session.queue,
            )
            directory = mapping[prefix + "subscriptions.assets:article_audio"]
            assert {p.name for p in directory.iterdir()} == (
                {a.wav.path.name for _, a in expected.values()}
                if target_layout == "common"
                else {audio.wav.path.name}
            )
            assert (
                ArticleAudioAssets(store, profile_data_root=directory.parent)
                .read_verified(audio.generation)
                .wav.sha256
                == audio.wav.sha256
            )


def assert_shared_replacement_preserves_unqualified_retirement(
    archive, source_db, live, expected, tmp_path, monkeypatch, helper_resource_root
):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        credentials,
        crypto,
        later_rollback,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path
    from tldw_chatbook.Backup_Recovery.restore_plan import retained_config_names
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    credential_store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    data = live / "data"
    create_private_directory(data)
    root = data / "default_user"
    create_private_directory(root)
    config = {"paths": {"data_dir": str(data)}}
    database = database_path(config, "subscriptions_db_path")
    shutil.copyfile(source_db, database)
    database.chmod(0o600)
    selectors = {profile: live / (profile + ".toml") for profile in expected}
    controls = bootstrap.default_bootstrap_root()
    authority = admission_authority(controls)
    authority.register("local", (live,))
    with authority.maintenance(("local", "bootstrap.unbound"), 30) as session:
        for selector in selectors.values():
            selector.write_text(toml.dumps(config), encoding="utf8")
            selector.chmod(0o600)
            bind_profile(
                controls, selector, ("local",), controls / "admission", session=session
            )
    paths = tuple(selectors.values())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(paths[0]))
    with _preview_reads():
        observed = selected_audio_inventory(paths)
    assert observed.complete, observed.issues
    plan = resolve_destinations(
        archive,
        mode="replace",
        profile_bases={},
        external_destinations={},
        target=observed,
        profile_names={},
        target_configs=selectors,
        data_groups=("research",),
    )
    names = set(_capture_names(authority, observed)) | set(retained_config_names(plan))
    with authority.maintenance(tuple(sorted(names)), 30) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "incoming", Event(), session=session
        )
    control = tmp_path / "replacement"
    operation = replacement.replace(
        plan, candidate, control_root=control, rollback_password=b"old", cancel=Event()
    )
    with _preview_reads():
        fresh = selected_audio_inventory(paths)
    assert fresh.complete, fresh.issues
    audio = root / "article_audio"
    originals = {p: p.read_bytes() for p in (*paths, database, *audio.iterdir())}
    assert {p.name for p in audio.iterdir()} == {
        a.wav.path.name for _, a in expected.values()
    }
    before = bootstrap._control_records(controls)
    with pytest.raises(ValueError, match="^local_snapshot_created_members_changed$"):
        later_rollback.preview_rollback(
            operation,
            control_root=control,
            old_password=b"old",
            target=fresh,
            cancel=Event(),
        )
    assert {p: p.read_bytes() for p in originals} == originals
    assert bootstrap._control_records(controls) == before


def test_shared_subscription_image_relocates_all_selected_partitions(
    records, tmp_path, monkeypatch
):
    import sqlite3
    from contextlib import closing
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    adapter, assets, config, root, m, binding, store, voice = owners(records)
    first = binding[0]
    other_selector = tmp_path / "second-config.toml"
    second = replace(first, profile=m.listening_profile_key(other_selector))
    foreign = replace(
        first, profile=m.listening_profile_key(tmp_path / "unselected.toml")
    )
    expected = {}
    for number, authority in enumerate((first, second, foreign), start=1):
        binding[0] = authority
        store = ArticleListeningStore(records, current_authority=lambda: binding[0])
        expected[authority.profile] = store.replace_session(
            prepare(store, authority, voice, item=number), expected_revision=None
        )
        with records.transaction() as c:
            c.execute(
                "INSERT INTO article_audio(profile,file_key,status,custody_witness) VALUES(?,?,'pending_delete',?)",
                (authority.profile, str(number) * 32 + ".wav", str(number) * 64),
            )
    records.close()
    peer = dict(config)
    peer[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(other_selector, second.profile)
    peer["general"] = {"users_name": "second"}
    create_private_directory(root.parent / "second")
    archive = capture_audio(
        root, config, adapter, assets, tmp_path, monkeypatch, peers=(peer,)
    )
    destination = tmp_path / "destinations"
    create_private_directory(destination)
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "restore-native"
    )
    profiles = {first.profile: "One", second.profile: "Two"}
    plan = resolve_destinations(
        archive,
        mode="isolated",
        profile_bases={key: destination / label for key, label in profiles.items()},
        external_destinations={},
        target=None,
        profile_names=profiles,
    )
    mapping = dict(plan.restore)
    databases = {mapping[f"profile:{key}:db.subscriptions"] for key in profiles}
    assert len(databases) == 1
    monkeypatch.setenv(
        "TLDW_CONFIG_PATH", str(mapping[f"profile:{first.profile}:config"])
    )
    restore_isolated(archive, plan, tmp_path / "shared-control", Event())
    (database,) = databases
    keys = {
        key: m.listening_profile_key(mapping[f"profile:{key}:config"])
        for key in profiles
    }
    assert len(set(keys.values()) | set(profiles)) == 4
    with closing(SubscriptionsDB(database)) as restored:
        for source, target in (*keys.items(), (foreign.profile, foreign.profile)):
            binding[0] = replace(first, profile=target)
            selected = ArticleListeningStore(
                restored, current_authority=lambda: binding[0]
            )
            assert selected.load_session().foreground == expected[source].foreground
    with closing(sqlite3.connect(database)) as db:
        assert dict(
            db.execute("SELECT profile,custody_witness FROM article_audio")
        ) == {key: None for key in (*keys.values(), foreign.profile)}


@pytest.mark.parametrize(
    "empty_first", [False, True], ids=["occupied-first", "empty-first"]
)
def test_shared_db_only_overlap_rebinds_the_original_partition(
    records, tmp_path, empty_first
):
    import sqlite3
    from contextlib import closing
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    adapter, _, config, root, m, binding, _, voice = owners(records)
    first = binding[0]
    middle_selector = tmp_path / "middle.toml"
    final_selector = tmp_path / "final.toml"
    middle = replace(first, profile=m.listening_profile_key(middle_selector))
    final = replace(first, profile=m.listening_profile_key(final_selector))
    unselected = replace(
        first, profile=m.listening_profile_key(tmp_path / "other.toml")
    )
    session, audio = seed_audio(records, first.profile, root)
    assert (session.foreground.bookmark.frame, session.queue[0].bookmark.frame) == (
        6,
        2,
    )
    audio.wav.path.unlink()
    other_store = ArticleListeningStore(records, current_authority=lambda: unselected)
    other_session = other_store.replace_session(
        prepare(other_store, unselected, voice, item=3), expected_revision=None
    )
    with records.transaction() as db:
        db.execute(
            "INSERT INTO article_audio(profile,file_key,status,custody_witness) VALUES(?,?,'pending_delete',?)",
            (unselected.profile, "a" * 32 + ".wav", "1" * 64),
        )
        db.execute(
            "UPDATE article_audio SET custody_witness=? WHERE generation=?",
            ("2" * 64, audio.generation),
        )
    peer = {
        **config,
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(middle_selector, middle.profile),
    }
    inputs = [
        replace(adapter.discover(data)[0], shared_group="shared:subscriptions")
        for data in (config, peer)
    ]
    if empty_first:
        inputs.reverse()
    mapping = {
        f"profile:{first.profile}:config": middle_selector,
        f"profile:{middle.profile}:config": final_selector,
    }
    records.close()
    candidate = tmp_path / "db-only.sqlite"
    shutil.copyfile(records.db_path, candidate)
    candidate.chmod(0o600)

    def audio_rows():
        with closing(sqlite3.connect(candidate)) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute("SELECT * FROM article_audio ORDER BY generation")
            ]

    original = audio_rows()
    assert {row["profile"] for row in original} == {first.profile, unselected.profile}
    with _preview_reads():
        adapter.rebind_listening_profile(
            inputs[0], candidate, mapping, aliases=(inputs[1],)
        )
    assert audio_rows() == [
        {
            **row,
            "profile": middle.profile
            if row["profile"] == first.profile
            else unselected.profile,
            "custody_witness": None,
        }
        for row in original
    ]
    with closing(SubscriptionsDB(candidate)) as db:

        def load(authority):
            return ArticleListeningStore(
                db, current_authority=lambda: authority
            ).load_session()

        assert load(middle) == replace(session, authority=middle)
        assert load(unselected) == other_session
        assert load(first) is None and load(final) is None


def test_detached_acquisition_rebinds_only_normalized_bytes(
    records, tmp_path, monkeypatch
):
    import sqlite3
    import zipfile
    from contextlib import closing
    from Tests.Backup_Recovery.test_detached_layout import _base_profile
    from tldw_chatbook.Backup_Recovery.detached_layout import produce_detached_archive
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf
    from tldw_chatbook.Subscriptions.article_listening_models import (
        listening_profile_key,
    )

    source, root, original = _base_profile(tmp_path, core_owners=())
    acquired_selector = source / "home" / ".config" / "tldw_cli" / "config.toml"
    historic_selector = original / "Data" / acquired_selector.relative_to(source)
    old_key, acquired_key = map(
        listening_profile_key, (historic_selector, acquired_selector)
    )
    state, audio = seed_audio(records, old_key, root)
    briefing = root / "briefing_audio" / audio.wav.path.name
    create_private_directory(briefing.parent)
    briefing.write_bytes(b"complete briefing")
    briefing.chmod(0o600)
    historic_path = original / "Data" / briefing.relative_to(source)
    with records.transaction() as cursor:
        cursor.execute(
            "INSERT INTO briefing_scripts(id,briefing_id,preset_name,roster_snapshot_json) VALUES(1,1,'saved','[]')"
        )
        cursor.execute(
            "INSERT INTO briefing_audio(script_id,voice_snapshot_json,status,file_path) VALUES(1,'[]','complete',?)",
            (str(historic_path),),
        )
    records.close()
    database = root / database_leaf("subscriptions_db_path")
    shutil.copyfile(records.db_path, database)
    database.chmod(0o600)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    from tldw_chatbook.TTS.TTS_Generation import TTSService
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
    from tldw_chatbook.TTS.audio_player import SimpleAudioPlayer
    from tldw_chatbook.TTS.controlled_wav_playback import ControlledWavPlayback

    def unexpected_audio(*args, **kwargs):
        pytest.fail("Passive recovery invoked a provider or player")

    for component, methods in (
        (TTSService, ("__init__", "synthesize", "synthesize_exact")),
        (TTSAdapterRegistry, ("__init__", "acquire")),
        (SimpleAudioPlayer, ("__init__", "play", "play_prepared")),
        (ControlledWavPlayback, ("__init__", "prepare", "start")),
    ):
        for method in methods:
            monkeypatch.setattr(component, method, unexpected_audio)
    result = produce_detached_archive(source, tmp_path / "normalized", Event())
    assert all(path.read_bytes() == data for path, data in before.items())
    with zipfile.ZipFile(result.archive) as archive:
        import json

        doc = json.loads(archive.read("manifest.json"))
        payload = next(
            row for row in doc["files"] if row["owner_id"] == "db.subscriptions"
        )
        candidate = tmp_path / "checked.db"
        candidate.write_bytes(archive.read(payload["payload"]))
    with closing(sqlite3.connect(candidate)) as connection:
        assert connection.execute(
            "SELECT profile,revision FROM listening_session"
        ).fetchall() == [(acquired_key, state.revision)]
        assert connection.execute(
            "SELECT profile,generation,custody_witness FROM article_audio"
        ).fetchall() == [(acquired_key, audio.generation, None)]
        assert connection.execute(
            "SELECT file_path FROM briefing_audio"
        ).fetchone() == (str(briefing),)

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets
    from tldw_chatbook.Subscriptions.article_listening_models import ListeningAuthority

    source.rename(tmp_path / "removed-acquired-profile")
    assert not source.exists() and not original.exists()
    archive = acquire(
        result.archive, tmp_path / "sealed", ArchiveLimits(), None, Event()
    )
    destinations = tmp_path / "destinations"
    create_private_directory(destinations)
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "new-native"
    )
    plan = resolve_destinations(
        archive,
        mode="isolated",
        profile_bases={acquired_key: destinations / "relocated"},
        external_destinations={},
        target=None,
        profile_names={acquired_key: "Final"},
    )
    mapped = dict(plan.restore)
    prefix = f"profile:{acquired_key}:"
    final = ListeningAuthority(
        listening_profile_key(mapped[prefix + "config"]), "local", 7
    )
    assert len({old_key, acquired_key, final.profile}) == 3
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(mapped[prefix + "config"]))
    restore_isolated(archive, plan, tmp_path / "final-control", Event())
    with closing(SubscriptionsDB(mapped[prefix + "db.subscriptions"])) as db:
        store = ArticleListeningStore(db, current_authority=lambda: final)
        restored = store.load_session()
        assert (restored.foreground, restored.queue) == (state.foreground, state.queue)
        actual_briefing = Path(db.get_briefing_audio(1)["file_path"])
        assert actual_briefing.read_bytes() == b"complete briefing"
        actual = ArticleAudioAssets(
            store, profile_data_root=actual_briefing.parent.parent
        )
        assert actual.read_verified(audio.generation).wav.sha256 == audio.wav.sha256


@pytest.mark.parametrize(
    "rollback",
    [
        "rejected",
        "created",
        "empty",
        "prior",
        "unrelated_briefing",
        "preview_topology",
        "held_topology",
    ],
)
def test_research_replacement_rebinds_and_rolls_back_owned_audio(
    records, tmp_path, monkeypatch, helper_resource_root, rollback
):
    import sqlite3
    from contextlib import closing
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        credentials,
        crypto,
        later_rollback,
        publication,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.inventory import (
        classify_entries,
        _sqlite_sidecars,
    )
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.restore_plan import retained_config_names
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    credential_store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    adapter, assets, config, root, m, binding, _, _ = owners(records)
    live = tmp_path / "target"
    create_private_directory(live)
    create_private_directory(live / "data")
    local = live / "data" / "Local"
    create_private_directory(local)
    selector = live / "config.toml"
    db_path = live / records.db_path.name
    records.close()
    shutil.copyfile(records.db_path, db_path)
    db_path.chmod(0o600)
    # Both selectors use explicit DB names outside their independent data trees.
    source_state, source_audio = seed_audio(records, binding[0].profile, root)
    if rollback not in {"prior", "unrelated_briefing"}:
        create_private_directory(root / "briefing_audio")
    if rollback == "empty":
        source_audio.wav.path.unlink()
    records.close()
    archive = capture_audio(root, config, adapter, assets, tmp_path, monkeypatch)

    local_profile = m.listening_profile_key(selector)
    local_config = {
        "general": {"users_name": "Local", "retained": "keep"},
        "paths": {"data_dir": str(local.parent)},
        "database": {"subscriptions_db_path": str(db_path)},
    }
    selector.write_text(toml.dumps(local_config), encoding="utf8")
    selector.chmod(0o600)
    prior_config = selector.read_bytes()
    local_config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(selector, local_profile)
    controls = tmp_path / "target-native"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: controls)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    authority = admission_authority(controls)
    authority.register("local", (live,))
    bind_profile(controls, selector, ("local",), controls / "admission")
    if rollback != "prior":
        create_private_directory(local / "briefing_audio")
    if rollback == "unrelated_briefing":
        recording = local / "briefing_audio" / "retained.wav"
        recording.write_bytes(b"unrelated retained briefing")
        recording.chmod(0o600)

    def inventory():
        with _preview_reads():
            declared = (
                *config_adapter().discover(local_config),
                *adapter.discover(local_config),
                *assets.discover(local_config),
            )
            rows = _sqlite_sidecars(declared, (adapter,))
            result = classify_entries((*declared, *rows))
        assert result.complete, result.issues
        return result

    prior = None
    if rollback in {"rejected", "prior"}:
        with closing(SubscriptionsDB(db_path)) as old:
            prior, previous_audio = seed_audio(old, local_profile, local)
            with old.transaction() as cursor:
                prior_witness = cursor.execute(
                    "SELECT custody_witness FROM article_audio"
                ).fetchone()[0]
        old_payload = previous_audio.wav.path.read_bytes()
        assert previous_audio.wav.path.name != source_audio.wav.path.name
    observed = inventory()

    def review():
        return resolve_destinations(
            archive,
            mode="replace",
            profile_bases={},
            external_destinations={},
            target=observed,
            profile_names={},
            target_configs={binding[0].profile: selector},
            data_groups=("research",),
        )

    if rollback == "unrelated_briefing":
        with pytest.raises(ValueError, match="selected_absence_mapping_required"):
            review()
        assert recording.read_bytes() == b"unrelated retained briefing"
        assert selector.read_bytes() == prior_config
        return
    plan = review()
    assert len(plan.retained_configs) == 1
    names = set(_capture_names(authority, observed)) | set(retained_config_names(plan))
    with authority.maintenance(tuple(sorted(names)), 30) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "incoming", Event(), session=session
        )
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    control = tmp_path / "replacements"
    if rollback == "rejected":
        validated = publication._validate_installed

        def inject(*args, **kwargs):
            validated(*args, **kwargs)
            raise ValueError("test_failure_after_verified_publication")

        monkeypatch.setattr(publication, "_validate_installed", inject)
        with pytest.raises(ValueError, match="replacement_rolled_back"):
            replacement.replace(
                plan,
                candidate,
                control_root=control,
                rollback_password=b"old",
                cancel=Event(),
            )
    else:
        operation = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"old",
            cancel=Event(),
        )
        with closing(sqlite3.connect(db_path)) as db:
            assert db.execute(
                "SELECT profile,revision FROM listening_session"
            ).fetchone() == (local_profile, source_state.revision)
        new_root = local / "article_audio"
        fresh = inventory()
        if rollback in {"preview_topology", "held_topology"}:
            from tldw_chatbook.Utils.platform_files import os as native_os

            stage = new_root / ("." + "e" * 32 + ".wav.stage")
            added = []
            read = native_os.read
            inspect = later_rollback._created_destination_target

            def guard_content(fd, count):
                info = native_os.fstat(fd)
                assert (info.st_dev, info.st_ino) not in added, "read unreviewed stage"
                return read(fd, count)

            def add_stage(*args, session=None, **kwargs):
                if (session is not None) == (rollback == "held_topology") and not added:
                    with create_private_file(stage) as fd:
                        info = native_os.fstat(fd)
                        added.append((info.st_dev, info.st_ino))
                return inspect(*args, session=session, **kwargs)

            monkeypatch.setattr(native_os, "read", guard_content)
            monkeypatch.setattr(
                later_rollback, "_created_destination_target", add_stage
            )

            def preview():
                return later_rollback.preview_rollback(
                    operation,
                    control_root=control,
                    old_password=b"old",
                    target=fresh,
                    cancel=Event(),
                )

            with pytest.raises(
                ValueError, match="local_snapshot_created_members_changed"
            ):
                if rollback == "preview_topology":
                    preview()
                else:
                    approved = preview()
                    later_rollback.execute_rollback(
                        operation,
                        control_root=control,
                        old_password=b"old",
                        new_password=b"new",
                        cancel=Event(),
                        approved_plan=approved,
                    )
            assert len(added) == 1 and stage.is_file()
            return
        p = later_rollback.preview_rollback(
            operation,
            control_root=control,
            old_password=b"old",
            target=fresh,
            cancel=Event(),
        )
        if prior is None:
            assert new_root in dict(p.retire).values()
        change = new_root / source_audio.wav.path.name
        if rollback in {"created", "prior"}:
            initial = change.read_bytes()
            change.write_bytes(initial[:-1] + b"z")
            with pytest.raises(
                ValueError, match="local_snapshot_created_members_changed"
            ):
                later_rollback.preview_rollback(
                    operation,
                    control_root=control,
                    old_password=b"old",
                    target=fresh,
                    cancel=Event(),
                )
            change.write_bytes(initial)
        p = later_rollback.preview_rollback(
            operation,
            control_root=control,
            old_password=b"old",
            target=inventory(),
            cancel=Event(),
        )
        if rollback == "prior":
            from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
            from tldw_chatbook.Backup_Recovery.recovery_copies import _locked_copy

            evidence = tmp_path / "prior-snapshot"
            create_private_directory(evidence)
            with _locked_copy(control, operation, exclusive=False) as (
                entry,
                _,
                proof,
                _,
            ):
                saved = later_rollback._acquire(entry, proof, evidence, b"old", Event())
                local_image = next(
                    row
                    for row in verify_sealed(saved).files
                    if row.owner_id == "db.subscriptions"
                )
        later_rollback.execute_rollback(
            operation,
            control_root=control,
            old_password=b"old",
            new_password=b"new",
            cancel=Event(),
            approved_plan=p,
        )
        if prior is None:
            assert not new_root.exists()
        if rollback == "prior":
            import hashlib

            assert (
                hashlib.sha256(db_path.read_bytes()).hexdigest() == local_image.sha256
            )
    if prior is not None:
        with closing(sqlite3.connect(db_path)) as db:
            assert db.execute(
                "SELECT profile,revision FROM listening_session"
            ).fetchone() == (local_profile, prior.revision)
            assert db.execute(
                "SELECT custody_witness FROM article_audio"
            ).fetchone() == (prior_witness,)
        assert previous_audio.wav.path.read_bytes() == old_payload
        assert not (local / "article_audio" / source_audio.wav.path.name).exists()
    assert (local / "briefing_audio").is_dir() == (rollback != "prior")
    assert selector.read_bytes() == prior_config


@pytest.mark.parametrize("version", [2, 3])
def test_reviewed_rebind_preserves_archived_predecessor_image(tmp_path, version):
    from Tests.DB.historical_subscriptions import historical_database
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Subscriptions.article_listening_models import (
        listening_profile_key,
    )

    source = tmp_path / "archived.db"
    historical_database(source, version)
    prefix = "profile:" + listening_profile_key(tmp_path / "old.toml") + ":"
    item = StorageItem(
        "db.subscriptions",
        prefix + "db.subscriptions",
        source,
        "included",
        (prefix + "config",),
    )
    owner = next(a for a in database_adapters() if a.owner_id == item.owner)
    before = source.read_bytes()
    with _preview_reads():
        owner.rebind_listening_profile(
            item, source, {prefix + "config": tmp_path / "new.toml"}
        )
        assert owner.validate_restore_dependencies(item, source, {}, topology={}) == ()
    assert source.read_bytes() == before


def test_capture_refuses_declared_corrupt_payload(records, tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired

    adapter, assets, config, root, _, binding, _, _ = owners(records)
    _, audio = seed_audio(records, binding[0].profile, root)
    records.close()
    previous = audio.wav.path.read_bytes()
    audio.wav.path.write_bytes(previous[:-1] + b"x")
    with pytest.raises(CaptureReviewRequired) as error:
        capture_audio(root, config, adapter, assets, tmp_path, monkeypatch)
    assert error.value.issues == ("dependency_unavailable",)
    assert not (tmp_path / "source.tldw-backup.zip").exists()


@pytest.mark.parametrize("move", [False, True])
def test_incoming_candidate_clears_local_custody_for_every_partition(
    records, tmp_path, move
):
    from contextlib import closing
    import sqlite3
    from tldw_chatbook.Subscriptions.article_listening_models import (
        listening_profile_key,
    )

    adapter, _, config, _, _, binding, _, _ = owners(records)
    source = binding[0].profile
    foreign = listening_profile_key(tmp_path / "unrelated.toml")
    with records.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO article_audio(profile,file_key,status,custody_witness) VALUES(?,?,'pending_delete',?)",
            [
                (source, "a" * 32 + ".wav", "1" * 64),
                (foreign, "b" * 32 + ".wav", "2" * 64),
            ],
        )
    item = adapter.discover(config)[0]
    target = tmp_path / ("other.toml" if move else "config.toml")
    records.close()
    candidate = tmp_path / "copy.db"
    shutil.copyfile(records.db_path, candidate)
    source_bytes = records.db_path.read_bytes()
    with _preview_reads():
        adapter.rebind_listening_profile(
            item, candidate, {f"profile:{source}:config": target}
        )
    with closing(sqlite3.connect(candidate)) as db:
        assert db.execute(
            "SELECT profile,custody_witness FROM article_audio ORDER BY generation"
        ).fetchall() == [(listening_profile_key(target), None), (foreign, None)]
    assert records.db_path.read_bytes() == source_bytes


@pytest.mark.parametrize("layout", ["absent", "explicit_missing", "orphan"])
def test_detached_listening_absence_preserves_exact_ownership(tmp_path, layout):
    import json
    import zipfile
    from Tests.Backup_Recovery.test_detached_layout import _base_profile
    from tldw_chatbook.Backup_Recovery.detached_layout import produce_detached_archive
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf
    from tldw_chatbook.Subscriptions.article_listening_models import (
        listening_profile_key,
    )
    from tldw_chatbook.Subscriptions.recovery import article_id

    source, root, origin = _base_profile(tmp_path, core_owners=())
    selector = source / "home" / ".config" / "tldw_cli" / "config.toml"
    work = tmp_path / "detached"
    if layout == "explicit_missing":
        with selector.open("a", encoding="utf-8") as stream:
            stream.write(
                toml.dumps(
                    {
                        "database": {
                            "subscriptions_db_path": str(
                                origin / "Data" / "missing.sqlite"
                            )
                        }
                    }
                )
            )
        with pytest.raises(ValueError, match="detached_required_database_missing"):
            produce_detached_archive(source, work, Event())
    elif layout == "orphan":
        sidecar = root / (database_leaf("subscriptions_db_path") + "-wal")
        sidecar.write_bytes(b"unclaimed")
        with pytest.raises(ValueError, match="detached_sqlite_orphan_sidecar"):
            produce_detached_archive(source, work, Event())
        assert sidecar.read_bytes() == b"unclaimed"
    else:
        result = produce_detached_archive(source, work, Event())
        with zipfile.ZipFile(result.archive) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        profile = listening_profile_key(selector)
        absent = {
            row["logical_id"]
            for row in manifest["producer_inventory"]
            if row["owner_id"] == "subscriptions.assets" and row["status"] == "unused"
        }
        assert absent == {
            f"profile:{profile}:subscriptions.assets",
            article_id(profile),
        }
        assert not any(
            row["owner_id"] == "subscriptions.assets" for row in manifest["files"]
        )
        assert not (root / "article_audio").exists()


@pytest.mark.parametrize("version", [2, 3])
def test_detached_packaging_preserves_predecessor_catalog(tmp_path, version):
    import json
    import sqlite3
    import zipfile
    from contextlib import closing
    from Tests.Backup_Recovery.test_detached_layout import _base_profile
    from Tests.DB.historical_subscriptions import historical_database, catalog
    from tldw_chatbook.Backup_Recovery.detached_layout import produce_detached_archive
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf

    source, root, _ = _base_profile(tmp_path, core_owners=())
    db = root / database_leaf("subscriptions_db_path")
    historical_database(db, version)
    with closing(sqlite3.connect(db)) as previous, previous:
        previous.execute(
            "INSERT INTO subscriptions(id,name,type,source) VALUES(1,'archived','rss','local')"
        )
        expected = catalog(previous)
    original = db.read_bytes()
    result = produce_detached_archive(source, tmp_path / "work", Event())
    assert db.read_bytes() == original
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        record = next(
            row for row in manifest["files"] if row["owner_id"] == "db.subscriptions"
        )
        restored = tmp_path / "verified.sqlite"
        restored.write_bytes(archive.read(record["payload"]))
    with closing(sqlite3.connect(restored)) as copied:
        assert catalog(copied) == expected
        assert copied.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            version,
        )
        assert copied.execute("SELECT name FROM subscriptions").fetchone() == (
            "archived",
        )


@pytest.mark.skipif(sys.platform != "win32", reason="qualified Windows predecessor")
def test_replacement_rechecks_installed_live_companion_before_destination_mutation(
    records, monkeypatch
):
    import subprocess
    import time
    import _winapi
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RestorePlan,
        _fingerprint,
        _paths,
        recheck_targets,
    )
    from tldw_chatbook.TTS._controlled_wav_windows import _creation_time
    from tldw_chatbook.Utils import windows_job

    adapter, assets, config, root, _, _, store, _ = owners(records)
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        with store.session_lease(root) as owner:
            owner.seal_predecessor(child.pid, _creation_time(int(child._handle)))
        companion = records.db_path.with_name(
            records.db_path.name + ".article-listening.lock"
        )
        original = companion.read_bytes(), records.db_path.read_bytes()
        db_item = adapter.discover(config)[0]
        inventory = Inventory((db_item, *assets.discover(config)), True, "", ())
        plan = RestorePlan(
            "digest",
            "replace",
            ((db_item.logical_id, records.db_path),),
            (),
            (),
            "",
            target=inventory,
        )
        plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), inventory))
        with pytest.raises(ValueError, match="previous_player_unconfirmed"):
            recheck_targets(plan)
        assert (companion.read_bytes(), records.db_path.read_bytes()) == original
        child.stdin.close()
        assert child.wait(10) == 0
        assert (
            _winapi.WaitForSingleObject(int(child._handle), 0) == _winapi.WAIT_OBJECT_0
        )

        def uncertain(*args):
            raise PermissionError

        with monkeypatch.context() as patch:
            patch.setattr(windows_job, "predecessor_retired", uncertain)
            with pytest.raises(ValueError, match="previous_player_unconfirmed"):
                recheck_targets(plan)
        assert (companion.read_bytes(), records.db_path.read_bytes()) == original
        observed = time.monotonic()
        recheck_targets(plan)
        assert time.monotonic() - observed >= 0.25
        assert (companion.read_bytes(), records.db_path.read_bytes()) == original
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(10)
