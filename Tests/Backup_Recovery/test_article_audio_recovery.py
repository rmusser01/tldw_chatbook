"""The subscriptions recovery owner binds both audio trees and selector partitions."""

from dataclasses import replace
from tldw_chatbook.Subscriptions.article_listening_store import ArticleListeningStore
from pathlib import Path
import shutil
from threading import Event

import pytest
import toml

from Tests.Subscriptions.test_article_listening_store import prepare, setup_store
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.native_files import (
    create_private_directory,
    create_private_file,
)
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
from tldw_chatbook.DB.recovery_operations import recovery_adapters as database_adapters
from tldw_chatbook.Subscriptions.recovery import recovery_adapters as asset_adapters

pytest_plugins = ["Tests.Subscriptions.test_article_listening_store"]


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


def test_capture_classifies_absence_and_exact_private_stages(records):
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
        os.write(handle, b"unexpected content")
    assert (
        next(row.status for row in assets.discover(config) if row.path == lock)
        == "unsupported"
    )


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


def seed_audio(db, profile, root):
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
        audio = publish(store, session, root, session.foreground)
    else:
        # Portable recovery accepts historical recordings from a qualified producer.
        claim = store.reserve_audio(
            session.foreground.entry_id, expected_revision=session.revision
        )
        create_private_directory(root / "article_audio")
        data = wav()
        with create_private_file(root / "article_audio" / claim.file_key) as fd:
            with os.fdopen(os.dup(fd), "wb") as stream:
                stream.write(data)
        store.complete_audio(claim, metadata(data))
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


def capture_audio(records, root, config, adapter, assets, tmp_path, monkeypatch):
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

    context = config[DISCOVERY_CONTEXT_KEY]
    selector = context.config_path
    selector.write_text(
        toml.dumps({k: v for k, v in config.items() if k != DISCOVERY_CONTEXT_KEY}),
        encoding="utf8",
    )
    selector.chmod(0o600)
    config_item = StorageItem(
        "config", f"profile:{context.profile_id}:config", selector, "included", ()
    )

    def discover(*args, **kwargs):
        return classify_entries(
            (config_item, *adapter.discover(config), *assets.discover(config))
        )

    controls = tmp_path / "native"
    authority = admission_authority(controls)
    authority.register("subscriptions", (selector, records.db_path, root))
    bind_profile(controls, selector, ("subscriptions",), controls / "admission")
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: controls)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
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
            (selector,),
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
    archive = capture_audio(
        records, root, config, adapter, assets, tmp_path, monkeypatch
    )
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


def test_detached_acquisition_rebinds_only_normalized_bytes(records, tmp_path):
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


@pytest.mark.parametrize(
    "rollback", ["rejected", "created", "empty", "prior", "unrelated_briefing"]
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
    archive = capture_audio(
        records, root, config, adapter, assets, tmp_path, monkeypatch
    )

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
        capture_audio(records, root, config, adapter, assets, tmp_path, monkeypatch)
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
