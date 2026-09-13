"""Default backup uses existing local profile locators, not saved storage roots."""

import os
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, capture_service


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    root = home / ".config" / "tldw_cli" / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    result = []
    for name in ("first", "second"):
        selector = tmp_path / name / "config.toml"
        selector.parent.mkdir(mode=0o700)
        base = tmp_path / (name + "-data")
        selector.write_text(
            f'[general]\nusers_name="{name}"\n[paths]\ndata_dir="{base}"\n'
        )
        selector.chmod(0o600)
        config = {"general": {"users_name": name}, "paths": {"data_dir": str(base)}}
        data = user_data_dir(config)
        data.mkdir(parents=True, mode=0o700)
        database = database_path(config, "research_db_path")
        result.append((selector, database, data))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(result[0][0]))

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )

    startup_key = (os.getpid(), str(root))
    previous_startup = storage._startups.get(startup_key)
    try:
        for name, (_, database, _) in zip(("first", "second"), result):
            store = LocalResearchService(database)
            try:
                store.create_session(title=name, query="retained " + name)
            finally:
                store.close()
    finally:
        # A cold constructor imports config and retains this fixture's startup
        # lease. Its stores are now closed; never retire another root's startup.
        if previous_startup is None:
            startup = storage._startups.pop(startup_key, None)
            if startup is not None:
                startup.close()
    for name, (selector, _, data) in zip(("first", "second"), result):
        authority.register(name, (selector.parent, data.parent))
    for name, (selector, _, _) in zip(("first", "second"), result):
        bind_profile(root, selector, (name,), root / "admission")
    return result


def test_default_review_includes_both_known_custom_profiles(profiles):
    preview = capture_service.preview_capture((), options={})
    # These real profiles have stored research but no initialized core media DB.
    # Selecting both must retain their actual required-store omissions too.
    assert not preview.complete
    assert (
        len(
            [
                item
                for item in preview.items
                if item.owner == "db.media.primary"
                and item.status == "missing_required"
            ]
        )
        == 2
    )
    assert {item.path for item in preview.items if item.owner == "config"} == {
        row[0] for row in profiles
    }
    assert {
        item.path
        for item in preview.items
        if item.owner == "research.local" and item.status == "included"
    } == {row[1] for row in profiles}


def test_explicit_subset_stays_exact_and_known_additions_are_opt_in(profiles, tmp_path):
    explicit = capture_service.preview_capture((profiles[0][0],), options={})
    assert {item.path for item in explicit.items if item.owner == "config"} == {
        profiles[0][0]
    }
    added = tmp_path / "added.toml"
    added.write_bytes(profiles[0][0].read_bytes())
    added.chmod(0o600)
    preview = capture_service.preview_capture(
        (added,), options={}, include_known_profiles=True
    )
    assert {item.path for item in preview.items if item.owner == "config"} == {
        added,
        *(row[0] for row in profiles),
    }


def test_recorded_missing_profile_is_retained_as_missing(profiles):
    missing = profiles[1][0]
    missing.unlink()
    preview = capture_service.preview_capture((), options={})
    assert any(
        item.owner == "config"
        and item.path == missing
        and item.status == "missing_required"
        for item in preview.items
    )
    assert not preview.complete


def test_existing_canonical_config_is_added_and_lexically_deduplicated(
    profiles, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.profile_paths import default_config_path

    canonical = default_config_path()
    canonical.write_bytes(profiles[0][0].read_bytes())
    canonical.chmod(0o600)
    selected = capture_service._selectors(())
    assert set(selected) == {canonical, *(row[0] for row in profiles)}
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(canonical.parent / "." / canonical.name))
    assert capture_service._selectors(()) == (canonical, profiles[0][0], profiles[1][0])


@pytest.mark.parametrize("added", [False, True])
def test_new_known_profile_requires_new_review_before_capture(
    profiles, tmp_path, added
):
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    extra = tmp_path / "added.toml"
    extra.write_bytes(profiles[0][0].read_bytes())
    extra.chmod(0o600)
    selected = (extra,) if added else ()
    options = {"allow_partial": True, "staging_parent": tmp_path}
    preview = capture_service.preview_capture(
        selected, options=options, include_known_profiles=added
    )
    later = tmp_path / "later" / "config.toml"
    later.parent.mkdir(mode=0o700)
    later.write_bytes(profiles[1][0].read_bytes())
    later.chmod(0o600)
    root = bootstrap.default_bootstrap_root()
    authority = admission_authority(root)
    authority.register("later", (later.parent,))
    bind_profile(root, later, ("later", "second"), root / "admission")
    destination = tmp_path / "changed.tldw-backup.zip"
    with pytest.raises(CaptureReviewRequired) as error:
        capture_service.capture(
            selected,
            preview.scope_digest,
            destination,
            options=options,
            cancel=Event(),
            include_known_profiles=added,
        )
    assert error.value.issues == ("scope_changed",)
    assert not destination.exists()
    assert not list(tmp_path.glob("capture-*"))
    assert not bootstrap._records(root)[0]


def test_unknown_canonical_data_is_still_reported(profiles):
    from tldw_chatbook.Backup_Recovery.profile_paths import default_base_data_dir

    orphan = default_base_data_dir() / "unlocated"
    orphan.mkdir(parents=True, mode=0o700)
    (orphan / "important.db").write_bytes(b"unknown historical bytes")
    preview = capture_service.preview_capture((), options={})
    assert not preview.complete
    assert any(
        item.owner == "unknown" and item.path == orphan and item.status == "unsupported"
        for item in preview.items
    )
    assert not any(item.path == orphan / "important.db" for item in preview.items)


def test_malformed_known_record_is_not_silently_ignored(profiles):
    root = bootstrap.default_bootstrap_root()
    record = root / ("profile-" + bootstrap._key(str(profiles[1][0])) + ".json")
    record.write_bytes(b"{}")
    with pytest.raises(ValueError, match="record_version"):
        capture_service.preview_capture((), options={})


def test_absent_canonical_config_is_not_fabricated(profiles):
    from tldw_chatbook.Backup_Recovery.profile_paths import default_config_path

    canonical = default_config_path()
    assert not canonical.exists()
    assert capture_service._selectors(()) == tuple(row[0] for row in profiles)
    assert not canonical.exists()
