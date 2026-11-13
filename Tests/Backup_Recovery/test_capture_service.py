"""Public backup uses the reviewed policy and an actual native capture session."""

from threading import Event

import pytest

from Tests.Backup_Recovery.test_operational_owners import STORES
from tldw_chatbook.Backup_Recovery import bootstrap, owner_registry
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture


def _populate_required_dependencies(preview):
    """Build actual source stores needed by the installed dependency declarations."""
    import sqlite3
    from contextlib import closing
    from importlib import import_module

    from tldw_chatbook.Chat.console_semantic_revision import (
        install_semantic_coordination,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.TTS.profile_schema import open_profile_store

    # Callers include fresh child processes, each its own composition root;
    # the chat rows seeded below need the fail-closed port (idempotent).
    install_semantic_coordination()

    # An owner may also declare sidecars (the TTS lease lock, for example).
    # Populate its primary logical record instead of the last path encountered.
    paths = {
        item.owner: item.path
        for item in preview.items
        if item.path is not None and item.logical_id.endswith(":" + item.owner)
    }
    for owner, cls in (
        ("db.chachanotes.primary", CharactersRAGDB),
        ("db.media.primary", MediaDatabase),
        ("db.prompts.primary", PromptsDatabase),
    ):
        path = paths[owner]
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        store = cls(path, "backup-fixture")
        if owner == "db.chachanotes.primary":
            character = store.add_character_card({"name": "Backup fixture"})
            conversation = store.add_conversation(
                {"title": "Retained media", "character_id": character}
            )
            message = store.add_message(
                {
                    "conversation_id": conversation,
                    "sender": "user",
                    "content": "Retained media",
                }
            )
        store.close()
    for name in ("workspaces", "agent_runs", "file_notes"):
        module, symbol, _, owner, _ = STORES[name]
        cls = getattr(import_module("tldw_chatbook." + module), symbol)
        if name == "file_notes":
            # Initialize the actual installed SQLite schema without invoking
            # unrelated process-global directory/config fixture enrollment.
            store = cls(":memory:")
            with closing(sqlite3.connect(paths[owner])) as destination:
                store._get_connection().backup(destination)
        else:
            store = cls(paths[owner])
        store.close()
    with closing(open_profile_store(paths["tts.profile_store"])):
        pass
    paths["skills"].mkdir(mode=0o700)
    return message


@pytest.mark.parametrize(
    ("damage_payload", "include_external"),
    [(False, False), (True, False), (False, True)],
)
def test_public_config_capture_and_publication_without_profile_rebinding(
    tmp_path, monkeypatch, damage_payload, include_external
):
    import json

    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive

    source = tmp_path / "source" / "config.toml"
    source.parent.mkdir(mode=0o700)
    source.write_text(
        f'[general]\nusers_name="fixture"\n[paths]\ndata_dir="{tmp_path / "data"}"\n[api_settings.openai]\napi_key="synthetic-secret"\n'
    )
    source.chmod(0o600)
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    options = {"allow_partial": True, "staging_parent": tmp_path}
    if include_external:
        external = tmp_path / "external"
        external.mkdir()
        (external / "empty").mkdir()
        (external / "document.txt").write_text("explicitly selected external content")
        options["external_roots"] = (external,)
    preview = preview_capture((source,), options=options)
    message = _populate_required_dependencies(preview)
    audio_history = next(
        item.path for item in preview.items if item.owner == "audio.history"
    )
    audio_history.write_text(
        '{"items":[{"id":1,"text":"retained speech","content_base64":"YXVkaW8="}]}'
    )
    audio_history.chmod(0o600)
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    media = tmp_path / "source-media.bin"
    media.write_bytes(b"saved opaque media bytes")
    recovered_root = next(
        item.path for item in preview.items if item.owner == "recovered.media"
    )
    profile = next(
        item.logical_id.split(":")[1]
        for item in preview.items
        if item.owner == "config"
    )
    store = RecoveredMedia(recovered_root)
    asset_id = store.retain(
        media, profile=profile, message=message, slug="saved", media_type="image/png"
    )
    preview = preview_capture((source,), options=options)
    assert not any(
        item.owner == "audio.history"
        and item.logical_id.endswith(":participant_pending")
        for item in preview.items
    )
    assert not set(preview.issues) - {
        "unsupported",
        "unavailable",
        "missing_required",
        "unsupported_owner",
    }, preview.issues
    destination = tmp_path / "saved.tldw-backup.zip"
    original_bytes = {
        item.path: item.path.read_bytes()
        for item in preview.items
        if item.status == "included" and item.path is not None
    }
    if damage_payload:
        from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
        from tldw_chatbook.Backup_Recovery.recovered_media import _RecoveredAdapter

        original_capture = _RecoveredAdapter.capture

        def corrupt_staged_payload(self, item, destination, cancel):
            original_capture(self, item, destination, cancel)
            if item.path.name.endswith(".payload"):
                destination.write_bytes(b"damaged staged copy")

        monkeypatch.setattr(_RecoveredAdapter, "capture", corrupt_staged_payload)
        with pytest.raises(CaptureReviewRequired) as error:
            capture(
                (source,),
                preview.scope_digest,
                destination,
                options=options,
                cancel=Event(),
            )
        assert error.value.issues == ("recovered_payload_missing",)
        assert not destination.exists()
        assert not list(tmp_path.glob("capture-*"))
        assert all(
            path.read_bytes() == content for path, content in original_bytes.items()
        )
        return
    result = capture(
        (source,), preview.scope_digest, destination, options=options, cancel=Event()
    )
    assert bootstrap._records(root) == ([], [])
    manifest = json.loads(result.manifest_bytes)
    producers = {item["logical_id"]: item for item in manifest["producer_inventory"]}
    assert set(producers) == {
        item["logical_id"]
        for item in (
            *manifest["files"],
            *manifest["directories"],
            *manifest["exclusions"],
        )
    }
    for file in manifest["files"]:
        assert producers[file["logical_id"]]["owner_id"] == file["owner_id"]
        assert file["metadata"]["version"] == 1
    audio_payload = next(
        file for file in manifest["files"] if file["owner_id"] == "audio.history"
    )
    assert (
        result.root / audio_payload["payload"]
    ).read_bytes() == audio_history.read_bytes()
    assert manifest["consistency"] == "partial"
    payload = next(
        file
        for file in manifest["files"]
        if file["relative_path"] == asset_id + ".payload"
    )
    assert (result.root / payload["payload"]).read_bytes() == media.read_bytes()
    if include_external:
        external_payload = next(
            file for file in manifest["files"] if file["owner_id"] == "external.files"
        )
        assert (result.root / external_payload["payload"]).read_bytes() == (
            external / "document.txt"
        ).read_bytes()
        assert any(
            item.owner == "external.files"
            and item.path == external / "empty"
            and item.status == "included_directory"
            for item in result.inventory.items
        )
    assert not result.inventory.complete
    assert all(
        b"synthetic-secret" not in (result.root / file["payload"]).read_bytes()
        for file in manifest["files"]
    )
    sealed = write_archive(result, destination, password=None, cancel=Event())
    assert sealed.path == destination
    assert destination.is_file()
    assert "synthetic-secret" in source.read_text()
    assert all(path.read_bytes() == content for path, content in original_bytes.items())


def test_budget_change_requires_preview_before_control_creation(tmp_path, monkeypatch):
    source = tmp_path / "config.toml"
    source.write_text(
        f'[general]\nusers_name="fixture"\n[paths]\ndata_dir="{tmp_path / "data"}"\n'
    )
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    options = {"allow_partial": True, "byte_budget": 1024**2}
    preview = preview_capture((source,), options=options)
    with pytest.raises(ValueError, match="capture_review_required"):
        capture(
            (source,),
            preview.scope_digest,
            tmp_path / "saved.tldw-backup.zip",
            options={**options, "byte_budget": 1024**3},
            cancel=Event(),
        )
    assert not root.exists()


def test_unenrolled_default_profile_backs_up_through_the_known_profiles_selection(
    tmp_path, monkeypatch
):
    """Pin for TASK-34100.16.1: a fresh profile's own config stays in capture scope.

    Backup & Restore asks for "all known local profiles" plus the running
    config. Builds from 2026-10-05 refused that capture on a fresh, unenrolled
    profile with ``capture_source_outside_scope``: the selected config path was
    outside the maintenance session's roots. It no longer reproduces live
    (4de437a5c5, and again on the dev 26b2e677dd rebase: g161v5-a/05). This is
    a non-reproduction pin, not a RED test: it passes on that base, and fails
    when a mutation drops the config from the capture roots. It keeps the UI's
    selection shape, the default config location under a scratch HOME and real
    stores, through a verified archive. Should the refusal return, it now
    reaches the user as its own actionable code (``test_group_issues``).
    """
    import json

    from tldw_chatbook.Backup_Recovery import capture_service
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    home = tmp_path / "home"
    source = home / ".config" / "tldw_cli" / "config.toml"
    source.parent.mkdir(parents=True, mode=0o700)
    data = home / ".local" / "share" / "tldw_cli"
    source.write_text(
        f'[general]\nusers_name="default_user"\n[paths]\ndata_dir="{data}"\n'
    )
    source.chmod(0o600)
    root = source.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    # The running profile is the default one: both locators name this file.
    monkeypatch.setattr(capture_service, "effective_config_path", lambda: source)
    monkeypatch.setattr(capture_service, "default_config_path", lambda: source)
    options = {"allow_partial": True, "staging_parent": tmp_path}
    preview = preview_capture((source,), options=options, include_known_profiles=True)
    _populate_required_dependencies(preview)
    preview = preview_capture((source,), options=options, include_known_profiles=True)
    destination = tmp_path / "fresh.tldw-backup.zip"

    result = capture(
        (source,),
        preview.scope_digest,
        destination,
        options=options,
        cancel=Event(),
        include_known_profiles=True,
    )

    assert bootstrap._records(root) == ([], []), "capture must not enroll the profile"
    manifest = json.loads(result.manifest_bytes)
    owners = {file["owner_id"] for file in manifest["files"]}
    assert {"config", "db.chachanotes.primary"} <= owners
    sealed = write_archive(result, destination, password=None, cancel=Event())
    verified = acquire(
        sealed.path, tmp_path / "inspect", ArchiveLimits(), None, Event()
    )
    assert verified.digest == sealed.digest


def test_device_state_store_lifecycle_does_not_void_a_reviewed_scope(
    tmp_path, monkeypatch
):
    """The excluded Notes device-state store may open and close between steps.

    TASK-34100.16.1: Library opens ``tldw_chatbook_notes_sync_state.db`` in WAL
    mode. A Review taken while it was open recorded its ``-wal``/``-shm`` files
    with file metadata; a backup's pause then closed the store, SQLite removed
    both, and Create rediscovered them without metadata. The reviewed digest no
    longer matched and the backup ended in ``scope_changed`` (live inventory
    diff: only those two items' metadata). The store is device-local and never
    captured, so its files coming and going is not a change of backup scope.
    """
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import capture_service
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    home = tmp_path / "home"
    source = home / ".config" / "tldw_cli" / "config.toml"
    source.parent.mkdir(parents=True, mode=0o700)
    data = home / ".local" / "share" / "tldw_cli"
    source.write_text(
        f'[general]\nusers_name="default_user"\n[paths]\ndata_dir="{data}"\n'
    )
    source.chmod(0o600)
    root = source.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    monkeypatch.setattr(capture_service, "effective_config_path", lambda: source)
    monkeypatch.setattr(capture_service, "default_config_path", lambda: source)
    options = {"allow_partial": True, "staging_parent": tmp_path}

    def review():
        return preview_capture((source,), options=options, include_known_profiles=True)

    first = review()
    database = next(
        item.path
        for item in first.items
        if item.owner == "notes.sync_state"
        and item.logical_id.endswith(":notes.sync_state")
    )
    sidecars = [Path(f"{database}{suffix}") for suffix in ("-wal", "-shm")]
    # A started profile whose Library has not opened the device-state store yet.
    _populate_required_dependencies(first)
    assert not database.exists()
    before = review().scope_digest
    store = NotesDeviceStateStore(database)
    try:
        store.initialize()
        with store.transaction() as connection:  # a held WAL read, as Library's
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        assert database.exists() and all(path.exists() for path in sidecars)
        while_open = review().scope_digest
    finally:
        store.close()
    assert not any(path.exists() for path in sidecars), "SQLite kept its sidecars"
    after_close = review().scope_digest

    assert after_close == while_open, "closing the store voided the review"
    assert before == while_open, "opening the store voided the review"


@pytest.mark.parametrize(
    "owner",
    [
        "notes.sync_state",
        "personal_context.interview_drafts",
        pytest.param(
            "personal_context.repository",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "TASK-34100.16.1.1: its present files stay explicit "
                    "unsupported rows (owner decision 2026-11-06), and a "
                    "present row's path is reviewed scope"
                ),
            ),
        ),
    ],
)
def test_local_sqlite_wal_lifecycle_does_not_void_a_reviewed_scope(
    tmp_path, monkeypatch, owner
):
    """No uncaptured local SQLite store's WAL open/close changes the scope.

    TASK-34100.16.1 review: the excluded stores (the Notes device state and
    the Personal Context interview drafts) share the fixed code path; Create
    runs with the digest reviewed while the store was open.
    ``personal_context.repository`` is declared but not excluded: its
    ``-wal``/``-shm`` rows flip between unsupported (present, path in scope)
    and unused (absent) as Personal Context opens and closes per-operation
    WAL connections, so a Review taken during one fails at Create with
    ``scope_changed``. That case is held for TASK-34100.16.1.1.
    """
    import sqlite3
    from contextlib import closing
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import capture_service

    home = tmp_path / "home"
    source = home / ".config" / "tldw_cli" / "config.toml"
    source.parent.mkdir(parents=True, mode=0o700)
    data = home / ".local" / "share" / "tldw_cli"
    source.write_text(
        f'[general]\nusers_name="default_user"\n[paths]\ndata_dir="{data}"\n'
    )
    source.chmod(0o600)
    root = source.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    monkeypatch.setattr(capture_service, "effective_config_path", lambda: source)
    monkeypatch.setattr(capture_service, "default_config_path", lambda: source)
    options = {"allow_partial": True, "staging_parent": tmp_path}

    def review():
        return preview_capture((source,), options=options, include_known_profiles=True)

    first = review()
    database = next(
        item.path
        for item in first.items
        if item.owner == owner and item.logical_id.endswith(":" + owner)
    )
    sidecars = [Path(f"{database}{suffix}") for suffix in ("-wal", "-shm")]
    _populate_required_dependencies(first)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:  # a store in use before
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS t (x)")
        connection.commit()
    assert not any(path.exists() for path in sidecars)
    before = review().scope_digest
    held = sqlite3.connect(database)
    try:
        held.execute("BEGIN")
        held.execute("SELECT count(*) FROM t").fetchone()  # a held WAL read
        assert all(path.exists() for path in sidecars)
        while_open = review().scope_digest
    finally:
        held.close()
    assert not any(path.exists() for path in sidecars), "SQLite kept its sidecars"
    after_close = review().scope_digest

    assert after_close == while_open, "closing the store voided the review"
    assert before == while_open, "opening the store voided the review"
    result = capture(
        (source,),
        while_open,
        tmp_path / "after-close.tldw-backup.zip",
        options=options,
        cancel=Event(),
        include_known_profiles=True,
    )
    assert result.manifest_bytes


def test_console_tokenizer_lookup_leaves_no_tokenizers_root_to_restore(
    tmp_path, monkeypatch
):
    """A profile that only used Console carries no custom-tokenizers root.

    TASK-34100.16.1 review (AC#3): opening Console constructed the tokenizer
    manager, which created an empty ``~/.config/tldw_cli/tokenizers`` (0755).
    Every later archive then carried ``tokenizers.custom`` as an included
    root, and isolated restore on a new machine (where that shared folder does
    not exist as an empty owner-only container) refused every folder with
    ``isolated_destination_parent_overlaps_control``. Only a write creates
    the folder now, owner-only.
    """
    from tldw_chatbook.Backup_Recovery import capture_service
    from tldw_chatbook.Utils.custom_tokenizers import CustomTokenizerManager

    home = tmp_path / "home"
    source = home / ".config" / "tldw_cli" / "config.toml"
    source.parent.mkdir(parents=True, mode=0o700)
    data = home / ".local" / "share" / "tldw_cli"
    source.write_text(
        f'[general]\nusers_name="default_user"\n[paths]\ndata_dir="{data}"\n'
    )
    source.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    root = source.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    monkeypatch.setattr(capture_service, "effective_config_path", lambda: source)
    monkeypatch.setattr(capture_service, "default_config_path", lambda: source)
    options = {"allow_partial": True, "staging_parent": tmp_path}
    tokenizers = source.parent / "tokenizers"

    manager = CustomTokenizerManager()  # what Console's token display builds
    assert manager.get_tokenizer_for_model("gpt-4o", "openai") is None
    assert manager.list_available_tokenizers() == []
    preview = preview_capture((source,), options=options, include_known_profiles=True)

    assert not tokenizers.exists(), "a read created the tokenizers folder"
    assert {
        item.status for item in preview.items if item.owner == "tokenizers.custom"
    } == {"unused"}
    manager.add_mapping("model", "tokenizer")  # the first write creates it
    assert tokenizers.is_dir() and tokenizers.stat().st_mode & 0o077 == 0
