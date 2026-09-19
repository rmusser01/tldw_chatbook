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

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.TTS.profile_schema import open_profile_store

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
