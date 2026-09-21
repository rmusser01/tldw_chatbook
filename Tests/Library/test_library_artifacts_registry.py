"""Registered Chatbooks compose with report owners without changing inventory."""

import json
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from Tests.Chatbooks.test_artifact_registry_snapshot import seed_registry
from tldw_chatbook.Chat.console_save_targets import console_chatbook_artifact_payload
from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Library.library_artifacts_catalog import (
    ArtifactReadError,
    LibraryArtifactsCatalog,
)
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("usable_zip", [False, True])
def test_chatbook_actions_require_a_usable_bundle(saved, usable_zip):
    from tldw_chatbook.Library.library_artifacts_state import chatbook_actions

    expected = {"preview", "manage_packs"}
    if usable_zip:
        expected.add("share")
    assert chatbook_actions(
        is_saved_response=saved, usable_zip=usable_zip
    ) == frozenset(expected)


@pytest.fixture
def service(tmp_path):
    return LocalChatbookService(registry_path=tmp_path / "chatbooks.json")


def catalog_for(service, *, subs=None, kept=None):
    return LibraryArtifactsCatalog(
        subscriptions_db=subs, chachanotes_db=kept, chatbook_service=service
    )


@pytest.mark.asyncio
async def test_registry_catalog_pages_and_locates_beyond_the_old_tail(
    service, monkeypatch
):
    await seed_registry(service, 1005)
    catalog = catalog_for(service)
    scope = ArtifactScope(view="chatbooks")
    real_load = service._load_registry
    loads = 0

    def counted_load():
        nonlocal loads
        loads += 1
        return real_load()

    monkeypatch.setattr(service, "_load_registry", counted_load)
    target = ArtifactKey("chatbook", 1)
    located = catalog.locate(scope, target)
    assert loads == 1
    assert located.start == 1000
    assert located.total == 1005
    assert [row.key.native_id for row in located.items] == [5, 4, 3, 2, 1]
    page = catalog.read_page(scope)
    seen = list(page.items)
    while page.start + len(page.items) < page.total:
        page = catalog.read_page(scope, boundary=page.items[-1].order_key)
        assert page.start == len(seen)
        assert 0 < len(page.items) <= 20
        seen.extend(page.items)
    assert [row.key.native_id for row in seen] == list(range(1005, 0, -1))
    previous = catalog.read_page(
        scope, boundary=page.items[0].order_key, direction="before"
    )
    assert previous.start == 980
    assert [row.key.native_id for row in previous.items] == list(range(25, 5, -1))


@pytest.mark.asyncio
async def test_all_artifacts_interleaves_complete_sources_and_kept_filter_is_exclusive(
    service, tmp_path
):
    subs = SubscriptionsDB(tmp_path / "subs.db", "registry-test")
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="registry-test")
    try:
        watch = WatchlistBundleService(subs).create(name="Live report")["id"]
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        expected = []
        for index in range(27):
            saved = kept.create_kept_briefing(
                source_briefing_id=index + 1,
                watchlist_name="Kept report",
                body_markdown=f"Saved {index}",
                origin="manual",
                original_created_at=(epoch + timedelta(seconds=index * 3)).isoformat(),
            )
            live = subs.insert_briefing(watch)
            subs.update_briefing(live, status="complete", body_markdown=f"Live {index}")
            with subs.transaction() as connection:
                connection.execute(
                    "UPDATE briefings SET created_at = ? WHERE id = ?",
                    ((epoch + timedelta(seconds=index * 3 + 1)).isoformat(), live),
                )
            record = await service.create_chatbook(name=f"Pack {index:02d}")
            registry = service._load_registry()
            registry["records"][-1]["created_at"] = (
                epoch + timedelta(seconds=index * 3 + 2)
            ).isoformat()
            service._save_registry(registry)
            expected[0:0] = [
                ArtifactKey("chatbook", record["chatbook_id"]),
                ArtifactKey("live_report", live),
                ArtifactKey("kept_report", saved),
            ]
        catalog = catalog_for(service, subs=subs, kept=kept)
        scope = ArtifactScope(view="all")
        page = catalog.read_page(scope)
        seen = list(page.items)
        while page.start + len(page.items) < page.total:
            page = catalog.read_page(scope, boundary=page.items[-1].order_key)
            assert page.start == len(seen)
            seen.extend(page.items)
        assert page.total == 81
        assert [row.key for row in seen] == expected
        offsets = {"kept_report": 0, "live_report": 1, "chatbook": 2}
        assert all(
            row.created_at
            == (
                epoch
                + timedelta(
                    seconds=(row.key.native_id - 1) * 3 + offsets[row.key.source]
                )
            ).isoformat()
            for row in seen
        )
        located = catalog.locate(scope, expected[-2])
        assert located.start == 60
        assert expected[-2] in [row.key for row in located.items]
        kept_page = catalog.read_page(ArtifactScope(view="all", kept_only=True))
        assert kept_page.total == 27
        assert {row.key.source for row in kept_page.items} == {"kept_report"}
    finally:
        kept.close_connection()
        subs.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [1500, 21000])
async def test_saved_response_detail_uses_stored_body_and_exact_source(
    service, tmp_path, length
):
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="registry-test")
    try:
        conversation_id = kept.add_conversation({"title": "Exact source"})
        payload = console_chatbook_artifact_payload(
            title="Saved answer",
            message_text="x" * length,
            message_role="Assistant",
            conversation_id=conversation_id,
            message_id="exact-message",
            provider="Test provider",
            model="Test model",
        )
        record = await service.create_chatbook(**payload)
        detail = catalog_for(service, kept=kept).read_detail(
            ArtifactKey("chatbook", record["chatbook_id"])
        )
        assert detail.body == payload["metadata"]["content"]
        assert len(detail.body) == min(length, 20000)
        assert detail.truncated is (length > 20000)
        assert (
            not detail.can_share
            and not detail.can_export
            and not detail.can_keep
            and not detail.can_play
        )
        assert detail.source_available
        assert detail.source_conversation_id == conversation_id
        assert detail.source_message_id == "exact-message"
        assert dict(detail.details)["Provider"] == "Test provider"
        assert dict(detail.details)["Model"] == "Test model"
        assert catalog_for(service).read_detail(detail.key).source_available is False
        assert catalog_for(service).read_detail(ArtifactKey("chatbook", 999999)) is None
    finally:
        kept.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_kind", ["valid", "missing", "invalid", "symlink", "none"]
)
async def test_share_capability_requires_a_selected_regular_zip(
    service, tmp_path, file_kind, monkeypatch
):
    bundle = tmp_path / "pack.zip"
    manifest = ChatbookManifest(
        version=ChatbookVersion.V2, name="Bundle snapshot", description="Exported bytes"
    )
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
    path = bundle
    if file_kind == "missing":
        path = tmp_path / "missing.zip"
    elif file_kind == "invalid":
        path = tmp_path / "invalid.zip"
        path.write_text("Not a ZIP")
    elif file_kind == "symlink":
        path = tmp_path / "linked.zip"
        path.symlink_to(bundle)
    elif file_kind == "none":
        path = None
    record = await service.create_chatbook(
        name="Registered metadata", description="Metadata preview", file_path=path
    )
    catalog = catalog_for(service)
    real_is_zip = zipfile.is_zipfile
    validations = []

    def checked_zip(path):
        validations.append(path)
        return real_is_zip(path)

    monkeypatch.setattr(zipfile, "is_zipfile", checked_zip)
    catalog.read_page(ArtifactScope(view="chatbooks"))
    assert validations == []  # Only selection may inspect the file.
    detail = catalog.read_detail(ArtifactKey("chatbook", record["chatbook_id"]))
    assert detail.can_share is (file_kind == "valid")
    assert not detail.can_export
    assert "Metadata preview" in detail.body
    assert dict(detail.details)["Sharing"]


@pytest.mark.asyncio
async def test_failed_registry_remains_explicit_but_report_view_stays_available(
    service,
):
    service.registry_path.write_text("{broken")
    catalog = catalog_for(service)
    for view in ("chatbooks", "all"):
        with pytest.raises(ArtifactReadError) as caught:
            catalog.read_page(ArtifactScope(view=view))
        assert caught.value.source == "chatbook"
    assert catalog.read_page(ArtifactScope(view="reports")).total == 0
