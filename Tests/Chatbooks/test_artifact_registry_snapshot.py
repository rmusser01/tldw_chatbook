"""Real-registry artifact reads stay complete and detached for one request."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope


@pytest.fixture
def service(tmp_path):
    return LocalChatbookService(registry_path=tmp_path / "chatbooks.json")


async def seed_registry(service, count):
    """Expand a real service-created record while preserving its registry envelope."""
    template = await service.create_chatbook(name="Template")
    registry = service._load_registry()
    records = []
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(count):
        record = deepcopy(template)
        record.update(
            id=str(index + 1),
            chatbook_id=index + 1,
            name=f"Pack {index:04d}",
            description=f"Description {index:04d}",
            created_at=(epoch + timedelta(seconds=index)).isoformat(),
            updated_at=(epoch + timedelta(seconds=index)).isoformat(),
        )
        records.append(record)
    registry["records"] = records
    registry["next_id"] = count + 1
    service._save_registry(registry)
    return records


@pytest.mark.asyncio
async def test_snapshot_reads_the_complete_registry_once_before_bounded_windows(
    service, monkeypatch
):
    records = await seed_registry(service, 1005)
    real_load = service._load_registry
    loads = 0

    def counting_load():
        nonlocal loads
        loads += 1
        return real_load()

    monkeypatch.setattr(service, "_load_registry", counting_load)
    snapshot = service.artifact_read_snapshot()
    scope = ArtifactScope(view="chatbooks")
    first = snapshot.read_artifact_window(
        scope, boundary=None, direction="after", limit=20
    )
    assert first.total == 1005
    assert [row.key.native_id for row in first.items] == list(range(1005, 985, -1))
    assert (
        snapshot.get_artifact_summary(scope, ArtifactKey("chatbook", 1)).title
        == "Pack 0000"
    )
    last = snapshot.read_artifact_window(
        scope, boundary=None, direction="before", limit=20
    )
    assert [row.key.native_id for row in last.items] == list(range(20, 0, -1))
    assert last.before_boundary == 1005
    assert loads == 1
    assert len(tuple(snapshot.iter_records())) == 1005
    assert loads == 1
    # A new registry commit must not change this request's metadata or body.
    await service.update_chatbook(1005, name="Changed after snapshot")
    assert snapshot.get_record(1005)["name"] == records[-1]["name"]
    assert (
        snapshot.get_artifact_summary(scope, ArtifactKey("chatbook", 1005)).title
        == "Pack 1004"
    )
    copied = snapshot.get_record(1005)
    copied["metadata"]["content"] = "Caller mutation"
    assert "content" not in snapshot.get_record(1005)["metadata"]
    # Snapshot access is read-only, including the original provenance envelope.
    assert json.loads(service.registry_path.read_text())["next_id"] == 1006


@pytest.mark.asyncio
async def test_snapshot_filters_metadata_before_slicing_and_never_searches_saved_body(
    service,
):
    await seed_registry(service, 35)
    await service.update_chatbook(35, tags=["tail-tag"])
    await service.update_chatbook(34, categories=["tail-category"])
    await service.update_chatbook(
        33,
        metadata={
            "artifact_source": "console",
            "artifact_kind": "assistant-response",
            "content": "body-only-needle",
            "content_truncated": False,
        },
    )
    snapshot = service.artifact_read_snapshot()
    for query, expected in [
        ("tail-tag", 35),
        ("tail-category", 34),
        ("Description 0030", 31),
        ("Console", 33),
    ]:
        window = snapshot.read_artifact_window(
            ArtifactScope(view="chatbooks", query=query),
            boundary=None,
            direction="after",
            limit=20,
        )
        assert window.total == 1
        assert [row.key.native_id for row in window.items] == [expected]
    assert (
        snapshot.read_artifact_window(
            ArtifactScope(view="chatbooks", query="body-only-needle"),
            boundary=None,
            direction="after",
            limit=20,
        ).total
        == 0
    )
    for view in ("reports", "chatbooks", "all"):
        assert (
            snapshot.read_artifact_window(
                ArtifactScope(view=view, kept_only=True),
                boundary=None,
                direction="after",
                limit=20,
            ).total
            == 0
        )


@pytest.mark.asyncio
async def test_snapshot_title_keys_match_sqlite_and_newest_uses_creation(service):
    await seed_registry(service, 35)
    await service.update_chatbook(35, name="AAA first")
    await service.update_chatbook(1, name="ÉCLAIR")
    await service.update_chatbook(2, name="éclair")
    snapshot = service.artifact_read_snapshot()
    titles = snapshot.read_artifact_window(
        ArtifactScope(view="chatbooks", sort="title"),
        boundary=None,
        direction="after",
        limit=20,
    )
    assert titles.items[0].key == ArtifactKey("chatbook", 35)
    newest = snapshot.read_artifact_window(
        ArtifactScope(view="chatbooks"),
        boundary=None,
        direction="after",
        limit=20,
    )
    assert newest.items[0].key == ArtifactKey("chatbook", 35)
    assert newest.items[0].revision == titles.items[0].revision
    # SQLite LOWER folds ASCII only: do not introduce a different Unicode key.
    query = snapshot.read_artifact_window(
        ArtifactScope(view="chatbooks", query="ÉCLAIR"),
        boundary=None,
        direction="after",
        limit=20,
    )
    assert [row.key.native_id for row in query.items] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [1.5, True, False, 0, -1, None, "01", "bad"])
async def test_snapshot_rejects_malformed_native_identity(service, bad_id):
    await seed_registry(service, 1)
    registry = service._load_registry()
    registry["records"][0]["chatbook_id"] = bad_id
    service._save_registry(registry)
    with pytest.raises((TypeError, ValueError)):
        service.artifact_read_snapshot()


@pytest.mark.asyncio
async def test_snapshot_rejects_duplicate_or_inconsistent_identity(service):
    await seed_registry(service, 2)
    registry = service._load_registry()
    registry["records"][1]["chatbook_id"] = 3
    service._save_registry(registry)
    with pytest.raises(ValueError):
        service.artifact_read_snapshot()
    registry["records"][1]["chatbook_id"] = 1
    registry["records"][1]["id"] = "1"
    service._save_registry(registry)
    with pytest.raises(ValueError):
        service.artifact_read_snapshot()


@pytest.mark.parametrize(
    "invalid", ["bad\x00.zip", "../../escape.zip", "pack;command.zip"]
)
def test_invalid_bundle_path_is_refused_before_filesystem_probes(
    tmp_path, monkeypatch, invalid
):
    import zipfile
    from pathlib import Path

    from tldw_chatbook.Chatbooks.artifact_registry_snapshot import (
        usable_chatbook_bundle,
    )

    raw = str(tmp_path / invalid)
    probes = []
    for name in ("is_symlink", "is_file"):
        original = getattr(Path, name)

        def probe(path, _name=name, _original=original):
            if str(path) == raw:
                probes.append(_name)
                return _name == "is_file"
            return _original(path)

        monkeypatch.setattr(Path, name, probe)

    def probe_zip(path):
        probes.append("zip")
        return False

    monkeypatch.setattr(zipfile, "is_zipfile", probe_zip)
    assert not usable_chatbook_bundle(raw)[0]
    assert probes == []


def test_home_relative_bundle_remains_shareable(tmp_path, monkeypatch):
    import zipfile

    from tldw_chatbook.Chatbooks.artifact_registry_snapshot import (
        usable_chatbook_bundle,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    with zipfile.ZipFile(tmp_path / "registered-pack.zip", "w") as bundle:
        bundle.writestr("manifest.json", "{}")
    assert usable_chatbook_bundle("~/registered-pack.zip")[0]
