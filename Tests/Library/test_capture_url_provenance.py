"""Durable capture URL authority survives first extraction and retry."""

from __future__ import annotations

import asyncio

import pytest

from Tests.Library.test_collections_capture_scope_service import _local_service
from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
from tldw_chatbook.Utils.egress import UrlProvenance, provenance_of_stored_source


@pytest.mark.parametrize(
    "options",
    [
        None,
        "{",
        "[]",
        {},
        {"url_provenance": None},
        {"url_provenance": "invented"},
        {"url_provenance": "unknown"},
    ],
)
def test_unproven_stored_source_never_earns_private_authority(options):
    assert provenance_of_stored_source(options) is UrlProvenance.UNKNOWN


def test_explicit_stored_user_authority_survives():
    assert (
        provenance_of_stored_source('{"url_provenance":"user_entered"}')
        is UrlProvenance.USER_ENTERED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provenance", [UrlProvenance.UNKNOWN, UrlProvenance.USER_ENTERED]
)
async def test_capture_first_extraction_and_retry_keep_durable_authority(
    tmp_path, provenance
):
    seen = []

    def extractor(url, *, url_provenance):
        seen.append((url, url_provenance))
        if len(seen) == 1:
            raise RuntimeError("controlled first failure")
        return {"content": "Extracted article"}

    authority, db, repo, service = _local_service(tmp_path, extractor=extractor)
    try:
        request = CaptureSaveRequest(
            authority.key, "http://intranet/article", url_provenance=provenance
        )
        outcome = await service.save_capture(request)
        await asyncio.gather(*service._extraction_tasks)
        detail = repo.get_detail(outcome.capture.identity)
        assert detail.url_provenance is provenance
        await service.retry_extraction(detail.identity)
        await asyncio.gather(*service._extraction_tasks)
        assert seen == [(request.submitted_url, provenance)] * 2
    finally:
        db.close()


def test_v4_capture_migration_defaults_unknown_and_reopens(tmp_path):
    import sqlite3

    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )

    authority, db, repo, _service = _local_service(tmp_path)
    saved = repo.save_capture(
        CaptureSaveRequest(authority.key, "http://intranet/article")
    )
    db.close()
    path = tmp_path / "collections.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE collection_capture_items DROP COLUMN url_provenance"
        )
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version VALUES (4)")
    for _ in range(2):
        restored = LibraryCollectionsDB(path)
        try:
            detail = CollectionsCaptureRepository(
                restored, authority_key=authority.key
            ).get_detail(saved.capture.identity)
            assert detail.url_provenance is UrlProvenance.UNKNOWN
            assert detail.submitted_url == "http://intranet/article"
            assert restored.get_schema_version() == 5
        finally:
            restored.close()


def test_v5_column_on_old_marker_is_refused_without_repair(tmp_path):
    import sqlite3

    from tldw_chatbook.DB.Library_Collections_DB import (
        LibraryCollectionsDB,
        LibraryCollectionsSchemaError,
    )

    _authority, db, _repo, _service = _local_service(tmp_path)
    db.close()
    path = tmp_path / "collections.db"
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version VALUES (4)")
    with pytest.raises(
        LibraryCollectionsSchemaError, match="unsupported_hybrid_schema"
    ):
        LibraryCollectionsDB(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchall() == [
            (4,)
        ]


def test_capture_request_does_not_accept_serialized_authority_claim():
    from tldw_chatbook.Library.collections_capture_models import CollectionsCaptureError

    with pytest.raises(CollectionsCaptureError, match="invalid_url_provenance"):
        CaptureSaveRequest(
            "local:a", "http://intranet/article", url_provenance="user_entered"
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "provenance, first_host, redirect, sends_per_attempt",
    [
        (UrlProvenance.UNKNOWN, "private.example", False, 0),
        (UrlProvenance.UNKNOWN, "public.example", True, 1),
        (UrlProvenance.USER_ENTERED, "private.example", False, 1),
        (UrlProvenance.USER_ENTERED, "private.example", True, 1),
    ],
)
async def test_actual_capture_first_and_retry_apply_egress_at_every_hop(
    tmp_path, monkeypatch, provenance, first_host, redirect, sends_per_attempt
):
    import httpx

    from tldw_chatbook import app_service_wiring
    from tldw_chatbook.Utils import egress

    real_client = httpx.Client
    sent = []
    monkeypatch.setattr(
        egress,
        "_resolve",
        lambda host: ["93.184.216.34" if host == "public.example" else "192.168.1.1"],
    )

    def transport(request):
        sent.append(str(request.url))
        if redirect:
            return httpx.Response(
                302,
                request=request,
                headers={"location": "http://other-private.example/secret"},
            )
        # A fetched, non-article response proves the positive network boundary
        # without importing optional extraction engines into the minimal lane.
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/pdf"},
            content=b"%PDF",
        )

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(transport), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    authority, db, repo, service = _local_service(
        tmp_path, extractor=app_service_wiring._extract_collections_article
    )
    try:
        url = f"http://{first_host}/article"
        outcome = await service.save_capture(
            CaptureSaveRequest(authority.key, url, url_provenance=provenance)
        )
        await asyncio.gather(*service._extraction_tasks)
        assert len(sent) == sends_per_attempt
        assert repo.get_detail(outcome.capture.identity).processing_state == "failed"
        await service.retry_extraction(outcome.capture.identity)
        await asyncio.gather(*service._extraction_tasks)
        assert sent == [url] * (2 * sends_per_attempt)
        assert all("other-private.example" not in request for request in sent)
    finally:
        await service.cancel_extractions()
        db.close()


@pytest.mark.parametrize(
    "damage", ["empty_marker", "missing_marker", "missing_provenance"]
)
def test_capture_malformed_current_schema_is_refused_without_repair(tmp_path, damage):
    import sqlite3

    from tldw_chatbook.DB.Library_Collections_DB import (
        LibraryCollectionsDB,
        LibraryCollectionsSchemaError,
    )

    path = tmp_path / "collections.db"
    database = LibraryCollectionsDB(path)
    database.close()
    with sqlite3.connect(path) as connection:
        if damage == "empty_marker":
            connection.execute("DELETE FROM schema_version")
        elif damage == "missing_marker":
            connection.execute("DROP TABLE schema_version")
        else:
            connection.execute(
                "ALTER TABLE collection_capture_items DROP COLUMN url_provenance"
            )
        before = connection.execute(
            "SELECT type,name,sql FROM sqlite_schema ORDER BY type,name"
        ).fetchall()
    with pytest.raises(LibraryCollectionsSchemaError):
        LibraryCollectionsDB(path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT type,name,sql FROM sqlite_schema ORDER BY type,name"
            ).fetchall()
            == before
        )
