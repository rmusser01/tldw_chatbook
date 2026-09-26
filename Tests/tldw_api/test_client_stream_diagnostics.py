"""Malformed-line diagnostics on the NDJSON ingest stream (task-32902).

Tier-2 review slice S06 P3: ``_stream_request`` dropped an undecodable line
with a bare ``print()`` -- the only one in the package. Textual runs the app
under ``redirect_stdout``, so it reached neither the loguru sinks nor the
in-app Logs window: ``process_mediawiki_dump`` / ``ingest_mediawiki_dump``
skipped pages silently.
"""

from __future__ import annotations

import contextlib

import pytest
from loguru import logger

from tldw_chatbook.tldw_api.client import TLDWAPIClient


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    @contextlib.asynccontextmanager
    async def stream(self, *args, **kwargs):
        yield _FakeStreamResponse(self._lines)


@pytest.mark.asyncio
async def test_malformed_ndjson_line_reaches_the_log_sinks(monkeypatch):
    client = TLDWAPIClient(base_url="https://server.example")
    fake = _FakeClient(['{"page": 1}', "{ not json", '{"page": 2}'])

    async def _get_client():
        return fake

    monkeypatch.setattr(client, "_get_client", _get_client)
    monkeypatch.setattr(
        TLDWAPIClient, "_raise_if_redirected", staticmethod(lambda *a, **k: _noop())
    )

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    try:
        pages = [page async for page in client._stream_request("POST", "/x")]
    finally:
        logger.remove(sink_id)

    assert pages == [{"page": 1}, {"page": 2}]
    assert any("Could not decode JSON line" in message for message in records)


async def _noop() -> None:
    return None
