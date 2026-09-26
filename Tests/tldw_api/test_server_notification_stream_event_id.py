"""`stream_server_notifications` resolves the SSE event id it is sent.

Tier-2 review S06, P2 [D1]: two client methods hit
`/api/v1/notifications/stream` with two different event models.
`_sse_request._flush_event` emits `{"event", "data", "event_id"}`;
`NotificationStreamEvent` declares `event_id`, but
`ServerNotificationStreamEvent` declared **`id`** with `extra` defaulting
to ignore, so the real key was dropped and `.id` was always `None` --
`Last-Event-ID` resumption was structurally impossible on that path.

The only existing test for it (`test_server_notifications_client.py::
test_client_streams_server_notifications_from_sse_path`) mocks
`_stream_sse_request` wholesale and hand-constructs the event with
`id="11"`, so it proves the model can *hold* an id, never that the SSE
path fills one. This drives the real parser over a real SSE body.
"""

from __future__ import annotations

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient

_SSE_BODY = (
    b"event: notification\n"
    b"id: 42\n"
    b'data: {"notification_id": 11, "title": "Reminder due"}\n'
    b"\n"
)


def _client() -> TLDWAPIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        async def chunks():
            yield _SSE_BODY

        return httpx.Response(
            200, content=chunks(), headers={"content-type": "text/event-stream"}
        )

    client = TLDWAPIClient("http://api.test", "secret")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


@pytest.mark.asyncio
async def test_the_sse_event_id_reaches_the_parsed_event():
    events = [e async for e in _client().stream_server_notifications(after=0)]

    assert len(events) == 1
    assert events[0].event == "notification"
    assert events[0].data == {"notification_id": 11, "title": "Reminder due"}
    assert events[0].id == "42", (
        "the SSE `id:` line must reach the model, or Last-Event-ID "
        "resumption is impossible on this stream"
    )


@pytest.mark.asyncio
async def test_the_sibling_model_on_the_same_endpoint_also_fills_its_id():
    """Both models read the same wire event; neither may silently drop it."""
    events = [e async for e in _client().stream_notification_events(after=0)]

    assert len(events) == 1
    assert events[0].event_id == "42"
