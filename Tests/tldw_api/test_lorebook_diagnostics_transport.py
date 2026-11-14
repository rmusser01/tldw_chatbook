"""Diagnostics query serialization over real loopback HTTP, not live-server UAT."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient

DIAGNOSTICS = {
    "chat_id": "chat-1",
    "character_id": None,
    "total_turns_with_diagnostics": 0,
    "turns": [],
    "page": 2,
    "size": 10,
    "pagination": {
        "page": 2,
        "per_page": 10,
        "total": 0,
        "total_pages": 0,
        "has_more": False,
    },
}


@pytest.fixture
def diagnostics_http_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            requests.append((request.path, parse_qs(request.query)))
            body = json.dumps(DIAGNOSTICS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "scope_kwargs",
    [
        {"scope_type": "workspace", "workspace_id": "ws-1"},
        {"workspace_id": "ws-1"},
    ],
)
async def test_workspace_diagnostics_query_reaches_http_endpoint(
    diagnostics_http_endpoint, scope_kwargs
):
    url, requests = diagnostics_http_endpoint
    client = TLDWAPIClient(url)
    try:
        result = await client.export_lorebook_diagnostics(
            "chat-1", page=2, size=10, order="desc", **scope_kwargs
        )
    finally:
        await client.close()

    assert requests == [
        (
            "/api/v1/chats/chat-1/diagnostics/lorebook",
            {
                "page": ["2"],
                "size": ["10"],
                "order": ["desc"],
                "scope_type": ["workspace"],
                "workspace_id": ["ws-1"],
            },
        )
    ]
    assert result == DIAGNOSTICS
