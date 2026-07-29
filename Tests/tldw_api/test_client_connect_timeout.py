"""The API client must not spend minutes failing to connect.

`TLDWAPIClient` took a single `timeout` (default 300s) and handed it to
httpx, which applies one value to connect, read, write and pool alike. A
server that is configured but unreachable -- a blackholing firewall, a VPN
that is down, a stale host -- therefore hung the *connect* phase for five
minutes.

That is not merely slow. Screens fetch during mount, and Textual awaits a
screen's mount inside the App's own `NavigateToScreen` handler, so the App's
message pump is blocked for the whole call: the entire app stops responding
to clicks and keys until the connect finally gives up.

Long reads are legitimate (uploads, transcription, batch jobs), so the read
budget must stay generous. A five-minute *connection* attempt never is.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connect_timeout_is_bounded_even_with_a_long_overall_timeout():
    """Connect is capped short while read stays long."""
    client = TLDWAPIClient(base_url="http://example.invalid", timeout=300.0)
    try:
        httpx_client = await client._get_client()
        timeout = httpx_client.timeout

        assert timeout.connect is not None, "connect must not be unbounded"
        assert timeout.connect <= 15.0, (
            f"connect timeout is {timeout.connect}s; an unreachable host freezes "
            "the app for that long during a screen mount"
        )
        # The long budget is still honored where it is actually needed.
        assert timeout.read == 300.0
    finally:
        await client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_caller_supplied_connect_timeout_is_respected():
    """An explicit connect budget overrides the default cap."""
    client = TLDWAPIClient(
        base_url="http://example.invalid", timeout=300.0, connect_timeout=3.0
    )
    try:
        httpx_client = await client._get_client()
        assert httpx_client.timeout.connect == 3.0
    finally:
        await client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_overall_timeout_is_not_lengthened_by_the_cap():
    """A caller asking for a short timeout keeps it; the cap is a ceiling."""
    client = TLDWAPIClient(base_url="http://example.invalid", timeout=2.0)
    try:
        httpx_client = await client._get_client()
        assert httpx_client.timeout.connect == 2.0
        assert httpx_client.timeout.read == 2.0
    finally:
        await client.close()
