"""TASK-32894: the Settings endpoint probe's CHAT branch had no egress check.

`UI/Screens/settings_endpoint_probe.probe_settings_endpoint` is the single
funnel for all four production Test-button callers (`speech_catalog_mixin`,
`chat_screen`, `settings_screen`, `FirstRunSetupWizard`). Its TTS branch
called `check_url_or_raise_async`; its chat branch went straight to the
network with the resolved provider credential attached.

These tests drive the real helper with an `httpx.MockTransport` client, so
nothing leaves the machine either way: the blocked case is proved by the
transport never being ASKED for a request, not by a timeout.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbePurpose,
    probe_settings_endpoint,
)


@pytest.fixture(autouse=True)
def _no_real_dns_or_config(monkeypatch):
    """Same idiom as `Tests/Utils/test_egress.py`: no DNS, no config read.

    The config read is what trips ADR-126's recovery gate in a clean
    worktree, and stubbing the resolver keeps the assertion about the
    POLICY rather than about this machine's `/etc/hosts`.
    """
    monkeypatch.setattr(egress, "_resolve", lambda host: [host])

    async def _fake_async(host):
        return [host]

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _recording_client() -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "some-model"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, seen


def test_chat_probe_never_reaches_a_cloud_metadata_endpoint() -> None:
    """The exact SSRF target the egress policy blocks even for trusted origins.

    A restored profile archive can set `api_settings.<provider>.base_url`;
    the user then only has to press Test. Before the fix the credential
    went to 169.254.169.254 on the first hop.
    """

    async def scenario() -> None:
        client, seen = _recording_client()
        try:
            outcome = await probe_settings_endpoint(
                "http://169.254.169.254/v1",
                provider="custom",
                api_key="sk-should-never-be-sent",
                http_client=client,
            )
        finally:
            await client.aclose()
        assert seen == [], (
            "the chat probe issued a request to the cloud metadata endpoint: "
            f"{[str(r.url) for r in seen]}"
        )
        assert outcome.state == "unreachable"
        assert outcome.category == "connection_error"

    asyncio.run(scenario())


def test_chat_probe_still_reaches_a_configured_local_provider() -> None:
    """The guard trusts the configured origin, so local servers keep working."""

    async def scenario() -> None:
        client, seen = _recording_client()
        try:
            outcome = await probe_settings_endpoint(
                "http://127.0.0.1:11434",
                provider="custom",
                api_key="sk-local",
                http_client=client,
            )
        finally:
            await client.aclose()
        assert [str(r.url) for r in seen] == ["http://127.0.0.1:11434/v1/models"]
        assert outcome.state == "reachable"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "purpose",
    [
        SettingsEndpointProbePurpose.CHAT_CATALOG,
        SettingsEndpointProbePurpose.TTS_CATALOG,
    ],
)
def test_both_purposes_block_the_metadata_endpoint(purpose) -> None:
    """Neither branch of the shared probe may reach link-local metadata."""

    async def scenario() -> None:
        client, seen = _recording_client()
        try:
            await probe_settings_endpoint(
                "http://169.254.169.254/v1",
                provider="openai",
                api_key="sk-should-never-be-sent",
                purpose=purpose,
                http_client=client,
            )
        finally:
            await client.aclose()
        assert seen == [], f"{purpose} reached the metadata endpoint"

    asyncio.run(scenario())
