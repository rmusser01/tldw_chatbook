"""TASK-34100.2 AC#3/#5 (gap-01, gap-03): a key check that can actually fail.

The 2026-10-02 review found setup accepting any OpenRouter or Hugging Face
key (their model lists answer a random key, so the "check" was vacuous: the
expired repo-root OpenRouter key got a 466-model list and Summary ✓), and no
Gemini model list or key check at all ("Key staged… Connection testing is
unavailable"). These tests pin the shared discovery behaviour every surface
reads: the public-listing set lives on the provider record, OpenRouter and
Hugging Face get an authenticated key check that returns the provider's own
reason, and Gemini lists its chat models from Google's models.list, which
also verifies the key. No test here touches a live server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_chatbook.Chat.provider_test_evidence import (
    PUBLIC_MODEL_LISTING_PROVIDER_KEYS,
    UNCHECKED_KEY_LISTING_PROVIDER_KEYS,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    discover_openai_compatible_models,
    supports_openai_compatible_model_discovery,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY
from tldw_chatbook.UI.Screens.settings_endpoint_probe import key_check_probe_result


# --- the public-listing set lives on the provider record ---------------------


def test_the_public_listing_set_is_the_records_and_includes_hf_and_novita() -> None:
    from_records = {
        key
        for key, record in RECORDS_BY_KEY.items()
        if record.public_model_listing and key == record.key
    }

    assert from_records == {"openrouter", "nvidia", "huggingface", "novita"}
    assert PUBLIC_MODEL_LISTING_PROVIDER_KEYS == from_records


def test_only_providers_without_a_key_check_read_key_not_checked() -> None:
    assert UNCHECKED_KEY_LISTING_PROVIDER_KEYS == {"nvidia", "novita"}
    assert RECORDS_BY_KEY["openrouter"].key_check_url == (
        "https://openrouter.ai/api/v1/key"
    )
    assert RECORDS_BY_KEY["huggingface"].key_check_url == (
        "https://huggingface.co/api/whoami-v2"
    )


# --- OpenRouter / Hugging Face: an authenticated key check -------------------

_KEY = "sk-or-v1-test-key-check-value"


def _transport(routes: dict[str, httpx.Response], seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        for prefix, response in routes.items():
            if str(request.url).startswith(prefix):
                return response
        return httpx.Response(599, json={"error": "unexpected route"})

    return httpx.MockTransport(handler)


async def _discover(provider: str, endpoint: str, routes, api_key=_KEY):
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(routes, seen)) as client:
        result = await discover_openai_compatible_models(
            provider=provider,
            provider_list_key=provider,
            endpoint=endpoint,
            api_key=api_key,
            client=client,
        )
    return result, seen


@pytest.mark.asyncio
async def test_an_expired_openrouter_key_is_rejected_with_openrouters_reason() -> None:
    result, seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "https://openrouter.ai/api/v1/models": httpx.Response(
                200, json={"data": [{"id": "openai/gpt-4.1-mini"}]}
            ),
            "https://openrouter.ai/api/v1/key": httpx.Response(
                401, json={"error": {"message": "API key expired.", "code": 401}}
            ),
        },
    )

    assert result.status == "error"
    assert result.error.kind == "missing_credentials"
    assert result.error.category == "unauthorized"
    assert "API key expired." in result.error.message
    assert "OpenRouter" in result.error.message
    assert result.models == ()
    assert seen[-1].headers["Authorization"] == f"Bearer {_KEY}"
    assert _KEY not in result.error.message


@pytest.mark.asyncio
async def test_an_accepted_openrouter_key_is_verified_by_the_key_endpoint() -> None:
    result, seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "https://openrouter.ai/api/v1/models": httpx.Response(
                200, json={"data": [{"id": "openai/gpt-4.1-mini"}]}
            ),
            "https://openrouter.ai/api/v1/key": httpx.Response(
                200, json={"data": {"label": "sk-or-v1-abc...xyz"}}
            ),
        },
    )

    assert result.status == "success"
    assert result.key_verified is True
    assert [str(request.url) for request in seen] == [
        "https://openrouter.ai/api/v1/models",
        "https://openrouter.ai/api/v1/key",
    ]


@pytest.mark.asyncio
async def test_no_key_means_no_key_check_and_nothing_verified() -> None:
    result, seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "https://openrouter.ai/api/v1/models": httpx.Response(
                200, json={"data": [{"id": "openai/gpt-4.1-mini"}]}
            ),
        },
        api_key=None,
    )

    assert result.status == "success"
    assert result.key_verified is False
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_bogus_hugging_face_token_is_rejected_by_whoami() -> None:
    result, seen = await _discover(
        "huggingface",
        "https://router.huggingface.co/v1/chat/completions",
        {
            "https://router.huggingface.co/v1/models": httpx.Response(
                200, json={"data": [{"id": "openai/gpt-oss-120b"}]}
            ),
            "https://huggingface.co/api/whoami-v2": httpx.Response(
                401, json={"error": "Invalid credentials in Authorization header"}
            ),
        },
        api_key="hf_test_token_value",
    )

    assert result.status == "error"
    assert result.error.kind == "missing_credentials"
    assert "Invalid credentials in Authorization header" in result.error.message
    assert str(seen[-1].url) == "https://huggingface.co/api/whoami-v2"


def _openrouter_listing() -> dict[str, httpx.Response]:
    return {
        "https://openrouter.ai/api/v1/models": httpx.Response(
            200, json={"data": [{"id": "openai/gpt-4.1-mini"}]}
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_answer",
    [
        httpx.Response(503, text="busy"),
        httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}}),
        # Owner ruling 2026-10-02 (PR #2958): only a 401 rejects a key.
        httpx.Response(403, json={"error": {"message": "Forbidden"}}),
    ],
)
async def test_an_unanswered_key_check_keeps_the_listing_and_verifies_nothing(
    key_answer: httpx.Response,
) -> None:
    """Review round 1 (F6c): a 429 or 5xx from the key endpoint threw away a
    listing that succeeded, failing the whole discovery. A check that got no
    answer verifies nothing and blocks nothing (fails closed on the KEY)."""
    result, _seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {**_openrouter_listing(), "https://openrouter.ai/api/v1/key": key_answer},
    )

    assert result.status == "success"
    assert [model.model_id for model in result.models] == ["openai/gpt-4.1-mini"]
    assert result.key_verified is False
    assert key_check_probe_result(result, key_sent=True).key_accepted is False


@pytest.mark.asyncio
async def test_a_compressed_401_from_the_key_endpoint_still_rejects_the_key() -> None:
    """Review round 1 (F1/F6a): an unreadable (gzip) key-check body was
    treated as status 200, so an expired key was reported verified."""
    import gzip

    result, _seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {
            **_openrouter_listing(),
            "https://openrouter.ai/api/v1/key": httpx.Response(
                401,
                headers={"Content-Encoding": "gzip"},
                content=gzip.compress(b'{"error": {"message": "API key expired."}}'),
            ),
        },
    )

    assert result.status == "error"
    assert result.error.kind == "missing_credentials"
    assert result.error.category == "unauthorized"
    assert result.key_verified is False


@pytest.mark.asyncio
async def test_a_compressed_200_from_the_key_endpoint_verifies_nothing() -> None:
    import gzip

    result, _seen = await _discover(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        {
            **_openrouter_listing(),
            "https://openrouter.ai/api/v1/key": httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                content=gzip.compress(b'{"data": {}}'),
            ),
        },
    )

    assert result.status == "success"
    assert result.key_verified is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "endpoint", "listing"),
    [
        (
            "openrouter",
            "https://llm-proxy.example.test/api/v1/chat/completions",
            "https://llm-proxy.example.test/api/v1/models",
        ),
        (
            "huggingface",
            "https://my-endpoint.endpoints.example.test/v1/chat/completions",
            "https://my-endpoint.endpoints.example.test/v1/models",
        ),
    ],
)
async def test_a_key_for_another_host_is_never_sent_to_the_key_check(
    provider: str, endpoint: str, listing: str
) -> None:
    """Review round 1 (F6b): the key check used the record's fixed host
    whatever endpoint was configured, so a proxy's or a dedicated endpoint's
    key went to openrouter.ai / huggingface.co and came back 'rejected'."""
    result, seen = await _discover(
        provider,
        endpoint,
        {listing: httpx.Response(200, json={"data": [{"id": "model-a"}]})},
    )

    assert result.status == "success"
    assert result.key_verified is False
    assert [str(request.url) for request in seen] == [listing]


@pytest.mark.parametrize(
    ("provider", "verified", "accepted"),
    [
        ("openrouter", False, False),  # listed, key never checked
        ("openrouter", True, True),
        ("huggingface", True, True),
        ("nvidia", False, False),
        ("novita", False, False),
        ("openai", False, True),  # an authenticated listing proves the key
    ],
)
def test_a_listing_accepts_a_key_only_when_something_checked_it(
    provider: str, verified: bool, accepted: bool
) -> None:
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )

    result = ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint="https://example.test/v1",
        status="success",
        models=(
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider,
                model_id="model-a",
                display_name="model-a",
                source="runtime_discovered",
                endpoint_fingerprint="https://example.test/v1",
                discovered_at="2026-10-03T00:00:00Z",
            ),
        ),
        key_verified=verified,
    )

    probe = key_check_probe_result(result, key_sent=True)

    assert probe is not None
    assert probe.key_accepted is accepted


# --- Gemini: native listing doubles as the key check -------------------------

_GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta"
_GOOGLE_MODELS = [
    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 1048576},
    {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.1-flash-image", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3-flash-preview", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
]


def test_the_google_api_base_supports_discovery() -> None:
    assert supports_openai_compatible_model_discovery("google", _GOOGLE_BASE)


@pytest.mark.asyncio
async def test_gemini_lists_chat_models_only_and_verifies_the_key() -> None:
    result, seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {
            f"{_GOOGLE_BASE}/models": httpx.Response(
                200, json={"models": _GOOGLE_MODELS}
            ),
        },
        api_key="AIza-test-google-key",
    )

    assert result.status == "success"
    assert result.key_verified is True
    ids = [model.model_id for model in result.models]
    assert ids[0] == "gemini-flash-latest"  # the rolling alias leads
    assert set(ids) == {"gemini-flash-latest", "gemini-3.8-flash", "gemini-2.5-flash"}
    request = seen[0]
    assert request.headers["x-goog-api-key"] == "AIza-test-google-key"
    assert "Authorization" not in request.headers
    assert "key=" not in str(request.url)  # the key never rides in the URL


@pytest.mark.asyncio
async def test_gemini_follows_next_page_tokens() -> None:
    pages = iter(
        [
            httpx.Response(
                200,
                json={"models": _GOOGLE_MODELS[:2], "nextPageToken": "page-2"},
            ),
            httpx.Response(200, json={"models": _GOOGLE_MODELS[2:3]}),
        ]
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return next(pages)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_openai_compatible_models(
            provider="google",
            provider_list_key="Google",
            endpoint=_GOOGLE_BASE,
            api_key="AIza-test-google-key",
            client=client,
        )

    assert result.status == "success"
    assert len(result.models) == 3
    assert seen[1].url.params["pageToken"] == "page-2"


@pytest.mark.asyncio
async def test_an_invalid_gemini_key_is_rejected_with_googles_reason() -> None:
    body = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "API_KEY_INVALID"}],
        }
    }
    result, _seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {f"{_GOOGLE_BASE}/models": httpx.Response(400, content=json.dumps(body))},
        api_key="AIza-bogus",
    )

    assert result.status == "error"
    assert result.error.kind == "missing_credentials"
    assert result.error.category == "unauthorized"
    assert "API key not valid" in result.error.message
    assert "AIza-bogus" not in result.error.message
    probe = key_check_probe_result(result, key_sent=True)
    assert (probe.endpoint, probe.category) == ("unreachable", "unauthorized")


@pytest.mark.asyncio
async def test_a_gemini_403_is_never_called_a_rejected_key() -> None:
    """Review round 2 (R2-8): only a 401 -- or Google's own 'API key not
    valid' 400 -- rejects a key (owner ruling, PR #2958). A 403 (the Gemini
    API disabled for the key's project, a key restricted to other APIs) keeps
    the shared 403 shape every other listing uses (``forbidden``), carries
    Google's reason, and never says the key was rejected."""
    reason = (
        "Generative Language API has not been used in project 123 before or "
        "it is disabled."
    )
    body = {
        "error": {
            "code": 403,
            "message": reason,
            "status": "PERMISSION_DENIED",
            "details": [{"reason": "SERVICE_DISABLED"}],
        }
    }
    result, _seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {f"{_GOOGLE_BASE}/models": httpx.Response(403, content=json.dumps(body))},
        api_key="AIza-project-disabled",
    )

    assert result.status == "error"
    assert result.key_verified is False
    assert result.error.category == "forbidden"
    assert "rejected" not in result.error.message.lower()
    assert reason in result.error.message
    assert result.error.provider_reason == reason
    # Settings reads it exactly as it reads any listing's 403 (dev's
    # TASK-33008 D2 keeps the 403's own "forbidden" on the evidence).
    probe = key_check_probe_result(result, key_sent=True)
    assert (probe.endpoint, probe.category) == ("model_listing_unavailable", "forbidden")
