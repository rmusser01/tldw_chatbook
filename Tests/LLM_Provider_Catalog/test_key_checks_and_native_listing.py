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


@pytest.mark.asyncio
async def test_an_accepted_hugging_face_token_is_verified_by_whoami() -> None:
    """Review F5: HF's whoami-v2 200 path was untested (and no real HF token
    exists to check it live). TASK-34100.6's Check key reads
    ``key_verified``/``key_check_passed``, so an accepted token must set both."""
    result, seen = await _discover(
        "huggingface",
        "https://router.huggingface.co/v1/chat/completions",
        {
            "https://router.huggingface.co/v1/models": httpx.Response(
                200, json={"data": [{"id": "openai/gpt-oss-120b"}]}
            ),
            "https://huggingface.co/api/whoami-v2": httpx.Response(
                200, json={"type": "user", "name": "someone"}
            ),
        },
        api_key="hf_test_token_value",
    )

    assert result.status == "success"
    assert result.key_verified is True
    assert key_check_probe_result(result, key_sent=True).key_check_passed is True
    assert [str(request.url) for request in seen] == [
        "https://router.huggingface.co/v1/models",
        "https://huggingface.co/api/whoami-v2",
    ]
    assert seen[-1].headers["Authorization"] == "Bearer hf_test_token_value"


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
    assert key_check_probe_result(result, key_sent=True).key_check_passed is False


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
    """The probe reports the listing and the key check separately; the
    evidence record decides whether the key was accepted (spec §5).

    ``key_accepted`` keeps dev's meaning -- the listing answered the key it was
    sent -- and ``key_check_passed`` carries the authenticated check
    (``ModelDiscoveryResult.key_verified``). TASK-34100.6's wizard Check key
    reads exactly that pair: an answered public listing whose key check did
    not pass says 'its model list answers without a key', not 'answered with
    an empty model list'. Settings still records an accepted key only for an
    authenticated listing or a passed key check.
    """
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import (
        ProviderDraftIdentity,
        ProviderTestEvidenceStore,
        connection_credential_revision,
    )
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
    assert probe.key_accepted is True  # the listing answered the key it was sent
    assert probe.key_check_passed is verified
    identity = ProviderDraftIdentity(
        provider_key=provider,
        connection_identity=canonical_connection_identity(
            provider, "https://example.test/v1"
        ),
        credential_source="stored",
        credential_revision=connection_credential_revision(_KEY),
        draft_generation=0,
    )
    app = type("App", (), {})()
    store = ProviderTestEvidenceStore(lambda: app)
    assert store.settle(store.begin(identity), probe)
    assert (store.evidence_for(identity).credential == "listing_accepted") is accepted


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
    # Review v5r4 (G2C-R4-1): no rolling alias leads any more; only the
    # shipped default is promoted, and the order is pinned below.
    assert set(ids) == {"gemini-flash-latest", "gemini-3.8-flash", "gemini-2.5-flash"}
    request = seen[0]
    assert request.headers["x-goog-api-key"] == "AIza-test-google-key"
    assert "Authorization" not in request.headers
    assert "key=" not in str(request.url)  # the key never rides in the URL


#: Google's own order, trimmed from models.list as served on 2026-11-12: a
#: dated id ahead of the rolling aliases, with ids Google still lists but
#: that did not serve the repo key that day (404, or 503 'high demand')
#: mixed in, and the shipped default in the middle.
_GOOGLE_LISTING_ORDER = (
    "gemini-2.5-flash",  # 404 'no longer available to new users'
    "gemini-3.5-flash",
    "gemini-2.5-pro",  # 404
    "gemini-pro-latest",
    "gemini-flash-latest",  # 503 'high demand' on every answered call
    "gemini-flash-lite-latest",  # the shipped default
    "gemini-3.8-flash",  # 503 'high demand' on every answered call
    "gemini-3.1-flash-lite",
)


@pytest.mark.asyncio
async def test_gemini_promotes_only_the_shipped_default_and_lists_known_bad_ids_last() -> None:
    """Review v5r4 (G2C-R4-1, LV3-2): setup's Model step and the Settings
    listing put ``gemini-flash-latest`` in row 2, under the recommended
    default, because every ``-latest`` alias was promoted over dated ids. It
    is one of the ids that came back 503 'high demand' on every answered
    call with the repo key on 2026-11-12, and the 2.5 ids answer 404 for a
    new key. Now only the shipped default is promoted; the rest keep
    Google's order, and an id the shipped pins name (retired or not serving)
    sorts last -- still listed, since another key may be served."""
    from tldw_chatbook.config import DEFAULT_GOOGLE_CHAT_MODEL
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        NOT_ANSWERING_GEMINI_IDS,
        RETIRED_GEMINI_IDS,
    )

    listing = [
        {"name": f"models/{model_id}", "supportedGenerationMethods": ["generateContent"]}
        for model_id in _GOOGLE_LISTING_ORDER
    ]
    result, _seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {f"{_GOOGLE_BASE}/models": httpx.Response(200, json={"models": listing})},
        api_key="AIza-test-google-key",
    )

    assert DEFAULT_GOOGLE_CHAT_MODEL == "gemini-flash-lite-latest"
    assert {"gemini-2.5-flash", "gemini-2.5-pro"} <= RETIRED_GEMINI_IDS
    assert {"gemini-flash-latest", "gemini-3.8-flash"} == NOT_ANSWERING_GEMINI_IDS
    assert result.status == "success"
    assert [model.model_id for model in result.models] == [
        "gemini-flash-lite-latest",  # the shipped default, the recommended row
        "gemini-3.5-flash",  # Google's order: no alias boost over a dated id
        "gemini-pro-latest",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",  # known not to serve a new key: last, not hidden
        "gemini-2.5-pro",
        "gemini-flash-latest",
        "gemini-3.8-flash",
    ]


#: Recorded from Google's models.list on 2026-11-05 (name, displayName and
#: description as served). "Nano Banana" is Google's brand for its image
#: models; ``gemini-nano-banana-2.1`` names no image token in its id.
_GOOGLE_IMAGE_BRAND_MODELS = [
    {
        "name": "models/gemini-nano-banana-2.1",
        "displayName": "Nano Banana 2.1",
        "description": "Gemini Nano Banana 2.1.",
        "supportedGenerationMethods": ["generateContent", "countTokens", "batchGenerateContent"],
    },
    {
        "name": "models/gemini-3.1-flash-image",
        "displayName": "Nano Banana 2",
        "description": "Gemini 3.1 Flash Image.",
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    },
    {
        "name": "models/gemini-3.5-flash",
        "displayName": "Gemini 3.5 Flash",
        "description": "Gemini 3.5 Flash",
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    },
]


@pytest.mark.asyncio
async def test_gemini_lists_no_image_model_under_googles_image_brand() -> None:
    """Review IV-3: setup listed ``gemini-nano-banana-2.1`` ('Nano Banana
    2.1', Google's image-model brand) as a chat model; a free-tier key gets
    429 'limit: 0' on it. An id or display name that carries the brand, or
    any non-chat token, keeps a model out of the chat list."""
    result, _seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {
            f"{_GOOGLE_BASE}/models": httpx.Response(
                200, json={"models": _GOOGLE_IMAGE_BRAND_MODELS}
            ),
        },
        api_key="AIza-test-google-key",
    )

    assert result.status == "success"
    assert [model.model_id for model in result.models] == ["gemini-3.5-flash"]


@pytest.mark.asyncio
async def test_gemini_drops_a_model_whose_display_name_alone_names_the_image_brand() -> None:
    """Review v5r2 TE-3: every recorded image model above also carries
    'banana' or 'image' in its id, so the display-name half of IV-3's filter
    was unguarded (matching the id alone left every test green). This row is
    SYNTHETIC: an id with no non-chat token whose display name is the image
    brand, the shape ``gemini-nano-banana-2.1`` showed an id can take. It is
    dropped; a chat model's display name drops nothing."""
    synthetic = [
        {
            "name": "models/gemini-3.9-flash-n",
            "displayName": "Nano Banana 3",
            "supportedGenerationMethods": ["generateContent"],
        },
        _GOOGLE_IMAGE_BRAND_MODELS[-1],  # gemini-3.5-flash, 'Gemini 3.5 Flash'
    ]
    result, _seen = await _discover(
        "google",
        _GOOGLE_BASE,
        {f"{_GOOGLE_BASE}/models": httpx.Response(200, json={"models": synthetic})},
        api_key="AIza-test-google-key",
    )

    assert result.status == "success"
    assert [model.model_id for model in result.models] == ["gemini-3.5-flash"]


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
