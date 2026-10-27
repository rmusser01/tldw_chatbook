"""TASK-34100.2 AC#2 (new-cross-cutting-01): Ready means a turn can be sent.

The first-run review found the Console header, Settings and the model
switcher all saying 'Ready · not tested' for a model the send preflight
refused every time ('This request cannot fit the selected model'), and still
'Ready' after a send failed 401 (a bad key) or 404 (a retired model). The
shared verdict now carries one capacity blocker computed from the same inputs
as the preflight, and the current model's last real send outcome.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tldw_chatbook.Chat import console_session_settings as session_settings
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SEND_OUTCOMES,
    ConsoleSessionSettings,
    build_console_settings_readiness,
    model_readiness_words,
    readiness_words,
    verdict_readiness_words,
)
from tldw_chatbook.Chat.provider_endpoint_contract import canonical_connection_identity
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderReadinessSnapshot,
    ProviderTestEvidence,
    connection_credential_revision,
)

KEY = "sk-test-first-chat-readiness"
OPENAI = {"api_settings": {"openai": {"api_key": KEY}}}
LLAMA = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
SEEN = datetime(2026, 10, 3, 9, 7).astimezone()


@pytest.fixture(autouse=True)
def _fresh_ledger():
    CONSOLE_SEND_OUTCOMES.clear()
    yield
    CONSOLE_SEND_OUTCOMES.clear()


def _readiness(provider="openai", model="gpt-4o", config=OPENAI, max_tokens=None, **kw):
    return build_console_settings_readiness(
        ConsoleSessionSettings(provider=provider, model=model, max_tokens=max_tokens),
        app_config=config,
        environ={},
        **kw,
    )


# --- one capacity blocker ----------------------------------------------------


def test_a_reservation_that_fills_a_known_window_reads_not_ready_with_a_fix() -> None:
    # gpt-4o's catalog window is 128,000 tokens; a 128,000-token reply
    # reservation leaves the send preflight no input capacity.
    readiness = _readiness(max_tokens=128_000)

    assert readiness.operability == "not_ready"
    assert readiness.blocker == "capacity_exhausted"
    assert readiness.recovery_action == "adjust_model_limits"
    assert readiness_words(readiness) == "Not ready · window too small"
    assert "Lower the response maximum or choose another model" in readiness.detail


def test_a_reservation_that_fits_stays_ready() -> None:
    assert readiness_words(_readiness(max_tokens=4096)) == "Ready · not tested"


def test_an_estimated_window_is_never_pre_blocked() -> None:
    # Unknown model: the window is only estimated (32,000), so the reply
    # reservation is capped (AC#1) and the model stays sendable.
    readiness = _readiness(model="gpt-9-unlisted", max_tokens=32_000)

    assert readiness.operability == "ready_to_send"


def test_a_window_the_server_reported_is_what_readiness_checks() -> None:
    # A local server said n_ctx = 2048; a 4,096-token reservation cannot fit.
    CONSOLE_SEND_OUTCOMES.record_window("llama_cpp", "qwen-small", 2048, True)

    readiness = _readiness(
        provider="llama_cpp", model="qwen-small", config=LLAMA, max_tokens=4096
    )

    assert readiness.blocker == "capacity_exhausted"


def _llama_blocker() -> str | None:
    return _readiness(
        provider="llama_cpp", model="qwen-small", config=LLAMA, max_tokens=4096
    ).blocker


def test_a_server_window_stands_only_under_the_settings_it_was_seen_with(
    monkeypatch,
) -> None:
    """Review round 2 (R2-3): the window a server reported never expired.
    capacity_exhausted refuses at the composer, so no send could refresh it:
    pointing the provider at another server with the same model id left the
    chat 'window too small' until a restart. Review round 1 of pass 2 (F8):
    an unrelated save keeps it -- only the provider's own settings (or the
    gateway TTL below) retire it."""
    _publish(monkeypatch, LLAMA)
    CONSOLE_SEND_OUTCOMES.record_window("llama_cpp", "qwen-small", 4096, True)
    assert _llama_blocker() == "capacity_exhausted"

    _publish(monkeypatch, {**LLAMA, "library": {"rail_state": {"lifecycle": "x"}}})
    assert _llama_blocker() == "capacity_exhausted"

    _publish(
        monkeypatch,
        {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9431"}}},
    )
    assert _llama_blocker() is None


def test_a_server_window_expires_with_the_gateway_metadata_cache(monkeypatch) -> None:
    """The server was restarted with -c 32768. The gateway re-reads a
    server's window after ``METADATA_SUCCESS_TTL``; readiness's copy must not
    outlive it, or the composer refuses the send that would re-read it."""
    from tldw_chatbook.Chat.console_context_window import METADATA_SUCCESS_TTL

    now = [1_000.0]
    monkeypatch.setattr(CONSOLE_SEND_OUTCOMES, "_clock", lambda: now[0])
    CONSOLE_SEND_OUTCOMES.record_window("llama_cpp", "qwen-small", 4096, True)

    now[0] += METADATA_SUCCESS_TTL - 1
    assert _llama_blocker() == "capacity_exhausted"

    now[0] += 2
    assert _llama_blocker() is None


def test_readiness_keeps_a_server_window_exactly_as_long_as_the_gateway() -> None:
    """Structural pin (two constants agree, not behavioural evidence): the
    window's expiry is covered by the TTL expiry test."""
    from tldw_chatbook.Chat.console_context_window import METADATA_SUCCESS_TTL

    assert session_settings.OBSERVED_WINDOW_TTL_SECONDS == METADATA_SUCCESS_TTL


def test_a_server_window_belongs_to_the_endpoint_that_reported_it() -> None:
    """Pass 2 (final review, major 3): the window was kept per provider and
    model only, so a chat pointed at another server serving the same model
    id inherited the first server's 'window too small' with no config save
    to clear it. It now stands only for the endpoint it was reported by."""
    CONSOLE_SEND_OUTCOMES.record_window(
        "llama_cpp", "qwen-small", 4096, True, endpoint="http://127.0.0.1:9099"
    )
    assert _llama_blocker() == "capacity_exhausted"  # LLAMA sends to :9099

    other_server = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="qwen-small",
            max_tokens=4096,
            base_url="http://127.0.0.1:9431",
        ),
        app_config=LLAMA,
        environ={},
    )

    assert other_server.blocker != "capacity_exhausted"
    # The same endpoint spelled with a trailing slash is the same server.
    assert CONSOLE_SEND_OUTCOMES.window_for(
        "llama_cpp", "qwen-small", "http://127.0.0.1:9099/"
    ) == (4096, True)


def test_a_larger_server_report_unblocks_readiness() -> None:
    CONSOLE_SEND_OUTCOMES.record_window("llama_cpp", "qwen-small", 4096, True)
    assert _llama_blocker() == "capacity_exhausted"

    CONSOLE_SEND_OUTCOMES.record_window("llama_cpp", "qwen-small", 32_768, True)

    assert _llama_blocker() is None


@pytest.mark.asyncio
async def test_every_send_path_hands_the_server_window_to_readiness(
    monkeypatch,
) -> None:
    """Review round 2 (R2-3, live g2-r2-llama/04-05): the window was recorded
    only in ``prepare_chat_request``, which the default Agents send path
    calls only with trace capture on. With llama.cpp at -c 4096 under the
    shipped 4,096 max_tokens the header read 'Ready · not tested' while every
    send came back 400 from the server. ``resolve_for_send`` -- shared by
    every path -- now hands readiness the server's cached window."""
    import asyncio

    import httpx

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )
    from tldw_chatbook.Utils import egress

    # The real URL policy, with isolated settings (test_console_context_window).
    monkeypatch.setattr(egress, "_config_enabled", lambda: True)
    monkeypatch.setattr(egress, "_config_allowed_hosts", frozenset)

    async def handler(request):
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 4096}}
            )
        return httpx.Response(200, json={"data": [{"id": "qwen-small"}]})

    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        # The endpoint LLAMA's readiness sends to: a window belongs to the
        # endpoint that reported it (pass 2).
        base_url="http://127.0.0.1:9099",
        explicit_model="qwen-small",
        max_tokens=4096,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        await gateway.resolve_for_send(selection)
        await asyncio.gather(
            *set(gateway._context_window_refreshes),
            *list(gateway._reasoning_metadata_refreshes.values()),
        )
        assert _llama_blocker() is None  # no server report yet: estimated
        second = await gateway.resolve_for_send(selection)

    assert (second.context_window.tokens, second.context_window.verified) == (4096, True)
    assert _llama_blocker() == "capacity_exhausted"


# --- the last real send ------------------------------------------------------


def test_a_401_on_send_takes_the_key_out_of_ready_until_a_reply_arrives() -> None:
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, 401)

    rejected = _readiness()
    assert rejected.blocker == "key_rejected_on_send"
    assert rejected.recovery_action == "configure_credential"
    assert readiness_words(rejected) == "Not ready · key rejected"
    # The key, not the model, was refused: every model of that key is out.
    assert _readiness(model="gpt-4o-mini").blocker == "key_rejected_on_send"

    CONSOLE_SEND_OUTCOMES.record_success("openai", "gpt-4o", KEY)
    assert _readiness().operability == "ready_to_send"


def test_a_new_key_starts_clean() -> None:
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", "sk-old-rejected", 401)

    assert _readiness().operability == "ready_to_send"


def test_a_404_on_send_takes_only_that_model_out_of_ready() -> None:
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, 404)

    lost = _readiness()
    assert lost.blocker == "model_unavailable"
    assert lost.recovery_action == "select_model"
    assert readiness_words(lost) == "Not ready · model not found"
    assert _readiness(model="gpt-4o-mini").operability == "ready_to_send"


def _publish(monkeypatch, published: dict) -> None:
    """A config publication: a new generation and the settings it installed."""
    import tldw_chatbook.config as config_module

    generation = config_module.get_runtime_config_generation() + 1
    monkeypatch.setattr(config_module, "get_runtime_config_generation", lambda: generation)
    monkeypatch.setattr(config_module, "settings", published)


_UNRELATED = {
    "api_settings": {"openai": {"api_key": KEY, "model": "gpt-4o"}},
    "console": {"onboarding": {"first_send_completed": False}},
}


def test_saving_this_providers_settings_clears_a_send_outcome(monkeypatch) -> None:
    _publish(monkeypatch, _UNRELATED)
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, 404)
    assert _readiness().blocker == "model_unavailable"

    _publish(
        monkeypatch,
        {
            "api_settings": {
                "openai": {
                    "api_key": KEY,
                    "model": "gpt-4o",
                    "api_base_url": "https://gateway.example/v1",
                }
            }
        },
    )

    assert _readiness().operability == "ready_to_send"


@pytest.mark.parametrize(
    "unrelated",
    [
        # The first reply's onboarding flag (g2b-v-oai, 13:09:36) ...
        {"console": {"onboarding": {"first_send_completed": True}}},
        # ... the first Library visit (g2b-v-ant/09-11) ...
        {"library": {"rail_state": {"lifecycle": "graduated"}}},
        # ... and another provider's settings.
        {"api_settings": {"anthropic": {"model": "claude-sonnet-5"}}},
    ],
)
def test_an_unrelated_save_keeps_a_send_outcome(monkeypatch, unrelated) -> None:
    """Review round 1 (F8, live F2): the 401/404 marks were keyed to the
    GLOBAL config generation, which every save bumps. Live, one click on
    Library saved its rail state and the header went from 'Not ready ·
    model not found' back to 'Ready · not tested' with nothing about the
    provider, key or model changed."""
    _publish(monkeypatch, _UNRELATED)
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, 404)
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", "sk-other-bad-key", 401)

    published = {
        **_UNRELATED,
        **unrelated,
        "api_settings": {
            **_UNRELATED["api_settings"],
            **unrelated.get("api_settings", {}),
        },
    }
    _publish(monkeypatch, published)

    assert _readiness().blocker == "model_unavailable"
    other_key = {"api_settings": {"openai": {"api_key": "sk-other-bad-key"}}}
    assert _readiness(config=other_key).blocker == "key_rejected_on_send"


def test_a_provider_with_no_readable_table_falls_back_to_any_save(
    monkeypatch,
) -> None:
    """Review round 2 (R3-6): a readiness key whose settings lookup comes
    back empty ('mistral', 'custom_hosted') fingerprinted to a constant, so
    its 401/404 mark could never be cleared by a save -- only a reply or a
    restart. With nothing to fingerprint, any save retires the mark (the
    documented fallback)."""
    _publish(monkeypatch, _UNRELATED)
    CONSOLE_SEND_OUTCOMES.record_status("mistral", "mistral-large-latest", KEY, 404)
    mistral = {"api_settings": {"mistral": {"api_key": KEY}}}
    lost = _readiness(provider="mistral", model="mistral-large-latest", config=mistral)
    assert lost.blocker == "model_unavailable"

    _publish(monkeypatch, {**_UNRELATED, "console": {"onboarding": {"x": 1}}})

    back = _readiness(provider="mistral", model="mistral-large-latest", config=mistral)
    assert back.operability == "ready_to_send"


def test_other_statuses_never_block() -> None:
    for status in (400, 403, 429, 500, 503):
        CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, status)

    assert _readiness().operability == "ready_to_send"


def test_a_403_on_one_model_never_rejects_the_key() -> None:
    """Review round 1 (F1): only a 401 rejects a key (owner ruling 2026-10-02,
    PR #2958). A 403 is often about one message or model -- OpenRouter's
    moderation, a model permission, a region limit -- so it must not take
    every model of the key out of Ready."""
    CONSOLE_SEND_OUTCOMES.record_status("openrouter", "meta-llama/guarded", KEY, 403)
    openrouter = {"api_settings": {"openrouter": {"api_key": KEY}}}

    other = _readiness(provider="openrouter", model="openai/gpt-4.1-mini", config=openrouter)
    same = _readiness(provider="openrouter", model="meta-llama/guarded", config=openrouter)

    assert other.operability == "ready_to_send"
    assert same.operability == "ready_to_send"


def test_a_saved_context_window_reaches_readiness_without_a_restart(monkeypatch) -> None:
    """Review round 1 (F3): readiness memoized the catalog window for the
    whole process, so a window saved in Settings (which reloads the model
    catalog) never reached the header until a restart."""
    import tldw_chatbook.model_capabilities as capabilities_module

    class _Catalog:
        def __init__(self, window: int | None) -> None:
            self.window = window

        def get_model_capabilities(self, provider: str, model: str) -> dict:
            return {"context_window": self.window} if self.window else {}

    monkeypatch.setattr(capabilities_module, "_global_capabilities", _Catalog(None))
    assert _readiness(model="my-model", max_tokens=4096).operability == "ready_to_send"

    # Settings ▸ Providers & Models saves a 2,048-token window, then reloads.
    capabilities_module.reload_capabilities()
    monkeypatch.setattr(capabilities_module, "_global_capabilities", _Catalog(2048))

    assert _readiness(model="my-model", max_tokens=4096).blocker == "capacity_exhausted"


# --- a past send's outcome never locks a first chat (review round 2, R2-2) ---


def _first_chat_card(readiness):
    from tldw_chatbook.Chat.console_onboarding_state import (
        build_console_setup_card_state,
        console_setup_is_blocking,
    )

    card = build_console_setup_card_state(
        readiness=readiness,
        provider_label=readiness.provider_display_name,
        has_model=True,
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    blocking = console_setup_is_blocking(
        readiness=readiness, has_model=True, first_send_completed=False
    )
    return card, blocking


@pytest.mark.parametrize(
    ("status", "word"),
    [(401, "Not ready · key rejected"), (404, "Not ready · model not found")],
)
def test_a_failed_first_send_never_puts_the_setup_card_back_over_the_composer(
    status: int, word: str
) -> None:
    """The first send of a new profile got a 401 (a bad key) or a 404 (an
    Ollama model not pulled yet). The gate admits the retry, but the empty
    chat went back to the blocking Get started card ('Composer unlocks after
    setup'), so the composer could not take it. The card now says why the
    last send failed on its one line and leaves the composer usable."""
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, status)
    readiness = _readiness()
    assert readiness_words(readiness) == word

    card, blocking = _first_chat_card(readiness)

    assert card.mode == "ready_line"
    assert card.body_copy.startswith(word)
    assert readiness.detail in card.body_copy
    assert blocking is False


def test_a_window_too_small_explains_itself_without_locking_the_composer() -> None:
    """Pass 2 (final review, major 2): a zero-capacity model kept the
    blocking Get started card over an empty chat, so the composer could not
    even take the message while the user fixed the limit. The send gate
    already refuses that send with copy naming the fix ('Lower the response
    maximum or choose another model'), so the card says why on its one line
    -- next to the Fix model limits action -- and the composer stays usable.
    """
    readiness = _readiness(max_tokens=128_000)

    card, blocking = _first_chat_card(readiness)

    assert card.mode == "ready_line"
    assert card.body_copy.startswith("Not ready · window too small")
    assert "Lower the response maximum or choose another model" in card.body_copy
    assert blocking is False
    assert readiness.recovery_action == "adjust_model_limits"  # the fix action


# --- the switcher judges each model row --------------------------------------


def test_switcher_rows_judge_capacity_per_model() -> None:
    provider = _readiness(max_tokens=128_000)  # the representative row is blocked

    assert model_readiness_words(
        provider, "gpt-4o", provider="openai", max_tokens=128_000
    ) == (
        "Not ready · window too small"
    )
    assert model_readiness_words(
        provider, "gpt-4.1-2025-04-14", provider="openai", max_tokens=128_000
    ) == "Ready · not tested"


def test_a_provider_level_blocker_reads_the_same_on_every_row() -> None:
    provider = _readiness(config={"api_settings": {"openai": {}}})

    assert model_readiness_words(provider, "gpt-4o", provider="openai") == (
        "Not ready · no key"
    )


# --- 'Key verified', not 'Ready · verified', for a listing-only check ------
# One word on every surface (Settings' check, the Console header, the switcher).


def _accepted_listing() -> ProviderTestEvidence:
    identity = ProviderDraftIdentity(
        provider_key="openai",
        connection_identity=canonical_connection_identity(
            "openai", "https://api.openai.com/v1"
        ),
        credential_source="stored",
        credential_revision=connection_credential_revision(KEY),
        draft_generation=0,
    )
    return ProviderTestEvidence(
        identity,
        "reachable",
        ("gpt-4o",),
        credential="listing_accepted",
        observed_at=SEEN,
    )


def test_the_settings_key_check_reads_key_verified() -> None:
    evidence = _accepted_listing()
    word = verdict_readiness_words(
        ProviderReadinessSnapshot(
            configuration="configured", endpoint="reachable", model="confirmed"
        ),
        evidence,
    )

    assert word == "Key verified 09:07"


def test_the_console_says_the_key_was_verified_not_the_model() -> None:
    owner = ProviderConnectionEvidence()
    owner.publish(_accepted_listing(), order=1)

    readiness = _readiness(connection_evidence=owner)

    assert readiness_words(readiness) == "Key verified 09:07"


def test_every_new_blocker_has_a_reason_that_fits_the_switcher_column() -> None:
    for blocker in ("capacity_exhausted", "key_rejected_on_send", "model_unavailable"):
        reason = session_settings._NOT_READY_REASONS[blocker]
        assert len(f"Not ready · {reason}") <= 28


# --- the gateway records what a real send proved -----------------------------


def _send_resolution(model: str = "gpt-4o"):
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

    return ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model=model,
        ready=True,
        execution_key="openai",
        api_key=KEY,
        streaming=False,
    )


async def _send(response_or_error) -> None:
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    def call(**_kwargs):
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return response_or_error

    gateway = ConsoleProviderGateway(chat_api_call_fn=call)
    _ = [
        item
        async for item in gateway.stream_chat(
            _send_resolution(), [{"role": "user", "content": "Reply with OK"}]
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "blocker", "word"),
    [
        ("auth", "key_rejected_on_send", "Not ready · key rejected"),
        ("missing", "model_unavailable", "Not ready · model not found"),
    ],
)
async def test_a_failed_real_send_leaves_ready_and_a_reply_restores_it(
    error, blocker, word
) -> None:
    from tldw_chatbook.Chat.Chat_Deps import (
        ChatAPIError,
        ChatAuthenticationError,
        ChatBadRequestError,
    )

    exc = (
        ChatAuthenticationError("bad key", provider="openai")
        if error == "auth"
        else ChatBadRequestError("no such model", provider="openai", status_code=404)
    )
    assert readiness_words(_readiness()) == "Ready · not tested"

    with pytest.raises(ChatAPIError):
        await _send(exc)

    failed = _readiness()
    assert failed.blocker == blocker
    assert readiness_words(failed) == word

    await _send({"choices": [{"message": {"content": "OK"}}]})

    assert readiness_words(_readiness()) == "Ready · not tested"


@pytest.mark.asyncio
async def test_a_rate_limit_or_outage_does_not_leave_ready() -> None:
    from tldw_chatbook.Chat.Chat_Deps import ChatAPIError, ChatRateLimitError

    with pytest.raises(ChatAPIError):
        await _send(ChatRateLimitError("slow down", provider="openai"))

    assert _readiness().operability == "ready_to_send"


# --- Settings' Readiness row reads the same capacity blocker (AC#2) ----------


def test_settings_verdict_judges_the_model_it_tests() -> None:
    """Settings' Test result read 'Key verified 11:29' for gpt-4o while a
    200,000-token reply reservation filled its 128k window (live, g2-capacity
    05-08): the verdict now takes the Console's model-level blocker."""
    snapshot = ProviderReadinessSnapshot(
        configuration="configured", endpoint="reachable", model="confirmed"
    )
    evidence = _accepted_listing()

    blocked = verdict_readiness_words(
        snapshot, evidence, provider="openai", model="gpt-4o", max_tokens=200_000
    )
    fits = verdict_readiness_words(
        snapshot, evidence, provider="openai", model="gpt-4o", max_tokens=4_096
    )

    assert blocked == "Not ready · window too small"
    assert fits == "Key verified 09:07"


def test_settings_rows_lead_with_the_model_and_name_the_fix() -> None:
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    readiness = get_provider_readiness("openai", OPENAI, environ={})

    rows = SettingsScreen._provider_test_rows(
        readiness,
        display_name="OpenAI",
        model="gpt-4o",
        endpoint="",
        evidence=_accepted_listing(),
        max_tokens=200_000,
    )

    assert rows[0] == ("Readiness", "Not ready · window too small")
    label, text = rows[1]
    assert label == "Model"
    assert "replies fill its whole context window" in text
    assert "lower Max tokens or choose another model" in text


# --- the gateway feeds a server-reported window to readiness (review C6) -----


@pytest.mark.parametrize(
    ("source", "verified", "blocked"),
    [
        # A llama.cpp server said n_ctx = 2,048: readiness must see it.
        ("server metadata", True, True),
        # A catalog window is re-read from the catalog, never pinned by a send
        # (a window saved in Settings would otherwise be shadowed, review F3).
        ("model catalog", True, False),
    ],
)
def test_the_send_preflight_records_only_a_server_reported_window(
    source: str, verified: bool, blocked: bool
) -> None:
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_kwargs: None)
    gateway.prepare_chat_request(
        ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            model="qwen-small",
            ready=True,
            max_tokens=4096,
            context_window=ContextWindowResolution(2048, source, verified),
        ),
        [{"role": "user", "content": "Reply with OK"}],
    )

    readiness = _readiness(
        provider="llama_cpp", model="qwen-small", config=LLAMA, max_tokens=4096
    )
    assert (readiness.blocker == "capacity_exhausted") is blocked


# --- a key setup's check rejected reaches the verdict (review round 2, F4) ----

BAD_OPENROUTER_KEY = "sk-or-v1-review-round-2-rejected-key"
OPENROUTER = {"api_settings": {"openrouter": {"api_key": BAD_OPENROUTER_KEY}}}


def _openrouter_readiness(config=OPENROUTER):
    return _readiness(
        provider="openrouter", model="openai/gpt-4.1-mini", config=config
    )


def _bump_config_generation(monkeypatch) -> None:
    import tldw_chatbook.config as config_module

    generation = config_module.get_runtime_config_generation()
    monkeypatch.setattr(
        config_module, "get_runtime_config_generation", lambda: generation + 1
    )


async def _discover_openrouter(verdict: str) -> list[dict]:
    """Setup's discovery through the shared catalog service; the discovery
    client answers as the real one does for a key check (no network)."""
    import copy

    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryError,
        ModelDiscoveryResult,
    )

    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    sent: list[dict] = []

    async def client(**kwargs) -> ModelDiscoveryResult:
        sent.append(kwargs)
        if verdict == "rejected":
            return ModelDiscoveryResult(
                provider=kwargs["provider"],
                provider_list_key=kwargs["provider_list_key"],
                endpoint_fingerprint=None,
                status="error",
                error=ModelDiscoveryError(
                    kind="missing_credentials",
                    message="OpenRouter rejected this key: User not found.",
                    recovery_hint="Check the API key configured for this provider.",
                    category="unauthorized",
                    provider_reason="User not found.",
                ),
            )
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=None,
            status="success",
            key_verified=verdict == "verified",
        )

    service = LocalLLMProviderCatalogService(
        settings_loader=lambda: config,
        provider_catalog_loader=lambda: copy.deepcopy(config["providers"]),
        discovery_client=client,
        environ={},
    )
    await service.discover_models(
        provider="openrouter",
        staged_settings={
            "api_settings": {
                "openrouter": {
                    "api_key": BAD_OPENROUTER_KEY,
                    "credential_source": "draft",
                }
            }
        },
        use_shared_cache=False,
    )
    return sent


@pytest.mark.asyncio
async def test_a_key_setup_rejected_reads_key_rejected_in_the_console(
    monkeypatch,
) -> None:
    """Setup's authenticated check rejected an OpenRouter key and the Summary
    said so, but the Console header read 'Ready · not tested' until a send
    came back 401 (live, g2-v2-or/05-09): the check's verdict never reached
    the shared readiness verdict."""
    sent = await _discover_openrouter("rejected")
    assert sent and sent[0]["api_key"] == BAD_OPENROUTER_KEY
    _bump_config_generation(monkeypatch)  # setup's commit publishes the config

    readiness = _openrouter_readiness()

    assert readiness.blocker == "key_rejected_on_send"
    assert readiness_words(readiness) == "Not ready · key rejected"
    assert "when it was checked" in readiness.detail
    assert "last message" not in readiness.detail
    # Another key starts clean.
    other = {"api_settings": {"openrouter": {"api_key": "sk-or-v1-another-key"}}}
    assert _openrouter_readiness(other).operability == "ready_to_send"


@pytest.mark.asyncio
async def test_a_passing_key_check_clears_a_rejection() -> None:
    await _discover_openrouter("rejected")
    assert _openrouter_readiness().blocker == "key_rejected_on_send"

    await _discover_openrouter("listed-unverified")  # a public list proves nothing
    assert _openrouter_readiness().blocker == "key_rejected_on_send"

    await _discover_openrouter("verified")
    assert _openrouter_readiness().operability == "ready_to_send"


def test_a_reply_clears_a_key_check_rejection() -> None:
    CONSOLE_SEND_OUTCOMES.record_key_check("openrouter", BAD_OPENROUTER_KEY, rejected=True)
    assert _openrouter_readiness().blocker == "key_rejected_on_send"

    CONSOLE_SEND_OUTCOMES.record_success(
        "openrouter", "openai/gpt-4.1-mini", BAD_OPENROUTER_KEY
    )

    assert _openrouter_readiness().operability == "ready_to_send"


def test_a_blank_max_tokens_is_judged_with_the_value_it_inherits() -> None:
    """Review round 1 (F6): Chat settings' blank Max tokens (inherit) was
    judged against the Console default reservation and read 'Ready', while
    the applied chat inherits the provider table's 200,000 and the header
    then read 'Not ready · window too small' (g2b-cap/04-05)."""
    config = {
        "api_settings": {"openai": {"api_key": KEY, "max_tokens": 200_000}},
        "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
    }

    blank = _readiness(config=config, max_tokens=None)

    assert blank.blocker == "capacity_exhausted"
    assert "Replies reserve 128,000" in blank.detail or "200,000" in blank.detail
    # A value set in the draft is judged as typed.
    assert _readiness(config=config, max_tokens=4096).operability == "ready_to_send"


def test_a_blank_max_tokens_with_nothing_to_inherit_uses_the_default() -> None:
    config = {"api_settings": {"openai": {"api_key": KEY}}}

    assert _readiness(config=config, max_tokens=None).operability == "ready_to_send"
