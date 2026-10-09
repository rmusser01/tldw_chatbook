"""Library RAG Answer resolves the persisted provider AND model (TASK-34000.21).

The 2026-10-02 Library UX review (L-06) ran RAG Answer with `[chat_defaults]`
set to gpt-4.1-mini and watched the paid call go to the OpenAI handler's own
default, a reasoning model nobody chose -- and, with `[chat_defaults]` set to
Anthropic, still to OpenAI. `resolve_library_rag_answer_provider` returned
`(default_api_endpoint, None)` by design, and the pre-run line could name
only the provider. These pins hold the new contract: the pair comes from
the persisted `[chat_defaults]` the way briefings already resolve it
(`resolve_remembered_provider_model`, never borrowing a model across
providers), and the line above Run names both halves.

Nothing here reaches a network: `load_settings` is replaced with a mapping
and no key is ever real.
"""

from __future__ import annotations

import pytest

from tldw_chatbook import config as app_config
from tldw_chatbook.Library.library_rag_answer_service import (
    library_rag_answer_provider_gate,
    resolve_library_rag_answer_provider,
)
from tldw_chatbook.Library.library_rag_state import (
    LibraryRagPanelState,
    LibraryRagQueryState,
    library_rag_paid_mode_notice,
)
from tldw_chatbook.Scheduling import automation_execution
from tldw_chatbook.Scheduling.automation_execution import resolve_execution_target
from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
    library_rag_query_quiet_text,
)

pytestmark = pytest.mark.unit


def _settings(
    *,
    provider: str | None,
    model: str | None,
    openai_model: str | None = "gpt-5.6-terra",
    anthropic_model: str | None = "claude-sonnet-5",
    openai_key: str | None = "sk-test-openai-key",
    anthropic_key: str | None = "sk-ant-test-anthropic-key",
) -> dict:
    """A persisted-config mapping in the harness `launch.sh` spelling.

    `[chat_defaults] provider` is written as the display alias (`OpenAI`,
    `Anthropic`) exactly as First Run and Settings persist it; the provider
    tables carry their OWN remembered models so a cross-provider borrow has
    something to borrow and the negative pins can catch it.
    """
    chat_defaults: dict = {}
    if provider is not None:
        chat_defaults["provider"] = provider
    if model is not None:
        chat_defaults["model"] = model
    openai: dict = {}
    anthropic: dict = {}
    if openai_model is not None:
        openai["model"] = openai_model
    if openai_key is not None:
        openai["api_key"] = openai_key
    if anthropic_model is not None:
        anthropic["model"] = anthropic_model
    if anthropic_key is not None:
        anthropic["api_key"] = anthropic_key
    return {
        "chat_defaults": chat_defaults,
        "api_settings": {"openai": openai, "anthropic": anthropic},
    }


@pytest.fixture
def persisted(monkeypatch):
    """Install one persisted-config mapping behind `app_config.load_settings`.

    `default_api_endpoint` is pinned to `openai` -- the loader's own fallback
    when `[llm_api_settings] default_api` is absent, and the value every
    review capture showed -- so a test that still resolves `openai` under an
    Anthropic `[chat_defaults]` is reading the wrong source, not a blank one.
    """

    def _install(settings: dict, *, endpoint: str | None = "openai") -> dict:
        monkeypatch.setattr(app_config, "load_settings", lambda *a, **k: settings)
        monkeypatch.setattr(app_config, "default_api_endpoint", endpoint, raising=False)
        return settings

    return _install


def test_resolves_openai_and_its_model_from_chat_defaults(persisted):
    """AC#1, AC#5: the openai/gpt-4.1-mini pair the review's personas had set."""
    persisted(_settings(provider="OpenAI", model="gpt-4.1-mini"))

    assert resolve_library_rag_answer_provider() == ("openai", "gpt-4.1-mini")


def test_resolves_anthropic_pair_and_never_borrows_openais_model(persisted):
    """AC#2, AC#5: `[chat_defaults]` Anthropic beats the `openai` endpoint
    fallback, and a blank Anthropic model falls back to ANTHROPIC's own
    remembered model -- never to the OpenAI table's."""
    persisted(_settings(provider="Anthropic", model="claude-haiku-4-5"))
    assert resolve_library_rag_answer_provider() == ("anthropic", "claude-haiku-4-5")

    persisted(_settings(provider="Anthropic", model=""))
    provider, model = resolve_library_rag_answer_provider()
    assert (provider, model) == ("anthropic", "claude-sonnet-5")
    assert model != "gpt-5.6-terra"


def test_falls_back_to_the_default_endpoint_and_its_own_model(persisted):
    """`[chat_defaults]` naming no usable provider keeps today's endpoint
    fallback, now with that endpoint's own remembered model; nothing named
    anywhere is still `(None, None)`."""
    persisted(_settings(provider=None, model=None))
    assert resolve_library_rag_answer_provider() == ("openai", "gpt-5.6-terra")

    persisted(_settings(provider="not-a-provider", model="x"), endpoint="anthropic")
    assert resolve_library_rag_answer_provider() == ("anthropic", "claude-sonnet-5")

    persisted(_settings(provider=None, model=None, openai_model=None))
    assert resolve_library_rag_answer_provider() == ("openai", None)

    persisted(_settings(provider=None, model=None), endpoint="")
    assert resolve_library_rag_answer_provider() == (None, None)


def test_pre_run_line_names_provider_and_model(persisted):
    """AC#3, AC#5: the quiet line above Run names both halves, with the
    footer's own `·` joiner and the RAW provider key the in-flight line and
    the footer also print."""
    assert (
        library_rag_paid_mode_notice("openai", "gpt-4.1-mini")
        == "To openai · gpt-4.1-mini: question + evidence"
    )
    assert (
        library_rag_paid_mode_notice("openai")
        == "To openai · its default model: question + evidence"
    )

    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What changed?",
        mode="rag",
        provider_name="openai",
        provider_model="gpt-4.1-mini",
    )
    assert state.query_state.ready_answer_provider == "openai"
    assert state.query_state.ready_answer_model == "gpt-4.1-mini"
    assert (
        library_rag_query_quiet_text(state)
        == "To openai · gpt-4.1-mini: question + evidence"
    )

    # The gate's whole pair reaches the line unchanged.
    persisted(_settings(provider="Anthropic", model="claude-haiku-4-5"))
    gate = library_rag_answer_provider_gate()
    assert (gate.provider, gate.model) == ("anthropic", "claude-haiku-4-5")
    live = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What changed?",
        mode="rag",
        provider_name=gate.provider,
        provider_model=gate.model,
    )
    assert (
        library_rag_query_quiet_text(live)
        == "To anthropic · claude-haiku-4-5: question + evidence"
    )


def test_search_mode_never_names_a_model() -> None:
    """Keyword Search calls no provider: the reserved row stays empty and
    `ready_answer_model` is derived under the same rag-and-ready condition
    as `ready_answer_provider`, so the two cannot disagree."""
    state = LibraryRagQueryState.from_values(
        query="tides",
        mode="search",
        provider_name="openai",
        provider_model="gpt-4.1-mini",
    )
    assert state.ready_answer_provider == ""
    assert state.ready_answer_model == ""


def test_scheduled_fallback_forwards_the_resolved_model(persisted, monkeypatch):
    """AC#4, un-mocked: an automation that sets no provider or model of its
    own resolves the SAME pair interactive RAG Answer would bill."""
    persisted(_settings(provider="Anthropic", model="claude-haiku-4-5"))
    monkeypatch.setattr(automation_execution, "get_cli_setting", lambda *a, **k: None)

    target = resolve_execution_target({"input": {"question": "q"}})

    assert (target["provider"], target["model"]) == ("anthropic", "claude-haiku-4-5")


def test_gate_blocks_a_configured_provider_without_a_key_and_names_no_model(
    persisted, monkeypatch
):
    """Review Focus 1: a provider configured but with no key anywhere is
    blocked with the credential remedy, and the panel never names a model
    for a provider it cannot bill."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    persisted(
        _settings(provider="OpenAI", model="gpt-4.1-mini", openai_key=None)
    )

    gate = library_rag_answer_provider_gate()

    assert gate.provider is None
    assert gate.credential_recovery
    assert gate.model == "gpt-4.1-mini"
    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What changed?",
        mode="rag",
        provider_name=gate.provider,
        provider_model=gate.model,
        provider_credential_recovery=gate.credential_recovery,
    )
    assert state.query_state.ready_answer_provider == ""
    assert state.query_state.ready_answer_model == ""
    assert not state.query_state.run_action.enabled
    assert library_rag_query_quiet_text(state) == ""
