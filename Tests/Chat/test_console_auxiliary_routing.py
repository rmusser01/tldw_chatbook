"""TASK-26024: route side tasks (compaction) to a cheaper auxiliary model.

The Console's one auxiliary LLM call is compaction (titling here is pure
string truncation, no model). These pin the pure routing decisions: build
an override selection only when configured, and fall back to the main
resolution when the auxiliary is unconfigured or not ready.
"""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Chat.console_auxiliary_routing import (
    auxiliary_selection_from_config,
    select_auxiliary_or_main,
)


@dataclass
class _Selection:
    provider: str
    explicit_model: str | None = None
    configured_model: str | None = None
    base_url: str | None = "https://main.example"


@dataclass
class _Resolution:
    ready: bool
    provider: str = ""
    model: str = ""


def test_no_auxiliary_configured_returns_none():
    """AC#2: unconfigured => today's behavior (the caller keeps the main)."""
    main = _Selection(provider="anthropic", explicit_model="claude-opus-5")
    assert auxiliary_selection_from_config(main, provider=None, model=None) is None
    assert auxiliary_selection_from_config(main, provider="", model="  ") is None


def test_auxiliary_overrides_provider_and_model_but_keeps_the_rest():
    main = _Selection(
        provider="anthropic",
        explicit_model="claude-opus-5",
        configured_model="claude-opus-5",
    )
    aux = auxiliary_selection_from_config(
        main, provider="anthropic", model="claude-haiku-4-5-20251001"
    )
    assert aux is not None
    assert aux.provider == "anthropic"
    assert aux.explicit_model == "claude-haiku-4-5-20251001"
    assert aux.configured_model == "claude-haiku-4-5-20251001"
    # a cross-provider auxiliary drops the main's base_url so the new
    # provider's own endpoint resolves
    cross = auxiliary_selection_from_config(main, provider="groq", model="llama-x")
    assert cross.provider == "groq"
    assert cross.base_url is None


def test_model_only_config_keeps_the_main_provider():
    main = _Selection(provider="anthropic", explicit_model="claude-opus-5")
    aux = auxiliary_selection_from_config(main, provider=None, model="claude-haiku")
    assert aux is not None
    assert aux.provider == "anthropic"
    assert aux.explicit_model == "claude-haiku"


def test_fallback_uses_main_when_auxiliary_not_ready():
    """AC#3."""
    main_res = _Resolution(ready=True, provider="anthropic", model="opus")
    ready_aux = _Resolution(ready=True, provider="anthropic", model="haiku")
    unready_aux = _Resolution(ready=False)

    assert select_auxiliary_or_main(ready_aux, main_res) is ready_aux
    assert select_auxiliary_or_main(unready_aux, main_res) is main_res
    assert select_auxiliary_or_main(None, main_res) is main_res


def test_auxiliary_routing_never_touches_the_send_path():
    """AC#5 (structural): the auxiliary resolver is only ever called from
    compaction/summarize methods, never a user-visible chat send. A source
    scan is the honest pin -- the routing lives in the controller and the
    send dispatch must not reference it."""
    import inspect

    from tldw_chatbook.Chat import console_chat_controller as mod

    source = inspect.getsource(mod)
    # every call site of the auxiliary resolver
    call_lines = [
        line.strip()
        for line in source.splitlines()
        if "_auxiliary_compaction_resolution(" in line
        and "def _auxiliary_compaction_resolution" not in line
    ]
    assert call_lines, "the resolver must be wired somewhere"
    # the send path methods must not appear as the enclosing context of any
    # call -- assert the resolver is never referenced inside _run_agent_reply
    send_method = source[source.index("async def _run_agent_reply") :]
    send_method = send_method[: send_method.index("\n    async def ", 10)]
    assert "_auxiliary_compaction_resolution" not in send_method
