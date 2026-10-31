"""TASK-33621.11: Inspect badge and Send strip come from codes, never copy.

Each case builds the real ``ConsoleSettingsReadiness`` (from a config, via
``build_console_settings_readiness``, or -- for evidence-only blockers -- the
validated dataclass itself), passes it through the real Inspector rows
(``ConsoleInspectorState.from_values``, fed as ChatScreen feeds it), the real
rail-state badge and the real pinned Run line, and asserts what renders.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_display_state import ConsoleInspectorState
from tldw_chatbook.Chat.console_rail_state import (
    build_console_inspector_rail_badge,
    console_badge_inspector_rows,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    build_console_readiness_presentation,
)

_KEY = {"api_settings": {"openai": {"api_key": "sk-u09readinesscodes0123456789ab"}}}
_NO_KEY = {"api_settings": {"openai": {"api_key_env_var": "ABSENT_U09_TEST_KEY"}}}

#: (scenario id, provider, model, config) -> the readiness blocker it yields.
READINESS_CASES = [
    ("ready", "openai", "gpt-4.1", _KEY, None),
    ("credential_missing", "openai", "gpt-4.1", _NO_KEY, "credential_missing"),
    ("model_missing", "openai", None, _KEY, "model_missing"),
    ("provider_missing", "", "gpt-4.1", _KEY, "provider_missing"),
    ("provider_unsupported", "no-such-provider", "x", _KEY, "provider_unsupported"),
]


def _readiness(provider, model, config):
    return build_console_settings_readiness(
        ConsoleSessionSettings(provider=provider, model=model),
        app_config=config,
        environ={},
    )


def _inspector(readiness, **values) -> ConsoleInspectorState:
    """The Inspector state, fed the provider facts as ChatScreen feeds them."""
    ready = readiness.operability == "ready_to_send"
    return ConsoleInspectorState.from_values(
        provider_label=readiness.provider_display_name or "Provider",
        model_label="gpt-4.1",
        provider_ready=ready,
        provider_recovery=""
        if ready
        else build_console_readiness_presentation(readiness).detail,
        **values,
    )


def _badge(state: ConsoleInspectorState, *, run_status=None) -> str:
    return build_console_inspector_rail_badge(
        run_status=run_status,
        inspector_rows=console_badge_inspector_rows(state),
        can_save_chatbook=state.can_save_chatbook,
    )


@pytest.mark.parametrize(
    ("provider", "model", "config", "blocker"),
    [case[1:] for case in READINESS_CASES],
    ids=[case[0] for case in READINESS_CASES],
)
def test_each_readiness_blocker_badges_setup_and_a_healthy_run_badges_running(
    provider, model, config, blocker
):
    """AC#1/#5: setup only for a real provider blocker; a run is "running"."""
    readiness = _readiness(provider, model, config)
    assert readiness.blocker == blocker
    idle = _inspector(readiness)
    running = _inspector(readiness, run_active=True)
    if blocker is None:
        assert _badge(idle) == ""
        assert project_console_send_authority(idle).run == "Ready"
        assert _badge(running, run_status="streaming") == "running"
        assert project_console_send_authority(running).run == "Running"
    else:
        assert _badge(idle) == "setup"
        assert _badge(running, run_status="streaming") == "setup"
        assert project_console_send_authority(idle).run == "Blocked"


def test_the_badge_reads_row_status_not_row_words():
    """AC#5: words in a row's copy never choose the badge.

    Two of two evidence references staged, one of them blocked: the
    Evidence row is ready ("Some evidence needs review: 1 blocked.") and the
    next send is not blocked. Dev matched "blocked" in that copy and badged
    the rail "blocked"; a row that merely says "failed" read "failed".
    """
    readiness = _readiness("openai", "gpt-4.1", _KEY)
    partial = _inspector(
        readiness,
        evidence_summary="1/2 available (available)",
        evidence_status="ready",
        evidence_recovery="Some evidence needs review: 1 blocked.",
    )
    assert _badge(partial) == ""
    worded = _inspector(
        readiness,
        evidence_summary="1/1 available (lookup failed once, retried)",
        evidence_status="ready",
    )
    assert _badge(worded) == ""

    # The row's STATUS still decides: no available evidence is blocked.
    none_available = _inspector(
        readiness,
        evidence_summary="0/1 available (blocked)",
        evidence_status="blocked",
        evidence_recovery="No available evidence. Review source authority before sending.",
    )
    assert _badge(none_available) == "blocked"


def test_an_uncoded_row_never_reads_as_provider_setup():
    """AC#5: "setup" comes from the Provider row's code and status only.

    Dev turned any row whose LABEL contained "model" or "provider" and whose
    copy contained "missing"/"blocked" into "setup".
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Row:
        label: str
        value: str = ""
        status: str = "ready"
        recovery: str = ""

    assert build_console_inspector_rail_badge(
        inspector_rows=(Row("Model", value="Missing", recovery="blocked"),)
    ) == ""
    assert build_console_inspector_rail_badge(
        inspector_rows=(Row("FAILED run", value="see transcript"),)
    ) == ""
