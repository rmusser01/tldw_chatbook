"""Settings failure logs preserve correlation without serializing private state."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from loguru import logger

from Tests.Chat.test_console_display_name_fork_lifetime import _coordinator
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultMutationIntent,
    ConsoleDefaultMutationOutcome,
    ConsoleDefaultSavePhase,
)
from tldw_chatbook.UI.Console_Modules import settings_durability


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    ("conversation_persist", "default_reserve", "default_apply", "default_publish"),
)
async def test_coordinator_failure_logs_safe_context_and_keeps_recovery(
    monkeypatch, phase
):
    session_id, submission_id = str(uuid4()), uuid4().hex
    settings = ConsoleSessionSettings(
        provider="openai", model="private-model-canary", system_prompt="draft-canary"
    )
    policy = ConsoleContextPolicyOverrides()
    submission = ConsoleSettingsSubmission(
        submission_id=submission_id,
        action=ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        surface=ConsoleSettingsSurface.QUICK_POPOVER,
        origin=ConsoleSettingsOrigin(session_id, None, 0),
        draft=ConsoleSettingsDraftState(settings, policy, (), (), None),
        user_display_name_override="private-name-canary",
        default_field_mask=frozenset(),
    )
    committed = ConsoleSettingsCommittedSubmission(
        submission,
        ConsoleSettingsLiveCommit(
            submission_id, session_id, None, 0, 0, 0, settings, policy
        ),
    )
    intent = ConsoleDefaultMutationIntent(
        7,
        submission.action,
        "private-provider-canary",
        "private-model-canary",
        frozenset(),
        {},
        None,
    )
    outcome = ConsoleDefaultMutationOutcome(7, True, True, {}, None)
    events = []

    def fault(at):
        events.append(at)
        if at == phase:
            raise RuntimeError("credential-canary /private/path-canary draft-canary")

    async def persist(*args, **kwargs):
        fault("conversation_persist")

    async def reserve(*args):
        fault("default_reserve")
        return intent

    def apply(*args):
        fault("default_apply")
        return outcome

    async def publish(*args):
        fault("default_publish")
        return True

    coordinator = _coordinator(
        SimpleNamespace(persist_console_settings_commit_serialized=persist),
        recovery=lambda: events.append("recovery_sync"),
    )
    coordinator._reserve_console_default_intent_off_event_loop = reserve
    coordinator._publish_console_default_outcome_off_event_loop = publish
    failures = []
    coordinator._record_console_default_failure = lambda *args: failures.append(args)
    monkeypatch.setattr(settings_durability, "apply_console_default_intent", apply)
    records, rendered = [], []

    def capture(message):
        records.append(message.record)
        rendered.append(str(message))

    sink = logger.add(capture, level="ERROR", format="{message}")
    try:
        await coordinator._coordinate_console_settings_submission(committed, None)
    finally:
        logger.remove(sink)

    assert len(records) == 1
    text = "".join(rendered)
    assert "operation=console_settings" in text
    assert f"phase={phase}" in text
    assert "failure=RuntimeError" in text
    assert f"session_id={session_id}" in text
    assert f"submission_id={submission_id}" in text
    if phase in ("default_apply", "default_publish"):
        assert "generation=7" in text
    assert records[0]["exception"] is None
    assert "canary" not in text
    assert "recovery_sync" in events
    assert "conversation_persist" in events
    if phase == "default_reserve":
        assert "default_apply" not in events
    elif phase == "default_apply":
        assert failures == [(intent, ConsoleDefaultSavePhase.BEFORE_REPLACE)]
    elif phase == "default_publish":
        assert failures == [(intent, ConsoleDefaultSavePhase.CACHE_PUBLICATION)]
    else:
        assert "default_publish" in events


@pytest.mark.parametrize(
    "value",
    (
        "ordinary-label",
        "api_key-canary",
        "/private/canary",
        True,
        object(),
        "z" * 32,
        "f47ac10b-58cc-11cf-a447-001122334455",
        "FFFFFFFF-FFFF-4FFF-8FFF-FFFFFFFFFFFF",
    ),
)
def test_settings_diagnostic_rejects_nonopaque_ids_without_stringification(value):
    from tldw_chatbook.UI.Console_Modules.settings_diagnostics import (
        log_settings_failure,
    )

    class Unprintable:
        def __str__(self):
            raise AssertionError("private object must not be serialized")

        __repr__ = __str__

    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        log_settings_failure(
            "default_apply",
            RuntimeError(Unprintable()),
            session_id=value,
            submission_id=Unprintable(),
            generation=True,
        )
    finally:
        logger.remove(sink)
    text = "".join(lines)
    assert "session_id=invalid" in text
    assert "submission_id=invalid" in text
    assert "generation=invalid" in text
    assert "failure=RuntimeError" in text
    assert "canary" not in text
