"""Repeated status projection keeps fresh owners without unchanged layout writes."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_polling_static_update_idempotence import observe_updates

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "context", "default"])
async def test_recovery_rail_skips_same_text_and_keeps_current_retry_owner(
    monkeypatch, kind
):
    from Tests.UI.test_console_rail_progress_timer import _ProgressRailHost
    from Tests.UI.test_console_resize_reflow import _failed_default_state
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleSettingsComponent,
        ConsoleSettingsPersistenceFailure,
        ConsoleSettingsPolicyFailureLabel,
    )
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Chat.console_settings_defaults import (
        ConsoleDefaultDurabilityState,
        ConsoleDefaultSavePhase,
    )
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    class Host(_ProgressRailHost):
        CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    host = Host()
    failure = ConsoleSettingsPersistenceFailure(
        component=ConsoleSettingsComponent.CONTEXT_POLICY,
        revision=2,
        persisted_conversation_id="conv-a",
        conversation_binding_revision=1,
        policy_failure_label=ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
    )
    failures = {failure.component: failure} if kind == "context" else {}
    default = (
        _failed_default_state(ConsoleDefaultSavePhase.BEFORE_REPLACE)
        if kind == "default"
        else ConsoleDefaultDurabilityState()
    )
    async with host.run_test(size=(120, 45)) as pilot:
        rail = host.query_one(ConsoleLeftRail)
        rail.sync_model_recovery(
            session_id="a", failures=failures, default_state=default
        )
        await pilot.pause()
        expected_title = "Model ⚠" if kind != "empty" else "Model"
        assert (
            str(rail.query_one("#console-rail-section-title-model", Static).renderable)
            == expected_title
        ), "Recovery warning was lost during layout"
        calls = observe_updates(
            monkeypatch,
            {
                "console-rail-section-title-model",
                "console-context-recovery-copy",
                "console-default-recovery-copy",
            },
        )
        for _ in range(8):
            rail.sync_model_recovery(
                session_id="a", failures=failures, default_state=default
            )
        assert calls == [], f"Unchanged recovery repaints: {calls}"
        if failures:
            failures = {failure.component: replace(failure, revision=3)}
        if default.recovery_intent is not None:
            default = replace(
                default,
                newest_intent_generation=8,
                recovery_intent=replace(default.recovery_intent, generation=8),
            )
        rail.sync_model_recovery(
            session_id="b", failures=failures, default_state=default
        )
        if failures:
            retry = rail.query_one("#console-retry-context-settings", Button)
            assert retry.console_settings_session_id == "b"
            assert retry.console_settings_revision == 3
        if default.recovery_intent is not None:
            assert (
                rail.query_one(
                    "#console-retry-default-save", Button
                ).console_default_intent_generation
                == 8
            )
        assert calls == [], "A fresh action owner with identical copy must not repaint"
        # Genuine warning changes still update the title and copy.
        failures = {failure.component: failure} if kind == "empty" else {}
        rail.sync_model_recovery(
            session_id="b",
            failures=failures,
            default_state=ConsoleDefaultDurabilityState(),
        )
        assert str(
            rail.query_one("#console-rail-section-title-model", Static).renderable
        ) == ("Model ⚠" if failures else "Model")
        assert "console-rail-section-title-model" in calls


@pytest.mark.asyncio
async def test_workspace_attention_accepts_new_generations_without_repainting_same_copy(
    monkeypatch,
):
    from Tests.UI.test_console_workspace_files_modal import _Host, _Inspector
    from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
        ConsoleWorkspaceFilesModal,
        WorkspaceFilesAttention,
        WorkspaceFilesBinding,
    )

    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),
        ),
    )
    attention = WorkspaceFilesAttention("Console has new activity")
    host = _Host()
    async with host.run_test(size=(120, 40)) as pilot:
        await host.push_screen(modal)
        await pilot.pause()
        assert modal.update_attention(attention, 1)
        calls = observe_updates(monkeypatch, {"console-workspace-files-attention"})
        for generation in range(2, 10):
            assert modal.update_attention(attention, generation)
        assert calls == [], f"Unchanged attention repaints: {calls}"
        changed_flags = replace(attention, has_failed_activity=True)
        assert modal.update_attention(changed_flags, 10)
        assert modal._attention is changed_flags
        assert modal._attention_generation == 10
        assert not modal.update_attention(WorkspaceFilesAttention("stale"), 9)
        assert calls == []
        assert modal.update_attention(
            WorkspaceFilesAttention("Console needs attention"), 11
        )
        assert (
            str(
                modal.query_one("#console-workspace-files-attention", Static).renderable
            )
            == "Console needs attention"
        )
        assert calls == ["console-workspace-files-attention"]


@pytest.mark.asyncio
async def test_llama_status_poll_keeps_same_copy_and_paints_verified_connection(
    monkeypatch, tmp_path
):
    from Tests.UI.test_llamacpp_setup_view import Harness
    from tldw_chatbook.LLM_Management.llamacpp_connection import LlamaCppProbeResult
    from tldw_chatbook.UI.LLM_Management import llamacpp_setup_view as module

    async def probe(request, **kwargs):
        return LlamaCppProbeResult(request, "ready", ("org/model",), "org/model")

    monkeypatch.setattr(module, "probe_llamacpp_target", probe)
    host = Harness(tmp_path / "profiles.json")
    async with host.run_test() as pilot:
        await pilot.pause()
        view = host.query_one(module.LlamaCppSetupView)
        view.refresh_state()
        calls = observe_updates(
            monkeypatch,
            {
                "llamacpp-preview-title",
                "llamacpp-connection-status",
                "llamacpp-bind-status",
            },
        )
        for _ in range(8):
            view.refresh_state()
        assert calls == [], f"Unchanged llama status repaints: {calls}"
        view.query_one("#llamacpp-existing-url", Input).value = "http://127.0.0.1:8181"
        await pilot.pause()
        await view.check_connection()
        assert "verified" in str(
            view.query_one("#llamacpp-connection-status", Static).renderable
        )
        assert not view.query_one("#llamacpp-use-console", Button).disabled
        assert "llamacpp-connection-status" in calls
