"""TASK-32533: the quick popover mounts when its provider draft is not an option.

A fresh no-provider profile can hand ``ConsoleModelPopover`` a draft whose
``settings.provider`` is empty, or a key that ``_provider_select_options()``
does not list. Textual's ``Select`` raises ``InvalidSelectValueError`` from
``_validate_value`` for any initial value outside its options, the widget's
``_pre_process`` forwards that to ``App._handle_exception``, and the whole app
exited (critique #3, assessor A, capture 14). The model select four lines down
was already guarded (TASK-16502); these tests pin the same guard on the
provider select.
"""

from __future__ import annotations

import pytest
from textual.widgets import Select

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

pytestmark = pytest.mark.asyncio


def _popover(provider: str) -> ConsoleModelPopover:
    settings = ConsoleSessionSettings(provider=provider, model=None)
    origin = ConsoleSettingsOrigin("session-a", None, 0)
    draft = ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=origin.session_id,
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )

    return ConsoleModelPopover(
        origin=origin,
        app_config={"api_settings": {"llama_cpp": {}}},
        initial_draft=draft,
        providers_models={"llama_cpp": ["model-a"]},
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=lambda state, **_kwargs: state,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: ConsoleSettingsReadiness(
            "Ready", "Ready.", True
        ),
    )


class _PopoverHarness(ConsolidatedCSSApp):
    """Push one popover over an otherwise empty app with the real CSS stack."""

    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, popover: ConsoleModelPopover) -> None:
        super().__init__()
        self._popover = popover

    async def on_mount(self) -> None:
        await self.push_screen(self._popover)


async def _assert_mounts_with_a_blank_provider(provider: str) -> None:
    app = _PopoverHarness(_popover(provider))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(200):
            if isinstance(app.screen, ConsoleModelPopover) and app.screen.query(
                "#console-popover-provider"
            ):
                break
            await pilot.pause(0.01)
        await pilot.pause()

        assert app.is_running, "the app stopped while mounting the popover"
        assert app._exception is None, f"popover mount raised {app._exception!r}"
        provider_select = app.screen.query_one("#console-popover-provider", Select)
        assert provider_select.value is Select.NULL
        # The membership guard must not have quietly dropped the real options.
        assert "llama_cpp" in {value for _, value in provider_select._options}


async def test_popover_mounts_with_an_empty_provider_draft() -> None:
    """A no-provider draft (provider == '') mounts with a blank provider select."""
    await _assert_mounts_with_a_blank_provider("")


async def test_popover_mounts_with_a_provider_absent_from_its_options() -> None:
    """A provider the option builder does not list mounts blank, not dead."""
    await _assert_mounts_with_a_blank_provider("zq-not-a-provider")
