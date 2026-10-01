"""TASK-32533: the quick popover mounts when its provider draft is not an option.

TASK-33004.4 replaced the provider Select with Switch model's pair list; the
tests below keep the mount and typed-id pins against the new surface.

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
from textual.widgets import Input

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


async def _open(app, pilot) -> None:
    for _ in range(200):
        if isinstance(app.screen, ConsoleModelPopover) and app.screen.query(
            "#console-popover-find"
        ):
            break
        await pilot.pause(0.01)
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _assert_mounts_with_a_blank_provider(provider: str) -> None:
    """Rewritten for TASK-33004.4: the provider Select is gone, so the crash
    it guarded (TASK-32533) cannot recur; the pin is now that Switch model
    mounts, lists pairs and applies nothing without one."""
    app = _PopoverHarness(_popover(provider))
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)

        assert app.is_running, "the app stopped while mounting the popover"
        assert app._exception is None, f"popover mount raised {app._exception!r}"
        rows = app.screen._rows
        assert ("pair", "llama_cpp", "model-a") in {row.key for row in rows}


async def test_blank_popover_survives_a_custom_model_id_keystroke() -> None:
    """TASK-32533 review, Important #1, kept for the typed-id path: a blank
    popover plus a typed model id must neither crash nor apply a pair with no
    provider (spec rule 1)."""
    app = _PopoverHarness(_popover(""))
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("g")
        await pilot.pause()

        assert app.is_running, "the popover took the app down on the update path"
        assert app._exception is None, f"the update path raised {app._exception!r}"
        # The key reached Find (so the typed-id path ran) and made no row.
        assert app.screen.query_one("#console-popover-find", Input).value == "g"
        assert all(row.kind != "typed" for row in app.screen._rows)
        assert isinstance(app.screen, ConsoleModelPopover)


async def test_popover_mounts_with_an_empty_provider_draft() -> None:
    """A no-provider draft (provider == '') mounts and lists ready pairs."""
    await _assert_mounts_with_a_blank_provider("")


async def test_popover_mounts_with_a_provider_absent_from_its_options() -> None:
    """A provider the option builder does not list mounts, not dead."""
    await _assert_mounts_with_a_blank_provider("zq-not-a-provider")
