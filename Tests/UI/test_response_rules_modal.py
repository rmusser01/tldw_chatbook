"""Rule editor tests use the public runtime and real private SQLite owner."""

from dataclasses import replace
from io import StringIO

import pytest
from rich.console import Console
from textual.widgets import Button, Input

from Tests.Chat.test_response_rules_runtime import activate, native, seed
from Tests.Chat.response_rules_store_fixtures import learning
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


class ManagerHost(ConsolidatedCSSApp):
    CSS_PATH = [str(p) for p in APP_STYLESHEETS]

    def __init__(self, modal):
        super().__init__()
        self.modal = modal

    async def on_mount(self):
        await self.push_screen(self.modal)


@pytest.mark.asyncio
async def test_global_manager_does_not_inherit_chat_only_rules(native):
    from textual.widgets import OptionList

    _owner, rules, _chats, session, _gateway, _controller = native
    activate(native)
    global_scope = rules.scopes(session.id)[2]
    modal = ResponseRulesModal(global_scope, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        assert modal.query_one("#rr-rules", OptionList).option_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_editor_testing_uses_public_runtime_and_never_autoactivates(native, size):
    _owner, rules, _chats, session, _gateway, _controller = native
    seed(native)
    _gateway.reply = "Evidence: corrected answer"
    learned = await rules.learn(session.id, "The answer omitted evidence")
    assert learned.state == "active", learned.reason
    scope = rules.scopes(session.id)[0]
    binding = rules.store.list_bindings(scope)[0]
    tested = []
    actual = rules.test_edit

    async def public(*args, **kwargs):
        tested.append((args, kwargs))
        return await actual(*args, **kwargs)

    rules.test_edit = public
    modal = ResponseRulesModal(scope, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=size) as pilot:
        for _ in range(5):
            await pilot.pause()
        assert "Missing proof" in str(modal.query_one("#rr-examples").render())
        modal.query_one("#rr-title", Input).value = "Evidence required"
        modal.query_one("#rr-test", Button).press()
        for _ in range(50):
            await pilot.pause()
            if modal.tested is not None:
                break
        assert tested and modal.tested.reason in {"tested", "validation_reused"}
        assert rules.store.list_bindings(scope)[0] == binding
        assert rules.store.list_drafts(scope)[-1].rule.revision > binding.revision
        console = Console(record=True, width=size[0], file=StringIO())
        console.print(modal._compositor.render_full_update())
        painted = console.export_text()
        assert "Test" in painted and "Save" in painted and "Close" in painted
        save = modal.query_one("#rr-save", Button)
        for _ in range(35):
            if host.focused is save:
                break
            await pilot.press("tab")
        assert host.focused is save
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert (
            rules.store.list_bindings(scope)[0].revision == modal.tested.rule.revision
        )


@pytest.mark.asyncio
async def test_inherited_exclusion_masks_and_exact_promotion_preview(native):
    _owner, rules, _chats, session, _gateway, _controller = native
    activate(native)
    chat, _workspace, global_scope = rules.scopes(session.id)
    rules.store.promote("rule", 1, global_scope, expected_binding_revision=0)
    rules.store.delete_binding(chat, "rule", expected_binding_revision=1)
    modal = ResponseRulesModal(chat, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if modal._selected is not None:
                break
        assert modal._selected is not None
        modal.query_one("#rr-exclude", Button).press()
        for _ in range(6):
            await pilot.pause()
        assert rules.effective_rules(session.id) == (), (
            modal._current_scope(),
            modal._busy,
            str(modal.query_one("#rr-status").render()),
        )
        assert rules.store.list_bindings(global_scope)[0].state == "enabled"


@pytest.mark.asyncio
async def test_missing_original_requires_selected_replacement_and_fresh_examples(
    native,
):
    from textual.widgets import Select
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    _owner, rules, chats, session, _gateway, _controller = native
    old = activate(native)
    chats.delete_message(old.id)
    user = chats.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Explain result again",
        persist=True,
    )
    replacement = chats.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Missing proof again",
        persist=True,
    )
    chats.persist_message_if_needed(user.id)
    chats.persist_message_if_needed(replacement.id)
    modal = ResponseRulesModal(rules.scopes(session.id)[0], rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(10):
            await pilot.pause()
        modal.query_one("#rr-test", Button).press()
        for _ in range(12):
            await pilot.pause()
        assert modal.tested.reason == "original_evidence_unavailable"
        modal.query_one("#rr-example", Select).value = replacement.id
        modal.query_one("#rr-test", Button).press()
        for _ in range(30):
            await pilot.pause()
            if modal.tested is not None and modal.tested.validation is not None:
                break
        assert modal.tested.reason == "tested"
        assert (
            modal.tested.validation.source.message_id
            == replacement.persisted_message_id
        )
        assert rules.store.list_bindings(modal.scope)[0].revision == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["storage", "binding", "scope"])
async def test_save_failure_preserves_edits_and_current_pin(native, fault, monkeypatch):
    from tldw_chatbook.Chat.response_rules.models import RuleBinding

    _owner, rules, chats, session, gateway, _controller = native
    seed(native)
    gateway.reply = "Evidence: repaired"
    learned = await rules.learn(session.id, "The answer omitted evidence")
    scope = rules.scopes(session.id)[0]
    pin = rules.store.list_bindings(scope)[0]
    modal = ResponseRulesModal(scope, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        title = modal.query_one("#rr-title", Input)
        title.value = "Evidence required"
        modal.query_one("#rr-test", Button).press()
        for _ in range(25):
            await pilot.pause()
            if modal.tested is not None:
                break
        assert modal.tested.validation is not None
        activate = rules.store.activate
        if fault == "storage":

            def refuse(*args, **kwargs):
                raise OSError("fixture storage unavailable")

            monkeypatch.setattr(rules.store, "activate", refuse)
        elif fault == "binding":
            pin = rules.store.set_binding(
                replace(pin, state="disabled"),
                expected_binding_revision=pin.binding_revision,
            )
        else:
            chats.create_session(title="Another Chat")
        assert await pilot.click("#rr-save")
        for _ in range(10):
            await pilot.pause()
        assert title.value == "Evidence required"
        assert rules.store.list_bindings(scope)[0] == pin
        if fault == "storage":
            monkeypatch.setattr(rules.store, "activate", activate)
            assert await pilot.click("#rr-save")
            for _ in range(10):
                await pilot.pause()
            assert (
                rules.store.list_bindings(scope)[0].revision
                == modal.tested.rule.revision
            )


@pytest.mark.asyncio
async def test_promotion_previews_exact_pin_without_private_examples(native):
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    _owner, rules, _chats, session, _gateway, _controller = native
    activate(native)
    chat, _workspace, global_scope = rules.scopes(session.id)
    modal = ResponseRulesModal(chat, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        assert await pilot.click("#rr-promote")
        for _ in range(5):
            await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        assert "revision 1" in host.screen.message
        assert "Missing proof" not in host.screen.message
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(10):
            await pilot.pause()
        assert rules.store.list_bindings(global_scope)[0].revision == 1
        assert rules.store.list_drafts(global_scope) == ()


@pytest.mark.asyncio
async def test_changing_scope_or_row_retains_unsaved_editor_text(native):
    from textual.widgets import Select

    _owner, rules, _chats, session, _gateway, _controller = native
    activate(native)
    chat, _workspace, global_scope = rules.scopes(session.id)
    modal = ResponseRulesModal(chat, rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        modal.query_one("#rr-title", Input).value = "Keep my unsaved title"
        modal.query_one("#rr-scope", Select).value = "2"
        for _ in range(8):
            await pilot.pause()
        assert modal.scope == chat
        assert modal.query_one("#rr-title", Input).value == "Keep my unsaved title"


@pytest.mark.asyncio
async def test_cancelled_testing_keeps_edits_and_pinned_revision(native):
    import asyncio

    _owner, rules, _chats, session, gateway, _controller = native
    seed(native)
    gateway.reply = "Evidence: repaired"
    assert (
        await rules.learn(session.id, "The answer omitted evidence")
    ).state == "active"
    scope = rules.scopes(session.id)[0]
    pin = rules.store.list_bindings(scope)[0]
    modal = ResponseRulesModal(scope, rules.store, rules)
    host = ManagerHost(modal)
    entered = asyncio.Event()
    actual = rules.builder.validate_edit

    async def held(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        return await actual(*args, **kwargs)

    rules.builder.validate_edit = held
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        modal.query_one("#rr-title", Input).value = "Evidence required"
        assert await pilot.click("#rr-test")
        for _ in range(20):
            await pilot.pause()
            if entered.is_set():
                break
        assert entered.is_set()
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert host.screen is modal
        assert modal.query_one("#rr-title", Input).value == "Evidence required"
        assert rules.store.list_bindings(scope)[0] == pin
        assert rules.state(session.id).phase == "idle"
        assert modal.query_one("#rr-save", Button).disabled


@pytest.mark.asyncio
async def test_dirty_escape_discards_only_after_confirmation(native):
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    _owner, rules, _chats, session, _gateway, _controller = native
    activate(native)
    modal = ResponseRulesModal(rules.scopes(session.id)[0], rules.store, rules)
    host = ManagerHost(modal)
    async with host.run_test(size=(80, 24)) as pilot:
        for _ in range(8):
            await pilot.pause()
        modal.query_one("#rr-title", Input).value = "Unsaved title"
        await pilot.press("escape")
        for _ in range(5):
            await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#cancel-button", Button).press()
        for _ in range(5):
            await pilot.pause()
        assert host.screen is modal
        assert modal.query_one("#rr-title", Input).value == "Unsaved title"
        await pilot.press("escape")
        for _ in range(5):
            await pilot.pause()
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(8):
            await pilot.pause()
        assert modal not in host.screen_stack
