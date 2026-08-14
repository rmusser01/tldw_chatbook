"""Contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from types import MethodType
from typing import Any

import pytest
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRowTotals
from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterOption,
    ConsoleCharacterPickerModal,
)
from tldw_chatbook.Widgets.Console.console_citation_sources_modal import (
    ConsoleCitationSourcesModal,
)
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal
from tldw_chatbook.Widgets.Console.console_cost_modal import ConsoleCostModal
from tldw_chatbook.Widgets.Console.console_image_viewer_modal import (
    ConsoleImageViewerModal,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    MODE_INSERT,
    ConsolePromptPickerModal,
)
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal
from tldw_chatbook.Widgets.Console.console_scope_picker_modal import (
    ScopeListPage,
    TagCount,
    ConsoleScopePickerModal,
)
from tldw_chatbook.Widgets.Console.console_skill_picker_modal import (
    ConsoleSkillPickerModal,
)
from tldw_chatbook.Widgets.Console.console_style_picker_modal import (
    ConsoleStylePickerModal,
)
from tldw_chatbook.Widgets.modal_dismissal import (
    SafeModalDismissMixin,
    is_modal_backdrop_click,
)


@dataclass(frozen=True)
class _Task2ModalContract:
    modal_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    cancel_result: object
    opener: str
    pre_cancel_hook: str | None
    guard: str
    focus_postcondition: str


_RESTORE_OPENER = "restore opener or Console composer fallback"


async def _empty_context_snapshot() -> ConsoleContextSnapshot:
    return ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _empty_records(_query: str) -> list[dict[str, object]]:
    return []


class _EmptySourceLister:
    async def list_page(self, **_kwargs: object) -> ScopeListPage:
        return ScopeListPage(items=(), total_matching=0)

    async def list_ids(self, **_kwargs: object) -> tuple[str, ...]:
        return ()


async def _empty_tags(_query: str) -> tuple[TagCount, ...]:
    return ()


def _citation_factory() -> ConsoleCitationSourcesModal:
    modal = ConsoleCitationSourcesModal(
        native_message_id="native-1",
        persisted_message_id="persisted-1",
        current_body="body",
        repository=object(),
        request_is_current=lambda: True,
    )
    modal._worker_started = True
    return modal


def _image_factory() -> ConsoleImageViewerModal:
    modal = ConsoleImageViewerModal(object())  # type: ignore[arg-type]
    modal._build_full_size_widget = MethodType(  # type: ignore[method-assign]
        lambda _self: Static("image", id="console-image-viewer-image"), modal
    )
    return modal


def _queue_factory() -> ConsolePromptQueueModal:
    registry = ConsolePromptQueueRegistry()
    snapshot = registry.snapshot("contract-session")
    return ConsolePromptQueueModal(
        session_id="contract-session",
        revision=snapshot.revision,
        queue_controller=registry,
    )


def _scope_factory() -> ConsoleScopePickerModal:
    source_lister = _EmptySourceLister()
    return ConsoleScopePickerModal(
        "contract target",
        None,
        None,
        lambda _scope: None,
        media_lister=source_lister,
        notes_lister=source_lister,
        tag_lister=_empty_tags,
    )


TASK2_MODAL_CONTRACTS = (
    _Task2ModalContract(
        ConsoleCharacterPickerModal,
        lambda: ConsoleCharacterPickerModal(options=[]),
        "#console-character-picker",
        None,
        "Console character chip",
        "_cancel_query_debounce",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleCitationSourcesModal,
        _citation_factory,
        "#console-citation-sources-modal",
        None,
        "Console citation marker",
        "increment _request_generation",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleContextModal,
        lambda: ConsoleContextModal(_empty_context_snapshot),
        "#console-context-modal",
        None,
        "Console context action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleCostModal,
        lambda: ConsoleCostModal(
            [], ConsoleCostRowTotals(0, 0.0, False, 0)
        ),
        "#console-cost-modal",
        None,
        "Console cost action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleImageViewerModal,
        _image_factory,
        "#console-image-viewer",
        None,
        "Console avatar",
        None,
        "intentional click-anywhere cancel",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleModelPopover,
        lambda: ConsoleModelPopover(
            settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
            providers_models={"openai": ["gpt-test"]},
        ),
        "#console-model-popover",
        None,
        "Console model chip",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsolePromptPickerModal,
        lambda: ConsolePromptPickerModal(
            mode=MODE_INSERT, prompt_search=_empty_records
        ),
        "#console-prompt-picker-modal",
        None,
        "Console composer prompt command",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsolePromptQueueModal,
        _queue_factory,
        "#console-prompt-queue-dialog",
        None,
        "Console prompt queue",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleRunLogModal,
        lambda: ConsoleRunLogModal(run_id="run-1", log_text="log"),
        "#console-run-log-modal",
        None,
        "Console run log action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleScopePickerModal,
        _scope_factory,
        "#console-scope-picker-modal",
        None,
        "Console RAG scope action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleSkillPickerModal,
        lambda: ConsoleSkillPickerModal(skill_search=_empty_records),
        "#console-skill-picker-modal",
        None,
        "Console composer skill command",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleStylePickerModal,
        lambda: ConsoleStylePickerModal(),
        "#console-style-picker-modal",
        None,
        "Console image style action",
        "_cancel_search_debounce",
        "none",
        _RESTORE_OPENER,
    ),
)


def _binding_key_action(binding: object) -> tuple[str, str]:
    if isinstance(binding, Binding):
        return binding.key, binding.action
    return binding[0], binding[1]  # type: ignore[index,return-value]


def test_task2_modal_contract_table_is_complete_and_adopted() -> None:
    assert len(TASK2_MODAL_CONTRACTS) == 12
    assert {contract.modal_type.__name__ for contract in TASK2_MODAL_CONTRACTS} == {
        "ConsoleCharacterPickerModal",
        "ConsoleCitationSourcesModal",
        "ConsoleContextModal",
        "ConsoleCostModal",
        "ConsoleImageViewerModal",
        "ConsoleModelPopover",
        "ConsolePromptPickerModal",
        "ConsolePromptQueueModal",
        "ConsoleRunLogModal",
        "ConsoleScopePickerModal",
        "ConsoleSkillPickerModal",
        "ConsoleStylePickerModal",
    }
    expected_hooks = {
        "ConsoleCharacterPickerModal": "_cancel_query_debounce",
        "ConsoleCitationSourcesModal": "increment _request_generation",
        "ConsoleStylePickerModal": "_cancel_search_debounce",
    }
    for contract in TASK2_MODAL_CONTRACTS:
        assert issubclass(contract.modal_type, SafeModalDismissMixin)
        assert contract.modal_type.SAFE_MODAL_CONTENT == contract.content_selector
        escape_actions = [
            action
            for binding in contract.modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == ["request_safe_cancel"]
        assert contract.cancel_result is None
        assert contract.opener
        assert contract.pre_cancel_hook == expected_hooks.get(
            contract.modal_type.__name__
        )
        assert contract.guard == (
            "intentional click-anywhere cancel"
            if contract.modal_type is ConsoleImageViewerModal
            else "none"
        )
        assert contract.focus_postcondition == _RESTORE_OPENER


class _Task2Harness(App[None]):
    CSS = """
    Screen { align: center middle; }
    #console-citation-sources-modal,
    #console-style-picker-modal { width: 60; height: 20; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


@pytest.mark.parametrize(
    "contract", TASK2_MODAL_CONTRACTS, ids=lambda row: row.modal_type.__name__
)
@pytest.mark.asyncio
async def test_task2_contract_selector_exists_and_escape_returns_cancel_result(
    contract: _Task2ModalContract,
) -> None:
    app = _Task2Harness()
    modal = contract.factory()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert modal.query_one(contract.content_selector)
        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [contract.cancel_result]


class _LifecycleCharacterModal(ConsoleCharacterPickerModal):
    def __init__(self) -> None:
        super().__init__(options=[])
        self.initialization_calls = 0

    async def _refresh_results(self, query: str) -> None:
        self.initialization_calls += 1
        await super()._refresh_results(query)


@pytest.mark.asyncio
async def test_textual_mro_runs_mixin_and_modal_mount_once(monkeypatch) -> None:
    mixin_mount_calls = 0
    original_mixin_mount = SafeModalDismissMixin.on_mount

    def count_mixin_mount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_mount_calls
        mixin_mount_calls += 1
        original_mixin_mount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_mount", count_mixin_mount)
    app = _Task2Harness()
    modal = _LifecycleCharacterModal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert mixin_mount_calls == 1
        assert modal.initialization_calls == 1


@pytest.mark.asyncio
async def test_textual_mro_runs_citation_mixin_unmount_once(monkeypatch) -> None:
    mixin_unmount_calls = 0
    original_mixin_unmount = SafeModalDismissMixin.on_unmount

    def count_mixin_unmount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_unmount_calls
        mixin_unmount_calls += 1
        original_mixin_unmount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_unmount", count_mixin_unmount)
    app = _Task2Harness()
    modal = _citation_factory()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        generation = modal._request_generation

        modal.dismiss(None)
        await pilot.pause()

        assert mixin_unmount_calls == 1
        assert modal._request_generation == generation + 1


class _TrackedCharacterModal(ConsoleCharacterPickerModal):
    def __init__(self) -> None:
        super().__init__(options=[ConsoleCharacterOption(1, "Ada")])
        self.cleanup_calls = 0
        self.order: list[str] = []

    def _cancel_query_debounce(self) -> None:
        self.cleanup_calls += 1
        self.order.append("cleanup")
        super()._cancel_query_debounce()

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.order.append("dismiss")
        return super().dismiss(result)


class _TrackedCitationModal(ConsoleCitationSourcesModal):
    def __init__(self) -> None:
        super().__init__(
            native_message_id="native-1",
            persisted_message_id="persisted-1",
            current_body="body",
            repository=object(),
            request_is_current=lambda: True,
        )
        self._worker_started = True
        self.generation_at_dismiss: list[int] = []

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.generation_at_dismiss.append(self._request_generation)
        return super().dismiss(result)


class _TrackedStyleModal(ConsoleStylePickerModal):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_calls = 0
        self.order: list[str] = []

    def _cancel_search_debounce(self) -> None:
        self.cleanup_calls += 1
        self.order.append("cleanup")
        super()._cancel_search_debounce()

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.order.append("dismiss")
        return super().dismiss(result)


async def _request_task2_cancel(modal, pilot, source: str) -> None:  # type: ignore[no-untyped-def]
    if source == "visible":
        if isinstance(modal, _TrackedCitationModal):
            await pilot.click("#console-citation-sources-close")
        else:
            result = modal.action_dismiss_picker()
            if inspect.isawaitable(result):
                await result
    elif source == "escape":
        await pilot.press("escape")
    else:
        await pilot.click(offset=(0, 0))
    await pilot.pause()


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_character_cancel_sources_run_debounce_cleanup_once(source: str) -> None:
    app = _Task2Harness()
    modal = _TrackedCharacterModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._query_debounce_timer = modal.set_timer(60, lambda: None)
        modal.cleanup_calls = 0
        modal.order.clear()

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.cleanup_calls == 1
    assert modal.order[:2] == ["cleanup", "dismiss"]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_citation_cancel_sources_invalidate_generation_once_before_dismiss(
    source: str,
) -> None:
    app = _Task2Harness()
    modal = _TrackedCitationModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._request_generation = 10

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.generation_at_dismiss == [11]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_style_cancel_sources_run_debounce_cleanup_once(source: str) -> None:
    app = _Task2Harness()
    modal = _TrackedStyleModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._search_debounce_timer = modal.set_timer(60, lambda: None)
        modal.cleanup_calls = 0
        modal.order.clear()

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.cleanup_calls == 1
    assert modal.order[:2] == ["cleanup", "dismiss"]


@dataclass(frozen=True)
class _FakeRegion:
    contains_point: bool

    def contains(self, _x: int, _y: int) -> bool:
        return self.contains_point


@dataclass(frozen=True)
class _FakeContent:
    region: _FakeRegion


@pytest.mark.parametrize(
    ("button", "known", "descendant", "contains", "expected"),
    [
        pytest.param(1, True, False, False, True, id="primary-outside"),
        pytest.param(1, True, True, False, False, id="primary-descendant"),
        pytest.param(1, True, False, True, False, id="primary-in-region"),
        pytest.param(2, True, False, False, False, id="non-primary-outside"),
        pytest.param(1, False, False, False, False, id="unknown-provenance"),
    ],
)
def test_classifier_identifies_only_known_primary_backdrop_clicks(
    button: int,
    known: bool,
    descendant: bool,
    contains: bool,
    expected: bool,
) -> None:
    content = _FakeContent(_FakeRegion(contains))

    assert (
        is_modal_backdrop_click(
            button=button,
            provenance_known=known,
            target_is_content_or_descendant=descendant,
            point_is_in_content_region=content.region.contains(7, 9),
        )
        is expected
    )


class _HostScreen(Screen[None]):
    """Revealed screen with the same optional focus seam as Console."""

    def __init__(self) -> None:
        super().__init__()
        self.composer_fallback_calls: list[bool] = []
        self.underlying_button_presses = 0
        self.unrelated_button_presses = 0
        self.screen_mouse_ups = 0
        self.screen_clicks = 0

    def compose(self) -> ComposeResult:
        yield Button("Underlying action", id="modal-test-underlying-action")
        yield Input(id="modal-test-opener")
        yield Input(id="modal-test-other-focus")
        yield Button("Unrelated action", id="modal-test-unrelated-action")
        yield Static("host", id="modal-test-host-label")

    @on(Button.Pressed, "#modal-test-underlying-action")
    def _underlying_action(self) -> None:
        self.underlying_button_presses += 1

    @on(Button.Pressed, "#modal-test-unrelated-action")
    def _unrelated_action(self) -> None:
        self.unrelated_button_presses += 1

    def on_mouse_up(self, _event: events.MouseUp) -> None:
        self.screen_mouse_ups += 1

    def on_click(self, _event: events.Click) -> None:
        self.screen_clicks += 1

    def _focus_console_composer_if_needed(self, *, force: bool) -> None:
        self.composer_fallback_calls.append(force)


class _ModalHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.host = _HostScreen()

    async def on_mount(self) -> None:
        await self.push_screen(self.host)


class _NestedModal(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("nested", id="modal-test-nested")


CancelEffect = Callable[[], Awaitable[None]]


class _SafeTestModal(SafeModalDismissMixin, ModalScreen[bool | None]):
    SAFE_MODAL_CONTENT = "#modal-test-content"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    CSS = """
    _SafeTestModal {
        align: center middle;
    }

    #modal-test-content {
        width: 30;
        height: 7;
        background: $surface;
    }
    """

    def __init__(
        self,
        *,
        result: bool | None = False,
        cancel_effect: CancelEffect | None = None,
    ) -> None:
        super().__init__()
        self._cancel_result = result
        self._cancel_effect = cancel_effect

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-test-content"):
            yield Static("safe modal", id="modal-test-descendant")
            yield Button("Cancel", id="modal-test-cancel")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._cancel_effect is not None:
            await self.run_cancel_effect_once(self._cancel_effect)
        self.dismiss_safe_once(self._cancel_result)

    @on(Button.Pressed, "#modal-test-cancel")
    async def _cancel_from_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")


async def _mount_modal(
    app: _ModalHarness,
    pilot,
    modal: _SafeTestModal,
    results: list[bool | None],
    *,
    opener_selector: str = "#modal-test-opener",
) -> Input:
    opener = app.host.query_one(opener_selector, Input)
    opener.focus()
    await pilot.pause()
    assert app.host.focused is opener
    app.push_screen(modal, results.append)
    await pilot.pause()
    assert app.screen is modal
    return opener


def _outside_click(
    modal: _SafeTestModal, screen_x: int = 0, screen_y: int = 0
) -> events.Click:
    return events.Click(
        modal,
        screen_x,
        screen_y,
        0,
        0,
        1,
        False,
        False,
        False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


@pytest.mark.asyncio
async def test_single_shot_consumes_repeated_escape_and_backdrop_while_pending():
    entered = asyncio.Event()
    release = asyncio.Event()
    effect_calls = 0

    async def delayed_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        entered.set()
        await release.wait()

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=delayed_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        first_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await entered.wait()
        second_escape = asyncio.create_task(modal.action_request_safe_cancel())
        backdrop = _outside_click(modal)
        backdrop_request = asyncio.create_task(modal.on_click(backdrop))
        try:
            await asyncio.sleep(0)
            assert second_escape.done()
            assert backdrop_request.done()
            assert effect_calls == 1
            assert app.screen is modal
            assert backdrop._stop_propagation
            assert backdrop._no_default_action
        finally:
            release.set()
            await asyncio.gather(
                first_escape,
                second_escape,
                backdrop_request,
                return_exceptions=True,
            )
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_top_screen_check_preserves_nested_modal_and_retry_skips_effect():
    effect_calls = 0
    nested = _NestedModal()

    async def push_nested() -> None:
        nonlocal effect_calls
        effect_calls += 1
        app.push_screen(nested)

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=push_nested)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is nested
        assert modal.is_mounted
        assert results == []
        assert effect_calls == 1

        nested.dismiss(None)
        await pilot.pause()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is app.host
        assert results == [False]
        assert effect_calls == 1


@pytest.mark.asyncio
async def test_single_shot_cancel_effect_commitment_survives_exception():
    effect_calls = 0

    async def failing_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        raise RuntimeError("cancel failed")

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=failing_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        with pytest.raises(RuntimeError, match="cancel failed"):
            await modal.action_request_safe_cancel()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert effect_calls == 1
        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.parametrize("source", ["button", "backdrop"])
@pytest.mark.asyncio
async def test_visible_cancel_and_backdrop_return_exact_typed_value(source: str):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(result=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        if source == "button":
            await pilot.click("#modal-test-cancel")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_restore_focus_returns_to_the_mounted_opener():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        app.host.set_focus(app.host.query_one("#modal-test-other-focus", Input))
        assert app.host.focused is not opener

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.focused is opener
        assert app.host.composer_fallback_calls == []


@pytest.mark.asyncio
async def test_restore_focus_uses_console_fallback_when_opener_was_removed():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        await opener.remove()

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.composer_fallback_calls == [True]


@pytest.mark.asyncio
async def test_backdrop_shield_is_inert_to_revealed_screen_and_focus():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (
            underlying.region.x + 1,
            underlying.region.y + 1,
        )
        opener = await _mount_modal(app, pilot, modal, results)
        assert not modal.query_one("#modal-test-content", Vertical).region.contains(
            *click_point
        )

        await pilot.click(offset=click_point)
        await pilot.click(offset=click_point)
        await pilot.pause()

        assert results == [False]
        assert app.host.underlying_button_presses == 0
        assert app.host.screen_mouse_ups == 0
        assert app.host.screen_clicks == 0
        assert app.host.focused is opener
        assert app.mouse_captured is None

        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)

        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_backdrop_shield_allows_an_unrelated_coordinate_action():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        unrelated = app.host.query_one("#modal-test-unrelated-action", Button)
        origin = (underlying.region.x + 1, underlying.region.y + 1)
        unrelated_point = (unrelated.region.x + 1, unrelated.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=origin)
        await pilot.click(offset=unrelated_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 0
        assert app.host.unrelated_button_presses == 1
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_safe_modal_state_resets_when_same_instance_is_repushed():
    effect_calls = 0

    async def effect() -> None:
        nonlocal effect_calls
        effect_calls += 1

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)
        await modal.action_request_safe_cancel()
        await pilot.pause()
        assert app.screen is app.host

        second_opener = await _mount_modal(
            app,
            pilot,
            modal,
            results,
            opener_selector="#modal-test-other-focus",
        )
        app.host.set_focus(app.host.query_one("#modal-test-opener", Input))
        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.screen is app.host
        assert results == [False, False]
        assert effect_calls == 2
        assert app.host.focused is second_opener


@pytest.mark.asyncio
async def test_real_click_dispatch_keeps_descendant_and_inside_geometry_open():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        await pilot.click("#modal-test-descendant")
        await pilot.pause()
        assert app.screen is modal

        content = modal.query_one("#modal-test-content", Vertical)
        inside_blank_point = (content.region.right - 2, content.region.bottom - 1)
        await pilot.click(offset=inside_blank_point)
        await pilot.pause()

        assert app.screen is modal
        assert results == []


@pytest.mark.parametrize("source", ["escape", "button"])
@pytest.mark.asyncio
async def test_non_backdrop_cancel_does_not_shield_revealed_screen(source: str):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)
        if source == "escape":
            await modal.action_request_safe_cancel()
        else:
            await pilot.click("#modal-test-cancel")
        await pilot.pause()

        assert app.screen is app.host
        assert app.mouse_captured is None
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_pending_escape_records_backdrop_before_terminal_dismissal():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_effect() -> None:
        entered.set()
        await release.wait()

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=delayed_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        pending_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await entered.wait()
        await pilot.click(offset=click_point)
        assert app.screen is modal

        release.set()
        await pending_escape
        await pilot.pause()
        assert app.screen is app.host

        await pilot.click(offset=click_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 0
        assert app.host.screen_mouse_ups == 0
        assert app.host.screen_clicks == 0
        assert app.mouse_captured is None

        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.parametrize("retry_source", ["escape", "button"])
@pytest.mark.asyncio
async def test_stale_backdrop_attempt_does_not_shield_later_retry(
    retry_source: str,
):
    app = _ModalHarness()
    nested = _NestedModal()

    async def push_nested() -> None:
        app.push_screen(nested)

    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=push_nested)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=(0, 0))
        await pilot.pause()
        assert app.screen is nested

        nested.dismiss(None)
        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)
        assert app.screen is modal

        if retry_source == "escape":
            await modal.action_request_safe_cancel()
        else:
            await pilot.click("#modal-test-cancel")
        await pilot.pause()
        await pilot.pause()

        assert app.screen is app.host
        assert app.mouse_captured is None

        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_expired_backdrop_chain_adds_no_shield():
    app = _ModalHarness()

    async def outlive_click_chain() -> None:
        await asyncio.sleep(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)

    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=outlive_click_chain)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=click_point)
        assert app.screen is app.host

        await pilot.click(offset=click_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 1
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_old_request_generation_cannot_dismiss_repushed_presentation():
    first_entered = asyncio.Event()
    first_release = asyncio.Event()
    second_entered = asyncio.Event()
    second_release = asyncio.Event()
    effect_calls = 0

    async def generation_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        if effect_calls == 1:
            first_entered.set()
            await first_release.wait()
        else:
            second_entered.set()
            await second_release.wait()

    app = _ModalHarness()
    first_results: list[bool | None] = []
    second_results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=generation_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, first_results)
        old_request = asyncio.create_task(modal.action_request_safe_cancel())
        await first_entered.wait()

        modal.dismiss(None)
        await pilot.pause()
        assert first_results == [None]
        assert app.screen is app.host

        await _mount_modal(app, pilot, modal, second_results)
        new_request = asyncio.create_task(
            modal.on_click(_outside_click(modal, *click_point))
        )
        await second_entered.wait()

        first_release.set()
        await old_request
        await pilot.pause()

        try:
            assert app.screen is modal
            assert second_results == []
            await modal.action_request_safe_cancel()
            assert app.screen is modal
            assert effect_calls == 2
        finally:
            second_release.set()
            await new_request
        await pilot.pause()

        assert app.screen is app.host
        assert second_results == [False]
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 0
