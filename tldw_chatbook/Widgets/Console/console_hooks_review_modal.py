"""Native review of saved hook definitions and their exact consent."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Static

from tldw_chatbook.Constants import TAB_SETTINGS
from tldw_chatbook.UI.Console_Modules.hooks import HookReviewResult, in_worker_task
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from textual.await_complete import AwaitComplete

    from tldw_chatbook.Agents.hook_permissions import (
        HookPermissions,
        HookReviewSnapshot,
    )
    from tldw_chatbook.Agents.run_hooks import HookSpec


def command_json(spec: HookSpec) -> str:
    """Display exact argument boundaries, with control characters escaped."""
    return json.dumps(list(spec.command), ensure_ascii=True, indent=2)


class ConsoleHooksReviewModal(SafeModalDismissMixin, ModalScreen[HookReviewResult]):
    """Review current saved hooks; decisions never derive authority from labels."""

    SAFE_MODAL_CONTENT = "#console-hooks-review"
    BINDINGS = (("escape", "request_safe_cancel", "Cancel"),)
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "css" / "screen_agentic_settings.tcss"
    )

    def __init__(
        self,
        *,
        snapshot: HookReviewSnapshot,
        waiting_for_send: bool,
        approve: Callable[
            [HookReviewSnapshot, Collection[str]], Awaitable[HookReviewSnapshot]
        ],
        revoke: Callable[[HookReviewSnapshot, str], Awaitable[HookReviewSnapshot]],
        disable: Callable[[HookReviewSnapshot, str], Awaitable[HookReviewSnapshot]],
        recover: Callable[[], Awaitable[HookReviewSnapshot]],
        reset: Callable[[HookReviewSnapshot], Awaitable[HookReviewSnapshot]],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self.add_class("settings-hooks-review-modal")
        self.snapshot = snapshot
        self._waiting = waiting_for_send
        self._approve, self._revoke, self._disable = approve, revoke, disable
        self._recover, self._reset, self._cancel = recover, reset, on_cancel
        self._all = False
        self._selected: set[str] = set()
        self._busy = False
        self._generation = 0
        self._answer: asyncio.Future[HookReviewResult] | None = None

    def answer(self) -> asyncio.Future[HookReviewResult]:
        """This review's result, settled by the modal itself.

        TASK-33621.28: not a ``push_screen`` result callback. Textual runs
        that through the requester pump's ``call_next``, and a requester that
        awaits the answer is that very pump, blocked -- the callback never
        ran and the Console's Send never settled. ``dismiss`` settles this
        future synchronously instead, and leaving the DOM without a dismissal
        settles it as a cancel, so a waiting caller always resumes.
        """
        if self._answer is None:
            self._answer = asyncio.get_running_loop().create_future()
        return self._answer

    def _settle(self, result: HookReviewResult) -> None:
        answer = self.answer()
        if not answer.done():
            answer.set_result(result)

    def dismiss(self, result: HookReviewResult | None = None) -> AwaitComplete:
        self._settle(result or HookReviewResult("cancel"))
        return super().dismiss(result)

    def compose(self) -> ComposeResult:
        with Vertical(id="console-hooks-review"):
            yield Static("Review hooks", id="console-hooks-title")
            yield Static(self._notice(), id="console-hooks-notice", markup=False)
            with Horizontal(id="console-hooks-tabs"):
                yield Button(
                    "Needs review",
                    id="console-hooks-needs",
                    classes="hook-review-action",
                )
                yield Button(
                    "All hooks", id="console-hooks-all", classes="hook-review-action"
                )
                yield Button(
                    "Retry refresh",
                    id="console-hooks-retry",
                    classes="hook-review-action",
                )
                yield Button(
                    "Reset state",
                    id="console-hooks-reset",
                    disabled=self.snapshot.store_revision != ("", 0),
                    classes="hook-review-action",
                )
            rows = VerticalScroll(id="console-hooks-list")
            rows.compose = self._rows
            yield rows
            yield Button(
                "Manage in Settings",
                id="console-hooks-settings",
                classes="hook-review-action",
            )
            with Horizontal(id="console-hooks-actions"):
                yield Button(
                    "Not now" if self._waiting else "Close",
                    id="console-hooks-cancel",
                    classes="hook-review-action",
                )
                yield Button(
                    "Allow all",
                    id="console-hooks-allow-all",
                    disabled=not self._pending(),
                    classes="hook-review-action",
                )
                yield Button(
                    "Allow selected",
                    id="console-hooks-allow-selected",
                    disabled=True,
                    classes="hook-review-action",
                )
                ready = Button(
                    "Continue Send",
                    id="console-hooks-ready",
                    disabled=not self.snapshot.ready,
                    classes="hook-review-action",
                )
                ready.display = self._waiting
                yield ready

    def _notice(self) -> str:
        if self.snapshot.notice:
            return self.snapshot.notice
        if not self.snapshot.ready:
            return self.snapshot.blocked_reason or "Review hooks."
        return "No enabled hooks need review. Permissions are remembered for these exact definitions."

    def _pending(self) -> set[str]:
        return {
            row.entry.key
            for row in self.snapshot.rows
            if row.entry and row.state == "pending"
        }

    def _rows(self) -> ComposeResult:
        visible = [
            row
            for row in self.snapshot.rows
            if self._all or row.state in {"pending", "invalid", "recovery"}
        ]
        if not visible:
            yield Static(
                "No hooks need review." if not self._all else "No hooks configured.",
                markup=False,
            )
        for index, row in enumerate(self.snapshot.rows):
            if row not in visible:
                continue
            entry = row.entry
            with Vertical(classes="hook-review-row", id=f"hook-review-row-{index}"):
                if entry is None:
                    yield Static(
                        self.snapshot.blocked_reason
                        or "Permission state needs recovery.",
                        markup=False,
                    )
                    continue
                raw_rows = (
                    self.snapshot.config.section.get("hook", [])
                    if isinstance(self.snapshot.config.section, dict)
                    else []
                )
                raw = raw_rows[entry.index] if entry.index < len(raw_rows) else {}
                title = raw.get("name") if isinstance(raw, dict) else None
                title = (
                    title
                    if isinstance(title, str)
                    else entry.spec.event + " / " + Path(entry.spec.command[0]).name
                    if entry.spec
                    else "Invalid hook"
                )
                # One-line JSON escapes prevent terminal controls or markup in chrome.
                title = json.dumps(title, ensure_ascii=True)[1:-1][:48]
                with Horizontal(classes="hook-review-heading"):
                    yield Checkbox(
                        Text(f"{entry.index + 1} - {title}"),
                        id=f"hook-review-select-{index}",
                        classes="hook-review-select",
                        disabled=row.state != "pending",
                    )
                    yield Button(
                        "Details",
                        id=f"hook-review-details-{index}",
                        classes="hook-review-action hook-review-details",
                    )
                yield Static(
                    f"{row.change} | {row.state.capitalize()}",
                    classes="hook-review-state",
                    markup=False,
                )
                if entry.error:
                    yield Static(
                        entry.error + " Repair in Settings or disable this saved row.",
                        markup=False,
                    )
                with Horizontal(classes="hook-review-controls"):
                    if row.state == "approved":
                        yield Button(
                            "Revoke now",
                            id=f"hook-review-revoke-{index}",
                            classes="hook-review-action",
                        )
                    if entry.enabled is not False and isinstance(raw, dict):
                        yield Button(
                            "Disable now",
                            id=f"hook-review-disable-{index}",
                            classes="hook-review-action",
                        )

    async def _render_rows(self) -> None:
        self._selected.clear()
        region = self.query_one("#console-hooks-list", VerticalScroll)
        await region.recompose()
        if self.app.screen is not self:
            return
        for notice in self.query("#console-hooks-notice").results(Static):
            notice.update(self._notice())
        self._sync_actions()

    def _sync_actions(self) -> None:
        for button in self.query(Button):
            if button.id not in {"console-hooks-cancel", "console-hooks-settings"}:
                button.disabled = self._busy
        for checkbox in self.query(Checkbox):
            index = int((checkbox.id or "").rsplit("-", 1)[-1])
            checkbox.disabled = (
                self._busy or self.snapshot.rows[index].state != "pending"
            )
        self.query_one("#console-hooks-allow-all", Button).disabled = (
            self._busy or not self._pending()
        )
        self.query_one("#console-hooks-allow-selected", Button).disabled = (
            self._busy or not self._selected
        )
        self.query_one("#console-hooks-ready", Button).disabled = (
            self._busy or not self._waiting or not self.snapshot.ready
        )
        self.query_one("#console-hooks-reset", Button).disabled = (
            self._busy or self.snapshot.store_revision != ("", 0)
        )

    def on_mount(self) -> None:
        self._sync_actions()

    @on(Checkbox.Changed)
    def _selection_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        if not event.checkbox.is_mounted:
            return
        index = int((event.checkbox.id or "").rsplit("-", 1)[-1])
        if index >= len(self.snapshot.rows):
            return
        row = self.snapshot.rows[index]
        if row.entry and row.state == "pending":
            if event.value:
                self._selected.add(row.entry.key)
            else:
                self._selected.discard(row.entry.key)
        self._sync_actions()

    @on(Button.Pressed)
    async def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        action = event.button.id or ""
        if action == "console-hooks-cancel":
            await self.request_safe_cancel(source="button")
            return
        if action == "console-hooks-settings":
            self._cancel()
            self._generation += 1
            self.dismiss_safe_once(HookReviewResult("settings"))
            return
        if self._busy:
            return
        if action in {"console-hooks-needs", "console-hooks-all"}:
            self._all = action == "console-hooks-all"
            await self._render_rows()
        elif action.startswith("hook-review-details-"):
            index = int(action.rsplit("-", 1)[-1])
            containers = self.query(f"#hook-review-row-{index}")
            if not containers:
                return
            container = containers.first(Vertical)
            existing = container.query(".hook-review-detail")
            if existing:
                await existing.remove()
            else:
                entry = self.snapshot.rows[index].entry
                if entry and entry.spec:
                    spec = entry.spec
                    details = f"Source: User config\nEvent: {spec.event}\nCommand (argv):\n{command_json(spec)}\nMatcher: {json.dumps(spec.matcher)}\nTimeout: {spec.timeout_s:g}s"
                else:
                    details = "Invalid definition. Open Settings to inspect and repair the saved entry."
                detail = Static(details, classes="hook-review-detail", markup=False)
                controls = container.query(".hook-review-controls")
                if not controls:
                    return
                await container.mount(detail, before=controls.first())
                if self.app.screen is self and detail.is_mounted:
                    detail.scroll_visible(top=True, immediate=True)
        elif action == "console-hooks-ready" and self.snapshot.ready:
            self.dismiss_safe_once(HookReviewResult("ready", self.snapshot))
        else:
            self._busy = True
            self._sync_actions()
            self.run_worker(
                self._mutate(action), group="hook-review-write", exclusive=True
            )

    async def _mutate(self, action: str) -> None:
        generation = self._generation
        try:
            if action == "console-hooks-retry":
                updated = await self._recover()
            elif action == "console-hooks-reset":
                confirmed = asyncio.get_running_loop().create_future()

                def answer(result: bool | None) -> None:
                    if not confirmed.done():
                        confirmed.set_result(bool(result))

                self.app.push_screen(
                    ConfirmationDialog(
                        "Reset hook permissions?",
                        "Forget invalid permission state. Every enabled hook will need review.",
                        confirm_label="Reset",
                        confirm_callback=lambda: None,
                    ),
                    callback=answer,
                )
                if not await confirmed:
                    return
                updated = await self._reset(self.snapshot)
            elif action.startswith(("hook-review-revoke-", "hook-review-disable-")):
                index = int(action.rsplit("-", 1)[-1])
                entry = self.snapshot.rows[index].entry
                assert entry is not None
                mutate = (
                    self._revoke
                    if action.startswith("hook-review-revoke-")
                    else self._disable
                )
                updated = await mutate(self.snapshot, entry.key)
            elif action in {"console-hooks-allow-all", "console-hooks-allow-selected"}:
                keys = (
                    self._pending()
                    if action.endswith("-all")
                    else self._selected.copy()
                )
                updated = await self._approve(self.snapshot, keys)
            else:
                return
            if generation != self._generation or self.app.screen is not self:
                return
            self.snapshot = updated
            if self._waiting and updated.ready:
                self.dismiss_safe_once(HookReviewResult("ready", updated))
            else:
                await self._render_rows()
        except Exception:  # noqa: BLE001 -- retain draft and show a bounded failure
            if (
                generation == self._generation
                and self.is_mounted
                and self.app.screen is self
            ):
                try:
                    updated = await self._recover()
                    if (
                        generation != self._generation
                        or not self.is_mounted
                        or self.app.screen is not self
                    ):
                        return
                    self.snapshot = updated
                    await self._render_rows()
                except (OSError, ValueError):
                    pass
                if (
                    generation != self._generation
                    or not self.is_mounted
                    or self.app.screen is not self
                ):
                    return
                for notice in self.query("#console-hooks-notice").results(Static):
                    notice.update(
                        "Hooks or permissions changed, or the save failed. Review current state and retry."
                    )
        finally:
            self._busy = False
            if self.is_mounted and self.app.screen is self:
                self._sync_actions()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        self._generation += 1
        self._cancel()
        self.dismiss_safe_once(HookReviewResult("cancel"))

    def on_unmount(self) -> None:
        self._generation += 1
        if not self._safe_dismiss_committed:
            self._cancel()
        self._settle(HookReviewResult("cancel"))


async def request_hook_review(
    screen: Screen,
    owner: HookPermissions,
    snapshot: HookReviewSnapshot,
    waiting: bool,
    on_cancel: Callable[[], None],
) -> HookReviewResult:
    """Present the shared modal and wait for the modal's own answer.

    Await this from a worker. The answer does not depend on any pump's
    ``call_next`` (see ``ConsoleHooksReviewModal.answer``), but the APP pump
    delivers every key and click the modal needs, so awaiting it on the app
    pump -- a handler or an ``app.call_later`` callback -- still freezes the
    whole app (TASK-33621.28). ``ConsoleHooksController.dispatch`` hands the
    review of a Send made off a worker's own task to a worker for exactly
    this reason.
    """
    if not in_worker_task():
        # W003 cannot see this await (the modal settles its own answer), so
        # a new caller on a pump is reported here rather than found frozen.
        # Class and widget id, and a bool: no permission content is logged.
        logger.error(
            "Hook review awaited outside a worker task (screen={}, "
            "waiting_for_send={}): the caller's message pump is blocked until "
            "the review closes (TASK-33621.28).",
            type(screen).__name__ + (f"#{screen.id}" if screen.id else ""),
            waiting,
        )
    modal = ConsoleHooksReviewModal(
        snapshot=snapshot,
        waiting_for_send=waiting,
        approve=lambda saved, keys: asyncio.to_thread(owner.approve, saved, keys),
        revoke=lambda saved, key: asyncio.to_thread(owner.revoke, saved, key),
        disable=lambda saved, key: asyncio.to_thread(owner.disable, saved, key),
        recover=lambda: asyncio.to_thread(owner.recover),
        reset=lambda saved: asyncio.to_thread(owner.reset_invalid_state, saved),
        on_cancel=on_cancel,
    )
    answer = modal.answer()
    screen.app.push_screen(modal)
    try:
        result = await answer
    except asyncio.CancelledError:
        on_cancel()
        if screen.app.screen is modal:
            await modal.request_safe_cancel(source="caller")
        raise
    if result.kind == "settings":
        screen.post_message(NavigateToScreen(TAB_SETTINGS, {"category": "hooks"}))
    return result
