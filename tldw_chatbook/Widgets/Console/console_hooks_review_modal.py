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
    from tldw_chatbook.Chat.console_hook_review import ConsoleHookReviewProjection
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


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
        on_result: Callable[[HookReviewResult], bool] | None = None,
        on_detach: Callable[[], None] | None = None,
        runtime_projection: ConsoleHookReviewProjection | None = None,
    ) -> None:
        super().__init__()
        self.add_class("settings-hooks-review-modal")
        self.snapshot = snapshot
        self._waiting = waiting_for_send
        self._approve, self._revoke, self._disable = approve, revoke, disable
        self._recover, self._reset, self._cancel = recover, reset, on_cancel
        self._all = False
        self._selected: set[str] = set()
        self._busy = bool(runtime_projection and runtime_projection.busy)
        self._runtime_ready_pending = False
        self._runtime_result = on_result
        self._runtime_detach = on_detach
        self._console_hook_review_projection = runtime_projection
        self._generation = 0
        self._answer: asyncio.Future[HookReviewResult] | None = None

    def update_runtime_projection(
        self, projection: ConsoleHookReviewProjection
    ) -> None:
        """Refresh disposable rows from the same resident review, including busy."""
        previous = self._console_hook_review_projection
        self._console_hook_review_projection = projection
        if not projection.busy:
            self._runtime_ready_pending = False
        if (
            previous
            and previous.snapshot is projection.snapshot
            and previous.busy == projection.busy
            and self._busy == projection.busy
        ):
            return
        self.snapshot = projection.snapshot
        self._busy = projection.busy
        if self.is_mounted:
            self.call_later(self._render_rows)

    def _submit_result(self, result: HookReviewResult) -> None:
        if self._runtime_result is None:
            self.dismiss_safe_once(result)
            return
        accepted = self._runtime_result(result)
        if result.kind == "ready":
            # Keep the presentation/token live until the runtime's original
            # permission reread has settled. Ready is not cached authority.
            if accepted:
                self._runtime_ready_pending = True
                self._busy = True
                self._sync_actions()
            return
        self.dismiss_safe_once(result)
        if accepted and result.kind == "settings":
            self.app.post_message(NavigateToScreen(TAB_SETTINGS, {"category": "hooks"}))

    def answer(self) -> asyncio.Future[HookReviewResult]:
        """This review's result, settled by the modal itself.

        TASK-33621.28: not a ``push_screen`` result callback. Textual runs
        that through the requester pump's ``call_next``, and a requester that
        awaits the answer is that very pump, blocked -- the callback never
        ran and the Console's Send never settled. ``dismiss`` settles this
        future synchronously instead, and leaving the DOM without a dismissal
        settles it as a cancel, so a waiting caller always resumes.

        Returns:
            The same future on every call, created on first use. It resolves
            to the ``HookReviewResult`` the modal was dismissed with --
            ``HookReviewResult("cancel")`` for a bare dismissal or an unmount
            without one -- and is never cancelled or failed by the modal.
        """
        if self._answer is None:
            self._answer = asyncio.get_running_loop().create_future()
        return self._answer

    def _settle(self, result: HookReviewResult) -> None:
        answer = self.answer()
        if not answer.done():
            answer.set_result(result)

    def dismiss(self, result: HookReviewResult | None = None) -> AwaitComplete:
        """Settle ``answer()`` first, then dismiss as Textual does.

        Settling here, synchronously, is the TASK-33621.28 fix: a waiting
        caller resumes without any pump flushing a result callback.

        Args:
            result: The review outcome; ``None`` settles the answer as
                ``HookReviewResult("cancel")``. Passed on to Textual unchanged.

        Returns:
            Textual's ``Screen.dismiss`` awaitable, which completes once the
            modal has been popped.
        """
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
                    self.snapshot.config.section.get(entry.source, [])
                    if isinstance(self.snapshot.config.section, dict)
                    else []
                )
                raw = (
                    raw_rows[entry.index]
                    if isinstance(raw_rows, list) and entry.index < len(raw_rows)
                    else {}
                )
                title = raw.get("name") if isinstance(raw, dict) else None
                title = (
                    title
                    if isinstance(title, str)
                    else entry.spec.event
                    + " / "
                    + (
                        entry.spec.id
                        if entry.source == "handler"
                        else Path(entry.spec.command[0]).name
                    )
                    if entry.spec
                    else "Invalid hook"
                )
                # One-line JSON escapes prevent terminal controls or markup in chrome.
                title = json.dumps(title, ensure_ascii=True)[1:-1][:48]
                with Horizontal(classes="hook-review-heading"):
                    yield Checkbox(
                        Text(f"{index + 1} - {title}"),
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
                        entry.error + " Repair in Settings or disable Console hooks.",
                        markup=False,
                    )
                with Horizontal(classes="hook-review-controls"):
                    if row.state == "approved":
                        yield Button(
                            "Revoke now",
                            id=f"hook-review-revoke-{index}",
                            classes="hook-review-action",
                        )
                    if (
                        entry.source == "hook"
                        and entry.enabled is not False
                        and isinstance(raw, dict)
                    ):
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
            self._submit_result(HookReviewResult("settings"))
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
                    if entry.source == "handler":
                        details = "Source: User config · v2\n" + json.dumps(
                            spec.model_dump(mode="json"), ensure_ascii=True, indent=2
                        )
                    else:
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
            self._submit_result(HookReviewResult("ready", self.snapshot))
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
                self._submit_result(HookReviewResult("ready", updated))
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
                except RuntimeError:
                    if self._runtime_result is None:
                        raise
                    # A displaced runtime review can refuse both the action
                    # and its refresh. Keep that refusal out of worker teardown.
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
            projection = self._console_hook_review_projection
            self._busy = (
                self._runtime_ready_pending or projection.busy
                if projection is not None
                else False
            )
            if self.is_mounted and self.app.screen is self:
                self._sync_actions()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        self._generation += 1
        self._cancel()
        self._submit_result(HookReviewResult("cancel"))

    def on_unmount(self) -> None:
        self._generation += 1
        if self._runtime_detach is not None:
            self._runtime_detach()
        elif not self._safe_dismiss_committed:
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


def project_runtime_hook_review(runtime: ConsoleRuntime, pending) -> bool | None:
    """Project the resident hook head synchronously, with no retained view callback."""
    view = runtime.view
    if view is None:
        return (
            False if getattr(pending, "decision_type", None) == "hook_review" else None
        )
    try:
        app = view.app
        current = app.screen
    except (AttributeError, RuntimeError):
        return (
            False if getattr(pending, "decision_type", None) == "hook_review" else None
        )
    selected = getattr(pending, "decision_type", None) == "hook_review"
    presented = [
        screen
        for screen in getattr(app, "screen_stack", (current,))
        if getattr(screen, "_console_hook_review_projection", None) is not None
    ]
    matching = None
    for screen in presented:
        prior = screen._console_hook_review_projection
        if (
            selected
            and prior.review_id == pending.decision_id
            and prior.generation == pending.payload["generation"]
            and prior.attachment_generation == runtime._attached_generation
        ):
            matching = screen
        else:
            dismiss = getattr(screen, "dismiss_safe_once_when_on_top", None)
            if callable(dismiss):
                dismiss(HookReviewResult("cancel"))
    if not selected:
        return None
    if not callable(getattr(app, "push_screen", None)):
        return False
    if pending.session_id != runtime.chat_controller.store.active_session_id:
        return False
    if matching is None and not runtime.has_answerable_view():
        return False
    host = runtime.chat_controller._interrupt_host
    projection = host.claim_hook_review_presentation(
        pending.decision_id,
        pending.payload["generation"],
        runtime._attached_generation,
        # ScreenResume can precede the popped modal's late Unmount release.
        replace_existing=matching is None,
    )
    if projection is None:
        return False
    if matching is not None:
        if (
            matching._console_hook_review_projection.presentation_token
            is projection.presentation_token
        ):
            matching.update_runtime_projection(projection)
            return True
        matching.dismiss_safe_once_when_on_top(HookReviewResult("cancel"))
        return False

    def apply(action, expected, keys=()):
        return runtime.apply_hook_review_action(
            projection.review_id,
            projection.generation,
            action,
            expected,
            tuple(keys),
            presentation_token=projection.presentation_token,
        )

    modal = ConsoleHooksReviewModal(
        snapshot=projection.snapshot,
        waiting_for_send=projection.waiting_for_send,
        approve=lambda saved, keys: apply("approve", saved, keys),
        revoke=lambda saved, key: apply("revoke", saved, (key,)),
        disable=lambda saved, key: apply("disable", saved, (key,)),
        recover=lambda: apply("recover", modal.snapshot),
        reset=lambda saved: apply("reset", saved),
        on_cancel=lambda: None,
        on_result=lambda result: runtime.resolve_initial_hook_review(
            projection.review_id,
            projection.generation,
            result,
            presentation_token=projection.presentation_token,
        ),
        on_detach=lambda: host.release_hook_review_presentation(
            projection.review_id, projection.generation, projection.presentation_token
        ),
        runtime_projection=projection,
    )
    try:
        app.push_screen(modal)
    except BaseException:
        host.release_hook_review_presentation(
            projection.review_id, projection.generation, projection.presentation_token
        )
        raise
    return True
