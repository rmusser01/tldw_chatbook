"""Local Change Review controls that preserve surrounding workspace drafts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_RETENTION_DAYS,
    change_review_setting,
)
from tldw_chatbook.Workspaces.change_review_consent import (
    ChangeReviewConsent,
    ChangeReviewState,
    ChangeReviewStateConflict,
    ChangeReviewStatus,
    RootReadinessState,
)

_PREFIX = "settings-workspace-change-review-"
_STATUS_ROWS = (
    "unavailable",
    "global-off",
    "retention",
    "state",
    "preparing",
    "failed",
    "ready",
)


class _ConsentToggle(Button):
    """Capture the visible intent when Textual posts an activation."""

    intent: tuple[ChangeReviewConsent, bool] | None = None

    def post_message(self, message: Message) -> bool:
        if isinstance(message, Button.Pressed) and message.button is self:
            message.change_review_intent = self.intent
        return super().post_message(message)


class _ChangeReviewReceipt(Static):
    def on_resize(self) -> None:
        if isinstance(self.parent, WorkspaceChangeReviewPanel):
            self.parent.reveal_result()


class WorkspaceChangeReviewPanel(Vertical):
    """Observe existing consent/readiness; keep controls and input neighbors intact."""

    BUNDLED_CSS = """
    WorkspaceChangeReviewPanel {
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        read_status: Callable[[], ChangeReviewStatus | None],
        toggle: Callable[[ChangeReviewConsent, bool], object],
        retry: Callable[[], int],
        git_available: bool,
    ) -> None:
        super().__init__(id=_PREFIX + "panel")
        self._read_status = read_status
        self._toggle = toggle
        self._retry = retry
        self._git_available = git_available
        self._status = read_status() if git_available else None
        self._result_origin: Button | None = None
        self._poll_worker = None

    def compose(self) -> ComposeResult:
        yield Static("Change review (post-run diffs)", classes="destination-section")
        for name in _STATUS_ROWS:
            yield Static(
                "", id=_PREFIX + name, classes="settings-detail-row", markup=False
            )
        yield Button("Retry failed preparation", id=_PREFIX + "retry", compact=True)
        yield _ConsentToggle(
            "Enable change review", id=_PREFIX + "toggle", compact=True
        )
        yield _ChangeReviewReceipt(
            "", id=_PREFIX + "result", classes="settings-status-row", markup=False
        )

    def on_mount(self) -> None:
        self._paint_status()
        self.set_interval(0.5, self._poll_preparing)

    def _current(self) -> bool:
        return self.is_attached and self.app.screen is self.screen

    def _poll_preparing(self) -> None:
        if (
            not self._current()
            or self._status is None
            or not any(
                root.state is RootReadinessState.PREPARING
                for root in self._status.roots
            )
            or (self._poll_worker is not None and self._poll_worker.is_running)
        ):
            return
        self._poll_worker = self._refresh_preparing()

    @work(exclusive=True, exit_on_error=False)
    async def _refresh_preparing(self) -> None:
        observed = self._status
        latest = await asyncio.to_thread(self._read_status)
        # A toggle/retry may have installed a newer observation during this read.
        if self._current() and self._status is observed and latest != observed:
            self._status = latest
            self._paint_status()

    def _row(self, name: str, text: str = "") -> None:
        row = self.query_one("#" + _PREFIX + name, Static)
        row.update(text)
        row.display = bool(text)

    def _paint_status(self) -> None:
        toggle = self.query_one("#" + _PREFIX + "toggle", _ConsentToggle)
        retry = self.query_one("#" + _PREFIX + "retry", Button)
        retry_focused = self.app.focused is retry
        for name in _STATUS_ROWS:
            self._row(name)
        can_toggle = can_retry = False
        status = self._status
        if not self._git_available:
            self._row("unavailable", "Change review needs git — install git to enable.")
        elif (
            status is not None and status.capability.state is ChangeReviewState.DISABLED
        ):
            self._row(
                "global-off",
                "Change review is disabled globally ([change_review] enabled = false).",
            )
        elif (
            status is None
            or status.capability.state is ChangeReviewState.UNAVAILABLE
            or status.consent.state is ChangeReviewState.UNAVAILABLE
        ):
            self._row(
                "unavailable",
                "Change Review state could not be read; chat and tools continue.",
            )
        else:
            retention = change_review_setting("retention_days", DEFAULT_RETENTION_DAYS)
            self._row(
                "retention",
                "Change Review stores shadow Git history in application data, "
                f"including file contents, for {retention} days by default. "
                "Disabling stops new review snapshots but does not erase existing history.",
            )
            enabled = status.consent.state is ChangeReviewState.ENABLED
            self._row(
                "state",
                "Tracking enabled: agent runs record per-turn diffs for this workspace's folders."
                if enabled
                else "Tracking disabled for this workspace: runs record no diffs and offer no review.",
            )
            if enabled:
                counts = {
                    state: sum(root.state is state for root in status.roots)
                    for state in RootReadinessState
                }
                if count := counts[RootReadinessState.PREPARING]:
                    self._row(
                        "preparing",
                        f"Preparing change history for {count} folder(s) in the background; chat and tools continue.",
                    )
                if count := counts[RootReadinessState.FAILED]:
                    self._row(
                        "failed",
                        f"Change history preparation failed for {count} folder(s); chat and tools continue.",
                    )
                    can_retry = True
                if count := counts[RootReadinessState.READY]:
                    self._row("ready", f"Change history ready for {count} folder(s).")
            toggle.label = (
                "Disable change review" if enabled else "Enable change review"
            )
            toggle.intent = (status.consent, not enabled)
            can_toggle = True
        toggle.display, toggle.disabled = can_toggle, not can_toggle
        retry.display, retry.disabled = can_retry, not can_retry
        if retry_focused and not retry.display and toggle.display:
            toggle.focus()
            if self._result_origin is retry:
                self._result_origin = toggle
        self.call_after_refresh(self.reveal_result)

    def _publish(self, text: str, origin: Button) -> None:
        self._result_origin = origin
        self.query_one("#" + _PREFIX + "result", Static).update(text)
        self._status = self._read_status()
        self._paint_status()

    def reveal_result(self) -> None:
        """Reveal the active action and its receipt without moving newer focus."""
        if not self._current():
            return
        focused = self.app.focused
        if focused not in self.query(Button) or not focused.display:
            return
        receipt = self.query_one("#" + _PREFIX + "result", Static)
        if focused is self._result_origin and receipt.renderable:
            children = list(self.children)
            if children.index(receipt) != children.index(focused) + 1:
                self.move_child(receipt, after=focused)
                self.call_after_refresh(self.reveal_result)
                return
            receipt.scroll_visible(animate=False, immediate=True)
        else:
            focused.scroll_visible(animate=False, immediate=True)

    @on(Button.Pressed, "#settings-workspace-change-review-toggle")
    def toggle_review(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._current() or event.button.disabled:
            return
        intent = getattr(event, "change_review_intent", None)
        if (
            not isinstance(intent, tuple)
            or len(intent) != 2
            or not isinstance(intent[0], ChangeReviewConsent)
            or not isinstance(intent[1], bool)
        ):
            return
        expected, enabled = intent
        try:
            self._toggle(expected, enabled)
        except ChangeReviewStateConflict:
            text = "Change Review changed elsewhere; refreshed current state."
        except Exception:  # noqa: BLE001 - do not expose storage paths or backend details
            text = "Change Review state could not be changed. Try again."
        else:
            text = (
                "Change Review enabled."
                if enabled
                else "Change Review disabled; existing history is retained."
            )
        self._publish(text, event.button)

    @on(Button.Pressed, "#settings-workspace-change-review-retry")
    def retry_preparation(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._current() or event.button.disabled:
            return
        try:
            scheduled = self._retry()
        except Exception:  # noqa: BLE001 - failure is different from no eligible roots
            text = "Change Review preparation could not be retried. Try again."
        else:
            text = (
                f"Retry scheduled for {scheduled} folder(s)."
                if scheduled
                else "No failed Change Review folders were ready to retry."
            )
        self._publish(text, event.button)
