"""Console turn widgets and transcript memory presentations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from time import monotonic

from rich.console import Console
from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_ACTIVITY_STATUSES,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleActivityPresentation,
    ConsoleActivityStatus,
    RawCliPresentation,
    console_activity_status_word,
)


from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Chat.console_context_repository import (
    MemoryCoverageKind,
    MemoryOriginKind,
)


_RAW_CLI_ELAPSED_STATES = frozenset({"running", "stopping"})
_RAW_CLI_ELAPSED_TICK_SECONDS = 0.1


def raw_cli_status_copy(
    presentation: RawCliPresentation,
    *,
    now: float | None = None,
) -> str:
    """Return explicit, terminal-safe lifecycle copy for a raw command row."""
    labels = {
        "starting": "Starting",
        "running": "Running",
        "stopping": "Stopping…",
        "exited": (
            "Exited"
            if presentation.exit_code is None
            else f"Exited {presentation.exit_code}"
        ),
        "timed_out": "Timed out",
        "cancelled": "Stopped",
        "cleanup_unproven": "Cleanup unproven",
        "failed": "Failed",
    }
    elapsed = presentation.elapsed_seconds
    if (
        presentation.lifecycle_state in _RAW_CLI_ELAPSED_STATES
        and presentation.started_at_monotonic is not None
    ):
        elapsed = max(
            elapsed,
            (monotonic() if now is None else now)
            - presentation.started_at_monotonic,
        )
    copy = f"{labels[presentation.lifecycle_state]} · {elapsed:.1f}s"
    if (
        presentation.cleanup_proven is False
        and presentation.lifecycle_state not in _RAW_CLI_ELAPSED_STATES
        and presentation.lifecycle_state != "cleanup_unproven"
    ):
        copy += " · Cleanup unproven"
    return copy


def tool_activity_status_copy(presentation: ConsoleActivityPresentation) -> str:
    """Describe actual lifecycle state with execution-only elapsed time.

    Args:
        presentation: Current tool state and execution timing, when available.

    Returns:
        Status text with an elapsed duration only after execution starts.
    """
    label = {"success": "Succeeded", "stopped": "Stopped"}.get(
        presentation.status,
        console_activity_status_word(presentation.status).capitalize(),
    )
    elapsed = presentation.elapsed_seconds
    if (
        presentation.status == "running"
        and presentation.started_at_monotonic is not None
    ):
        elapsed = max(0.0, monotonic() - presentation.started_at_monotonic)
    return f"{label} · {elapsed:.1f}s" if elapsed is not None else label


class ConsoleToolPreview(Static):
    """Literal output preview bounded by painted rows, including its omission hint."""

    def __init__(self, activity_id: str) -> None:
        self._output = ""
        super().__init__(
            "",
            id=f"console-tool-preview-{activity_id}",
            classes="console-tool-preview",
            markup=False,
        )

    def set_output(self, output: str) -> None:
        """Rewrap retained output on lifecycle changes without replacing the widget.

        Args:
            output: Literal result text retained for width-dependent wrapping.
        """
        self._output = output
        self._rewrap()

    def on_resize(self, event: events.Resize) -> None:
        """Rewrap the preview using the widget's current content width.

        Args:
            event: Textual's notification that the widget dimensions changed.
        """
        self._rewrap()

    def _rewrap(self) -> None:
        width = max(1, self.content_size.width)
        lines = Text(self._output).wrap(Console(width=width), width, overflow="fold")
        if len(lines) > 3:
            hint = Text(f"… {len(lines) - 2} more lines — expand")
            hint.truncate(width, overflow="ellipsis")
            lines = list(lines[:2]) + [hint]
        self.update(Content("\n".join(line.plain for line in lines)))


class ConsoleActivityActivated(Message):
    """Request selection and, when available, disclosure-state toggling."""

    def __init__(self, activity_message_id: str, *, toggle_requested: bool) -> None:
        self.message_id = activity_message_id
        self.toggle_requested = toggle_requested
        super().__init__()

    @property
    def activity_message_id(self) -> str:
        """Backward-compatible alias for the canonical original message ID."""
        return self.message_id


class ConsoleActivityHeader(Horizontal):
    """Focusable literal-text header for one Assistant activity marker."""

    can_focus = True

    def __init__(
        self,
        activity_message_id: str,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool = False,
        expandable: bool = False,
        selected: bool = False,
        raw_cli_presentation: RawCliPresentation | None = None,
        tool_presentation: ConsoleActivityPresentation | None = None,
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self.tool_presentation = tool_presentation
        self._raw_cli_elapsed_timer: Timer | None = None
        self.label_widget = Static(
            self._label_content(),
            id=f"console-activity-label-{activity_message_id}",
            classes="console-activity-label",
            markup=False,
        )
        self.status_widget = Static(
            self._status_content(),
            id=f"console-activity-status-{activity_message_id}",
            classes="console-activity-status",
            markup=False,
        )
        super().__init__(
            self.label_widget,
            self.status_widget,
            id=f"console-activity-header-{activity_message_id}",
            classes="console-activity-header",
        )
        self._sync_classes()

    def _label_content(self) -> Content:
        """Build the flexible literal label without interpreting its text."""
        chevron = ""
        if self.expandable:
            chevron = "▾ " if self.expanded else "▸ "
        return Content(f"{chevron}{self.label}")

    def _status_content(self) -> Content:
        """Build the fixed terminal-status copy kept separate from the label."""
        if self.raw_cli_presentation is not None:
            return Content(f"· {raw_cli_status_copy(self.raw_cli_presentation)}")
        if self.tool_presentation is not None:
            return Content(f"· {tool_activity_status_copy(self.tool_presentation)}")
        return Content(f"· {console_activity_status_word(self.status)}")

    @property
    def renderable(self) -> Content:
        """Retain the former combined-text inspection seam for callers/tests."""
        return Content(
            f"{self._label_content().plain} {self._status_content().plain}"
        )

    def on_mount(self) -> None:
        """Own the elapsed repaint cadence for this command activity row."""
        self._sync_raw_cli_timer()

    def _tick_raw_cli_elapsed(self) -> None:
        if self.raw_cli_presentation is not None or self.tool_presentation is not None:
            # Once-per-second elapsed repaint of a `height: 1` CSS-pinned row --
            # content cannot change the box size, so skip the screen reflow
            # (the 21692/21595 Static.update layout=True default).
            self.status_widget.update(self._status_content(), layout=False)

    def _sync_raw_cli_timer(self) -> None:
        timer = self._raw_cli_elapsed_timer
        active = (
            self.raw_cli_presentation is not None
            and self.raw_cli_presentation.lifecycle_state in _RAW_CLI_ELAPSED_STATES
            and self.raw_cli_presentation.started_at_monotonic is not None
        )
        active = active or (
            self.tool_presentation is not None
            and self.tool_presentation.status == "running"
        )
        if active and timer is None:
            self._raw_cli_elapsed_timer = self.set_interval(
                _RAW_CLI_ELAPSED_TICK_SECONDS,
                self._tick_raw_cli_elapsed,
            )
        elif active:
            timer.resume()
        elif timer is not None:
            timer.stop()
            self._raw_cli_elapsed_timer = None

    def _sync_classes(self) -> None:
        self.status_widget.set_class(
            self.tool_presentation is not None, "console-activity-status-tool"
        )
        self.set_class(self.selected, "console-activity-header-selected")
        self.set_class(self.expanded, "console-activity-header-expanded")
        self.set_class(self.expandable, "console-activity-header-expandable")
        for status in CONSOLE_ACTIVITY_STATUSES:
            self.status_widget.set_class(
                self.status == status,
                f"console-activity-status-{status}",
            )

    def sync_header(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        expandable: bool,
        selected: bool,
        raw_cli_presentation: RawCliPresentation | None = None,
        tool_presentation: ConsoleActivityPresentation | None = None,
    ) -> None:
        """Project transcript-owned disclosure state onto this header."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self.tool_presentation = tool_presentation
        self._sync_classes()
        self.label_widget.update(self._label_content())
        self.status_widget.update(self._status_content())
        self._sync_raw_cli_timer()

    def _activate(self) -> None:
        self.post_message(
            ConsoleActivityActivated(
                self.activity_message_id,
                toggle_requested=self.expandable,
            )
        )

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.focus()
        self._activate()

    def on_key(self, event: events.Key) -> None:
        if event.key not in {"enter", "space"}:
            return
        event.stop()
        event.prevent_default()
        self._activate()


class ConsoleActivityDisclosure(Vertical):
    """Externally controlled disclosure for one structured activity marker."""

    def __init__(
        self,
        activity_message_id: str,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool = False,
        selected: bool = False,
        action_widgets: Iterable[Widget] = (),
        detail_widgets: Iterable[Widget] = (),
        detail_available: bool | None = None,
        raw_cli_presentation: RawCliPresentation | None = None,
        tool_presentation: ConsoleActivityPresentation | None = None,
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self.tool_presentation = tool_presentation
        action_children = tuple(action_widgets)
        detail_children = tuple(detail_widgets)
        self._has_actions = bool(action_children)
        self.detail_available = (
            bool(detail_children) if detail_available is None else detail_available
        )
        self._has_detail = bool(detail_children)
        self.header = ConsoleActivityHeader(
            activity_message_id,
            label,
            status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=raw_cli_presentation,
            tool_presentation=tool_presentation,
        )
        self.preview = ConsoleToolPreview(activity_message_id)
        self.approval_button = Button(
            "Review approval",
            id=f"console-tool-approval-{activity_message_id}",
            classes="console-tool-approval",
        )
        self.action_stack = Vertical(
            *action_children,
            id=f"console-activity-actions-{activity_message_id}",
            classes="console-activity-action-stack",
        )
        self.detail_stack = Vertical(
            *detail_children,
            id=f"console-activity-detail-{activity_message_id}",
            classes="console-activity-detail-stack",
        )
        super().__init__(
            self.header,
            *((self.preview, self.approval_button) if tool_presentation else ()),
            self.action_stack,
            self.detail_stack,
            id=f"console-activity-disclosure-{activity_message_id}",
            classes="console-activity-disclosure",
        )
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        self.set_class(self.selected, "console-activity-disclosure-selected")
        self.set_class(self.expanded, "console-activity-disclosure-expanded")
        tool = self.tool_presentation
        self.preview.display = bool(
            tool is not None and tool.result_preview is not None and not self.expanded
        )
        if tool is not None and tool.result_preview is not None:
            self.preview.set_output(tool.result_preview or "Completed — no output")
        self.approval_button.display = bool(
            tool is not None and tool.status == "awaiting_approval"
        )
        self.action_stack.display = self.selected and self._has_actions
        self.detail_stack.display = self.expanded and self._has_detail

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Reach the existing approval card without adding a second decision surface."""
        if event.button is self.approval_button:
            event.stop()
            await self.app.run_action("screen.review_pending_approval")

    async def replace_detail_widgets(self, detail_widgets: Iterable[Widget]) -> None:
        """Replace lazy detail children without replacing the disclosure."""
        replacements = tuple(detail_widgets)
        if self.detail_stack.children:
            await self.detail_stack.remove_children()
        if replacements:
            await self.detail_stack.mount(*replacements)
        self._has_detail = bool(replacements)
        self.header.sync_header(
            self.label,
            self.status,
            expanded=self.expanded,
            expandable=self.detail_available,
            selected=self.selected,
        )
        self._sync_visibility()

    def sync_state(self, *, expanded: bool, selected: bool) -> None:
        """Apply transcript-owned selection and expansion state in place."""
        self.expanded = expanded
        self.selected = selected
        self.header.sync_header(
            self.label,
            self.status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=self.raw_cli_presentation,
            tool_presentation=self.tool_presentation,
        )
        self._sync_visibility()

    def sync_activity(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        selected: bool,
        raw_cli_presentation: RawCliPresentation | None = None,
        tool_presentation: ConsoleActivityPresentation | None = None,
    ) -> None:
        """Apply new structured copy and transcript-owned state in place."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self.tool_presentation = tool_presentation
        self.header.sync_header(
            label,
            status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=raw_cli_presentation,
            tool_presentation=tool_presentation,
        )
        self._sync_visibility()


class ConsoleAssistantTurnWidget(Vertical):
    """Stable Assistant header/activity/answer/adjunct presentation shell."""

    def __init__(
        self,
        assistant_message_id: str,
        header_widget: Widget,
        activity_widgets: Iterable[Widget],
        answer_widget: Widget,
        adjunct_widgets: Iterable[Widget] = (),
    ) -> None:
        self.assistant_message_id = assistant_message_id
        self.header_widget = header_widget
        self.answer_widget = answer_widget
        self.activity_stack = Vertical(
            *tuple(activity_widgets),
            id=f"console-assistant-activities-{assistant_message_id}",
            classes="console-assistant-activity-stack",
        )
        self.adjunct_stack = Vertical(
            *tuple(adjunct_widgets),
            id=f"console-assistant-adjuncts-{assistant_message_id}",
            classes="console-assistant-adjunct-stack",
        )
        super().__init__(
            header_widget,
            self.activity_stack,
            answer_widget,
            self.adjunct_stack,
            id=f"console-assistant-turn-{assistant_message_id}",
            classes="console-assistant-turn",
        )

    async def replace_activity_widgets(
        self, activity_widgets: Iterable[Widget]
    ) -> None:
        """Replace only mounted activity children, retaining turn identity."""
        replacements = tuple(activity_widgets)
        if self.activity_stack.children:
            await self.activity_stack.remove_children()
        if replacements:
            await self.activity_stack.mount(*replacements)


#: SP2 /rewind: render-derived (never a tree node) one-line banner shown above
#: the boundary message when "summarize up to here" is in effect.
CONSOLE_SUMMARY_BANNER_COPY = (
    "⤵ Earlier turns summarized for context — full history above"
)


@dataclass(frozen=True, slots=True)
class ConsoleMemoryBannerPresentation:
    """One content-free banner derived from validated effective memory."""

    kind: Literal["prefix", "range"]
    render_anchor_message_id: str
    start_message_id: str | None
    end_message_id: str
    copy: str

    def __post_init__(self) -> None:
        if self.kind not in {"prefix", "range"}:
            raise ValueError("memory banner kind must be prefix or range")
        for name in ("render_anchor_message_id", "end_message_id", "copy"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"memory banner {name} must be non-empty text")
        if self.kind == "range":
            if not isinstance(self.start_message_id, str) or not self.start_message_id:
                raise ValueError("range memory banner requires a start identity")
        elif self.start_message_id is not None:
            raise ValueError("prefix memory banner cannot carry a start identity")


def derive_console_memory_banner_presentation(
    effective: EffectiveMemoryResult,
    active_messages: Iterable[ConsoleChatMessage],
) -> ConsoleMemoryBannerPresentation | None:
    """Derive one banner from the same typed memory result used by dispatch.

    Every persisted identity lookup is exact. Missing or duplicate anchors,
    malformed effective state, and prefix memories without a real placement
    row return ``None`` rather than guessing from content or proximity.

    Args:
        effective: Validated effective-memory result used by provider dispatch.
        active_messages: Ordered native messages on the visible active branch.

    Returns:
        A content-free banner presentation, or ``None`` when exact placement
        cannot be proven.

    Raises:
        TypeError: If ``effective`` is not an ``EffectiveMemoryResult``.
    """

    if not isinstance(effective, EffectiveMemoryResult):
        raise TypeError("effective must be an EffectiveMemoryResult")
    rows = tuple(active_messages)
    positions_by_persisted_id: dict[str, list[int]] = {}
    for index, message in enumerate(rows):
        persisted_id = message.persisted_message_id
        if isinstance(persisted_id, str) and persisted_id:
            positions_by_persisted_id.setdefault(persisted_id, []).append(index)

    def exact_index(persisted_id: str | None) -> int | None:
        if not isinstance(persisted_id, str) or not persisted_id:
            return None
        matches = positions_by_persisted_id.get(persisted_id, ())
        return matches[0] if len(matches) == 1 else None

    def prefix_presentation(
        *, render_index: int, end_message_id: str
    ) -> ConsoleMemoryBannerPresentation:
        return ConsoleMemoryBannerPresentation(
            kind="prefix",
            render_anchor_message_id=rows[render_index].id,
            start_message_id=None,
            end_message_id=end_message_id,
            copy=CONSOLE_SUMMARY_BANNER_COPY,
        )

    if effective.kind is EffectiveMemoryKind.RAW:
        return None
    if effective.kind is EffectiveMemoryKind.LEGACY_PREFIX:
        legacy = effective.legacy
        if legacy is None:
            return None
        boundary_index = exact_index(legacy.boundary_message_id)
        if boundary_index is None:
            return None
        return prefix_presentation(
            render_index=boundary_index,
            end_message_id=legacy.boundary_message_id,
        )

    memory = effective.memory
    scope = effective.scope
    if (
        memory is None
        or scope is None
        or not memory.active
        or memory.source_kind != "generated"
        or memory.memory_id != scope.memory_id
        or memory.conversation_id != scope.conversation_id
    ):
        return None
    boundary_index = exact_index(memory.boundary_message_id)
    if boundary_index is None:
        return None

    if effective.kind is EffectiveMemoryKind.GENERATED_RANGE:
        if (
            scope.coverage_kind is not MemoryCoverageKind.RANGE
            or scope.origin_kind is not MemoryOriginKind.MANUAL_REWIND
        ):
            return None
        start_index = exact_index(scope.selection_anchor_message_id)
        if (
            start_index is None
            or start_index >= boundary_index
            or rows[start_index].role is not ConsoleMessageRole.USER
        ):
            return None
        user_ordinals: dict[int, int] = {}
        ordinal = 0
        for index, message in enumerate(rows):
            if message.role is ConsoleMessageRole.USER:
                ordinal += 1
                user_ordinals[index] = ordinal
        start_ordinal = user_ordinals.get(start_index)
        end_ordinal = next(
            (
                user_ordinals[index]
                for index in range(boundary_index, start_index - 1, -1)
                if index in user_ordinals
            ),
            None,
        )
        if start_ordinal is None or end_ordinal is None:
            return None
        return ConsoleMemoryBannerPresentation(
            kind="range",
            render_anchor_message_id=rows[start_index].id,
            start_message_id=scope.selection_anchor_message_id,
            end_message_id=memory.boundary_message_id,
            copy=(
                "Context uses a summary of turns "
                f"#{start_ordinal}-#{end_ordinal} - full transcript remains visible."
            ),
        )

    if (
        effective.kind is not EffectiveMemoryKind.GENERATED_PREFIX
        or scope.coverage_kind is not MemoryCoverageKind.PREFIX
    ):
        return None
    if scope.origin_kind is MemoryOriginKind.MANUAL_REWIND:
        anchor_index = exact_index(scope.selection_anchor_message_id)
        if (
            anchor_index is None
            or boundary_index >= anchor_index
            or rows[anchor_index].role is not ConsoleMessageRole.USER
        ):
            return None
        return prefix_presentation(
            render_index=anchor_index,
            end_message_id=memory.boundary_message_id,
        )
    if (
        scope.origin_kind is not MemoryOriginKind.AUTOMATIC
        or scope.selection_anchor_message_id is not None
    ):
        return None
    next_user_index = next(
        (
            index
            for index in range(boundary_index + 1, len(rows))
            if rows[index].role is ConsoleMessageRole.USER
        ),
        None,
    )
    if next_user_index is None:
        return None
    return prefix_presentation(
        render_index=next_user_index,
        end_message_id=memory.boundary_message_id,
    )
