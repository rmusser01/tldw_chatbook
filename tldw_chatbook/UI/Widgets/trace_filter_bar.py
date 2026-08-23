"""Single-owner structured filters for the Console Trace screen."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Select, Static

from tldw_chatbook.Chat.trajectory import TrajectoryRecord

__all__ = [
    "TraceFilterBar",
    "TraceFilterOptions",
    "TraceFilterState",
    "TraceFiltersDialog",
]


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _agent_value(record: TrajectoryRecord) -> str | None:
    kind = (record.actor_kind or "").strip().lower()
    if kind not in {"agent", "primary", "subagent", "child_agent"}:
        return None
    return (record.run_id or "").strip() or None


@dataclass(frozen=True)
class TraceFilterOptions:
    kinds: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceFilterState:
    """All structured filter truth, including the timeline time brush."""

    kind: str | None = None
    status: str | None = None
    agent: str | None = None
    provider: str | None = None
    time_range: tuple[float, float] | None = None

    @property
    def active_count(self) -> int:
        return sum(
            value is not None
            for value in (
                self.kind,
                self.status,
                self.agent,
                self.provider,
                self.time_range,
            )
        )

    @property
    def is_active(self) -> bool:
        return self.active_count > 0

    @property
    def summary(self) -> str:
        parts = []
        for label, value in (
            ("Kind", self.kind),
            ("State", self.status),
            ("Agent", self.agent),
            ("Provider", self.provider),
        ):
            if value:
                parts.append(f"{label}: {_humanize(value)}")
        if self.time_range is not None:
            parts.append("Time range")
        return " · ".join(parts) if parts else "No structured filters"

    def matches(self, record: TrajectoryRecord) -> bool:
        if self.kind is not None and record.kind != self.kind:
            return False
        if self.status is not None and record.status != self.status:
            return False
        if self.agent is not None and _agent_value(record) != self.agent:
            return False
        if self.provider is not None and record.provider != self.provider:
            return False
        if self.time_range is not None:
            if record.step_started_at is None:
                return False
            start = record.step_started_at
            end = record.completed_at
            if end is None or end < start:
                end = start
            lo, hi = sorted(self.time_range)
            if start > hi or end < lo:
                return False
        return True

    @staticmethod
    def options_from(records: Iterable[TrajectoryRecord]) -> TraceFilterOptions:
        material = tuple(records)
        return TraceFilterOptions(
            kinds=tuple(sorted({record.kind for record in material if record.kind})),
            statuses=tuple(
                sorted({record.status for record in material if record.status})
            ),
            agents=tuple(
                sorted(
                    {
                        agent
                        for record in material
                        if (agent := _agent_value(record)) is not None
                    }
                )
            ),
            providers=tuple(
                sorted({record.provider for record in material if record.provider})
            ),
        )


class TraceFiltersDialog(ModalScreen[TraceFilterState | None]):
    """Compact-terminal editor for the same state used by wide controls."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("a", "apply", "Apply"),
        Binding("x", "clear", "Clear"),
    ]

    BUNDLED_SCREEN_CSS = """
    TraceFiltersDialog { align: center middle; background: $background 70%; }
    #trace-filter-dialog { width: 58; max-width: 94%; height: 17; padding: 0 2; background: $panel; }
    #trace-filter-dialog-title { height: 1; text-style: bold; }
    #trace-filter-dialog-time { height: 1; color: $text-muted; }
    TraceFiltersDialog #trace-filter-dialog Select { width: 1fr; min-width: 0; height: 3; margin: 0; }
    #trace-filter-dialog-actions { height: 3; align-horizontal: right; }
    #trace-filter-dialog-actions Button { min-width: 9; }
    """

    def __init__(self, state: TraceFilterState, options: TraceFilterOptions) -> None:
        super().__init__()
        self._state = state
        self._options = options

    @staticmethod
    def _select(
        prompt: str, values: Sequence[str], value: str | None, widget_id: str
    ) -> Select[str]:
        select = Select(
            [(_humanize(item), item) for item in values],
            prompt=prompt,
            allow_blank=True,
            value=value if value is not None else Select.NULL,
            id=widget_id,
            compact=True,
        )
        # Inline geometry wins over the later app stylesheet's global
        # ``Select { width: 100%; margin-bottom: 1; }`` rule.
        select.styles.width = "1fr"
        select.styles.min_width = 0
        select.styles.margin = 0
        return select

    def compose(self) -> ComposeResult:
        with Vertical(id="trace-filter-dialog"):
            yield Static("Trace filters", id="trace-filter-dialog-title")
            yield self._select(
                "Event kind", self._options.kinds, self._state.kind, "dialog-kind"
            )
            yield self._select(
                "State", self._options.statuses, self._state.status, "dialog-status"
            )
            yield self._select(
                "Agent", self._options.agents, self._state.agent, "dialog-agent"
            )
            yield self._select(
                "Provider",
                self._options.providers,
                self._state.provider,
                "dialog-provider",
            )
            time_text = (
                "Time: timeline range active" if self._state.time_range else "Time: all"
            )
            yield Static(time_text, id="trace-filter-dialog-time")
            with Horizontal(id="trace-filter-dialog-actions"):
                yield Button("Clear", id="dialog-clear")
                yield Button("Cancel", id="dialog-cancel")
                yield Button("Apply", id="dialog-apply", variant="primary")

    @staticmethod
    def _value(select: Select[str]) -> str | None:
        return None if select.value is Select.NULL else str(select.value)

    def _edited_state(self) -> TraceFilterState:
        return replace(
            self._state,
            kind=self._value(self.query_one("#dialog-kind", Select)),
            status=self._value(self.query_one("#dialog-status", Select)),
            agent=self._value(self.query_one("#dialog-agent", Select)),
            provider=self._value(self.query_one("#dialog-provider", Select)),
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply(self) -> None:
        self.dismiss(self._edited_state())

    def action_clear(self) -> None:
        self.dismiss(TraceFilterState())

    @on(Button.Pressed)
    def _on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "dialog-apply":
            self.action_apply()
        elif event.button.id == "dialog-clear":
            self.action_clear()
        else:
            self.action_cancel()


class TraceFilterBar(Widget):
    """Responsive filter controls and counts with one immutable state owner."""

    can_focus = True

    BINDINGS = [Binding("enter", "open_filters", "Filters", show=False)]

    BUNDLED_CSS = """
    TraceFilterBar { width: 1fr; height: 3; }
    #trace-filter-wide { width: 1fr; height: 3; }
    #trace-filter-counts { width: 20; height: 3; content-align: left middle; color: $text-muted; }
    TraceFilterBar #trace-filter-wide > Select { width: 1fr; min-width: 0; height: 3; margin: 0; }
    #trace-filter-compact { width: 1fr; height: 1; color: $text-muted; }
    TraceFilterBar:focus #trace-filter-compact { text-style: reverse; color: $text; }
    """

    class Changed(Message):
        def __init__(self, state: TraceFilterState) -> None:
            super().__init__()
            self.state = state

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.state = TraceFilterState()
        self.options = TraceFilterOptions()
        self.shown_count = 0
        self.matching_count = 0
        self.total_count = 0
        self._compact = False
        self._syncing = False

    @property
    def compact(self) -> bool:
        return self._compact

    @property
    def summary_text(self) -> str:
        return (
            f"{self.count_text} · {self.state.active_count} active · "
            f"{self.state.summary}"
        )

    @property
    def count_text(self) -> str:
        return (
            f"{self.shown_count} shown · {self.matching_count} matches · "
            f"{self.total_count} total"
        )

    @property
    def wide_count_text(self) -> str:
        return (
            f"Shown {self.shown_count}\n"
            f"Matches {self.matching_count}\n"
            f"Total {self.total_count}"
        )

    @property
    def visible_count(self) -> int:
        """Compatibility alias for the total number of filter matches."""

        return self.matching_count

    def render(self) -> str:
        return self.summary_text

    @staticmethod
    def _select(prompt: str, widget_id: str) -> Select[str]:
        select = Select([], prompt=prompt, allow_blank=True, id=widget_id, compact=True)
        select.styles.width = "1fr"
        select.styles.min_width = 0
        select.styles.margin = 0
        return select

    def compose(self) -> ComposeResult:
        with Horizontal(id="trace-filter-wide"):
            yield Static("", id="trace-filter-counts", markup=False)
            yield self._select("Kind", "trace-filter-kind")
            yield self._select("State", "trace-filter-status")
            yield self._select("Agent", "trace-filter-agent")
            yield self._select("Provider", "trace-filter-provider")
        yield Static("", id="trace-filter-compact", markup=False)

    def on_mount(self) -> None:
        self._refresh_options()
        self._sync_controls()
        self._refresh_presentation()

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        self.can_focus = compact
        self.styles.height = 1 if compact else 3
        self._refresh_presentation()

    def set_records(self, records: Iterable[TrajectoryRecord]) -> None:
        derived = TraceFilterState.options_from(records)

        def retain(values: tuple[str, ...], selected: str | None) -> tuple[str, ...]:
            return tuple(sorted(set(values) | ({selected} if selected else set())))

        self.options = TraceFilterOptions(
            kinds=retain(derived.kinds, self.state.kind),
            statuses=retain(derived.statuses, self.state.status),
            agents=retain(derived.agents, self.state.agent),
            providers=retain(derived.providers, self.state.provider),
        )
        self._refresh_options()
        self._sync_controls()

    def update_counts(self, shown: int, matching: int, total: int) -> None:
        self.shown_count = shown
        self.matching_count = matching
        self.total_count = total
        self._refresh_presentation()
        self.refresh()

    def set_state(self, state: TraceFilterState, *, emit: bool = True) -> None:
        if state == self.state:
            return
        self.state = state
        self._sync_controls()
        self._refresh_presentation()
        self.refresh()
        if emit:
            self.post_message(self.Changed(state))

    def set_time_range(self, time_range: tuple[float, float] | None) -> None:
        self.set_state(replace(self.state, time_range=time_range))

    def clear(self) -> None:
        self.set_state(TraceFilterState())

    def _refresh_options(self) -> None:
        if not self.is_mounted:
            return
        for widget_id, values in (
            ("#trace-filter-kind", self.options.kinds),
            ("#trace-filter-status", self.options.statuses),
            ("#trace-filter-agent", self.options.agents),
            ("#trace-filter-provider", self.options.providers),
        ):
            self.query_one(widget_id, Select).set_options(
                [(_humanize(value), value) for value in values]
            )

    def _sync_controls(self) -> None:
        if not self.is_mounted:
            return
        self._syncing = True
        try:
            for widget_id, value in (
                ("#trace-filter-kind", self.state.kind),
                ("#trace-filter-status", self.state.status),
                ("#trace-filter-agent", self.state.agent),
                ("#trace-filter-provider", self.state.provider),
            ):
                self.query_one(widget_id, Select).value = (
                    value if value is not None else Select.NULL
                )
        finally:
            self._syncing = False

    def _refresh_presentation(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#trace-filter-wide").display = not self._compact
        self.query_one("#trace-filter-counts", Static).update(self.wide_count_text)
        compact = self.query_one("#trace-filter-compact", Static)
        compact.display = self._compact
        copy = f"{self.count_text} · {self.state.active_count} active · g filters"
        compact.update(Text(copy, no_wrap=True, overflow="ellipsis"))

    @on(Select.Changed)
    def _on_select_changed(self, event: Select.Changed) -> None:
        if (
            self._syncing
            or not event.select.id
            or not event.select.id.startswith("trace-filter-")
            or event.value != event.select.value
        ):
            return
        field = event.select.id.removeprefix("trace-filter-")
        value = None if event.value is Select.NULL else str(event.value)
        self.set_state(replace(self.state, **{field: value}))

    def _dialog_dismissed(self, result: TraceFilterState | None) -> None:
        if result is not None:
            self.set_state(result)

    async def open_dialog(self) -> None:
        await self.app.push_screen(
            TraceFiltersDialog(self.state, self.options),
            callback=self._dialog_dismissed,
        )

    async def action_open_filters(self) -> None:
        """Open the compact editor from the bar's actionable Tab stop."""

        if self._compact:
            await self.open_dialog()
