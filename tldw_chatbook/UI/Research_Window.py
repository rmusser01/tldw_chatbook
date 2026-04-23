"""Research Sessions source-switched TUI window."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static

from tldw_chatbook.UI.Research_Modules import ResearchController


class ResearchWindow(Vertical):
    """Research Sessions container for local/server run browsing."""

    def __init__(self, app_instance: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_source = "local"
        self.runs: list[Any] = []
        self.selected_run: Any | None = None
        self.status_message = ""
        self.controller = ResearchController(
            getattr(app_instance, "research_scope_service", None)
        )

    def compose(self) -> ComposeResult:
        yield Label("Research Sessions")
        with Horizontal(id="research-toolbar"):
            yield Select(
                [("Local", "local"), ("Server", "server")],
                value=self.current_source,
                allow_blank=False,
                id="research-source-select",
            )
            yield Button("Refresh", id="research-refresh-runs")
        with Horizontal(id="research-create-row"):
            yield Input(placeholder="Research question", id="research-query-input")
            yield Button("Create Run", id="research-create-run", variant="primary")
        yield Static(self.status_message, id="research-status")
        with Horizontal(id="research-body"):
            yield ListView(id="research-run-list")
            with Vertical(id="research-detail-panel"):
                yield Static("No research run selected.", id="research-run-detail")
                with Horizontal(id="research-run-actions"):
                    yield Button("Resume", id="research-resume-run")
                    yield Button("Pause", id="research-pause-run")
                    yield Button("Cancel", id="research-cancel-run", variant="error")

    def save_state(self) -> dict[str, Any]:
        return {"source": self.current_source}

    def restore_state(self, state: dict[str, Any]) -> None:
        source = str((state or {}).get("source") or "local").strip().lower()
        self.current_source = source if source in {"local", "server"} else "local"

    async def switch_source(self, source: str) -> list[Any]:
        self.current_source = self._normalize_source(source)
        self.runs = []
        self.selected_run = None
        self._set_status("")
        return await self.load_runs(self.current_source)

    async def load_runs(self, source: str | None = None) -> list[Any]:
        selected_source = self._normalize_source(source or self.current_source)
        self.current_source = selected_source
        try:
            self.runs = await self.controller.load_runs(selected_source)
        except Exception as exc:
            self.runs = []
            self._set_status(str(exc))
            await self._refresh_run_list()
            return []
        self._set_status(f"Loaded {len(self.runs)} {selected_source} research run(s).")
        await self._refresh_run_list()
        return self.runs

    async def create_run(self, payload: dict[str, Any]) -> Any:
        created = await self.controller.create_run(self.current_source, payload)
        await self.load_runs(self.current_source)
        return created

    def select_run(self, run: Any) -> None:
        self.selected_run = run
        self._update_detail(run)

    async def pause_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.pause_run(self.current_source, run_id)
        self.select_run(updated)
        return updated

    async def resume_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.resume_run(self.current_source, run_id)
        self.select_run(updated)
        return updated

    async def cancel_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.cancel_run(self.current_source, run_id)
        self.select_run(updated)
        return updated

    async def _refresh_run_list(self) -> None:
        if not self.is_mounted:
            return
        list_view = self.query_one("#research-run-list", ListView)
        await list_view.clear()
        for run in self.runs:
            title = self._run_title(run)
            item = ListItem(Static(title), id=f"research-run-{self._record_get(run, 'id')}")
            item.run_record = run
            await list_view.append(item)
        try:
            self.query_one("#research-status", Static).update(self.status_message)
        except Exception:
            pass

    def _update_detail(self, run: Any) -> None:
        detail = (
            f"{self._run_title(run)}\n"
            f"Status: {self._record_get(run, 'status', 'unknown')}\n"
            f"Phase: {self._record_get(run, 'phase', 'unknown')}\n"
            f"Control: {self._record_get(run, 'control_state', 'unknown')}"
        )
        if not self.is_mounted:
            return
        try:
            self.query_one("#research-run-detail", Static).update(detail)
        except Exception:
            pass

    def _set_status(self, message: str) -> None:
        self.status_message = message
        if not self.is_mounted:
            return
        try:
            self.query_one("#research-status", Static).update(message)
        except Exception:
            pass

    def _selected_run_id(self) -> str:
        if self.selected_run is None:
            raise ValueError("No research run is selected.")
        return str(self._record_get(self.selected_run, "id") or "")

    @staticmethod
    def _normalize_source(source: str) -> str:
        return source if source in {"local", "server"} else "local"

    @staticmethod
    def _record_get(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    def _run_title(self, run: Any) -> str:
        query = str(self._record_get(run, "query", "") or "").strip()
        run_id = str(self._record_get(run, "id", "") or "").strip()
        return query or run_id or "Untitled research run"

    @on(Select.Changed, "#research-source-select")
    async def _on_source_changed(self, event: Select.Changed) -> None:
        await self.switch_source(str(event.value or "local"))

    @on(Button.Pressed, "#research-refresh-runs")
    async def _on_refresh_pressed(self, _event: Button.Pressed) -> None:
        await self.load_runs(self.current_source)

    @on(Button.Pressed, "#research-create-run")
    async def _on_create_pressed(self, _event: Button.Pressed) -> None:
        query = ""
        try:
            query = self.query_one("#research-query-input", Input).value.strip()
        except Exception:
            pass
        if not query:
            self._set_status("Research query is required.")
            return
        await self.create_run({"query": query})

    @on(ListView.Selected, "#research-run-list")
    def _on_run_selected(self, event: ListView.Selected) -> None:
        run = getattr(event.item, "run_record", None)
        if run is not None:
            self.select_run(run)

    @on(Button.Pressed, "#research-pause-run")
    async def _on_pause_pressed(self, _event: Button.Pressed) -> None:
        await self.pause_selected_run()

    @on(Button.Pressed, "#research-resume-run")
    async def _on_resume_pressed(self, _event: Button.Pressed) -> None:
        await self.resume_selected_run()

    @on(Button.Pressed, "#research-cancel-run")
    async def _on_cancel_pressed(self, _event: Button.Pressed) -> None:
        await self.cancel_selected_run()
