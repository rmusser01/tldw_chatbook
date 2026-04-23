"""Research Sessions source-switched TUI window."""

from __future__ import annotations

import json
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static, TextArea

from tldw_chatbook.UI.Research_Modules import ResearchController


class ResearchWindow(Vertical):
    """Research Sessions container for local/server run browsing."""

    def __init__(self, app_instance: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_source = "local"
        self.runs: list[Any] = []
        self.selected_run: Any | None = None
        self.current_bundle: dict[str, Any] | None = None
        self.current_artifact: Any | None = None
        self.event_log_entries: list[str] = []
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
                    yield Button("Watch Events", id="research-watch-events")
                    yield Button("Cancel", id="research-cancel-run", variant="error")
                with Horizontal(id="research-observe-actions"):
                    yield Input(placeholder="Artifact name", id="research-artifact-name")
                    yield Button("Load Artifact", id="research-load-artifact")
                    yield Button("Load Bundle", id="research-load-bundle")
                yield Input(placeholder="Checkpoint id (defaults to latest)", id="research-checkpoint-id")
                yield TextArea("{}", id="research-checkpoint-patch")
                with Horizontal(id="research-checkpoint-actions"):
                    yield Button("Approve Checkpoint", id="research-approve-checkpoint")
                    yield Button("Clear Events", id="research-clear-events")
                yield Static("No bundle loaded.", id="research-bundle-detail")
                yield Static("No artifact loaded.", id="research-artifact-detail")
                yield Static("No research events captured yet.", id="research-event-log")

    def save_state(self) -> dict[str, Any]:
        return {"source": self.current_source}

    def restore_state(self, state: dict[str, Any]) -> None:
        source = str((state or {}).get("source") or "local").strip().lower()
        self.current_source = source if source in {"local", "server"} else "local"

    async def switch_source(self, source: str) -> list[Any]:
        self.current_source = self._normalize_source(source)
        self.runs = []
        self.selected_run = None
        self._reset_run_payload_state()
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
        self._set_selected_run(run, reset_payload_state=True)

    async def pause_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.pause_run(self.current_source, run_id)
        self._set_selected_run(updated, reset_payload_state=False)
        return updated

    async def resume_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.resume_run(self.current_source, run_id)
        self._set_selected_run(updated, reset_payload_state=False)
        return updated

    async def cancel_selected_run(self) -> Any:
        run_id = self._selected_run_id()
        updated = await self.controller.cancel_run(self.current_source, run_id)
        self._set_selected_run(updated, reset_payload_state=False)
        return updated

    async def load_selected_run_bundle(self) -> dict[str, Any]:
        run_id = self._selected_run_id()
        bundle = await self.controller.get_bundle(self.current_source, run_id)
        self.current_bundle = dict(bundle or {})
        self._render_bundle_detail()
        if self.current_bundle and self.is_mounted:
            first_artifact_name = next(iter(self.current_bundle.keys()), "")
            if first_artifact_name:
                try:
                    self.query_one("#research-artifact-name", Input).value = str(first_artifact_name)
                except Exception:
                    pass
        self._set_status(f"Loaded research bundle for {run_id}.")
        return self.current_bundle

    async def load_selected_run_artifact(self, artifact_name: str | None = None) -> Any:
        run_id = self._selected_run_id()
        resolved_artifact_name = self._resolve_artifact_name(artifact_name)
        if not resolved_artifact_name:
            self._set_status("Artifact name is required.")
            return None
        artifact = await self.controller.get_artifact(self.current_source, run_id, resolved_artifact_name)
        self.current_artifact = artifact
        self._render_artifact_detail()
        self._set_status(f"Loaded research artifact {resolved_artifact_name}.")
        return artifact

    async def approve_selected_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        patch_payload: dict[str, Any] | None = None,
    ) -> Any:
        try:
            resolved_checkpoint_id = self._resolve_checkpoint_id(checkpoint_id)
            if not resolved_checkpoint_id and self.current_source != "local":
                self._set_status("Checkpoint id is required.")
                return None
            if not resolved_checkpoint_id:
                resolved_checkpoint_id = "local-checkpoint-unavailable"
            resolved_patch_payload = (
                patch_payload if patch_payload is not None else self._parse_checkpoint_patch_payload()
            )
            updated = await self.controller.patch_and_approve_checkpoint(
                self.current_source,
                self._selected_run_id(),
                resolved_checkpoint_id,
                resolved_patch_payload,
            )
        except Exception as exc:
            self._set_status(str(exc))
            return None
        self._set_selected_run(updated, reset_payload_state=False)
        self._set_status(f"Approved research checkpoint {resolved_checkpoint_id}.")
        return updated

    async def watch_selected_run_events(self, *, after_id: int = 0) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        events: list[dict[str, Any]] = []
        try:
            async for event in self.controller.stream_run_events(
                self.current_source,
                run_id,
                after_id=after_id,
            ):
                event_data = dict(event or {})
                events.append(event_data)
                self._apply_stream_event(event_data)
        except Exception as exc:
            self._set_status(str(exc))
            return events
        if not events:
            self._set_status("Research event stream ended without events.")
        return events

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
            f"Control: {self._record_get(run, 'control_state', 'unknown')}\n"
            f"Latest checkpoint: {self._record_get(run, 'latest_checkpoint_id', 'none')}\n"
            f"Progress: {self._record_get(run, 'progress_message', '') or 'n/a'}"
        )
        if not self.is_mounted:
            return
        try:
            self.query_one("#research-run-detail", Static).update(detail)
        except Exception:
            pass

    def _apply_stream_event(self, event: dict[str, Any]) -> None:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("event") == "snapshot" and isinstance(data, dict) and isinstance(data.get("run"), dict):
            run_payload = dict(data["run"])
            run_payload.setdefault("query", self._record_get(self.selected_run, "query", ""))
            self._set_selected_run(run_payload, reset_payload_state=False)
        message = self._stream_event_message(event)
        self._set_status(message)
        self._append_event_log_entry(message)

    def _stream_event_message(self, event: dict[str, Any]) -> str:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        event_name = str(event.get("event") or "event")
        event_id = event.get("id")
        if isinstance(data, dict):
            progress_message = data.get("progress_message")
            if not progress_message and isinstance(data.get("run"), dict):
                progress_message = data["run"].get("progress_message")
            if progress_message:
                return f"Research event {event_name} {event_id or ''}: {progress_message}".strip()
        return f"Research event {event_name} {event_id or ''}".strip()

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

    def _set_selected_run(self, run: Any, *, reset_payload_state: bool) -> None:
        self.selected_run = run
        if reset_payload_state:
            self._reset_run_payload_state()
        self._update_detail(run)

    def _reset_run_payload_state(self) -> None:
        self.current_bundle = None
        self.current_artifact = None
        self.event_log_entries = []
        self._render_bundle_detail()
        self._render_artifact_detail()
        self._render_event_log()

    def _append_event_log_entry(self, message: str) -> None:
        self.event_log_entries.append(message)
        self._render_event_log()

    def _render_bundle_detail(self) -> None:
        if not self.is_mounted:
            return
        renderable = "No bundle loaded."
        if self.current_bundle is not None:
            renderable = json.dumps(self.current_bundle, indent=2, sort_keys=True, default=str)
        try:
            self.query_one("#research-bundle-detail", Static).update(renderable)
        except Exception:
            pass

    def _render_artifact_detail(self) -> None:
        if not self.is_mounted:
            return
        renderable = "No artifact loaded."
        if self.current_artifact is not None:
            artifact = self.current_artifact
            renderable = (
                f"Artifact: {self._record_get(artifact, 'artifact_name', '')}\n"
                f"Type: {self._record_get(artifact, 'content_type', '')}\n"
                f"Version: {self._record_get(artifact, 'artifact_version', 1)}\n"
                f"Content:\n{self._render_value(self._record_get(artifact, 'content'))}"
            )
        try:
            self.query_one("#research-artifact-detail", Static).update(renderable)
        except Exception:
            pass

    def _render_event_log(self) -> None:
        if not self.is_mounted:
            return
        renderable = "\n".join(self.event_log_entries) if self.event_log_entries else "No research events captured yet."
        try:
            self.query_one("#research-event-log", Static).update(renderable)
        except Exception:
            pass

    def _resolve_artifact_name(self, artifact_name: str | None) -> str:
        resolved = str(artifact_name or "").strip()
        if not resolved and self.is_mounted:
            try:
                resolved = self.query_one("#research-artifact-name", Input).value.strip()
            except Exception:
                resolved = ""
        if not resolved and self.current_bundle:
            resolved = str(next(iter(self.current_bundle.keys()), "")).strip()
        return resolved

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        resolved = str(checkpoint_id or "").strip()
        if not resolved and self.is_mounted:
            try:
                resolved = self.query_one("#research-checkpoint-id", Input).value.strip()
            except Exception:
                resolved = ""
        if not resolved:
            resolved = str(self._record_get(self.selected_run, "latest_checkpoint_id", "") or "").strip()
        return resolved

    def _parse_checkpoint_patch_payload(self) -> dict[str, Any] | None:
        raw_text = ""
        if self.is_mounted:
            try:
                raw_text = self.query_one("#research-checkpoint-patch", TextArea).text.strip()
            except Exception:
                raw_text = ""
        if not raw_text:
            return None
        payload = json.loads(raw_text)
        if payload in ({}, None):
            return None
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint patch payload must be a JSON object.")
        return payload

    @staticmethod
    def _render_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, sort_keys=True, default=str)

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

    @on(Button.Pressed, "#research-watch-events")
    async def _on_watch_events_pressed(self, _event: Button.Pressed) -> None:
        await self.watch_selected_run_events()

    @on(Button.Pressed, "#research-cancel-run")
    async def _on_cancel_pressed(self, _event: Button.Pressed) -> None:
        await self.cancel_selected_run()

    @on(Button.Pressed, "#research-load-bundle")
    async def _on_load_bundle_pressed(self, _event: Button.Pressed) -> None:
        await self.load_selected_run_bundle()

    @on(Button.Pressed, "#research-load-artifact")
    async def _on_load_artifact_pressed(self, _event: Button.Pressed) -> None:
        await self.load_selected_run_artifact()

    @on(Button.Pressed, "#research-approve-checkpoint")
    async def _on_approve_checkpoint_pressed(self, _event: Button.Pressed) -> None:
        await self.approve_selected_checkpoint()

    @on(Button.Pressed, "#research-clear-events")
    def _on_clear_events_pressed(self, _event: Button.Pressed) -> None:
        self.event_log_entries = []
        self._render_event_log()
