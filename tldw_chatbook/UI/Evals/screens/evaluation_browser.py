"""Read-oriented evaluation browser that consumes the shared evaluation scope seam."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from ....Evaluations_Interop import (
    EvaluationScopeService,
)
from ..navigation.nav_bar import EvalNavigationBar, EvalStatus, QuickAction

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvaluationBrowserScreen(Screen):
    """Browse and launch evaluations through the compat scope service."""

    BINDINGS = [
        Binding("ctrl+r", "refresh_browser", "Refresh", show=True),
        Binding("escape", "app.pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    EvaluationBrowserScreen {
        background: $background;
    }

    .browser-container {
        width: 100%;
        height: 100%;
        layout: vertical;
        padding: 1 2;
    }

    .context-strip {
        width: 100%;
        padding: 1 2;
        margin-bottom: 1;
        background: $panel;
        border: round $primary;
        color: $text-muted;
    }

    .browser-body {
        width: 100%;
        height: 1fr;
    }

    .browser-pane {
        width: 1fr;
        height: 100%;
        border: round $primary;
        background: $panel;
        padding: 1;
    }

    .browser-pane.left {
        margin-right: 1;
    }

    .browser-pane.right {
        margin-left: 1;
    }

    .pane-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .pane-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    .browser-table {
        height: 1fr;
    }

    .launcher-panel {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: round $primary-background;
        background: $surface;
    }

    .launcher-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .launcher-input {
        width: 1fr;
        margin-right: 1;
    }

    .launcher-input:last-child {
        margin-right: 0;
    }

    #runs-table {
        height: 12;
        margin-bottom: 1;
    }

    #evaluation-detail {
        height: 1fr;
        border: solid $primary-background;
    }
    """

    def __init__(self, app_instance: "TldwCli", *, view_mode: str = "manage", **kwargs: Any):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.view_mode = view_mode
        self.nav_bar: Optional[EvalNavigationBar] = None
        self.evaluations: List[Dict[str, Any]] = []
        self.datasets_by_id: Dict[str, Dict[str, Any]] = {}
        self.runs: List[Dict[str, Any]] = []
        self.targets: List[Dict[str, Any]] = []
        self._evaluation_row_map: Dict[str, Dict[str, Any]] = {}
        self._run_row_map: Dict[str, Dict[str, Any]] = {}
        self._selected_evaluation_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        self.nav_bar = EvalNavigationBar(self.app_instance)
        yield self.nav_bar

        with Container(classes="browser-container"):
            yield Static("", id="browser-context", classes="context-strip")
            with Horizontal(classes="browser-body"):
                with Vertical(classes="browser-pane left"):
                    yield Static("Evaluations", classes="pane-title")
                    yield Static(
                        "Read-oriented parity surface for local/server evaluation definitions.",
                        classes="pane-subtitle",
                    )
                    table = DataTable(id="evaluations-table", classes="browser-table")
                    table.cursor_type = "row"
                    table.zebra_stripes = True
                    yield table
                with Vertical(classes="browser-pane right"):
                    yield Static("Runs", classes="pane-title")
                    if self.view_mode == "manage":
                        with Container(classes="launcher-panel"):
                            yield Static("Run Launcher", classes="pane-title")
                            yield Static(
                                "Local runs use saved targets. Server runs accept an explicit target model string.",
                                classes="pane-subtitle",
                            )
                            with Horizontal(classes="launcher-row"):
                                yield Select(
                                    [],
                                    prompt="Select local target...",
                                    allow_blank=True,
                                    id="target-select",
                                    classes="launcher-input",
                                )
                                yield Input(
                                    placeholder="openai:gpt-4.1-mini",
                                    id="target-model-input",
                                    classes="launcher-input",
                                )
                            with Horizontal(classes="launcher-row"):
                                yield Input(
                                    placeholder="Run name (optional)",
                                    id="run-name-input",
                                    classes="launcher-input",
                                )
                                yield Input(
                                    placeholder='{"temperature": 0.2}',
                                    id="run-config-input",
                                    classes="launcher-input",
                                )
                                yield Button("Create Run", id="create-run-button", variant="primary")
                    runs_table = DataTable(id="runs-table", classes="browser-table")
                    runs_table.cursor_type = "row"
                    runs_table.zebra_stripes = True
                    yield runs_table
                    yield Static("Details", classes="pane-title")
                    yield TextArea("", id="evaluation-detail", read_only=True)

    def on_mount(self) -> None:
        self.call_after_refresh(self._configure_nav_bar)

        eval_table = self.query_one("#evaluations-table", DataTable)
        eval_table.add_columns("Name", "Type", "Dataset", "Updated")
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.add_columns("Run", "Status", "Model", "Progress", "Created")
        self._sync_launcher_visibility(self._runtime_backend())

        self.run_worker(self._refresh_browser(), exclusive=True)

    def _configure_nav_bar(self) -> None:
        if self.nav_bar:
            self.nav_bar.push_breadcrumb("Evaluations", "tasks")
            self.nav_bar.set_status(EvalStatus.IDLE)
            self.nav_bar.can_run = False
            self.nav_bar.can_stop = False
            self.nav_bar.can_export = False

    def _runtime_backend(self) -> str:
        candidates = (
            getattr(self.app_instance, "current_runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def _scope_service(self) -> Optional[EvaluationScopeService]:
        service = getattr(self.app_instance, "evaluation_scope_service", None)
        if service is not None:
            return service
        return None

    def _notify(self, message: str, severity: str = "warning") -> None:
        notifier = getattr(self.app_instance, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    def _sync_launcher_visibility(self, mode: str) -> None:
        if self.view_mode != "manage":
            return

        is_local = mode == "local"
        target_select = self.query_one("#target-select", Select)
        target_input = self.query_one("#target-model-input", Input)
        target_select.display = is_local
        target_input.display = not is_local

    def _set_target_options(self, targets: list[dict[str, Any]]) -> None:
        if self.view_mode != "manage":
            return

        self.targets = list(targets or [])
        target_select = self.query_one("#target-select", Select)
        options = [
            (
                str(target.get("display_name") or target.get("name") or target.get("target_model") or "Unnamed target"),
                str(target.get("backing_id") or target.get("record_id") or ""),
            )
            for target in self.targets
            if target.get("backing_id") or target.get("record_id")
        ]
        if not options:
            options = [("No local targets available", Select.BLANK)]
        target_select.set_options(options)
        target_select.value = options[0][1]

    def action_refresh_browser(self) -> None:
        self.run_worker(self._refresh_browser(), exclusive=True)

    @on(QuickAction)
    def handle_quick_action(self, message: QuickAction) -> None:
        if message.action == "refresh":
            self.action_refresh_browser()
            return
        self._notify(
            "Use Quick Test for ad-hoc prompts. This screen is for evaluation definitions, runs, and results.",
            severity="information",
        )

    @on(DataTable.RowSelected, "#evaluations-table")
    def handle_evaluation_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        row_key = str(event.row_key.value)
        record = self._evaluation_row_map.get(row_key)
        if not record:
            return
        self._selected_evaluation_id = str(record.get("backing_id") or "")
        self._render_evaluation_detail(record)
        if self._selected_evaluation_id:
            self.run_worker(self._refresh_runs(self._selected_evaluation_id), exclusive=True)

    @on(DataTable.RowSelected, "#runs-table")
    def handle_run_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        row_key = str(event.row_key.value)
        record = self._run_row_map.get(row_key)
        if not record:
            return
        run_id = str(record.get("backing_id") or "")
        if run_id:
            self.run_worker(self._load_run_artifacts(run_id), exclusive=True)

    @on(Button.Pressed, "#create-run-button")
    def handle_create_run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self._create_run(), exclusive=True)

    async def _refresh_browser(self) -> None:
        mode = self._runtime_backend()
        scope = self._scope_service()
        context = self.query_one("#browser-context", Static)
        if self.nav_bar:
            self.nav_bar.set_status(EvalStatus.RUNNING)

        if scope is None:
            context.update("Evaluation browser is unavailable because the app-owned evaluation scope service is unavailable.")
            self._set_empty_tables()
            if self.nav_bar:
                self.nav_bar.set_status(EvalStatus.ERROR)
            return

        try:
            evaluations = await scope.list_evaluations(mode=mode, limit=200)
            datasets = await scope.list_datasets(mode=mode, limit=200)
            targets: list[dict[str, Any]] = []
            if self.view_mode == "manage" and mode == "local":
                targets = await scope.list_targets(mode=mode, limit=200)
        except Exception as exc:
            logger.error("Failed to load evaluation browser data", exc_info=True)
            context.update(f"{mode.title()} backend load failed: {exc}")
            self._set_empty_tables()
            if self.nav_bar:
                self.nav_bar.set_status(EvalStatus.ERROR)
            self._notify(f"Failed to load evaluations: {exc}", severity="error")
            return

        self.evaluations = list(evaluations or [])
        self.datasets_by_id = {
            str(record.get("backing_id") or record.get("record_id")): dict(record)
            for record in list(datasets or [])
        }
        self._sync_launcher_visibility(mode)
        if self.view_mode == "manage":
            self._set_target_options(targets if mode == "local" else [])
        self._populate_evaluations_table()

        context.update(
            f"Backend: {mode} | Evaluations: {len(self.evaluations)} | Datasets: {len(self.datasets_by_id)} | "
            "Local results include sample detail; server results are currently summary-only."
        )

        if self.evaluations:
            first_record = self.evaluations[0]
            self._selected_evaluation_id = str(first_record.get("backing_id") or "")
            self._render_evaluation_detail(first_record)
            if self._selected_evaluation_id:
                await self._refresh_runs(self._selected_evaluation_id)
        else:
            self._render_empty_detail(mode)
            self._clear_runs_table()

        if self.nav_bar:
            self.nav_bar.set_status(EvalStatus.SUCCESS)

    async def _refresh_runs(self, evaluation_id: str) -> None:
        mode = self._runtime_backend()
        scope = self._scope_service()
        if scope is None:
            self._clear_runs_table()
            return

        try:
            runs = await scope.list_runs(mode=mode, eval_id=evaluation_id, limit=200)
        except Exception as exc:
            logger.error("Failed to load evaluation runs for {}", evaluation_id, exc_info=True)
            self._clear_runs_table(error_message=str(exc))
            self._notify(f"Failed to load evaluation runs: {exc}", severity="error")
            return

        self.runs = list(runs or [])
        self._populate_runs_table()

    async def _create_run(self) -> None:
        if self.view_mode != "manage":
            return

        mode = self._runtime_backend()
        scope = self._scope_service()
        if scope is None:
            self._notify("Evaluation backend is unavailable.", severity="error")
            return
        if not self._selected_evaluation_id:
            self._notify("Select an evaluation before creating a run.", severity="warning")
            return

        run_name = self.query_one("#run-name-input", Input).value.strip() or None
        config_text = self.query_one("#run-config-input", Input).value.strip()
        target_id: str | None = None
        target_model: str | None = None

        if mode == "local":
            target_select = self.query_one("#target-select", Select)
            if target_select.value == Select.BLANK:
                self._notify("Select a local target before creating a run.", severity="warning")
                return
            target_id = str(target_select.value)
        else:
            target_model = self.query_one("#target-model-input", Input).value.strip() or None
            if not target_model:
                self._notify("Enter a target model before creating a server run.", severity="warning")
                return

        config: dict[str, Any] = {}
        if config_text:
            try:
                parsed = json.loads(config_text)
            except (TypeError, ValueError) as exc:
                self._notify(f"Run config must be valid JSON: {exc}", severity="error")
                return
            if not isinstance(parsed, dict):
                self._notify("Run config must be a JSON object.", severity="error")
                return
            config = dict(parsed)

        try:
            run = await scope.create_run(
                mode=mode,
                eval_id=self._selected_evaluation_id,
                target_id=target_id,
                target_model=target_model,
                config=config,
                run_name=run_name,
            )
        except Exception as exc:
            logger.error("Failed to create evaluation run", exc_info=True)
            self._notify(f"Failed to create run: {exc}", severity="error")
            return

        self._notify(f"Created run {run.get('backing_id') or run.get('name')}.", severity="information")
        await self._refresh_runs(self._selected_evaluation_id)
        self._render_run_detail(self._evaluation_row_map.get(f"{mode}:evaluation:{self._selected_evaluation_id}"), {
            "run": run,
            "metrics": run.get("results") or {},
            "results": None,
            "detail_available": mode == "local",
        })

    def _set_empty_tables(self) -> None:
        self.evaluations = []
        self.datasets_by_id = {}
        self.runs = []
        self.targets = []
        self._populate_evaluations_table()
        self._clear_runs_table()
        if self.view_mode == "manage":
            self._set_target_options([])
        self.query_one("#evaluation-detail", TextArea).text = "No evaluation data available."

    def _populate_evaluations_table(self) -> None:
        table = self.query_one("#evaluations-table", DataTable)
        table.clear()
        self._evaluation_row_map.clear()

        for record in self.evaluations:
            row_key = str(record.get("record_id") or "")
            self._evaluation_row_map[row_key] = record
            dataset_name = self._dataset_name(record.get("dataset_id"))
            table.add_row(
                record.get("name") or "(unnamed)",
                record.get("eval_type") or "-",
                dataset_name,
                str(record.get("updated_at") or record.get("created_at") or "-"),
                key=row_key,
            )

    def _clear_runs_table(self, error_message: Optional[str] = None) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        self.runs = []
        self._run_row_map.clear()
        if error_message:
            table.add_row("Unable to load runs", "error", "-", "-", error_message[:32], key="runs-error")

    def _populate_runs_table(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        self._run_row_map.clear()

        for record in self.runs:
            row_key = str(record.get("record_id") or "")
            self._run_row_map[row_key] = record
            table.add_row(
                record.get("name") or record.get("backing_id") or "(unnamed)",
                record.get("status") or "-",
                record.get("target_model") or "-",
                self._progress_label(record.get("progress")),
                str(record.get("created_at") or "-"),
                key=row_key,
            )

    def _render_empty_detail(self, mode: str) -> None:
        detail = self.query_one("#evaluation-detail", TextArea)
        detail.text = (
            f"No {mode} evaluations are available.\n\n"
            "This browser only exposes evaluation definitions, linked datasets, and recent runs."
        )

    def _dataset_name(self, dataset_id: Any) -> str:
        if dataset_id in (None, ""):
            return "-"
        dataset = self.datasets_by_id.get(str(dataset_id))
        if dataset is None:
            return str(dataset_id)
        return str(dataset.get("name") or dataset_id)

    def _progress_label(self, progress: Any) -> str:
        if not isinstance(progress, dict):
            return "-"
        completed = progress.get("completed_samples")
        total = progress.get("total_samples")
        percent = progress.get("percent_complete")
        if completed is None and total is None and percent is None:
            return "-"
        return f"{percent or 0:.0f}% ({completed or 0}/{total or 0})"

    def _render_evaluation_detail(self, record: Dict[str, Any]) -> None:
        dataset = self.datasets_by_id.get(str(record.get("dataset_id") or ""))
        detail = self.query_one("#evaluation-detail", TextArea)
        detail_payload = {
            "backend": record.get("backend"),
            "id": record.get("backing_id"),
            "name": record.get("name"),
            "description": record.get("description"),
            "eval_type": record.get("eval_type"),
            "dataset": {
                "id": record.get("dataset_id"),
                "name": dataset.get("name") if dataset else None,
                "sample_count": dataset.get("sample_count") if dataset else None,
                "source_path": dataset.get("source_path") if dataset else None,
                "format": dataset.get("format") if dataset else None,
            } if record.get("dataset_id") else None,
            "metadata": record.get("metadata") or {},
            "eval_spec": record.get("eval_spec") or {},
        }
        detail.text = json.dumps(detail_payload, indent=2, sort_keys=True)

    async def _load_run_artifacts(self, run_id: str) -> None:
        mode = self._runtime_backend()
        scope = self._scope_service()
        if scope is None:
            return

        try:
            artifacts = await scope.get_run_artifacts(mode=mode, run_id=run_id)
        except Exception as exc:
            logger.error("Failed to load run artifacts for {}", run_id, exc_info=True)
            self._notify(f"Failed to load run detail: {exc}", severity="error")
            return

        evaluation_record = None
        if self._selected_evaluation_id:
            evaluation_record = self._evaluation_row_map.get(f"{mode}:evaluation:{self._selected_evaluation_id}")
        self._render_run_detail(evaluation_record, artifacts)

    def _render_run_detail(self, evaluation_record: Optional[Dict[str, Any]], artifacts: Dict[str, Any]) -> None:
        run = dict(artifacts.get("run") or {})
        dataset = None
        if evaluation_record and evaluation_record.get("dataset_id"):
            dataset = self.datasets_by_id.get(str(evaluation_record.get("dataset_id")))

        detail_payload = {
            "evaluation": {
                "id": evaluation_record.get("backing_id") if evaluation_record else run.get("evaluation_id"),
                "name": evaluation_record.get("name") if evaluation_record else None,
                "dataset": {
                    "id": evaluation_record.get("dataset_id") if evaluation_record else None,
                    "name": dataset.get("name") if dataset else None,
                } if evaluation_record else None,
            },
            "run": {
                "id": run.get("backing_id"),
                "name": run.get("name"),
                "status": run.get("status"),
                "target_model": run.get("target_model"),
                "progress": run.get("progress"),
                "config": run.get("config"),
                "results_summary": run.get("results"),
                "error_message": run.get("error_message"),
            },
            "metrics": artifacts.get("metrics") or {},
            "results": artifacts.get("results") if artifacts.get("detail_available") else None,
            "detail_available": bool(artifacts.get("detail_available")),
            "detail_note": None if artifacts.get("detail_available") else "Server-backed runs currently expose summary-level results only.",
        }
        self.query_one("#evaluation-detail", TextArea).text = json.dumps(
            detail_payload,
            indent=2,
            sort_keys=True,
        )
