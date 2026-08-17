"""Research Sessions source-switched TUI window."""

from __future__ import annotations

import json
from typing import Any

from textual import on

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.UI.Research_Modules import ResearchController
from tldw_chatbook.UI.Research_Modules.bundle_rendering import (
    default_artifact_for_bundle,
    render_artifact,
    render_bundle_summary,
)


def _parse_provider_tokens(text: str | None) -> list[str]:
    """Parse and validate the providers input (task-16791/Qodo +
    task-16792): the raw text goes through the shared input validator, then
    tokens -- source ids ("pubmed") OR category names ("biomedical",
    "repositories", ...) -- are validated against the catalog and
    deduplicated in order. Unknown tokens drop to a warning rather than
    silently narrowing; invalid input drops the whole list."""
    from tldw_chatbook.Utils.input_validation import validate_text_input

    from ..Research_Interop.research_source_catalog import (
        CATEGORIES,
        catalog_entries,
    )

    raw = str(text or "")
    if not validate_text_input(raw, max_length=200):
        return []
    known = {e.source_id for e in catalog_entries()} | set(CATEGORIES)
    seen: list[str] = []
    unknown: list[str] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in known:
            if token not in seen:
                seen.append(token)
        else:
            unknown.append(token)
    if unknown:
        logger.warning(f"providers input ignored unknown tokens: {unknown}")
    return seen


def _parse_limits_text(text: str | None) -> tuple[dict[str, float], list[str]]:
    """Parse a limits input like "max_searches=5, max_runtime_seconds=120"
    into a limits_json dict (task-16334). Invalid pairs are excluded and
    reported as warnings so one typo never blocks run creation."""
    limits: dict[str, float] = {}
    warnings: list[str] = []
    for raw_pair in str(text or "").split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            warnings.append(f"unparsed limit (expected key=value): {pair!r}")
            continue
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        try:
            value = float(raw_value.strip())
        except ValueError:
            warnings.append(f"non-numeric limit for {key!r}: {raw_value.strip()!r}")
            continue
        if key:
            limits[key] = value
    return limits, warnings


def _parse_config_bool(value: Any) -> bool:
    """Parse a config bool that may arrive as an actual bool or a string
    (task-16814: ``bool("false")`` is True -- truthy strings must not
    enable the network-costing lane)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "on"}


def _rounds_options(current: int) -> list[tuple[str, int]]:
    """Rounds options that always include ``current`` (Qodo, PR 1769).

    The control offered a fixed 1-4 with ``allow_blank=False``, so an install
    configuring more rounds -- or a restored state holding more -- raised
    InvalidSelectValueError and the window failed to mount entirely. Extending
    the options rather than clamping the value keeps the control honest: it
    shows what the run will actually do instead of quietly displaying a
    different number.

    Args:
        current: The value that must be selectable.

    Returns:
        Ascending (label, value) pairs covering 1-4 plus ``current``.
    """
    values = sorted({1, 2, 3, 4, max(1, int(current or 1))})
    return [(f"{n} round" if n == 1 else f"{n} rounds", n) for n in values]


def _iteration_rounds_default() -> int:
    """Rounds the window offers by default (task-17371).

    Deliberately delegates to the engine's own resolver so the number a user
    sees is the number a run would have used anyway -- a second default here
    would drift from it. Lazy import keeps the window's module import cheap.
    """
    try:
        from ..Research_Interop.local_research_engine import (
            _configured_max_iterations,
        )

        return max(1, int(_configured_max_iterations()))
    except Exception:  # noqa: BLE001 - a UI default must never fail to load
        return 1


def _academic_lane_default() -> bool:
    """Config default for the academic lane toggle: [SearchSettings]
    research_academic_lane (default False). Failures default OFF -- the
    lane costs network calls, so the safe default is opt-in."""
    try:
        from tldw_chatbook.config import get_cli_setting

        return _parse_config_bool(
            get_cli_setting("SearchSettings", "research_academic_lane", False)
        )
    except Exception:
        return False


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
        # task-16328: academic lane toggle (arXiv + Semantic Scholar papers
        # join the run's evidence pool when enabled).
        self.academic_enabled = _academic_lane_default()
        # task-17371: rounds this window will launch with (multi-hop). Shown
        # rather than buried in the limits box, and persisted like the lane
        # toggle; a typed max_iterations still wins on create.
        self.iteration_rounds = _iteration_rounds_default()
        # task-16334: budget limits text (parsed into limits_json on create)
        # and the rendered follow-up answer.
        self.limits_text = ""
        self.followup_answer_text = ""
        # task-16791: per-run lane routing + academic provider selection.
        self.source_policy = "balanced"
        self.providers_text = ""
        self.controller = ResearchController(
            getattr(app_instance, "research_scope_service", None)
        )

    def compose(self) -> ComposeResult:
        """Build the window: toolbar, run-creation row, run list and detail pane.

        Yields:
            The window's child widgets, in mount order.
        """
        yield Label("Research Sessions")
        with Horizontal(id="research-toolbar"):
            yield Select(
                [("Local", "local"), ("Server", "server")],
                value=self.current_source,
                allow_blank=False,
                id="research-source-select",
            )
            yield Button("Refresh", id="research-refresh-runs")
            yield Checkbox(
                "Academic (arXiv + S2)",
                value=self.academic_enabled,
                id="research-academic-toggle",
            )
        with Horizontal(id="research-create-row"):
            yield Input(placeholder="Research question", id="research-query-input")
            yield Input(
                placeholder="Limits: max_searches=5, max_runtime_seconds=120",
                id="research-limits-input",
            )
            yield Select(
                [("Balanced", "balanced"), ("Web only", "web_only"),
                 ("Academic only", "academic_only"), ("Web first", "web_first"),
                 ("Academic first", "academic_first")],
                value=self.source_policy,
                allow_blank=False,
                id="research-policy-select",
            )
            yield Select(
                _rounds_options(self.iteration_rounds),
                value=self.iteration_rounds,
                allow_blank=False,
                id="research-rounds-select",
            )
            yield Input(
                placeholder="Providers: arxiv, pubmed",
                id="research-providers-input",
            )
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
                    yield Input(
                        placeholder="Artifact name", id="research-artifact-name"
                    )
                    yield Button("Load Artifact", id="research-load-artifact")
                    yield Button("Load Bundle", id="research-load-bundle")
                yield Input(
                    placeholder="Checkpoint id (defaults to latest)",
                    id="research-checkpoint-id",
                )
                yield TextArea("{}", id="research-checkpoint-patch")
                with Horizontal(id="research-checkpoint-actions"):
                    yield Button("Approve Checkpoint", id="research-approve-checkpoint")
                    yield Button("Clear Events", id="research-clear-events")
                with Horizontal(id="research-followup-row"):
                    yield Input(
                        placeholder="Ask a follow-up about the selected run",
                        id="research-followup-input",
                    )
                    yield Button("Ask Follow-up", id="research-followup-ask")
                yield Static("No follow-up yet.", id="research-followup-answer")
                yield Static("No bundle loaded.", id="research-bundle-detail")
                yield Static("No artifact loaded.", id="research-artifact-detail")
                yield Static(
                    "No research events captured yet.", id="research-event-log"
                )

    def save_state(self) -> dict[str, Any]:
        """Capture the operator's run-creation choices for later restoration.

        Returns:
            The source, academic-lane flag, limits text, source policy,
            provider tokens and multi-hop round count.
        """
        return {
            "source": self.current_source,
            "academic": self.academic_enabled,
            "limits": self.limits_text,
            "policy": self.source_policy,
            "providers": self.providers_text,
            "rounds": self.iteration_rounds,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Reapply saved choices, falling back to defaults for bad values.

        Every field is validated rather than trusted: an unknown source or
        policy, or a non-positive round count, resolves to its default instead
        of reaching a widget that would reject it.

        Args:
            state: A mapping previously produced by ``save_state`` (or empty).
        """
        source = str((state or {}).get("source") or "local").strip().lower()
        self.current_source = source if source in {"local", "server"} else "local"
        self.academic_enabled = bool((state or {}).get("academic"))
        self.limits_text = str((state or {}).get("limits") or "")
        policy = str((state or {}).get("policy") or "balanced")
        self.source_policy = (
            policy if policy in {
                "balanced", "web_only", "academic_only", "web_first", "academic_first",
            } else "balanced"
        )
        self.providers_text = str((state or {}).get("providers") or "")
        try:
            rounds = int((state or {}).get("rounds") or 0)
        except (TypeError, ValueError):
            rounds = 0
        self.iteration_rounds = rounds if rounds >= 1 else _iteration_rounds_default()
        self._sync_academic_toggle()
        try:
            rounds_select = self.query_one("#research-rounds-select", Select)
            # Widen first: assigning a value the mounted control does not offer
            # raises InvalidSelectValueError (Qodo, PR 1769).
            rounds_select.set_options(_rounds_options(self.iteration_rounds))
            rounds_select.value = self.iteration_rounds
        except Exception:
            pass  # not mounted yet; compose()'s initial value covers it
        try:
            self.query_one("#research-limits-input", Input).value = self.limits_text
        except Exception:
            pass  # not mounted yet; compose()'s initial value covers it

    def _sync_academic_toggle(self) -> None:
        try:
            self.query_one("#research-academic-toggle", Checkbox).value = (
                self.academic_enabled
            )
        except Exception:
            pass  # not mounted yet; the compose() initial value covers it

    @on(Button.Pressed, "#research-followup-ask")
    async def _on_followup_ask(self, event: Button.Pressed) -> None:
        try:
            question = self.query_one("#research-followup-input", Input).value
        except Exception:
            question = ""
        await self.ask_follow_up(question or None)

    async def ask_follow_up(self, question: str | None) -> Any:
        """Answer a follow-up from the selected local run's stored claims
        (task-16334). Renders the answer, or the engine's explicit
        insufficient-evidence verdict -- never a fabricated one."""
        try:
            run_id = self._selected_run_id()
        except ValueError:
            run_id = ""
        if not run_id:
            self._set_status("Select a research run before asking a follow-up.")
            return None
        if self.current_source != "local":
            self._set_status("Follow-up Q&A is available for local runs only.")
            return None
        local_service = getattr(self.app_instance, "local_research_service", None)
        if local_service is None:
            self._set_status("Local research service unavailable for follow-ups.")
            return None
        if not question or not str(question).strip():
            self._set_status("Type a follow-up question first.")
            return None
        question = str(question).strip()

        # The default follow-up answerer reads the synthesis LLM from
        # search_params; assemble them the same way the tool does so the
        # window uses the configured pipeline.
        search_params: dict[str, Any] = {}
        try:
            from ..Tools.web_tool_impls import _deep_search_settings

            final_llm = _deep_search_settings().get("final_answer_llm")
            if final_llm:
                search_params["final_answer_llm"] = final_llm
        except Exception:
            pass

        from ..Research_Interop.local_research_engine import LocalResearchEngine

        engine = LocalResearchEngine(local_service, search_params=search_params)
        try:
            result = await engine.answer_follow_up(run_id, question)
        except Exception as exc:  # noqa: BLE001 - report, never crash the window
            self._set_status(f"Follow-up failed: {exc}")
            return None
        if result.get("status") == "answered":
            self.followup_answer_text = (
                f"Q: {question}\n{result.get('answer') or ''}"
            )
        else:
            self.followup_answer_text = (
                f"Q: {question}\n[insufficient evidence] "
                f"{result.get('reason') or 'stored evidence does not support this question'}\n"
                f"{result.get('suggestion') or ''}"
            )
        try:
            self.query_one("#research-followup-answer", Static).update(
                self.followup_answer_text
            )
        except Exception:
            pass  # not mounted (tests); the state carries the text
        self._set_status(f"Follow-up {result.get('status')}.")
        return result

    @on(Input.Changed, "#research-limits-input")
    def _on_limits_input_changed(self, event: Input.Changed) -> None:
        self.limits_text = event.value

    @on(Input.Changed, "#research-providers-input")
    def _on_providers_input_changed(self, event: Input.Changed) -> None:
        self.providers_text = event.value

    @on(Select.Changed, "#research-policy-select")
    def _on_policy_changed(self, event: Select.Changed) -> None:
        self.source_policy = str(event.value or "balanced")
        self._set_status(f"Source policy: {self.source_policy}")

    @on(Select.Changed, "#research-rounds-select")
    def _on_rounds_changed(self, event: Select.Changed) -> None:
        try:
            self.iteration_rounds = max(1, int(event.value))
        except (TypeError, ValueError):
            return
        if self.iteration_rounds == 1:
            self._set_status("Rounds: 1 (single pass -- no gap-driven follow-up).")
        else:
            self._set_status(
                f"Rounds: {self.iteration_rounds}. Each extra round researches the "
                "gaps the previous answer left open -- more evidence, and "
                "proportionally more searches and LLM calls."
            )

    @on(Checkbox.Changed, "#research-academic-toggle")
    def _on_academic_toggle_changed(self, event: Checkbox.Changed) -> None:
        self.academic_enabled = bool(event.value)
        self._set_status(
            "Academic sources (arXiv + Semantic Scholar) "
            + ("enabled" if self.academic_enabled else "disabled")
            + " for local runs."
        )

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
        """Create a run from the payload plus this window's live inputs.

        Reads the limits and providers inputs directly (rather than trusting
        the seeded attributes), folds in the source policy, provider overrides
        and the multi-hop round count, then starts the local engine for a
        freshly created local run.

        Args:
            payload: Base run fields from the caller, e.g. the query.

        Returns:
            The created run record, or whatever the controller returned.
        """
        # Qodo (PR 1722): read the inputs live -- restore_state seeds the
        # attributes, but typing in the widgets must reach the payload.
        try:
            self.limits_text = self.query_one("#research-limits-input", Input).value
        except Exception:
            pass
        try:
            self.providers_text = self.query_one(
                "#research-providers-input", Input
            ).value
        except Exception:
            pass
        limits, warnings = _parse_limits_text(self.limits_text)
        if warnings:
            self._set_status("Limits: " + "; ".join(warnings))
        # task-17371: the rounds control contributes max_iterations unless the
        # limits box already states one -- a typed value is the more specific
        # statement of intent, and is what the engine treats as authoritative.
        # Qodo (PR 1769): _parse_limits_text preserves key casing, so a typed
        # "Max_Iterations=1" used to be invisible here AND invisible to the
        # engine (which reads the canonical lowercase key), leaving the control's
        # value in charge of a run the user had explicitly bounded. Any casing
        # now counts, and is normalized onto the canonical key; a later variant
        # wins, as a repeated key would.
        typed_variants = [key for key in limits if key.lower() == "max_iterations"]
        if typed_variants:
            typed_value = limits[typed_variants[-1]]
            limits = {
                key: value
                for key, value in limits.items()
                if key not in typed_variants
            }
            limits["max_iterations"] = typed_value
        else:
            limits = {**limits, "max_iterations": max(1, int(self.iteration_rounds or 1))}
        if limits:
            payload = {**payload, "limits_json": limits}
        # task-16791: lane routing + provider selection ride the run record.
        payload = {**payload, "source_policy": self.source_policy}
        providers = _parse_provider_tokens(self.providers_text)
        if self.providers_text.strip() and not providers:
            self._set_status("Providers input invalid or all-unknown; ignored.")
        if providers:
            payload = {**payload, "provider_overrides": {
                "academic_providers": providers,
            }}
        created = await self.controller.create_run(self.current_source, payload)
        if self.current_source == "local":
            created_id = (
                created.get("id") if isinstance(created, dict) else getattr(created, "id", None)
            )
            if created_id:
                self._start_local_engine(str(created_id))
        await self.load_runs(self.current_source)
        return created

    def _start_local_engine(self, run_id: str) -> None:
        """Start the local research execution engine for ``run_id``
        (task-16322, ADR-068) in a Textual worker when mounted.

        A resumed local run re-enters the engine, which restarts the phase
        machine from the top (phase-level resume is out of scope for v1).
        Without a mounted app (headless/tests) or without the app's
        ``local_research_service``, this reports and does nothing -- it must
        never block the create/resume flow it hangs off of.
        """
        local_service = getattr(self.app_instance, "local_research_service", None)
        if local_service is None:
            self._set_status("Local research engine unavailable (no local service).")
            return
        # Lazy import keeps the window's module import cheap and lets tests
        # patch LocalResearchEngine on its own module.
        from tldw_chatbook.Research_Interop.local_research_engine import (
            LocalResearchEngine,
        )

        # task-16328: the academic lane joins the evidence pool (with DOI
        # dedup) when the window toggle is on; off keeps today's web-only
        # behavior with a None paper_search_fn.
        from ..Research_Interop.academic_providers import search_papers

        # task-17371: the window used to construct the engine with NO
        # search_params, so the pipeline rejected every window-launched run
        # ("Invalid search_params parameter") before a single search, and the
        # engine's gap analysis -- which reads final_answer_llm from these
        # params -- could never fire. Assemble them the way the Console
        # /research command and the baseline recorder do (one shared
        # assembly, task-16484); a failure to assemble is reported instead of
        # being spent on a run that cannot succeed.
        try:
            from ..Tools.web_tool_impls import deep_search_pipeline_params

            search_params = deep_search_pipeline_params()
        except Exception as exc:  # noqa: BLE001 - report, never crash the window
            logger.error(f"Research engine params unavailable: {exc}")
            self._set_status(
                "Research engine unavailable: deep-search settings could not be "
                f"assembled ({exc}). Check [SearchSettings] in your config."
            )
            return

        engine = LocalResearchEngine(
            local_service,
            search_params=search_params,
            paper_search_fn=search_papers if self.academic_enabled else None,
        )

        async def _run_engine() -> None:
            try:
                await engine.execute_run(run_id)
            except Exception as exc:  # noqa: BLE001 - worker must not crash the app
                # task-16814: dispatch through the UI message pump rather
                # than mutating widgets from worker context (async workers
                # run on the loop, but deferring is correct for either).
                try:
                    self.call_later(self._set_status, f"Local research engine error: {exc}")
                except Exception:
                    self._set_status(f"Local research engine error: {exc}")

        if self.is_mounted:
            self.run_worker(
                _run_engine(),
                group=f"research-engine-{run_id}",
                exclusive=True,
                description=f"Local research engine: {run_id}",
            )
            self._set_status(f"Local research engine started for {run_id}.")

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
        if self.current_source == "local":
            status = self._record_field(updated, "status")
            if status not in {"completed", "failed", "cancelled"}:
                self._start_local_engine(run_id)
        return updated

    @staticmethod
    def _record_field(record: Any, field: str, default: str = "") -> str:
        if isinstance(record, dict):
            value = record.get(field)
        else:
            value = getattr(record, field, None)
        return str(value) if value is not None else default

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
        # task-16483: open the most useful artifact right away (the report
        # when present -- never the local shape's run record).
        default_name = default_artifact_for_bundle(self.current_bundle)
        if default_name:
            if self.is_mounted:
                try:
                    self.query_one("#research-artifact-name", Input).value = default_name
                except Exception:
                    pass
            await self.load_selected_run_artifact(default_name)
        self._set_status(f"Loaded research bundle for {run_id}.")
        return self.current_bundle

    async def load_selected_run_artifact(self, artifact_name: str | None = None) -> Any:
        run_id = self._selected_run_id()
        resolved_artifact_name = self._resolve_artifact_name(artifact_name)
        if not resolved_artifact_name:
            self._set_status("Artifact name is required.")
            return None
        artifact = await self.controller.get_artifact(
            self.current_source, run_id, resolved_artifact_name
        )
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
            run_id = self._selected_run_id()
            if not resolved_checkpoint_id:
                # task-16482: local runs resolve their latest PENDING
                # checkpoint (the engine parks at one boundary at a time).
                local_service = getattr(
                    self.app_instance, "local_research_service", None
                )
                pending = (
                    local_service.latest_pending_checkpoint(run_id)
                    if local_service is not None
                    else None
                )
                if pending is None:
                    self._set_status("No pending checkpoint for this run.")
                    return None
                resolved_checkpoint_id = str(pending["id"])
            resolved_patch_payload = (
                patch_payload
                if patch_payload is not None
                else self._parse_checkpoint_patch_payload()
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
        if self.current_source == "local":
            # task-16482: approval unblocks the boundary -- restart the
            # engine so the run continues to its next phase/checkpoint.
            self._start_local_engine(run_id)
        return updated

    async def watch_selected_run_events(
        self, *, after_id: int = 0
    ) -> list[dict[str, Any]]:
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
            item = ListItem(
                Static(title), id=f"research-run-{self._record_get(run, 'id')}"
            )
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
        if (
            event.get("event") == "snapshot"
            and isinstance(data, dict)
            and isinstance(data.get("run"), dict)
        ):
            run_payload = dict(data["run"])
            run_payload.setdefault(
                "query", self._record_get(self.selected_run, "query", "")
            )
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

    def on_mount(self) -> None:
        # task-16486: while a local engine run is in flight, keep the
        # selected run's detail current without manual Refresh.
        self.set_interval(2.0, self._auto_refresh_selected_run)

    async def _auto_refresh_selected_run(self) -> None:
        """Refresh the selected LOCAL run's detail while it is non-terminal
        (task-16486). Payload state (bundle/artifact selections) is
        preserved; server-source selections and terminal runs are skipped
        (server observation already has its own streaming surface)."""
        if self.current_source != "local" or self.selected_run is None:
            return
        if self._record_field(self.selected_run, "status") in {
            "completed", "failed", "cancelled", "draft",
        }:
            return
        try:
            run_id = self._selected_run_id()
            updated = await self.controller.get_run(self.current_source, run_id)
        except Exception:
            return  # transient controller errors must not spam the status line
        if updated is None:
            return
        self.selected_run = updated
        self._update_detail(updated)

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
        renderable = render_bundle_summary(self.current_bundle)
        try:
            self.query_one("#research-bundle-detail", Static).update(renderable)
        except Exception:
            pass

    def _render_artifact_detail(self) -> None:
        if not self.is_mounted:
            return
        renderable = render_artifact(self.current_artifact)
        try:
            self.query_one("#research-artifact-detail", Static).update(renderable)
        except Exception:
            pass

    def _render_event_log(self) -> None:
        if not self.is_mounted:
            return
        renderable = (
            "\n".join(self.event_log_entries)
            if self.event_log_entries
            else "No research events captured yet."
        )
        try:
            self.query_one("#research-event-log", Static).update(renderable)
        except Exception:
            pass

    def _resolve_artifact_name(self, artifact_name: str | None) -> str:
        resolved = str(artifact_name or "").strip()
        if not resolved and self.is_mounted:
            try:
                resolved = self.query_one(
                    "#research-artifact-name", Input
                ).value.strip()
            except Exception:
                resolved = ""
        if not resolved and self.current_bundle:
            resolved = str(next(iter(self.current_bundle.keys()), "")).strip()
        return resolved

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        resolved = str(checkpoint_id or "").strip()
        if not resolved and self.is_mounted:
            try:
                resolved = self.query_one(
                    "#research-checkpoint-id", Input
                ).value.strip()
            except Exception:
                resolved = ""
        if not resolved:
            resolved = str(
                self._record_get(self.selected_run, "latest_checkpoint_id", "") or ""
            ).strip()
        return resolved

    def _parse_checkpoint_patch_payload(self) -> dict[str, Any] | None:
        raw_text = ""
        if self.is_mounted:
            try:
                raw_text = self.query_one(
                    "#research-checkpoint-patch", TextArea
                ).text.strip()
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
