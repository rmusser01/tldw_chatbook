"""Library-owned local authoring screen with durable, recoverable A/B state."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections import deque
from functools import partial
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Chunking import lab_state
from tldw_chatbook.Chunking.lab_models import RunResult
from tldw_chatbook.Chunking.lab_preflight import current_local_runtime, prepare_recipe
from tldw_chatbook.Chunking.lab_recovery import (
    MAX_ENVELOPE_BYTES,
    export_recovery,
    parse_recovery,
)
from tldw_chatbook.RAG_Admin.chunking_lab_service import (
    ExpectedTemplate,
    save_lab_template,
)
from tldw_chatbook.UI.Chunking_Lab_Modules import ChunkingTemplatesChanged
from tldw_chatbook.UI.Chunking_Lab_Modules.dialogs import LabDialog, TemplateDialog
from tldw_chatbook.UI.Chunking_Lab_Modules.editor_region import EditorRegion
from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion
from tldw_chatbook.UI.Chunking_Lab_Modules.sample_region import (
    SAMPLE_BYTES,
    SampleRegion,
    read_sample_excerpt,
    read_sample_file,
    sample_source_label,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_bytes,
    open_private_binary,
)


def _validation(draft: dict) -> str:
    if draft["parse_error"]:
        error = draft["parse_error"]
        return f"JSON line {error['line']}, column {error['column']}: {error['message']}. Correct it or Discard invalid edit."
    if draft["pending_controls"]:
        return (
            "Incomplete controls: "
            + ", ".join(draft["pending_controls"])
            + ". Correct them or Discard invalid edit before using JSON."
        )
    try:
        prepare_recipe(
            json.loads(draft["parsed_json"]), runtime=current_local_runtime()
        )
    except (TypeError, ValueError) as exc:
        return f"{exc}. Incompatible options are preserved; edit them explicitly in Full JSON."
    return "Ready for local preview. Advanced metadata is preserved; classifier rules do not run."


def _save_payload(session, role: str) -> tuple[dict, dict, dict | None, str]:
    candidate_id, candidate = next(
        (key, value)
        for key, value in session.candidates.items()
        if value["role"] == role
    )
    if role == "A":
        recipe = candidate["pinned_recipe"]
        body = json.loads(recipe["authored_json"])
        effective = json.loads(recipe["effective_json"])
        for section in ("preprocessing", "chunking", "postprocessing"):
            body[section] = effective[section]
        record = candidate.get("template_record") or {}
        fields = {
            key: record.get(key, [] if key == "tags" else "")
            for key in ("name", "description", "tags")
        }
        expected = (
            {
                key: record[key]
                for key in ("id", "uuid", "version", "is_builtin")
                if key in record
            }
            if all(key in record for key in ("id", "uuid", "version"))
            else None
        )
    else:
        draft = candidate["draft"]
        if draft["parse_error"] or draft["pending_controls"]:
            raise ValueError(
                "Correct or discard the current invalid edit before saving B"
            )
        body, fields, expected = (
            json.loads(draft["parsed_json"]),
            draft["record_fields"],
            draft["expected_record"],
        )
    prepare_recipe(body, runtime=current_local_runtime())
    return body, lab_state.captured_record_fields(fields), expected, candidate_id


def _export_template(session) -> bytes:
    """Preserve a syntactically valid authored object without Run/Save admission."""
    candidate = next(
        value for value in session.candidates.values() if value["role"] == "B"
    )
    draft = candidate["draft"]
    body = json.loads(draft["raw_json"])
    if not isinstance(body, dict):
        raise TypeError(
            "Template export requires a JSON object; use recovery export to preserve any raw draft"
        )
    return json.dumps(
        {
            **lab_state.captured_record_fields(draft["record_fields"]),
            "template_json": body,
            "version": "1.0",
            "source": "tldw_chatbook",
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def _write_selected_file(selected: str, payload: bytes, overwrite: bool) -> None:
    path = validate_path_simple(selected, probe_existing=False)
    if not path.is_absolute():
        raise ValueError("Choose an absolute output path")
    precondition = PrivateFileWritePrecondition.missing()
    if path.exists():
        if not overwrite:
            raise ValueError(
                "File already exists; explicitly choose Replace existing file"
            )
        with open_private_binary(path) as opened:
            precondition = PrivateFileWritePrecondition.from_opened(opened)
    atomic_private_write_bytes(path, payload, target_precondition=precondition)


def _inspect_recovery(selected: str) -> tuple[bytes, str]:
    """Read and validate once for bounded inspection, with no store authority."""
    path = validate_path_simple(selected, require_exists=True)
    if not path.is_file():
        raise ValueError("Choose a regular recovery JSON file")
    with path.open("rb") as stream:
        payload = stream.read(MAX_ENVELOPE_BYTES + 1)
    imported = parse_recovery(payload)
    sample = imported.samples[imported.view["sample_hash"]]
    lines = [
        f"Sample: {len(sample['text']):,} characters · {len(sample['text'].encode('utf-8')):,} UTF-8 bytes",
        "Sample preview: "
        + sample["text"][:240]
        + ("…" if len(sample["text"]) > 240 else ""),
    ]
    for candidate in sorted(
        imported.candidates.values(), key=lambda value: value["role"]
    ):
        record = (
            (candidate.get("draft") or {}).get("record_fields")
            or candidate.get("template_record")
            or {}
        )
        lines.append(
            f"{candidate['role']}: {record.get('name', '')[:80] or 'Unnamed recipe'}"
        )
    for result in imported.results.values():
        report = result.get("report")
        lines.append(
            f"Retained result: {result['status']} · {len(report['chunks']) if report else 0:,} chunks · {result['request']['recipe']['runtime']['backend'][:80]}"
        )
    lines.append(
        "Replacement preserves the displaced session for Undo restore until a content edit."
    )
    return payload, "\n".join(lines)


def _inspection_runs(session, candidate_id: str) -> tuple[str | None, str | None]:
    """Keep current batch membership distinct from retained successful evidence."""
    candidate = session.candidates[candidate_id]
    current, previous = (
        candidate.get("current_run_id"),
        candidate.get("previous_run_id"),
    )
    if session.batch:
        member = next(
            (
                key
                for key, request in session.batch["requests"].items()
                if request["candidate_id"] == candidate_id
            ),
            None,
        )
        if member is not None and member != current:
            if session.results.get(current, {}).get("status") == "completed":
                previous = current
            current = member
    return current, previous


class ChunkingLabScreen(BaseAppScreen):
    """Compose native sample, authoring and captured-result regions."""

    BINDINGS: ClassVar = [
        Binding("r", "lab_run", "Run B", show=False),
        Binding("p", "lab_pin", "Pin A", show=False),
        Binding("s", "lab_save", "Save B", show=False),
    ]
    BUNDLED_CSS = """
    ChunkingLabScreen #lab-title, ChunkingLabScreen #lab-status, ChunkingLabScreen #lab-message { height: auto; padding: 0 1; }
    ChunkingLabScreen #lab-title { text-style: bold; background: $surface; }
    ChunkingLabScreen #lab-controls, ChunkingLabScreen #lab-actions { height: 3; }
    ChunkingLabScreen #lab-controls Button, ChunkingLabScreen #lab-actions Button { min-width: 8; width: auto; padding: 0 1; }
    ChunkingLabScreen #lab-menu-slot { width: 1fr; min-width: 20; height: 3; }
    ChunkingLabScreen #lab-menu { width: 100%; }
    ChunkingLabScreen #lab-work { height: 1fr; min-height: 0; }
    ChunkingLabScreen #lab-inputs { width: 2fr; height: 1fr; min-width: 0; }
    ChunkingLabScreen #lab-results-scroll { width: 3fr; height: 1fr; min-width: 0; min-height: 0; }
    ChunkingLabScreen #lab-result-choices { height: 1; }
    ChunkingLabScreen #lab-result-choices Button { height: 1; min-width: 10; width: auto; padding: 0 1; }
    ChunkingLabScreen ResultsRegion { width: 1fr; height: 1fr; min-width: 0; }
    ChunkingLabScreen.-narrow #lab-inputs, ChunkingLabScreen.-narrow #lab-results-scroll { width: 1fr; }
    ChunkingLabScreen.-narrow #lab-actions { height: 1; }
    ChunkingLabScreen.-narrow ResultsRegion { height: 20; min-height: 20; }
    ChunkingLabScreen.-narrow #result-tables { height: 7; min-height: 7; }
    ChunkingLabScreen.-narrow #chunk-inspector { height: 5; min-height: 5; }
    """

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "chunking_lab", **kwargs)
        self.nav_bar_active = "library"
        self.coordinator = None
        self.return_route = "library"
        self.local_media_id = None
        self._ready = asyncio.Event()
        self._unsubscribe = None
        self._edits = deque()
        self._edit_task = None
        self._render_generation = 0
        self._input_generation = 0
        self._rendered_session = None
        self._results_cache = {}
        self._result_epoch = None
        self._result_revisions = {}
        self._active_region = "sample"
        self._initializing = True
        self._leaving = False

    def compose_content(self) -> ComposeResult:
        yield Static(
            "Chunking Lab · Library · Local execution", id="lab-title", markup=False
        )
        yield Static("Loading local recovery…", id="lab-status", markup=False)
        with Horizontal(id="lab-controls"):
            yield Button("Back", id="lab-back")
            yield Button("Sample", id="lab-show-sample")
            yield Button("Configure", id="lab-show-configure")
            yield Button("Results", id="lab-show-results")
            with Vertical(id="lab-menu-slot"):
                yield Select(
                    [
                        (label, value)
                        for label, value in (
                            ("Load text file", "sample-file"),
                            ("Load explicit excerpt", "excerpt"),
                            ("Saved templates", "catalog"),
                            ("Import template JSON", "import-template"),
                            ("Export template B", "export-template"),
                            ("Export recovery", "export-recovery"),
                            ("Restore recovery", "restore"),
                            ("Undo restore", "undo-restore"),
                            ("Clear local recovery", "clear"),
                        )
                    ],
                    prompt="Files / Recovery",
                    id="lab-menu",
                )
        with Horizontal(id="lab-actions"):
            for label, identity in (
                ("Run B", "run"),
                ("Run both", "both"),
                ("Cancel", "cancel"),
                ("Pin A", "pin"),
                ("Save A", "save-a"),
                ("Save B", "save-b"),
                ("Retry", "retry"),
            ):
                yield Button(
                    label,
                    id=f"lab-{identity}",
                    disabled=True,
                    variant="primary" if identity == "run" else "default",
                )
        yield Static(
            "Full sample text and completed results are stored locally for recovery.",
            id="lab-message",
            markup=False,
        )
        with Horizontal(id="lab-work"):
            with Vertical(id="lab-inputs"):
                yield SampleRegion(id="lab-sample", disabled=True)
                yield EditorRegion(id="lab-editor", disabled=True)
            with VerticalScroll(id="lab-results-scroll"):
                with Horizontal(id="lab-result-choices"):
                    for side in ("a", "b"):
                        yield Button(
                            f"{side.upper()} current",
                            id=f"lab-current-{side}",
                            disabled=True,
                        )
                        yield Button(
                            f"{side.upper()} Previous",
                            id=f"lab-previous-{side}",
                            disabled=True,
                        )
                yield ResultsRegion(id="lab-results")

    def apply_navigation_context(self, context: dict) -> None:
        route = context.get("return_route", "library")
        self.return_route = (
            route
            if isinstance(route, str)
            and route != "chunking_lab"
            and resolve_screen_route(route)
            else "library"
        )
        media_id = context.get("local_media_id")
        if media_id is not None and (type(media_id) is not int or media_id < 1):
            raise ValueError("Chunking Lab requires a positive local media ID")
        self.local_media_id = media_id

    def on_mount(self) -> None:
        # Textual 8 binds TextArea F6 to select_line. Release only this screen's
        # instance maps so ADR-031's app-global pane key can reach its delegate.
        for editor in self.query(TextArea):
            editor._bindings.key_to_bindings.pop("f6", None)
        self._layout()
        self.run_worker(
            self._load, group="lab-load", exclusive=True, exit_on_error=False
        )

    async def wait_until_ready(self) -> None:
        """Wait for load completion, including a visible failed-load outcome."""
        await self._ready.wait()

    async def _load(self) -> None:
        if not self.is_mounted or not self.query("#lab-message"):
            return
        self._ready.clear()
        try:
            owner = await self.app_instance.get_chunking_lab_coordinator()
            if not self.is_mounted or not self.query("#lab-message"):
                return
            self.coordinator = owner
            self._unsubscribe = self.coordinator.subscribe(self._coordinator_changed)
            region = self.coordinator.session.view.get("region", "sample")
            self._active_region = (
                region if region in ("sample", "configure", "results") else "sample"
            )
            self._initializing = False
            await self._refresh_session()
            if self.coordinator.recovery_warning:
                self._message(
                    "Recovered the previous valid local checkpoint; the newest checkpoint could not be read. Review the restored draft before continuing."
                )
            if self.local_media_id is not None:
                media_id, self.local_media_id = self.local_media_id, None
                try:
                    await self._load_local_media(media_id)
                except (ValueError, TypeError, OSError) as exc:
                    self._message(
                        f"Could not load Library source: {exc}. Your recovered draft is unchanged."
                    )
        except Exception as exc:  # noqa: BLE001 - load boundary must retain unavailable recovery, never replace it.
            if not self.is_mounted or not self.query("#lab-message"):
                return
            self._message(
                f"Could not load local recovery ({type(exc).__name__}). Retry reads it again; no empty draft is writable."
            )
            self.query_one("#lab-status", Static).update("Recovery load failed")
            self.query_one("#lab-retry", Button).disabled = False
        finally:
            self._ready.set()

    def _message(self, text: str) -> None:
        self.query_one("#lab-message", Static).update(text)

    def _coordinator_changed(self, event) -> None:
        if self.is_mounted:
            self._observe_result_revisions()
            self.run_worker(
                self._refresh_session,
                group="lab-render",
                exclusive=True,
                exit_on_error=False,
            )

    def _observe_result_revisions(self) -> None:
        """Remember result publication before any later UI delta advances revision."""
        if self.coordinator is None:
            return
        session = self.coordinator.session
        if self._result_epoch != session.epoch:
            self._result_epoch = session.epoch
            self._result_revisions.clear()
        self._result_revisions = {
            run_id: self._result_revisions.get(run_id, session.revision)
            for run_id in session.results
        }

    def _b_id(self, session=None) -> str:
        return next(
            key
            for key, value in (session or self.coordinator.session).candidates.items()
            if value["role"] == "B"
        )

    def queue_edit(self, operation, *args, **kwargs) -> None:
        """Queue a delta, never a stale whole-session computation."""
        if self._initializing or self._leaving or self.coordinator is None:
            return
        self._observe_result_revisions()
        self._input_generation += 1
        self._edits.append(partial(operation, *args, **kwargs))
        if self._edit_task is None or self._edit_task.done():
            self._edit_task = asyncio.create_task(self._apply_edits())

    async def _apply_edits(self) -> None:
        while True:
            while self._edits:
                edit = self._edits.popleft()
                while True:
                    before = self.coordinator.session
                    if self.coordinator.guarded:
                        await asyncio.sleep(0.01)
                        continue
                    try:
                        changed = await asyncio.to_thread(edit, before)
                        if self.coordinator.session is not before:
                            continue
                        self._observe_result_revisions()
                        self.coordinator.set_session(changed)
                    except (ValueError, TypeError, RuntimeError) as exc:
                        self._message(str(exc))
                    break
            await self._refresh_session(edit_complete=True)
            # Rendering yields: retain ownership of deltas queued during it.
            if not self._edits:
                return

    async def drain_edits(self) -> None:
        """Drain deltas before taking any Run/Save/export/navigation snapshot."""
        while self._edits or (
            self._edit_task is not None and not self._edit_task.done()
        ):
            if self._edit_task is None or self._edit_task.done():
                self._edit_task = asyncio.create_task(self._apply_edits())
            await asyncio.shield(self._edit_task)

    @on(SampleRegion.Changed)
    def sample_changed(self, event: SampleRegion.Changed) -> None:
        event.stop()
        self.queue_edit(
            lambda session: lab_state.replace_sample(
                session, event.text, {"kind": "paste"}
            )
        )

    @on(EditorRegion.Edited)
    def editor_changed(self, event: EditorRegion.Edited) -> None:
        event.stop()
        self._queue_editor_edit(event.kind, event.field, event.value)

    def _queue_editor_edit(self, kind: str, field: str, value: str) -> None:
        if kind == "record":
            field = "tags_text" if field == "tags" else field
            self.queue_edit(
                lambda session: lab_state.edit_record_fields(
                    session, self._b_id(session), {field: value}
                )
            )
        elif kind == "json":
            self.queue_edit(
                lambda session: lab_state.edit_json(session, self._b_id(session), value)
            )
        else:
            self.queue_edit(
                lambda session: lab_state.edit_control(
                    session, self._b_id(session), field, value
                )
            )

    async def _refresh_session(self, *, edit_complete: bool = False) -> None:
        if (
            self.coordinator is None
            or not self.is_mounted
            or not self.query("#lab-message")
        ):
            return
        self._observe_result_revisions()
        self._render_generation += 1
        generation, inputs = self._render_generation, self._input_generation
        session = self.coordinator.session
        draft = session.candidates[self._b_id(session)]["draft"]
        validation, document = await asyncio.to_thread(
            lambda: (
                _validation(draft),
                json.loads(draft["parsed_json"]) if draft["parsed_json"] else {},
            )
        )
        a = b = None
        stale = set()
        previous_ids, previous_sides = set(), set()
        choices = session.view.get("result_choices", {})
        current_complete = True
        controls = {}
        for candidate_id, candidate in session.candidates.items():
            current_id, previous_id = _inspection_runs(session, candidate_id)
            current_complete &= (
                session.results.get(current_id, {}).get("status") == "completed"
            )
            previous = choices.get(candidate_id) == "previous"
            run_id = previous_id if previous else current_id
            controls[candidate["role"]] = (previous_id is not None, previous)
            if previous:
                previous_sides.add(candidate["role"])
                if run_id is not None:
                    previous_ids.add(run_id)
            result = None
            if run_id in session.results:
                stored = session.results[run_id]
                cached = self._results_cache.get(run_id)
                if cached is None or cached[0] is not stored:
                    cached = (
                        stored,
                        await asyncio.to_thread(RunResult.model_validate, stored),
                    )
                    self._results_cache[run_id] = cached
                result = cached[1]
                if await asyncio.to_thread(
                    lab_state.is_result_stale, session, candidate_id, run_id
                ):
                    stale.add(run_id)
            if candidate["role"] == "A":
                a = result
            else:
                b = result
        if (
            generation != self._render_generation
            or session is not self.coordinator.session
            or not self.is_mounted
            or not self.query("#lab-message")
        ):
            return
        self._results_cache = {
            key: value
            for key, value in self._results_cache.items()
            if key in session.results
        }
        # Never replace visible newer keystrokes with a delayed preparation.
        editing = self._edit_task is not None and not self._edit_task.done()
        if (
            inputs == self._input_generation
            and not self._edits
            and (edit_complete or not editing)
        ):
            snapshot = session.samples[session.view["sample_hash"]]
            sample = snapshot["text"]
            self.query_one("#lab-sample-source", Static).update(
                sample_source_label(snapshot["source"])
            )
            editor = self.query_one("#lab-sample-text", TextArea)
            with self.query_one(SampleRegion).prevent(TextArea.Changed):
                if editor.text != sample:
                    editor.load_text(sample)
            self.query_one(EditorRegion).present(draft, validation, document)
        self.query_one(SampleRegion).disabled = (
            self.coordinator.guarded or self._leaving
        )
        self.query_one(EditorRegion).disabled = (
            self.coordinator.guarded or self._leaving
        )
        results = self.query_one(ResultsRegion)
        for side in ("A", "B"):
            available, previous = controls.get(side, (False, False))
            for choice in ("current", "previous"):
                button = self.query_one(f"#lab-{choice}-{side.lower()}", Button)
                button.display = side in controls
                button.disabled = (
                    self.coordinator.guarded
                    or self._leaving
                    or (choice == "previous" and not available)
                )
                button.variant = (
                    "primary" if (choice == "previous") == previous else "default"
                )
        results.configure_view(
            session.view,
            previous_ids=frozenset(previous_ids),
            previous_sides=frozenset(previous_sides),
        )
        results.show_results(
            a,
            b,
            stale_ids=frozenset(stale),
            comparison_block="Inspecting Previous output; current batch is incomplete. Run both to compare."
            if previous_sides and not current_complete
            else None,
        )
        status = self.coordinator.save_status
        saved = (
            status.acknowledged is not None
            and status.acknowledged.revision == session.revision
            and status.state == "saved"
        )
        status_text = (
            "Saved locally"
            if saved
            else "Save conflict · Export recovery; reopen deliberately"
            if status.state == "conflict"
            else "Save failed · Retry or Export recovery"
            if status.state == "failed"
            else "Saving…"
            if status.state == "saving"
            else "Not yet saved"
        )
        unsaved_results = any(
            status.acknowledged is None
            or status.acknowledged.epoch != session.epoch
            or status.acknowledged.revision < revision
            for revision in self._result_revisions.values()
        )
        if unsaved_results:
            status_text += " · Unsaved result"
        self.query_one("#lab-status", Static).update(
            status_text + (" · Preview running" if self.coordinator.busy else "")
        )
        for identity in ("run", "both", "pin", "save-a", "save-b", "retry"):
            self.query_one(f"#lab-{identity}", Button).disabled = (
                self.coordinator.guarded or self._leaving
            )
        for identity in ("run", "both"):
            self.query_one(f"#lab-{identity}", Button).disabled |= self.coordinator.busy
        has_a = any(c["role"] == "A" for c in session.candidates.values())
        self.query_one("#lab-both", Button).disabled |= not has_a
        self.query_one("#lab-save-a", Button).disabled |= not has_a
        self.query_one("#lab-pin", Button).label = "Replace A" if has_a else "Pin A"
        self.query_one("#lab-cancel", Button).disabled = not self.coordinator.busy
        self._rendered_session = session
        self._layout()

    def on_resize(self) -> None:
        if self.is_mounted:
            self._layout()

    def _layout(self) -> None:
        narrow = self.size.width < 120
        self.set_class(narrow, "-narrow")
        self.query_one("#lab-inputs").display = self._active_region != "results"
        self.query_one(SampleRegion).display = self._active_region == "sample"
        self.query_one(EditorRegion).display = self._active_region == "configure"
        self.query_one("#lab-results-scroll").display = (
            not narrow or self._active_region == "results"
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action.startswith("lab_") and isinstance(
            self.focused, (Input, TextArea, Select)
        ):
            return False
        return super().check_action(action, parameters)

    def action_lab_run(self) -> None:
        self.run_worker(self._safe(self.run_candidates()), exit_on_error=False)

    def action_lab_pin(self) -> None:
        self.run_worker(self._safe(self._pin()), exit_on_error=False)

    def action_lab_save(self) -> None:
        self.run_worker(self._safe(self._save("B")), exit_on_error=False)

    def action_focus_next_workbench_pane(self) -> None:
        """Honor the global F6 delegate by cycling the three task regions."""
        regions = ("sample", "configure", "results")
        self._active_region = regions[
            (regions.index(self._active_region) + 1) % len(regions)
        ]
        self._layout()
        target = {
            "sample": "#lab-sample-text",
            "configure": "#lab-name",
            "results": "#view-b",
        }[self._active_region]
        self.call_after_refresh(self.query_one(target).focus)
        selected_region = self._active_region
        self.queue_edit(
            lambda session: lab_state.update_view(session, {"region": selected_region})
        )

    def on_descendant_focus(self) -> None:
        local_keys = (
            not isinstance(self.focused, (Input, TextArea, Select))
            and self.coordinator is not None
        )
        hints = (("f6", "region"), ("f1", "help"))
        if local_keys:
            hints += (("r", "run B"), ("p", "pin A"), ("s", "save B"))
        self.register_footer_shortcuts(source="chunking_lab", shortcuts=hints)

    async def _safe(self, operation) -> None:
        try:
            await operation
        except Exception as exc:  # noqa: BLE001 - worker boundary reports recoverable action failures to the user.
            self._message(str(exc))

    async def run_candidates(self, both: bool = False) -> None:
        """Capture drained current inputs, then supervise real local execution."""
        await self.drain_edits()
        if self.coordinator is None:
            raise ValueError("Load local recovery before running")
        ids = tuple(self.coordinator.session.candidates) if both else (self._b_id(),)
        await self.coordinator.run(ids)
        await self._refresh_session()

    async def _pin(self) -> None:
        await self.drain_edits()
        replace = any(
            c["role"] == "A" for c in self.coordinator.session.candidates.values()
        )
        if (
            replace
            and await self._dialog(
                LabDialog(
                    "Replace pinned A?",
                    "Replace the frozen baseline with B's current completed result.",
                    accept="Replace A",
                )
            )
            is None
        ):
            return
        self.queue_edit(
            lambda session: lab_state.pin_baseline(session, replace=replace)
        )
        await self.drain_edits()

    async def _dialog(self, dialog: LabDialog) -> dict | None:
        future = asyncio.get_running_loop().create_future()
        await self.app.push_screen(
            dialog,
            lambda result: future.set_result(result) if not future.done() else None,
        )
        return await future

    def _catalog_service(self):
        from tldw_chatbook.Chunking.chunking_interop_library import (
            ChunkingInteropService,
        )

        database = getattr(self.app_instance, "media_db", None)
        if database is None:
            raise ValueError(
                "The local Media database is unavailable; preview and recovery remain available"
            )
        return ChunkingInteropService(database)

    async def _save(self, role: str) -> None:
        await self.drain_edits()
        captured_session = self.coordinator.session
        body, fields, expected, candidate_id = await asyncio.to_thread(
            _save_payload, captured_session, role
        )
        answer = await self._dialog(
            LabDialog(
                f"Save {role} as local template",
                "Saving creates or updates a reusable template. Source content and defaults are unchanged.",
                fields={
                    "name": ("Name", fields.get("name", "")),
                    "description": ("Description", fields.get("description", "")),
                    "tags": (
                        "Tags · comma separated",
                        captured_session.candidates[candidate_id]["draft"][
                            "record_fields"
                        ].get("tags_text", ", ".join(fields.get("tags", [])))
                        if role == "B"
                        else ", ".join(fields.get("tags", [])),
                    ),
                },
                checks={"new": "Save as new (required for built-ins)"}
                if expected
                else {},
                checked=frozenset({"new"})
                if expected and expected.get("is_builtin")
                else frozenset(),
                accept="Save template",
                on_edit=(
                    lambda field, value: self._queue_editor_edit("record", field, value)
                )
                if role == "B"
                else None,
            )
        )
        if answer is None:
            return
        await self.drain_edits()
        if role == "B":
            # Capture again after the dialog's durable record edits have drained.
            captured_session = self.coordinator.session
            body, fields, expected, candidate_id = await asyncio.to_thread(
                _save_payload, captured_session, role
            )
        else:
            fields = {
                "name": answer["name"],
                "description": answer["description"],
                "tags": [
                    tag.strip() for tag in answer["tags"].split(",") if tag.strip()
                ],
            }
        try:
            await self.save_candidate(
                role,
                fields=fields,
                as_new=bool(answer.get("new")),
                captured=(body, expected, candidate_id, captured_session),
            )
        except Exception as exc:  # noqa: BLE001 - catalog failure must retain the authored draft.
            await self._dialog(
                LabDialog(
                    "Template was not saved",
                    f"{exc}\nYour draft is retained. Correct the name/options and Save again, choose Save as new, or reload through Saved templates.",
                    accept="Keep editing",
                )
            )

    async def save_candidate(
        self,
        role: str,
        *,
        fields: dict | None = None,
        as_new: bool = False,
        captured=None,
    ) -> dict:
        """Save captured authored B or pinned authored/effective A through the catalog."""
        await self.drain_edits()
        session = self.coordinator.session
        if captured is None:
            body, authored_fields, expected, candidate_id = await asyncio.to_thread(
                _save_payload, session, role
            )
            fields = fields or authored_fields
        else:
            body, expected, candidate_id, session = captured
        candidate = session.candidates[candidate_id]
        draft, generation = candidate.get("draft"), candidate.get("draft_generation")
        record = await asyncio.to_thread(
            save_lab_template,
            self._catalog_service(),
            body=body,
            name=fields["name"],
            description=fields.get("description", ""),
            tags=fields.get("tags", []),
            expected=None
            if as_new or expected is None
            else ExpectedTemplate(
                **{key: expected[key] for key in ("id", "uuid", "version")}
            ),
        )
        if role == "B":
            self.queue_edit(
                lambda current: lab_state.associate_saved_record(
                    current,
                    candidate_id,
                    record,
                    captured_draft=draft,
                    captured_generation=generation,
                )
            )
            await self.drain_edits()
        self.post_message(ChunkingTemplatesChanged(record["id"], record["version"]))
        self._message(
            f"Saved local template: {record['name']}. Available in Library ingest."
        )
        return record

    async def _load_local_media(self, media_id: int) -> None:
        database = getattr(self.app_instance, "media_db", None)
        if database is None:
            raise ValueError(
                "Local Library text is unavailable; paste text or choose a local UTF-8 file"
            )
        record = await asyncio.to_thread(database.get_media_by_id, media_id)
        if not record or record.get("deleted"):
            raise ValueError("That local Library item is no longer available")
        text = record.get("content")
        if not isinstance(text, str) or not text:
            raise ValueError(
                "This local item has no extracted text. Extract it in Library first."
            )
        source = {"kind": "local_media", "local_media_id": media_id}
        if await asyncio.to_thread(lambda: len(text.encode("utf-8"))) > SAMPLE_BYTES:
            choice = await self._dialog(
                LabDialog(
                    "Choose a Library excerpt",
                    "Full text exceeds 2 MiB. Select exact character positions; nothing is silently truncated.",
                    fields={
                        "start": ("Start · zero-based character", "0"),
                        "end": ("End · exclusive character", ""),
                    },
                    accept="Copy excerpt",
                )
            )
            if choice is None:
                return
            start, end = int(choice["start"]), int(choice["end"])
            if start < 0 or end <= start or end > len(text):
                raise ValueError("Choose a nonempty range within the Library text")
            text = text[start:end]
            source = {
                **source,
                "kind": "local_media_excerpt",
                "start": start,
                "end": end,
            }
        self.queue_edit(lambda session: lab_state.replace_sample(session, text, source))
        await self.drain_edits()

    @on(Select.Changed, "#lab-menu")
    def menu_changed(self, event: Select.Changed) -> None:
        event.stop()
        if not isinstance(event.value, str):
            return
        action = event.value
        with event.select.prevent(Select.Changed):
            event.select.value = Select.NULL
        self.run_worker(self._safe(self._menu_action(action)), exit_on_error=False)

    async def _menu_action(self, action: str) -> None:
        await self.drain_edits()
        if self.coordinator is None and action != "restore":
            raise ValueError("Retry loading local recovery before authoring")
        if action == "undo-restore":
            await self.coordinator.undo_restore()
        elif action == "clear":
            if (
                await self._dialog(
                    LabDialog(
                        "Clear local recovery?",
                        "Remove the Lab sample, drafts and retained results. Saved reusable templates remain. This is not secure erasure.",
                        accept="Clear recovery",
                    )
                )
                is not None
            ):
                await self.coordinator.clear()
        elif action == "catalog":
            from tldw_chatbook.RAG_Admin.local_rag_admin_service import (
                LocalRAGAdminService,
            )

            service = self._catalog_service()
            records = await asyncio.to_thread(
                LocalRAGAdminService(
                    service.media_db, chunking_service=service
                ).list_templates
            )
            record = await self._dialog(TemplateDialog(records))
            if record is not None:
                self.queue_edit(
                    lambda session: lab_state.replace_template(
                        session,
                        self._b_id(session),
                        record["template_json"],
                        record_fields={
                            key: record.get(key, [] if key == "tags" else "")
                            for key in ("name", "description", "tags")
                        },
                        expected_record={
                            **{key: record[key] for key in ("id", "uuid", "version")},
                            "is_builtin": bool(record.get("is_builtin")),
                        },
                    )
                )
                await self.drain_edits()
                self._active_region = "configure"
        else:
            fields = {"path": ("Absolute file path", "")}
            if action == "excerpt":
                fields.update(
                    {
                        "start": ("Start · zero-based character", "0"),
                        "end": ("End · exclusive character", ""),
                    }
                )
            checks = (
                {"overwrite": "Replace existing file"}
                if action.startswith("export-")
                else {}
            )
            answer = await self._dialog(
                LabDialog(
                    action.replace("-", " ").capitalize(),
                    "Recovery files contain full sample text and results. Choose your own file path; imported data never chooses write targets.",
                    fields=fields,
                    checks=checks,
                )
            )
            if answer is not None:
                await self.file_operation(action, answer)
        await self._refresh_session()

    async def file_operation(self, action: str, choice: dict) -> None:
        """Execute one explicit user-selected bounded local file operation off-loop."""
        await self.drain_edits()
        selected = choice["path"]
        if action == "restore":
            payload, summary = await asyncio.to_thread(_inspect_recovery, selected)
            owner = self.coordinator
            unavailable = self._replacement_unavailable()
            answer = await self._dialog(
                LabDialog(
                    "Inspect recovery snapshot",
                    summary + ("\n\n" + unavailable if unavailable else ""),
                    accept="Replace current session",
                    accept_disabled=bool(unavailable),
                )
            )
            if answer is None:
                return
            if owner is not self.coordinator or self._replacement_unavailable():
                raise ValueError(
                    "Replacement authority changed; inspect the snapshot again after Retry"
                )
            # Retained bytes are immutable: replacement validates this same snapshot,
            # never a later file revision. Validation precedes coordinator quiescence.
            await owner.replace_recovery(payload)
            region = owner.session.view.get("region", "sample")
            self._active_region = (
                region if region in ("sample", "configure", "results") else "sample"
            )
            await self._refresh_session()
            return
        if action in {"sample-file", "excerpt"}:
            text, source = (
                await asyncio.to_thread(
                    read_sample_excerpt,
                    selected,
                    int(choice["start"]),
                    int(choice["end"]),
                )
                if action == "excerpt"
                else await asyncio.to_thread(read_sample_file, selected)
            )
            self.queue_edit(
                lambda session: lab_state.replace_sample(session, text, source)
            )
            await self.drain_edits()
        elif action in {"export-recovery", "export-template"}:
            if action == "export-recovery":
                payload = await asyncio.to_thread(
                    export_recovery, self.coordinator.session
                )
            else:
                payload = await asyncio.to_thread(
                    _export_template, self.coordinator.session
                )
            await asyncio.to_thread(
                _write_selected_file, selected, payload, bool(choice.get("overwrite"))
            )
            self._message("Exported to the selected file.")
        else:

            def read():
                path = validate_path_simple(selected, require_exists=True)
                if not path.is_file():
                    raise ValueError("Choose a regular template JSON file")
                limit = 8 * 1024 * 1024
                # A regular path can become a FIFO after admission. Open without
                # waiting for a writer, then verify the actual descriptor before
                # reading. Imported files keep their existing permissions.
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_NOCTTY", 0)
                    | getattr(os, "O_BINARY", 0)
                )
                descriptor = os.open(path, flags)
                try:
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise ValueError("Choose a regular template JSON file")
                    with os.fdopen(descriptor, "rb", closefd=False) as stream:
                        payload = stream.read(limit + 1)
                finally:
                    os.close(descriptor)
                if len(payload) > limit:
                    raise ValueError("Import exceeds the supported file limit")
                return payload

            payload = await asyncio.to_thread(read)
            if action == "import-template":

                def parse():
                    record = json.loads(payload.decode("utf-8"))
                    if not isinstance(record, dict):
                        raise TypeError("Template import must be a JSON object")
                    body = record.get("template_json", record)
                    fields = {
                        key: record.get(key, [] if key == "tags" else "")
                        for key in ("name", "description", "tags")
                    }
                    return body, fields

                body, fields = await asyncio.to_thread(parse)
                self.queue_edit(
                    lambda session: lab_state.replace_template(
                        session,
                        self._b_id(session),
                        body,
                        record_fields=fields,
                        expected_record=None,
                    )
                )
                await self.drain_edits()
                self._active_region = "configure"
            else:
                raise ValueError("Unknown file operation")
        await self._refresh_session()

    def _replacement_unavailable(self) -> str | None:
        if self.coordinator is None:
            return "Replacement unavailable: local recovery could not be loaded. Retry the existing store before replacing; this snapshot remains available for read-only inspection."
        if self.coordinator.guarded or self._leaving:
            return "Replacement unavailable while another session transition settles."
        if self.coordinator.save_status.state in ("failed", "conflict"):
            return "Replacement unavailable until local persistence and authority are repaired. Retry or deliberately reopen the profile so the displaced session can be preserved."
        return None

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        identity = event.button.id or ""
        if not identity.startswith("lab-"):
            return
        event.stop()
        if identity.startswith(("lab-current-", "lab-previous-")):
            _, choice, side = identity.split("-")

            def select_result(session):
                candidate_id = next(
                    key
                    for key, candidate in session.candidates.items()
                    if candidate["role"] == side.upper()
                )
                return lab_state.update_view(
                    session,
                    {
                        "result_choices": {
                            **session.view.get("result_choices", {}),
                            candidate_id: choice,
                        }
                    },
                )

            self.queue_edit(select_result)
            return
        if identity.startswith("lab-show-"):
            self._active_region = identity.removeprefix("lab-show-")
            self._layout()
            self.queue_edit(
                lambda session: lab_state.update_view(
                    session, {"region": self._active_region}
                )
            )
            return
        self.run_worker(self._safe(self._action(identity)), exit_on_error=False)

    async def _action(self, identity: str) -> None:
        if identity == "lab-back":
            from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

            self.app.post_message(NavigateToScreen(self.return_route))
        elif identity == "lab-retry":
            if self.coordinator is None:
                await self._load()
            else:
                await self.drain_edits()
                await self.coordinator.cancel()
                await self._refresh_session()
        elif identity in {"lab-run", "lab-both"}:
            await self.run_candidates(identity == "lab-both")
        elif identity == "lab-cancel":
            await self.drain_edits()
            await self.coordinator.cancel()
        elif identity == "lab-pin":
            await self._pin()
        elif identity in {"lab-save-a", "lab-save-b"}:
            await self._save(identity[-1].upper())
        elif identity == "lab-discard":
            self.queue_edit(
                lambda session: lab_state.discard_pending_edit(
                    session, self._b_id(session)
                )
            )
        elif identity == "lab-undo":
            self.queue_edit(lab_state.undo_edit)

    @on(ResultsRegion.SelectionChanged)
    def result_selection(self, event: ResultsRegion.SelectionChanged) -> None:
        event.stop()
        self.queue_edit(
            lambda session: lab_state.update_view(session, {"results": event.view})
        )

    @on(ResultsRegion.RerunRequested)
    def rerun_requested(self, event: ResultsRegion.RerunRequested) -> None:
        event.stop()
        self.run_worker(self._safe(self.run_candidates(True)), exit_on_error=False)

    async def confirm_navigation(self) -> bool:
        """Cancel and commit before route teardown; failures retain visible recovery."""
        self._leaving = True
        try:
            await self.drain_edits()
            if self.coordinator is not None:
                await self.coordinator.cancel()
            return True
        except Exception:  # noqa: BLE001 - any failed checkpoint must veto navigation.
            self._message(
                "Could not save local recovery. Stay here to Retry or Export recovery before leaving."
            )
            return False
        finally:
            self._leaving = False

    async def confirm_quit(self) -> bool:
        return await self.confirm_navigation()

    async def prepare_for_quit(self) -> bool:
        return await self.confirm_navigation()

    async def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        await self.drain_edits()
        if self.coordinator is not None and self.coordinator.busy:
            await self.coordinator.cancel()
