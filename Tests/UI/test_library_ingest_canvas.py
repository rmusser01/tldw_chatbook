"""Tests for ``LibraryIngestCanvas`` rendering and message contracts.

Widget-only tests mount the canvas directly in a bare ``App`` subclass and
assert on widget existence, rendered text, and posted messages. The canvas is
render-only: all state is supplied by ``build_library_ingest_state``.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import ComposeResult
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from Tests.UI.background_signals import wait_for_background_signal
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
    build_active_ingest_consent_scope,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestCanvasState,
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import (
    LibraryScreen,
    _LibraryIngestStartConsent,
)
from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
    LibraryIngestCanvas,
    LibraryIngestQueuePanel,
    _summarise_option,
)


class _CanvasHost(ConsolidatedCSSApp):
    def __init__(
        self,
        state: LibraryIngestCanvasState,
        *,
        external_busy: bool = False,
        external_status: str = "",
    ) -> None:
        super().__init__()
        self._state = state
        self._external_busy = external_busy
        self._external_status = external_status

    def compose(self) -> ComposeResult:
        kwargs: dict[str, object] = {"id": "library-ingest-canvas"}
        if self._external_busy or self._external_status:
            kwargs.update(
                external_busy=self._external_busy,
                external_status=self._external_status,
            )
        yield LibraryIngestCanvas(self._state, **kwargs)


@pytest.mark.asyncio
async def test_chunking_template_save_invalidates_mounted_picker_cache():
    from tldw_chatbook.UI.Chunking_Lab_Modules import ChunkingTemplatesChanged
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        INGEST_CHUNK_TEMPLATE_PICKER_ID,
    )

    assert hasattr(LibraryIngestCanvas, "invalidate_chunk_templates")
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _CanvasHost(state)
    names = ["before"]

    class Catalog:
        async def list_templates(self, *, mode):
            assert mode == "local"
            return [{"name": name} for name in names]

    host.rag_admin_scope_service = Catalog()
    async with host.run_test() as pilot:
        canvas = host.query_one(LibraryIngestCanvas)
        await canvas._fetch_chunk_templates()
        picker = canvas.query_one(f"#{INGEST_CHUNK_TEMPLATE_PICKER_ID}", Select)
        picker.value = "before"
        await pilot.pause()
        names.append("after")
        canvas.post_message(ChunkingTemplatesChanged(1, 2))
        await pilot.pause()
        await host.workers.wait_for_complete()
        assert canvas._chunk_template_names == ["before", "after"]
        assert picker.value == "before"
        picker.value = "after"
        await pilot.pause()
        assert picker.value == "after"


class _QueuePanelHost(ConsolidatedCSSApp):
    """Mount only the real queue panel with the shipped app stylesheet."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, state: LibraryIngestCanvasState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield LibraryIngestQueuePanel(
            self._state,
            id="library-ingest-queue-panel",
        )


class _MessageRecordingHost(ConsolidatedCSSApp):
    """Host that records ``OptionValueChanged`` and ``OptionPanelToggled``."""

    def __init__(self, state: LibraryIngestCanvasState) -> None:
        super().__init__()
        self._state = state
        self.option_changes: list[LibraryIngestCanvas.OptionValueChanged] = []
        self.panel_toggles: list[LibraryIngestCanvas.OptionPanelToggled] = []
        self.tooling_detail_toggles: list[
            LibraryIngestCanvas.ToolingDetailToggled
        ] = []
        self.parakeet_install_requests = 0
        self.directory_browse_requests: list[tuple[str, str]] = []
        self.transcribe_cpp_gguf_requests = 0
        self.external_cancel_requests = 0

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")

    @on(LibraryIngestCanvas.OptionValueChanged)
    def _record_option_change(
        self, event: LibraryIngestCanvas.OptionValueChanged
    ) -> None:
        self.option_changes.append(event)

    @on(LibraryIngestCanvas.OptionPanelToggled)
    def _record_panel_toggle(
        self, event: LibraryIngestCanvas.OptionPanelToggled
    ) -> None:
        self.panel_toggles.append(event)

    @on(LibraryIngestCanvas.ToolingDetailToggled)
    def _record_tooling_detail_toggle(
        self, event: LibraryIngestCanvas.ToolingDetailToggled
    ) -> None:
        self.tooling_detail_toggles.append(event)

    @on(LibraryIngestCanvas.ParakeetInstallRequested)
    def _record_parakeet_install_request(
        self, _event: LibraryIngestCanvas.ParakeetInstallRequested
    ) -> None:
        self.parakeet_install_requests += 1

    @on(LibraryIngestCanvas.TranscribeCppGGUFRequested)
    def _record_transcribe_cpp_gguf_request(
        self, _event: LibraryIngestCanvas.TranscribeCppGGUFRequested
    ) -> None:
        self.transcribe_cpp_gguf_requests += 1

    def on_library_ingest_canvas_external_preparation_cancel_requested(
        self, _event: object
    ) -> None:
        self.external_cancel_requests += 1

    def on_library_ingest_canvas_directory_browse_requested(self, event) -> None:
        self.directory_browse_requests.append((event.group, event.name))


def _default_form() -> LibraryIngestFormState:
    return LibraryIngestFormState(path="/tmp/test")


@pytest.mark.asyncio
async def test_preflight_checking_renders_status_static():
    """When ``preflight_checking`` is true, a "Checking…" status appears."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight_checking=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        status = pilot.app.query_one("#ingest-preflight-status", Static)
        assert "Checking…" in str(status.renderable)


@pytest.mark.asyncio
async def test_preflight_errors_render_with_retry_button():
    """Pre-flight errors are listed and a Retry button is provided."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        error_static = pilot.app.query_one("#ingest-preflight-error-0", Static)
        assert "Path not found" in str(error_static.renderable)
        retry_button = pilot.app.query_one("#ingest-preflight-retry", Button)
        assert "Retry" in str(retry_button.label)


@pytest.mark.asyncio
async def test_preflight_warnings_render_with_prefix():
    """Pre-flight warnings are rendered with a warning prefix."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[
                {
                    # Production tooling warnings ALWAYS carry ``feature``
                    # (get_tooling_warnings); a featureless warning is an
                    # advisory and renders elsewhere by design.
                    "feature": "pdf_processing",
                    "label": "PDF processing",
                    "hint": "PyMuPDF is not installed.",
                }
            ],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        warning_static = pilot.app.query_one("#ingest-preflight-warning-0", Static)
        text = str(warning_static.renderable)
        assert text.startswith("⚠")
        assert "PDF processing" in text
        assert "PyMuPDF is not installed." in text


@pytest.mark.asyncio
async def test_type_breakdown_and_estimate_render():
    """The type-breakdown and estimate Statics render the expected copy."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf", "/tmp/b.pdf"],
                "generic": ["/tmp/c.txt"],
            },
            warnings=[],
            errors=[],
            total_size=2048,
            truncated=False,
            total_files=3,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        breakdown = pilot.app.query_one("#ingest-type-breakdown", Static)
        breakdown_text = str(breakdown.renderable)
        assert "2 PDF documents" in breakdown_text
        assert "1 plain text file" in breakdown_text

        estimate = pilot.app.query_one("#ingest-estimate", Static)
        assert "3 files · 2.0 KB" == str(estimate.renderable)


@pytest.mark.asyncio
async def test_unsupported_files_summary_renders():
    """Unsupported files are summarized with a failure note."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/weird.xyz"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#ingest-unsupported-summary", Static)
        # (task-2100) Unsupported-only selection is gate-blocked, so the
        # line names the file instead of promising a failure row that a
        # blocked submit never records.
        assert str(summary.renderable) == (
            "Unsupported: weird.xyz."
            " Supported: PDF documents, Word/Office documents, audio/video"
            " files, e-books, images, plain text files, web pages (by URL)."
        )


@pytest.mark.asyncio
async def test_existing_controls_are_still_present():
    """The path input, Browse, Start import, and queue heading remain."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-ingest-path", Input)
        assert pilot.app.query_one("#library-ingest-browse", Button)
        assert pilot.app.query_one("#library-ingest-start", Button)
        assert pilot.app.query_one("#library-ingest-queue-heading", Static)
        assert pilot.app.query_one("#library-ingest-queue-empty", Static)


@pytest.mark.asyncio
async def test_no_preflight_renders_no_summary_widgets():
    """Without a pre-flight result, only the generic panel is mounted."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        for widget_id in (
            "#ingest-preflight-status",
            "#ingest-preflight-error-0",
            "#ingest-preflight-retry",
            "#ingest-preflight-warning-0",
            "#ingest-type-breakdown",
            "#ingest-estimate",
            "#ingest-unsupported-summary",
            "#type-group-pdf",
        ):
            assert len(pilot.app.query(widget_id)) == 0
        # Generic panel is always rendered so global options stay accessible.
        assert len(pilot.app.query("#type-group-generic")) == 1


@pytest.mark.asyncio
async def test_preflight_checking_suppresses_summary():
    """``preflight_checking=True`` hides the full summary, even if a result is
    already available."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=1024,
            truncated=False,
            total_files=1,
        ),
        preflight_checking=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#ingest-preflight-status")) == 1
        for widget_id in (
            "#ingest-type-breakdown",
            "#ingest-estimate",
            "#ingest-unsupported-summary",
        ):
            assert len(pilot.app.query(widget_id)) == 0
        # (task-2042) Options panels stay mounted during a re-analysis:
        # hiding them made ``preflight_checking`` a STRUCTURAL flag, and the
        # resulting full recompose swallowed clicks in flight. A re-check
        # lasts well under a second; last-known panels are less flicker.
        assert len(pilot.app.query("#type-group-pdf")) == 1


@pytest.mark.asyncio
async def test_multiple_errors_and_warnings_are_enumerated():
    """Several errors/warnings each get their own indexed Static."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[
                {
                    "feature": "pdf_processing",
                    "label": "PDF processing",
                    "hint": "PyMuPDF is not installed.",
                },
                {
                    "feature": "audio_processing",
                    "label": "Audio",
                    "hint": "ffmpeg not found.",
                },
            ],
            errors=["Path not found", "URL unreachable"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert "Path not found" in str(
            pilot.app.query_one("#ingest-preflight-error-0", Static).renderable
        )
        assert "URL unreachable" in str(
            pilot.app.query_one("#ingest-preflight-error-1", Static).renderable
        )
        assert pilot.app.query_one("#ingest-preflight-retry", Button)

        assert "PyMuPDF is not installed." in str(
            pilot.app.query_one("#ingest-preflight-warning-0", Static).renderable
        )
        assert "ffmpeg not found." in str(
            pilot.app.query_one("#ingest-preflight-warning-1", Static).renderable
        )


@pytest.mark.asyncio
async def test_error_and_warning_markup_is_escaped():
    """Rich markup metacharacters in errors/warnings are rendered verbatim."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[
                {"feature": "pdf_processing", "label": "Hint", "hint": "[/bracket]"}
            ],
            errors=["[bold]not bold[/bold]"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        error_static = pilot.app.query_one("#ingest-preflight-error-0", Static)
        assert error_static.visual.plain == "[bold]not bold[/bold]"

        warning_static = pilot.app.query_one("#ingest-preflight-warning-0", Static)
        assert warning_static.visual.plain == (
            "⚠ Hint isn't installed — needed for [/bracket]."
        )


# --- Per-type options panels ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_type_group_panels_render_for_detected_groups():
    """One collapsible panel is rendered per detected supported type group."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "generic": ["/tmp/b.txt"],
            },
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
        ),
    )
    app = _CanvasHost(state)
    # (task-14825 #7) Pinned installed: a title only advertises values of
    # fields the user can actually edit, so what this asserts must not
    # depend on which extras this venv happens to have.
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            pdf_panel = pilot.app.query_one("#type-group-pdf", Collapsible)
            generic_panel = pilot.app.query_one(
                "#type-group-generic", Collapsible
            )
            assert "PDF documents" in str(pdf_panel.title)
            assert str(pdf_panel.title) == "PDF documents"
            assert "Import behavior" in str(generic_panel.title)
            # (task-28007 AC#6) The collapsed header states the analysis
            # state the fold hides; the default is off.
            assert str(generic_panel.title) == "Import behavior · analysis off"

            scope = pilot.app.query_one(
                "#type-group-pdf .type-group-scope", Static
            )
            assert "Applies to every PDF document in this import." in str(
                scope.renderable
            )

            assert pilot.app.query_one("#opt-pdf-reset", Button)
            assert pilot.app.query_one("#opt-generic-reset", Button)


@pytest.mark.asyncio
async def test_expand_collapse_all_buttons_render_when_type_groups_present():
    """Bulk expand/collapse buttons render only when MULTIPLE panels exist
    (task-2016: a single panel has nothing to expand "all" of)."""
    with_groups = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"], "generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
        ),
    )
    app = _CanvasHost(with_groups)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#ingest-expand-all", Button)
        assert pilot.app.query_one("#ingest-collapse-all", Button)


@pytest.mark.asyncio
async def test_dependent_controls_disabled_when_dependency_missing():
    """Fields whose ``depends_on`` feature is unavailable render disabled."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=False,
    ):
        async with app.run_test() as pilot:
            engine_select = pilot.app.query_one("#opt-pdf-pdf_engine", Select)
            extract_checkbox = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            assert engine_select.disabled is True
            assert extract_checkbox.disabled is True


@pytest.mark.asyncio
async def test_non_dependent_controls_stay_enabled():
    """Fields with no ``depends_on`` dependency render enabled."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        analyze_checkbox = pilot.app.query_one("#opt-generic-analyze", Checkbox)
        chunk_checkbox = pilot.app.query_one("#opt-generic-chunk", Checkbox)
        encoding_input = pilot.app.query_one("#opt-generic-encoding", Select)
        assert analyze_checkbox.disabled is False
        assert chunk_checkbox.disabled is False
        assert encoding_input.disabled is False


@pytest.mark.asyncio
async def test_chunk_size_disabled_when_chunk_unchecked():
    """Chunk size and overlap inputs are disabled until Chunk is checked."""
    form = _default_form()
    # The panel renders from ``type_options``; the screen writes the scalar
    # ``form.chunk`` mirror and this dict together on every toggle. Chunking
    # is on by default now, so the off case has to be stated explicitly.
    form.chunk = False
    form.type_options = {"generic": {"chunk": False}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        chunk_size_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        chunk_overlap_input = pilot.app.query_one("#opt-generic-chunk_overlap", Input)
        assert chunk_size_input.disabled is True
        assert chunk_overlap_input.disabled is True


def _import_behavior_state(*, backend: str, analyze: bool = False) -> LibraryIngestCanvasState:
    """Build one expanded generic panel for a selected effective backend."""
    form = _default_form()
    form.expanded_type_groups.add("generic")
    form.type_options = {"generic": {"analyze": analyze}}
    return build_library_ingest_state(
        (),
        form=form,
        ingest_backend=backend,
        runtime_source="server",
        server_ingest_available=True,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_local_import_behavior_renders_shared_controls_and_prompt_reasons():
    """Local mode renders only controls that can affect a local import."""
    app = _CanvasHost(_import_behavior_state(backend="local"))

    async with app.run_test(size=(80, 50)) as pilot:
        assert "Import behavior" in str(
            pilot.app.query_one("#type-group-generic", Collapsible).title
        )
        assert pilot.app.query_one("#opt-generic-overwrite_existing", Checkbox)
        assert pilot.app.query_one("#opt-generic-generate_embeddings", Checkbox)
        custom_prompt = pilot.app.query_one("#opt-generic-custom_prompt", TextArea)
        system_prompt = pilot.app.query_one("#opt-generic-system_prompt", TextArea)
        assert custom_prompt.disabled is True
        assert system_prompt.disabled is True
        assert len(pilot.app.query("#opt-generic-keep_original_file")) == 0
        labels = [str(static.renderable) for static in pilot.app.query(Static)]
        assert labels.count("Custom prompt — needs Analyze after import on") == 1
        assert labels.count("System prompt — needs Analyze after import on") == 1


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_server_import_behavior_adds_keep_original_file_and_fits_compact_width():
    """Server-only controls stay inside the compact canvas without clipping."""
    app = _CanvasHost(_import_behavior_state(backend="server", analyze=True))

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        keep_original = pilot.app.query_one(
            "#opt-generic-keep_original_file", Checkbox
        )
        for widget_id in (
            "#opt-generic-custom_prompt",
            "#opt-generic-system_prompt",
            "#opt-generic-keep_original_file",
        ):
            assert pilot.app.query_one(widget_id).region.right <= 80
        assert keep_original.disabled is False


def test_populated_multiline_prompt_has_bounded_collapsed_title() -> None:
    """Prompt bodies must not leak into a collapsed option-panel receipt."""
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    custom_prompt = next(
        field
        for field in get_capabilities("generic").fields
        if field.name == "custom_prompt"
    )

    summary = _summarise_option(
        custom_prompt,
        "Summarize each claim in detail.\n" * 20,
    )

    assert summary == "Custom prompt: set"


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_option_value_changed_posted_on_multiline_prompt_change():
    """Editing a prompt forwards its complete multiline value to the owner."""
    app = _MessageRecordingHost(_import_behavior_state(backend="local", analyze=True))

    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#opt-generic-custom_prompt", TextArea)
        prompt.text = "Summarize the key claims.\nPreserve names."
        await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "generic"
        and event.name == "custom_prompt"
        and event.value == "Summarize the key claims.\nPreserve names."
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_checkbox_change():
    """Toggling a checkbox posts ``OptionValueChanged`` with the right group/name."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            checkbox = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            checkbox.value = True
            await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "pdf"
        and event.name == "ocr"
        and event.value is True
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_select_change():
    """Changing a select posts ``OptionValueChanged`` with the new value."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            select = pilot.app.query_one("#opt-pdf-pdf_engine", Select)
            select.value = "pymupdf"
            await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "pdf" and event.name == "pdf_engine" and event.value == "pymupdf"
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_input_change():
    """Typing in a number/text option input posts ``OptionValueChanged``."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    async with app.run_test() as pilot:
        option_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        option_input.value = "1234"
        await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "generic" and event.name == "chunk_size" and event.value == "1234"
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_panel_toggled_posted_on_expand_collapse():
    """Expanding/collapsing a type-group panel posts ``OptionPanelToggled``."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    async with app.run_test() as pilot:
        panel = pilot.app.query_one("#type-group-generic", Collapsible)
        panel.collapsed = False
        await pilot.pause()
        panel.collapsed = True
        await pilot.pause()

    assert len(app.panel_toggles) == 2
    assert app.panel_toggles[0].group == "generic"
    assert app.panel_toggles[0].expanded is True
    assert app.panel_toggles[1].group == "generic"
    assert app.panel_toggles[1].expanded is False


@pytest.mark.asyncio
async def test_type_group_number_input_renders_with_value_and_placeholder():
    """Generic number options render as Inputs with their default value/placeholder."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(type_options={"generic": {"chunk_size": 750}}),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        chunk_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        assert chunk_input.value == "750"
        assert chunk_input.placeholder == "Chunk size"


# --- Progress, structured errors, retry, and recent ingests -----------------


@pytest.mark.asyncio
async def test_parsing_row_reserves_progress_line_before_first_worker_tick():
    """Removing the active-row reservation makes the first tick shift the queue."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.PARSING,
        progress=None,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _QueuePanelHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        assert progress.display is True
        assert str(progress.renderable) == "Preparing import"


@pytest.mark.asyncio
async def test_writing_row_reserves_saving_line_without_progress_payload():
    """A defensive payload gap must not collapse the WRITING reservation."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.WRITING,
        progress=None,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _QueuePanelHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        assert progress.display is True
        assert str(progress.renderable) == "Saving to Library"


@pytest.mark.asyncio
async def test_progress_line_uses_formatter_without_repeating_state():
    """Bypassing the formatter repeats ``parsing`` and loses truthful percent copy."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.PARSING,
        progress={
            "phase": "extracting",
            "message": "Extracting page 2 of 5",
            "percent": 40.0,
        },
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _QueuePanelHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        rendered = str(progress.renderable)
        assert rendered == "40% · Extracting page 2 of 5"
        assert "Â·" not in rendered


@pytest.mark.asyncio
async def test_progress_line_absent_when_not_present():
    """A queued job without progress does not render a progress line."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.QUEUED,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-progress-ingest-job-1")) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [LIBRARY_TEST_SIZE, (72, 18)])
async def test_progress_detail_paints_below_row_without_obscuring_actions_or_neighbor(
    size,
):
    """The final compositor keeps dim telemetry in normal and constrained layouts."""
    parsing = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/interview.wav",
        state=IngestJobState.PARSING,
        progress={
            "phase": "transcribing",
            "message": "Transcribing speaker one of two",
            "percent": 25.0,
        },
    )
    queued = LibraryIngestJob(
        job_id="ingest-job-2",
        source_path="/tmp/neighbor.txt",
        state=IngestJobState.QUEUED,
    )
    state = build_library_ingest_state(
        (parsing, queued),
        form=_default_form(),
    )
    app = _QueuePanelHost(state)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        row = pilot.app.query_one("#library-ingest-row-0", Static)
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        actions = pilot.app.query_one(".library-ingest-row-actions")
        neighbor = pilot.app.query_one("#library-ingest-row-1", Static)
        counts = pilot.app.query_one("#library-ingest-queue-counts", Static)

        assert row.region.bottom <= progress.region.y
        assert progress.region.bottom <= actions.region.y
        assert actions.region.bottom <= neighbor.region.y
        assert progress.styles.color == counts.styles.color
        assert progress.styles.color != row.styles.color

        compositor = pilot.app.screen._compositor
        for widget in (row, progress, actions, neighbor):
            assert widget in compositor.visible_widgets
        strips = compositor.render_strips()

        def painted_text(widget) -> str:
            return "\n".join(
                strips[y].text
                for y in range(widget.region.y, widget.region.bottom)
            )

        assert "parsing" in painted_text(row)
        assert "25%" in painted_text(progress)
        assert "Transcribing speaker" in painted_text(progress)
        assert "Cancel" in painted_text(actions)
        assert "queued" in painted_text(neighbor)
        assert "neighbor.txt" in painted_text(neighbor)


@pytest.mark.parametrize(
    "copy",
    [
        "Import active. Start again to queue a duplicate.",
        "2 active files. Start again to queue all.",
        "Import active; 2 may fail. Start again to queue.",
    ],
)
@pytest.mark.asyncio
async def test_active_ingest_confirm_copy_fits_fixed_gate_at_72x18(copy):
    """Each binding instruction paints whole above Start at minimum geometry."""
    form = LibraryIngestFormState(path="C:/docs/a.txt")
    state = build_library_ingest_state(
        (),
        form=form,
        start_confirm_armed=True,
        start_confirm_line=copy,
    )
    app = _CanvasHost(state)

    async with app.run_test(size=(72, 18)) as pilot:
        await pilot.pause()
        quiet = app.query_one("#library-ingest-start-quiet-line", Static)
        start = app.query_one("#library-ingest-start", Button)
        strips = app.screen._compositor.render_strips()
        painted = "".join(
            strip.text
            for strip in strips[quiet.region.y : quiet.region.bottom]
        )

        assert quiet.region.height == 1
        assert copy in painted
        assert "…" not in painted
        assert quiet.region.bottom <= start.region.y


@pytest.mark.asyncio
async def test_active_confirm_update_preserves_start_input_focus_cursor_and_scroll(
    tmp_path,
):
    """Arming consent updates only the mounted gate line and warning class."""
    app = _build_test_app()
    _seed_conversations(app, ())
    registry = LibraryIngestJobRegistry()
    for index in range(12):
        registry.submit(source_path=f"C:/docs/queued-{index}.txt")
    app.library_ingest_jobs = registry
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        path_input = screen.query_one("#library-ingest-path", Input)
        start = screen.query_one("#library-ingest-start", Button)
        canvas = screen.query_one(LibraryIngestCanvas)
        path_input.value = str(tmp_path / "active.txt")
        path_input.focus()
        await pilot.pause()
        path_input.cursor_position = 3
        canvas.scroll_to(y=2, animate=False, force=True, immediate=True)
        start_region = start.region
        scroll_y = canvas.scroll_y
        compositor = screen._compositor
        strips = compositor.render_strips()
        painted_start = "\n".join(
            strips[y].text[start_region.x : start_region.right]
            for y in range(start_region.y, start_region.bottom)
        )
        assert screen.focused is path_input
        assert path_input.cursor_position == 3
        assert scroll_y == 2
        assert start in compositor.visible_widgets
        assert start_region.width > 0
        assert start_region.height > 0
        assert "Start import" in painted_start

        screen._library_ingest_start_consent = _LibraryIngestStartConsent(
            fingerprint="active-test",
            admission_scope=build_active_ingest_consent_scope(
                [str(tmp_path / "active.txt")],
                origin="local",
                active_job_ids=("ingest-job-1",),
                active_source_count=1,
            ),
            tooling_affected_count=0,
            is_folder=False,
        )
        screen._update_library_ingest_gate(screen._build_library_ingest_state())
        await pilot.pause()

        assert screen.query_one("#library-ingest-path", Input) is path_input
        assert screen.query_one("#library-ingest-start", Button) is start
        assert screen.focused is path_input
        assert path_input.cursor_position == 3
        assert canvas.scroll_y == scroll_y
        assert start.region == start_region


@pytest.mark.asyncio
async def test_local_stt_cancel_then_force_stop_actions_render_exclusively():
    running = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/speech.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing"},
    )
    state = build_library_ingest_state((running,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-ingest-cancel-ingest-job-1", Button)
        assert not list(pilot.app.query("#library-ingest-force-stop-ingest-job-1"))

    cancelling = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/speech.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing", "cancel_requested": True},
    )
    state = build_library_ingest_state((cancelling,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        force = pilot.app.query_one(
            "#library-ingest-force-stop-ingest-job-1",
            Button,
        )
        assert str(force.label) == "Force stop"
        assert not list(pilot.app.query("#library-ingest-cancel-ingest-job-1"))


@pytest.mark.asyncio
async def test_show_details_button_renders_for_error_detail():
    """A failed job with structured error detail gets a Show details button."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Bad codec",
        error_detail={"category": "codec_error", "message": "Codec missing"},
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        btn = pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)
        assert "Show details" in str(btn.label)


@pytest.mark.asyncio
async def test_show_details_button_absent_without_error_detail():
    """A failed job without error detail does not render Show details."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Bad codec",
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-details-ingest-job-1")) == 0


@pytest.mark.asyncio
async def test_retry_button_renders_for_retryable_failure():
    """A retryable failed job still renders the Retry action."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Network error",
        permanent=False,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        btn = pilot.app.query_one("#library-ingest-retry-ingest-job-1", Button)
        assert "Retry" in str(btn.label)


@pytest.mark.asyncio
async def test_retry_button_hidden_for_unsupported_file_type():
    """Retry is withheld when the error category is unsupported_file_type."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.xyz",
        state=IngestJobState.FAILED,
        error="Unsupported file type",
        permanent=False,
        error_detail={
            "category": "unsupported_file_type",
            "message": "Unsupported extension",
        },
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-retry-ingest-job-1")) == 0
        # The structured-error surface is still available.
        assert pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)


@pytest.mark.asyncio
async def test_transcribe_cpp_failure_renders_only_eligible_recovery_actions():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/private/voice.wav",
        state=IngestJobState.FAILED,
        error="The selected GGUF cannot be used by transcribe.cpp.",
        permanent=False,
        error_detail={
            "category": "stt_failure",
            "code": "artifact_incompatible",
            "message": "The selected GGUF cannot be used by transcribe.cpp.",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)

    async with app.run_test() as pilot:
        choose = pilot.app.query_one(
            "#library-ingest-choose-gguf-ingest-job-1", Button
        )
        retry = pilot.app.query_one(
            "#library-ingest-retry-faster-whisper-ingest-job-1", Button
        )
        assert "Choose another GGUF" in str(choose.label)
        assert str(retry.label) == "Retry with faster-whisper"
        assert not list(pilot.app.query("#library-ingest-retry-ingest-job-1"))


@pytest.mark.asyncio
async def test_research_failure_renders_only_honest_catalog_retry_action():
    """Research ownership suppresses provider overrides that bypass its receipt."""

    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/private/voice.wav",
        state=IngestJobState.FAILED,
        error="The selected GGUF cannot be used by transcribe.cpp.",
        permanent=False,
        error_detail={
            "category": "stt_failure",
            "code": "artifact_incompatible",
            "message": "The selected GGUF cannot be used by transcribe.cpp.",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
        research_source_operation_id="source-op-retry-library-row",
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)

    async with app.run_test() as pilot:
        retry = pilot.app.query_one("#library-ingest-retry-ingest-job-1", Button)
        assert str(retry.label) == "Retry Research source"
        assert not list(
            pilot.app.query("#library-ingest-retry-faster-whisper-ingest-job-1")
        )
        assert not list(pilot.app.query("#library-ingest-choose-gguf-ingest-job-1"))


@pytest.mark.asyncio
async def test_transcribe_cpp_provider_shows_path_free_configured_picker():
    form = _default_form()
    form.type_options = {
        "audio_video": {"transcription_provider": "transcribe-cpp"}
    }
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/voice.wav"]},
            warnings=[],
            errors=[],
            total_size=1,
            truncated=False,
            total_files=1,
        ),
        transcribe_cpp_configured=True,
    )
    app = _MessageRecordingHost(state)

    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            button = pilot.app.query_one(
                "#opt-audio_video-choose-transcribe-cpp-gguf", Button
            )
            status = pilot.app.query_one(
                "#opt-audio_video-transcribe-cpp-status", Static
            )
            assert "Choose another GGUF" in str(button.label)
            assert "configured" in str(status.renderable).lower()
            assert "/" not in str(status.renderable)

            button.press()
            await pilot.pause()

    assert app.transcribe_cpp_gguf_requests == 1


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_terminal_jobs():
    """The Recent imports collapsible lists done/failed jobs but not queued."""
    done = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/done.txt",
        state=IngestJobState.DONE,
        media_id=1,
    )
    failed = LibraryIngestJob(
        job_id="ingest-job-2",
        source_path="/tmp/failed.txt",
        state=IngestJobState.FAILED,
        error="boom",
    )
    queued = LibraryIngestJob(
        job_id="ingest-job-3",
        source_path="/tmp/queued.txt",
        state=IngestJobState.QUEUED,
    )
    state = build_library_ingest_state(
        (done, failed, queued), form=_default_form()
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        recent = pilot.app.query_one("#library-ingest-recent", Collapsible)
        assert str(recent.title) == "Recent imports"
        assert recent.collapsed is True
        items = pilot.app.query(".library-ingest-recent-item")
        texts = [str(item.renderable) for item in items]
        assert any("done.txt" in t and "done" in t for t in texts)
        assert any("failed.txt" in t and "failed" in t for t in texts)
        assert not any("queued.txt" in t for t in texts)


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_when_queue_empty():
    """(task-2100) Recent imports is HIDDEN when there is nothing recent --
    it used to render always, and after a clear it expanded to an empty,
    unlabeled shell (round-3 critique evidence)."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert not list(pilot.app.query("#library-ingest-recent"))
        assert len(pilot.app.query("#library-ingest-queue-empty")) == 1


@pytest.mark.asyncio
async def test_mounting_option_panels_posts_no_option_changes():
    """Mounting a type-group panel must not look like a user edit.

    Textual's ``Select`` posts ``Changed`` as soon as it mounts, and an
    ``Input`` does the same for a non-empty ``value=``. Bubbling those as
    ``OptionValueChanged`` made the screen recompose, which remounted the
    select, which posted again -- an unbounded recompose cycle that pinned
    the UI at 100% CPU for every pdf/audio/ebook pre-flight (task-673).

    Every type group is mounted at once: ``pdf``, ``audio_video`` and
    ``ebook`` each carry a select and so each reproduced the freeze on its
    own, while ``generic`` (the one group with no select) is the reason
    plain text was the only content type that ever worked.
    """
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "audio_video": ["/tmp/a.mp3"],
                "ebook": ["/tmp/a.epub"],
                "generic": ["/tmp/a.txt"],
            },
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=4,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

    assert app.option_changes == [], (
        "mounting option panels emitted "
        f"{[(e.group, e.name, e.value) for e in app.option_changes]}"
    )


@pytest.mark.asyncio
async def test_audio_video_defaults_to_semantic_provider_with_exact_controls_disabled():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            provider = pilot.app.query_one(
                "#opt-audio_video-transcription_provider", Select
            )
            model = pilot.app.query_one("#opt-audio_video-transcription_model", Select)
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            assert provider.value == "default"
            assert model.disabled is True
            assert model_dir.disabled is True
            assert model_dir.value == ""
            install = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )
            assert "Install verified Parakeet v2 INT8" in str(install.label)
            assert install.disabled is True
            install.press()
            await pilot.pause()
            assert app.parakeet_install_requests == 0


@pytest.mark.asyncio
async def test_explicit_parakeet_enables_model_dir_and_verified_installer():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            provider = pilot.app.query_one(
                "#opt-audio_video-transcription_provider", Select
            )
            model = pilot.app.query_one("#opt-audio_video-transcription_model", Select)
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            install = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )

            assert provider.value == "parakeet-onnx"
            assert model.disabled is True
            assert model_dir.disabled is False
            assert install.disabled is False
            install.press()
            await pilot.pause()
            assert app.parakeet_install_requests == 1


@pytest.mark.asyncio
async def test_parakeet_model_directory_has_adjacent_browse_action():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            browse = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir-browse", Button
            )

            assert model_dir.parent is browse.parent
            assert browse.disabled is False
            browse.press()
            await pilot.pause()

    assert app.directory_browse_requests == [("audio_video", "transcription_model_dir")]


@pytest.mark.asyncio
async def test_parakeet_directory_row_fits_real_eighty_column_viewport() -> None:
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test(size=(80, 50)) as pilot:
            await pilot.pause()
            field = app.query_one("#opt-audio_video-transcription_model_dir", Input)
            browse = app.query_one(
                "#opt-audio_video-transcription_model_dir-browse", Button
            )
            row = field.parent

            assert row is browse.parent
            assert field.region.width > browse.region.width
            assert browse.region.right <= row.region.right <= 80


@pytest.mark.asyncio
async def test_external_preparation_status_and_cancel_are_stable_and_path_free() -> (
    None
):
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        runtime_source="server",
        server_ingest_available=True,
    )
    idle = _CanvasHost(state)
    async with idle.run_test():
        status = idle.query_one("#library-external-prepare-status", Static)
        cancel = idle.query_one("#library-external-prepare-cancel", Button)
        assert status.display is False
        assert cancel.display is False

    busy = _CanvasHost(
        state,
        external_busy=True,
        external_status="Verifying external Parakeet model…",
    )
    async with busy.run_test():
        status = busy.query_one("#library-external-prepare-status", Static)
        cancel = busy.query_one("#library-external-prepare-cancel", Button)
        assert status.display is True
        assert str(status.renderable) == "Verifying external Parakeet model…"
        assert "/" not in str(status.renderable)
        assert cancel.display is True
        assert "Cancel" in str(cancel.label)
        assert busy.query_one("#library-ingest-start", Button).disabled is True
        assert busy.query_one("#library-ingest-path", Input).disabled is True
        assert busy.query_one("#library-ingest-browse", Button).disabled is True
        assert busy.query_one("#library-ingest-backend-switch", Button).disabled is True


@pytest.mark.asyncio
async def test_external_busy_disables_routing_fields_and_shows_scope_helper() -> None:
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state, external_busy=True, external_status="Installing VAD…")
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test():
            helper = app.query_one("#library-external-scope-helper", Static)
            assert str(helper.renderable) == (
                "This import and its retries only · does not change Lab Models "
                "or your global source."
            )
            assert (
                app.query_one(
                    "#opt-audio_video-transcription_provider", Select
                ).disabled
                is True
            )
            assert (
                app.query_one(
                    "#opt-audio_video-transcription_model_dir", Input
                ).disabled
                is True
            )
            assert (
                app.query_one(
                    "#opt-audio_video-transcription_model_dir-browse", Button
                ).disabled
                is True
            )


@pytest.mark.asyncio
async def test_external_preparation_cancel_is_a_physical_message_action() -> None:
    class _CancelHost(_CanvasHost):
        def __init__(self, state: LibraryIngestCanvasState) -> None:
            super().__init__(
                state,
                external_busy=True,
                external_status="Verifying external Parakeet model…",
            )
            self.cancel_requests = 0

        def on_library_ingest_canvas_external_preparation_cancel_requested(
            self, _event: object
        ) -> None:
            self.cancel_requests += 1

    app = _CancelHost(build_library_ingest_state((), form=_default_form()))
    async with app.run_test() as pilot:
        await pilot.click("#library-external-prepare-cancel")
        await pilot.pause()

    assert app.cancel_requests == 1


@pytest.mark.asyncio
async def test_parakeet_model_directory_picker_updates_only_the_submission_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    screen = object.__new__(LibraryScreen)
    screen.app_instance = MagicMock()
    screen._library_ingest_form = LibraryIngestFormState(
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "language": "en",
            }
        }
    )
    prior = deepcopy(screen._library_ingest_form.type_options)
    event = SimpleNamespace(
        group="audio_video",
        name="transcription_model_dir",
        stop=MagicMock(),
    )

    LibraryScreen.handle_library_ingest_directory_browse(screen, event)

    event.stop.assert_called_once_with()
    fake_app.push_screen.assert_called_once()
    picker, callback = fake_app.push_screen.call_args.args
    assert isinstance(picker, SelectDirectory)
    selected = tmp_path / "user-owned-parakeet"
    selected.mkdir()
    await callback(selected)

    assert screen._library_ingest_form.type_options == {
        **prior,
        "audio_video": {
            **prior["audio_video"],
            "transcription_model_dir": str(selected),
        },
    }
    screen.app_instance._ensure_parakeet_source_service.assert_not_called()


@pytest.mark.asyncio
async def test_idle_external_fence_preserves_focused_form_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle semantic edits fence stale callbacks without remounting the form."""

    monkeypatch.setattr(
        library_screen_module, "get_cli_setting", lambda *_args, **_kwargs: None
    )
    app = _build_test_app()
    _seed_conversations(app, ())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-title")

        title = screen.query_one("#library-ingest-title", Input)
        title.focus()
        await pilot.pause()
        refresh = MagicMock(wraps=screen.refresh)
        screen.refresh = refresh

        title.value = "Atlas notes"
        title.cursor_position = 5
        await pilot.pause()
        screen.post_message(
            LibraryIngestCanvas.OptionValueChanged("pdf", "ocr_language", "fr")
        )
        screen.post_message(
            LibraryIngestCanvas.OptionValueChanged("generic", "chunk_size", "2048")
        )
        await pilot.pause()
        await pilot.pause()

        assert screen._library_ingest_form.title == "Atlas notes"
        assert screen._library_ingest_form.type_options["pdf"]["ocr_language"] == "fr"
        assert (
            screen._library_ingest_form.type_options["generic"]["chunk_size"]
            == "2048"
        )
        assert screen.query_one("#library-ingest-title", Input) is title
        assert screen.app.focused is title
        assert title.cursor_position == 5
        refresh.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_library_screen_multiline_prompt_typing_preserves_widget_and_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Textarea edits must use the in-place branch rather than remounting the form."""
    monkeypatch.setattr(
        library_screen_module, "get_cli_setting", lambda *_args, **_kwargs: None
    )
    app = _build_test_app()
    app._resolve_ingest_backend = lambda: "local"
    _seed_conversations(app, ())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    screen._library_ingest_form.analyze = True
    screen._library_ingest_form.expanded_type_groups.add("generic")
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#opt-generic-custom_prompt")

        prompt = screen.query_one("#opt-generic-custom_prompt", TextArea)
        prompt.focus()
        await pilot.press("a", "b", "enter", "c")
        await pilot.pause()

        assert prompt.text == "ab\nc"
        assert screen.query_one("#opt-generic-custom_prompt", TextArea) is prompt
        assert screen.app.focused is prompt


async def _wait_for_thread_signal(
    signal: threading.Event,
    pilot,
    *,
    what: str,
) -> None:
    """Bound a mounted wait for one production thread-worker checkpoint."""

    for _ in range(200):
        if signal.is_set():
            return
        await pilot.pause(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _backend_switch_screen(app, backend: dict[str, str]) -> LibraryScreen:
    """Build the real mounted ingest canvas around a controllable owner."""

    app._resolve_ingest_backend = lambda: backend["value"]
    _seed_conversations(app, ())
    screen = LibraryScreen(app)
    screen._build_library_ingest_state = lambda: build_library_ingest_state(
        (),
        form=screen._library_ingest_form,
        ingest_backend=backend["value"],
        runtime_source="server",
        server_ingest_available=True,
    )
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    screen._library_ingest_form.analyze = True
    screen._library_ingest_form.expanded_type_groups.add("generic")
    return screen


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_backend_switch_repaints_after_delayed_persistence_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed Server-to-Local save removes Server-only controls."""

    backend = {"value": "server"}
    save_entered = threading.Event()
    release_save = threading.Event()
    real_save = library_screen_module.save_setting_to_cli_config

    def delayed_save(section: str, key: str, target: str) -> bool:
        if (section, key) != ("library.ingest", "backend"):
            return real_save(section, key, target)
        save_entered.set()
        assert release_save.wait(5.0), "test never released backend persistence"
        backend["value"] = target
        return True

    app = _build_test_app()
    screen = _backend_switch_screen(app, backend)
    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", delayed_save
    )
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")

            screen.query_one("#library-ingest-backend-switch", Button).press()
            await _wait_for_thread_signal(
                save_entered, pilot, what="backend persistence start"
            )
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")
            await pilot.pause()
            assert screen.query_one("#opt-generic-keep_original_file", Checkbox)
            release_save.set()
            for _ in range(200):
                if (
                    backend["value"] == "local"
                    and screen._library_ingest_backend_target is None
                    and len(screen.query("#opt-generic-keep_original_file")) == 0
                ):
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError("backend persistence never completed")
            await _wait_for_selector(screen, pilot, "#opt-generic-custom_prompt")

            prompt = screen.query_one("#opt-generic-custom_prompt", TextArea)
            prompt.text = "Summarize this import."
            await pilot.pause()

            title = str(screen.query_one("#type-group-generic", Collapsible).title)
            assert "Keep original file" not in title
    finally:
        release_save.set()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_backend_switch_failure_restores_persisted_server_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Local preference save repaints the persisted owner."""

    backend = {"value": "server"}
    save_entered = threading.Event()
    release_save = threading.Event()
    real_save = library_screen_module.save_setting_to_cli_config

    def failing_save(section: str, key: str, target: str) -> bool:
        if (section, key) != ("library.ingest", "backend"):
            return real_save(section, key, target)
        save_entered.set()
        assert release_save.wait(5.0), "test never released backend persistence"
        raise OSError("private fixture detail")

    app = _build_test_app()
    screen = _backend_switch_screen(app, backend)
    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", failing_save
    )
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")

            screen.query_one("#library-ingest-backend-switch", Button).press()
            await _wait_for_thread_signal(
                save_entered, pilot, what="failing backend persistence start"
            )
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")
            await pilot.pause()
            persisted_control = screen.query_one(
                "#opt-generic-keep_original_file", Checkbox
            )

            release_save.set()
            for _ in range(200):
                controls = screen.query("#opt-generic-keep_original_file")
                if (
                    screen._library_ingest_backend_target is None
                    and len(controls) == 1
                    and controls.first() is not persisted_control
                ):
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError("failed preference never repainted owner state")
            assert backend["value"] == "server"
    finally:
        release_save.set()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_rapid_backend_switch_keeps_latest_server_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale Local completion cannot repaint or outlast a newer Server choice."""

    backend = {"value": "server"}
    entered = {"local": threading.Event(), "server": threading.Event()}
    release = {"local": threading.Event(), "server": threading.Event()}
    saves: list[str] = []
    real_save = library_screen_module.save_setting_to_cli_config

    def delayed_save(section: str, key: str, target: str) -> bool:
        if (section, key) != ("library.ingest", "backend"):
            return real_save(section, key, target)
        saves.append(target)
        entered[target].set()
        assert release[target].wait(5.0), f"test never released {target} save"
        backend["value"] = target
        return True

    app = _build_test_app()
    screen = _backend_switch_screen(app, backend)
    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", delayed_save
    )
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")

            screen.query_one("#library-ingest-backend-switch", Button).press()
            await _wait_for_thread_signal(
                entered["local"], pilot, what="first Local preference save"
            )
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")
            await pilot.pause()

            screen.query_one("#library-ingest-backend-switch", Button).press()
            await _wait_for_selector(screen, pilot, "#opt-generic-keep_original_file")
            await pilot.pause()
            latest_pending_control = screen.query_one(
                "#opt-generic-keep_original_file", Checkbox
            )

            release["local"].set()
            await _wait_for_thread_signal(
                entered["server"], pilot, what="latest Server preference save"
            )
            assert backend["value"] == "local"
            await pilot.pause()
            assert (
                screen.query_one("#opt-generic-keep_original_file", Checkbox)
                is latest_pending_control
            )

            release["server"].set()
            for _ in range(200):
                controls = screen.query("#opt-generic-keep_original_file")
                if (
                    backend["value"] == "server"
                    and screen._library_ingest_backend_target is None
                    and len(controls) == 1
                    and controls.first() is not latest_pending_control
                ):
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError("latest Server preference never persisted")
            await pilot.pause()
            assert saves == ["local", "server"]
            assert screen.query_one("#opt-generic-keep_original_file", Checkbox)
    finally:
        release["local"].set()
        release["server"].set()


@pytest.mark.asyncio
@pytest.mark.allow_network
@pytest.mark.parametrize("size", [LIBRARY_TEST_SIZE, (120, 48)])
async def test_library_screen_ingest_layout_contains_metadata_and_start_for_local_prompt(
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
) -> None:
    """Normal and compact Library screens contain the local prompt form chrome."""
    monkeypatch.setattr(
        library_screen_module, "get_cli_setting", lambda *_args, **_kwargs: None
    )
    app = _build_test_app()
    app._resolve_ingest_backend = lambda: "local"
    _seed_conversations(app, ())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    screen._library_ingest_form.path = "/tmp/notes.txt"
    screen._library_ingest_form.analyze = True
    screen._library_ingest_form.expanded_type_groups.add("generic")
    screen._library_ingest_form.type_options = {
        "generic": {"custom_prompt": "Keep headings.\nPreserve citations."}
    }
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#opt-generic-custom_prompt")
        screen._sync_library_ingest_rail_for_width(size[0])
        await pilot.pause()

        canvas = screen.query_one("#library-ingest-canvas", LibraryIngestCanvas)
        metadata = screen.query_one("#library-ingest-metadata-row")
        start = screen.query_one("#library-ingest-start", Button)
        prompt = screen.query_one("#opt-generic-custom_prompt", TextArea)

        assert prompt.text == "Keep headings.\nPreserve citations."
        assert len(screen.query("#opt-generic-keep_original_file")) == 0
        assert metadata.region.x >= canvas.region.x
        assert metadata.region.right <= canvas.region.right
        assert metadata.region.y >= canvas.region.y
        assert metadata.region.height > 0
        assert start.region.x >= canvas.region.x
        assert start.region.right <= canvas.region.right
        assert start.region.y >= canvas.region.y
        assert start.region.bottom <= canvas.region.bottom


def test_external_override_defers_submit_until_preparation_finishes() -> None:
    screen = object.__new__(LibraryScreen)
    submit = MagicMock()
    source_service = MagicMock()
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: source_service,
    )
    screen._library_ingest_form = LibraryIngestFormState(
        path="/tmp/speech.wav",
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/user-owned/parakeet-v2",
                "transcription_precision": "int8",
                "language": "en",
            }
        },
    )
    screen._library_external_submit_generation = 0
    screen._library_external_submit_scope_id = None
    screen._library_external_submit_worker = None
    pending_worker = MagicMock()
    screen._prepare_library_external_submission = MagicMock(return_value=pending_worker)

    LibraryScreen._do_submit_ingest(screen, "/tmp/speech.wav")

    submit.assert_not_called()
    source_service.records.assert_not_called()
    assert screen._library_external_submit_worker is pending_worker
    args = screen._prepare_library_external_submission.call_args.args
    generation, scope_id, key, directory, submit_kwargs = args
    assert generation == 1
    assert scope_id.startswith("library-external-")
    assert "/user-owned/parakeet-v2" not in scope_id
    assert key.value == "v2_int8"
    assert directory == Path("/user-owned/parakeet-v2")
    assert (
        "transcription_external_scope_id"
        not in submit_kwargs["ingest_options"]["audio_video"]
    )


def _vad_only_report(tmp_path: Path):
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_vad_descriptor,
        parakeet_vad_reference,
    )
    from tldw_chatbook.Model_Artifacts import ProvenanceClass
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactPreflightEntry,
        PreflightReport,
    )

    reference = parakeet_vad_reference()
    descriptor = parakeet_vad_descriptor()
    entry = ArtifactPreflightEntry(
        ref=reference,
        source_url=descriptor.source_url,
        repository=descriptor.upstream_repository,
        revision=descriptor.upstream_revision,
        license_id=descriptor.license_id,
        license_url=descriptor.license_url,
        precision=descriptor.precision,
        total_bytes=descriptor.expected_installed_bytes,
        file_count=len(descriptor.files),
        already_installed=False,
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
    )
    return PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(entry,),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=tmp_path / "managed",
        free_bytes=10**12,
        required_bytes=descriptor.expected_installed_bytes,
        sufficient_space=True,
        gating_errors=(),
    )


def test_external_prepare_retains_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = MagicMock()
    service = MagicMock()
    service.prepare_external.side_effect = lambda *_args, **_kwargs: (
        events.append("prepare") or prepared
    )
    service.prepare_config_commit.side_effect = lambda _prepared: events.append(
        "readiness"
    )
    service.retain_prepared.side_effect = lambda *_args: events.append("retain")
    submit = MagicMock(side_effect=lambda **_kwargs: events.append("enqueue"))
    saved_settings: list[dict[str, object]] = []
    fake_app = SimpleNamespace(
        call_from_thread=lambda callback, *args: callback(*args),
    )
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_current_worker",
        lambda: SimpleNamespace(is_cancelled=False),
    )
    screen = object.__new__(LibraryScreen)
    # task-15470: the actual write moved into a `@work(thread=True)`
    # instance method (`_save_library_ingest_options`), which needs a real
    # running app to dispatch through `run_worker` -- `fake_app` above is a
    # bare `SimpleNamespace` stand-in for sequencing, not a mounted app, and
    # has no `_thread_id`. Patching the instance method (rather than the
    # module-level `save_settings_to_cli_config` it wraps) keeps this
    # test's own subject -- that sensitive/internal fields are stripped
    # before persisting -- intact.
    screen._save_library_ingest_options = lambda values: saved_settings.append(values)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
    )
    screen._library_ingest_form = LibraryIngestFormState(path="/tmp/speech.wav")
    screen._library_external_submit_generation = 1
    screen._library_external_submit_scope_id = "library-external-scope"
    screen._library_external_submit_worker = None
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Verifying external model…"
    screen._parakeet_v2_install_progress = MagicMock()
    screen._library_ingest_registry = lambda: None
    screen._invalidate_library_ingest_preflight = MagicMock()
    screen.refresh = MagicMock()
    screen.call_after_refresh = MagicMock()
    submit_kwargs = {
        "source_path": "/tmp/speech.wav",
        "ingest_options": {
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/private/parakeet",
            }
        },
    }

    LibraryScreen._prepare_library_external_submission.__wrapped__(
        screen,
        1,
        "library-external-scope",
        MagicMock(),
        Path("/private/parakeet"),
        submit_kwargs,
    )

    assert events == ["prepare", "readiness", "retain", "enqueue"]
    assert (
        submit.call_args.kwargs["ingest_options"]["audio_video"][
            "transcription_external_scope_id"
        ]
        == "library-external-scope"
    )
    assert "/private/parakeet" not in str(saved_settings)
    assert "transcription_external_scope_id" not in str(saved_settings)
    assert screen._library_external_submit_busy is False
    assert screen._library_external_submit_status == ""
    assert screen._parakeet_v2_install_progress is None


def test_external_vad_plan_is_exact_and_cancel_releases_without_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
    )
    screen._library_ingest_form = LibraryIngestFormState(path="/tmp/speech.wav")
    prior = deepcopy(screen._library_ingest_form)
    screen._library_external_submit_generation = 4
    screen._library_external_submit_scope_id = "library-external-vad"
    screen._library_external_submit_worker = None
    report = _vad_only_report(tmp_path)

    LibraryScreen._apply_library_external_preparation(
        screen,
        4,
        "library-external-vad",
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        report,
        None,
    )

    submit.assert_not_called()
    modal, callback = fake_app.push_screen.call_args.args
    assert modal.report.root == report.root
    assert {entry.ref for entry in modal.report.entries} == {report.root}
    callback(False)
    service.release_scope.assert_called_once_with("library-external-vad")
    assert screen._library_ingest_form == prior
    submit.assert_not_called()


def test_external_vad_plan_rejects_any_non_vad_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dataclasses import replace
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL

    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    service = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=MagicMock(),
        _ensure_parakeet_source_service=lambda: service,
        notify=MagicMock(),
    )
    screen._library_external_submit_generation = 2
    screen._library_external_submit_scope_id = "library-external-changed-plan"
    screen._library_external_submit_worker = None
    report = _vad_only_report(tmp_path)
    changed_entry = replace(
        report.entries[0],
        ref=parakeet_reference(PARAKEET_V2_MODEL, "int8"),
    )

    LibraryScreen._apply_library_external_preparation(
        screen,
        2,
        "library-external-changed-plan",
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        replace(report, entries=report.entries + (changed_entry,)),
        None,
    )

    fake_app.push_screen.assert_not_called()
    service.release_scope.assert_called_once_with("library-external-changed-plan")


def test_stale_external_result_releases_scope_without_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
    )
    screen._library_external_submit_generation = 8
    screen._library_external_submit_scope_id = "library-external-current"

    LibraryScreen._apply_library_external_preparation(
        screen,
        7,
        "library-external-stale",
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        None,
        None,
    )

    service.release_scope.assert_called_once_with("library-external-stale")
    submit.assert_not_called()
    fake_app.push_screen.assert_not_called()


def test_external_validation_failure_releases_and_preserves_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        notify=MagicMock(),
    )
    screen._library_ingest_form = LibraryIngestFormState(path="/tmp/speech.wav")
    prior = deepcopy(screen._library_ingest_form)
    screen._library_external_submit_generation = 3
    screen._library_external_submit_scope_id = "library-external-failed"
    screen._library_external_submit_worker = MagicMock()
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Verifying external model…"

    LibraryScreen._apply_library_external_preparation(
        screen,
        3,
        "library-external-failed",
        None,
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        None,
        "validation_failed",
    )

    service.release_scope.assert_called_once_with("library-external-failed")
    assert screen._library_ingest_form == prior
    assert screen._library_external_submit_busy is False
    assert screen._library_external_submit_status.startswith(
        "Directory verification failed."
    )
    submit.assert_not_called()


def test_external_submit_exception_releases_before_any_registry_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    service = MagicMock()
    registry = MagicMock()
    registry.jobs.return_value = ()
    submit = MagicMock(side_effect=RuntimeError("submit failed"))
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        notify=MagicMock(),
    )
    screen._library_ingest_form = LibraryIngestFormState(path="/tmp/speech.wav")
    prior = deepcopy(screen._library_ingest_form)
    screen._library_ingest_registry = lambda: registry
    screen._library_external_submit_generation = 5
    screen._library_external_submit_scope_id = "library-external-submit-error"
    screen._library_external_submit_worker = MagicMock()
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Queueing import…"

    LibraryScreen._enqueue_library_ingest_snapshot(
        screen,
        {
            "source_path": "/tmp/speech.wav",
            "ingest_options": {"audio_video": {}},
        },
        generation=5,
        scope_id="library-external-submit-error",
    )

    service.release_scope.assert_called_once_with("library-external-submit-error")
    assert registry.jobs.call_count == 2
    assert screen._library_ingest_form == prior
    assert screen._library_external_submit_busy is False
    assert screen._library_external_submit_status.startswith("Queueing failed.")


def test_external_override_is_not_prepared_for_server_backend() -> None:
    submit = MagicMock()
    service = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        _resolve_ingest_backend=lambda: "server",
    )
    screen._library_ingest_form = LibraryIngestFormState(
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/private/external-parakeet",
                "transcription_precision": "int8",
            }
        }
    )
    screen._library_external_submit_generation = 0
    screen._library_external_submit_scope_id = None
    screen._library_external_submit_worker = None
    screen._prepare_library_external_submission = MagicMock()
    screen._enqueue_library_ingest_snapshot = MagicMock()

    LibraryScreen._do_submit_ingest(screen, "/tmp/speech.wav")

    screen._prepare_library_external_submission.assert_not_called()
    screen._enqueue_library_ingest_snapshot.assert_called_once()
    service.prepare_external.assert_not_called()


def test_backend_switch_during_external_hash_cancels_and_fences_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {"value": "local"}
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        _resolve_ingest_backend=lambda: backend["value"],
    )
    screen._library_ingest_form = LibraryIngestFormState(
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/private/external-parakeet",
                "transcription_precision": "int8",
            }
        }
    )
    screen._library_external_submit_generation = 0
    screen._library_external_submit_scope_id = None
    screen._library_external_submit_worker = None
    screen._library_ingest_start_consent = None
    worker = MagicMock(is_finished=False)
    screen._prepare_library_external_submission = MagicMock(return_value=worker)
    screen.refresh = MagicMock()

    def save_backend(target: str, _generation: int) -> None:
        backend["value"] = target

    # task-15470: the actual persistence call moved into a
    # `@work(thread=True)` instance method (`_save_library_ingest_backend`),
    # which needs a running app to dispatch through `run_worker` -- this
    # bare, unmounted screen has none. Patching the instance method itself
    # (rather than the module-level `save_setting_to_cli_config`) keeps this
    # test's actual subject -- cancellation/fencing of the external hash
    # worker -- decoupled from how the backend choice eventually reaches
    # disk, which has its own coverage.
    screen._save_library_ingest_backend = save_backend
    LibraryScreen._do_submit_ingest(screen, "/tmp/speech.wav")
    generation, scope_id, *_rest = (
        screen._prepare_library_external_submission.call_args.args
    )

    LibraryScreen.handle_library_ingest_backend_switch(
        screen, SimpleNamespace(stop=MagicMock())
    )
    worker.cancel.assert_called_once_with()
    LibraryScreen._apply_library_external_preparation(
        screen,
        generation,
        scope_id,
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        None,
        None,
    )

    assert backend["value"] == "server"
    assert screen._library_external_submit_scope_id is None
    assert screen._library_external_submit_generation > generation
    submit.assert_not_called()
    service.release_scope.assert_any_call(scope_id)


def test_option_reset_during_external_hash_preserves_reset_and_fences_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        _resolve_ingest_backend=lambda: "local",
    )
    screen._library_ingest_form = LibraryIngestFormState(
        path="/tmp/speech.wav",
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/private/external-parakeet",
                "transcription_precision": "int8",
            }
        },
    )
    screen._library_external_submit_generation = 0
    screen._library_external_submit_scope_id = None
    screen._library_external_submit_worker = None
    worker = MagicMock(is_finished=False)
    screen._prepare_library_external_submission = MagicMock(return_value=worker)
    screen._refresh_library_ingest_canvas_preserving_context = MagicMock()
    # task-15470: the actual persistence call moved into a
    # `@work(thread=True)` instance method (`_save_library_ingest_options`),
    # which needs a running app to dispatch through `run_worker` -- this
    # bare, unmounted screen has none. Patching the instance method itself
    # keeps this test's actual subject -- cancellation/fencing of the
    # external hash worker -- decoupled from how reset options eventually
    # reach disk, which has its own coverage.
    screen._save_library_ingest_options = MagicMock()

    LibraryScreen._do_submit_ingest(screen, "/tmp/speech.wav")
    generation, scope_id, *_rest = (
        screen._prepare_library_external_submission.call_args.args
    )
    event = SimpleNamespace(
        stop=MagicMock(),
        button=SimpleNamespace(id="opt-audio_video-reset"),
    )
    LibraryScreen.handle_library_ingest_option_reset(screen, event)
    worker.cancel.assert_called_once_with()
    LibraryScreen._apply_library_external_preparation(
        screen,
        generation,
        scope_id,
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        None,
        None,
    )

    assert screen._library_ingest_form.path == "/tmp/speech.wav"
    assert screen._library_ingest_form.type_options["audio_video"] == {}
    submit.assert_not_called()
    service.release_scope.assert_any_call(scope_id)


@pytest.mark.asyncio
async def test_external_vad_worker_cancellation_reaches_underlying_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def gated_provision(*_args: object, **_kwargs: object) -> None:
        entered.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.run_parakeet_vad_provision",
        gated_provision,
    )
    provision = LibraryScreen._provision_library_external_vad.__wrapped__
    assert inspect.iscoroutinefunction(provision)
    service = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
    )
    screen.post_message = MagicMock()
    screen._apply_library_external_preparation = MagicMock()
    task = asyncio.create_task(
        provision(
            screen,
            1,
            "library-external-vad",
            MagicMock(),
            {"source_path": "/tmp/speech.wav", "ingest_options": {}},
            _vad_only_report(tmp_path),
        )
    )
    await wait_for_background_signal(
        entered,
        task,
        what="the external VAD provision entering the underlying install",
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stopped.is_set()
    screen._apply_library_external_preparation.assert_not_called()
    submit.assert_not_called()


def test_vad_install_failure_has_exact_zero_job_copy_and_recovery() -> None:
    service = MagicMock()
    notify = MagicMock()
    submit = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=submit,
        _ensure_parakeet_source_service=lambda: service,
        notify=notify,
    )
    screen._library_ingest_form = LibraryIngestFormState(path="/tmp/speech.wav")
    screen._library_external_submit_generation = 2
    screen._library_external_submit_scope_id = "library-external-vad-failed"
    screen._library_external_submit_worker = MagicMock()
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Installing Silero VAD dependency…"
    screen._parakeet_v2_install_progress = MagicMock()
    screen.refresh = MagicMock()

    LibraryScreen._apply_library_external_preparation(
        screen,
        2,
        "library-external-vad-failed",
        MagicMock(),
        {"source_path": "/tmp/speech.wav", "ingest_options": {}},
        None,
        "vad_failed",
    )

    notify.assert_called_once_with(
        "Silero VAD could not be installed; no import was queued.",
        severity="error",
    )
    assert screen._library_external_submit_busy is False
    assert "Retry" in screen._library_external_submit_status
    assert "Faster Whisper" in screen._library_external_submit_status
    assert "managed model" in screen._library_external_submit_status
    assert screen._parakeet_v2_install_progress is None
    submit.assert_not_called()


def test_external_invalidation_clears_busy_status_and_shared_vad_progress() -> None:
    service = MagicMock()
    progress_widget = SimpleNamespace(display=True)
    worker = MagicMock(is_finished=False)
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        _ensure_parakeet_source_service=lambda: service,
    )
    screen._library_external_submit_generation = 3
    screen._library_external_submit_scope_id = "library-external-stale-progress"
    screen._library_external_submit_worker = worker
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Installing Silero VAD dependency…"
    screen._parakeet_v2_install_progress = MagicMock()
    screen._is_mounted = True
    screen.query_one = MagicMock(return_value=progress_widget)
    screen.refresh = MagicMock()

    LibraryScreen._invalidate_library_external_submission(screen)

    worker.cancel.assert_called_once_with()
    service.release_scope.assert_called_once_with("library-external-stale-progress")
    assert screen._library_external_submit_busy is False
    assert screen._library_external_submit_status == ""
    assert screen._parakeet_v2_install_progress is None
    assert progress_widget.display is False
    screen.refresh.assert_called_once_with(recompose=True)


def test_physical_external_cancel_releases_scope_and_preserves_form() -> None:
    service = MagicMock()
    worker = MagicMock(is_finished=False)
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(
        _ensure_parakeet_source_service=lambda: service,
    )
    screen._library_ingest_form = LibraryIngestFormState(
        path="/tmp/speech.wav",
        type_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "/private/external-parakeet",
            }
        },
    )
    prior = deepcopy(screen._library_ingest_form)
    screen._library_external_submit_generation = 1
    screen._library_external_submit_scope_id = "library-external-cancel"
    screen._library_external_submit_worker = worker
    screen._library_external_submit_busy = True
    screen._library_external_submit_status = "Verifying external model…"
    event = SimpleNamespace(stop=MagicMock())

    LibraryScreen.handle_library_external_preparation_cancel(screen, event)

    event.stop.assert_called_once_with()
    worker.cancel.assert_called_once_with()
    service.release_scope.assert_called_once_with("library-external-cancel")
    assert screen._library_ingest_form == prior
    assert screen._library_external_submit_busy is False
    assert screen._library_external_submit_status == (
        "External preparation cancelled; no import was queued."
    )


def test_external_vad_progress_is_generation_fenced_and_labeled() -> None:
    event = MagicMock()
    label = MagicMock()
    progress = MagicMock()
    screen = object.__new__(LibraryScreen)
    screen.app_instance = SimpleNamespace(_resolve_ingest_backend=lambda: "local")
    screen._library_external_submit_generation = 7
    screen._library_external_submit_scope_id = "library-external-progress"
    screen._library_external_submit_backend = "local"
    screen.query_one = MagicMock(side_effect=[label, progress])

    LibraryScreen._apply_library_external_vad_progress(
        screen,
        7,
        "library-external-progress",
        event,
    )

    assert screen._parakeet_v2_install_progress is event
    assert screen._library_model_install_progress_label == "Silero VAD dependency"
    label.update.assert_called_once_with("Silero VAD dependency")
    assert label.display is True
    assert progress.display is True
    progress.update_progress.assert_called_once_with(event)

    screen.query_one.reset_mock()
    LibraryScreen._apply_library_external_vad_progress(
        screen,
        6,
        "library-external-progress",
        MagicMock(),
    )
    screen.query_one.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_faster_whisper_enables_only_whisper_model_control():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "faster-whisper"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_provider", Select
                ).value
                == "faster-whisper"
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_model", Select
                ).disabled
                is False
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_model_dir", Input
                ).disabled
                is True
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-install-parakeet-v2", Button
                ).disabled
                is True
            )


@pytest.mark.asyncio
async def test_chunk_size_enabled_when_chunk_checked():
    """Chunk size and overlap become editable once Chunk is on.

    They were gated through the installed-feature lookup on the name
    "chunk", which is a sibling field rather than a package, so they were
    disabled no matter what the user did (task-676).
    """
    form = _default_form()
    form.chunk = True
    form.type_options = {"generic": {"chunk": True}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#opt-generic-chunk_size", Input).disabled is False
        assert (
            pilot.app.query_one("#opt-generic-chunk_overlap", Input).disabled is False
        )


@pytest.mark.asyncio
async def test_path_error_offers_a_way_to_pick_a_file_not_retry():
    """A path that cannot be found offers correction, not Retry.

    Re-running the same analysis against the same bad path fails identically,
    so Retry was the wrong verb for the most common pre-flight error a
    first-time user hits (task-666).
    """
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found: /tmp/nope.txt"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=True,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#ingest-preflight-retry")) == 0
        assert pilot.app.query_one("#ingest-preflight-choose", Button)


@pytest.mark.asyncio
async def test_retryable_error_still_offers_retry():
    """A URL that failed to respond is worth another attempt."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["URL unreachable: timed out"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=False,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#ingest-preflight-retry", Button)
        assert len(pilot.app.query("#ingest-preflight-choose")) == 0


@pytest.mark.asyncio
async def test_option_panel_title_reads_as_plain_language():
    """The collapsed panel title describes settings, not internal field names."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(pilot.app.query_one("#type-group-pdf", Collapsible).title)

    assert "pdf_engine=" not in title
    assert "ocr=False" not in title
    assert title == "PDF documents"
    # (task-14825 #7) "Enable OCR" is gated by the chosen PDF engine, so the
    # control reads "— needs the docext engine". Advertising its value in
    # the title told the user they had a setting they cannot change, so it
    # is dropped from the receipt.
    assert "Enable OCR: off" not in title, title
    # (xhigh review round, G4) ...but a closed WITHIN-FORM gate is the form
    # working as designed, not a broken panel: with every package installed
    # this title used to lead "3 options unavailable — needs Enable OCR on".
    # The packaging-gated case is asserted in
    # ``test_a_packaging_gate_still_leads_the_panel_receipt``.
    assert "unavailable" not in title, title


def test_warning_line_does_not_repeat_itself():
    """Warning copy names the gap once, then how to close it."""
    from tldw_chatbook.Library.library_ingest_state import build_warning_lines

    lines = build_warning_lines(
        [
            {
                "feature": "pdf_processing",
                "label": "PDF processing",
                "hint": "PDF ingestion",
                "command": 'pip install -e ".[pdf]"',
            },
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "Audio processing",
                "command": 'pip install -e ".[audio]"',
            },
        ]
    )

    assert lines[0] == (
        "PDF processing isn't installed — needed for PDF ingestion. "
        'Install it with: pip install -e ".[pdf]"'
    )
    # The label and the capability are the same words here, so say it once.
    assert lines[1] == (
        'Audio processing isn\'t installed. Install it with: pip install -e ".[audio]"'
    )


@pytest.mark.asyncio
async def test_first_visit_explains_what_can_be_imported():
    """An untouched form orients the user instead of showing blank space."""
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        first = str(pilot.app.query_one("#library-ingest-intro-0", Static).renderable)
        second = str(pilot.app.query_one("#library-ingest-intro-1", Static).renderable)

    assert "folder" in first and "URL" in first
    assert "PDF documents" in first and "e-books" in first
    assert "searchable" in second


@pytest.mark.asyncio
async def test_intro_gives_way_to_the_real_summary():
    """Orientation never competes with an actual pre-flight result."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        # (task-2042) Intro Statics stay mounted (display-managed) so their
        # appearance/disappearance never changes the canvas structure; the
        # user-visible contract is that they are not DISPLAYED.
        intro = pilot.app.query_one("#library-ingest-intro-0")
        assert intro.display is False


@pytest.mark.asyncio
async def test_clear_button_appears_only_with_a_path():
    """The path field can be emptied in one press once it has content."""
    empty = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(empty)
    async with app.run_test() as pilot:
        # (task-2042) Always mounted, display-managed -- the visible
        # contract is unchanged (hidden without a path, shown with one),
        # but the widget STRUCTURE no longer flips while typing.
        clear = pilot.app.query_one("#library-ingest-clear-path", Button)
        assert clear.display is False

    filled = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(filled)
    async with app.run_test() as pilot:
        clear = pilot.app.query_one("#library-ingest-clear-path", Button)
        assert clear.display is True


@pytest.mark.asyncio
async def test_backend_switch_appears_only_with_a_server_configured():
    """A dead toggle is worse than none, so it is offered only when real."""
    local_only = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(local_only)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-backend-switch")) == 0
        assert len(pilot.app.query("#library-ingest-server-line")) == 0

    with_server = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server",
        server_ingest_available=True
    )
    app = _CanvasHost(with_server)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-ingest-backend-switch", Button)
        assert "server" in str(button.label).lower()
        line = pilot.app.query_one("#library-ingest-server-line", Static)
        assert "this machine" in str(line.renderable).lower()


@pytest.mark.asyncio
async def test_backend_switch_offers_the_way_back_when_targeting_the_server():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend="server",
        runtime_source="server",
        server_ingest_available=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-ingest-backend-switch", Button)
        assert "this machine" in str(button.label).lower()
        line = pilot.app.query_one("#library-ingest-server-line", Static)
        assert "server" in str(line.renderable).lower()


@pytest.mark.asyncio
async def test_backend_state_is_rendered_before_the_switch():
    """"Imports run on X" must precede the button that changes it.

    With the button first, the pane read as a contradiction top-to-bottom:
    "Import on the server" directly above "Imports run on this machine."
    """
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server",
        server_ingest_available=True
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        children = list(pilot.app.query_one(LibraryIngestCanvas).children)
        ids = [c.id for c in children if c.id]
        assert "library-ingest-server-line" in ids
        assert "library-ingest-backend-switch" in ids
        assert ids.index("library-ingest-server-line") < ids.index(
            "library-ingest-backend-switch"
        ), f"switch rendered before the state line: {ids[:6]}"


@pytest.mark.asyncio
async def test_unsupported_files_summary_pluralizes_correctly():
    """(task-2015) Plural counts must not read "recorded as a failures"."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/a.xyz", "/tmp/b.xyz"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#ingest-unsupported-summary", Static)
        # (task-2100) Gate-blocked (nothing importable): names only.
        assert str(summary.renderable) == (
            "Unsupported: a.xyz, b.xyz."
            " Supported: PDF documents, Word/Office documents, audio/video"
            " files, e-books, images, plain text files, web pages (by URL)."
        )


# --- task-2016: P3 polish ----------------------------------------------------


@pytest.mark.asyncio
async def test_done_row_progress_line_has_no_state_prefix():
    """(task-2016) The row line already says "✓ done"; prefixing the progress
    line with "done" again read as stuttering. Terminal states render the
    message alone; active states keep the prefix (previous test)."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.DONE,
        media_id=1,
        progress={"message": "Imported report.txt"},
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        assert str(progress.renderable) == "Imported report.txt"


@pytest.mark.asyncio
async def test_expand_collapse_all_hidden_for_single_panel():
    """(task-2016) Bulk expand/collapse over exactly one panel is noise."""
    single = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(single)
    async with app.run_test() as pilot:
        assert not list(pilot.app.query("#ingest-expand-all"))
        assert not list(pilot.app.query("#ingest-collapse-all"))


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_generic_scope_line_reworded_when_no_generic_files_staged():
    """(task-2016) The always-present generic panel claimed scope even when the import
    contained zero such files."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    # Expand the generic panel so its scope line mounts.
    state.form.expanded_type_groups.add("generic")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        scopes = [
            str(w.renderable)
            for w in pilot.app.query(".type-group-scope").results(Static)
        ]
        generic_scope = [s for s in scopes if "imported item" in s.lower()]
        assert generic_scope, f"generic scope line missing: {scopes}"
        assert "if this import contains any" in generic_scope[0]
        assert "in this import." not in generic_scope[0]


# --- task-2043: P2 batch ----------------------------------------------------


@pytest.mark.asyncio
async def test_expanded_details_render_inline_and_flip_button_label():
    """(task-2043) Expanded rows render their detail lines inline (the old
    surface was a ~4s uncopyable toast) and the button reads Hide details."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="Failed to process pdf file: PDF Extraction Error.",
        error_detail={
            "category": "parse_error",
            "message": "Failed to process pdf file: PDF Extraction Error.",
        },
    )
    state = build_library_ingest_state(
        (job,),
        form=_default_form(),
        expanded_details={"ingest-job-1"},
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        detail = pilot.app.query_one("#library-ingest-detail-ingest-job-1-0", Static)
        # (task-14821) The first detail line is the user-facing REASON, not
        # the raw internal category token ("Category: parse error").
        assert str(detail.renderable).startswith("Reason: ")
        button = pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)
        assert str(button.label) == "Hide details"


@pytest.mark.asyncio
async def test_select_fields_carry_visible_labels():
    """(task-2043) Selects missed task-2012's labeling: 'pymupdf4llm' bare
    carries no meaning. Every select gets a label Static."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    state.form.expanded_type_groups.add("pdf")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        labels = [
            str(w.renderable)
            for w in pilot.app.query(".type-group-field-label").results(Static)
        ]
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities

        select_labels = [
            f.label for f in get_capabilities("pdf").fields if f.type == "select"
        ]
        assert select_labels, "pdf group unexpectedly has no selects"
        for label in select_labels:
            # (task-3304) A schema-disabled select's label carries a
            # " — <reason>" annotation, and whether one applies here
            # depends on which optional packages THIS environment has --
            # so accept the label with or without the suffix, pinned to
            # the exact separator so a missing label still fails.
            assert any(
                text == label or text.startswith(f"{label} — ")
                for text in labels
            ), f"select {label!r} has no visible label; rendered: {labels!r}"


@pytest.mark.asyncio
async def test_checkbox_glyph_tracks_state_without_color():
    """(task-2043) Stock ToggleButton renders 'X' for both states; the
    subclass carries on/off in the glyph itself."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        StateGlyphCheckbox,
    )

    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    state.form.expanded_type_groups.add("generic")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        analyze = pilot.app.query_one("#opt-generic-analyze", StateGlyphCheckbox)
        chunk = pilot.app.query_one("#opt-generic-chunk", StateGlyphCheckbox)
        assert chunk.value is True and chunk.BUTTON_INNER == "✓"
        assert analyze.value is False and analyze.BUTTON_INNER == " "
        analyze.value = True
        await pilot.pause()
        assert analyze.BUTTON_INNER == "✓"


@pytest.mark.asyncio
async def test_duplicate_forecast_line_renders_in_summary():
    """(task-2043) The pre-flight duplicate forecast renders as a quiet
    line in the summary block."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
            already_in_library=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        line = pilot.app.query_one("#ingest-duplicate-summary", Static)
        assert "already be in your Library" in str(line.renderable)


@pytest.mark.asyncio
async def test_severity_colour_supplements_glyphs_and_invalid_field_marked() -> None:
    """Severity colour supplements glyphs; invalid fields stay marked.

    (task-2230 a11y) Failed/skipped rows carry a severity class ON TOP of
    the glyph+word they already have (never colour-only), and an invalid
    option field stays marked without focus -- the gate line's
    "highlighted" pointed at a border that only existed while focused.
    """
    failed = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="Failed to process pdf file.",
    )
    skipped = LibraryIngestJob(
        job_id="ingest-job-2",
        source_path="/tmp/photo.jpg",
        state=IngestJobState.SKIPPED,
        error="Unsupported file type: .jpg.",
    )
    done = LibraryIngestJob(
        job_id="ingest-job-3",
        source_path="/tmp/report.txt",
        state=IngestJobState.DONE,
    )
    form = LibraryIngestFormState(path="/tmp/report.txt")
    form.type_options["generic"] = {"chunk_size": "abc"}
    state = build_library_ingest_state((failed, skipped, done), form=form)

    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        rows = list(pilot.app.query(".library-ingest-row"))
        classes = [set(row.classes) for row in rows]
        assert any("library-ingest-row-failed" in c for c in classes)
        assert any("library-ingest-row-skipped" in c for c in classes)
        # The done row carries neither severity class.
        assert sum(
            1
            for c in classes
            if "library-ingest-row-failed" in c
            or "library-ingest-row-skipped" in c
        ) == 2

        # Glyph + word survive alongside the colour (monochrome contract).
        by_id = {row.job_id: row for row in state.queue_rows}
        assert by_id["ingest-job-1"].line.startswith("✗ failed")
        assert by_id["ingest-job-2"].line.startswith("○ skipped")

        invalid = pilot.app.query_one("#opt-generic-chunk_size", Input)
        assert invalid.has_class("-ingest-option-invalid"), (
            "an invalid field must stay marked without focus"
        )
        assert pilot.app.focused is not invalid


@pytest.mark.asyncio
async def test_analysis_hint_renders_when_state_carries_one():
    """task-3301: the Analyze-readiness hint renders beside the Start gate."""
    form = LibraryIngestFormState(path="/tmp/test", analyze=True)
    state = build_library_ingest_state(
        (),
        form=form,
        analysis_unready_hint=(
            "Analyze after import is on, but OpenAI is not ready: Missing "
            "API key. Imports will run without analysis."
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        hint = pilot.app.query_one("#library-ingest-analysis-hint", Static)
        assert hint.display is True
        assert "OpenAI" in str(hint.renderable)
        assert "without analysis" in str(hint.renderable)


@pytest.mark.asyncio
async def test_analysis_hint_hidden_when_state_has_none():
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        hint = pilot.app.query_one("#library-ingest-analysis-hint", Static)
        assert hint.display is False


# ---------------------------------------------------------------------------
# task-3303: document panel, PDF OCR gating, AV translate gating, web honesty
# ---------------------------------------------------------------------------


def _preflight_for(groups: dict[str, list[str]]) -> PreflightResult:
    return PreflightResult(
        type_groups=groups,
        warnings=[],
        errors=[],
        total_size=0,
        truncated=False,
        total_files=sum(len(v) for v in groups.values()),
    )


@pytest.mark.asyncio
async def test_document_panel_renders_with_processing_and_ocr_controls():
    """(task-3303 AC1) A .docx selection gets its own options panel."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=_preflight_for({"document": ["/tmp/report.docx"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            panel = pilot.app.query_one("#type-group-document", Collapsible)
            assert "Word/Office documents" in str(panel.title)
            method = pilot.app.query_one(
                "#opt-document-processing_method", Select
            )
            assert method.value == "auto"
            ocr = pilot.app.query_one("#opt-document-ocr", Checkbox)
            # Under the default "auto" method OCR is offerable (docling is
            # "installed" in this test), and the label carries its scope.
            assert ocr.disabled is False
            assert "docling" in str(ocr.label)
            language = pilot.app.query_one("#opt-document-ocr_language", Input)
            # OCR is off, so the language field rides the gate.
            assert language.disabled is True


@pytest.mark.asyncio
async def test_pdf_ocr_checkbox_inert_under_pymupdf_engine_with_reason():
    """(task-3303 AC2) OCR under a non-OCR engine is inert, with the reason."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=_preflight_for({"pdf": ["/tmp/a.pdf"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            ocr = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            # Default engine is pymupdf4llm, which cannot OCR.
            assert ocr.disabled is True
            assert "docling or docext engines only" in str(ocr.label)


@pytest.mark.asyncio
async def test_pdf_ocr_checkbox_enabled_under_docling_engine():
    form = _default_form()
    form.type_options = {"pdf": {"pdf_engine": "docling"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"pdf": ["/tmp/a.pdf"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            ocr = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            assert ocr.disabled is False
            backend = pilot.app.query_one("#opt-pdf-ocr_backend", Select)
            # Backend selection applies to the docext engine only.
            assert backend.disabled is True


@pytest.mark.asyncio
async def test_translate_checkbox_inert_under_parakeet_provider():
    """(task-3303 AC4) Only faster-whisper translates; parakeet renders it inert."""
    form = _default_form()
    form.type_options = {
        "audio_video": {"transcription_provider": "parakeet-onnx"}
    }
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"audio_video": ["/tmp/talk.mp3"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            translate = pilot.app.query_one(
                "#opt-audio_video-translate_to_english", Checkbox
            )
            assert translate.disabled is True
            assert "faster-whisper" in str(translate.label)
            vad = pilot.app.query_one("#opt-audio_video-vad_filter", Checkbox)
            assert vad.disabled is False


@pytest.mark.asyncio
async def test_web_local_multi_page_note_visible_when_sitemap_selected():
    """(task-3303 AC5) A local sitemap selection says it fetches one page."""
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "sitemap"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is True
        assert "one page" in str(note.renderable)
        assert "server" in str(note.renderable)


@pytest.mark.asyncio
async def test_web_note_hidden_for_single_page_selection():
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "individual"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is False


@pytest.mark.asyncio
async def test_web_note_hidden_when_targeting_the_server():
    """Server behavior is unchanged: the clip path honors multi-page methods."""
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "sitemap"}}
    state = build_library_ingest_state(
        (),
        form=form,
        runtime_source="server",
        server_ingest_available=True,
        ingest_backend="server",
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is False


# --- task-3305: copy & labels batch -----------------------------------------


@pytest.mark.asyncio
async def test_every_select_renders_human_labels_never_raw_tokens():
    """(task-3305, MI-09) Every option select shows human display copy while
    persisting the internal value: no rendered option prompt may be the raw
    schema token (``pymupdf4llm``, ``url_level``, ``recursive_scraping``…)."""
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "document": ["/tmp/b.docx"],
                "audio_video": ["/tmp/c.mp3"],
                "ebook": ["/tmp/d.epub"],
                "web": ["https://example.com/article"],
            },
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=5,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            selects = list(pilot.app.query(Select))
            assert selects, "no selects rendered -- panel sweep is broken"
            seen_groups: set[str] = set()
            for select in selects:
                widget_id = select.id or ""
                assert widget_id.startswith("opt-")
                _, group, name = widget_id.split("-", 2)
                seen_groups.add(group)
                cap = get_capabilities(group)
                field = next(
                    (f for f in cap.fields if f.name == name),
                    None,
                )
                if field is None:
                    # (task 11) The chunking-template picker is an opt-*
                    # select with NO schema field: its values ARE the
                    # user-facing template names (not internal tokens), and
                    # its default option is the spec §9.3 None label. The
                    # token-label rule below is a schema-select contract.
                    assert name == "chunk_template" and group == "generic"
                    continue
                rendered = [
                    (str(prompt), value)
                    for prompt, value in select._options
                    if value is not Select.BLANK
                ]
                assert {value for _prompt, value in rendered} == set(
                    field.options
                ), f"{group}.{name}: persisted values must stay the tokens"
                for prompt, value in rendered:
                    assert prompt != value, (
                        f"{group}.{name}: option {value!r} renders its raw "
                        "internal token"
                    )
                # The persisted selection is still an internal token.
                assert select.value in field.options
            # Every group with a select was actually swept.
            assert {"pdf", "document", "audio_video", "ebook", "web",
                    "generic"} <= seen_groups


@pytest.mark.asyncio
async def test_recent_import_bracket_filename_renders_clean():
    """(task-3305, MI-14) escape_markup on a ``markup=False`` Static is
    double defense: the escape backslashes render literally. A bracketed
    filename must come out byte-identical."""
    done = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/[draft] notes.txt",
        state=IngestJobState.DONE,
        media_id=1,
    )
    state = build_library_ingest_state((done,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        items = [
            str(item.renderable)
            for item in pilot.app.query(".library-ingest-recent-item")
        ]
        assert any("[draft] notes.txt" in text for text in items), items
        assert not any("\\[" in text for text in items), items
        paths = [
            str(item.renderable)
            for item in pilot.app.query(".library-ingest-recent-path")
        ]
        assert any("/tmp/[draft] notes.txt" == text for text in paths), paths
        assert not any("\\[" in text for text in paths), paths


@pytest.mark.asyncio
async def test_collapsed_title_caps_pairs_and_skips_empty_values():
    """(task-3305, MI-16) The audio panel title was a ~140-char run-on with
    a dangling empty value ("Local Parakeet model folder: ,"). The title
    caps at the most salient pairs and never renders an empty value."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(
                pilot.app.query_one("#type-group-audio_video", Collapsible).title
            )
    assert "Local Parakeet model folder" not in title, (
        "empty value must be skipped, not rendered dangling"
    )
    assert ": ," not in title
    assert title == "Audio & video"


@pytest.mark.asyncio
async def test_metadata_fields_keep_persistent_labels_after_values_are_entered():
    """TASK-15702: populated metadata remains identifiable without placeholders."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        expected = {
            "title": "Title (optional)",
            "author": "Author (optional)",
            "keywords": "Keywords (optional)",
        }
        for name, copy in expected.items():
            field = pilot.app.query_one(f"#library-ingest-{name}", Input)
            field.value = f"filled {name}"
            label = pilot.app.query_one(f"#library-ingest-{name}-label", Static)
            assert str(label.renderable) == copy
            assert label.region.y < field.region.y


@pytest.mark.asyncio
async def test_collapsed_title_promotes_changed_values_first():
    """A changed option is what the user cares to see in the receipt; it
    must outrank untouched defaults when the cap bites."""
    form = _default_form()
    form.type_options["audio_video"] = {"diarization": True}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(
                pilot.app.query_one("#type-group-audio_video", Collapsible).title
            )
    _, _, pairs_text = title.partition(" — ")
    assert pairs_text.startswith("Speaker diarization: on"), title


@pytest.mark.asyncio
async def test_parakeet_folder_placeholder_is_an_example_not_the_label():
    """(task-3305, MI-16) The empty model-folder Input showed its own label
    as placeholder -- pure stutter. The placeholder is example content."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            assert model_dir.placeholder == "/path/to/parakeet-model"
            assert model_dir.placeholder != "Local Parakeet model folder"


@pytest.mark.asyncio
async def test_failed_queue_row_bracket_error_renders_clean():
    """(task-3312 #2) The queue-row Static rendered ``escape_markup``ed
    text WITH markup enabled. ``rich.markup.escape`` skips a bracket run
    that never closes as a tag (``[remedy: … [``) while escaping the inner
    closed ones (``[keys]``), and Textual's content markup then leaves the
    FIRST escape's backslash literal -- the live 2026-08-08 receipt showed
    ``\\[web_security]`` verbatim (REPL-pinned: escape+from_markup on this
    exact shape leaks). Verbatim ``markup=False`` rendering: byte-identical
    text, no escape artifacts, nothing swallowed as a tag."""
    error = (
        "note [remedy: check [keys] in config.toml, or set [keys] off]"
    )
    failed = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/[draft] notes.txt",
        state=IngestJobState.FAILED,
        error=error,
    )
    state = build_library_ingest_state((failed,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        row = pilot.app.query_one("#library-ingest-row-0", Static)
        text = row.visual.plain
        assert "[draft] notes.txt" in text, text
        assert "\\[" not in text, text
        # Nothing swallowed as a markup tag: the error text is intact.
        assert error in text, text


# --- live-verify round: gated text fields lost their hint --------------------


def _audio_video_state() -> LibraryIngestCanvasState:
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    return build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )


def _label_before(pilot, widget_id: str) -> str:
    """The ``.type-group-field-label`` Static rendered above ``widget_id``."""
    field = pilot.app.query_one(f"#{widget_id}")
    siblings = list(field.parent.children)
    index = siblings.index(field)
    for candidate in reversed(siblings[:index]):
        if isinstance(candidate, Static) and candidate.has_class(
            "type-group-field-label"
        ):
            return candidate.visual.plain
    raise AssertionError(f"no field label found above #{widget_id}")


@pytest.mark.asyncio
async def test_enabled_text_field_label_carries_its_hint():
    """Baseline: the unit/format hint rides the label line (task-2223)."""
    app = _CanvasHost(_audio_video_state())
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            label = _label_before(pilot, "opt-audio_video-start_time")
            assert "HH:MM:SS or seconds" in label, label
            cookies = _label_before(pilot, "opt-audio_video-cookies_file")
            assert "video URLs only" in cookies, cookies


@pytest.mark.asyncio
async def test_disabled_text_field_label_keeps_its_hint_too():
    """The disabled branch rebuilt the label from ``field.label`` alone, so
    the hint vanished exactly when the field is inert and the user most
    needs to know what it wants -- on a stock install that is every gated
    text field ("video URLs only", "HH:MM:SS or seconds"). The checkbox
    branch never had the bug; the text branch must match it."""
    app = _CanvasHost(_audio_video_state())
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=False,
    ):
        async with app.run_test() as pilot:
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-cookies_file", Input
                ).disabled
                is True
            )
            cookies = _label_before(pilot, "opt-audio_video-cookies_file")
            assert "video URLs only" in cookies, (
                f"the gated field lost its hint: {cookies!r}"
            )
            assert "needs" in cookies, (
                f"the disabled reason must still be there: {cookies!r}"
            )
            start = _label_before(pilot, "opt-audio_video-start_time")
            assert "HH:MM:SS or seconds" in start, start


# --- live-verify round: stacked copy buttons were indistinguishable ----------


@pytest.mark.asyncio
async def test_each_copy_install_button_names_its_own_extra():
    """Six or seven "Copy install command" buttons stacked under the audio
    warnings all RENDERED the same string: the disambiguator was spelled
    ``.[audio]`` and a Button label is parsed as content markup, so
    ``[audio]`` was eaten as a style tag and every button read "Copy
    install command (.)". Each must name its own extra, visibly."""
    warnings = [
        {
            "feature": "audio_processing",
            "label": "Audio processing",
            "hint": "audio transcription",
            "command": 'pip install -e ".[audio]"',
        },
        {
            "feature": "video_processing",
            "label": "Video processing",
            "hint": "video ingestion",
            "command": 'pip install -e ".[video]"',
        },
        {
            "feature": "faster_whisper",
            "label": "faster-whisper",
            "hint": "local transcription",
            "command": 'pip install -e ".[transcription_faster_whisper]"',
        },
    ]
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
            warnings=warnings,
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        rendered = [
            pilot.app.query_one(
                f"#ingest-preflight-copy-command-{index}", Button
            ).label.plain
            for index in range(len(warnings))
        ]
        assert len(set(rendered)) == len(rendered), (
            f"stacked copy buttons render identically: {rendered}"
        )
        assert any("audio" in text for text in rendered), rendered
        assert any("video" in text for text in rendered), rendered
        assert any(
            "transcription_faster_whisper" in text for text in rendered
        ), rendered


# --- task-14822: the tooling-warning wall folds -----------------------------
#
# A 21-file mixed folder rendered 11 warning Statics (CSS double-spaces them
# into ~22 rows) followed by 9 stacked "Copy install command (…)" buttons --
# ~31 rows, the whole 52-row viewport, before the type breakdown, the options
# or Start. The honest reading of that block was "this app is broken", when
# the truth was "3 of your 21 files need optional extras".


def _tooling_warnings(count: int) -> list[dict[str, str]]:
    """``count`` distinct tooling warnings, each with its own extra."""
    return [
        {
            "feature": f"feature_{index}",
            "label": f"Backend {index}",
            "hint": f"capability {index}",
            "command": f'pip install -e ".[extra{index}]"',
        }
        for index in range(count)
    ]


def _warned_state(count: int = 11):
    return build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=_tooling_warnings(count),
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )


@pytest.mark.asyncio
async def test_tooling_warnings_collapse_to_one_summary_line():
    """AC#1: eleven warnings render ONE summary line at canvas level; the
    per-warning detail moves behind a collapsed fold."""
    app = _CanvasHost(_warned_state(11))
    async with app.run_test() as pilot:
        summary_block = pilot.app.query_one(
            "#library-ingest-preflight-summary"
        )
        summary = pilot.app.query_one(
            "#ingest-preflight-tooling-summary", Static
        )
        assert summary.visual.plain.startswith("⚠")
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        assert fold.collapsed is True, "the detail must start folded away"
        # Every warning line still exists -- inside the fold, not stacked
        # above the rest of the form.
        for index in range(11):
            warning = pilot.app.query_one(
                f"#ingest-preflight-warning-{index}", Static
            )
            assert fold in warning.ancestors, (
                f"warning {index} is not inside the fold"
            )
        # No warning Static is a direct child of the summary block.
        assert not [
            child
            for child in summary_block.children
            if (child.id or "").startswith("ingest-preflight-warning-")
        ]


class _StubForecast:
    """The forecast reads the fold makes (task-14820's object).

    (xhigh review round, G1) Carries the doomed/degraded SPLIT as well as
    the affected total, because the fold's verb follows it: a stub with
    only ``consent_affected`` would exercise nothing but the defensive
    branch, which is how "may fail" survived over 21 certain failures.
    """

    def __init__(
        self,
        *,
        doomed: int = 0,
        degraded: int = 0,
        staged_total: int = 0,
    ) -> None:
        self.will_fail_tooling = doomed
        self.at_risk = degraded
        self.consent_affected = doomed + degraded
        if staged_total:
            self.staged_total = staged_total


class _StubWarnedState:
    """Just enough state for the pure summary-line function."""

    def __init__(self, warning_count: int, forecast=None) -> None:
        self.warning_lines = [f"line {i}" for i in range(warning_count)]
        self.forecast = forecast


def test_tooling_summary_line_names_affected_files_from_the_one_forecast():
    """AC#1: the count comes from task-14820's forecast object -- the fold
    never computes a second one (two independently-derived counts is the
    P1 this arc exists to fix)."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    with_total = ingest_tooling_summary_line(
        _StubWarnedState(11, _StubForecast(degraded=3, staged_total=21))
    )
    assert with_total == (
        "⚠ 3 of 21 files need optional tooling — those imports may fail."
    ), with_total

    without_total = ingest_tooling_summary_line(
        _StubWarnedState(11, _StubForecast(degraded=1))
    )
    assert without_total == (
        "⚠ 1 file needs optional tooling — that import may fail."
    ), without_total

    # The same shape when the missing feature is REQUIRED: certain, not
    # possible (xhigh review round, G1).
    doomed = ingest_tooling_summary_line(
        _StubWarnedState(11, _StubForecast(doomed=21, staged_total=21))
    )
    assert doomed == (
        "⚠ 21 of 21 files need tooling that isn't installed — "
        "those imports will fail."
    ), doomed

    # A forecast that puts NO staged file at risk still explains the block.
    none_affected = ingest_tooling_summary_line(
        _StubWarnedState(4, _StubForecast())
    )
    assert none_affected == (
        "⚠ 4 optional components aren't installed — "
        "no staged file needs them."
    ), none_affected


def test_tooling_summary_line_degrades_without_a_forecast():
    """No forecast means no file count may be stated -- the block is
    described by what IS known (how many components are missing)."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    line = ingest_tooling_summary_line(_StubWarnedState(3, None))
    assert line == (
        "⚠ 3 optional components aren't installed — some imports may fail."
    ), line


@pytest.mark.asyncio
async def test_rendered_tooling_summary_is_the_shared_line():
    """The mounted summary renders exactly the shared function's output --
    one source for the copy, whatever the state carries."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    state = _warned_state(11)
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        text = pilot.app.query_one(
            "#ingest-preflight-tooling-summary", Static
        ).visual.plain
    assert text == ingest_tooling_summary_line(state)
    assert text.startswith("⚠")


@pytest.mark.asyncio
async def test_one_combined_install_command_is_offered_outside_the_fold():
    """AC#4: a single command installs the union of missing extras, and it
    is reachable without opening the fold."""
    copied: list[str] = []

    class _ClipboardHost(_CanvasHost):
        def copy_to_clipboard(self, text: str) -> None:
            copied.append(text)

    app = _ClipboardHost(_warned_state(3))
    async with app.run_test() as pilot:
        button = pilot.app.query_one(
            "#ingest-preflight-copy-all-commands", Button
        )
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        assert fold not in button.ancestors, (
            "the combined command must not be hidden behind the fold"
        )
        button.press()
        await pilot.pause()

    assert copied == ['pip install -e ".[extra0,extra1,extra2]"'], copied


@pytest.mark.asyncio
async def test_per_extra_copy_labels_have_one_shape_at_any_count():
    """AC#4: the ``(extra)`` suffix used to VANISH at exactly one command,
    so the same control had two label shapes.

    (xhigh review round, G5) The per-extra family now only exists where it
    disambiguates -- at two or more commands. At exactly one, the combined
    button above the fold IS that command and the pair rendered it twice;
    ``test_a_single_install_command_yields_a_single_copy_control`` pins
    that case.
    """
    for count in (2, 4):
        app = _CanvasHost(_warned_state(count))
        async with app.run_test() as pilot:
            labels = [
                pilot.app.query_one(
                    f"#ingest-preflight-copy-command-{index}", Button
                ).label.plain
                for index in range(count)
            ]
        assert labels == [
            f"Copy install command (extra{index})" for index in range(count)
        ], labels


@pytest.mark.asyncio
async def test_outcome_lines_do_not_share_the_tooling_warning_class():
    """AC#3: unsupported/empty files are OUTCOMES of this import, not
    environment facts -- they must not render at the tooling warnings'
    weight."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/a.txt"],
                "unsupported": ["/tmp/b.xyz"],
            },
            warnings=_tooling_warnings(3),
            errors=[],
            total_size=10,
            truncated=False,
            total_files=2,
            empty_files=["/tmp/c.txt"],
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        unsupported = pilot.app.query_one("#ingest-unsupported-summary", Static)
        empty = pilot.app.query_one("#ingest-empty-summary", Static)
        tooling = pilot.app.query_one(
            "#ingest-preflight-tooling-summary", Static
        )
        for outcome in (unsupported, empty):
            assert outcome.has_class("library-ingest-outcome-line"), (
                f"{outcome.id} carries no outcome weight: {outcome.classes}"
            )
            assert not outcome.has_class("library-ingest-quiet-line"), (
                f"{outcome.id} still shares the quiet/warning weight"
            )
        assert not tooling.has_class("library-ingest-outcome-line")


# --- task-14825 #7 / task-14826 AC#2: what a collapsed title must say ------


@pytest.mark.asyncio
async def test_collapsed_title_omits_values_of_disabled_fields():
    """task-14825 #7: the title read ``Extract text (OCR): on`` while the
    control below it read ``— needs OCR backend installed``. Advertising a
    value the user cannot change is a promise the panel does not keep."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=False,
    ):
        async with app.run_test() as pilot:
            panel = pilot.app.query_one("#type-group-pdf", Collapsible)
            title = str(panel.title)
            disabled_labels = [
                widget
                for widget in panel.query(Select)
                if widget.disabled
            ]
            assert disabled_labels, "precondition: the group is gated"
            # Not one gated field's value may appear in the receipt.
            assert "PDF engine:" not in title, title
            assert "unavailable" in title and "installed" in title, title


@pytest.mark.asyncio
async def test_collapsed_panel_with_an_invalid_value_is_marked_in_its_title():
    """task-14826 AC#2: the gate said "Fix the highlighted options" while
    the highlight (`-ingest-option-invalid`) sat on an Input inside the
    COLLAPSED body -- nothing on screen was marked, and the collapsed title
    cheerfully reported ``Chunk size: 7``."""
    form = _default_form()
    form.type_options = {"generic": {"chunk": True, "chunk_size": "7"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        panel = pilot.app.query_one("#type-group-generic", Collapsible)
        assert panel.collapsed is True, "precondition: the panel is folded"
        title = str(panel.title)
        assert title.count("⚠") == 1, title
        assert "Chunk size needs fixing" in title, title
        # The invalid value itself must not be presented as a setting.
        assert "Chunk size: 7" not in title, title
        # And the marker survives the screen's in-place title update, which
        # assigns `Collapsible.title` and nothing else.
        from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
            build_type_group_title,
        )
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities

        assert build_type_group_title(
            get_capabilities("generic"), form.type_options["generic"]
        ) == title


@pytest.mark.asyncio
async def test_opening_the_tooling_fold_is_not_reported_as_an_option_panel():
    """The fold is a ``Collapsible`` inside the summary; the canvas's
    option-panel handler must ignore it, or opening it would persist a
    bogus expanded type group under the id ``ingest-preflight-tooling-
    detail``."""
    app = _MessageRecordingHost(_warned_state(3))
    async with app.run_test() as pilot:
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        fold.collapsed = False
        await pilot.pause()
        assert fold.collapsed is False
    assert app.panel_toggles == [], (
        f"the fold leaked an option-panel toggle: {app.panel_toggles}"
    )


# --- xhigh review round: the fold must not contradict its own forecast -------
#
# G1, the arc's own headline defect re-created inside the fold it added: the
# summary hard-coded "optional tooling" and "may fail" while
# ``consent_affected`` sums DOOMED (required-missing) and DEGRADED
# (optional-missing) files. Live: 21 PDFs without the pdf extra rendered
# "⚠ 21 of 21 files need optional tooling — those imports may fail." beside a
# commit line reading "0 will import · 21 will fail (need tooling)" and a
# consent line reading "21 files will fail without more tooling".


def _pdf_warning() -> dict[str, str]:
    """The pdf group's REQUIRED feature -- a doomed selection."""
    return {
        "feature": "pdf_processing",
        "label": "PDF processing",
        "hint": "PDF ingestion",
        "command": 'pip install -e ".[pdf]"',
    }


def _pdf_optional_warning() -> dict[str, str]:
    """An OPTIONAL pdf feature -- degraded, not doomed."""
    return {
        "feature": "docling",
        "label": "Docling",
        "hint": "layout-aware PDF extraction",
        "command": 'pip install -e ".[docling]"',
    }


def _ebook_optional_warning() -> dict[str, str]:
    return {
        "feature": "html2text",
        "label": "html2text",
        "hint": "ebook HTML conversion",
        "command": 'pip install -e ".[ebook]"',
    }


def _forecast_state(type_groups, warnings):
    total = sum(len(files) for files in type_groups.values())
    return build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups=type_groups,
            warnings=warnings,
            errors=[],
            total_size=1024,
            truncated=False,
            total_files=total,
        ),
    )


def _doomed_state(file_count: int = 21):
    return _forecast_state(
        {"pdf": [f"/tmp/pdfs/{i}.pdf" for i in range(file_count)]},
        [_pdf_warning()],
    )


def _degraded_state(file_count: int = 3):
    return _forecast_state(
        {"pdf": [f"/tmp/pdfs/{i}.pdf" for i in range(file_count)]},
        [_pdf_optional_warning()],
    )


def _mixed_forecast_state():
    return _forecast_state(
        {
            "pdf": [f"/tmp/mixed/{i}.pdf" for i in range(5)],
            "ebook": [f"/tmp/mixed/{i}.epub" for i in range(3)],
        },
        [_pdf_warning(), _ebook_optional_warning()],
    )


def test_the_fold_line_says_will_fail_for_a_doomed_selection():
    """G1: 21 PDFs with the REQUIRED pdf feature missing are certain
    failures. The fold said "need optional tooling — those imports may
    fail." while the commit line beside it said "21 will fail (need
    tooling)" -- two forecasts for one selection, which is the exact
    defect this arc exists to remove."""
    from tldw_chatbook.Library.library_ingest_state import (
        forecast_consent_line,
        forecast_summary_line,
    )
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    state = _doomed_state(21)
    forecast = state.forecast
    assert forecast is not None
    assert (forecast.will_fail_tooling, forecast.at_risk) == (21, 0), (
        f"precondition: the forecast must call these doomed: {forecast}"
    )

    line = ingest_tooling_summary_line(state)
    assert "will fail" in line, (
        f"the fold still softens a certain failure: {line!r} beside "
        f"{forecast_summary_line(forecast)!r}"
    )
    assert "may fail" not in line, line
    assert "optional" not in line, (
        f"required tooling described as optional: {line!r}"
    )
    assert "21 of 21 files" in line, line
    # And it must not disagree with the two lines derived from the same
    # object at the commit point.
    assert "will fail" in forecast_summary_line(forecast)
    assert "will fail without more tooling" in forecast_consent_line(forecast)


def test_the_fold_line_keeps_may_fail_for_a_degraded_selection():
    """G1's other half: an OPTIONAL feature only degrades the import, and
    the softer wording is correct there."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    state = _degraded_state(3)
    forecast = state.forecast
    assert (forecast.will_fail_tooling, forecast.at_risk) == (0, 3), forecast
    line = ingest_tooling_summary_line(state)
    assert "may fail" in line and "will fail" not in line, line
    assert "optional tooling" in line, line
    assert "3 of 3 files" in line, line


def test_the_fold_line_states_both_halves_of_a_mixed_selection():
    """G1: a selection with both doomed and degraded files must say so --
    collapsing them into one verb is what made the fold lie."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    state = _mixed_forecast_state()
    forecast = state.forecast
    assert (forecast.will_fail_tooling, forecast.at_risk) == (5, 3), forecast
    line = ingest_tooling_summary_line(state)
    assert "5 will fail" in line, line
    assert "3 may fail" in line, line
    assert "8 of 8 files" in line, line


@pytest.mark.asyncio
async def test_the_rendered_fold_line_agrees_with_the_commit_line():
    """The mounted surface, not just the function: the fold, the commit
    line and the consent line must never state different fates for one
    selection."""
    from tldw_chatbook.Library.library_ingest_state import forecast_summary_line

    state = _doomed_state(21)
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        fold_line = pilot.app.query_one(
            "#ingest-preflight-tooling-summary", Static
        ).visual.plain
    commit_line = forecast_summary_line(state.forecast)
    assert "will fail" in commit_line and "will fail" in fold_line, (
        f"fold={fold_line!r} commit={commit_line!r}"
    )
    assert "may fail" not in fold_line, fold_line


# --- G2: a note that names no component is not a missing component ----------


_URL_NOTE = (
    "The site answered 403 to our check, so it could not be confirmed "
    "ahead of time. The import will still be attempted."
)


def _state_with_advisory_note(tooling_warnings: int = 2):
    """A state carrying both tooling warnings and a featureless note.

    ``advisory_lines`` is the seam the state builder fills for warnings
    with no ``feature`` key (the URL probe's note is the shipped one). It
    is set here directly because the canvas is the half under test.
    """
    state = _warned_state(tooling_warnings)
    object.__setattr__(
        state, "warning_lines", list(state.warning_lines) + [_URL_NOTE]
    )
    object.__setattr__(state, "advisory_lines", (_URL_NOTE,))
    return state


def test_a_featureless_note_is_not_counted_as_a_missing_component():
    """G2: the URL probe's "Could not check the link" note has no
    ``feature`` -- counting it into "N optional components aren't
    installed" describes a note as a package."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    state = _state_with_advisory_note(2)
    object.__setattr__(state, "forecast", None)
    line = ingest_tooling_summary_line(state)
    assert line.startswith("⚠ 2 optional components"), (
        f"a featureless note was counted as a component: {line!r}"
    )


@pytest.mark.asyncio
async def test_a_featureless_note_renders_outside_the_fold():
    """G2: the note is the only thing on screen that says the link could
    not be checked -- folding it away behind "What's missing" hides the
    real message and mislabels it as missing tooling."""
    app = _CanvasHost(_state_with_advisory_note(2))
    async with app.run_test() as pilot:
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        note = pilot.app.query_one("#ingest-preflight-note-0", Static)
        assert _URL_NOTE in note.visual.plain, note.visual.plain
        assert fold not in note.ancestors, (
            "the advisory note is hidden inside the tooling fold"
        )
        # And it is not repeated inside the fold as a tooling warning.
        folded_text = " ".join(
            widget.visual.plain for widget in fold.query(Static)
        )
        assert _URL_NOTE not in folded_text, folded_text


@pytest.mark.asyncio
async def test_a_note_alone_renders_no_missing_tooling_summary():
    """G2: with no feature-bearing warning there is no tooling to
    summarise -- a "⚠ 1 optional component isn't installed" line over a
    403 note is a fabricated diagnosis."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"web": ["https://example.com/a"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    object.__setattr__(state, "warning_lines", [_URL_NOTE])
    object.__setattr__(state, "advisory_lines", (_URL_NOTE,))
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#ingest-preflight-note-0", Static)
        assert _URL_NOTE in note.visual.plain
        assert not pilot.app.query("#ingest-preflight-tooling-summary"), (
            "a note-only pre-flight still claims components are missing"
        )
        assert not pilot.app.query("#ingest-preflight-tooling-detail"), (
            "an empty fold was rendered for a note-only pre-flight"
        )


# --- G3: the fold must survive the recompose that runs on every job tick ----


@pytest.mark.asyncio
async def test_an_expanded_fold_survives_a_dynamic_region_recompose():
    """G3: ``_update_library_ingest_dynamic_regions`` assigns
    ``summary.state`` and calls ``refresh(recompose=True)`` on every
    registry tick, so an open fold snapped shut under the user mid-read
    during an active import. Same convention as the option panels'
    ``expanded_type_groups``: the expansion is state, not a widget
    accident."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        LibraryIngestPreflightSummary,
    )

    app = _CanvasHost(_warned_state(4))
    async with app.run_test() as pilot:
        summary = pilot.app.query_one(LibraryIngestPreflightSummary)
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        fold.collapsed = False
        await pilot.pause()

        # Exactly what the screen's in-place update does per tick.
        summary.state = summary.state
        summary.refresh(recompose=True)
        await pilot.pause()

        reborn = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        assert reborn.collapsed is False, (
            "the fold snapped shut under the user on a registry tick"
        )


@pytest.mark.asyncio
async def test_toggling_the_fold_is_reported_so_the_screen_can_persist_it():
    """G3: the canvas is render-only, so the durable half of the
    expansion is a message the screen persists -- the same contract
    ``OptionPanelToggled`` has for the option panels."""
    app = _MessageRecordingHost(_warned_state(3))
    async with app.run_test() as pilot:
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        fold.collapsed = False
        await pilot.pause()
        fold.collapsed = True
        await pilot.pause()
    assert [event.expanded for event in app.tooling_detail_toggles] == [
        True,
        False,
    ], app.tooling_detail_toggles


@pytest.mark.asyncio
async def test_a_state_carried_expansion_opens_the_fold_on_compose():
    """G3's durable half: once the screen persists the flag, a FULL
    recompose (a structural change rebuilds the whole canvas) restores the
    fold the same way ``expanded_type_groups`` restores an option panel."""
    state = _warned_state(3)
    object.__setattr__(state, "tooling_detail_expanded", True)
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        fold = pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        assert fold.collapsed is False, (
            "a persisted expansion did not survive the rebuild"
        )


# --- G4: only a gate the user must act on OUTSIDE the app is "unavailable" --


@pytest.mark.asyncio
async def test_a_healthy_panel_does_not_lead_with_options_unavailable():
    """G4: ``blocked_count`` counted ordinary closed WITHIN-FORM gates, so
    a fully working default Web panel led its receipt with "2 options
    unavailable — single-page fetch selected". task-14824's intent was to
    surface PACKAGING gates -- work the user must do outside the app."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        build_type_group_title,
    )
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    for group in ("web", "pdf", "audio_video"):
        title = build_type_group_title(
            get_capabilities(group), {}, is_installed=lambda _feature: True
        )
        assert "unavailable" not in title, (
            f"a healthy {group} panel still reads broken: {title!r}"
        )

    # The same panel with a sibling toggle turned off is still healthy.
    title = build_type_group_title(
        get_capabilities("generic"),
        {"chunk": False},
        is_installed=lambda _feature: True,
    )
    assert "unavailable" not in title, title


@pytest.mark.asyncio
async def test_a_packaging_gate_still_leads_the_panel_receipt():
    """G4's counterpart: the gate task-14824 exists for -- a missing
    package -- must still be stated on the (keyboard-reachable) title."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        build_type_group_title,
    )
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    title = build_type_group_title(
        get_capabilities("pdf"), {}, is_installed=lambda _feature: False
    )
    assert "3 options unavailable" in title, title
    assert "needs PDF processing installed" in title, title

    audio = build_type_group_title(
        get_capabilities("audio_video"), {}, is_installed=lambda _feature: False
    )
    assert "13 options unavailable" in audio, audio


# --- G5: one install command must yield exactly one copy control ------------


@pytest.mark.asyncio
async def test_a_single_install_command_yields_a_single_copy_control():
    """G5: at exactly one command the canvas rendered BOTH the combined
    button and the per-extra button, copying the identical string under
    two labels -- the one-label-shape rule task-14822 added, defeated."""
    copied: list[str] = []

    class _ClipboardHost(_CanvasHost):
        def copy_to_clipboard(self, text: str) -> None:
            copied.append(text)

    app = _ClipboardHost(_warned_state(1))
    async with app.run_test() as pilot:
        buttons = list(
            pilot.app.query(".ingest-preflight-copy-command")
        )
        assert len(buttons) == 1, (
            "one command, two copy controls: "
            f"{[(b.id, b.label.plain) for b in buttons]}"
        )
        assert buttons[0].id == "ingest-preflight-copy-all-commands", (
            "the surviving control must be the one outside the fold"
        )
        buttons[0].press()
        await pilot.pause()
    assert copied == ['pip install -e ".[extra0]"'], copied
