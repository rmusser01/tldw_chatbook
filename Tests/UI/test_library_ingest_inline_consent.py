"""task-3314: inline two-press Start consent (guardrail modal retired).

The owner ruled (task-3310, ruling 2) that tooling-warning consent folds
into the inline commit/gate grammar: Start with active tooling warnings no
longer raises ``IngestGuardrailModal`` — the FIRST press converts the gate
line into an explicit confirm state naming the blast radius, and the
SECOND press submits. The mechanism MIRRORS the queue's incumbent
two-press pattern ("Clear finished", task-2015/2160): screen-attr state
carrier, arming updates only in place, a double-press dead zone, and
"the thing you armed against changed" disarms.

Unit tests drive the state machine on an unmounted screen (the
``_minimal_library_screen`` shape the retired guardrail-modal suite used);
pilot tests under ``LibraryHarness`` pin the rendered confirm line, the
Enter,Enter keyboard path, Esc-declines, and the in-place update
discipline (object identity across dynamic-region ticks).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Input, Static

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    ActiveIngestConsentScope,
    ActiveIngestJobRef,
    ActiveIngestSubmissionRefused,
    IngestJobState,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    active_ingest_start_confirm_line,
    build_ingest_forecast,
    build_library_ingest_state,
    count_warning_affected_files,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _preflight(**overrides) -> PreflightResult:
    defaults = {
        "type_groups": {},
        "warnings": [],
        "errors": [],
        "total_size": 0,
        "truncated": False,
        "total_files": 0,
    }
    defaults.update(overrides)
    return PreflightResult(**defaults)


_WARNING = {
    "feature": "pdf_processing",
    "label": "PDF processing",
    "hint": "Install pdfplumber.",
    "command": "pip install pdfplumber",
}


def _minimal_library_screen() -> LibraryScreen:
    """A LibraryScreen without the full UI (the guardrail suite's shape).

    ``_update_library_ingest_gate``/``_build_library_ingest_state`` are
    mocked: the unit tests here assert the consent STATE MACHINE; the
    rendered line is pinned by the pilot tests below.
    """
    screen = object.__new__(LibraryScreen)
    screen._library_ingest_form = LibraryIngestFormState()
    screen._library_ingest_preflight_worker = None
    screen._library_ingest_preflight_generation = 0
    # Keeps ``_update_library_ingest_dynamic_regions`` a no-op (different
    # canvas selected), matching the unmounted-screen constraint.
    screen._library_selected_row_id = ""
    screen._library_ingest_start_consent = None
    screen._library_ingest_start_confirm_armed_at = 0.0
    screen._library_ingest_last_submission = None
    screen._library_external_submit_generation = 0
    screen._library_external_submit_scope_id = None
    screen._library_external_submit_worker = None
    screen._library_external_submit_backend = None
    screen._library_external_submit_consent = None
    screen._notify_library_ingest_warning = MagicMock()
    screen._update_library_ingest_gate = MagicMock()
    screen._build_ingest_options_snapshot = MagicMock(
        side_effect=lambda: {
            group: dict(values)
            for group, values in screen._library_ingest_form.type_options.items()
        }
    )
    screen.refresh = MagicMock()
    screen.call_after_refresh = MagicMock()
    screen._refresh_library_ingest_canvas_preserving_context = MagicMock()
    screen.app_instance = MagicMock()
    screen.app_instance.library_ingest_jobs = LibraryIngestJobRegistry()
    screen.app_instance._resolve_ingest_backend = lambda: "local"
    screen._build_library_ingest_state = MagicMock(
        side_effect=lambda: SimpleNamespace(
            selection_has_nothing_importable=False,
            start_enabled=True,
            ingest_backend=screen.app_instance._resolve_ingest_backend(),
            forecast=build_ingest_forecast(
                screen._library_ingest_form.preflight,
                targets_server=(
                    screen.app_instance._resolve_ingest_backend() == "server"
                ),
            ),
        )
    )
    screen._clear_library_external_vad_progress = MagicMock()
    screen._set_library_external_status = MagicMock()
    screen._save_library_ingest_options = MagicMock()
    return screen


def _stage_warned_pdf(screen: LibraryScreen, tmp_path) -> str:
    pdf = tmp_path / "file.pdf"
    pdf.write_text("dummy")
    form = screen._library_ingest_form
    form.path = str(pdf)
    form.preflight = _preflight(
        type_groups={"pdf": [str(pdf)]},
        warnings=[dict(_WARNING)],
        total_files=1,
    )
    return str(pdf)


def _stage_plain_file(screen: LibraryScreen, tmp_path) -> str:
    source = tmp_path / "file.txt"
    source.write_text("body")
    screen._library_ingest_form.path = str(source)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"generic": [str(source)]},
        total_files=1,
    )
    return str(source)


def _stage_external_parakeet_audio(screen: LibraryScreen, tmp_path) -> str:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"RIFF")
    form = screen._library_ingest_form
    form.path = str(source)
    form.preflight = _preflight(
        type_groups={"audio_video": [str(source)]},
        total_files=1,
    )
    form.type_options["audio_video"] = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model_dir": str(tmp_path / "model"),
    }
    screen._prepare_library_external_submission = MagicMock()
    return str(source)


def _stage_warned_external_audio(screen: LibraryScreen, tmp_path) -> str:
    source = _stage_external_parakeet_audio(screen, tmp_path)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"audio_video": [source]},
        warnings=[
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "audio transcription",
                "command": 'pip install -e ".[audio]"',
            }
        ],
        total_files=1,
    )
    return source


@pytest.mark.parametrize(
    ("active_files", "is_folder", "tooling_files", "expected"),
    [
        (1, False, 0, "Import active. Start again to queue a duplicate."),
        (2, True, 0, "2 active files. Start again to queue all."),
        (1, False, 2, "Import active; 2 may fail. Start again to queue."),
    ],
)
def test_active_ingest_confirm_copy_is_exact(
    active_files, is_folder, tooling_files, expected
):
    assert active_ingest_start_confirm_line(
        active_source_count=active_files,
        is_folder=is_folder,
        tooling_affected_count=tooling_files,
    ) == expected
    assert len(expected) <= 48


def test_active_confirm_override_arms_state_without_tooling_warning():
    form = LibraryIngestFormState(path="/tmp/file.txt")
    form.preflight = _preflight(
        type_groups={"generic": ["/tmp/file.txt"]}, total_files=1
    )

    state = build_library_ingest_state(
        (),
        form=form,
        start_confirm_armed=True,
        start_confirm_line="Import active. Start again to queue a duplicate.",
    )

    assert state.start_confirm_armed is True
    assert state.start_quiet_line == (
        "Import active. Start again to queue a duplicate."
    )


def test_active_job_lifecycle_transition_preserves_armed_consent(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    job = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    armed = screen._library_ingest_start_consent
    screen.app_instance.library_ingest_jobs.mark_parsing(job.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(job.job_id)

    assert screen._current_library_ingest_start_consent(source).fingerprint == (
        armed.fingerprint
    )


def test_tooling_only_consent_cannot_override_late_duplicate(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    duplicate = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen._library_ingest_start_consent.active_job_ids == (
        duplicate.job_id,
    )


def test_active_duplicate_second_press_passes_one_shot_override(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    job = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    screen.app_instance.library_ingest_jobs.mark_parsing(job.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(job.job_id)
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert isinstance(kwargs["active_duplicate_consent"], ActiveIngestConsentScope)
    assert kwargs["active_duplicate_consent"].active_job_ids == (job.job_id,)
    assert screen._library_ingest_start_consent is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda screen: setattr(screen._library_ingest_form, "title", "changed"),
        lambda screen: setattr(screen._library_ingest_form, "author", "changed"),
        lambda screen: setattr(screen._library_ingest_form, "keywords", "changed"),
        lambda screen: screen._library_ingest_form.type_options.setdefault(
            "generic", {}
        ).update({"custom_prompt": "changed"}),
        lambda screen: setattr(
            screen.app_instance, "_resolve_ingest_backend", lambda: "server"
        ),
    ],
)
def test_request_mutation_changes_active_consent_fingerprint(
    mutation, tmp_path
):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)
    before = screen._current_library_ingest_start_consent(source).fingerprint

    mutation(screen)

    after = screen._current_library_ingest_start_consent(source).fingerprint
    assert after != before


def test_source_change_changes_active_consent_fingerprint(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)
    before = screen._current_library_ingest_start_consent(source).fingerprint

    after = screen._current_library_ingest_start_consent(
        str(tmp_path / "other.txt")
    ).fingerprint

    assert after != before


def test_warning_change_changes_active_consent_fingerprint(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_pdf(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)
    before = screen._current_library_ingest_start_consent(source).fingerprint

    screen._library_ingest_form.preflight = _preflight(
        type_groups={"pdf": [source]},
        warnings=[{**_WARNING, "hint": "Different warning."}],
        total_files=1,
    )

    after = screen._current_library_ingest_start_consent(source).fingerprint
    assert after != before


def test_identical_warning_text_with_changed_affected_count_rearms_consent(
    tmp_path,
):
    screen = _minimal_library_screen()
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_text("first")
    second.write_text("second")
    form = screen._library_ingest_form
    form.path = str(tmp_path)
    form.preflight = _preflight(
        type_groups={"pdf": [str(first)], "generic": [str(second)]},
        warnings=[dict(_WARNING)],
        total_files=2,
    )
    before = screen._current_library_ingest_start_consent(str(tmp_path))

    form.preflight = _preflight(
        type_groups={"pdf": [str(first), str(second)]},
        warnings=[dict(_WARNING)],
        total_files=2,
    )
    after = screen._current_library_ingest_start_consent(str(tmp_path))

    assert before.admission_scope.candidate_digest == (
        after.admission_scope.candidate_digest
    )
    assert before.tooling_affected_count == 1
    assert after.tooling_affected_count == 2
    assert before.fingerprint != after.fingerprint


def test_active_membership_change_requires_fresh_second_press(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    first = screen.app_instance.library_ingest_jobs.submit(source_path=source)
    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen.app_instance.library_ingest_jobs.mark_done(first.job_id, media_id=1)
    second = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen._library_ingest_start_consent.active_job_ids == (second.job_id,)


def test_folder_preview_counts_distinct_active_files(tmp_path):
    screen = _minimal_library_screen()
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
        screen.app_instance.library_ingest_jobs.submit(source_path=str(path))
    screen._library_ingest_form.path = str(folder)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"generic": [str(path) for path in paths]},
        total_files=2,
    )

    screen._submit_library_ingest_form()

    assert screen._library_ingest_start_consent.active_source_count == 2
    consent = screen._library_ingest_start_consent
    assert active_ingest_start_confirm_line(
        active_source_count=consent.active_source_count,
        is_folder=consent.is_folder,
        tooling_affected_count=consent.tooling_affected_count,
    ) == "2 active files. Start again to queue all."


def test_combined_tooling_and_active_warning_takes_two_presses(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_pdf(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    consent = screen._library_ingest_start_consent
    assert active_ingest_start_confirm_line(
        active_source_count=consent.active_source_count,
        is_folder=consent.is_folder,
        tooling_affected_count=consent.tooling_affected_count,
    ) == "Import active; 1 may fail. Start again to queue."
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert screen.app_instance.submit_library_ingest_job.call_count == 1
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is not None
    )


def test_active_preview_blocks_external_preparation_before_retain(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_external_parakeet_audio(screen, tmp_path)
    screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()

    screen._prepare_library_external_submission.assert_not_called()
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    service.retain_prepared.assert_not_called()
    assert screen._library_ingest_form.path == source


def test_late_active_refusal_preserves_form_without_generic_error(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen.app_instance.submit_library_ingest_job.side_effect = (
        ActiveIngestSubmissionRefused(
            (ActiveIngestJobRef("ingest-job-7", IngestJobState.QUEUED),)
        )
    )

    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {}}
    )

    assert screen._library_ingest_form.path == source
    assert screen._library_ingest_start_consent is not None
    screen.app_instance.notify.assert_not_called()


def test_late_active_refusal_releases_untransferred_external_scope(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    screen._library_external_submit_generation = 3
    screen._library_external_submit_scope_id = "scope-3"
    screen._library_external_submit_backend = "local"
    screen.app_instance.submit_library_ingest_job.side_effect = (
        ActiveIngestSubmissionRefused(
            (ActiveIngestJobRef("ingest-job-7", IngestJobState.PARSING),)
        )
    )

    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {"audio_video": {}}},
        generation=3,
        scope_id="scope-3",
    )

    service.release_scope.assert_called_once_with("scope-3")
    assert screen._library_ingest_form.path == source
    assert screen._library_ingest_start_consent is not None


def test_late_refusal_fallback_single_file_can_be_confirmed(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    hidden = tmp_path / "worker-canonical.txt"
    hidden.write_text("body")
    job = screen.app_instance.library_ingest_jobs.submit(source_path=str(hidden))
    refusal = ActiveIngestSubmissionRefused(
        (ActiveIngestJobRef(job.job_id, IngestJobState.QUEUED),)
    )
    screen.app_instance.submit_library_ingest_job.side_effect = [refusal, None]

    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {}}
    )
    screen.app_instance.library_ingest_jobs.mark_parsing(job.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(job.job_id)
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert screen.app_instance.submit_library_ingest_job.call_count == 2
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is not None
    )


def test_late_refusal_fallback_folder_can_be_confirmed(tmp_path):
    screen = _minimal_library_screen()
    folder = tmp_path / "batch"
    folder.mkdir()
    child = folder / "captured.txt"
    child.write_text("body")
    hidden = tmp_path / "expanded-later.txt"
    hidden.write_text("body")
    job = screen.app_instance.library_ingest_jobs.submit(source_path=str(hidden))
    screen._library_ingest_form.path = str(folder)
    screen._library_ingest_form.preflight = _preflight(
        type_groups={"generic": [str(child)]}, total_files=1
    )
    refusal = ActiveIngestSubmissionRefused(
        (ActiveIngestJobRef(job.job_id, IngestJobState.PARSING),)
    )
    screen.app_instance.submit_library_ingest_job.side_effect = [refusal, None]

    screen._enqueue_library_ingest_snapshot(
        {"source_path": str(folder), "ingest_options": {}}
    )
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert screen.app_instance.submit_library_ingest_job.call_count == 2
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is not None
    )


def test_external_preparation_revalidates_active_membership_before_enqueue(
    tmp_path,
):
    screen = _minimal_library_screen()
    source = _stage_external_parakeet_audio(screen, tmp_path)
    first = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    prepare_args = screen._prepare_library_external_submission.call_args.args
    generation, scope_id = prepare_args[:2]
    submit_kwargs = prepare_args[-1]
    screen.app_instance.library_ingest_jobs.mark_done(first.job_id, media_id=1)
    second = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._apply_library_external_preparation(
        generation,
        scope_id,
        MagicMock(),
        submit_kwargs,
        None,
        None,
    )

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen._library_ingest_start_consent.active_job_ids == (second.job_id,)
    screen.app_instance._ensure_parakeet_source_service.return_value.release_scope.assert_called_once_with(
        scope_id
    )


def test_option_reset_disarms_pending_consent_before_repaint(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    screen._library_ingest_form.type_options["generic"] = {"custom_prompt": "old"}
    screen.app_instance.library_ingest_jobs.submit(source_path=source)
    screen._submit_library_ingest_form()
    assert screen._library_ingest_start_consent is not None
    event = MagicMock()
    event.button.id = "opt-generic-reset"

    with patch.object(library_screen_module, "save_settings_to_cli_config"):
        screen.handle_library_ingest_option_reset(event)

    assert screen._library_ingest_start_consent is None
    screen._refresh_library_ingest_canvas_preserving_context.assert_called_once_with()


@pytest.mark.parametrize("is_folder", [False, True], ids=["file", "folder"])
def test_external_authoritative_fallback_second_preparation_queues_override(
    tmp_path, is_folder
):
    screen = _minimal_library_screen()
    source = _stage_external_parakeet_audio(screen, tmp_path)
    submitted_source = source
    if is_folder:
        folder = tmp_path / "batch"
        folder.mkdir()
        child = folder / "audio.wav"
        child.write_bytes(b"RIFF")
        submitted_source = str(folder)
        screen._library_ingest_form.path = submitted_source
        screen._library_ingest_form.preflight = _preflight(
            type_groups={"audio_video": [str(child)]}, total_files=1
        )
    hidden = tmp_path / "expanded-later.wav"
    hidden.write_bytes(b"RIFF")
    job = screen.app_instance.library_ingest_jobs.submit(source_path=str(hidden))
    refusal = ActiveIngestSubmissionRefused(
        (ActiveIngestJobRef(job.job_id, IngestJobState.QUEUED),)
    )
    screen.app_instance.submit_library_ingest_job.side_effect = [refusal, None]

    screen._submit_library_ingest_form()
    first_prepare = screen._prepare_library_external_submission.call_args.args
    screen._apply_library_external_preparation(
        first_prepare[0], first_prepare[1], MagicMock(), first_prepare[-1], None, None
    )
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    second_prepare = screen._prepare_library_external_submission.call_args.args
    screen._apply_library_external_preparation(
        second_prepare[0],
        second_prepare[1],
        MagicMock(),
        second_prepare[-1],
        None,
        None,
    )

    assert screen.app_instance.submit_library_ingest_job.call_count == 2
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is not None
    )
    assert screen._library_ingest_form.path == ""
    assert screen._library_ingest_start_consent is None
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    service.release_scope.assert_called_once_with(first_prepare[1])


def test_unrelated_active_job_churn_preserves_authoritative_fallback(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    hidden = tmp_path / "worker-canonical.txt"
    hidden.write_text("body")
    matched = screen.app_instance.library_ingest_jobs.submit(source_path=str(hidden))
    refusal = ActiveIngestSubmissionRefused(
        (ActiveIngestJobRef(matched.job_id, IngestJobState.QUEUED),)
    )
    screen.app_instance.submit_library_ingest_job.side_effect = [refusal, None]
    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {}}
    )
    unrelated_path = tmp_path / "unrelated.txt"
    unrelated_path.write_text("other")
    unrelated = screen.app_instance.library_ingest_jobs.submit(
        source_path=str(unrelated_path)
    )
    screen.app_instance.library_ingest_jobs.mark_parsing(unrelated.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(unrelated.job_id)
    screen.app_instance.library_ingest_jobs.mark_done(unrelated.job_id, media_id=2)

    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is not None
    )


@pytest.mark.parametrize("terminal", ["done", "replacement"])
def test_authoritative_fallback_matching_membership_change_requires_new_consent(
    tmp_path, terminal
):
    screen = _minimal_library_screen()
    source = _stage_plain_file(screen, tmp_path)
    hidden = tmp_path / "worker-canonical.txt"
    hidden.write_text("body")
    matched = screen.app_instance.library_ingest_jobs.submit(source_path=str(hidden))
    refusal = ActiveIngestSubmissionRefused(
        (ActiveIngestJobRef(matched.job_id, IngestJobState.QUEUED),)
    )
    screen.app_instance.submit_library_ingest_job.side_effect = [refusal, None]
    screen._enqueue_library_ingest_snapshot(
        {"source_path": source, "ingest_options": {}}
    )
    screen.app_instance.library_ingest_jobs.mark_parsing(matched.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(matched.job_id)
    screen.app_instance.library_ingest_jobs.mark_done(matched.job_id, media_id=1)
    if terminal == "replacement":
        replacement = screen.app_instance.library_ingest_jobs.submit(
            source_path=str(hidden)
        )
        assert replacement.job_id != matched.job_id
        screen.app_instance.submit_library_ingest_job.side_effect = [
            ActiveIngestSubmissionRefused(
                (
                    ActiveIngestJobRef(
                        replacement.job_id, IngestJobState.QUEUED
                    ),
                )
            )
        ]

    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()

    assert screen.app_instance.submit_library_ingest_job.call_count == 2
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is None
    )
    if terminal == "done":
        assert screen._library_ingest_start_consent is None
    else:
        assert screen._library_ingest_start_consent is not None
        assert screen._library_ingest_start_consent.active_job_ids == (
            replacement.job_id,
        )


def test_combined_external_confirm_survives_matching_job_finishing(tmp_path):
    screen = _minimal_library_screen()
    source = _stage_warned_external_audio(screen, tmp_path)
    matched = screen.app_instance.library_ingest_jobs.submit(source_path=source)

    screen._submit_library_ingest_form()
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    prepare = screen._prepare_library_external_submission.call_args.args
    screen.app_instance.library_ingest_jobs.mark_parsing(matched.job_id)
    screen.app_instance.library_ingest_jobs.mark_writing(matched.job_id)
    screen.app_instance.library_ingest_jobs.mark_done(matched.job_id, media_id=1)

    screen._apply_library_external_preparation(
        prepare[0], prepare[1], MagicMock(), prepare[-1], None, None
    )

    screen.app_instance.submit_library_ingest_job.assert_called_once()
    assert (
        screen.app_instance.submit_library_ingest_job.call_args.kwargs[
            "active_duplicate_consent"
        ]
        is None
    )
    assert screen._library_ingest_start_consent is None
    service = screen.app_instance._ensure_parakeet_source_service.return_value
    service.release_scope.assert_not_called()


# --- migrated from the retired guardrail suite: submit-flow contracts -------


def test_submit_with_blank_path_warns_to_import_not_ingest():
    """(task-2857 review, migrated) A blank path warns with the form's
    "import" wording and submits nothing — no consent state is touched."""
    screen = _minimal_library_screen()
    screen._library_ingest_form.path = ""

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()

    screen._notify_library_ingest_warning.assert_called_once_with(
        "Please choose a file to import."
    )
    screen.app_instance.submit_library_ingest_job.assert_not_called()
    mock_app.push_screen.assert_not_called()
    assert screen._library_ingest_start_consent is None


def test_first_start_with_warnings_arms_instead_of_submitting(tmp_path):
    """AC#1/AC#2: the first press with active tooling warnings must NOT
    submit and must NOT push any modal — it arms the inline confirm."""
    screen = _minimal_library_screen()
    _stage_warned_pdf(screen, tmp_path)

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    mock_app.push_screen.assert_not_called()
    assert screen._library_ingest_start_consent is not None
    assert screen._library_ingest_start_consent.tooling_affected_count == 1
    # Arming re-renders the gate line in place (never a recompose).
    screen._update_library_ingest_gate.assert_called()
    screen.refresh.assert_not_called()


def test_second_start_submits_after_the_dead_zone(tmp_path):
    """AC#1: the second press (a decision, not a double-click) submits."""
    screen = _minimal_library_screen()
    pdf = _stage_warned_pdf(screen, tmp_path)

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()
        assert screen._library_ingest_start_consent is not None
        # Rewind past the double-press dead zone (the Clear-finished rule).
        screen._library_ingest_start_confirm_armed_at -= 1.0
        screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_called_once()
    call_kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert call_kwargs["source_path"] == pdf
    assert screen._library_ingest_start_consent is None
    mock_app.push_screen.assert_not_called()


def test_press_inside_the_dead_zone_is_one_gesture_not_consent(tmp_path):
    """task-2160's double-click rule, mirrored: a press landing within the
    dead zone of the arming press must neither submit nor disarm."""
    screen = _minimal_library_screen()
    _stage_warned_pdf(screen, tmp_path)

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()
        screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen._library_ingest_start_consent is not None


def test_submit_without_warnings_is_a_single_press(tmp_path):
    """AC#4: no warnings — one press submits, nothing ever arms."""
    txt = tmp_path / "file.txt"
    txt.write_text("hello")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(txt)
    form.preflight = _preflight(
        type_groups={"generic": [str(txt)]}, total_files=1
    )

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()

    mock_app.push_screen.assert_not_called()
    screen.app_instance.submit_library_ingest_job.assert_called_once()
    call_kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert call_kwargs["source_path"] == str(txt)
    assert screen._library_ingest_start_consent is None


def test_submit_clears_the_stale_preflight_summary(tmp_path):
    """(task-665, migrated) Submitting must not leave the previous file's
    summary on screen."""
    txt = tmp_path / "file.txt"
    txt.write_text("hello")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(txt)
    form.title = "Some title"
    form.preflight = _preflight(
        type_groups={"generic": [str(txt)]}, total_files=1
    )

    mock_app = MagicMock()
    with patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)
    ):
        screen._submit_library_ingest_form()

    assert form.path == ""
    assert form.title == ""
    assert form.preflight is None, "stale pre-flight summary survived the submit"
    assert form.preflight_checking is False


# --- the reset set: what invalidates the forecast clears the consent --------


def test_invalidating_the_preflight_disarms_a_pending_confirm(tmp_path):
    """Submit/Clear/reset all route through the invalidator — a consent
    armed against a forecast that no longer exists must not survive it."""
    screen = _minimal_library_screen()
    _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    assert screen._library_ingest_start_consent is not None

    screen._invalidate_library_ingest_preflight()

    assert screen._library_ingest_start_consent is None


def test_result_with_different_warnings_disarms_the_pending_confirm(tmp_path):
    """A fresh forecast carrying DIFFERENT warnings invalidates the consent
    (the Clear-finished "the queue you armed against changed" rule)."""
    screen = _minimal_library_screen()
    _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    assert screen._library_ingest_start_consent is not None

    other = _preflight(
        type_groups={"audio_video": ["/tmp/talk.mp3"]},
        warnings=[
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "audio transcription",
            }
        ],
        total_files=1,
    )
    screen._apply_library_ingest_preflight_result(
        other, screen._library_ingest_preflight_generation
    )

    assert screen._library_ingest_start_consent is None


def test_result_with_identical_warnings_keeps_the_pending_confirm(tmp_path):
    """The Enter-in-path re-trigger lands an IDENTICAL forecast — that must
    not steal the pending confirm, or Enter,Enter could never submit."""
    screen = _minimal_library_screen()
    pdf = _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    armed = screen._library_ingest_start_consent
    assert armed is not None

    same = _preflight(
        type_groups={"pdf": [pdf]},
        warnings=[dict(_WARNING)],
        total_files=1,
    )
    screen._apply_library_ingest_preflight_result(
        same, screen._library_ingest_preflight_generation
    )

    assert screen._library_ingest_start_consent == armed


@pytest.mark.asyncio
async def test_escape_while_armed_declines_and_stays_on_the_canvas(tmp_path):
    """AC#4: Esc is the consent "no" — it drops the pending confirm and
    stays on the Ingest canvas; a second Esc leaves as before."""
    screen = _minimal_library_screen()
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA
    _stage_warned_pdf(screen, tmp_path)
    screen._submit_library_ingest_form()
    assert screen._library_ingest_start_consent is not None
    screen._select_library_rail_row = AsyncMock()
    # ``object.__new__`` bypasses Widget.__init__; the action's tail reads
    # ``is_mounted`` on the non-armed path.
    screen._is_mounted = False

    await screen.action_library_ingest_back()

    assert screen._library_ingest_start_consent is None
    screen._select_library_rail_row.assert_not_called()

    # Second Esc: not armed anymore — leaves for the hub as before.
    await screen.action_library_ingest_back()
    screen._select_library_rail_row.assert_awaited_once_with("")


# --- the blast-radius count (replaces the modal's _affected_counts) ---------


def test_count_warning_affected_files_counts_distinct_staged_files():
    """(migrated from test_affected_counts_aggregates_by_feature) Files in
    groups that depend on a warned feature are counted once each."""
    preflight = _preflight(
        type_groups={
            "pdf": ["/a.pdf", "/b.pdf"],
            "audio_video": ["/a.mp3"],
            "generic": ["/c.txt"],
            "unsupported": ["/weird.xyz"],
        },
        warnings=[dict(_WARNING)],
        total_files=5,
    )
    # Only the pdf group depends on pdf_processing.
    assert count_warning_affected_files(preflight) == 2


def test_count_warning_affected_files_zero_without_warnings():
    preflight = _preflight(
        type_groups={"pdf": ["/a.pdf"]},
        total_files=1,
    )
    assert count_warning_affected_files(preflight) == 0


# --- state builder: the confirm copy on the gate line ------------------------


def _warned_state(*, armed: bool, files: list[str] | None = None):
    files = files if files is not None else ["/tmp/a.pdf"]
    form = LibraryIngestFormState(path=files[0])
    form.preflight = _preflight(
        type_groups={"pdf": list(files)},
        warnings=[dict(_WARNING)],
        total_files=len(files),
    )
    return build_library_ingest_state(
        (), form=form, start_confirm_armed=armed
    )


def test_armed_state_converts_the_gate_line_into_the_confirm_copy():
    state = _warned_state(armed=True)
    assert state.start_confirm_armed is True
    # (task-14820) ``pdf_processing`` is the pdf group's REQUIRED feature,
    # so this file cannot import at all -- "may fail" understated a
    # certainty the forecast beside it now states outright.
    assert state.start_quiet_line == (
        "⚠ Press Start again to import anyway — 1 file will fail without "
        "more tooling."
    )
    # The gate itself stays open: the second press must be possible.
    assert state.start_enabled is True


def test_confirm_copy_pluralizes_the_file_count():
    """(migrated from the modal's pluralization pin) "2 files", never
    "(1 files)"."""
    state = _warned_state(armed=True, files=["/tmp/a.pdf", "/tmp/b.pdf"])
    assert "2 files will fail" in state.start_quiet_line
    single = _warned_state(armed=True)
    assert "1 file will fail" in single.start_quiet_line
    assert "1 files" not in single.start_quiet_line


def test_confirm_count_is_the_forecast_count_not_a_second_computation():
    """(task-14820 AC#1) The consent line and the commit forecast are two
    renderings of ONE object -- live saw them disagree by 8 files."""
    state = _warned_state(armed=True, files=["/tmp/a.pdf", "/tmp/b.pdf"])
    assert "2 will fail" in state.commit_summary_line
    assert "2 files will fail" in state.start_quiet_line
    assert count_warning_affected_files(state.form.preflight) == 2


def test_unarmed_state_keeps_the_plain_gate_line():
    state = _warned_state(armed=False)
    assert state.start_confirm_armed is False
    assert "Press Start again" not in state.start_quiet_line


def test_armed_flag_without_warnings_never_paints_confirm_copy():
    """A stale armed flag with no active warnings must not manufacture a
    consent state the forecast doesn't justify."""
    form = LibraryIngestFormState(path="/tmp/a.txt")
    form.preflight = _preflight(
        type_groups={"generic": ["/tmp/a.txt"]}, total_files=1
    )
    state = build_library_ingest_state((), form=form, start_confirm_armed=True)
    assert state.start_confirm_armed is False
    assert "Press Start again" not in state.start_quiet_line


def test_confirm_css_uses_theme_tokens():
    """(migrated from the modal's token pin) The confirm treatment's CSS
    carries theme tokens, no off-token literals."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        LibraryIngestCanvas,
    )

    css = LibraryIngestCanvas.DEFAULT_CSS
    assert "-ingest-start-confirm" in css
    assert "$warning" in css
    assert "black" not in css
    assert "gray" not in css


# --- pilot: the rendered flow -------------------------------------------------


async def _warned_ingest_screen(host, pilot, monkeypatch, tmp_path):
    """Enter Ingest with a staged source whose forecast carries a warning."""
    source = tmp_path / "file.pdf"
    source.write_text("%PDF-1.4 dummy")
    result = _preflight(
        type_groups={"pdf": [str(source)]},
        warnings=[dict(_WARNING)],
        total_files=1,
    )
    screen = host.screen_stack[-1]
    await _wait_for_library_shell(screen, pilot)
    monkeypatch.setattr(
        library_screen_module,
        "analyze_path",
        lambda path, scan_limit=1000, **_kwargs: result,
    )
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    path_input = await _wait_for_selector(screen, pilot, "#library-ingest-path")
    path_input.value = str(source)
    screen._trigger_library_ingest_preflight(str(source))
    await _wait_for_condition(
        pilot,
        lambda: screen._library_ingest_form.preflight is not None
        and bool(screen._library_ingest_form.preflight.warnings),
        message="warned pre-flight never landed",
    )
    await pilot.pause()
    await pilot.pause()
    # The forecast changed the type-group set, which takes the STRUCTURAL
    # recompose path and replaces every form widget -- the pre-forecast
    # Input reference is a detached widget by now (focusing it would send
    # keys nowhere). Re-query for the live one.
    path_input = screen.query_one("#library-ingest-path", Input)
    return screen, path_input, str(source)


def _pilot_app():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = object()
    return app


@pytest.mark.asyncio
async def test_enter_enter_two_press_flow_renders_confirm_then_submits(
    monkeypatch, tmp_path
):
    """AC#1 + the pinned keyboard path: Enter arms (confirm copy on the
    gate line, warning treatment, copy affordance still reachable, no
    modal), Enter again submits."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert submitted == []
        assert host.screen_stack[-1] is screen, "a modal was pushed (AC#2)"
        quiet = screen.query_one("#library-ingest-start-quiet-line", Static)
        text = str(quiet.renderable)
        assert "Press Start again to import anyway" in text
        # (task-14820) A missing REQUIRED feature is a certainty, not a risk.
        assert "1 file will fail without more tooling" in text
        assert quiet.has_class("-ingest-start-confirm")
        # AC#3: the copy-install-command affordance stays reachable at the
        # inline warnings while the confirm is pending. Queried by CLASS,
        # not by index: which button carries the command (one combined
        # button, or one per distinct extra) is the canvas's call and has
        # changed with the warning fold -- that it is REACHABLE is the
        # contract this test owns.
        assert list(screen.query(".ingest-preflight-copy-command"))

        # The confirm is gate-updater-owned chrome: it must survive the
        # in-place hot path with object identity (task-2042 discipline).
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert (
            screen.query_one("#library-ingest-start-quiet-line", Static) is quiet
        )
        assert "Press Start again" in str(quiet.renderable)

        # Second Enter (past the dead zone) submits.
        screen._library_ingest_start_confirm_armed_at -= 1.0
        await pilot.press("enter")
        await pilot.pause()

        assert [k.get("source_path") for k in submitted] == [source]
        assert screen._library_ingest_start_consent is None
        # The gate line left the confirm treatment with the submit.
        quiet_after = screen.query_one(
            "#library-ingest-start-quiet-line", Static
        )
        assert not quiet_after.has_class("-ingest-start-confirm")


@pytest.mark.asyncio
async def test_escape_declines_the_pending_confirm_and_stays(
    monkeypatch, tmp_path
):
    """AC#4 rendered: Esc while armed clears the confirm copy and stays on
    the Ingest canvas."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, _source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._library_ingest_start_consent is not None

        await pilot.press("escape")
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        assert screen._library_ingest_start_consent is None
        quiet = screen.query_one("#library-ingest-start-quiet-line", Static)
        assert "Press Start again" not in str(quiet.renderable)
        assert not quiet.has_class("-ingest-start-confirm")
        assert submitted == []


@pytest.mark.asyncio
async def test_editing_the_path_resets_the_pending_confirm(
    monkeypatch, tmp_path
):
    """AC#4: editing the form (a genuine path change) invalidates the
    forecast the consent was armed against."""
    app = _pilot_app()
    monkeypatch.setattr(
        app, "submit_library_ingest_job", lambda **kwargs: None, raising=False
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, _source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._library_ingest_start_consent is not None

        await pilot.press("x")
        await pilot.pause()

        assert screen._library_ingest_start_consent is None
        quiet = screen.query_one("#library-ingest-start-quiet-line", Static)
        assert "Press Start again" not in str(quiet.renderable)


# --- xhigh review + live-verify round: consent state-machine defects --------


@pytest.mark.asyncio
async def test_enter_armed_consent_survives_the_start_click_and_submits(
    monkeypatch, tmp_path
):
    """The confirm copy says "Press Start again to import anyway" — so the
    Start CLICK must be the second press, not a reset.

    Live defect: the click blurs the path field first, the blur handler
    disarmed the pending consent, and the press then merely RE-ARMED.
    Nothing ever submitted — the copy instructed the exact gesture that
    made submission impossible. A blur is not a forecast-invalidating
    edit; only real edits/invalidation disarm.
    """
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()
        assert screen._library_ingest_start_consent is not None
        assert submitted == []

        # The human reads the confirm before clicking; model that pause so
        # the double-press dead zone (a repeat-gesture guard) is not what
        # this test is measuring.
        screen._library_ingest_start_confirm_armed_at -= 1.0

        # The Start button sits below the fold at this terminal size, and
        # ``pilot.click`` addresses SCREEN coordinates -- an unscrolled
        # click lands on whatever occupies that cell and the press handler
        # never runs at all (which would make this test vacuous).
        start_button = screen.query_one("#library-ingest-start")
        start_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#library-ingest-start")
        await pilot.pause()

        assert [k.get("source_path") for k in submitted] == [source], (
            "the Start click the confirm copy asks for did not submit "
            f"(armed={screen._library_ingest_start_consent is not None})"
        )


@pytest.mark.asyncio
async def test_path_blur_alone_keeps_the_pending_confirm(monkeypatch, tmp_path):
    """Unit-level statement of the same rule: leaving the path field is
    not an edit, so it must not disarm."""
    app = _pilot_app()
    monkeypatch.setattr(
        app, "submit_library_ingest_job", lambda **kwargs: None, raising=False
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, _source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._library_ingest_start_consent is not None

        screen.set_focus(screen.query_one("#library-ingest-browse"))
        await pilot.pause()

        assert screen._library_ingest_start_consent is not None, (
            "a bare blur cancelled the pending Start consent"
        )
        quiet = screen.query_one("#library-ingest-start-quiet-line", Static)
        assert "Press Start again" in str(quiet.renderable)


@pytest.mark.asyncio
async def test_browse_picking_a_new_file_disarms_the_pending_confirm(
    monkeypatch, tmp_path
):
    """A consent armed against file A must never cover file B.

    The Browse… callback wrote ``form.path`` directly; the recomposed
    Input then re-announced that value, which ``handle_library_ingest_
    path_changed``'s echo guard drops — so the disarm on the genuine-edit
    seam never ran and B could be submitted under A's consent.
    """
    app = _pilot_app()
    monkeypatch.setattr(
        app, "submit_library_ingest_job", lambda **kwargs: None, raising=False
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, path_input, source = await _warned_ingest_screen(
            host, pilot, monkeypatch, tmp_path
        )
        screen.set_focus(path_input)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._library_ingest_start_consent is not None

        other = tmp_path / "second.pdf"
        other.write_text("dummy")

        callbacks: list = []
        monkeypatch.setattr(
            screen.app,
            "push_screen",
            lambda *args, **kwargs: callbacks.append(
                kwargs.get("callback", args[1] if len(args) > 1 else None)
            ),
            raising=False,
        )
        screen.query_one("#library-ingest-browse").press()
        await pilot.pause()
        assert callbacks and callbacks[0] is not None, "Browse pushed no picker"

        await callbacks[0](other)
        await pilot.pause()

        assert screen._library_ingest_form.path == str(other)
        assert screen._library_ingest_start_consent is None, (
            "a consent armed against "
            f"{source} still covers the newly picked {other}"
        )
