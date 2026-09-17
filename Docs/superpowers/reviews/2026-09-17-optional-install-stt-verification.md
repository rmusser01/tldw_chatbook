# Optional-install clipboard and YouTube STT verification

Date: 2026-09-17. Tasks: TASK-32754, TASK-32755.
Base: dev `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`.
The affected production files also match latest main
`b0dadf19414f5f8faf69b854d8e59007d275083e`.

## Root causes and repairs

The Library copy button called Textual's unacknowledged OSC 52 clipboard route
and immediately reported success. The missing-feature dialog called pyperclip
on the UI thread without readback; a subprocess backend can fail silently.
The dialog also rendered extras as Rich markup and its development command
contained an uninterpolated placeholder.

Both buttons now share bounded native copy/readback off the event loop. A failed,
remote, browser, headless or process-owned Qt clipboard uses Textual's route with
an unconfirmed-delivery warning. Failed native processes are terminated before
returning, and the Library reveals literal commands for manual selection. The
feature dialog preserves and quotes the extras in both displayed commands.

Audio/video capability metadata listed alternative STT backends cumulatively,
including retired MLX providers. It surfaced those missing packages even with
the selected backend installed, affecting both warnings and start consent.
The underlying YouTube route already respected the chosen provider. The repair
removes retired providers from this inventory, includes supported transcribe.cpp,
and projects the captured warnings through the current selection before display
and consent. Provider changes reuse the original snapshot. Auto remains
faster-whisper; other installed backends are explicitly selectable.

ADR required: no. Existing STT policy:
[ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md).
These are routine repairs to the existing interfaces, without a new dependency,
automatic provider fallback, schema or ownership policy.

## Automated verification

All Python runs used Python 3.12 and targeted tests only.

| Check | Result |
|---|---|
| Affected capability, preflight, state, canvas, structural, inline-consent tests plus new mounted clipboard, STT-warning and YouTube-routing tests | 646 passed; 19 known baseline cases deselected |
| Final clipboard/native/design-token run | 19 passed: 8 mounted cases repeated from above, 5 native-process cases, 6 token-governance cases |
| Linux container: STT-warning and native-process tests | 13 passed |
| Native macOS final helper, then independent pbpaste after helper exit | Exact command matched; every original clipboard format restored and verified |
| Repository derived-artifact preflight | All seven guards passed; Mermaid inputs downloaded with declared size/hash verification |
| Ruff new production/test files | Check and format pass for all five files |
| Ruff modified existing files against untouched base | No new diagnostics; 29 versus 31 existing diagnostics |
| Diff whitespace and independent code review | Pass; review findings fixed and rechecked |

This is **657 distinct passing local tests**, plus the 13 Linux reruns. Initial
regressions failed before their fixes, including silent native failure, selected
STT warning/consent behavior, Qt self-readback and literal development commands.

Reproduction file set:

```text
Tests/Library/test_ingest_capabilities.py
Tests/Library/test_ingest_preflight.py
Tests/Library/test_library_ingest_state.py
Tests/UI/test_library_ingest_canvas.py
Tests/UI/test_library_ingest_structural.py
Tests/UI/test_library_ingest_inline_consent.py
Tests/UI/test_install_command_clipboard.py
Tests/Library/test_ingest_stt_warnings.py
Tests/Local_Ingestion/test_youtube_stt_selection.py
Tests/Utils/test_install_clipboard_native.py
Tests/UI/test_design_token_governance.py
```

Use `PYTHONPATH="$PWD" python -m pytest <files> -q --no-cov`. For the qualified
run, deselect only the exact baseline cases below. Include the mounted clipboard
test module when running design governance: its collection-time app import
avoids the pre-existing configuration ownership fixture/import-order error.

The Linux checks ran in the existing disposable Python 3.12 Linux image
`tldw-task14-4-54-backend:magic-20260914`, with source mounted read-only. Actual
Linux optional-dependency discovery advertised only missing transcribe.cpp in
that image, with no MLX hints. No native Fedora desktop was available. UI tests
mount the real buttons; native-process tests substitute only the external
clipboard library. URL tests run the real job option builder and both media
processors through an injected transcription runner; metadata/download and
model inference are substituted. No live YouTube download or model inference
is claimed.

## Baseline failures

The same six original affected test files on an untouched archive of the base
produced **19 failed, 626 passed**. All 19 fail with the existing
`RecoveryRequired: raw_source_selection_changed` fixture/config ownership error.
They also failed on the candidate and are the only cases excluded from the
qualified run. Two additional failures initially used a retired MLX warning as
a display fixture; those fixtures now use supported faster-whisper and pass.
The full suite was not run.

- `Tests/Library/test_library_ingest_state.py::test_a_page_source_offers_its_scope_settings_in_the_canvas_state`
- `Tests/UI/test_library_ingest_canvas.py::test_active_confirm_update_preserves_start_input_focus_cursor_and_scroll`
- `Tests/UI/test_library_ingest_canvas.py::test_idle_external_fence_preserves_focused_form_input`
- `Tests/UI/test_library_ingest_canvas.py::test_library_screen_multiline_prompt_typing_preserves_widget_and_focus`
- `Tests/UI/test_library_ingest_canvas.py::test_backend_switch_repaints_after_delayed_persistence_success`
- `Tests/UI/test_library_ingest_canvas.py::test_backend_switch_failure_restores_persisted_server_controls`
- `Tests/UI/test_library_ingest_canvas.py::test_rapid_backend_switch_keeps_latest_server_selection`
- `Tests/UI/test_library_ingest_canvas.py::test_library_screen_ingest_layout_contains_metadata_and_start_for_local_prompt[size0]`
- `Tests/UI/test_library_ingest_canvas.py::test_library_screen_ingest_layout_contains_metadata_and_start_for_local_prompt[size1]`
- `Tests/UI/test_library_ingest_structural.py::test_submit_brings_the_queue_heading_into_view`
- `Tests/UI/test_library_ingest_structural.py::test_the_fold_pays_for_itself_in_the_shipped_screen`
- `Tests/UI/test_library_ingest_structural.py::test_the_open_fold_survives_a_registry_tick_in_the_shipped_screen`
- `Tests/UI/test_library_ingest_structural.py::test_start_and_forecast_are_visible_without_scrolling_at_52_rows`
- `Tests/UI/test_library_ingest_inline_consent.py::test_enter_enter_two_press_flow_renders_confirm_then_submits`
- `Tests/UI/test_library_ingest_inline_consent.py::test_escape_declines_the_pending_confirm_and_stays`
- `Tests/UI/test_library_ingest_inline_consent.py::test_editing_the_path_resets_the_pending_confirm`
- `Tests/UI/test_library_ingest_inline_consent.py::test_enter_armed_consent_survives_the_start_click_and_submits`
- `Tests/UI/test_library_ingest_inline_consent.py::test_path_blur_alone_keeps_the_pending_confirm`
- `Tests/UI/test_library_ingest_inline_consent.py::test_browse_picking_a_new_file_disarms_the_pending_confirm`
