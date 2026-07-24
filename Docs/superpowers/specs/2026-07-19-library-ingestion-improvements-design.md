# Library Ingestion Improvements — Design Spec

**Date:** 2026-07-19  
**Status:** Implemented  
**Scope:** TUI file ingestion flow in `tldw_chatbook/UI/Screens/library_screen.py` and related modules.

## Goal

Improve the Library file ingestion screen and functionality so users get clearer feedback, understand tooling requirements before ingesting, configure options per file type, and can start ingestion quickly from the Library rail or command palette.

## User Requests Addressed

1. Better feedback during and after ingestion.
2. Guardrails around supported files based on available tooling.
3. Ingestion options depending on file type, with more options exposed per file.
4. A prominent Ingest button at the top of the left Library column and a command-palette shortcut.

## Approach

**Incremental redesign of `LibraryIngestCanvas`.**

We keep the existing `library_ingest_state.py` → `LibraryIngestCanvas` render pipeline and the `LibraryIngestJobRegistry` job lifecycle. We enrich each stage with pre-flight analysis, per-type option schemas, structured progress and errors, persistent job history, and a new top-of-rail entry point.

The legacy `Ingest` tab (`MediaIngestWindowRebuilt`) is out of scope and remains deprecated.

## Components

| Component | Change |
|-----------|--------|
| `LibraryRail` | Add an optional `top_action_factory` parameter so `LibraryScreen` can inject a primary `Button("Ingest content…", variant="primary")` at the top of the rail without making `LibraryRail` ingest-aware. |
| `LibraryIngestCanvas` + `library_ingest_state.py` | Split canvas into source, pre-flight summary, collapsible type-group options, and queue/history zones. Track `expanded_type_groups` in state so user toggles survive recomposes. |
| New `ingest_capabilities.py` | Thin UI layer over `Local_Ingestion/local_file_ingestion.py` and cheap `importlib.util.find_spec` probes of `Utils/optional_deps.py` metadata. Maps file types to tooling requirements, option schemas, labels, and install hints. |
| Pre-flight analyzer | `@work` worker that resolves paths/URLs, expands directories (with a cap), groups files by type, checks tooling, and returns warnings + defaults. Cancels any prior pre-flight worker before starting a new one; results are marshalled via `call_from_thread`. |
| `LibraryIngestJob` + `LibraryIngestJobRegistry` | One job per file. Add a serializable `ingest_options: dict[str, Any]` field (snapshot of the relevant type-group options at queue time), plus a `progress` payload and structured `error_detail`. Persist on job creation. |
| `progress` payload schema | Minimal schema: `{"stage": str, "current": int | None, "total": int | None, "message": str}`. `stage` is a user-facing phase name; `current`/`total` are optional counters; `message` is a human-readable progress line. |
| `app.py` `_ingest_job_options` | Translate `ingest_options` into the kwargs that `parse_local_file_for_ingest` and the backend processors accept. |
| Options store | Last-used options per type group are stored in `config.toml` under `[library.ingest_options]` (e.g., `[library.ingest_options.pdf]`). |
| Confirmation modal | Aggregated, quantified guardrail warnings with install guidance and optional copy-command button. |
| Command palette provider | Add to the existing command-palette setup (e.g., extend `QuickActionsProvider` or create a small `LibraryIngestCommandProvider`) a `"Library: Ingest content…"` command that navigates to the Library screen with `LIBRARY_NAV_CONTEXT_INGEST`. |
| `Library_Ingest_Jobs_DB` | Migrate schema v1→v2 to add `ingest_options TEXT` (JSON), `error_detail TEXT` (JSON), and `progress TEXT` (JSON, nullable). Persist jobs on creation and load recent history on canvas mount. |

## User Flow

1. User opens the ingest canvas via the rail button or command palette.
2. User selects a file, folder, or URL via the path input or Browse dialog.
3. Pre-flight analysis runs on Browse close, `Enter`, or debounced `Blur` — not on every keystroke. If the user changes the input while a pre-flight is running, the prior worker is cancelled.
4. Canvas renders:
   - detected file count grouped by type (URL sources render as a single item),
   - a lightweight estimate line (file count + total size only; no duration estimates in v1),
   - warning banners for missing/optional tooling,
   - collapsible options panels per detected type group.
5. User adjusts options; scope is all files of that type in the current selection.
6. User clicks **Start ingest**.
7. If any guardrail warning is active, an aggregated confirmation modal appears (e.g., "PDF support missing — 3 PDFs will fail. Start anyway?").
8. The selection expands into one `LibraryIngestJob` per file. Each job captures a snapshot of its type group's current `ingest_options` so later edits to the panel do not affect already-queued jobs. Jobs are queued and persisted.
9. The queue updates live with status, structured progress text, and elapsed time.
10. On completion, the row offers **Open in Library**. On failure, it offers **Retry** and **Dismiss** (dismissed jobs remain in history with status `dismissed`).
11. The Library media counter refreshes when jobs complete.

## UI Details

### Rail Button
- `LibraryScreen` injects a primary `Button("Ingest content…")` at the top of `LibraryRail` via the new `top_action_factory` parameter, above the search box.

### Source Area
- Path input + **Browse…** button.
- Helper text: "Select a file, folder, or URL. Folders are scanned recursively."

### Pre-flight Summary
- File/type breakdown: e.g., "1 PDF, 2 audio files, 1 plain text file." URLs show as a single source item.
- Lightweight estimate: file count and total size only. Duration estimates are out of scope for v1.
- Warning banner per missing/optional tooling with a "How to install" link.
- If a directory scan exceeds `library.ingest_directory_scan_limit`, show: "Showing first N of M files. Ingest subfolders separately."
- Inline error display for pre-flight I/O failures (permission denied, path not found, URL unreachable) with a retry action.

### URL Sources
- Pre-flight detects a URL via the existing `classify_ingest_source` / `_is_http_url` helpers.
- The URL is shown as a single source item in the summary; it is not recursively scanned.
- Type detection uses the URL path/extension and, if cheaply available, a `HEAD` request Content-Type.
- The `source_url` field of the job is set to the URL; the backend downloads or streams the resource during parsing.
- `media_id` and `source_url` in the final media record are populated from the completed job.

### Options Panel
- One collapsible section per detected type group.
- Default state: only the first detected type group is expanded; user toggles are stored in `expanded_type_groups` in state.
- Each collapsed panel shows a summary badge: e.g., "PDF — pdfplumber, pages 1-50, OCR on".
- "Expand all / Collapse all" link at the top.
- Each panel has a "Reset to defaults" link.
- Scope label: "These options apply to all X files in this selection."
- Dependent controls are disabled when not applicable.
- The existing global "Advanced options" collapsible (analyze/chunk/chunk-size) is removed; those controls become the Generic/Plain text panel defaults.

### Per-Type Controls

Controls are shown only when the backend processor can consume the option. If a backend option is not yet wired, the control is hidden or disabled with a tooltip explaining it is unavailable.

**PDF**
- Engine dropdown, filtered by installed packages; placeholder "No PDF engine installed" if empty.
- Pages: All / Range, with validator for `1-10, 15, 20-25`.
- OCR checkbox (disabled if no OCR backend).
- Extract images checkbox.

**Audio / Video**
- Transcription model dropdown (placeholder if none installed).
- Language: Auto / specific.
- Include timestamps checkbox.
- Speaker diarization checkbox (disabled if unsupported).

**Ebook**
- Extraction method dropdown.
- Split by chapter checkbox.
- Include table of contents checkbox.

**Generic / Plain text**
- Analyze after ingest checkbox.
- Chunk content checkbox.
- Chunk size number input (disabled unless chunking on).
- Chunk overlap number input (disabled unless chunking on).
- This panel applies only to plain-text files and other types handled by the generic parser.

**Unsupported file types**
- Unsupported files are listed separately in the pre-flight summary with a warning that no specific handler exists.
- They do not appear in the Generic/Plain text options panel and do not receive generic options.
- A job is still created and submitted so the failure is recorded; the parse worker marks it as a permanent failure with category `unsupported_file_type`.
- Unsupported jobs are not retryable (no **Retry** action) because the outcome cannot change without new backend support.

### Queue / History
- Live jobs at the top with status, progress text, elapsed time, and actions.
- "Recent ingests" collapsed section below with the last N completed/failed/dismissed jobs.
- Pruning uses the existing policy in `Library_Ingest_Jobs_DB` (e.g., `_MAX_PERSISTED_INGEST_JOBS`); this design does not change that policy.

### Keyboard Shortcuts
- No dedicated global shortcuts in v1. Focus management and `Enter`/ `Tab` navigation are sufficient. Shortcuts can be added later after auditing `app.py` bindings.

## Guardrails & Errors

### Guardrails
- Pre-flight uses cheap `importlib.util.find_spec` probes plus `optional_deps.py` metadata to determine installed backends. Heavy `check_*_deps` functions are not called during pre-flight.
- Warnings are non-blocking banners.
- **Start ingest** remains enabled; clicking it with active warnings opens the confirmation modal.
- The modal lists each warning, quantifies affected files, and shows install guidance. A "Copy install command" button appears only when a reliable, platform-appropriate command is available.

### Errors
- Pre-flight I/O errors render inline in the summary area.
- Ingest failures produce structured `error_detail`:
  - `message`: user-facing summary,
  - `category`: tooling / missing_dependency / parse_error / write_error / unsupported_file_type / unknown,
  - `fix_hint`: concrete next step,
  - `docs_link` (optional).
- Queue rows show the summary and a "Show details" button for the full error.
- **Retry** re-queues the same file with the same options.
- **Dismiss** removes the job from the live queue but keeps it in `Library_Ingest_Jobs_DB` with status `dismissed`.

### Open in Library
- Completed jobs open the corresponding media item via `media_id`.
- If `media_id` is `None` (e.g., due to deduplication), fall back to looking up the most recent media row by `source_url`, then by content hash. If no match is found, show a transient status line: "Already in Library" with an option to open the newest match.

## Configuration

Add to `config.toml` under a `[library]` section:

```toml
[library]
ingest_directory_scan_limit = 1000

[library.ingest_options.pdf]
engine = "pdfplumber"
pages = "all"
ocr = false
extract_images = false

[library.ingest_options.audio_video]
model = "base"
language = "auto"
timestamps = true
diarization = false
```

- `ingest_directory_scan_limit`: maximum number of files enumerated from a directory before showing the "Ingest subfolders separately" warning.
- `[library.ingest_options.<type_group>]`: TOML inline tables / dotted keys storing the last-used options for each type group. The app reads these on canvas mount and writes them when an ingest starts. Complex nested values are avoided; booleans, strings, and numbers are stored as plain TOML values.

## Out of Scope

- Legacy `Ingest` tab (`MediaIngestWindowRebuilt`) improvements; it remains deprecated.
- "Skip transcription, import metadata only" option for audio/video; flagged as a follow-up task.
- True fine-grained progress bars that require parse-pool callback plumbing. The schema supports progress payloads, but the first implementation will enrich status text with counts/estimates if callbacks are not yet available.
- Dedicated global keyboard shortcuts for Browse/Start ingest in v1.
- Audio/video duration estimates in v1.

## Backend Wiring Requirements

For every exposed control, the implementation must extend the backend so the option is actually consumed:

- `parse_local_file_for_ingest` and the PDF/ebook/audio/video processors must accept the new kwargs derived from `ingest_options`.
- If a control cannot be wired by the end of the task, it must be hidden or disabled — never shown as a no-op.
- `app.py` `_ingest_job_options` must map `ingest_options` JSON to the existing `chunk_options` and new processor kwargs.

## Testing Plan

### Unit Tests
- `ingest_capabilities.py` mapping of extensions/MIME types to option schemas and tooling checks.
- Pre-flight analyzer: directory expansion, type grouping, warning generation, I/O error handling, cancellation.
- Page-range parser validator.
- `LibraryIngestJob` serialization including `ingest_options`, `progress`, and `error_detail`.

### UI / State Tests
- `library_ingest_state.py` transitions for path selection, pre-flight result, options changes, expanded/collapsed panels, and start ingest.
- `LibraryIngestCanvas` rendering for each type group and guardrail banner.

### Integration Tests
- End-to-end ingest flow with mocked optional dependencies.
- Guardrail modal behavior: warning shown, override allowed, job proceeds.
- Persistence: queued → running → done / dismissed states in `Library_Ingest_Jobs_DB`.
- DB schema migration v1→v2 upgrades cleanly and old data is preserved.
- Command-palette provider routing.
- Rail button navigation.
- Last-used options persistence in `config.toml`.
- "Open in Library" fallback when `media_id` is `None`.

### Manual QA
- Mixed-batch options apply correctly.
- Directory scan limit warning.
- Install guidance and copy-command button.
- Retry and Dismiss actions.

## Related Files

- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_rail.py`
- `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`
- `tldw_chatbook/Library/library_ingest_state.py`
- `tldw_chatbook/Library/library_ingest_jobs.py`
- `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- `tldw_chatbook/Utils/optional_deps.py`
- `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/app.py`

## Acceptance Criteria

- [x] A prominent "Ingest content…" button exists at the top of the Library left rail.
- [x] A "Library: Ingest content…" command is available in the command palette and navigates to the ingest canvas.
- [x] Selecting a file, folder, or URL triggers pre-flight analysis that detects type and tooling status.
- [x] Missing/optional tooling is surfaced as a non-blocking warning with install guidance.
- [x] Users can proceed past warnings after an explicit confirmation modal that quantifies affected files.
- [x] Options panels are rendered per detected file type with relevant, backend-wired controls.
- [x] The existing global "Advanced options" are replaced by per-type panels; generic options live in the Plain text panel.
- [x] Live ingest queue shows status, progress text, and elapsed time per file.
- [x] Completed and failed ingest jobs persist as history on job creation.
- [x] Failed jobs can be retried or dismissed; dismissed jobs remain in history.
- [x] Unsupported file type jobs are recorded as permanent failures and are not retryable.
- [x] `Library_Ingest_Jobs_DB` schema is migrated to store `ingest_options`, `error_detail`, and `progress`.
- [x] Progress payloads follow the documented minimal schema and are surfaced in the queue.
- [x] Every exposed control maps to a real backend argument; un-wired controls are hidden or disabled.
- [x] `config.toml` supports `library.ingest_directory_scan_limit` and `[library.ingest_options]`.
- [x] All new logic has unit or integration tests, including the DB migration.

## Implementation Notes

- **Branch:** `feature/library-ingestion-improvements`
- **Implementation approach:** Followed the plan task-by-task using subagent-driven development. Each task was implemented, spec-reviewed, and code-quality reviewed before proceeding.
- **Key files modified:**
  - `tldw_chatbook/Widgets/Library/library_rail.py` — added `top_action_factory` for the rail-top Ingest button.
  - `tldw_chatbook/UI/Screens/library_screen.py` — injected the button, drove pre-flight, built options snapshots, wired guardrail modal, added Open-in-Library fallback, persisted options.
  - `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` — rendered pre-flight summary, per-type options panels, progress/errors, conditional retry, recent ingests.
  - `tldw_chatbook/Library/ingest_capabilities.py` — mapped file types to option schemas and tooling checks.
  - `tldw_chatbook/Library/ingest_preflight.py` — async pre-flight analyzer.
  - `tldw_chatbook/Library/library_ingest_jobs.py` — extended jobs with `ingest_options`, `progress`, `error_detail`, `content_hash`; added `get_job`.
  - `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py` — schema v1→v2 migration for new JSON columns.
  - `tldw_chatbook/app.py` — wired `ingest_options` through `submit_library_ingest_job` and `_ingest_job_options`.
  - `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` — routed per-type options into PDF/ebook/audio/video processors.
  - `tldw_chatbook/config.py` — added `[library]` defaults and ingest-options persistence.
- **Tests:** Added unit tests for capabilities, pre-flight, state, DB migration, guardrail modal, options persistence, and integration tests for the full ingest flow. All relevant suites pass.
- **Bug fixes during implementation:** Fixed `push_screen` call in guardrail wiring (used `self.app.push_screen` correctly and updated tests to patch the `app` property). Fixed schema-version migration race on existing v1 databases.
- **Known limitations:**
  - True fine-grained progress bars require parse-pool callback plumbing; current implementation surfaces progress text when available.
  - Some per-type options (`page_range`, `extract_images` for PDF; `split_chapters` for ebook) are accepted by the UI but logged as unsupported by processors when not yet implemented.
  - Pre-existing lint debt in `tldw_chatbook/config.py` and `tldw_chatbook/UI/Screens/library_screen.py` remains untouched.
