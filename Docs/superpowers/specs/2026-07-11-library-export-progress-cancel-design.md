# Library chatbook export — progress reporting + cancel (task 157)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-157 — "Chatbook export progress reporting and cancel".
**Builds on:** F4 bulk Library export (PR #597). v1 deliberately shipped with a
static `Exporting…` line and no cancel.

## Problem

The Library export canvas dispatches a threaded worker that calls
`local_chatbook_service.export_chatbook(...)` →
`ChatbookCreator.create_chatbook(...)`, a single synchronous shot with no
progress or cancel hook. A large export (many media items, a big zip) shows a
frozen `Exporting… (N items)` line with no way to interrupt it. We want a live,
per-phase item-count status line and a working Cancel control, reusing the
existing chatbook export path.

## Goal / Acceptance

- **AC1** — the export canvas shows real, moving progress while a large export
  runs (per-phase item/file counts, text, e.g. `Collecting media…  42/318`,
  `Packaging archive…  210/540 files`).
- **AC2** — the user can cancel an in-flight export; cancel is effective at the
  next item/file checkpoint, leaves no partial artifact at the destination, and
  returns the canvas to the export form with an `Export cancelled.` status.
- **AC3** — `ChatbookCreator` exposes progress + cancel hooks (shared, reusable
  by other chatbook UIs later; this task wires the UI only in the Library
  export canvas).

## Chosen approach

**Callback + cancel-check carried by the creator**, chosen over (2) a polled
shared-progress object (needs a UI timer, less precise, the creator must
checkpoint anyway) and (3) rewriting `create_chatbook` as a progress-yielding
generator (invasive rewrite of a ~1000-line method, breaks the return contract
other callers depend on). Approach 1 is the smallest, most testable change and
adds no new widget or layout.

## Components

### 1. Creator hooks (`tldw_chatbook/Chatbooks/chatbook_creator.py`)

New public types (defined in this module):

```python
@dataclass(frozen=True)
class ExportProgress:
    phase: str      # "conversations" | "notes" | "characters" | "media"
                    # | "prompts" | "relationships" | "packaging"
    current: int    # 1-based items completed in this phase
    total: int      # items in this phase (0 → indeterminate/skip)


class ChatbookExportCancelled(Exception):
    """Raised internally when cancel_check() returns True at a checkpoint."""
```

`create_chatbook(...)` gains two optional keyword params, both defaulting to
`None` (→ today's behavior byte-for-byte):

```python
progress_callback: Optional[Callable[[ExportProgress], None]] = None
cancel_check: Optional[Callable[[], bool]] = None
```

At the top of `create_chatbook`, store them on the instance
(`self._progress_callback`, `self._cancel_check`) and clear them in a `finally`,
so the `_collect_*` helpers and `_create_zip_archive` can call two private
helpers without threading params through six signatures:

- `self._emit_progress(phase, current, total)` — no-op if
  `self._progress_callback is None`; else calls it with an `ExportProgress`.
  Must **never** raise into the collection logic: wrap the callback in
  `try/except Exception` and log at debug on failure (a flaky UI marshaller
  must not abort a real export).
- `self._check_cancel()` — if `self._cancel_check` and it returns `True`, raise
  `ChatbookExportCancelled`.

Instances are single-use (`creator = ChatbookCreator(...)` is constructed per
export in `export_chatbook`), so instance state is safe.

**Emit / check points:**
- Each `_collect_*` phase (`_collect_conversations/notes/characters/media/prompts`):
  call `self._check_cancel()` then `self._emit_progress(phase, i+1, total)` once
  per item in its loop. `total` = `len(ids)` for that phase.
- `_discover_relationships`: one `self._check_cancel()` + a single
  `_emit_progress("relationships", 1, 1)` (fast phase, coarse is fine).
- `_create_zip_archive`: materialize `files = [p for p in work_dir.rglob('*') if p.is_file()]`
  once (so `total = len(files)` is known), then per file:
  `self._check_cancel()` → `zf.write(...)` → `self._emit_progress("packaging", i+1, total)`.

**Atomic finalize (cancel/crash safety):** `_create_zip_archive` writes to a
sibling temp path in the **destination directory**
(`output_path.parent / (output_path.name + ".partial")`) and, only after the
zip completes, `os.replace(temp, output_path)` (atomic within one filesystem;
same-dir keeps it atomic even when the destination is on a different volume than
`self.temp_dir`). The destination file is never created or clobbered until
success — so cancel/failure never leaves a truncated `.zip` there.

**Cancel cleanup:** wrap the body of `create_chatbook` so that on
`ChatbookExportCancelled` (and on any exception) it removes the temp `work_dir`
and the `.partial` file if present, then:
- for `ChatbookExportCancelled`: return `(False, "Export cancelled", {"cancelled": True, "missing_dependencies": [], "auto_included": []})`
  (a distinct, recognizable cancelled outcome — not a generic failure).
- for other exceptions: preserve today's failure return.

### 2. Service passthrough (`tldw_chatbook/Chatbooks/local_chatbook_service.py`)

`export_chatbook(self, request_data, *, progress_callback=None, cancel_check=None)`
forwards both to `creator.create_chatbook(...)`. The returned dict gains a
`"cancelled": bool` key (derived from `dependency_info.get("cancelled", False)`,
or from `success is False and message == "Export cancelled"`). Callables are not
serialized — this is the in-process local path only; the server service is
untouched (export runs locally).

### 3. Library UI wiring (`tldw_chatbook/UI/Screens/library_screen.py`)

**Cancel signal.** Add `self._library_export_cancel_event: threading.Event | None`
(created fresh in `handle_library_export_submit` at dispatch, captured by the
worker closure). Two triggers, deliberately different:
- **Cancel button** (new `#library-export-cancel` in the running-state canvas):
  `event.set()` and set `_library_export_status = "Cancelling…"`, then refresh
  the status label. It does **NOT** bump `_library_export_run_id` — the run is
  still the current, visible one until the worker reports back.
- **Navigate-away / `_reset_library_export_transient_state`:** set the outgoing
  event (if any) **and** bump `_library_export_run_id` (existing behavior), so
  the worker stops soon and its eventual cancelled result no-ops on the DOM.

`cancel_check` handed to the service = `self._library_export_cancel_event.is_set`.

**Progress.** The worker builds a `progress_callback` that marshals to the UI
thread, mirroring the existing `_marshal_library_export_*` pattern:
- **Throttle (worker-thread local state):** only marshal when the phase changed,
  or `time.monotonic() - last_emit >= 0.1s`, or `current == total`. Always flush
  on phase-change and the final tick so the line never freezes mid-count.
- Each marshal = `self.app.call_from_thread(self._apply_library_export_progress, run_id, phase, current, total)`
  wrapped in `try/except Exception` (Textual `NoApp` is not a `RuntimeError`),
  exactly like `_marshal_library_export_success`.
- `_apply_library_export_progress(run_id, phase, current, total)` runs on the UI
  thread and **first** checks `if run_id != self._library_export_run_id: return`
  (no stomping a superseded canvas), then sets
  `self._library_export_status = f"{LABEL[phase]}…  {current}/{total}"` and
  updates the single status `Static` **without a full recompose** (update the
  label widget directly; fall back to a targeted refresh if the widget lookup
  fails, guarded).
  `LABEL` maps phase→friendly text: conversations→"Collecting conversations",
  notes→"Collecting notes", characters→"Collecting characters",
  media→"Collecting media", prompts→"Collecting prompts",
  relationships→"Resolving links", packaging→"Packaging archive".

**Completion.** `_run_library_export_via_service` propagates the service's
`"cancelled"` flag into the `outcome` dict it returns (alongside
`success`/`path`/`dependency_info`/`registry_recorded`/`message`). Because a
cancelled export reports `success is False`, the existing "registry record only
on success" guard already skips the `create_chatbook` registry step — no
artifact is recorded for a cancelled run. `_run_library_export_worker` then
inspects `outcome["cancelled"]`:
- cancelled → `_marshal_library_export_cancelled(run_id)` →
  `_apply_library_export_cancelled(run_id)`: same staleness guard as the other
  apply handlers, then `_library_export_running = False`,
  `_library_export_status = "Export cancelled."`,
  `_update_library_export_canvas_after_run()` (returns to the form).
- success/failure paths unchanged.

The running-state canvas gains the Cancel button beside the status line; it is
present only while `_library_export_running` and shows "Cancelling…" once
pressed (button disabled after first press; `event.set()` is idempotent).

## Data flow

```
Cancel button ── event.set() ─────────────────────────────┐
                                                           ▼
handle_submit → worker(thread) → service.export_chatbook(progress_cb, cancel_check)
                                       └→ creator.create_chatbook
                                            ├ _collect_* : check_cancel + emit(phase,i,total)
                                            ├ _discover_relationships : check_cancel + emit
                                            └ _create_zip_archive : per-file check_cancel + emit
                                                 └ os.replace(.partial → output_path)  [success only]
emit ─(throttled)→ call_from_thread → _apply_library_export_progress(run_id,…)
                                          └ if run_id current: update status label
outcome.cancelled → _marshal_library_export_cancelled → _apply_… → "Export cancelled." + form
```

## Error handling

- Cancel is clean control flow (`ChatbookExportCancelled`), distinct from
  failure (which keeps `_apply_library_export_failure`).
- Temp `work_dir` **and** `.partial` removed on cancel, failure, and success.
- Progress-callback exceptions are swallowed (debug-logged) in the creator and
  in the marshaller — a UI hiccup never aborts a real export nor crashes the
  worker on teardown.
- `os.replace` failure (e.g. destination vanished) surfaces as a normal export
  failure with the temp `.partial` cleaned up.

## Testing

Favor the seams (fast, deterministic) over a heavy full-threaded UI export
(the project's UI harness lacks the app stylesheet and has a ~33-failure
baseline; `AppTest` is unavailable → `app.run_test()`):

- **Creator seam (`Tests/Chatbooks/`):**
  - progress: a recording `progress_callback` over a small multi-type export →
    assert every expected phase appears, `current` is monotonic 1..`total`
    within each phase, and the final tick per phase is `current == total`.
  - cancel: a `cancel_check` returning `True` after *k* calls → assert the
    outcome is the cancelled tuple (`success is False`, `"cancelled": True`),
    the temp `work_dir` is gone, **and no file exists at `output_path`**
    (atomic finalize) — including the case where cancel fires during packaging.
  - atomic finalize (success): a pre-existing file at `output_path` is only
    replaced after a successful zip; a forced failure mid-zip leaves the
    pre-existing file intact.
  - default-None: `create_chatbook` with no hooks behaves exactly as before
    (one existing happy-path assertion re-run).
- **Throttle helper** — extract the "should I emit now?" decision into a small
  pure function/class and unit-test phase-change / interval / final-tick.
- **Cancelled-apply staleness** — unit-test `_apply_library_export_cancelled`
  (and the progress apply) honor the `run_id` guard: a stale `run_id` leaves
  `_library_export_running`/status untouched; a current one updates them.
- **UI smoke (minimal, `app.run_test()`):** open the export canvas in a running
  state, assert the Cancel button exists and pressing it sets the event /
  flips the status to "Cancelling…". No full real export in-harness.

## Scope / non-goals

- Only the **Library export canvas** gets UI wiring. Other chatbook creation
  UIs (ChatbookCreationWindow, wizard) may adopt the shared hooks later — out of
  scope here (YAGNI).
- No determinate percentage/progress-bar widget (chosen: text item counts).
- A single in-progress giant file copy / `zf.write` is not mid-interrupted;
  cancel takes effect at the next item/file boundary (acceptable v1).
- The server-mode export path is unchanged (export runs locally).
