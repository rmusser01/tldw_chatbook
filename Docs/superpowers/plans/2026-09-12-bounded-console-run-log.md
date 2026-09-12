# Bounded Console run-log implementation

> **For agentic workers:** Use subagent-driven-development to implement and review each task.

**Goal:** Complete TASK-18601 with bounded filesystem-log reading and a paged Console viewer.

**Architecture:** Add a byte-framed streaming page reader, preserve the bridge's scratch authority and child ownership, and replace the modal's whole-log document with one worker-loaded page. Availability scans also move off the UI thread.

**Tech Stack:** Python, pathlib/stdlib binary I/O, pytest, Textual.

**Spec:** `Docs/superpowers/specs/2026-09-12-bounded-console-run-log-design.md`.

ADR required: no new ADR
ADR path: backlog/decisions/082-console-per-chat-private-scratch-space.md
Reason: bounded read/presentation implementation of the existing byte-framed segmented log and scratch-lease contract; no migration or retention change.

## Global Constraints

- Work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr` on `codex/agent-orchestration-remaining`; main checkout stays untouched.
- Use `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`. Targeted tests only; all product imports/probes run under pytest isolation.
- Production limits: 100 fragments/page, 256,000 content bytes/page, 4,500,000 scanned bytes/call, 16,384 bytes/header, 256 previous cursors/modal. Constants, not new user settings.
- Preserve all stored valid body text through fragments, child filtering, segment ordering, malformed/incomplete-record handling, current scratch leases, and cancellation. No body retention across pages.
- No dependencies, schema changes, provider calls, external network, shared venv edits, or guard/ratchet increases. Preserve unrelated formatter debt.
- Follow DESIGN.md, existing Console modal controls, and semantic CSS tokens; regenerate bundled CSS from its source when changed.
- Root owns staging/commits and Backlog status. Workers leave changes unstaged; no worker subagents or reviewer dispatch.

### Task 1: Stream bounded log pages and preserve bridge authority

**Files:**
- Create `tldw_chatbook/Agents/run_log_paging.py` and `Tests/Agents/test_run_log_paging.py`.
- Modify `tldw_chatbook/Agents/run_log_format.py` only for a small shared header decoder if needed; preserve existing codec API/behavior.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py` page/availability seams.
- Modify `Tests/Chat/test_console_agent_tool_result_cap.py` for page and authority assertions.
- Read `run_log.py`, `run_log_search.py`, existing codec/search tests, and the spec.

**Interfaces:**

```python
@dataclass(frozen=True)
class RunLogPageCursor:
    segment_index: int
    record_offset: int
    content_offset: int = 0

@dataclass(frozen=True)
class RunLogRecordSlice:
    record: RunLogRecord
    content_offset: int
    stored_content_bytes: int
    slice_complete: bool

@dataclass(frozen=True)
class RunLogPage:
    slices: tuple[RunLogRecordSlice, ...]
    start_cursor: RunLogPageCursor
    next_cursor: RunLogPageCursor | None
    scanned_bytes: int
    diagnostics: tuple[str, ...] = ()

def load_record_page(log_dir: Path, *, cursor=None, run_id=None,
                     max_records=100, max_content_bytes=256_000,
                     max_scan_bytes=4_500_000) -> RunLogPage: ...
def format_record_page(page: RunLogPage) -> str: ...
```

The two ellipses above designate signatures, not implementation placeholders: implement the byte-framing algorithm below. Bridge `load_run_log_page(self, run_id, *, cursor=None) -> RunLogPage | None` captures ownership and holds a lease for each call. Task 2 consumes this API and the immutable page types. `run_log_available` retains its bool contract but uses streaming metadata rather than all-record allocation.

- [x] Read the task and spec; run existing affected reader/bridge tests as a baseline before editing their behavior. Add tests over real files encoded with `encode_record` for at least two pages, a segment-number gap, a child after a large unrelated body, and a single multimegabyte UTF-8 body. Pin reconstruction and budgets:

```python
assert b"".join(part.record.content.encode("utf-8") for part in fragments) == original
assert all(sum(len(s.record.content.encode("utf-8")) for s in p.slices) <= 256_000 for p in pages)
assert all(len(p.slices) <= 100 for p in pages)
```

- [x] Observe intended failures for the new missing page API before implementation. Add a recording binary-file wrapper that rejects unbounded `read` and records total read size; this must fail if a reader materializes an entire segment or oversized record before slicing.
- [x] Implement cursor validation, numeric segment discovery, bounded header reads, framing/terminator checks, seeking over unrelated bodies, fragment continuation and UTF-8 boundary preservation. Count scanning before returning; a budget stop yields an advancing continuation even with no matching slices. Resynchronize malformed/torn headers within the scan allowance and retain an explicit bounded diagnostic code; never log their text.
- [x] Share only the codec header parsing needed to avoid two conflicting framing implementations. Keep existing `iter_records` tests green; do not refactor search, writer, or manifest behavior.
- [x] Add the authority-scoped bridge page method using `_owning_run_id_for_log` and `_run_log_authority_for` and holding `access_scope()` per call. Add a metadata-only availability scan that retains no bodies, checking lease/cancellation between chunks where available. Task 2 calls potentially long scans only on workers.
- [x] Verify absent/revoked authority, revocation between pages, primary versus exact child filtering, large child bodies and skipped sibling reads. Keep the whole-text method temporarily until Task 2 migrates the production viewer and surveys its remaining concrete consumers.
- [x] Run the new paging module, existing format/search tests, and affected bridge cap/authority tests; record counts, read-budget evidence, scoped static checks and `git diff --check`. Leave changes unstaged and write report. Root commits and independently reviews before Task 2.

### Task 2: Page the Console modal and bound availability work

**Files:**
- Modify `tldw_chatbook/UI/Console_Modules/agent.py`.
- Modify `tldw_chatbook/Widgets/Console/console_run_log_modal.py`.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py` only to remove the obsolete full-viewer seam if concrete consumers are migrated.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` if pager layout requires it.
- Modify `Tests/UI/test_console_agent_rail.py`, `Tests/UI/test_console_modal_dismissal.py`; add focused modal tests in `Tests/UI/test_console_run_log_paging.py` if needed.
- Update TASK-18601 and its user documentation, preserving completed database work.

**Interfaces:**

```python
RunLogPageLoader = Callable[[RunLogPageCursor | None], RunLogPage | None]
# Modal constructor:
# ConsoleRunLogModal(run_id=..., first_page=page, page_loader=loader)
```

The loader closes over the exact bridge and run ID; every invocation repeats authority validation. The controller passes it through a thread worker and pushes the modal on the UI thread. The modal stores only current page, bounded cursors, and generation/loading/error state.

- [ ] Add mounted tests with two distinct page canaries and a gated loader. Assert loader thread differs from the UI thread, Next replaces old content, Previous reloads the old cursor, a second click cannot launch an overlapping request, and Close while gated prevents late publication. Assert the previous body is not retained in the modal/page state.
- [ ] Change `_load_console_agent_run_log` to load the first page directly, avoiding a second full availability scan. Keep initial empty/absent semantics and allow an empty continuable scan page to open honestly.
- [ ] Move cache-miss availability work from `_console_agent_full_log_available` onto a worker, preserving target-keyed caching and collapsed-section no-I/O behavior. Publish only if the target still matches; update the actual affordance through its existing UI callback. Add a slow probe regression proving the UI remains responsive and stale results cannot reveal another run's action.
- [ ] Replace `log_text` state with `first_page/page_loader`, Previous/Next/First/Close controls and one replaceable TextArea. Bound history to 256 cursors. Keep Close usable while loading; empty continuation and lost-authority errors have honest text. Do not clear the last good page on navigation failure.
- [ ] Reuse SafeModalDismissMixin and existing modal tokens/layout. Inspect wide and narrow mounted layouts in one batch, correct observed defects, then confirm once. Load the Impeccable craft-floor guidance before UI editing; existing product and DESIGN.md are the visual authority.
- [ ] Survey all `load_run_log_text` callers before migration/removal; keep genuine non-viewer API compatibility if needed and document that only the product viewer is qualified as bounded. Do not silently truncate a compatibility method to fit page limits.
- [ ] Run affected reader/bridge tests plus rail/modal/dismissal tests. Verify rendered pager controls, replacement, off-thread reads, cancellation/error paths and history cap. Run scoped static checks, CSS reproduction/token checks and whitespace checks without raising ratchets.
- [ ] Add implementation notes to TASK-18601, including completed DB work, actual segment-source paging, exact tests and limitations. Leave task In Progress and changes unstaged until root's independent task and combined plan review. Root closes the task via CLI only when all criteria are verified.
