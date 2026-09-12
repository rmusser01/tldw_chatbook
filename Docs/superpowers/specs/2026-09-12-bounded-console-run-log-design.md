# Bounded Console run-log viewing

## Outcome

TASK-18601's remaining acceptance criterion is a viewer that does not retain all steps at once. The indexed database storage already shipped. The actual full-log viewer reads the lossless filesystem segment log, so this change pages that source and preserves primary/child filtering and access to stored content.

ADR required: no new ADR
ADR path: backlog/decisions/082-console-per-chat-private-scratch-space.md
Reason: implement the existing process-local scratch-read authority and segmented-log contract in `Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md`; no storage format, retention, provider, or permission change.

## Reader contract

Add `Agents/run_log_paging.py`, reusing the byte-framing rules and metadata from `run_log_format.py`. A page cursor contains only a numeric segment identifier, the record header's byte offset, and the content byte offset. It contains no path and is scoped to the bridge loader that produced it. Reject invalid types and negative positions before opening anything.

Each production page holds at most 100 record fragments and 256,000 content bytes. Each call reads at most 4,500,000 bytes plus a fixed bounded header/UTF-8 lookahead allowance. Headers are bounded at 16,384 bytes; malformed or oversized headers produce an explicit page diagnostic, never silent apparent completion. A record larger than the content budget continues over subsequent pages; the configured writer record cap cannot enlarge viewer memory.

Preserve segment-number order without relying on MANIFEST. Existing discovery tolerates missing segment numbers, so missing the next integer does not mean EOF if a later segment exists. Stream directory entries when selecting a later segment rather than retaining log contents. A valid writer orders records globally across its segments; record fragments keep the original number and metadata.

Before yielding any fragment, verify the complete declared record and terminating newline exist, using file size and a bounded terminator read. An incomplete trailing record remains unread until complete. Do not expose a partial body merely because the requested first fragment fits. Torn/malformed records resynchronize using the existing line-anchored format, with all scanning charged to the budget.

UTF-8 fragment boundaries preserve complete code points. Concatenating the fragments of valid stored content reproduces it exactly without replacement characters or duplication. A scan-limited page can contain no matching records and still return a continuation; that is distinct from EOF. Child filtering reads metadata and seeks past nonmatching bodies instead of allocating them. No unbounded `read()`, `read_bytes()`, or whole-record decoding belongs in the paged path.

The page format keeps the existing record headings. Fragmented records show a concise continuation indication; writer truncation remains a separate truthful notice. No log body, path, credential, or malformed header content goes into diagnostic logging.

## Bridge and availability

`ConsoleAgentBridge.load_run_log_page(run_id, *, cursor=None)` returns a page or `None` for absent/revoked/unreadable authority. Resolve the requested child to its primary owner exactly as today. Every page read holds `authority.access_scope()`; persisted IDs and cursor values never recreate scratch authority. A primary page contains its whole run tree; a child page contains only that child's records.

Replace the production viewer's full-text load with this page API. Retain the old full-text method only if a concrete non-viewer consumer needs it; do not keep a second product viewer path. Existing search/tool APIs remain compatible and outside this change.

Availability must also avoid materializing all records. Use a bounded streaming metadata scan, retaining no bodies. Any potentially long availability probe runs in an existing Console worker; cache publication is keyed to the exact run and ignored when the target changes. A negative first scan is not proof that a child's later records do not exist. Continue scanning in the worker until a match or EOF, with cancellation/authority checks between bounded chunks. Add an optional keyword-only `cancelled: Callable[[], bool] | None = None` to `run_log_available`; check it before each metadata chunk/lease and return False when cancelled. The Console supplies the captured worker cancellation state; existing callers retain the bool contract. The UI remains responsive and only shows the action after a confirmed match, preserving its existing no-dangling-button contract. A negative availability result may be retried after one monotonic second on existing expanded-section ticks, so a later first record becomes visible for the same run. Do not overlap probes: mark the exact generation pending until completion, and start the retry deadline only after a negative result. Keep positive caching and collapsed steady-state no-I/O behavior; add no timer.

## Viewer behavior

The initial page and every navigation page load in a worker. The modal owns one current page, a bounded history of at most 256 page-start cursors, and loading/error state. Replacing the page replaces the TextArea document; it does not append to it or retain previous bodies. Previous reloads a cursor. First returns to the start when older history has been dropped. Next advances only after a successful load.

Use the incumbent Console modal: title, read-only scrollable log, concise page/range or continuation status, Previous/Next/First/Close buttons. Disable unavailable navigation, keep Close usable during loading, and preserve Escape-safe dismissal. No new keyboard shortcut, export feature, or global setting is needed. Use existing semantic design tokens and the bundled stylesheet.

Only one navigation load is admitted at a time. A late result after dismissal, target replacement, or generation change cannot update the UI. The modal accepts an optional `target_is_current: Callable[[], bool]` predicate, defaulting to true for standalone callers. The Console supplies its exact captured bridge/run predicate; evaluate it only on the UI thread before navigation admission and before result/error publication. Close remains usable after target replacement. A load failure retains the last good page and displays that the log is no longer available; no stale authority is retried implicitly. Loading an empty scan page with a continuation displays an honest continue message rather than “no log.”

## Verification

- Real temporary segment files cover ordered multi-page reads, gaps in segment numbering, primary/child filtering, malformed headers, incomplete/torn records, writer truncation, empty scan pages, and cursor rejection.
- A stored multi-megabyte record with multibyte text is reconstructed exactly across bounded fragments. Instrument read sizes/bytes and retained page sizes so the test would reject whole-file or whole-record loading.
- Bridge tests use real scratch authorities and revoke them between pages; no cross-child data is displayed, and availability skips irrelevant bodies.
- Mounted modal tests verify Next/Previous/First content replacement, off-thread loading, single admission, loading/error/close behavior, stale-result rejection, and bounded cursor/body retention. Inspect painted controls at wide and narrow supported Console sizes.
- Run the affected reader/bridge/modal/controller tests and scoped lint/format, CSS reproduction, token checks and whitespace checks. Do not run the full repository suite or raise existing ratchets.

## Design choices

Page filesystem segments instead of database step rows because only segments contain the full stored tool/model content. Split oversized records instead of silently truncating them because this viewer is how users reach content omitted from the transcript. Keep a small cursor history instead of caching page bodies because reload is cheap compared with retaining long runs. Preserve sparse segment discovery rather than assuming contiguous files because the accepted reader contract uses discovered segments, not a manifest.
