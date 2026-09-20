# Library artifacts — implementation evidence

Implemented on `codex/library-artifacts-design`, based on the fetched dev commit
`ebee42fab8`. ADR-172 governs the completed browse/navigation boundary. The
existing schema, registry inventory, report generators, ZIP manager, and
application-owned share service remain the owners of their data and actions.

## Delivered behavior

- Library has All artifacts, Chatbooks, and Reports. Reports starts on All
  reports; Kept reads independent copies without requiring Subscriptions.
- Namespaced identities preserve live/kept copies and imported-ID collisions.
  Snapshot reads produce bounded metadata pages and exact target location.
  Keep rejects incompatible saved snapshots before parent/script mutation.
- The existing adaptive reader supplies panes, local search, sorting, paging,
  Preview/Details, keyboard focus, and per-view restoration. Full stored Console
  responses are readable; truncated saves and missing ZIPs are labeled.
- The existing pack manager remains linked. Sharing retains multi-selection,
  app ownership, URLs, Manage/Stop across Library views, and local/LAN consent.
- Ctrl+6, the command palette, legacy routes, configured Artifacts defaults,
  and exact Console-save targets enter Library. The permanent Artifacts tab
  was removed after capability checks and native verification.

## Targeted evidence

These are independent focused runs, with some intentional overlap; counts are
not a single full-suite total.

| Area | Result |
| --- | --- |
| Report state/catalog and config import boundary | 29 passed |
| Service Keep conflict, rollback, snapshots, borrowed transactions | 43 passed |
| Complete registry, merged inventory, exact source, saved body | 60 passed |
| Reports canvas plus Library rail state | 45 passed |
| Final reader interactions, stale actions, scroll, Scripts deletion | 16 passed |
| Full Library Chatbooks/manager/actions/sharing integration | 7 passed |
| Share adapter and real share-owner lifecycle | 25 passed |
| Exact navigation, recovery, query and modal resume | 21 passed |
| Final legacy route, default, palette, shortcut and tab cutover | 9 passed |
| Design tokens, CSS build integrity and reproducibility | 42 passed |
| Final CSS changes: token references, spacing and reproducibility | 3 passed |
| New controller budgets and complete registration | 5 passed |

New reader files pass Ruff and formatting. The 49 changed Python files have no
Ruff diagnostics on added lines; existing diagnostics were not mass-reformatted.
`git diff --check` is clean. New controller budgets are pinned at their actual
sizes. The artifact stylesheet is a separate token-backed source module and the
bundles were rebuilt; its sheet-size check passes.

## Native application verification

A real `TldwCli` process ran in a private tmux TTY with `LinuxDriver` and
`isatty=True`, using production CSS and a temporary profile. The final capture
set is `/private/tmp/library-artifacts-native/final4/`; `proof.json` records:

- Native Enter, Down, Escape, then Down through the terminal driver correctly
  enters the reader, scrolls without changing selection, returns to Items, and
  changes selection there.
- An artifact-only profile graduates to the full Library rail.
- Reports uses bounded pages (20 of 27 copies in the final fixture).
- A kept report remains readable after deleting its source Watchlist.
- A Console response longer than 1,000 characters is read from its stored body.
- A real exported ZIP is selected through the dialog; the actual localhost
  child server returns HTTP 200 with the pack. Sharing survives Notes navigation;
  Stop clears the share, and the app exits afterward.
- The existing manager exposes Create Local Pack, Import Local Pack, and Browse
  Templates. The compatibility route returns to Library. The duplicate permanent
  Artifacts navigation button is absent without shifting the other shortcuts.

Captures cover 160×50 dark, 100×40 light, 64×30 dark, and 50×25 light. Visual
inspection corrected unreadable heading ink and excess sharing-strip spacing;
final captures show token-backed headings and a compact, wrapping URL strip.
No normal user profile or external service was used for this verification.

## Review corrections

Independent reviews and real mounted regressions caught and fixed stale detail
Retry, search-field confusion on resume, renamed-source re-Keep selection,
missing share URLs, retired queued share starts, exact handoff query mismatch,
Scripts deletion leaving a stale preview, and stale action revalidation. Real
view switching exposed asynchronous Markdown mount/scroll ordering; the reader
now waits for body mount before restoring its position. This incident is recorded
in `backlog/docs/lessons-live-verification.md`.

## Known verification limits

No full suite was requested or run. Existing direct old-screen/casting/demo
checks fail at `RecoveryRequired('raw_source_selection_changed')` during profile
initialization, before exercising the new Library UI. Specifically, the existing
kept-modal/export run had 30 passes and 4 casting failures; the report-view/export/
demo run had 23 passes and 12 demo initialization failures. The new mounted cases
use the repository's private-profile process harness and pass. Live demo network
fetching, paid LLM generation, and physical audio output were not exercised.

Existing repository-wide governance also remains red for raw dimension literals
in `_workflows.tcss` and previously exceeded Chat/Library screen size budgets.
Library's base file was already 35,793 lines against a 33,204-line ceiling before
this feature; this change keeps its new behavior in focused controllers and adds
thin screen wiring. Those unrelated baseline limits were not raised or hidden.

Implementation commits separate source contracts, Keep integrity, sharing,
compatibility routing, focused tests, and final reader wiring. Stage 2 groundwork
ran alongside Stage 1 verification once contracts were stable; permanent
navigation removal waited for the native parity check. The old Artifacts class
remains only for its existing static Console-launch payload helper and legacy
implementation tests; it is no longer registered as a browse destination.
