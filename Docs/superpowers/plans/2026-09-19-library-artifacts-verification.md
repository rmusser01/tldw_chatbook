# Library artifacts — implementation evidence

Implemented on `codex/library-artifacts-design`, originally based on dev commit
`ebee42fab8` and rebased onto latest dev `de10a62e67124a2b21b78edf1a4887cea03ff139`. ADR-172 governs the completed browse/navigation boundary. The
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

## PR #2754 independent review follow-up

The requested fresh review of `ebee42fab8` → `4942a4484b` found two issues:
completed artifact DB workers retained native SQLite caches and prevented backup
maintenance drain; the artifact reader was absent from Library’s adaptive-shell
recognition and global F6 pane targets. Controller storage reads/actions now use
Library’s existing finite worker boundary. Artifact focus cycles through Library,
Items and reader, with closed-pane grips; focusing a narrow reader reveals its
stage, and an empty zero-width reader is excluded.

Regression tests first reproduced retained handles, false shell classification,
and broken forward/collapsed-pane focus. The narrow follow-up additionally caught
focus at x=51 on a 50-column screen, despite the focus ID being correct. Tests
assert rendered bounds, not only focus ownership. A 64-column fixture initially
assumed zero reader width; it was corrected to the actual list-first state.

- 12 controller-driven, private-profile SQLite cases pass: page/locate/detail
  under success, failure, cancellation and borrowed-handle ownership. Native
  closure is checked on the same pool thread, cancellation is joined before
  checking, the UI-thread connection stays usable, and maintenance drains.
- Seven production-CSS focus cases pass, covering forward/reverse traversal,
  collapsed grips, adaptive-shell recognition, narrow reader reveal/return and
  empty results. A held-callback test also proves a newer F6 focus cannot be
  overwritten by deferred reader reveal; already-focused reveal changes geometry
  without requesting focus again. The affected Reports/Chatbooks integration run
  passed 21 cases; six adjacent Enter/Escape/layout/restoration cases passed again
  after the narrow reveal change.
- The five focused new-controller governance checks pass. Only the initial pin
  for this PR’s new controller was revised for finite-worker dispatch and the
  already-focused reader reveal; existing module/screen ceilings remain unchanged.
- Native `TldwCli` verification in `/private/tmp/artifact-review-native/run4/`
  records actual terminal F6/Shift+F6 cycles at 160×50, 64×30 and 50×25,
  followed by Escape/Enter, with `LinuxDriver`, `isatty=True` and exit 0.
  The narrow screenshot was inspected. Earlier native attempts were not counted:
  they raced startup and assumed a search field in the compact starter rail.
  The final fresh-profile run primed the supported terminal capability helper
  and exercised the actual starter-rail fallback as well as closed grips.
- The independent reviewer accepted all corrections with no remaining findings.
  New/changed focused modules pass Ruff and formatting, and the legacy files
  have zero diagnostics on added lines. `git diff --check` is clean.

The previous-head CI run also identified diagnostic-inventory drift. The
statement review against inventory commit `0e10b0b72b22` found four new warnings
in the two artifact controllers (fixed copy, internal action names and exception
types only), one formatting-only share-sweep count message, and one unchanged
re-indented exception call. No new sink or path-privacy candidate was introduced.
`Docs/security/production-diagnostic-inventory.json` was regenerated after that
review; it adds the two owners and updates the existing share-owner digest.
The same CI checker with `--diff` passes: no drift, 610 owners and 14 sink files.

## Qodo review and dev rebase

All seven Qodo findings were addressed before merge:

1. Ctrl+6 has a dedicated compatibility action, outside the permanent shell
   shortcut map. Navigation labels, binding counts and the actual CSS destination
   tour agree on fourteen permanent destinations; Ctrl+6 still opens All artifacts.
2. Registered bundle paths pass central validation before symlink/file/ZIP probes.
   Lexical home expansion preserves existing `~/pack.zip` registrations. Sharing
   still performs its own staging/authority checks after explicit review.
3. Detail capability checks and playback use the normalized path returned by the
   existing audio-directory validator; malformed/outside paths are unavailable.
4. All six public artifact state dataclasses document their attributes and
   identity, paging, revision and capability contracts.
5. Reader builders and constructors expose concrete parameter/return types, with
   cross-module imports guarded by `TYPE_CHECKING`.
6. Exact-location predecessor windows derive from `ARTIFACT_PAGE_SIZE - 1`.
7. Kept exports normalize complete status and original creation time (falling
   back to kept time) for both the filename and Markdown document.

Eight behavior regressions passed after reproducing the failures. Independent
review caught a lost home-relative bundle path during validation; its positive
regression was observed failing before the compatibility fix and passing after.
Three truthy non-string registry path values also reproduced a TypeError; they
now become unavailable without breaking the share chooser (19 registry cases
passed after this final correction).
Final focused evidence (overlapping runs, not a full suite):

- 83 registry, catalog/state, share-owner and Chatbooks integration cases passed.
- 49 permanent-navigation and mounted Ctrl+6 cases passed. Two existing config-
  loading tests now use the repository's private-profile process harness, and
  the Ctrl+6 fixture disables its own startup splash through the real config.
- The broader UI run passed all 38 artifact navigation/canvas cases, including
  normalized playback and kept export date/status with and without original time.
  Its four navigation/fixture failures were fixed and covered by the final 49.
- The actual permanent-destination CSS tour and five controller governance
  checks passed (6 total). Existing unrelated ceilings were not changed.
- Focused files pass Ruff and formatting; all sixteen changed Python files have
  zero Ruff diagnostics on added lines. `git diff --check` passes.
- Native `TldwCli` in `/private/tmp/artifact-review-native/run6/` receives Ctrl+6's
  extended terminal sequence and reaches Library / All artifacts. Actual
  F6/Shift+F6, Escape and Enter pass at 160, 64 and 50 columns. The process used
  a fresh private profile, `LinuxDriver`, `isatty=True`, and exited 0. An earlier
  tmux legacy-key attempt produced Control-circumflex rather than Ctrl+6 and is
  not counted as passing evidence.
- Rebase preserved upstream MCP changes. Its only conflict was the generated
  diagnostic count; the combined inventory rebuild matches exactly: 610 owners,
  7,765 TASK-494 calls and 14 sink files.

ADR-172 remains the governing decision; no new storage, publication, or runtime
boundary was introduced by these corrections.

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
