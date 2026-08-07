# ADR-043: Console rail compact-collapse yields to explicit toggles

Status: Accepted
Date: 2026-08-05
Related Task: [backlog/tasks/task-2154.2 - Console-make-Inspector-reachable-below-150-cols-LY-11-DS-06.md](../tasks/task-2154.2%20-%20Console-make-Inspector-reachable-below-150-cols-LY-11-DS-06.md)
Supersedes: N/A

## Decision

The Console side rails' compact-width collapse rules (Inspector below 150 columns, Context rail below 100) are the responsive **default rendering**, not a hard block: an explicit user toggle is honored at any terminal width, with the main column's minimum-width guarantee waived for that viewport so the workspace grid always resolves.

## Context

The persistent-rails spec (2026-05-24) defines compact-width protection as "a responsive rendering override, not a preference mutation": when the terminal is too narrow to safely render a rail, Console renders it collapsed without overwriting the stored preference. The implementation applied this as an unconditional hard block below the threshold. UX review finding LY-11 (2026-08-04) showed the failure mode: at 140 columns, clicking the Inspector handle persisted `right_open=true` with zero visual change — a silent no-op that left staged Sources, the retrieval-scope row, the run inspector, and the settings summary with no reachable surface (compounded by DS-06: the Sources/Tools status chips were focusable but inert). TASK-2154.1 reproduced the same inert-handle pattern for the left rail below 100 columns.

Two constraints shaped the fix. The persistence model (per-workspace `:layout` keys, coerced defaults, launch/band auto-open behavior) must not change — including the 118–128-column auto-open band and the pending-launch Inspector auto-open suppression below 150. And "too narrow to safely render" must remain true: honoring a toggle may never reintroduce the LY-08/LY-09 grid overflow.

The rails detect "explicit" differently because their defaults differ. The right rail's default is closed, so the coerced value suffices: default and explicitly-stored `right_open=False` both keep the collapse (preserving the launch auto-open suppression byte-for-byte), and only `right_open=True` yields. The left rail's default is open, and every write serializes the full preference payload — so neither the coerced value nor plain key presence can distinguish "never toggled" (keep the LY-08 force-collapse) from "explicitly opened below the threshold" (honor it), nor from "rode along in an unrelated toggle's write" (an early version of this change used key presence and UAT caught the left rail opening below 100 columns after a right-rail toggle). Explicitness is therefore recorded by a dedicated `left_open_explicit` payload marker, written alongside the toggle gesture and preserved across later unrelated writes; legacy payloads lack the marker and keep the force-collapse default. `_set_console_rail_preference` writes through on explicit rail toggles even when the coerced value is unchanged, so the gesture (and marker) is recorded. The main column's existing min-width waiver (single-pane mechanism, TASK-2154.1) extends to honored-below-threshold rails via the new `*_compact_override` state flags, which are computed only in `build_console_rail_state` so the band/launch `replace()` paths keep their exact prior rendering.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Open the rail as an overlay panel below the threshold | A second rendering of the same rail (absolute positioning, dismissal, focus layering) duplicates layout machinery for a case the in-grid waiver already solves; overlay focus/dismissal semantics diverge from every other Console surface. |
| Keep the hard block; handle shows why + chips open a content modal | A staged-sources/run-inspector modal would fork the Inspector's content into a second, watered-down surface to keep in sync; the handle remaining inert below the threshold still violates "every manual toggle produces visible feedback". |
| Honor explicit toggles below the threshold without waiving the main min-width | Reintroduces grid overflow between ~84–115 columns (rail mins 24/34 + main 56 + handles exceed the viewport), the exact LY-08 regression the force-collapse was added to prevent. |
| Track "explicitly toggled at this width" as transient session state | Does not survive restart; the spec requires manual toggles to persist, and a per-width memory multiplies state for no user-visible benefit. |

## Consequences

- A manual rail toggle can never again be a silent preference-only change: it either changes the rendered layout or (when already in the requested state) was never a toggle at all.
- Users who explicitly open a rail below its threshold get a denser layout (transcript narrower than the usual 56-column minimum) — an accepted tradeoff, chosen explicitly and reversible via the rail's own collapse button.
- The stored-preference schema, keys, coercion, and migration paths are unchanged; the only new persisted facts are the `left_open_explicit` marker on left-rail toggles (additive, ignored by older readers and by coercion) and that explicit rail toggles write through even when equal to the coerced default.
- Auto-open behavior is unchanged everywhere: the 118–128 band rule and the pending-launch auto-open evaluate exactly as before (verified by the value-based right-rail detection and by computing override flags before the `replace()` paths).
- Any future rail added to the workspace grid must decide whether it is default-open or default-closed to choose value-based vs marker-based explicit-toggle detection.

## Links

- [UX review findings LY-11/DS-06](../../Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md)
- [Persistent rails design spec](../../Docs/superpowers/specs/2026-05-24-console-persistent-rails-design.md)
- Parent epic: backlog/tasks/task-2154; sibling: task-2154.1 (responsive fallback below ~100 cols)
