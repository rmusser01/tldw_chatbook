---
target: "Console Inspect sidebar (Environment panel, PR #2411 worktree)"
total_score: 18
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 3
timestamp: 2026-09-05T06-46-15Z
slug: tldw-chatbook-ui-console-modules-right-rail-py
---
# Console Inspect rail (Environment panel) — critique 2026-09-04

Method: dual-agent. Target: tldw_chatbook/UI/Console_Modules/right_rail.py on worktree feat/console-inspector-environment @ 830c5828e6 (PR #2411). Live UAT at 235×52 and 80×24, isolated profiles.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | No freshness signal; Refresh silent; ~20s false "No git workspace" on cold start |
| 2 | Match System / Real World | 2 | "Local", "wt:", bare "+4 −0" with no referent; slugified task titles |
| 3 | User Control and Freedom | 2 | 10s poll resets focus; n/p evicts panel with no route back |
| 4 | Consistency and Standards | 2 | Five Enter outcome classes on identical rows; "Commit or push" breaks the …-opens convention; doing/todo vs in progress/to do |
| 5 | Error Prevention | 1 | After workspace switch, another repo's data stays on screen and "Commit or push" is still offered |
| 6 | Recognition Rather Than Recall | 2 | Nothing marks actionable rows or predicts Enter's outcome |
| 7 | Flexibility and Efficiency | 2 | No section jump reaches the panel; no Refresh key; Tab can't reach rail from composer |
| 8 | Aesthetic and Minimalist Design | 1 | Two-line rows; inert Local row; header duplicates rows; 80×24 shows net-zero info |
| 9 | Error Recovery | 1 | "No git workspace" names no recovery; Refresh inert in that state; stale is color-only |
| 10 | Help and Documentation | 3 | User Guide honesty table excellent; footer advertises n/p that can't reach these sections |
| **Total** | | **18/40** | **Poor** |

## Priority Issues

**[P0] Stale-root state model: the panel keeps another repository's data on screen after a workspace switch — permanently — and still offers "Commit or push"; on cold start it asserts "No git workspace" for ~20s inside a git worktree.** Root cause: `root is None` and pre-first-fetch have no representation in EnvironmentSnapshot — poll_tick/request_refresh return early, leaving the last paint. Fix: add UNBOUND + PENDING states, land them explicitly; use Change Review's own copy ("No folder is bound… this is not a report that nothing changed"); suppress Commit-or-push/counts/Tasks in those states; never render a negative before a gatherer answers.

**[P1] The 10s poll silently steals rail focus whenever the changed-file set changes** — measured: focus parked on a row is thrown above the section header at the next tick after an external file edit; two Tabs to recover; fires repeatedly during agent runs (the panel's core workflow). Fix: capture focused row_id before sync_state, restore after (the activation path already does this — reuse `_request_console_environment_row_focus` on the poll path).

**[P1] Two-line rows make 80×24 unusable and violate the density mandate everywhere.** 8 pinned lines + 3-line scroll body; the one visible row restates the header; 25% of Environment's row-lines are blank; a 2-file diff with two expansions eats 20 lines at 235×52; header needs 55 cols against 33. Fix: min-height 1, hide empty secondary, right-align secondary on the primary line when it fits; suppress header summary while open; drop the redundant Tasks counts row; kill or demote the inert Local row.

**[P1] Keyboard access and focus visibility fail together:** all five focus-indicator styles measure 1.03–1.79:1 (below any threshold), one Tab stop has no indication at all in either capture, Tab from the composer never reaches the rail (40 presses; F6/Alt+I only), and n/p evicts the panel from the viewport with no route back while the footer advertises "n/p Sections". Fix: real focus ring styling (the row's corner brackets are the right idea — extend to buttons), include ConsoleInspectorSection headers in the boundary ring or suppress the footer hint inside these sections.

**[P2] Affordance + consequence legibility:** five Enter outcome classes (expand / full-screen nav / leave app / append to composer draft / nothing) render identically; "Commit or push" performs navigation and omits the "…" its own destination uses; Refresh gives zero feedback over 11.7s when data is fresh; stale is color-only in a hue identical to error ($ds-status-blocked == $ds-status-error, 2.53:1 on banded rows). Fix: trailing-marker convention (▸ expand, … opens surface, distinct marker for composer-inserts), rename to "Review & commit… · N files", transient "Refreshing…" acknowledgment, text stale marker, $text-error for readable error text.

## Measured evidence highlights (Assessment B)

- Background banding: #2d2d2d originates in the left rail and bleeds full-width to col 233, splitting single rows across two backgrounds; same secondary fg measures 3.44:1 on one line and 5.24:1 on the next. **Overturns the prior critique's refutation** — `.console-inspector-section-row-secondary` does render in the right rail.
- Section inner scrollbar thumb: fg==bg #2d2d2d, 1.00:1 — invisible at both sizes.
- Refresh DOES force a re-read (≤0.3s when stale) — the control works; only the feedback is absent.
- Commit-or-push shows a transient false "No file changes recorded" for ≤0.5s before the real diff.
- Poll cadence confirmed ~10s (mid-cycle edit landed t+6..7s); branch truncation width varies with unrelated count digits.
- Detector: TCSS scan → 2 advisory undocumented colors (#6f7782 known frame border; rgb(245,245,245)); Python target exits 0 with `[]` — silently inapplicable, not evidence of cleanliness.

## Persona red flags

Jordan (first-timer): first impression is a false statement (~20s); Local is mystery-meat disclosing a non-feature; +4 −0 has no unit/base/tree; empty state dead-ends while its neighbor teaches recovery; no signal rows respond to Enter.
Alex (power user): poll steals focus mid-workflow; Tab never reaches the rail; n/p is a trapdoor; Refresh unverifiable; Tasks-without-branch-task = ~62 lines, zero actions, slug titles.
Sam (a11y lens): row focus is shape-carried (good) but 1.42:1 background lift; buttons have color-only focus at ~1.6:1; one stop indication-free; stale color-only sharing error's hue; secondary text under AA on banded rows.

## Minor observations

Task titles use filename slugs while parsed frontmatter holds real titles (one-line fix); "▸ Inspect" open vs "<-Inspect" closed glyph mismatch; Refresh floats orphaned (own margin + blank neighbor lines) and sits ~28 lines from its data when Changes is expanded; "doing/todo" vs "in progress/to do" one line apart; Change Review header "1 files" vs rail's correct "1 file"; four Change Review openers and counting; hidden-but-focusable blank widget in left rail is what breaks Tab routing; expansion children share parent indent (containment cue missing — add indent field or └ glyph).

## Questions to consider

1. If Changes expanded to the diff itself, would Commit-or-push need to exist as a rail row — or is the panel a launcher wearing a status panel's clothes?
2. The panel refuses to run `git fetch` on honesty grounds — why doesn't that standard extend to root-is-None, where it answers a question it never asked?
3. Three top sections use a different interaction grammar and live outside the rail's own section ring: one surface with two grammars, or two surfaces sharing a scroller?

## History note

Prior critiques of this rail (main-checkout snapshots): 15/40 (2026-08-29, pre-burndown) → 23/40 (post-burndown re-critique). This run (18/40) is not like-for-like: it reviews the NEW Environment panel top sections on the PR branch, where most findings concentrate; the older sections below scored well in strengths (Change Review round trip, in-place disclosure, no flicker).
