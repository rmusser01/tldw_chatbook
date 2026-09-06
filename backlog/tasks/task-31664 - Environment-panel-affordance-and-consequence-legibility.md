---
id: TASK-31664
title: >-
  Environment panel affordances: mark actionable rows, name consequences,
  acknowledge Refresh
status: Done
assignee: []
created_date: '2026-09-05 07:00'
updated_date: '2026-09-06 03:12'
labels:
  - console
  - inspector
  - ux
  - critique-2026-09-05
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P2: Enter has five outcome classes (expand in place, full-screen
navigation, leave the app, append to the composer draft, nothing) on
visually identical rows; "Commit or push" performs navigation and omits
the "…" its own destination uses; Refresh produces zero visible feedback
for 11.7 measured seconds when data is fresh (it works — ≤0.3s when stale
— but is indistinguishable from dead); "stale" is color-only in the exact
hue of error ($ds-status-blocked ≡ $ds-status-error, 2.53:1 on banded
rows). Repo precedent for the fix: the left rail's System line trailing ▸
(chat_screen ~7983) and Change Review's Commit…/Push… ellipses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A consistent trailing-marker convention distinguishes expand rows, rows that open another surface, and rows that modify the composer draft; inert rows carry none
- [x] #2 "Commit or push · N files" is renamed to name what it does (e.g. "Review & commit… · N files")
- [x] #3 Refresh shows a transient acknowledgment (e.g. "Refreshing…") even when the data comes back unchanged
- [x] #4 Stale state carries a text marker alongside color; failure/stale rows a user must read render the readable error hue (>=4.5:1 on both banded backgrounds). AMENDED at final review: the shipped design deliberately converges stale and error on $ds-status-error-readable and differentiates via the "(stale)" text marker — better than the hue-split originally worded here (controller ruling, whole-branch review)
- [x] #5 The UNBOUND copy names the true cause or goes cause-agnostic: workspace_roots == () also occurs when Change Review consent is not ENABLED for a bound folder (the common default), when the consent service is absent/raises, and when all bound roots are skipped — "No folder is bound" is wrong in those cases (31660 re-review obs; the "changes are not tracked here" clause stays true). Distinguish consent-off if the admission data allows; also restore the remediation half of Change Review's copy (bind/enable path)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the task file (5 ACs, #5 added after review) and the plan's Task 5 landmarks.
2. Design a trailing-marker convention in the projection (console_environment_state.py): expand chevrons (▸/▾), a navigation ellipsis, and a composer-insert prefix; apply to every clickable row; leave inert rows untouched.
3. Rename "Commit or push · N files" to "Review & commit… · N files" (AC#2), sweeping tests for the old literal first.
4. Add a stale text marker to secondary_text at every stale-driven row (AC#4) and split the CSS hue for the checks/error row to $ds-status-error-readable.
5. Add ConsoleInspectorSection.set_view_all_busy() (bypasses sync_state's equality guard) and wire it from the screen's ViewAllRequested handler + _land_console_environment (AC#3).
6. Investigate whether the UNBOUND cause is cheaply distinguishable at the _console_environment_root seam; reword the copy cause-agnostic either way and restore Change Review's enable-remediation half (AC#5).
7. Update/extend tests in both seams (projection + wiring + widget), regenerate the CSS bundle, update the User Guide, run targeted suites + preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all 5 ACs for the Environment panel's affordance/consequence legibility.

AC#1 (marker convention): trailing-marker vocabulary added in the pure projection
(console_environment_state.py), never in the widget -- `_with_expand_marker` (▸
collapsed / ▾ expanded, matching the section header's own chevron), `_with_surface_marker`
(trailing "…" for rows that navigate to Change Review or the OS browser),
`_with_insert_marker` ("+ " prefix for composer-insert rows). Applied to all 12
clickable rows (Changes/Local/branch/PR/checks/Tasks-head expand; Review-in-Change-Review/
Open-in-browser/commit-push navigate; Add-to-chat/Fix/Add-task-to-chat insert); file
rows, task entries, "Local instance ✓" and every expanded detail row stay unmarked
(inert). The expand marker ellipsizes its label BEFORE appending the marker (mirrors
`environment_summary`'s existing pattern) so a long branch name's overflow can never
eat the chevron via the terminal's own CSS `text-overflow: ellipsis` -- covered by a
dedicated test with an 80-character branch name.

AC#2 (rename): "Commit or push · N files" -> "Review & commit… · N files" (dirty case);
the ahead-only "Push ↑N" variant gained a trailing "…" too, since it navigates to the
same destination. Both now carry the same ellipsis Change Review's own Commit…/Push…
actions use. Swept and updated 4 test literals across 2 files before editing.

AC#3 (Refresh acknowledgment): new `ConsoleInspectorSection.set_view_all_busy(bool)`
flips the "view all" tail Button's label directly (VIEW_ALL_BUSY_LABEL = "Refreshing…"),
bypassing `sync_state`'s rows/summary equality guard -- that guard is a deliberate
no-op on a content-identical landing, which is exactly the scenario that left Refresh
looking dead for ~12 measured seconds. Wired from `on_console_inspector_section_view_all`
(sets busy=True before calling request_refresh) and cleared unconditionally at the end
of `_land_console_environment` (the controller's on_snapshot, which fires on EVERY
landing). Only the explicit Refresh handler arms it, so the 10s automatic poll never
flickers it -- covered by both a widget-seam test (direct set_view_all_busy calls) and
a wiring-seam test (real ViewAllRequested -> busy label -> real landing -> reverted
label), plus a negative control proving the poll path never shows it.

AC#4 (stale text carrier + hue split): new `_with_stale_marker` appends "(stale)" to a
row's secondary_text (or stands alone when secondary_text is empty) for the three rows
`git.stale`/`pr.stale` can mark (Changes, branch, PR). CSS: `.console-inspector-section-row-error`
now uses `$ds-status-error-readable` ($text-error, the AA-passing per-theme token
already documented for exactly this in `_variables.tcss`) instead of the decorative
`$ds-status-error` -- `$ds-status-blocked` (stale) is untouched and keeps aliasing
$error, so stale and error no longer resolve to the identical hue on rows a user must
read. Source edit in `_agentic_terminal.tcss`, regenerated via build_css.py into
`screen_agentic_console.tcss`; added a `test_console_rail_color_grammar.py` contract
entry pinning the new declaration in both the source and generated sheets.

AC#5 (UNBOUND copy): investigated whether the true cause is cheaply distinguishable at
the `_console_environment_root` seam. Traced the real admission path
(`ChangeReviewConsentService.admit_turn` -> `_admit_enabled_locked`): consent-off,
capability-unavailable, service-absent/raising, and genuinely-no-folder-bound all
return the SAME empty `ChangeReviewAdmission()` (no ready_roots, no skipped_roots) --
indistinguishable from each other at this seam. Only "bound roots that are still
preparing/failed" leaves a distinguishing trace (`skipped_roots` non-empty), and that
signal isn't currently plumbed up through `resolve_turn_execution_context` to the rail;
wiring it through would require broadening the `workspace_root_accessor` contract
across the controller for one narrow, uncommon sub-case. Shipped the AC-sanctioned
cause-agnostic fallback instead: "Changes aren't tracked for this workspace." + "Bind
a folder and enable Change Review in Settings ▸ Workspaces — this is not a report
that nothing changed." Also restored the missing ENABLE half of Change Review's own
empty-state remediation (`change_review_screen.py::_empty_history_copy`), which
previously named only the bind step.

Modified: tldw_chatbook/Chat/console_environment_state.py (markers, stale marker,
renamed labels, UNBOUND copy), tldw_chatbook/Widgets/Console/console_inspector_section.py
(set_view_all_busy + VIEW_ALL_BUSY_LABEL), tldw_chatbook/UI/Screens/chat_screen.py
(refresh-busy wiring in on_console_inspector_section_view_all + _land_console_environment),
tldw_chatbook/UI/Screens/change_review_screen.py (remediation copy),
tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated
screen_agentic_console.tcss). Tests: Tests/Chat/test_console_environment_state.py,
Tests/UI/test_console_environment_section.py, Tests/UI/test_console_environment_wiring.py,
Tests/UI/test_console_rail_color_grammar.py, Tests/UI/test_change_review_current_mode.py
(2 pre-existing assertions updated for the new cause-agnostic copy). Docs:
Docs/User_Guide/console/context-and-rag.md (row table, missing-things table, new
dated amendment).

Verification: targeted suite (environment state/section/wiring/controller, right_rail,
focus_carriers, fleet_panel, change_review_current_mode, rail_color_grammar,
inspector_section) = 301 passed, 1 failed. The 1 failure
(test_fresh_rail_compose_applies_the_agent_status_class, a ConsoleLeftRail constructor
kwarg mismatch in left_rail.py) is pre-existing and unrelated -- left_rail.py is
untouched by this task and the failure reproduces on the pre-task tree. preflight.sh
green (CSS bundle, profile-owned paths, diagnostic inventory, backlog id census, index
plan pins).

Known scope limit (documented, not a defect against the AC): the Refresh acknowledgment
reverts on the FIRST landing after a press, which for the (rare) UNBOUND workspace case
completes synchronously within the same call as the press (per TASK-31660's deliberate
`_land_unbound` design) -- so "Refreshing…" is set and cleared before any repaint can
show it there. This is a non-issue in practice: that landing is near-instant already, so
the acknowledgment is not needed for it; it reliably shows for the actually-measured
defect (the async `gh`/local-tier dispatch path).

## Round-1 review fixes

Independent review held up AC#1/#2/#5 but found 5 required defects, all fixed:

I1 (Refresh ack bound to the wrong event): the ack was clearing on the FIRST
landing (local, ~0.3s) while the `gh` fetch -- the actual measured ~12s the
control exists to cover -- ran on silently. Fixed by having the controller
track `pending_ack_tiers` (a `{local, net}` set populated by `request_refresh`
BEFORE dispatch, discarded in `_land` as each tier genuinely lands); the
screen now clears the ack only when this is empty. Verified the gh-timeout
path cannot leak the counter: both gatherers are documented and coded "never
raises" (`run_gh` catches `TimeoutExpired`/`OSError` into a returncode-124
`GhResult`; `gather_pr_env`/`gather_git_env` always return a state), so the
worker's `_marshal_to_ui(self._land, ...)` call always fires.

I2 (ack could wedge forever): fixed four separate paths -- (a) arming moved
from BEFORE `request_refresh` to AFTER it, gated on `pending_ack_tiers`
actually being non-empty, so a call that dispatches nothing (`UNKNOWN_ROOT`,
rail closed) never arms; (b) the UNBOUND branch pre-seeds pending with
`{local}` right before the synchronous `_land_unbound()` call, so it's
already empty by the time `request_refresh` returns; (c) `_land`'s
stale-scope guard now also clears `pending_ack_tiers` (a scope change means
no future landing will ever reach the tier-discard code for the OLD press);
(d) `_land`'s deferred-net-abandoned branch (rail closed before the deferred
net could be re-issued) now discards `net` from pending too. The screen's
own clear-check moved to the TOP of `_land_console_environment`, before the
rail-mount early return, so a landing arriving mid-recompose can't skip it
either.

Testing surfaced a fifth, unreported wedge: a STRUCTURAL row-set change
(the PENDING -> real-rows transition is the production case) recomposes
the whole `ConsoleInspectorSection`, rebuilding the tail Button from
`compose()` -- which had no idea an ack was in flight and silently reset
it to "Refresh". Fixed by tracking `_view_all_busy` as a widget attribute
`compose()` consults on every rebuild, not just a live-widget mutation.

I3 (ASCII fallback bypass): the row markers (▸/▾) were raw literals, so
`appearance.ascii_glyphs` correctly substituted the section header's own
chevron but printed the raw glyph on every row. Kept the projection
(`console_environment_state.py`) pure -- per the review's own suggested
seam split -- and resolve at the widget/render seam instead
(`ConsoleInspectorSectionRow`, via `glyph_fallback.resolve_glyph_text`),
matching every other consumer of that module (all are widget-layer, never
a pure-state module). Width/shape decisions still use the unresolved text;
every marker this repo assigns a fallback for is a 1-for-1 substitute, so
that's never wrong once ASCII mode resolves it.

I4 (a11y token missed its target rows): `$ds-status-error-readable` had
landed on `.console-inspector-section-row-error`, but the row the critique
actually measured (2.53:1) -- "Environment unavailable — Refresh to retry"
and every stale Changes/branch/PR row -- carries status="blocked", not
"error"; the fix never reached it. Applied the readable token to
`-row-blocked` too (both converge on one hue: a real failure and
possibly-stale data are both "must be read" states). Corrected the two
comments (CSS + `test_console_rail_color_grammar.py`) that had asserted
the wrong row was already fixed.

I5 (mechanical): the PR/checks table rows in the User Guide had 3 cells in
a 4-column table (the Actions cell got merged into "Enter expands to"
without the separating pipe). Restored the header's 4-cell shape.

Minor ride-alongs: `_with_surface_marker` no longer inserts a space before
"…" (now matches Change Review's `Commit…`/`Push…` precedent exactly);
verified clear-ack-on-row-expansion stays correct now that the clear is
gated on `pending_ack_tiers` rather than "any landing" (a no-op when
nothing is pending, which is the common case). Recorded follow-up path
for AC#5's cause-distinction: `ChangeReviewConsentService.status(workspace_id)`
exposes `capability.state`/`consent.state` directly (unlike `admit_turn`,
which collapses them), and `list_folder_bindings` can be read regardless of
consent -- combining the two COULD distinguish "consent off" from "nothing
bound" without needing a folder to be ready, but isn't plumbed to this
panel today; a real follow-up task, not done here.

New/changed tests: `Tests/UI/test_console_environment_controller.py` (+7,
`pending_ack_tiers` at the controller level via `DeferredFixture`),
`Tests/UI/test_console_environment_wiring.py` (replaced the stubbed ack
test with a deferred-fake one driving the REAL controller through both
landings; added the UNKNOWN_ROOT never-arms test), `Tests/UI/
test_console_environment_section.py` (+2: recompose-survival, ASCII
fallback), `Tests/UI/test_console_rail_color_grammar.py` (+1: the
-row-blocked declaration), `Tests/Chat/test_console_environment_state.py`
(surface-marker space removed from 2 assertions), `Docs/User_Guide/
console/context-and-rag.md` (table cell fix).

Verification: the same targeted suite (environment state/section/wiring/
controller, right_rail, focus_carriers, fleet_panel, change_review_current_
mode, rail_color_grammar, inspector_section, console_glyphs) = 319 passed,
2 failed. Both failures are pre-existing and unrelated: test_fresh_rail_
compose_applies_the_agent_status_class (ConsoleLeftRail kwarg mismatch,
untouched file, same as round 1) and test_live_work_widget_swaps_cover_
real_twenty_twenty_one_geometry[...] (the task brief's own documented
"known flaky... baseline before blaming" -- confirmed by re-running it
alone: 2 passed). preflight.sh green.
<!-- SECTION:NOTES:END -->
