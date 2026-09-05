---
id: TASK-31660
title: >-
  Environment panel state honesty: UNBOUND and PENDING states (stale-root P0)
status: In Progress
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX critique P0 (2026-09-05, dual-agent, live-measured): after a workspace
switch the Environment panel keeps ANOTHER repository's counts/branch and
still offers "Commit or push · N files" — permanently (Refresh is inert when
root is None); on cold start it asserts "No git workspace" for ~20s inside a
git worktree. Root cause: root-is-None and pre-first-fetch have no
representation in EnvironmentSnapshot; poll_tick/request_refresh return
early and the last paint stands. Owner chose state-honesty as the first
burn-down cluster. Snapshot: see .impeccable critique 2026-09-05.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 When the workspace root accessor returns None, the panel lands an explicit UNBOUND state within one poll cycle: counts, Commit-or-push, PR/checks, and Tasks suppressed; copy matches Change Review's ("No folder is bound to this conversation's workspace, so changes are not tracked here — this is not a report that nothing changed")
- [x] #2 Before the first local-tier landing, the panel renders a PENDING state (e.g. "Checking workspace…") or stays hidden — it never renders "No git workspace" (or any negative) before a gatherer has answered
- [x] #3 A workspace switch clears the previous root's data within one poll cycle even when the new root is None
- [x] #4 Refresh in the UNBOUND state either re-checks the binding or is not offered; it is never a visible no-op control
- [x] #5 Deferred-fake controller tests + screen-wiring tests cover: cold start, bound→unbound switch, unbound→bound recovery
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split `NOT_APPLICABLE`'s two meanings: add `PENDING` ("nobody has
   looked") and `UNBOUND` ("no folder is bound") to `EnvSourceAvailability`;
   default every `EnvironmentSnapshot()` tier to `PENDING`.
2. Give both new states a projection in `console_environment_state.py`,
   returning before any counts/action rows can be built.
3. Make the controller's `None`-root path land an UNBOUND snapshot through
   the ordinary `_land`/`on_snapshot` path instead of returning early.
4. TDD both seams (pure projection + deferred-fake controller) plus the
   screen-wiring seam; update the pins that encoded the old conflation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`NOT_APPLICABLE` was carrying three jobs — "checked: not a repo", "never
checked", and (by omission) "no folder bound" — and the projection rendered
all of them as the flat assertion **No git workspace**. It now carries only
the first; `PENDING` and `UNBOUND` carry the other two, and
`NOT_APPLICABLE`'s meaning and rendering are byte-identical to before.

**The root-cause fix is the controller's early return, not the copy.**
`poll_tick`/`request_refresh` returned on a `None` root, so nothing landed
and the last paint stood: after a switch to an unbound workspace the panel
kept the previous repository's branch and counts and still offered "Commit
or push · N files" — permanently, since those two methods were the only
things that could ever have replaced them. Both now land an explicit
UNBOUND snapshot through the ordinary `_land`/`on_snapshot` path with
`scope_root=None` (called directly, NOT through `marshal_to_ui` — the
production marshal is `app.call_from_thread`, which raises when called from
the app's own thread, and both call sites are already on it).

Two consequences that were easy to miss:

- `_land`'s local branch replaces the WHOLE snapshot on an UNBOUND result
  (via the new `unbound_snapshot()` factory) rather than
  `dataclasses.replace(git=…, tasks=…)`. A per-field replace would mark git
  unbound while leaving the *previous root's* PR number and check results
  painted — the same defect one field over. The net TTL/pending bookkeeping
  is retired with it, so a rebind re-fetches instead of inheriting a 60s
  window keyed on the old `(root, branch)`.
- `_landed_root = None` is now a REAL landed value, so it can no longer
  double as "nothing has landed yet". `_has_landed` carries that, and
  `poll_tick`'s root-change branch is gated on it — which is what makes
  BOTH crossings of the unbound boundary genuine root changes
  (`root → None` wipes within one tick; `None → root` retires the old TTL
  and refreshes both tiers). Keeping the old `_landed_root is not None`
  guard would have made the unbound→bound recovery silently unreachable.

Rendering choice (the brief allowed either): PENDING/UNBOUND render as
muted rows in the **Environment** section and as *absence* in the **Tasks**
card. That is one consistent rule — never assert anything before you know —
expressed through each section's own idiom: Environment is a permanently
mounted header with the Refresh tail and needs a visible row to stay
honest; the Tasks card has no header to keep honest, so its non-assertion
is simply its absence (which is also what it already did for every non-OK
tier). Making Tasks render a "Checking backlog…" row instead would have
popped a section in and out of the rail on every cold start.

AC #4 falls out of the controller change: the Refresh tail already posts
`request_refresh(include_net=True, force_net=True)`, which now re-reads the
accessor and re-lands — and recovers immediately if a folder was bound in
the meantime.

Modified: `tldw_chatbook/Chat/console_environment_state.py` (enum, three
dataclass defaults, `unbound_snapshot()`, two projection branches, copy
constants), `tldw_chatbook/UI/Console_Modules/environment.py` (`_land_
unbound`, `_has_landed`, both entry points, `_land`'s UNBOUND branch),
`tldw_chatbook/UI/Screens/chat_screen.py` + `UI/Console_Modules/
right_rail.py` (docstrings/comments only), `Docs/User_Guide/console/
context-and-rag.md` (the empty-state table gained the two situations it was
silently folding into one), and the three test suites.
<!-- SECTION:NOTES:END -->
