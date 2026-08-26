# Qodo audit of 25 merged fleet-programme PRs — verdicts and fixes

An owner question ("all Qodo comments addressed, yea?") had an honest answer of NO:
the programme's PRs merged within seconds of opening, so the bot's reviews were never
read. A read-only audit fetched all 69 findings across PRs 1461-1793 and triaged each
against current dev in a detached worktree. This report preserves the verdicts; the
three significant fixes and the minor batch landed on `fix/qodo-audit-findings`.

## Counts

| Total | Stale/fixed by later PRs | Noise (reason recorded) | Real-minor | Real-significant |
|---|---|---|---|---|
| 69 | 7 | 35 | 24 | 3 |

Two noise verdicts were established by DISPROVING the bot's premise against the
installed Textual source (Screen.dismiss pops synchronously; App.notify is documented
thread-safe). The 7 stale findings were all fixed by later PRs in this same programme.
Noise clusters: Qodo compliance rules over-applied to test files; per-file DB
conventions (AgentRuns_DB's inline versioned schema); the documented producer-boundary
validation design for steering; the repo's diagnostic-privacy logging posture
(TASK-15103) which the bot repeatedly flagged as "insufficient logging".

## The three significant fixes (this branch)

**S1 (`72ef7c083`) — a superseded Console leave stages no teardown notice.**
`_record_console_fleet_teardown` ignored `leave_console_runtime`'s return bool and
staged the "N session(s) cancelled when you left Console" notice unconditionally — on
an overlapping ChatScreen→ChatScreen navigation the superseded screen's leave no-ops
by design while sessions keep running under the successor, yet the user was told they
were cancelled. Red drove the real overlap; the fix gates staging on the bool; the
true-teardown path's existing pins stay green.

**S2 (`86b90c068`) — a viewless-from-birth approval round bound an orphaned Event.**
`_bind_visit_cancel_signal` inferred "no visit" from `_shutdown_requested.is_set()`;
a never-visited controller holds an UNSET E0, so a wake-at-launch approval round bound
E0 — which `begin_visit` then replaces, leaving an Event nothing in the process can
ever set: the round survived the leave that must deny it, and app-exit could not
cancel it. Fixed to bind the headless cancel signal when no visit has EVER opened;
red proved the survival, and the leave/app-exit denials are pinned.

**S3 (`092d65ecf`) — a steering draft typed for child A submitted to child B.**
The steering bar's Input survives drill-out (visibility is display-toggled), and
submit pairs the retained text with the LATEST target. Fixed: the input clears when
`target_id` changes — and deliberately does NOT clear on a same-target re-sync, so a
routine tick never eats a draft mid-typing. Both directions pinned.

**Minor batch (`a8b5dde86` + docstring/typing commits):** draft preserved on a
refused submit; terminal fleet rows drop the `steering queued (N)` segment; the
delivering-flag set only after `create_task` succeeds; teardown debug log carries
exception context; docstrings/typing across the flagged sites; test hygiene (join
asserts, fixed-sleep, import order, quiet-window parity). DEFERRED to avoid colliding
with the concurrent Task 4/5 work in `agent_service`/`fleet_coordinator`/
`agent_runtime`/`AgentRuns_DB`: fingerprint dedupe, wait-result cap-0 notice, the
drain-vs-step-budget note (the schema constant was closed by Task 4's v11).

## Process change

Remaining programme PRs wait for the Qodo review before merge; findings are triaged
inline (fix, or dismiss with a recorded reason) rather than merged past.
