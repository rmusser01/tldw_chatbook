# Library Phase C — Media Graduation (Resident Canvas) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The program's founding motivation, finally: eliminate the measured **139–380 ms main-thread freeze** on Library rail-mode switches by making the media canvas resident across mode switches instead of being torn down and remounted per click — the first phase-C graduation per the spec's "Phase C — region ownership" section. This is a DESIGNED BEHAVIOR-CHANGE series, explicitly outside the pure-move policy that governed waves 1–8.

**Architecture:** Spec: `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` §"Phase C — region ownership". Graduation criteria are met for media: series + cleanup landed (waves 7–8); mounted coverage dense once TASK-31880 closes (in flight on this branch); the motivating change is this plan. Evidence instrument: `Helper_Scripts/library_click_probe.py` under recipe §9's corrected same-location pairing; the last pre-phase-C baseline is §25's close-out table (settle 264–485 ms band; mount counts 177/89/85/114/38/179/114/175; recompose 0).

## Global Constraints

- **This is not a move series.** Byte-for-byte does not apply; TDD DOES (failing test first — including the probe-derived acceptance thresholds), and behavior changes need their own tests. The size-ratchet rows still bind: screen only shrinks or holds; new code belongs in Library_Modules or Widgets/Library.
- **Compose with TASK-31521 screen reuse, never fight it.** Dev's screen-level suspend/resume machinery (`_library_screen_suspended`, `_refresh_library_visit_surfaces`, the suspend timer stops) already keeps ONE LibraryScreen per app run. Phase C adds canvas residency WITHIN that resident screen — mode switches inside the Library. Every interaction (a resident canvas receiving events while its row is unselected; suspend/resume of a screen with resident canvases) needs an explicit ruling.
- **Evidence discipline (all recipe §3/§6/§7/§9/§24 rules stand):** same-location probe pairs, order-swapped, n>1 with interleaved round-robin for dispositions; every count/line-range verified live; §7 by-name before calling any red new; isolated worktrees + venv parity for baselines.
- **The dispatcher debt is adjacent, not absorbed:** `canvas_sync.py`'s shared dispatchers and the TASK-32089 accessor-guard debt sit on the exact path this plan changes. Task 1 must MAP the interaction; fixes to that debt land only if the residency mechanism requires them (then as their own labeled commits referencing TASK-32089).
- **Rollback posture:** each task independently revertible; the probe acceptance test is the no-regression floor for every subsequent Library change (it stays in the suite, band-pinned).

---

### Task 1: Mechanism spike + measured design decision
The current teardown: instrument and quantify exactly what a media↔other rail switch destroys and rebuilds today (mount/unmount counts per switch, CSS reflow cost, the 139–380 ms band's composition) using the probe + targeted instrumentation on a REAL app run. Evaluate the candidate residency mechanisms against Textual 8.x's actual behavior (visibility/display toggling of a persistent canvas; widget-instance caching with detach/reattach; lazy-mount-once-then-toggle) — a throwaway spike per mechanism, measured with the probe. Decide with numbers; record the decision + rejected alternatives + measured deltas in a design-record section appended to the spec (ADR style). Deliverable includes the FAILING acceptance test: a probe-derived pinned test asserting the mode-switch settle/mount numbers the chosen mechanism must hit (red today by construction). STOP-and-report if every mechanism measures worse than a threshold improvement or fights TASK-31521 irreconcilably.

### Task 2: Resident media canvas (the behavior change)
Implement the chosen mechanism for the media canvas: TDD from Task 1's failing acceptance test; the suspend/resume interaction rulings implemented and pinned (a resident-but-unselected canvas must not process row events — gate per the TASK-31521 suspended-activity precedent); the canvas_sync dispatcher path updated only as the mechanism requires (TASK-32089-labeled commits); all existing media suites + both dual-receiver guards + screen-reuse green; probe before/after pair (same-location, order-swapped, interleaved) recorded as the acceptance evidence.

### Task 3: Region-ownership migration (media, scope-honest)
Per the spec: canvas-origin `@on` handlers migrate from the screen's routing table into the media canvas widget where the residency work has ALREADY forced ownership clarity — migrate exactly the handlers the resident canvas now logically owns, not the full table (rail/footer/header rows are permanent per the spec's scope honesty). Each migration is a designed change with its own test; the wiring suites' delegator pins updated as handlers leave the screen; screen ratchet lowered same-commit.

### Task 4: Graduation close
Probe acceptance evidence finalized (the before/after table in the spec's design record + recipe §25 addendum: the freeze band then vs now); notes-graduation readiness note (what transfers, what doesn't); stale-doc sweep; durable SDD evidence; follow-up filings (id sweep as the LAST commit before push per the hygiene lesson); full battery + paired sweep vs this plan's start commit.

## Self-review record
- Task 1 buys the design with measurements instead of guessing the mechanism in this plan — the founding freeze was misdiagnosed twice before the probe existed; phase C does not repeat that.
- The TASK-31521 composition rulings are named per task; the two systems (screen reuse, canvas residency) are complementary but their seams are where silent breakage lives.
- Region ownership (Task 3) rides AFTER the motivating change and only as far as residency forces it — the spec's "never done for its own sake" rule.
- All evidence mechanics by reference to the recipe; only phase-C decisions are new here.
