# Schedules UAT Remediation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the UAT's 2 confirmed Blockers + 7 confirmed Majors (finding 10 refuted; the 404s were an environment artifact whose keepable fix is a capabilities handshake), per the root-cause report — `.superpowers/sdd/uat-remediation/root-causes.md` is the mechanism authority; deviate only with probe evidence.

**Spec/authority:** the root-cause report + `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md`. Rulings:
1. Blocker 2: NO kebab — move the existing `#scheduling-task-detail-lifecycle` row above the groups (the report's recommendation; spec §5's kebab is formally superseded, note it in the commit).
2. Stale-cluster (a): the scheduler loop fans out on the existing `on_queue_changed` seam (a real push beats making the ticker load — the ticker stays paint-only per its contract) — but if the seam trace shows it unreachable from the workbench, the fallback is a lightweight `post_message` bridge; document the choice.
3. Major 5: the destructive `except` in `load_tasks` becomes non-destructive (keep the last-good rows + an error notice) + a logging hook naming the raiser — the honest fix for an unpinned symptom.
4. Major 9: connectedness gates BOTH `_server_available` and `transfer_refusal` (the header's readiness probe is the source); an unreplayable orphaned mutation settles to `to_server_failed` (Retry/Cancel + "Last transfer error" already exist).
5. Capabilities handshake: probe `GET /scheduled-tasks/capabilities` per the `client.py:16269` precedent; missing routes surface honestly (kills Minor 24/Polish 31).
6. TEST DISCIPLINE (the blindness lesson): every display fix pins via the `render_strips()` geometry oracle (already imported in both test files for colour — now used for geometry); content-only assertions are insufficient for visibility claims.

## Global Constraints
Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-uat-fix`, branch `fix/schedules-uat-remediation` off f01cad196e. NEVER stash (git show / throwaway worktree for baselines); no pkill beyond own PIDs; FOREGROUND pytest; EXACT tail lines; tmp_path DBs; diagnostics pin SCRIPT on logger changes; class-targeted CSS + bundle rebuilds; census/ratchet parity; escape/Text discipline; commit trailer:

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv

### Task 1: Display blockers (findings 1+2 + minor 18)
compact=True at the 12 begin_edit sites + `#scheduling-queue-filter` (+ Escape blurs the filter); `TaskDetail, DefinitionDetail { overflow-y: auto }` + `#scheduling-task-detail-groups { height: auto }`; the lifecycle row moves above the groups (both panes' equivalents — check DefinitionDetail's ordering too). Tests: render_strips GEOMETRY pins for an open editor (its strip carries the value), the filter (typed text visible), the pane scroll (max_scroll_y > 0 on tall content), the lifecycle row painted at 235×52 AND 80×24-pushed.

### Task 2: The stale-display cluster (findings 3a/3c/3d/3e + Major 5 hardening)
(a) dispatch→UI fanout per ruling 2 — a fired reminder repaints within a tick, pinned; (c) `SyncOutcome.phase_errors` + per-phase toast truth + `last_push_at` stamped by the definition-push phase; (d) the liveness strip on its own 5s interval; (e) the one-line COMPLETED label; Major 5: non-destructive except + logging hook. Tests per item incl. the sync-success-never-toasts-failure pin.

### Task 3: Sync/transfer state (findings 4+5 + the handshake + finding 6's target)
Ghost row: `released_server_id` filtering of pulled_items (the adopted_server_id twin, 12 lines below); Major 9 per ruling 4; the capabilities handshake per ruling 5; conflict buttons: widen the click target (a container-level click or min-height 3 — smallest honest fix; finding 10 itself REFUTED, cite it). Tests: release→pull same-cycle → ONE row; no-server transfer refused with reason; orphaned mutation settles to failed w/ Retry visible; handshake gates the results affordances honestly.

### Task 4: Gates + scoped live re-verification
Full `Tests/Scheduling/ -q` + the schedules UI set + floor pins; census/ratchets/pin/bundle with attributions; then a SCOPED live tmux re-check of exactly the fixed legs (editor visible + typed text painted; pane scrolls to History; a fired reminder repaints without navigation; sync toast truth; the lifecycle row on screen) under a scratch profile — the UAT charter's evidence style, cleanup confirmed. Docs: User Guide touch-ups where fixes changed copy/behavior.

## After
Final whole-branch review (opus) → wave → PR `fix(scheduling): UAT remediation — display, staleness, transfer honesty` → bot round → in-loop merge → memory + file the refuted/deferred UAT minors to backlog.
