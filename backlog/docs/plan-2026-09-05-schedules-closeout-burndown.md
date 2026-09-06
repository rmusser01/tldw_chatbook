# Schedules Close-out Burndown

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Burn down 9 of the 11 filed close-out/UAT-minor tasks in one PR (5 review-gated commits); defer 2 as genuinely standalone. Authority: each task file's ACs + `.superpowers/sdd/burndown/burndown-scope.md` (the scoping survey).

**Deferred (NOT this PR, keep open):** 31415 (post-transfer history-identity bridge — DB-query-layer change, ~6 call sites, own regression surface); 31419 (shell-chrome narrow-floor — a live-measurement/classification exercise, may close as "nothing reclaimable"). Note in the PR body.

## Global Constraints
Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-burndown`, branch `fix/schedules-closeout-burndown` off origin/dev. NEVER stash (baselines: git show / throwaway detached worktree); no pkill beyond own PIDs; `git --no-pager`; FOREGROUND pytest; EXACT pasted tail lines; tmp_path DBs. Diagnostics pin SCRIPT on logger changes; class-targeted CSS + bundle rebuilds; census/bare-type ratchet parity; geometry tests via `CSS_PATH=BUNDLED_STYLESHEET` + compositor `render_strips()` for any visibility claim (the standing lesson); escape/Text discipline. Each task's ACs get ticked (`- [ ]`→`- [x]`) and an Implementation Notes section on completion (DoD). Ordinary commit trailer:

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv

### Task 1 (=31413): Delete TaskDetail's unreachable projection-rendering layer
Deletion risk warrants isolation. Re-verify unreachability at HEAD (TaskDetail single construction site, fed only include_projections=False ReminderTasks — the ~6 LIVE None-guards at the task-file-cited region MUST survive; delete only the projection/legacy-fields branches). Tests: the retained None-guards still pass; a projection-shaped input is provably impossible (assert or type). Commit `refactor(scheduling): remove TaskDetail's dead projection-rendering layer (31413)`.

### Task 2 (=31414): Stop backfilling unset definition-config on unrelated edits
`validate_recurring_question_config` (per the task file) backfills generation/scope/finding_policy to defaults on ANY edit — an edit path must preserve unset-ness (the display-honesty fix). Read the task's ACs for the exact seam (edit vs create must diverge). Shared-seam correctness — pin: editing one field on a definition with the others unset leaves them unset in storage; a create still normalizes. Commit `fix(scheduling): don't backfill unset config keys on an unrelated row edit (31414)`.

### Task 3 (=31416+31417): Per-profile server credentials + auth_token placeholder screen
Same file (`server_context.py` per the survey). 31416: wire the existing unused `ServerCredentialScope` scoping machinery into the credential callers so `TLDW_CONFIG_PATH` profiles don't share the OS keyring (the UAT incident). 31417: screen `auth_token` for placeholder values (mirror the api-key placeholder-screening precedent) before it outranks an explicit `api_key`. Security-sensitive — tests for both ACs incl. a placeholder auth_token NOT shadowing a real api_key, and a scoped credential not leaking across profiles. Commit `fix(config): per-profile server credential isolation + auth_token placeholder screen (31416,31417)`.

### Task 4 (=31418): Remove the double base on_unmount under MRO dispatch
Repo-wide convention fix (`BaseAppScreen` + ~10 screens). The task Description's example is stale (base body simplified by 52f6a3c9c, still idempotent) — fix the real MECHANISM per the ACs, not the stale illustration. Pin: a screen unmount fires the base handler exactly once (the probe-verified count from the earlier review). Verify no screen relied on the double-fire. Commit `fix(ui): base on_unmount fires once under Textual MRO dispatch (31418)`.

### Task 5 (=31710+31711+31712 + 31713's cosmetic/run-now parts): Schedules UI polish batch
All schedules-UI-local. Per each task's enumerated findings: 31710 copy/vocabulary (header subtitle truth, "Recent runs:" wrap, terminology), 31711 timestamp/timezone display (labeled UTC, human timestamps not raw ISO, no past-dated placeholder), 31712 form/detail-pane polish (reminder-form timezone, two-step dropdowns, dead space, all-dash inspector), 31713 owner-label consistency + the Run-now-against-missing-endpoint honest copy (reuse `sync_engine._automation_capabilities_available()` — the survey's named precedent; if the reviewer wants the sync-seam touch isolated it splits to its own commit). Painted/compositor assertions for every visibility change; class-targeted CSS. Commit `fix(scheduling): UAT copy/timestamp/form polish + run-now honesty (31710-31713)`.

## After
Final whole-branch review (opus) → one wave → PR `fix(scheduling): close-out burndown (9 tasks)` → bot round → in-loop merge (required-aware: Derived-green even if the census ratchet is red-at-base) → mark the 9 tasks Done via the backlog CLI, note 31415/31419 stay open.
