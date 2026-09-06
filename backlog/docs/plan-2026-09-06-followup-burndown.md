# Schedules/infra follow-up burndown (round 3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Close the five follow-ups surfaced by the close-out burndown reviews: 31821, 31824 (credential layer), 31822 (Textual lifecycle), 31825, 31823 (schedules UI).

**Spec:** the five backlog task files ARE the spec. Supporting context: `backlog/docs/lessons-textual.md` (MRO double-dispatch lesson, 31822), the redesign spec §5 (31823), task-31419's Implementation Notes measurement (31825).

## Global Constraints
- NEVER `git stash`; baselines via `git show <rev>:<path>` or throwaway detached worktrees, removed after.
- FOREGROUND pytest only; tmp_path DBs; exact pasted tail lines from the final run.
- Diagnostics pin is a SCRIPT (`scripts/check_persistent_diagnostic_inventory.py --write` + commit JSON), not a pytest.
- Geometry/visibility tests need the bundled stylesheet (`ConsolidatedCSSApp` harness — a bare App with CSS_PATH measures nothing); CSS targeted by class, never ancestor-scoped bare type.
- Vendored code (`Third_Party/`) is out of scope for every task.

### Task 1 (=31821): Route auth-account login bearer writes through the per-profile credential scope
`Auth_Account_Interop/auth_account_scope_service.py` (~:145,157) writes the login/account bearer via the PLAIN legacy store API (bare server_id slot), bypassing 31416's scope — in scoped mode a non-default profile's bearer lands where the default profile reads first. Route the writes through a profile-scoped provider method (reuse the 31416 machinery: `server_profile_id` scoping in `runtime_policy/server_credentials.py` / `server_context.py`; do NOT invent a parallel scheme). ACs: scoped writes; non-default bearer unreadable by default profile (pin both directions); default single-profile unaffected (no re-auth — the legacy-scope equivalence 31416 AC#4 used). Commit `fix(auth): route account bearer writes through the per-profile credential scope (31821)`.

### Task 2 (=31824): Scope clear_server to a single server under profile isolation
`runtime_policy/server_context.py` (~:455): in scoped mode `clear_server` filters on `server_profile_id`, clearing every server in the profile. Make it clear only the target server (scope key = profile AND origin), pin with a multi-server-profile test (clear A leaves B), default single-server unaffected. Rider (same area, from the same Qodo round): complete the three flagged docstrings — `resolve_tldw_api_auth_token` (config.py ~:1406), the profile helper's Returns (server_context.py ~:46), the scoped credential methods (server_credentials.py ~:45) — Google style. Commit `fix(config): clear_server clears one server, not the profile (31824)`.

### Task 3 (=31822): Convert remaining super().on_mount() calls under MRO dispatch
The mount-side twin of 31418 (see lessons-textual.md). ~19 live `super().on_mount()` calls each double-fire a separately-MRO-dispatched base. For each site CLASSIFY first (base handler in its own `__dict__` = redundant → remove with the standard comment; genuine run-once need → the `BaseWizard._post_mount_hook` plain-method pattern; `Third_Party/` untouched). Correct `change_review_screen.py`'s two on_mount docstrings that misdescribe the mechanism as "ordinary attribute lookup … SHADOWS". Extend the AST guard (`Tests/UI/test_on_unmount_mro_convention.py`) to cover `on_mount` (allowlist any legitimate plain-method exceptions), revert-checked. Behavioral risk is real: an on_mount that RELIED on running the base body twice, or on the in-line ordering of the base call, must be caught — read each site's base handler before removing. Commit `fix(ui): base on_mount fires once under Textual MRO dispatch (31822)`.

### Task 4 (=31825): Wire DestinationHeader's dormant compact density to a height-based trigger
Shared layer: `UI/Workbench/workbench_widgets.py` (~:164) ships a `density="compact"` CSS rule no caller triggers; no height-based responsive logic exists in the workbench layer. Wire an automatic trigger (app/screen height threshold — pick the mechanism from how the widget learns its size; on_resize or a checkpoint watcher) so every DestinationHeader user benefits with no per-screen overrides. Geometry tests with the bundled-CSS harness: compact rows measurably fewer at 80x24, normal density unchanged at standard sizes; `Tests/UI/test_schedules_responsive_floor.py` stays green minus the known pre-existing red (`test_the_docked_task_detail_pane_scrolls_to_reveal_history_past_the_fold`, fails at base). Commit `feat(workbench): DestinationHeader auto-compact below a height threshold (31825)`.

### Task 5 (=31823): Schedules detail-pane kebab — Duplicate / View runs / View results
Redesign spec §5's deferred affordance: Duplicate, View runs, and View results reachable from BOTH detail panes (kebab menu or equivalent per-action controls — pick the shape that fits the existing lifecycle row in `task_detail.py`/`definition_detail.py`). Disabled states follow the lifecycle-row disabled+reason idiom (UX-073 — grep for it). Existing lifecycle actions unaffected (pin). View runs/View results should navigate to the existing runs/results surfaces (find how the workbench reaches them today — reuse, don't rebuild); Duplicate creates a copy via the existing service create path with a name disambiguator. Commit `feat(scheduling): detail-pane Duplicate/View-runs/View-results affordance (31823)`.

## After
Final whole-branch review (most capable model — 5 tasks, shared credential + UI seams) → one fix wave → PR `fix: schedules/infra follow-up burndown (5 tasks)` → looped update+watch+merge → mark the 5 tasks Done.
