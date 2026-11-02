# PR3338 send-start lifecycle integration plan

> **For agentic workers:** Use superpowers:subagent-driven-development for one coupled integration and independent superpowers:requesting-code-review before publication. Existing backlog execution authorization remains in force.

**Goal:** Integrate each supported Console send's own bounded start window and directly verify cleanup when work ends without a provider request.

**Architecture:** Preserve the original context-carried SendStart and store-keyed observational registry. Runtime custody and controller submission remain the real execution owners; timing hints confer no authority or proof of native retirement. Add four focused lifecycle cases, not a new scheduler or ownership layer.

**Tech stack:** Existing Python/Textual/asyncio/pytest environment; no dependencies.

**Spec:** TASK-36610; existing ADR094/219/230; precise lifecycle assertions and preservation matrix in .superpowers/preparations/pr3338/repair-plan.md; initial source review at C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3338-initial-review.md. Preparation is now authorized for execution subject to the current refs and overrides here.

**ADR required:** no new ADR. **ADR paths:** backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md, 219-console-chat-destinations-and-bounded-starts.md, 230-console-send-preparation-and-io-ownership.md. **Reason:** Pin existing timing and lifetime contracts; introduce no new authority, storage or cancellation policy.

## Global constraints

- Selected source `808536969b4dcc0a8d6805ff75942ad5e66ad5d3`, source base `94a8a85f03c0480ae16c21382f26be71b2399421`, actual integration base `3da5f6451e4989b6b58ea4c4b344a483ffc30f34`. Existing isolated branch codex/pr3338-resync starts at selected source. Rebase first, preserving every concurrent contributor/base change and this untracked plan. Explain any skipped source commit.
- One exclusive source/index/test/import/collection/generator owner. Root owns remote mutation and publication. No subagents or additional worktrees, stash, main-checkout edits, dependency changes, draft promotion/merge, excluded PR3267/3297 action or release/main work. PR3245/3246 remain excluded.
- User waives hosted CI and numeric size limits and replaces Qodo with independent requesting-code-review. Preserve tests, security/privacy/source/custody guards, current caps, exact census prefix and floor. Numeric failures are WAIVED, not PASS; historical test counts are not fresh evidence.
- Original source semantic delta only: preserve received-intent terminal_callback forwarding, acknowledgement custody callback, turn_id/mark_admitted/return order, captured/sent draft ownership, preparation/admission/recovery/native drain, checked display refusal and asynchronous presentation preparation. Preserve current first-send imports, emergency_stop_hold and PR3336 idle-GC changes. Never restore older whole files over current owners.
- Add only test_custody_task_completion_retires_only_its_send_start (three modes) and test_exceptional_submission_retires_only_its_send_start as defined in the preparation. Real helpers, weak-referenceable identity owners, long_limit 600.0, same-store held sibling and separate-owner control, and post-registration callback signals are required. No fake _OPEN/SendStart.end/Task or sleeps to expiry. Retrieve all exceptions and clean every owned task in finally.
- The three task cases must detect a removed production done callback. A narrowly scoped reversible one-line mutation is authorized; verify intended failures, restore exact source, then qualify the positive cases. Removing only submission finally is not a causal oracle for exceptional cleanup because its inner handler also cleans up. Do not claim otherwise.
- TASK38460/38461 remain open. Reported owner acceptance of approximately +38 ms first-send bound and Inspector timing remain historical, with unavailable raw/sign-off evidence and mixed-head/interval limitations disclosed. No new latency certification, live-provider or broad benchmark campaign.
- Installed interpreter is C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe, read-only. Create private per-process roots under C:/Users/GDesktop-1/.codex/pr3338-private-fixtures-20261101 using existing Tests.windows_private_fixture_runner ownership machinery. Reference runner: .superpowers/sdd/2026-11-01-pr3336-backlog-integration/run-targeted-private.py. Adapt only inside this plan workspace; do not rewrite old receipts, normalize ACLs or weaken source guards.
- Enroll exact refs, changed sources, selected tests, applicable conftests and imported local fixture closure. Keep raw config/logs private, export content-free outcomes/hashes. Print full exec return objects, retain session IDs and drain to explicit exit before dependent work. Empty early output is not command completion. Commit with normal hooks; no hook-bypass request.
- Read AGENTS, actual tasks/ADRs/architecture and relevant testing/live/backlog lessons; apply .superpowers/preparations/backlog-targeted-verification-guidance.md. Preparation's fourteen regression files are a candidate pool, not a required sweep. Run only nodes needed for actual changed seams; rerun only amended/failing cases or concrete uncovered concerns.

## Review focus

1. Cancel before first step, return or exception before provider entry closes exactly its own start without relying on expiry or sibling masking.
2. Another store and overlapping same-store task stay open until their own retirement; actual callback delivery is observed.
3. Current receipt/acknowledgement/draft/native ownership survives integration, including early received-intent terminal callbacks.
4. Timing hints stay observational and the three-second cap, lazy Task construction, context propagation and auxiliary-request boundary remain explicit.
5. Current import order, GC policy, ordered census prefix/floor and historical performance qualifications survive without relaxed guards or invented passes.

### Task 1: Rebase, pin lifecycle cleanup and qualify current integration

**Files:** Source PR's actual changed files, especially Chat/console_send_start.py, console_runtime.py, console_chat_start.py, console_send_diagnostics.py, console_first_send_modules.py; UI/Console_Modules/config_sync.py and wiring.py; affected tests, UI census and existing task/lesson files. New lifecycle cases belong only in Tests/UI/test_console_send_start_window.py. Commit this canonical plan.

**Interfaces:** Real start_custody_task(owner, coroutine), submission_send_start(owner), send_start_open(owner, long_limit) and original SendStart/provider notification. Follow the exact preparation test recipe; adapt existing owners to actual current dev without adding another authority.

- [ ] Read requirements/current owners and rebase onto the exact integration base. Preserve all source commits, current callbacks/imports/census/lessons and report conflicts/skips. Record existing ADR disposition and the narrow test repair in TASK36610's plan/notes through installed Backlog CLI.
- [ ] Add the three no-provider custody completion cases and exceptional submission case with the specified non-vacuous controls and bounded cleanup. Verify syntax/lint prerequisites before collection. Establish focused missing-callback mutation failures at the intended owner-closed assertions, restore exact source and run positive cases.
- [ ] Qualify the existing send-start cases and only directly affected current receipt/custody/acknowledgement/submission/display/import seams. Inspect actual fixtures/readiness before invoking them; keep one frozen source/process owner and distinguish setup/teardown/source failures from behavioral results. Diagnose unexpected failures on exact evidence, not inherited author claims.
- [ ] Preserve current ordered census entries and floor; append the new whole-file selector exactly once and record actual collected case count. Validate changed derived artifacts canonically and run scoped lint/format with introduced versus inherited findings separated. Do not run unchanged performance/census campaigns automatically.
- [ ] Update task/architecture notes only as needed for real integration and evidence; keep follow-ups and historical latency limits explicit. Self-review, commit a clean head and run full BASE..FINAL whitespace validation.
- [ ] Write task-1-report.md and final-qa.md with exact refs, commands/exits, distinct latest case identities, source bridges, failures/waivers and limitations. Provide a compact whole-branch patch/hash manifest that distinguishes Git blobs from raw checkout bytes. Release every source/index/test/import/generator/process owner. Root obtains independent whole-branch review and performs fresh eligibility/exact-lease publication.
