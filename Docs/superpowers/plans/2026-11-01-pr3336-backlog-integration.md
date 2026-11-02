# PR3336 first-send warm-up integration plan

> **For agentic workers:** Use superpowers:subagent-driven-development for this single coupled integration, followed by one independent whole-branch review. Existing backlog execution authorization remains in force.

**Goal:** Integrate the selected warm-up PR and prevent first-send imports from classifying overlapping foreground cycles as compaction-exempt boot material.

**Architecture:** Retain the source PR's package-level imports, input pacing and boot counting optimization. Remove direct per-step/closing first-send boot freezes. At pass completion, marshal one request to the existing current attached Console idle owner. Preserve real navigation/custody admission, full collection, measured growth, stranded debt and quiet-time policy.

**Tech stack:** Python, Textual, existing CPython GC and private pytest fixtures; no dependency or new scheduler.

**Spec:** TASK33620.15.18, its existing follow-ups, ADR198 and the concrete I1 source review at C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3336-initial-review.md. Detailed repair requirements: .superpowers/preparations/pr3336/repair-plan.md. The resolved ruling and qualification limits are in .superpowers/preparations/pr3336-integration-preflight.md.

**ADR required:** yes, amend existing ADR198 before implementation; no new ADR number. **ADR path:** backlog/decisions/198-gc-policy-freeze-boot-heap.md. **Reason:** Narrow first-send boot classification to preserve the existing runtime collection/growth policy. ADR097 and ADR230 continue to govern import budgets and custody.

## Global constraints

- Selected source: 52d2bcbdfd926e43af07cb2290efa15524dcf37a; original base d424d1b6887a0f834ca4f68bcf2103e869e89a9d. Actual integration base: 3412c2b6a3c362cb710cdffdc8293a617e1bdd55. Rebase first in the existing isolated pr3298-resync/tldw_tui worktree, preserving the canonical plan and every current dev owner. Explain any dropped source commit.
- Root owns remote mutation, publication and merge. Implementer owns source/index/test/import/generator serially, creates no subagents or extra worktrees, and releases all processes before review. Main checkout and its existing interpreter/dependencies are read-only. No stash, draft merge/promotion, release, or action on3267/3297; release PR3245/3246 remain outside scope.
- The user authorizes hosted-CI and numeric-size waivers plus independent requesting-code-review instead of Qodo. Preserve correctness/security/source/custody guards, exact current numeric limits and ordered census prefix/floor. Numeric failures are WAIVED, never PASS.
- Root has chosen preparation option A: existing idle owner. The preparation's A/B approval request and historical active-worktree references are superseded. No new foreground lock, epoch, polling/retry scheduler, synthetic retirement checkpoint or debt reset.
- Preserve original boot freeze behavior and count-without-permanent-walk optimization. First-send completion requests runtime maintenance; no worker-thread collection or cached view. Current navigation admission and early global runtime custody are real owners, not controller-status/visible-task proxies.
- Preserve input gate, package/import order, optional failure isolation, source priming, shutdown, route-first behavior and existing collection/quiet/stranded-debt rules. Existing busy requests may drop; existing painted/settled requests provide later opportunities.
- TASK33620.15.18 AC2, .49 continuous-input starvation and unmeasured .50 qualification remain open. Historical .51 acceptance is narrow and not transferable to repaired source. Functional reclamation does not certify native navigation/provider latency, memory-byte bounds, or continuous-input progress.
- No broad benchmark campaign or paid-provider activity. Inspect concrete existing native setup, but if required native/raw evidence is unavailable, state it as unqualified and leave its criteria open. Targeted real-GC/owner controls remain mandatory. Preparation test lists are candidate pools, not a full sweep.
- Read AGENTS, task files, applicable ADRs, Architecture and testing/live/backlog lessons before editing. Apply .superpowers/preparations/backlog-targeted-verification-guidance.md. Keep one frozen-source test owner; distinguish intended RED from setup/teardown/precondition failures.

## Review focus

1. A real import overlaps navigation or early received custody: transient old-generation cycles are collectible after retirement, with an original-behavior retention control.
2. Baseline-before-first-look versus already-measured baseline: the repair must not hide retention in exempt boot accounting.
3. Completion callback runs after detachment, navigation or shutdown: resolve the current attached view on the UI loop and respect current owners.
4. Real idle maintenance, CollectorBusy and preexisting stranded debt retain policy; routing spies alone do not establish reclamation.
5. Historical timing and unmeasured starvation/native performance do not become current passes or closed acceptance criteria.

### Task 1: Rebase, repair unsafe first-send freezing, and qualify the integrated PR

**Files:** Actual selected source delta, including Chat/console_first_send_modules.py, Utils/ui_responsiveness.py, UI/Console_Modules/transcript.py and directly affected tests; ADR198, app-shell architecture and existing tasks. Reconcile actual census/inventory conflicts without source-era caps or counts.

**Interfaces:** Planned _request_idle_heap_refreeze(app) UI callback; existing request_console_heap_refreeze, _console_heap_busy, app navigation admission/call ledger and ConsoleRuntime.has_custodied_turns. The detailed preparation defines exact intended signatures and real-owner test controls. Adapt only to current dev interfaces, preserving their authority.

- [ ] Inspect selected source, tasks/spec/ADRs, current owners and nested guidance. Rebase onto 3412c2b6a3c362cb710cdffdc8293a617e1bdd55. Preserve the canonical plan and all actual dev/current-source work; report conflicts and any dropped commit.
- [ ] Amend ADR198 and task plan before production edits to record the existing-idle-owner decision and its performance tradeoff. Keep affected unfinished criteria open; use installed Backlog CLI for task metadata. No new ADR number.
- [ ] Establish causal real-GC RED using a blocked actual import and foreground navigation/custody lifetime, weakref-observable cycles promoted while held, original-behavior retention and unfreeze cleanup/control. Cover first-look variants in fresh disposable children. No fake GC/freeze counts, threshold changes or gc.disable.
- [ ] Remove per-step and closing first-send boot freezes and their now-unused module-count gate/import. Retain source priming and pacing. Queue one UI callback at completed pass; resolve current attached view on loop, exit for shutdown/no view, request existing idle maintenance. Use existing failure boundaries.
- [ ] Extend the existing busy predicate with actual navigation admission/lock and global early custody where current dev still needs them. Preserve every existing check and fixture contract. No new ownership state or direct maintenance from worker.
- [ ] Qualify real overlap/reclamation, early mounted received owner, callback routing/current view/shutdown, actual mounted idle maintenance, retained debt/CollectorBusy and directly affected input/import/pacing behavior. Select narrowly from the preparation; serialize, preserve failed attempts, and rerun only changed or failed cases. Do not claim performance from held barriers or recording mocks.
- [ ] Inspect concrete existing native qualification availability. Report unavailable native/proxy/raw evidence as unqualified; no new broad probe campaign. Leave .18AC2/.49/unmeasured.50 open and keep .51 historical. Update architecture/task notes with observed evidence and remaining limits.
- [ ] Validate only changed derived artifacts canonically. Preserve exact dev census prefix/floor and size limits. Run changed Python lint/format against base, introduced versus inherited debt separately, and full git diff --check BASE..FINAL. Any numeric excess remains WAIVED.
- [ ] Self-review, commit a clean head, and provide a compact immutable whole-branch diff/source/receipt package with exact base/head, commands/exits, distinct cases, source bindings and limitations. Release every source/index/test/import/generator/process owner. Root handles independent review and exact-lease publication after fresh live checks.
