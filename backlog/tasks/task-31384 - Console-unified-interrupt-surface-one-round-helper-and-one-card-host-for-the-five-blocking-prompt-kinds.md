---
id: TASK-31384
title: >-
  Console unified interrupt surface: one round helper and one card host for the
  five blocking prompt kinds
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-04 19:29'
updated_date: '2026-09-05 01:10'
labels:
  - console
  - refactor
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console controller now carries FIVE hand-cloned copies of the same blocking round loop -- MCP approvals, skill install, skill script, worktree merge, and (since PR #2379) ask_user questions -- each with its own registry, lock, retained-payload map, marshal, remount-at-activation trio, and revocation leg, and ChatTaskCards routes each kind to its own bespoke card. Every new kind is a fourth or fifth copy of ~150 lines and three activation-site edits, and every bug fix (PR #1836's round-keying, M2's atomic busy check) has to be applied per copy. Sub-project C of the design spec (2026-08-19-console-user-interaction-design.md section 4) is the extraction: one _run_pending_round helper the kinds parameterise, and one card host routing by kind. A first spine (C1) was designed and implemented on PR #1903 and closed unmerged on 2026-09-04 after dev's approvals-verdict rewrite outran it; its design doc (2026-08-20-console-interrupt-host-design.md) and plan are recoverable from that branch's history and remain valid. The extraction is a refactor with a parity oracle: the existing interrupt battery must pass unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One shared round helper backs all five kinds; the per-kind request_* methods keep their names and return shapes
- [x] #2 One host in the task-card slot routes payloads to cards by kind, including lazy mounting for kinds that stay off the boot path
- [x] #3 The interrupt battery (approval, skill-install, skill-script, worktree-merge, ask_user suites and the concurrency suites) passes with the same failure set as clean dev
- [x] #4 Per-kind behaviour differences (FIFO queueing for approvals vs busy for questions; verdict keying; revocation) are expressed as parameters, not copies
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline the interrupt battery at the current dev tip in a detached worktree.
2. Port the C1 host module (Chat/console_interrupt_rounds.py, recovered from PR #1903) and its unit tests, extended to five kinds; alias the legacy registry/payload/lock attribute names to the host in the controller constructor; the payload helpers become shims.
3. Migrate the five request_* bridges onto host.run_round with per-kind hooks (approvals: detached announce + decision stamping; questions: busy fast path stays outside).
4. Consolidate the three activation-site re-derive trios into host.remount_for_session and the revocation sweep into a host-side per-kind sweep parameterised by each kind's closed-decision stamp.
5. Card slot: a kind-to-card routing table in ChatTaskCards plus an identity guard on the approval card so re-syncs keep unsubmitted selections (the question card already has one).
6. Parity: the battery passes with the same failure set as clean dev; new host unit tests cover arm/park/promote/timeout/revoke per kind.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Ported the C1 `InterruptRoundHost` (design + host module recovered from closed PR #1903, head `09dbc187e8`) into `tldw_chatbook/Chat/console_interrupt_rounds.py`, extended from four kinds to five (`question` added), and moved every blocking-prompt bridge in `ConsoleChatController` onto it.

- **One lock, one registry/payload map per kind.** The controller's fifteen legacy names (`_approval_state_lock`, `_pending_*_rounds`, `_parked_*_payloads`, ...) are identity aliases of the host's objects, so every existing test and helper reads the same state. The six PR0 payload helpers are shims over the host (`_kind_for_store` recovers the kind by identity).
- **`run_round()`** is the single arm / mount-or-park / wait / teardown loop. Per-kind differences are parameters and hooks, not copies: approvals pass `on_cancelled`/`on_timeout` decision stamps, `before_wait` (permission summary), `announce_detached`, and an `on_teardown` that retains the payload in the `finishing` phase; skill-install and worktree-merge opt out of revocation (`check_revoked=False`); the question bridge keeps its atomic live-check-and-register fast path outside the host and maps `revoked` to a cancelled result.
- **Revocation and activation.** `revoke_for_run()` sweeps the named kinds under one lock with each kind's closed-decision stamp (`_REVOCATION_STAMPS`); the three `_revoke_*_rounds` methods are one-kind wrappers. `remount_for_session()` replaces the four per-kind re-derive calls at each of the three session-activation sites (the approval block there is untouched).
- **Card slot.** `ChatTaskCards.sync_state` routes payloads through one `_routes()` table; the question card entry appears only once the card has been lazily created, so it stays off the boot path (census 967/972).

**Deviation from the plan.** Step 5's "identity guard on the approval card" was not added: `ChatApprovalCard.set_batch` already keys on `round_id`: "Repeated resume-state syncs for one unchanged, identified round preserve its mounted controls" (its docstring), so there was nothing to guard.

**Review round.** A read-only bridge-by-bridge diff review (opus) found eight confirmed and two suspected behaviour deltas in the first cut; all are fixed and pinned:
- teardown promotion of a queued approval sibling no longer fired the ADR-090 permission summary -> the hook is a per-kind `after_remount` registry on the host that every remount path applies (`test_after_remount_hook_fires_for_a_teardown_promoted_head`);
- the payload shims required the host and identity-matched stores, breaking bare-double tests and the skill bridges' own dicts (`Tests/Chat/test_permission_summary_wiring.py`, which the first battery had missed) -> the payload layer is four module-level functions over `(lock, store)` shared by host and shims;
- the question marker, the revoked-approval audit rows and the approval decision snapshot had moved after the registry pop/unpark -> `run_round(on_outcome=...)` runs them after the wait and before teardown (`test_on_outcome_runs_before_the_registry_pop_and_unpark`); the approval "finishing" retention now happens only after a completed wait, never on an exception mid-wait;
- registration had moved after the timeout config read for four bridges -> they pre-register as before (`run_round` re-registers the same object);
- `_approval_view_is_detached()` was sampled before the park -> `announce_detached` is now evaluated after it and returns True when it announced;
- the question card was created before the three fixed cards synced -> `_routes()` is a generator;
- `InterruptRoundHost.resolve` had no production caller -> removed.
The eleven shim-using test files the first battery missed were added to it and baselined on clean dev.

**Qodo round 1 (PR #2396).** `run_round` refuses a state already stamped revoked at entry (a sweep can land between a bridge's pre-registration and host entry; the old path wrote it back, parked and mounted it for one poll tick); the four activation-site kinds are one `SESSION_REMOUNT_KINDS` constant; Google-style sections on `run_round` and the payload helpers.

**Boot census (ADR-097).** The controller is constructed before UI-ready, so `Chat/console_interrupt_rounds.py` is resident at boot even when imported lazily. Dev sits at 971/972 after #2391 and CI's reading of the same head swings by about two between runs, so the new module is paid for by demoting `Chat.console_visual_transcript` (PIL-backed visual compaction, used only inside `_apply_conversation_memory_preflight`) to a method-local import: the branch reads 971 locally, exactly dev's number. The host itself is still imported lazily (constructor and shims) so nothing else drags it in.

**Evidence.** Interrupt battery (22 files, listed in the PR) on this branch: 19 failed / 374 passed; clean `origin/dev` (`be2c3bf8ec`, detached worktree): 20 failed / 373 passed; the branch's failure set is a strict subset of dev's (the one extra dev failure, `test_navigation_guard_survives_stay_then_renavigate_then_leave_by_coordinates`, is a coordinate-click flake). Host unit tests: `Tests/Chat/test_console_interrupt_rounds.py` (27) and `Tests/Chat/test_console_interrupt_host_wiring.py` (alias/setter identity). Preflight: diagnostic inventory drift reviewed (five per-bridge teardown debug lines -> one host line interpolating only the kind key) and regenerated.

**Files.** `tldw_chatbook/Chat/console_interrupt_rounds.py` (new), `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`, `Docs/security/production-diagnostic-inventory.json`, `Docs/superpowers/specs/2026-08-20-console-interrupt-host-design.md`, `Docs/superpowers/plans/2026-08-20-console-interrupt-host-c1.md`, tests above.
<!-- SECTION:NOTES:END -->
