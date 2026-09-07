---
id: TASK-28238
title: Worktree isolation and stale-write guard for parallel sub-agents
status: Done
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred rows C3+C7 combined, promoted by TASK-26041: their shared precondition shipped — fleet_coordinator runs concurrent sub-agent children against ONE shared working tree, so concurrent-edit races are now a live risk, not a future one. Two complementary halves: (a) optional per-child git-worktree isolation like hermes; (b) a stale-write guard on the fs tools (read-version tokens; a write whose base changed since read refuses with a diff) — optimistic locking exists only for notes today (Tools/note_management_tools.py:318).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fleet child can opt into an isolated worktree; its changes merge back explicitly, never silently
- [x] #2 fs_write/fs_edit refuse when the target changed since the tool last read it, naming the conflict
- [x] #3 Single-agent behavior is unchanged by default
- [x] #4 The refusal path is exercised by a test that races two writers
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-28228, colliding with the
"MCP-wire-server-initiated-sampling-elicitation-to-the-live-chat-provider-and-approval-surface" task that arrived on origin/dev first (dev minted 28226-28228
within the hour after this batch's sweep; re-verified at the wave-2 dev merge,
2026-09-02). Per the TASK-19601 owner rule the younger task renumbers with
provenance; it is now TASK-28238.


## Design 2026-09-02 (approved, phased)

Design approved via brainstorming. Spec:
backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md

Scoped as two phases under this task:
- Phase 1 (primary): stale-write guard on fs_write/fs_edit/fs_patch, keyed by
  (run_id, resolved_path) -- NOT per-provider-instance, because fleet children
  share one provider (a per-instance ledger would mask a sibling's race). Auto-
  records each fs_read's whole-file hash (incl. an ABSENT sentinel); refuses at
  execution when a file read this run changed on disk, naming the conflict;
  blind writes/creates proceed (single-agent unchanged). No git dependency.
- Phase 2 (later, sketch): opt-in git-worktree isolation per fleet child with
  explicit (never silent) merge-back; refuses cleanly if the workspace is not a
  git repo.

## Implementation Notes (Phase 1)

Phase 1 (stale-write guard) implemented on branch `feat/task-28238-stale-write-guard`.

**Mechanism**: read-ledger keyed by `(run_id, resolved_path) -> (sha256, size) | ABSENT`; provider-side record on fs_read (via provider-local path resolution + whole-file hash, independent of fs_read success/failure); staleness check at write-execution time (after approval) with refusal naming conflict; fs_write injects `expected_sha256` into the existing atomic CAS (reusing `write_file`'s per-target lock), fs_edit/fs_patch pre-hash each target and refuse the whole operation if any is stale; post-write re-stamp of every target.

**Files touched**:
- NEW `tldw_chatbook/Agents/fs_read_ledger.py` (ledger module, 7 tests)
- MODIFIED `tldw_chatbook/Agents/local_tool_provider.py` (record on read, check on write/edit/patch, update on success, +13 guard tests)

**AC coverage**:
- AC#2 (refusal naming conflict): `test_two_writer_race_refuses_second_writer`, `test_edit_race_refuses_second_writer`, `test_patch_with_one_stale_target_refuses_whole_patch`, `test_absent_then_created_by_peer_refuses`
- AC#3 (unchanged by default): `test_blind_write_proceeds_unchanged`, `test_edit_without_prior_read_proceeds`, `test_model_supplied_precondition_wins_over_ledger`, plus two own-chain tests
- AC#4 (racing writers): `test_two_writer_race_refuses_second_writer` (sequential `use_run_id("A")`/`use_run_id("B")` binding against shared provider)

**Phase 2 remains**: AC#1 (worktree isolation) unticked; dispatch-wiring open question (shared single registry) recorded in spec §Open questions.

## Implementation Notes (Phase 2)

Phase 2 (git-worktree isolation) implemented on branch
`feat/task-28238-phase2-worktree-isolation` across seven sub-tasks (T1-T7),
each with a review-fix follow-up commit where review found something.

**Dispatch wiring — the phase-1 sketch's per-child-registry fear, resolved**:
the provider already modeled multiple roots as `RunAdmittedWorkspaceRoot` +
`_select_admitted_root`; phase 2 just adds a second, separate map,
`LocalToolProvider._agent_roots: dict[run_id -> RunAdmittedWorkspaceRoot]`
behind its own lock, with `admit_run_workspace_root`/
`retire_run_workspace_root`. `_select_admitted_root` checks this map before
falling back to the constructor's static alias map, so an isolated child's
fs/git calls auto-route to its worktree with no `root_alias` for the model
to remember, while unmapped runs and the static-alias path are byte-for-byte
unchanged (AC#3 discipline held into phase 2). No per-run registry per
child was needed — the existing per-run-keyed authority machinery from
phase 1 already generalized.

**Worktree lifecycle** (`tldw_chatbook/Agents/agent_worktree.py`, new,
T1): `create_agent_worktree` (`git worktree add <scratch>/agent-<run8> -b
agent/<run_id> HEAD` — base ref is HEAD at spawn time; uncommitted
shared-tree dirt does not carry into the child's checkout, by design) and
`discard_agent_worktree`/`prune_stale_agent_worktrees`, all reason-coded
refusals (non-git root, unwritable worktree base) rather than a silent
same-tree fallback.

**Dual-mode merge-back** (same module, T2): `merge_agent_worktree_changes`
supports `mode="apply"` (3-way-apply the child branch's diff into the
shared tree, left uncommitted for user review; the `agent/<run_id>` branch
survives as backup) and `mode="merge"` (`git merge --no-ff`, a real merge
commit). Both refuse atomically and name the conflicting file(s) on overlap
— nothing half-applies. `preview_agent_worktree_diffstat` backs the
confirm-card preview.

**Runtime tools** (`tldw_chatbook/Agents/agent_models.py` +
`agent_service.py`, T5): `merge_agent_worktree`/`discard_agent_worktree`
join `RUNTIME_TOOL_NAMES`, `mutates`-tagged (floored to `ask`). Both
dispatch closures fail closed — no confirm surface, non-terminal child, or
unknown handle each refuse with a named reason rather than doing nothing
silently or raising.

**Spawn integration** (`agent_service.py`, T4): `spawn_subagent` gains
`isolation="worktree"` (default absent = today's shared tree, AC#3
untouched); an isolated spawn creates the worktree, admits it as the
child's run authority, and retires the mapping at child finish (the
worktree itself survives until merged or discarded). Inline (non-fleet)
spawns refuse isolation outright rather than silently running unisolated.

**Console confirm plumbing** (`console_chat_controller.py` +
`console_agent_bridge.py`, T6): `request_worktree_merge_confirm` /
`set_pending_worktree_merge` carry a payload (handle, mode, branch,
diffstat, request id) to a confirm card and back; a stale or missing
request id, a timeout, or no UI bridge all deny rather than hang or
default-allow.

**Deviation from the T5 plan (controller ruling, T7)**: T5 originally
disclosed the two merge/discard schemas under the identical bare
`fleet_active` predicate as `wait_agents`/`check_agents`/`send_to_agent`.
Sweep review found this regressed the unrecognized-model 4096-token
fallback budget: the two ~300-token schemas pushed a fleet-active +
run-log-active first request from a 47-token margin into overflow, and
`build_first_request_schema_plan`'s fallback is fit-or-nothing, so the
*entire* schema plan (including run-log) collapsed. Separately, disclosing
them was pointless in every real session today: both tools already fail
closed with "no approval surface" whenever `request_worktree_merge_confirm`
is `None`, which is always true in production (see limitation below). Fix:
a second, independent gate, `worktree_merge_enabled` (mirrors the existing
`run_skill_script_enabled` pattern) on top of `fleet_active`, threaded
through **both** plan builders — `AgentService._run_one`'s own
`build_first_request_schema_plan` call, and
`console_agent_bridge.build_console_first_request_plan` (the one the live
Console path actually uses, since `run_reply` prebuilds the plan and passes
it into `run_turn` as `first_request_schema_plan`, bypassing `_run_one`'s
own call entirely). Both pass
`worktree_merge_enabled=request_worktree_merge_confirm is not None` at
their one live call site each. Dispatch/`RUNTIME_TOOL_NAMES` were
untouched — only schema *disclosure* is gated; the closures still refuse
safely either way.

**Known limitations (stated honestly)**:
- No Console UI card is wired yet: `set_pending_worktree_merge` is defined
  on the controller but never assigned anywhere in production, so
  merge/discard refuse fail-closed with "no approval surface" in every
  Console session today, and per the T7 gate they are also not disclosed to
  the model in that state.
- Merge-back is same-turn-only: the per-run worktree/authority map resets
  each `run_turn`, so a handle cannot be merged from a later turn.
- `build_console_first_request_plan`'s two preview call sites
  (`build_project_instruction_preview_request`,
  `build_personal_context_preview_snapshot`) were left at the
  `worktree_merge_enabled` default rather than threaded, so preview and
  live can diverge once a UI card lands and starts passing a real confirm
  callable through the live path only.
- `build_first_request_schema_plan`'s binary fit-or-nothing fallback
  collapse is pre-existing fragility (not introduced or fixed here) — this
  task's gate avoids tripping it, not repairing it.
- Follow-up backlog tasks (UI confirm card, cross-turn merge persistence,
  graceful fallback-budget degradation) are being filed separately.

**Files touched** (`git diff --name-only ac3777f658..HEAD`):
- NEW `tldw_chatbook/Agents/agent_worktree.py` (lifecycle + merge-back)
- NEW `Tests/Agents/test_agent_worktree.py`,
  `Tests/Agents/test_merge_discard_worktree_runtime_tool.py`,
  `Tests/Chat/test_console_worktree_merge_confirm.py`
- MODIFIED `tldw_chatbook/Agents/agent_models.py` (tool name constants),
  `tldw_chatbook/Agents/agent_runtime.py`,
  `tldw_chatbook/Agents/agent_service.py` (spawn isolation, merge/discard
  dispatch, schema-plan gate), `tldw_chatbook/Agents/local_tool_provider.py`
  (per-run agent-root map), `tldw_chatbook/Agents/tool_catalog.py`,
  `tldw_chatbook/Chat/console_agent_bridge.py` (confirm kwarg + gate
  threading), `tldw_chatbook/Chat/console_chat_controller.py` (confirm
  card plumbing)
- MODIFIED tests: `Tests/Agents/test_agent_models.py`,
  `Tests/Agents/test_agent_service.py`, `Tests/Agents/test_fleet_runtime.py`,
  `Tests/Agents/test_local_tool_provider.py`,
  `Tests/Agents/test_trace_approval_capture.py`
- `Docs/superpowers/plans/2026-09-03-task-28238-phase2-worktree-isolation.md`
  (implementation plan), `backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md`
  (phase-2 design section)

**AC#1 coverage (worktree isolation + explicit merge-back)**:
- Lifecycle/isolation: `test_create_yields_isolated_checkout_at_head`,
  `test_create_refuses_non_git_root`,
  `test_discard_removes_worktree_and_branch`,
  `test_uncommitted_shared_changes_do_not_carry`,
  `test_prune_removes_only_dead_runs`,
  `test_create_refuses_when_worktree_base_unwritable`
  (`Tests/Agents/test_agent_worktree.py`)
- Merge-back mechanics: `test_apply_mode_lands_uncommitted_diff`,
  `test_merge_mode_creates_merge_commit`,
  `test_apply_conflict_refuses_atomically_naming_file`,
  `test_merge_conflict_aborts_and_names_file`,
  `test_clean_worktree_is_nothing_to_merge`,
  `test_apply_conflict_names_file_for_already_exists_shape`,
  `test_apply_patch_tempfile_write_failure_is_refusal_not_exception`,
  `test_auto_commit_failure_is_refusal_not_silent_nothing_to_merge`,
  `test_preview_diffstat_sees_uncommitted_child_work`,
  `test_preview_diffstat_sees_untracked_new_file`,
  `test_preview_diffstat_empty_for_untouched_worktree`,
  `test_preview_diffstat_never_raises_on_missing_worktree`
  (`Tests/Agents/test_agent_worktree.py`)
- Per-run admitted-root routing: `test_admitted_run_routes_fs_tools_to_worktree`,
  `test_unmapped_run_unchanged_and_retire_restores`,
  `test_agent_root_write_permission_enforced`,
  `test_admit_rejects_alias_colliding_with_static_admitted_root`,
  `test_admit_rejects_alias_colliding_with_another_live_agent_run`,
  `test_retire_never_drops_a_static_alias_spec_cache`,
  `test_vanished_agent_alias_cache_returns_honest_refusal_not_crash`
  (`Tests/Agents/test_local_tool_provider.py`)
- Spawn + merge/discard integration end-to-end:
  `test_isolated_spawn_writes_are_invisible_until_merge`,
  `test_isolated_spawn_refuses_on_non_git_workspace`,
  `test_isolated_spawn_refuses_without_fleet`,
  `test_plain_inline_spawn_without_isolation_still_works`,
  `test_merge_unknown_handle_refuses`,
  `test_merge_refuses_while_child_still_running`,
  `test_merge_no_confirm_surface_refuses`,
  `test_merge_deny_refuses_and_leaves_tree_untouched`,
  `test_merge_allow_apply_lands_uncommitted`,
  `test_merge_allow_merge_creates_commit`,
  `test_discard_requires_confirm_and_refuses_when_none`,
  `test_discard_removes_worktree_branch_and_entry`,
  `test_discard_unknown_handle_refuses`,
  `test_discard_refuses_while_child_still_running`,
  `test_discard_deny_refuses_and_leaves_worktree_untouched`,
  `test_merge_refuses_without_a_local_provider`,
  `test_merge_confirm_payload_carries_uncommitted_diffstat`,
  `test_discard_confirm_payload_carries_diffstat`
  (`Tests/Agents/test_fleet_runtime.py`)
- Runtime tool registration/dispatch: `test_names_are_registered_as_runtime_tools`,
  `test_loop_dispatches_merge_to_the_injected_callable`,
  `test_merge_defaults_mode_to_apply_when_omitted`,
  `test_loop_dispatches_discard_to_the_injected_callable`,
  `test_unwired_merge_falls_through_to_the_permission_gate`,
  `test_unwired_discard_falls_through_to_the_permission_gate`
  (`Tests/Agents/test_merge_discard_worktree_runtime_tool.py`)
- Console confirm round-trip + schema-gate: `test_no_ui_bridge_denies_immediately`,
  `test_allow_round_trip`, `test_deny_round_trip`, `test_confirm_timeout_denies`,
  `test_confirm_payload_carries_handle_mode_branch_diffstat_and_request_id`,
  `test_decision_shape_matches_what_agent_service_reads`,
  `test_stale_request_id_is_dropped`,
  `test_console_plan_discloses_worktree_merge_schemas_when_enabled`,
  `test_console_plan_omits_worktree_merge_schemas_when_flag_is_omitted`,
  `test_bridge_forwards_the_confirm_kwarg_to_run_turn`,
  `test_bridge_forwards_none_when_omitted`,
  `test_bridge_enables_worktree_merge_disclosure_when_confirm_is_wired`,
  `test_bridge_disables_worktree_merge_disclosure_when_confirm_is_absent`
  (`Tests/Chat/test_console_worktree_merge_confirm.py`)
- Exact-enumeration regressions caught by sweep (T7):
  `test_runtime_tool_names` (`Tests/Agents/test_agent_models.py`),
  `test_first_request_plan_contains_exact_named_agent_and_fleet_schemas` +
  `test_first_request_plan_adds_worktree_merge_schemas_only_when_enabled`
  (`Tests/Agents/test_agent_service.py`)

**Task hygiene**: the backlog CLI is unavailable in this worktree, so the
frontmatter `status: Done` edit and the AC#1 checkbox above were made
directly in this file rather than via `backlog task edit`.
