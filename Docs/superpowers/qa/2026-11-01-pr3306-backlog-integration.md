# PR3306 backlog integration — 2026-11-01

Source head: `82bc5fcd6e70eafaf73e263606a72e4df0f10ce8`.
Tested dev base: `621b8f1550a9f5a93b36c7a1b5676d75d5e1b793`.

The rebase retains current dev's workspace activation, conversation acknowledgment,
continuation and recovery machinery. The remaining production change resolves
character defaults from the active worked chat's provider/model, with canonical
and control fallback for pristine/no-source launches. Card prompts and greetings
remain independently owned; source prompts and prefills are not cloned.

Additive CI reconciliation retains all 1,253 dev census entries in their original
order and appends five prompt nodes, for 1,258. Both admission-sensitive launch
selectors remain registered. No production guards or tests were weakened.

The duplicate TASK-37600 was resolved by creation/add-commit provenance: the older
character task retains its ID. The younger SSE task is TASK-38328 with current
inbound references updated and explicit provenance. Prior PR3298 integration
records describe their historical snapshot and remain unchanged.

## Fresh local evidence

- `../r-4c6d381a`: 48 passed, zero failures/errors/skips. Fifteen controller
  selectors cover worked/pristine/source/default/duplicate/concurrent behavior;
  both complete prompt and real model-picker modules ran. All six mounted
  Apply/Cancel entry-route combinations passed. Source hashes stayed unchanged.
- `../r-7e9dbb49`: 336 passed and three existing size-limit failures, zero
  errors/skips; source hashes stayed unchanged. All CI/census contracts and the
  changed controller's size check passed (6,981 lines, existing limit 6,982).
  The failures are console_chat_store.py (22,361 > 22,338),
  console_trace_service.py (6,787 > 6,765), and app_command_providers.py
  (1,133 > 1,123). All three files and the ratchet test are identical to the
  tested dev base after newline normalization. They are covered by the user's
  explicit size-limit waiver; their budgets were not changed.
- The census checker passes at 1,258 entries. The task-ID/path checker passes
  across 6,041 task files. `git diff --check` passes.
- Scoped Ruff across seven changed Python files adds no diagnostics relative to
  dev. Four files pass formatting; CI-contract, session and wiring files have
  pre-existing formatter debt on both base and candidate. No formatting sweep.

The 2026-10-29 screenshots and loopback-provider UAT are historical source
receipts. Native UAT was not rerun for this integration. Fresh mounted tests
qualify the current integrated launch/model-picker behavior. No full suite ran.

## Review and merge policy

A fresh independent reviewer used `superpowers:requesting-code-review` and its
reviewer template. Review of the rebased production/tests and identity delta
found no Critical, Important or Minor findings; the final exact-head receipt is
stored beside this worktree as `../pr3306-integration-review.md`.

The user's later authorization replaces Qodo with this independent review and
waives hosted CI and size limits for backlog clearance. Required thread
resolution and a live non-draft/head/base check remain mandatory. This supersedes
historical queue/CI requirements in the original qualification notes. No release
or main promotion is included.

ADR required: no. Existing ADR-006/095 and the TASK-15120 workspace ownership
ruling apply; no storage, security or service boundary is introduced.
