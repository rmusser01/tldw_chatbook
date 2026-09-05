# Chunking Lab PR startup-budget correction

> Execute inline using systematic-debugging and test-driven-development, with verification checkpoints.

**Goal:** Unblock PR #2421 by correcting the inherited startup import breach without raising the 972-module cap.
**Architecture:** Preserve controller ownership and handoff validation. Load Environment gatherers at the first admitted refresh; load vLLM setup types only when validating a real target.
**Tech stack:** Python 3.12, Textual 8, pytest.
**Spec:** TASK-31645 AC19 and the user's explicit authorization to repair the inherited startup regression before merge.

ADR required: no new ADR.
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: apply the existing deferral policy without changing service contracts. ADR-118 remains applicable to the Lab.

## Constraints

No raised limits, skipped guards, dependency changes, or full-suite sweep. Preserve the existing isolated worktree and UAT evidence. The user now authorizes publication and merge after current-head checks and review; this supersedes earlier publication/merge exclusions.

## Correction and verification

Measured refinement: the four original modules leave the mounted graph, but
the immediate scheduler can legitimately add emergency-stop and heartbeat
modules before readiness (observed 973 and 974). Preserve its time-sensitive
start policy. Shed two further first-use dependencies instead: subscription
credential support outside Anthropic readiness/dispatch, and the custom-PII
regex worker when no custom detection runs. Move imports only, retain all
credential opt-in/fail-closed and masking logic, and run their existing behavior
tests plus isolated import regressions. This is ADR-097 response 2, not a change
to authentication, privacy, or scheduler policy.

- [x] Reproduce the census failure before production changes: 976 modules versus 972, matching dev CI run 33979515367 and PR run 33980917972.
- [x] Add isolated-import regressions in `Tests/Packaging/test_console_interaction_import_closure.py`: controller construction plus closed-rail refresh must not load `Workspaces.environment_status`/`git_workspace`; importing handoff contracts must not load `UI.LLM_Management.vllm_setup`. Observe both fail before changing production.
- [x] In `UI/Console_Modules/environment.py`, replace eager scanner construction with `self._scanner = None`; import gatherers and initialize the scanner once in `_dispatch_local`, after refresh admission and before worker dispatch. Import the net gatherer in `_dispatch_net`. Preserve the per-controller scanner cache and worker/landing semantics.
- [x] In `UI/Navigation/vllm_handoff.py`, keep annotation-only types under `TYPE_CHECKING`; import exact runtime types locally in `_intent_fields_from_target` and `owner_has_current_intent`. Retain every exact-type, generation, endpoint and readiness check.
- [x] Update consumer tests to patch helpers at their defining modules, not removed eager-import aliases. Add all six deferred module names to the existing absent-at-ready contract without altering its cap or sampling; add RED/GREEN regressions for subscription and custom-PII first-use boundaries.
- [x] Run new regressions, Environment controller/wiring/state/gatherer tests, pending handoff and vLLM tests, Lab screen/recovery/results, and the complete selections in `.github/workflows/perf-guard.yml`. Record any genuine baseline failures explicitly; do not waive new failures.
- [x] Run scoped static checks, all derived-artifact preflight checks, and review the diff. Record evidence in `Docs/Chunking_Lab_Verification.md` and TASK-31645 notes.
- [ ] Fetch dev again, rebase if needed, push with a precise lease if history changes, address current-head Qodo findings, and merge only with passing checks and the expected head SHA.
