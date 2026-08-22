---
id: TASK-18610
title: Pass-2 triage — runner-environment failures and remaining suite red
status: In Progress
assignee: []
created_date: '2026-08-19 11:30'
labels:
  - ci
  - testing
  - triage
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pass 1 of the TASK-18609 triage (PR: fix/task-18609-triage-pass-1) fixed the
clusters that fail everywhere and have local reproductions. This task is the
remainder, ordered by size, each with its evidence from the complete CI
inventories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 git-push-service 16 (ubuntu) / 17 (macOS): read the canary output from pass-1's CI run (`test_git_network_pin_environment_contract` prints the full stat table naming the failing predicate + directory), DONE in pass 2: the artifact longrepr named the culprit -- the factory pins `sys.executable` for SSH dispatch (no override seam), and hosted runners install Python under /opt/hostedtoolcache, which fails the predicates. Added the `python_executable` seam to `NetworkContextFactory`, a `_pinnable_python_executable()` harness helper, and a `sys-executable` row to the canary. 645 Notes/git tests green locally.
- [ ] #2 (largest sub-cluster RESOLVED; remainder open) UI 119 failures / 57 files. **The 24 `test_library_prompts_canvas` cases are FIXED** -- they waited 15s for a `#library-prompt-row-5` that never mounted while "Loading prompts..." stayed on screen; PR #1838 reworked the browse-scope settlement wait, and run 32511976568 (2026-08-21, all 12 ubuntu shards, dev@2a15a72bb) shows 338 canvas rows with ZERO stuck-loading signatures (evidence and the dispatch workaround needed to get a dev verdict at all are in TASK-18611's notes). STILL OPEN in this AC: 4 `test_settings_configuration_hub` (Console Behavior edit does not mark the draft dirty -- reproducible locally, real), ~8 NoMatches selectors, `SimpleNamespace` lacks `set_annotation_previews` (test-double drift, chat_screen.py:15822), and ~45 1-3x stragglers. Note the canvas file is red again for an UNRELATED reason since PR #1893 -- owned by TASK-19602, not this AC.
- [x] #3 TTS 8 (ubuntu): 4 `test_tts_request_admission` (publication timing — saved model_id stays Old), 1 migration atomic-move message, 1 audio_cpp guided_text flag, 1 app-lifecycle drain, ALL fixed in pass 2: the admission trio + seal test updated to the fenced-activation/no-seal contract (37da4620a semantics); the guided-recipe test re-pointed from pocket_tts_english_safetensors (voice_or_reference_required since e206d0882) to dramabox_q8_0 (optional_reference_only); the app-ownership double grew the console-runtime shutdown stage. 4,076 TTS tests green locally.
- [ ] #4 Wizards 1: `test_mounted_model_owner_timeout_fences_late_result` — retry button stays `hidden` after timeout.
- [x] #5 git-integration 2: unborn-HEAD discovery returns `repository=None` where a branch is expected. FIXED in pass 2 (PR #1833): `git show-ref --exists` documents rc 2 as "missing" without constraining stderr, but newer git prints an error line there and the service required stderr to be EMPTY -- so every fresh `git init` repo before its first commit read as "HEAD reference lookup failed". Now rc 2 + any stderr = unborn. Real product bug, not test rot; 180 Notes/git tests green, and the pass-2 CI verdict showed the unborn trio cleared on both legs.
- [ ] #6 Local-only stragglers (clean-dev macOS, none in CI inventories): (not in CI's inventory; fail on a clean dev worktree macOS): `chat_screen.py` is 21,253 lines against TASK10's 20,943-line ratchet ceiling -- a real 310-line growth since the ceiling was last set (the ratchet did its job; the ceiling needs an owner decision: extract or re-baseline). Same file trips `test_console_wave6_inventory` (`assert 21253 <= 20943`). These two were failing before the sharding fix too -- they were among the failures the cancelled runs hid. Also macOS-local: `test_legacy_server_client_builder_matches_are_listed_in_migration_audit` (migration-audit drift).
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence per cluster is in the TASK-18609 notes and the run artifacts
(ui-test-results-0..11, core-test-results-{ubuntu,macos}, run 32268704382).

**Pass-2 disposition:** git-push (AC#1) and TTS (AC#3) fixed and verified
locally (645 Notes/git + 4,076 TTS green). Remaining: UI 119 (AC#2, largest:
the library canvas cluster now filed as TASK-18611 with both failure modes
evidenced), wizard 1 (AC#4), git-integration unborn-HEAD 2 (AC#5), local
stragglers (AC#6). Also discovered: pytest-json-report under xdist records
only ~20.6k of 37k executed tests (one worker's share) -- pre-existing
reporting gap worth its own fix so future triage sees everything.

**Pass-2 CI verdict (PR #1833, rebased head d64608b84):**
- macOS core: **58 → 15** (17 git-push cleared by the python_executable
  seam; TTS drift cleared; unborn trio cleared by the show-ref fix;
  summarization pair cleared by regenerating digests on the MERGED
  content — lesson recorded below).
- ubuntu core: **47 → 30 → (final run in flight; 15 of the prior 30
  were the two git-push leftovers now closed on-head plus the unborn
  trio plus summarization pair)**.
- UI: **119 → 108**; the only never-failing file
  (test_library_prompt_collections) is a timeout-poll flake passing on
  both branch and clean dev — the TASK-18611 flaky family.
- Rebase hazard worth remembering: the summarization ledger digests are
  cut against CONTENT; CI tests the merge of branch+dev, so dev-side
  LLM_Calls edits made after the digests were cut broke the boundary on
  the merge while local branch-only runs stayed green. Always regenerate
  inventory+digests AFTER rebasing onto the target dev.

**Remaining 15 macOS (all pre-existing, runner-env):** 6 audio_cpp
real-child subprocess tests, 2 TTS profile-repository spawned
concurrency, 2 git_service runner-shutdown timing, 2 git_push_service
macOS-runner variants (subprocess TimeoutExpired on the frozen route;
writable-ancestor arrangement), 1 Character_Chat fifo asset,
Architecture 15103 ledger-exact + app-state legacy-absent (both pass on
dev checkouts everywhere tried; macOS-runner-only).

<!-- SECTION:NOTES:END -->
