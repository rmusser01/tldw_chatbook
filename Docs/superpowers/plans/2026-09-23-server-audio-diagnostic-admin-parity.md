# Server audio diagnostic admin parity — TASK-32917

ADR required: yes
ADR path: backlog/decisions/178-server-audio-diagnostic-admin-boundary.md
Reason: This aligns a connected-client security and service contract with the server's administrator boundary.

## Stage 1: Contract tests
**Goal:** Demonstrate passive health, admin-only dispatch, and explicit denial through connected and scope services.
**Success Criteria:** Targeted tests fail on the current behavior.
**Tests:** `Tests/Audio_Services/test_server_audio_services_service.py`, `test_audio_services_scope_service.py`.
**Status:** Complete

## Stage 2: Service gate
**Goal:** Check the server's effective audio diagnostic capability using the same client and translate 403s.
**Success Criteria:** Explicitly denied calls never dispatch; admin and wildcard-authorized calls dispatch; passive health stays available.
**Tests:** The targeted tests from Stage 1.
**Status:** Complete

## Stage 3: Verify and publish
**Goal:** Verify the touched scope, document the contract, and open the Chatbook PR.
**Success Criteria:** Targeted tests and style checks pass; PR links the server review finding.
**Tests:** Targeted pytest, compile/style checks, `git diff --check`.
**Status:** Complete

The first pass checked only `user.role` and was rejected in Qodo review because the server also honors wildcard permission. The server's existing current-user capabilities route now exposes the effective diagnostic decision, and the Chatbook client reads that decision. Revised verification: 22 focused audio and API client tests passed; Ruff, format, compileall, and diff checks passed on touched small files. The large API client and lazy-export module retain pre-existing lint findings and were checked by targeted tests and compileall. An existing `from_config` test remains red in the isolated checkout with `RecoveryRequired(raw_source_selection_changed)` during configuration bootstrap, before the changed diagnostic methods are called.

## Stage 4: Qualify CI and integrate
**Goal:** Keep the PR aligned with current dev and merge only after required checks pass.
**Success Criteria:** The final head has no unresolved review finding, required derived-artifact CI passes, and GitHub confirms the merge.
**Tests:** Focused audio/API tests, the diagnostic inventory checker, `git diff --check`, and required GitHub checks.
**Status:** In Progress

On 2026-09-26, dev `c4225b5d3` merged tier-2 P3c and refreshed the inventory.
The rebase preserved that newer generated inventory, which supersedes this
PR's two-counter correction. All production patches are unchanged. The official
checker passes with 598 owners and 7542 TASK-494 calls; 22 focused audio/API
tests pass with the same known bootstrap test deselected. Fresh CI is required.

The required artifact gate subsequently passed, but dev advanced to `a3001f0ee`
with theme UX and MCP character-authoring changes. The final rebase preserves
all eight PR patches. The official inventory checker passes with 598 owners
and 7543 TASK-494 calls, and 22 focused audio/API tests pass with the same known
bootstrap test deselected. The generated inventory remains exactly upstream;
the new head needs fresh required CI before merge.

The PR and UI fast lanes passed on 2026-09-26. The required inventory step failed because its summary counters were stale while its owner rows, diagnostic digests, and sink topology already matched the source. After rebasing onto dev `0053a0a40`, the committed 597 unique owner rows total 7541 TASK-494 calls; the summary still said 596 owners and 7539 calls. The official generator changed only those two aggregate fields, and a fresh checker run passes with no drift. The 22 focused audio/API tests pass; one pre-existing bootstrap test remains deselected. Final remote CI and merge remain pending. No Python source changed in this repair, so prior touched-source Bandit remains applicable.

2026-09-27: all checks, including required Derived Artifacts, passed on `00c587ef51`, but dev advanced through tier-2 P2c and OmniVoice wizard merges to `38084b68c`. Rebased cleanly; all eight prior patches are identical by range-diff. The upstream generated inventory is preserved unchanged. The focused service/scope/capability run passed 21 cases with one known unrelated from_config bootstrap test deselected. No Python source changed; prior touched-source Bandit remains applicable. The official inventory checker passes with 597 owners, 7548 TASK-494 calls and 14 sink files. Diff check passes. Fresh required CI must qualify this latest-dev head before merge.

2026-09-27 second integration: all checks and required Derived Artifacts passed on 805c20ec5, but dev advanced through the tier-2 security merge to 906b6e253. Rebased cleanly; all eight prior patches are identical by range-diff. The upstream generated inventory remains unchanged; the official checker passes with 597 owners, 7543 TASK-494 calls and 14 sink files. The focused service/scope/capability run passed 21 cases with one known unrelated from_config bootstrap test deselected. Diff check passes. No PR Python source changed; prior touched-source Bandit remains applicable. Fresh required CI must qualify this latest-dev head before merge.

2026-09-27 final integration: the requester approved the manual task-ID repair.
The older audio task keeps TASK-32917; the younger provider-engine task becomes
TASK-33009, with its status, owner, criteria and notes preserved, provenance
appended, and ADR-179's inbound link updated. Rebased onto live dev `74965f694`;
all nine prior patches are identical by range-diff. The upstream inventory is
unchanged and the official checker passes with 600 owners, 1362 TASK-492 calls,
56 TASK-31551 calls, 7550 TASK-494 calls and 14 sink files. The task-ID guard
passes across 4436 task files.

The initial four-module test run passed 25 cases and reproduced the previously
documented `from_config` bootstrap failure. That same case fails on dev
`74965f694` with `RecoveryRequired(raw_source_selection_changed)`. Its real
client construction reads TLS configuration after the per-test config redirect;
the existing `bootstrap_profile` marker retains the isolated collection-time
profile for such consumers. Applying the marker to this one test preserves all
assertions and recovery guards. All four targeted modules now pass: **26 passed,
no exclusions**. Ruff lint/format on the five touched small service/schema/test
files, compileall on changed modules, diff checks and touched-source Bandit pass
(zero findings). Server dependencies #2968 and #3002 are merged. Final-head CI,
Qodo review and GitHub merge confirmation remain required.
