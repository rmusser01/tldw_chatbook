---
id: TASK-42659
title: Re-pin the voice-pair locator inventory at 31 so the Perf Guard voice step passes on dev
status: Done
assignee:
  - '@claude'
created_date: '2026-11-27 20:33'
updated_date: '2026-11-27 20:33'
labels:
  - dev-red
  - tests
  - perf-guard
dependencies:
  - TASK-42517
references:
  - Tests/Performance/test_speculative_voice_latency.py
  - Packaging/speculative_voice_history_gate.py
  - .github/workflows/perf-guard.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Perf Guard job "UI latency guardrails" has failed on every dev push since 2026-11-26 04:28 UTC (run 45755430693, dev 6e41919880) in its step "Run Console tab-publication and voice gates". `Tests/Performance/test_speculative_voice_latency.py::test_completed_pair_locator_inventory_is_complete_and_stable` asserts 30 locators and measures 31. The job stops at its first failing step, so the keystroke work census, the Console storage-unit ratchet and the compact-setup gates run on no pull request that reaches that step. Found on 2026-11-27 while verifying PR #3575 (TASK-40608), which inherits the red from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The commit that added the 31st locator is named, and the entry is either confirmed as intended or removed at its source
- [x] #2 The locator inventory node passes with exact pins, not a range
- [x] #3 The step "Run Console tab-publication and voice gates" passes on the branch
- [x] #4 The TASK-36510 strict-xfail history node keeps its marker and its body
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the three locator sets the gate reads with `git log -S` to the commit that added the entry, and read its intent.
2. If the entry is intended, re-split `_locator_distribution` so the forbidden locators sum to 1,000 with same-table columns sharing a count.
3. Re-pin the inventory node (locator count, distribution rows, the mix-in table's locators) and say why in a comment.
4. Run the workflow step, the other importers of the gate helper, the light tier's selection and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** b40d12adb3 (2026-11-24, TASK-42517 review) added `("console_prompt_mixin_attempts", "result_revision_id")` to `_VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS`. It reached dev through PR #3781 (merge 4dd7992120, 2026-11-26 04:27 UTC). The entry is intended. The column arrived with schema v86 (7970fbcada) and had never been classified, which kept `test_voice_promotion_locator_inventory_covers_owned_message_revision_tables` red. The same commit gave the column its v88 index, plan pin and census row. It did not touch the perf gate, which reads the same three sets: 4 mandatory + 21 forbidden message + 6 forbidden revision = 31 locators (was 30), 27 of them in the seeded distribution (was 26).

**Why PR #3781 did not show it.** Its Perf Guard run (45733242633) failed one step earlier, at "Run boot budget ratchets": 689 app modules against a limit of 688, the same assertion dev b534feabd3 failed (run 45726512388). So the voice step was skipped on the PR that broke it. PR #3785 fixed the boot budget in the same stack, and the first dev run after the merges went red at the voice step.

**Change.**
- `Packaging/speculative_voice_history_gate.py`: `_locator_distribution` gives the two checkpoint columns 50 rows and the other 25 locators 36. The old 44/38 split summed to 1,038 and raised `completed-pair baseline distribution drifted`. The hunk is byte-identical to the one on PR #3345's branch, so that branch re-syncs without a conflict here. PR #3771 splits the same rows 38/37 and will conflict on these lines whichever lands first.
- `Tests/Performance/test_speculative_voice_latency.py`: the inventory node pins 31 locators, 27 distribution rows and the mix-in table's three locators. A comment names the commit and says what a new locator has to move.

**The history node, unchanged on purpose.** `test_completed_pair_history_gate_meets_large_history_thresholds` keeps its strict xfail and its body. What it fails on did change:
- On dev it xfailed at the distribution check above, before it measured anything (`--runxfail` on c0a999302f).
- With the split repaired it runs the gate (about 3 s) and fails only at its own count pin, `assert report["inventory"]["count"] == 30`, in 3 of 3 `--runxfail` runs. `scan_count == 0` passes.

So the marker's stated reason no longer holds on dev: v88 indexed the columns. Removing the marker, re-pinning that count and dropping the gate's `ANALYZE` are on PR #3345's branch (TASK-36510 is Done there) and on #3771's. Re-pinning the inner count here without removing the marker would make the node XPASS and fail the step. A probe of the gate itself (3 runs, Windows, SQLite 3.49.1) had scan count 0 in all three. Two runs passed. One failed on a 160.9 ms fresh-commit large p95 against the 100 ms limit, the host-timing class TASK-38712 describes.

**Also repaired.** 21 nodes in `Tests/Packaging/test_voice_qualification_manifest.py` failed on dev with the same distribution error, because the manifest validator calls `_locator_distribution`. They pass now. That file runs in `release-voice-aec.yml`, and `Packaging/qualify_speculative_voice.py` raised for the same reason.

**Verified** on 2026-11-27 against dev c0a999302f (Windows 10, Python 3.12.10, SQLite 3.49.1, private test profile):
- The step's command, `pytest Tests/Performance/test_console_tab_sync_source_ownership.py Tests/Performance/test_speculative_voice_latency.py -v --timeout=300`. Base: 1 failed (`assert 31 == 30`), 25 passed, 1 xfailed. Branch: 26 passed, 1 xfailed.
- The four `Tests/Packaging` files that import the gate (`test_voice_qualification_manifest.py`, `test_qualify_speculative_voice.py`, `test_speculative_voice_soak.py`, `test_voice_source_digest.py`). Base: 22 failed, 90 passed, 1 skipped. Branch: 1 failed, 111 passed, 1 skipped. The one left, `test_checked_in_authority_is_valid_hard_off_package_data`, fails the same way at base and only on this checkout: `core.autocrlf=true` gives the working-copy manifest CRLF bytes, and the blob's sha256 matches the pinned identity.
- The light tier's selection for this diff (`scripts/ci/tier_policy.py`), one process per file as `run_light.py` runs them: `Tests/Performance/test_speculative_voice_latency.py` 2 passed, 1 xfailed; `Tests/test_smoke.py` 16 passed; `Tests/MCP/test_approval_timeout_policy.py` 21 passed; `Tests/Chat/test_console_approval_scope_journeys.py` 15 passed.
- `./scripts/preflight.sh`: 12 of 13 checks pass. "Canvas Mermaid generated assets" cannot run on this machine: it requires Python 3.12.11 exactly, and the machine has 3.12.10. This change touches nothing under `tldw_chatbook/Canvas`.
- `ruff check` and `ruff format --check` on the two Python files: clean.
- Simulated merges (`git merge-tree`) with the open branches that touch these files: #3345 clean, #3575 clean, #3771 one conflict, in `Packaging/speculative_voice_history_gate.py`.
- Not run locally: Linux. The pull request's own Perf Guard run is the Linux check.

Files: `Packaging/speculative_voice_history_gate.py`, `Tests/Performance/test_speculative_voice_latency.py`, `backlog/docs/lessons-testing-evidence.md` (two entries).
<!-- SECTION:NOTES:END -->
