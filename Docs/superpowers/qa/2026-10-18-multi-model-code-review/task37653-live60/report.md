# TASK-37653 live qualification: verifier completed (2026-11-02)

**Result: passed on the second approved run.** On `949ddbc452` (branch
`claude/burn-35582-stream` on `dev` `88af9f9379`), the live run did all of the following:

- sealed both reviewer originals;
- presented both reviewer originals before the fresh verifier started;
- sealed the verifier's report;
- produced a unified report that expands each of the three original findings exactly once.

The owner approved each run separately: 60 requests / 3 children for each, one invocation each, no retries, and no CI gate. The two runs together spent **90 generation requests and 5 product children**.

## Run 1: verifier blocked by the harness (`attempt-_2f9i5ac`)

- **Spend:** 40 generation requests, all HTTP 200: Qwen (local) 27, `gpt-6-luna` 6, `deepseek-flash` 7, Haiku 0. There were also 6 metadata GETs and 2 children.
- **Reviewers:**
  - R1 sealed with no findings and refuted the seeded false string-input claim.
  - R2 sealed with 4 findings.
- **Result:** `live_report_incomplete_V1`. The automatic follow-up turn tried to spawn the verifier twice, and both attempts returned `original_reports_not_presented`. The local primary then returned two empty responses and the run stopped as `stuck`.
- **Cause:** a gap in the harness, not a product defect.
  - The runner built its own `ConsoleChatController` and set `controller.app = app`.
  - That was the attach step before PR #3286 moved review presentation into the UI wiring.
  - Since then, the Console screen attaches through `wiring.attach_review_presenter(controller, app)`, which installs the original-report presenter. The runner never called it, so nothing presented the originals.

## Run 2: passed (`attempt-tnqqou64`)

The only change was the one-line harness fix: the runner now attaches through `attach_review_presenter`.

- **Spend:** 50 generation requests, all HTTP 200: Qwen 25, `gpt-6-luna` 6, `deepseek-flash` 8, `claude-haiku-5-5` 11. Plus 3 children. Every run ended `done`.
- **Reviewers:**
  - R1 (`gpt-6-luna`) sealed with no findings. It refuted the seeded false claim but missed the seeded defect, in both runs.
  - R2 (`deepseek-flash`) sealed three findings. Both originals were presented.
- **Verifier and unified report:** V1 (`claude-haiku-5-5`) sealed one disposition per finding, and the unified report expanded each finding exactly once.

| Original | Finding | Verifier |
|---|---|---|
| R2-F1 (high) | `average([])` now raises `ZeroDivisionError`: the seeded removal of the empty-input guard | confirmed |
| R2-F2 (low) | The QA claim that `average` accepts strings | refuted (the seeded false positive) |
| R2-F3 (low) | `isinstance(value, int)` admits `bool` | confirmed |

**The fixes from AC#1, seen live:**
- A refused report read now shows its code, for example `invalid_report_reference`, instead of a generic `review_refused`.
- No continuation-persistence error occurred.
- No refused path argument deferred a whole batch.

**What this run did not cover live:** R1 returned no findings, so duplicate grouping was not exercised. The failed-reviewer, truncation, source-drift, small-context and storage-overflow controls are the harness's scripted offline controls. They passed at this head (`offline_partial_controls_passed`, 0 network) and are kept separate from the live evidence above.

## Changes to the harness for these runs

The three harness changes below were reviewed against the consumed `macos-final60` preparation. Patches are in the archives.

- **History loss.** macOS's periodic `/tmp` cleanup deleted the raw history packets the runner re-hashes (23 of 24 history files and 194 of 278 spend-history files).
  - With the owner's approval, the runner now records them as lost instead of refusing.
  - The spend baseline constants, the physical 60-request cap, the 3-child guard, the single exclusive claim, and the receipt and head pins are unchanged.
- **Dev drift.** The `source_case` test fixture gained a `request` parameter on dev (`8ff29d408a`). The harness now passes a stand-in only when the fixture accepts it.
- **Run 2 only.**
  - The runner attaches the presenter.
  - The spend baseline moved to 524 requests / 35 children, so the aggregate ceiling is 584 / 38.
  - Run 1's 20 unused requests and 1 unused child slot retire with it.

## Evidence

- [Spending](spending.json)
- [Retention](retention.json): archives under `~/.codex/artifacts/task-37653/`, kept outside `/tmp` on purpose. The scratch home is excluded, and the credential scan found no key bytes.
- Raw packets: `/private/tmp/task37653-live60/attempt-_2f9i5ac` and `/private/tmp/task37653-live60b/attempt-tnqqou64`. These are subject to macOS `/tmp` cleanup; the archives are the durable copy.
