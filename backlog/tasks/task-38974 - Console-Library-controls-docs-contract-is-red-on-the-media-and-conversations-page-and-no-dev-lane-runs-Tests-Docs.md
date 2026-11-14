---
id: TASK-38974
title: "Console Library controls docs contract is red on the media-and-conversations page, and no dev lane runs Tests/Docs"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-10 18:28
labels:
- follow-up
- docs
- tests
- ci
priority: low
references:
- Tests/Docs/test_console_library_controls_docs.py
- Docs/User_Guide/library/media-and-conversations.md
- README.md
- .github/workflows/test.yml
- .github/workflows/nightly-deep.yml
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. In `Tests/Docs/test_console_library_controls_docs.py`, 4 of 21 nodes fail. The same 4 fail in a separate base checkout, and the whole file takes 0.5 s.

**The four reds.**

On `Docs/User_Guide/library/media-and-conversations.md`, not owned by any task:
- `test_console_library_control_docs_state_the_contract[...media-and-conversations.md-required_text6]` reports the page is missing `"does not change the conversation's Auto or Assistant setting"` and `'staged context'`. f09a7786d4 (2026-09-16, "consume live-work handoffs on warm returns") removed both phrases from the page.
- `test_console_library_control_docs_have_valid_local_links[...media-and-conversations.md]` reports `['media://…']` as a missing local link. The page has carried that example link text since 47f5bbea32 (2026-10-09, TASK-34000.25). The link check (`MARKDOWN_LINK_RE`, `:73`; the scheme skip at `:109`) skips only `http://` and `https://` targets.

On `README.md`, already listed in TASK-32583 (To Do), which was filed before the media-page pair broke:
- `[README.md-required_text0]`: missing 'Manual Search Library', 'Auto: Never / Automatic', 'Assistant: Blocked / Allowed', 'Direct / RAG'. b9fe7ee28a ("docs: finalize README rebase", 2026-08-30) changed the README's wording.
- `test_console_library_control_docs_have_valid_local_links[README.md]`: `['mailto:contact@rmusser.net']`, a link the README has had since 878ed258fb.

One PR can fix all four, and the README pair can then be ticked off in TASK-32583.

**Why dev looks green.** No pull-request lane runs `Tests/Docs`:
- `derived-artifacts.yml` has no target under it.
- `test.yml` runs `pytest Tests --ignore=Tests/UI` only on pushes to `main` and on manual dispatch (`test.yml:3-6`, `:133`).
- `nightly-deep.yml` runs `pytest ./Tests/` on a schedule (`:7-8`, `:133`), which runs from the default branch, `main`.

So dev's PR checks never collect these nodes. The drift shows up only when `main` is updated or when someone runs the file locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two media-and-conversations nodes pass, either by restoring the per-conversation Library control contract on the page at the shipped wording or by re-pinning the test at that wording (not by loosening it); the notes say which, per string
- [x] #2 The local-link check skips non-file URL schemes (mailto:, media:, and any other the docs use), and still fails on a broken relative link, shown with a temporary broken link
- [x] #3 Tests/Docs/test_console_library_controls_docs.py runs on a PR lane, with its measured time recorded (0.5 s for the file on dev), so a docs edit that breaks the contract fails its own PR
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-pin the media-and-conversations contract at the page's shipped wording (both strings), and the README entry at README's shipped wording, so the file is green.
2. Make the local-link check skip any URL scheme (mailto:, media:, http(s):) and show a temporary broken relative link still fails.
3. Add Tests/Docs to a PR lane (dreams-boundary sandboxed step, which has headroom; pr-fast-lane is at its cap) in derived-artifacts.yml and its Tests/CI pin, and record the file's measured time.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-pinned the stale contract strings at the shipped wording, taught the link check to skip URL schemes, and put the four Tests/Docs files on a PR lane.

- **AC#1** Both media-and-conversations strings are re-pinned at the page's wording, which f09a7786d4 rewrote on purpose; the page is unchanged:
  - "does not change the conversation's Auto or Assistant setting" → "does not send automatically or change the conversation's Auto or Assistant setting" (the page's sentence; the same no-change claim, plus no auto-send);
  - "staged context" → "Use as source stages a reference to the selected conversation in the current Console session" (what the handoff stages, and where).
  - The README pair (TASK-32583's two nodes) had to pass for the file to be laned. The README entry is re-pinned at README's own wording, which d01daac8b4 ("docs: restore layered project README") introduced: "manual Library search, automatic retrieval policy, assistant tool permission, and direct/RAG retrieval mode" and "New conversations do not silently grant every tool or source". The `mailto:` node passes through AC#2.
- **AC#2** `missing_local_links` skips any target that starts with a URL scheme (`URL_SCHEME_RE`: http, https, mailto, media, …). Negative control: a temporary `[the gone page](no-such-page.md)` and a `mailto:` link appended to the media page failed with `['no-such-page.md']` alone; reverted. Baseline red on dev: the 4 named nodes failed, 21 passed (`Tests/Docs`).
- **AC#3** `Tests/Docs/{test_console_library_controls_docs,test_user_guide_contract,test_user_guide_conflict_markers,test_personas_qa_capture_script}.py` run in `recovered-suites`' "Run the recovered suites" step (the owner's 2026-11-06 job for non-UI suites red in no PR lane). They sit in its sandboxed block, after `Tests/Prompts_DB/test_prompts_db_batch_mutations.py` and before the bootstrap_profile suites, and are pinned in `RECOVERED_SUITES_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`). Measured serially with that step's flags (`--storage-lease-guard --timeout=180`) at load 42-52 on 18 CPUs: the four files were 30 passed in 5.5 s and 4.8 s; the controls file alone was 21 passed in 1.7 s. `Tests/CI`: 1240 passed, 1 skipped.
- Files: `Tests/Docs/test_console_library_controls_docs.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
- **Review round 1 (r13h-rev1).** I-1: the first version put the files in `dreams-boundary`, against the owner's decision that keeps that job's margin. They moved to `recovered-suites`; the dreams-boundary step, its comment, `DREAMS_BOUNDARY_TARGETS` and `test_response_rules_critical_gate.py` are back as on dev. M-5: the README prose came from d01daac8b4, not b9fe7ee28a (which removed the old labels); the test comment and these notes are corrected.
- **Review round 2 (r13h-rev2).** The Files line no longer lists `Tests/CI/test_response_rules_critical_gate.py`, which round 1 restored to dev.
<!-- SECTION:NOTES:END -->
