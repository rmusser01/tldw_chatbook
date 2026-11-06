---
id: TASK-38974
title: "Console Library controls docs contract is red on the media-and-conversations page, and no dev lane runs Tests/Docs"
status: To Do
created_date: 2026-11-06 01:21
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
- [ ] #1 The two media-and-conversations nodes pass, either by restoring the per-conversation Library control contract on the page at the shipped wording or by re-pinning the test at that wording (not by loosening it); the notes say which, per string
- [ ] #2 The local-link check skips non-file URL schemes (mailto:, media:, and any other the docs use), and still fails on a broken relative link, shown with a temporary broken link
- [ ] #3 Tests/Docs/test_console_library_controls_docs.py runs on a PR lane, with its measured time recorded (0.5 s for the file on dev), so a docs edit that breaks the contract fails its own PR
<!-- AC:END -->
