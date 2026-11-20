---
id: TASK-42101
title: "Native change review is refused for every user with Git LFS installed"
status: To Do
assignee: []
created_date: 2026-11-20 00:40
labels:
- decision
- workspaces
- git
priority: high
references:
- tldw_chatbook/Workspaces/native_git.py
- backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md
- backlog/decisions/285-native-phased-agent-workflows.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-42071 (2026-11-19, dev 6a5a332f95).

**The refusal.** `native_git.qualified_git_command` refuses every native Git command when any `filter.<name>.clean`, `smudge` or `process` is configured anywhere in Git's config (system, global or repository). The refusal reads "native change review unavailable: configured filters". ADR-084 and ADR-285 make this deliberate: an external filter is a configured program that Git would run on workspace files during capture.

**Who it hits.** Running `git lfs install` writes exactly those keys to the user's global `~/.gitconfig`. Two cases are known:
- The owner's machine has `filter.lfs.clean`, `smudge` and `process` in `~/.gitconfig`.
- GitHub's runners have them in the system config.

So for anyone who has ever installed Git LFS, native change review is refused in every workspace, including workspaces with no `.gitattributes` and no LFS-tracked file. That also refuses Console `@staged` and `@diff` references and native workflow capture, which go through the same qualification.

**Why it went unnoticed.** The test suite cannot see it: `Tests/conftest.py` isolates `HOME`, and TASK-42071 isolated the system config for the tests that ran into it.

**The decision.** Whether Git LFS users should lose the feature is a product decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's choice is recorded in ADR-084 and ADR-285. The known options are: keep refusing any configured filter; refuse only filters that a workspace path actually selects through its attributes; or treat Git LFS specially.
- [ ] #2 A test with Git LFS filters in the global config and a workspace that tracks no LFS path pins the chosen behaviour for native change review and for `@staged`/`@diff`.
- [ ] #3 When capture is refused because of configured filters, the user can see that cause and what would change it. Verify this in the running app.
<!-- AC:END -->
