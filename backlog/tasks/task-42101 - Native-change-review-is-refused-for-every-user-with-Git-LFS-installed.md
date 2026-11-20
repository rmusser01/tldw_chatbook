---
id: TASK-42101
title: "Native change review is refused for every user with Git LFS installed"
status: In Progress
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
- [x] #1 The owner's choice is recorded in ADR-084 and ADR-285. Chosen (2026-11-19): refuse a configured filter only when a path the invocation can reach selects it through Git attributes. The other options were to keep refusing any configured filter, or to treat Git LFS specially.
- [x] #2 With Git LFS-style filters configured and a workspace that tracks no LFS path, native change review publishes and `@diff` reads, and the driver never runs. A path that selects the driver still refuses, including an ignored path force-staged from stdin and a top-level path while the app runs from a subdirectory. No configured driver runs even when detection misses one, and none runs inside a submodule (tests).
- [ ] #3 When capture is refused because of configured filters, the user can see that cause and what would change it. Verify this in the running app.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (ADR-084 and ADR-285 amended in place).

1. Pin the new rule with failing tests: a filter nothing selects (capture and `@staged`), and an ignored force-staged path that selects one.
2. In `qualified_git_command`, keep the partial-clone refusal unconditional. For filter-capable invocations, ask Git which paths select a configured driver, and refuse only then.
3. Pass `update-index --stdin` paths into the check, and give the references runner stdin.
4. Make the refusal name its remedy, then update the ADRs and the architecture doc.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Rule (`Workspaces/native_git.py`).** A configured clean/process/smudge driver refuses only when a reachable path selects it.
  - **When the check runs:** only when the config query finds filter names, and only for invocations that can hand work-tree content to a filter: add, update-index, rm, commit, diff, `ls-files -m`, and `cat-file --filters`.
  - **How it decides:** `_selects_configured_filter` runs `ls-files -z --cached --others --exclude-standard`, adds the exact paths an `update-index -z --stdin` stages, and asks `check-attr -z --stdin filter`. Git resolves the attributes, macros included.
  - **Inside the custody job:** the check runs in the same opaque job, again before every such invocation.
  - **Unchanged:** partial-clone remotes still always refuse.
- **Guarantee (from the security review).** Detection decides the refusal; it does not guarantee that no filter runs. The review reproduced three escapes on the first version (80f7b33ad6):
  - `@diff`'s dirty-submodule check ran `git status` inside a submodule, and that ran its filter.
  - With the app's cwd in a subdirectory of the root, Git scoped the check to that subdirectory, while `commit` refreshed the whole index.
  - A `.gitattributes` write racing the check got the filter run in 4 of 25 captures.

  Now:
  - **Neutralization:** `_neutralize_filters` empties every configured driver's commands for the command itself, through GIT_CONFIG_COUNT. That keeps driver names out of argv, and submodule children inherit it.
  - **Submodules:** `diff` gets `--ignore-submodules=dirty`, which also covers a driver configured only in a submodule's own config.
  - **Working directory:** ShadowRepo runs Git at its root, for both the native and legacy runners. This also fixes `add -- .` and stdin paths resolving against the app's cwd.
  - **Refused outright:** `cat-file --textconv` (a textconv driver is not a filter and is not qualified) and non-`-z` `update-index --stdin`.
  - **Checked empirically:** in a scratch HOME, the env pins kept a sentinel filter from running in the superproject and in a submodule child. A selected `required` driver fails the command instead.
- **Message.** "configured filters selected by Git attributes (remove the filter= attribute or its filter config)". The Console shows it as "⚠ change tracking failed for <root>: <error>".
- **Plumbing.**
  - `ShadowRepo._native_command` passes stdin.
  - The references runner accepts `input_data`.
  - Each references invocation qualifies a copy of the environment.
- **Tests.**
  - Red first, against the old any-filter rule:
    - no selecting path (capture publishes);
    - `@diff` with a sentinel driver nothing selects;
    - an ignored force-staged path that selects one (refused; without the stdin paths the helper really ran).
  - Red against 80f7b33ad6, all three:
    - detection forced to miss (the driver never runs);
    - a filter configured only inside a submodule (`@diff` never enters it);
    - the app's cwd in a subdirectory (refused, never run).
  - The reference refusal cases now write a selecting `.gitattributes`, and include a dotted, mixed-case driver name.
- **Cost.** Two extra Git calls per filter-capable invocation, only when a filter is configured. That took about 0.16 s over this repo's ~42k paths, measured by the reviewer.
- **Docs.** ADR-084, ADR-285 and `Docs/Architecture/console.md` are amended.
- **Open: AC #3.** Its "verify in the running app" is not done. The persisted text the row shows is pinned by a test, but the live TUI was not driven, so the AC stays open for the owner.
<!-- SECTION:NOTES:END -->
