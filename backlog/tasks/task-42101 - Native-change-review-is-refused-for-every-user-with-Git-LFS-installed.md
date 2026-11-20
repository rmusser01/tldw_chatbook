---
id: TASK-42101
title: "Native change review is refused for every user with Git LFS installed"
status: Done
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
- [x] #2 With Git LFS-style filters in the global config and a workspace that tracks no LFS path, native change review publishes and `@staged` reads, and the driver never runs. A path that selects the driver, including an ignored path force-staged from stdin, still refuses before Git can run it (tests).
- [x] #3 When capture is refused because a path selects a configured filter, the Console change-tracking row names that cause and the remedy (a test pins the persisted text the row shows).
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
  - **When the check runs:** only when the config query finds filter names, and only for invocations that can hand work-tree content to a filter: add, update-index, rm, commit, diff, and cat-file with --filters/--textconv.
  - **How it decides:** `_selects_configured_filter` runs `ls-files -z --cached --others --exclude-standard`, adds the exact paths an `update-index --stdin` stages (these may be ignored), and asks `check-attr -z --stdin filter`. Git applies its own attribute rules, macros included.
  - **Inside the custody job:** both commands use the invocation's prefix, pins and runner, so they stay in the same opaque job as the version and config discovery.
  - **Not cached:** the check runs again before every filter-capable invocation, as the config check already did.
  - **Unchanged:** partial-clone and promisor remotes still always refuse.
- **Message.** The refusal now reads "configured filters selected by Git attributes (remove the filter= attribute or its filter config)". That text is what the Console's "⚠ change tracking failed for <root>: <error>" row shows.
- **Plumbing.**
  - `ShadowRepo._native_command` passes the invocation's stdin.
  - The references `run_process` accepts `input_data`, for both its direct and custody paths.
- **Tests.**
  - New, each red first: a global LFS-style driver with no selecting path (capture publishes, the driver never runs); `@staged` with a repo-local driver nothing selects; and an ignored, force-staged path whose directory selects the driver. That last one is refused, and without the stdin paths the helper really ran (mutation check).
  - The reference refusal test now writes a selecting `.gitattributes`. The existing Workspaces refusal tests already did.
- **Cost.** The check adds two Git calls per filter-capable invocation, and only on machines with a filter configured. Before this change, those machines had no native capture at all.
- **Docs.** ADR-084 and ADR-285 are amended, and so is `Docs/Architecture/console.md`.
- **Not done:** I did not drive the live TUI. The row text is the existing marker's f-string over the persisted `tracking_error`, which the test pins.
<!-- SECTION:NOTES:END -->
