---
id: TASK-18413
title: Duplicate backlog task IDs 17650 and 17651 red the CI guard on dev
status: To Do
assignee: []
created_date: '2026-08-18'
labels: [backlog-hygiene, ci]
dependencies: []
priority: medium
---

## Description (the why)

**`No duplicate backlog task IDs` is RED on `origin/dev` and fails on every
PR opened against it**, including PRs that touch no task file at all (found
on PR #1812, which only modified an existing task).

Two programmes minted the same two ids **fourteen minutes apart** on
2026-08-17, and both merged:

| id | filed 18:00:58 (`68b6e3a87`) | filed 18:14:57 (`979a6914a`) |
|---|---|---|
| `task-17650` | Console — delete the zero-information rows in the bottom stack | Shared workspace creation modal |
| `task-17651` | Console — flatten the composer to a one-row dense form field | Project skills `.SKILLS/` discovery and import |

All four are `Done`. The guard caught it only after both had landed, which is
the known failure mode: the CLI assigns from the **local** max, so two
worktrees that never see each other mint the same number.

The harm is not only the red check. **A reference to "TASK-17651" is now
ambiguous** — and both meanings are actively cited in the tree, including in
production source comments.

## Why this was filed rather than fixed

Renumbering is the obvious repair, but it is **not** a mechanical rename, and
a wrong reference is worse than a red check.

**The inventory below is GENERATED, not curated — because my first,
hand-filtered one was wrong.** I separated the two meanings with a keyword
grep (excluding anything matching `console`), and that filter silently hid
`Docs/User_Guide/console/sessions-tabs-workspaces.md` — a *workspaces* page
whose path merely contains the word — plus `task-18310`. A reviewer caught
it. **Any repair must start from a generated file list**, because the very
heuristic that looks like it separates the two meanings is the thing that
drops sites.

**75 files reference the two ids:**

```
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
17650
17651
CI
Centralized-workspace-activation-seam-for-cross-screen-Console-runtime-sync.md
Console-delete-the-zero-information-rows-in-the-bottom-stack.md
Console-design-pass-inline-REPL-prompt-north-star.md
Console-flatten-the-composer-to-a-one-row-dense-form-field.md
Console-message-action-focus-walk-red-on-dev-after-selection-feedback.md
Console-one-row-breathing-room-below-the-composer.md
Console-raise-the-composer-draft-cap-from-4-to-8-rows.md
Console-status-chips-placement-setting-and-persistent-collapse.md
DESIGN.md
Docs/User_Guide/console/sessions-tabs-workspaces.md
Docs/User_Guide/library.md
Docs/User_Guide/library/skills.md
Docs/User_Guide/settings.md
Duplicate
Focused-compact-Input-Checkbox-content-invisible-in-WorkspaceCreateModal.md
IDs
Project-skills-SKILLS-folder-discovery-and-prompt-driven-import.md
Project-skills-follow-ups-offer-modal-footer-tests-checkbox-escape-assertion-off-thread-discovery.md
Shared-fixed-name-temp-files-race-across-app-instances-in-skills-store-and-trust-store.md
Shared-workspace-creation-modal-across-Console-Settings-and-Library.md
Tests/UI/test_console_composer_collapse.py
Tests/UI/test_console_internals_decomposition.py
Tests/UI/test_console_status_row_collapse.py
Tests/UI/test_non_obscuring_focus_contract.py
Tests/UI/test_workbench_visual_snapshots.py
Workspace-create-modal-follow-up-tests-and-typed-seams.md
and
backlog
backlog/decisions/069-project-skills-folder-convention.md
backlog/docs/lessons-testing-evidence.md
backlog/tasks/task-17650
backlog/tasks/task-17650
backlog/tasks/task-17651
backlog/tasks/task-17651
backlog/tasks/task-17652
backlog/tasks/task-17654
backlog/tasks/task-17655
backlog/tasks/task-17656
backlog/tasks/task-17657
backlog/tasks/task-17961
backlog/tasks/task-17962
backlog/tasks/task-17963
backlog/tasks/task-17964
backlog/tasks/task-18310
dev.md
guard
on
red
task
the
tldw_chatbook/UI/Console_Modules/frame.py
tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py
tldw_chatbook/UI/Console_Modules/transcript.py
tldw_chatbook/UI/Screens/chat_screen.py
tldw_chatbook/Widgets/Console/console_composer_bar.py
tldw_chatbook/css/components/_agentic_terminal.tcss
tldw_chatbook/css/tldw_cli_modular.tcss
```

Reading them by meaning:

- **Must NOT be touched** (they mean the *Console* pair): the production
  source comments (`chat_screen.py`, `Console_Modules/frame.py`,
  `transcript.py`, `provider_continuation_recovery.py`,
  `console_composer_bar.py`), both `.tcss` files, `DESIGN.md`, and tasks
  17652 / 17654 / 17655 / 17656 / 17657.
- **Must be re-pointed** (they mean the later-filed *workspace / project
  skills* pair): the merged **ADR-069** (two links, one URL-encoded to the
  filename), **four** User Guide pages (`settings.md`, `library.md`,
  `library/skills.md`, **`console/sessions-tabs-workspaces.md`**), two
  entries in `lessons-testing-evidence.md`, and tasks 17961, 17962, 17963,
  17964, **18310** — in frontmatter `dependencies:` **and** prose.
- **Ambiguous from the comment text alone**, needs the test read:
  `test_non_obscuring_focus_contract.py`, `test_workbench_visual_snapshots.py`,
  `test_console_composer_collapse.py`, `test_console_internals_decomposition.py`,
  `test_console_status_row_collapse.py`.

That last category is why this needs a human: the two ids cannot be
separated by pattern matching, only by reading each site for which programme
it means.

## Acceptance Criteria (the what)

- [ ] The later-filed pair (workspace modal, project skills) is renumbered to
      fresh ids taken by **leapfrog** from the max across ALL remotes and
      worktrees, not from the local max — the practice that caused this
- [ ] The repair starts from a **generated** file list (the grep in this
      task), never a hand-curated one — a keyword filter already dropped two
      sites when this task was first written
- [ ] Every reference is re-pointed **by meaning, not by pattern**: each
      workspace/skills site is read and updated; the Console sites
      (production comments, both `.tcss` files, `DESIGN.md`, tasks
      17652/17654/17655/17656/17657) are left untouched
- [ ] The five ambiguous test files are resolved by reading the tests, and
      which programme each meant is recorded
- [ ] ADR-069's links resolve, including the URL-encoded filename form
- [ ] `No duplicate backlog task IDs` passes on a PR into `dev`
- [ ] The frontmatter `id:` field and the filename agree for every renamed
      file (the guard checks both independently)
