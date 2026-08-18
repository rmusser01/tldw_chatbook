---
id: TASK-18413
title: Duplicate backlog task IDs 17650 and 17651 red the CI guard on dev
status: Done
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

**The inventory below is GENERATED, and it took two tries — both failures are
the trap the repair will face.** First I separated the two meanings with a
keyword grep excluding anything matching `console`, which silently dropped
`Docs/User_Guide/console/sessions-tabs-workspaces.md` — a *workspaces* page
whose path merely contains that word — and `task-18310`. A reviewer caught
it. Then, regenerating, I split grep's output on whitespace; **these task
filenames contain spaces** (`task-17650 - Console-….md`), so 33 paths
shattered into 75 fragments. Both times the instrument, not the repo, was
wrong.

So: **generate the list, split on newlines, and assert every entry ends in a
real extension** before trusting a count.

**33 files reference the two ids:**

```
DESIGN.md
Docs/User_Guide/console/sessions-tabs-workspaces.md
Docs/User_Guide/library.md
Docs/User_Guide/library/skills.md
Docs/User_Guide/settings.md
Tests/UI/test_console_composer_collapse.py
Tests/UI/test_console_internals_decomposition.py
Tests/UI/test_console_status_row_collapse.py
Tests/UI/test_non_obscuring_focus_contract.py
Tests/UI/test_workbench_visual_snapshots.py
backlog/decisions/069-project-skills-folder-convention.md
backlog/docs/lessons-testing-evidence.md
backlog/tasks/task-17650 - Console-delete-the-zero-information-rows-in-the-bottom-stack.md
backlog/tasks/task-17650 - Shared-workspace-creation-modal-across-Console-Settings-and-Library.md
backlog/tasks/task-17651 - Console-flatten-the-composer-to-a-one-row-dense-form-field.md
backlog/tasks/task-17651 - Project-skills-SKILLS-folder-discovery-and-prompt-driven-import.md
backlog/tasks/task-17652 - Console-status-chips-placement-setting-and-persistent-collapse.md
backlog/tasks/task-17654 - Console-raise-the-composer-draft-cap-from-4-to-8-rows.md
backlog/tasks/task-17655 - Console-design-pass-inline-REPL-prompt-north-star.md
backlog/tasks/task-17656 - Console-message-action-focus-walk-red-on-dev-after-selection-feedback.md
backlog/tasks/task-17657 - Console-one-row-breathing-room-below-the-composer.md
backlog/tasks/task-17961 - Focused-compact-Input-Checkbox-content-invisible-in-WorkspaceCreateModal.md
backlog/tasks/task-17962 - Workspace-create-modal-follow-up-tests-and-typed-seams.md
backlog/tasks/task-17963 - Shared-fixed-name-temp-files-race-across-app-instances-in-skills-store-and-trust-store.md
backlog/tasks/task-17964 - Project-skills-follow-ups-offer-modal-footer-tests-checkbox-escape-assertion-off-thread-discovery.md
backlog/tasks/task-18310 - Centralized-workspace-activation-seam-for-cross-screen-Console-runtime-sync.md
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

- [x] Renumbered to **18704** (workspace modal) and **18705** (project
      skills), leapfrogged from 18604 — the max across origin/dev, every
      local branch and every worktree, not the local max.
- [x] Started from the generated list in this task, then every site was
      READ before editing. No pattern-wide substitution was used anywhere.
- [x] **20 references re-pointed, 59 Console references left untouched**, each
      classified by reading it. The decisive case: `lessons-testing-evidence.md`
      holds TWO `TASK-17651` entries with OPPOSITE meanings — "Removing a
      widget's border box…" is the Console composer (untouched) and "A gate
      built in halves…" is the `.SKILLS/` import (re-pointed). A `sed` would
      have corrupted one of them.
- [x] All five read and resolved: **all mean the Console pair** —
      `test_console_composer_collapse` (zero chrome rows),
      `test_console_internals_decomposition` (dense-form composer),
      `test_console_status_row_collapse` (phantom footer row),
      `test_non_obscuring_focus_contract` (composer focus edge),
      `test_workbench_visual_snapshots` (grid bottom edge). None touched.
      `task-17656` was ambiguous from its text too and is likewise Console.
- [x] Both ADR-069 links updated and verified to resolve on disk after
      percent-decoding (the `task-18705%20-%20…` form included).
- [x] Both halves of the guard pass locally across **2,175** task files:
      zero duplicate filename ids, zero duplicate frontmatter ids. The check
      itself was verified to have teeth (an injected duplicate is detected) —
      `uniq -d` exits 0 either way, so a naive `&&` test always reports clean.
- [x] `task-18704` → `id: TASK-18704`; `task-18705` → `id: TASK-18705`.


## Implementation Notes

**The guard is green: 2,175 task files, zero duplicate ids on either half.**

**Which pair moved, and why.** The Console programme filed 17650-17655 at
18:00:58 on 2026-08-17 (`68b6e3a87`); the workspace-modal/project-skills pair
filed the same two ids **fourteen minutes later** at 18:14:57 (`979a6914a`).
The later arrival was renumbered, which also happened to be the side with
**no production-source references** — all 18 source/CSS comments mean the
Console pair.

**20 references re-pointed, 59 left untouched**, every one classified by
reading it rather than by pattern. The case that justifies the whole approach:
`backlog/docs/lessons-testing-evidence.md` contains **two `TASK-17651` entries
with opposite meanings** — "Removing a widget's border box activates the
global focus outline" is the *Console composer*, and "A gate built in halves
is no gate" is the *`.SKILLS/` import*. One was re-pointed, one was not. Any
file-wide substitution corrupts one of them.

All five test files flagged as ambiguous in the filing turned out to mean the
Console pair once read (composer chrome rows, dense-form geometry, phantom
footer row, focus edge, workbench grid), as did `task-17656`.

**Verification:** both guard halves re-implemented so they can actually fail
(`uniq -d` exits 0 regardless, so the original `&&` form reported "clean"
unconditionally — checked with an injected duplicate); every re-pointed
reference resolves; ADR-069's percent-encoded link resolves on disk; no
dependency dangles as a result of this change. One pre-existing dangling dep
(`TASK-2800` in `task-2818`) is present on dev and untouched.

**Files:** 2 renamed (`git mv`, history preserved), 5 sibling tasks, ADR-069,
4 User Guide pages, 1 lessons entry, this task. No production source changed.
