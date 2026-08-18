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
a wrong reference is worse than a red check. Measured on dev at the time of
filing: **79 references** to the two ids across the repo, of which roughly 25
mean the later-filed (workspace/skills) pair. They span:

- a **merged ADR** — `backlog/decisions/069-project-skills-folder-convention.md`
  (two links, one URL-encoded to the filename)
- **three User Guide pages** — `settings.md`, `library.md`, `library/skills.md`
- **two lessons entries** in `backlog/docs/lessons-testing-evidence.md`
- **four sibling task files** — 17961, 17962, 17963, 17964 (frontmatter
  `dependencies:` **and** prose)
- **production source comments** and `DESIGN.md` — these mean the *Console*
  pair, so they must NOT be touched
- **two test files** whose intent is genuinely ambiguous from the comment text
  alone (`test_non_obscuring_focus_contract.py:895`,
  `test_workbench_visual_snapshots.py:164`)

That last category is the reason this needs a human decision: the two ids
cannot be separated by pattern matching, only by reading each site for which
programme it means.

## Acceptance Criteria (the what)

- [ ] The later-filed pair (workspace modal, project skills) is renumbered to
      fresh ids taken by **leapfrog** from the max across ALL remotes and
      worktrees, not from the local max — the practice that caused this
- [ ] Every reference is re-pointed **by meaning, not by pattern**: each of
      the ~25 workspace/skills sites is read and updated; the Console sites
      (production comments, `DESIGN.md`, tasks 17652/17654/17655/17657/17662)
      are left untouched
- [ ] The two ambiguous test-file comments are resolved by reading the tests,
      and which programme each meant is recorded
- [ ] ADR-069's links resolve, including the URL-encoded filename form
- [ ] `No duplicate backlog task IDs` passes on a PR into `dev`
- [ ] The frontmatter `id:` field and the filename agree for every renamed
      file (the guard checks both independently)
