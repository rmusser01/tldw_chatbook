---
id: TASK-17381
title: Duplicate task-16812 fails the backlog ID guard on dev
status: Done
assignee: []
created_date: '2026-08-17 08:05'
updated_date: '2026-08-17'
labels:
  - backlog-hygiene
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two different task files claim TASK-16812 on dev — the Console local-provider thinking controls task and the research category-lane baselines task — so the duplicate-ID CI guard fails on dev itself and every pull request against dev inherits that red. This is the seventh ID collision in this repo, and it is the second one created by a renumber landing on an ID that was already taken: the Console file was added by a commit titled "chore(backlog): resolve duplicate task IDs".

Resolving it needs a decision rather than a mechanical rename, because the usual rule (never move a Done task) cannot break the tie: both tasks are Done, and both IDs are referenced from live artifacts. The Console ID is linked by filename from ADR-066 plus a plan and a QA script; the research ID is cited from source comments, a test section header, the eval baseline doc, and another task's justification. Whichever file moves, the references that point at it need to move with it, and the historical plan/QA artifacts that record the work under its old number need a decision about whether they are rewritten or left as history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The duplicate-ID guard passes on dev
- [x] #2 Exactly one task file claims each of the two affected IDs, in both the filename and the frontmatter
- [x] #3 Every live reference to the renumbered task resolves to it, including the ADR link that names the file
- [x] #4 The chosen treatment of historical plan and QA artifacts naming the old ID is recorded with its reason
- [x] #5 The lessons entry on ID collisions records this incident, since a renumber caused it
<!-- AC:END -->

## Implementation Plan

1. Decide which file moves: the Console task (File A) is linked by FILENAME from ADR-066 (and a spec/plan/QA script); the baselines task (File B) is cited only by prose `task-16812` mentions. Move File B — updating prose references is safe, whereas moving File A would break the ADR's filename-encoded link.
2. Pick a collision-free id above the true all-branch max (17384) → `17385`.
3. `git mv` File B and set its frontmatter `id:`; leave File A on 16812.
4. Grep the ENTIRE repo (not just `backlog/`+`Docs/`) for every `16812` reference and repoint the baselines ones to 17385.
5. Add the lessons entry; record the historical-artifact decision here.
6. Verify both backlog-guard checks (filename + frontmatter) pass.

## Implementation Notes

Resolved by renumbering the **later-created** baselines task (File B, `Record repository, research-graph, and biomedical-stress live baselines`, created 2026-08-16) from `TASK-16812` to **`TASK-17385`**. The Console thinking-controls task (File A, created 2026-08-15) keeps `TASK-16812`.

**Why File B moved.** Both tasks are Done, so "never move a Done task" cannot break the tie. File A's id is embedded in an ADR-066 *filename* link (`task-16812%20-%20Console-thinking...`), a spec, a plan, and a QA script; File B's id appears only as prose `task-16812` mentions. Moving the prose-referenced task avoids rewriting a filename-encoded ADR link, so File A — and ADR-066's link to it — is untouched (satisfies #3 without disturbing the ADR).

**References repointed to 17385** (found only by a whole-repo grep — the trap that made this "a decision, not a mechanical rename"):
- `tldw_chatbook/Research_Interop/academic_providers.py` (source comment)
- `Tests/Research/test_academic_providers.py` (test section header)
- `Helper_Scripts/Benchmarks/record_research_baseline.py` (benchmark comment)
- `Docs/Development/research-report-eval-baseline.md` (×2)
- `backlog/tasks/task-17066 - Source-type-aware-relevance-gate-for-non-paper-evidence.md` (justification)

**#4 — treatment of historical artifacts.** All of the renumbered task's references are *live* (active source/test/benchmark code and a living baseline doc), so they were **rewritten** to 17385 rather than left as history — a live reference must resolve to the actual task. There are **no frozen plan/QA/spec artifacts under `Docs/superpowers/` naming 16812 for the baselines task** (a whole-repo grep confirms every `superpowers/` mention of 16812 belongs to the Console task). File A's own historical plan/QA/spec/ADR artifacts keep 16812 unchanged, because File A did not move.

Both backlog-guard checks (filename prefix + frontmatter `id:`) pass; 0 duplicates across the task set. Lessons entry added to `backlog/docs/lessons-backlog-hygiene.md` recording that references hid in source/test/benchmark comments a `backlog/`+`Docs/` sweep missed.
