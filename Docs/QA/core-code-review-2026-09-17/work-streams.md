# Work streams — core-runtime code review 2026-09-17

97 backlog tasks were created from this review's 408 findings: **12 work-stream parents** (`TASK-32800`–`TASK-32811`) and **85 child tasks** (dotted subtask ids under them). Every child names the findings it closes, with the file and line each was verified at.

Branch: `docs/core-review-tasks-2026-09-18`, created off `origin/dev` `e89f28d751` in a dedicated worktree so the main checkout's in-progress work was not touched.

## Suggested order

The streams are independent enough to run in parallel, but this order front-loads what a user can hit today and puts the mechanical work last.

| # | Stream | Parent | Tasks | High | What it is |
|---|---|---|---:|---:|---|
| 1 | `review-crash` | TASK-32800 | 5 | 4 | The three app-killing crashes, plus the guard that would have caught all of them |
| 2 | `review-data` | TASK-32801 | 5 | 1 | The Media data loss and the transaction contract behind it |
| 3 | `review-markup` | TASK-32802 | 5 | 2 | User text deleted or mangled by the wrong escaper, ~40 sites |
| 4 | `review-security` | TASK-32806 | 8 | 4 | Credentials, the permission-store reset, the unbounded calculator, process groups |
| 5 | `review-ux` | TASK-32811 | 8 | 1 | Features that silently do nothing or show the wrong thing |
| 6 | `review-wire` | TASK-32805 | 5 | 1 | Streaming leaks, dropped token usage, summarizers that always error |
| 7 | `review-time` | TASK-32803 | 5 | 1 | One timestamp helper, the three comparisons already wrong, and a format guard |
| 8 | `review-loop` | TASK-32804 | 12 | 2 | Synchronous work on the event loop; starts with the 11 ms config read behind most of it |
| 9 | `review-helpers` | TASK-32808 | 11 | 0 | Adopt the shared helpers, delete the dead ones |
| 10 | `review-dead` | TASK-32807 | 6 | 0 | Delete ~12,000 lines with no production importer |
| 11 | `review-size` | TASK-32809 | 3 | 0 | Re-pin the 18 red ratchet rows, add budgets for the unbudgeted god modules |
| 12 | `review-polish` | TASK-32810 | 12 | 0 | The 209 P3 findings, bundled per subsystem |

**16 tasks are marked high priority.** They are the four crashes, the data loss, the markup class, the credential and permission-store defects, the unbounded calculator, the process-group fix, the RAG cache, the streaming leak, the flashcard due-date defect, and the two worst event-loop costs.

## Dependencies worth knowing

- **Stream 8 (`review-loop`) starts with one task.** `TASK-32804.1` makes cached config reads cheap; most of the other eleven exist because a surface pays that 11 ms cost repeatedly. Doing it first may shrink or close several of them, so measure again after it lands.
- **Stream 7 (`review-time`) also starts with one task.** The shared helper and its guard come before the four adoption and fix tasks.
- **Stream 3 (`review-markup`) has one keystone.** `TASK-32802.1` swaps in the correct escaper everywhere; the other four are separate surfaces or the opposite defect (double-escaping into a markup-off surface).
- **Stream 1's guard task** (`TASK-32800.4`) is worth landing early even though the three crash fixes are independent of it — it is what stops the fourth one.
- **Two tasks need an ADR before implementation**: the credential-precedence ruling (`review-security`) and the scope-service base class (`review-helpers`). Both are recorded as decisions to make, not patches to apply, because the contradictory behaviour is pinned by test names on both sides.

## How each task traces back

Every child task carries:

- a description written as the *why*, quoting the measured consequence;
- a **Findings this closes** block listing each finding's severity, dimension, slice and the file and line it was verified at;
- acceptance criteria written as outcomes, not steps;
- a link to `report.md`, the per-slice evidence in `slices/`, and the reproductions in `phase4-verification.md`.

Findings that already had an owning task in the backlog were not re-filed; the review cites the existing task instead. Those are listed in `report.md` under "Already handled".

## Commands

```bash
backlog task list --parent 32800 --plain          # one stream's tasks, grouped by status
backlog task list --priority high --plain         # the 16 high-priority tasks
backlog task 32802.1 --plain                      # one task with its findings block
grep -rl 'core-review' backlog/tasks              # everything from this review
grep -rl 'review-crash' backlog/tasks             # by stream label
```

`backlog task list` has no label filter (`-s`, `-a`, `-m`, `-p`, `--priority`, `--sort` only), so filter by label with `grep` over the task files or by `--parent` with the stream's id. Verified against backlog.md 1.44.0.
