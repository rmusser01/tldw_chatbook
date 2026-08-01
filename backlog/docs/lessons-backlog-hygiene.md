# Lessons: backlog, git and tooling traps

Mechanical traps in the task board and the plumbing around it. Cheap to avoid once
known, expensive to rediscover. Every entry states the incident that produced it.

---

## Task IDs collide constantly — sweep every remote, not just dev

**What happened.** This has recurred **ten-plus times**. Most recently, in one session:

- IDs verified free against `origin/dev` and the local worktree still collided with
  `feat/watchlists-phase-a-data-foundation`. True max across all 101 remote refs was
  **699**, while dev's view said 686.
- The backlog CLI auto-assigned **703** — already taken **on dev itself**.
- An earlier ID collided with dev's own task of the same number, filed by another agent
  mid-session.

Checking `origin/dev` *feels* like diligence. It is not: parallel agents hold IDs on
unmerged branches.

**What to do.** Before filing, sweep **every remote ref** plus every worktree, and
re-check at merge time — dev moves under you. Never trust the CLI's auto-assignment.

```bash
git for-each-ref --format='%(refname)' refs/remotes/ | while read -r b; do
  git ls-tree -r -z --name-only "$b" backlog/ | tr '\0' '\n'
done | grep -oE 'task-[0-9]+' | cut -d- -f2 | sort -n -u | tail -5
```

Note `sort -n`, not `sort -u` alone. Lexicographic order puts `task-100` before
`task-99`, so a plain sort reports the wrong maximum. This is not theoretical: run
against this repo, the lexicographic version reports **99** as the highest task ID
while the numeric version reports **838** -- it would have collided on essentially
every filing. That mistake was in the first draft of this very file, which is a fair
illustration of why these entries carry their evidence.

---

## `--ac` does not split on commas — you get one run-on criterion

**What happened.** 2026-07-28, filing task-1261: four acceptance criteria were passed
as `--ac "first,second,third,fourth"`, exactly the shape CLAUDE.md documents
(`--ac "Must work,Must be tested"`). The CLI (v1.44.0) wrote **a single** criterion
whose text was the whole comma-joined string. Confirmed on a trivial control:
`--ac "alpha,beta,gamma"` produces `- [ ] #1 alpha,beta,gamma`, not three items.

A single run-on AC is not a cosmetic problem: it cannot be checked off
independently, so the Definition of Done ("all `- [ ]` changed to `- [x]`") becomes
all-or-nothing and the task stops describing what is actually left.

**What to do.** Pass `--ac` **once per criterion** if the flag repeats, or write the
`## Acceptance Criteria` block into the task file directly and verify with
`backlog task <id> --plain` before moving on. Whichever route, read the rendered AC
list back — the CLI accepted the comma form silently rather than erroring, so nothing
warns you.

---

## `git ls-tree` octal-escapes non-ASCII filenames

**What happened.** Several task titles contain an em-dash. `git ls-tree` emits those
paths as `"…\342\200\224…"`, so comparing its output to real filenames reported
**phantom collisions** for four unrelated tasks — twice in one session.

**What to do.** Use `git ls-tree -r -z --name-only` and split on NUL. That emits raw
UTF-8 paths.

---

## The backlog CLI collapses multiple `--ac` values into one criterion

**What happened.** Passing three acceptance criteria produced a single criterion with
the three joined by commas — on two separate tasks, unnoticed until later.

**What to do.** After filing, re-read the `AC:BEGIN` block. If it collapsed, rewrite the
block directly in the file. Also note the CLI strips some free-form sections, and
backticks in `--notes` are interpreted by the shell — a phrase in backticks vanished
from a task note this way.

**A related trap (TASK-595 Task 10, 2026-07-31).** `backlog task edit <id> --notes "..."`
does not append to an existing `## Implementation Notes` section — it REPLACES the
entire content between `<!-- SECTION:NOTES:BEGIN -->` and `<!-- SECTION:NOTES:END -->`.
A detailed Implementation Notes section written by hand-editing the task file *before*
running the close-out `-s Done --notes "..."` command was silently discarded, confirmed
by diffing the file before and after. Recovered by re-adding the detailed text inside
the same markers, after the CLI's short summary.

**What to do.** Run the CLI `--notes` command first (or use only the CLI's text), then
hand-edit to elaborate — never the other order. Diff the task file after any `--notes`
call to confirm what survived.

---

## Never `git add -A` while resolving a rebase conflict

**What happened.** `git add -A` during conflict resolution swept **dev's own renames**
back to their pre-rename names, silently reverting another agent's work. Caught only by
auditing `git ls-files` against `git ls-tree origin/dev` afterwards.

**What to do.** Stage conflicts **file by file**. After any rebase touching `backlog/`,
diff your tree's task filenames against dev's.

---

## Regenerate the CSS bundle; never hand-merge it

**What happened.** `tldw_cli_modular.tcss` is generated and conflicts on essentially
every rebase, and running the app rewrites its timestamp.

**What to do.** On conflict, take either side, then re-run `python
tldw_chatbook/css/build_css.py` and stage the result. Verify by diffing the bundle's
top-level selectors before and after — that shows what actually changed, past the
timestamp noise.

---

## Scripted `.replace()` matches the *first* occurrence

**What happened.** Twice in one session, a scripted edit landed in the wrong place:

- `import dataclasses` stranded in its own block, breaking import grouping.
- A screen attribute initialised inside `_reset_library_ingest_transient_state()`
  instead of `__init__`, because the anchor string appears in both. Restored sessions
  never run that reset, so **the media viewer stopped mounting** — shipped and caught
  two commits later.

**What to do.** Anchor on a string unique to the target, or verify where the edit landed
before moving on. For state that must exist on every path, prefer a **class-level
attribute** over an assignment in one method.

---

## Audit the board; do not trust your own summary

**What happened.** Two tasks had shipped and been verified live, but their status was
never moved off `To Do`. Found only by auditing every task in the programme at the end.

**What to do.** Before declaring a programme complete, list every task's status **from
`origin/dev`**. "I fixed it in that PR" is not the same as the board saying so.

---

## Verify a reported bug still exists before filing it

**What happened.** A whole-branch reviewer flagged that `LabModeStrip`'s active-mode
label was invisible — "the second occurrence" of a one-row strip inheriting the global
`.is-active { border: round }` and clipping its own text row. The claim was accurate
about the mechanism and the file, so it was filed as a task (864, renumbered 875) and
carried through a PR body and two memory notes.

It had already been fixed. Commit `880febc05`, four days earlier, moved the chip rules
app-tier into a new `css/features/_lab.tcss` for exactly this reason. The task was
implemented by confirming the existing fix and changing no production code.

The reviewer was not wrong about the bug class — the Watchlists tab strip really did
have it, and really was fixed in the same programme. What went unchecked was whether
the *second* instance was still live on current `dev`.

**What to do.** A finding about code the current task does not touch is a **report, not
a filing**. Before it becomes a task, grep `origin/dev` for the fix, not just for the
bug: `git log -S'<the selector or symbol>' origin/dev -- <path>`. This is cheap, and the
alternative is a task that reaches an implementer, consumes a full cycle, and closes as
"already fixed" — plus stale claims in whatever PR body and notes quoted it in between.

---

## Search the board for the defect before investigating it

**TASK-1022 / TASK-1210, 2026-07-28.** A runtime import trace established that
scheduled watchlist checks never ran: the feature flag had no `else` branch, the
old scheduler had no construction path, and the flag shipped false. That was
filed as TASK-1210 and fixed.

TASK-1022 already said all of it — filed a day earlier, from a plain reading of
the code, with the same four load-bearing facts and the same conclusion. The
investigation was duplicated because nobody grepped `backlog/tasks/` first.

**What to do.** Before investigating a defect, grep the board for its subject —
here, `grep -il "watchlist.*schedul" backlog/tasks/` would have surfaced it in
one command. Do it even when the finding feels new, and *especially* when it
feels like a discovery: a confident diagnosis is exactly the state in which you
skip the check.

Closing the duplicate is not enough on its own. Say in the surviving task which
one was first and what it already knew, so the board records that the second
investigation was avoidable rather than quietly implying two independent
confirmations.

---

## Related

- `lessons-testing-evidence.md`
- `backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md`
