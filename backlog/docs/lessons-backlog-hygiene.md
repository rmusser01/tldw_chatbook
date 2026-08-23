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
- PR #1448 exposed a duplicate already merged to `dev`: QwenCloud had claimed
  `TASK-3603` first, then the Watchlists phase-3 task reused it the next day. The
  branch that finally tripped Backlog Guard did not introduce either side of the
  collision; fixing the gate required tracing both files' add commits and renumbering
  the later task to `TASK-3791` across its code, tests, plan, and task record.

- **2026-08-10, supervisor-fleet PR 2a.** Two distinct failures in one branch. (i) The
  branch was cut from dev *before* a `task-3401` renumber merged, so Backlog Guard went
  red on a duplicate **dev had already fixed** — the branch introduced neither side, and
  the fix was `git rebase origin/dev`, not renumbering anything. (ii) A task filed
  mid-session as `13213` collided with a task another session landed on dev while the PR
  was in flight. The ceiling moved from **13,212 to 14,826 in a few hours** of concurrent
  work — an ID scanned at the start of a long PR is not safe at the end of it.

- **2026-08-11, RAG P2ab arc (TASK-15020).** Three collisions in ONE arc, and the gap
  widened each time. Task 1: the CLI offered `14913` against a true max of `14920`.
  Task 5: the CLI offered `15021` while the global max was `15270` — the brief had
  carried a "next safe id" estimate of ~15021 derived hours earlier, and it was **249
  short**. Task 9: the CLI offered `15401` against a true max of **15482**, swept live
  across 166 worktrees and 39 remote refs. Every offer was plausible; every one was
  wrong, by 7, then 249, then 81. The Task-5 lesson bears repeating on its own:
  **never pass a "next safe id" between tasks in a brief** — an ID is only safe at the
  instant it is derived, so re-derive it at filing time.

Checking `origin/dev` *feels* like diligence. It is not: parallel agents hold IDs on
unmerged branches. And a *green* Backlog Guard at branch time proves nothing later —
a duplicate can arrive from dev moving underneath you, in which case rebasing is the
fix and renumbering is actively wrong.

**Cheap habit that makes the collision visible before it costs anything:** create one
throwaway task FIRST and read which ID the CLI assigned it. That is the CLI's answer,
exposed before it is attached to anything real — delete the probe, then leapfrog past
the swept maximum. All three P2ab collisions were caught this way, and no file ever
carried a bad number.

**What to do.** Before filing, sweep **every remote ref** plus every worktree, and
re-check at merge time — dev moves under you. Never trust the CLI's auto-assignment.
When a collision is found after both tasks have started, use add-commit provenance:
the later claimant moves, and every reference in its shipped slice moves with it.

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

**MAX+20 leapfrogging has now failed TWICE against concurrent minting — 2026-08-21
(TASK-19573/TASK-19601).** Seven ids were each claimed by two unrelated task files,
in two internally-clean batches: `16320`, `16322`, `16323`, `16324` (note `16321`
was never duplicated) and `18912`, `18913`, `18915` (note `18914` was never
duplicated). Each batch's minting session DID sweep and leapfrog past an observed
maximum -- that is exactly why every id within a batch was unique. The failure was
that two different sessions performed that sweep against the *same* pre-merge
maximum from different worktrees/branches, so their two leapfrogged blocks
overlapped wholesale. Leapfrogging further (MAX+30, MAX+50, ...) does not fix this
class of failure -- it only widens the window a collision can hide in, because the
race is not "did you leapfrog far enough," it is "did anyone else observe the same
maximum before you merged."

Four of the seven pairs were also **Done vs. Done**, which the standing "a Done
task never moves" reading made unresolvable at the file level -- fixing only the
three resolvable pairs would not have turned the guard green, since the other four
still collided.

**The fix was a tie-break rule, not more distance.** TASK-19601 decided: the
OLDER arrival (by `created_date`) keeps the id regardless of status; the younger
task renumbers, carrying a `## Renumbering provenance` section that names the old
id, with every inbound reference (`dependencies:`, docs, plans, code comments)
moved to match. `.github/workflows/backlog-guard.yml`'s failure message now states
this rule directly, so the next collision is actionable without re-deriving policy
from scratch.

**What to do.** Sweeping and leapfrogging reduces collision probability but cannot
eliminate it while multiple sessions mint ids from an observed maximum
concurrently -- treat every freshly-claimed id as provisional until it lands on
`origin/dev`, and when a collision is found anyway, apply the older-keeps-id rule
instead of trying to out-sweep the next session.

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

## A clean scoped rebase can still invalidate a repository-wide manifest

**TASK-856, 2026-08-08.** Tasks 1–3 and their scoped reviews were clean when the
branch rebased onto a newer `dev`. None of TASK-856's scoped production or test
files had changed upstream, yet the cross-cutting diagnostic inventory turned
red: five new `app.py` diagnostics plus related owner and sink line movement
made the generated manifest differ from the checked one.

**What to do.** After every rebase, rerun repository-wide generated-manifest and
architecture gates even when the scoped file diff is unchanged. Review the
upstream semantic delta and commit it as a separate boundary before claiming
that the feature branch changes only its intended inventory entry.

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

**TASK-15810, 2026-08-14 — the same trap one level up: the board was grepped,
and it still happened.** A session picked up TASK-15810 (High, `status: To Do`),
wrote a spec and a plan, and spent a full task building a reproduction. Another
session had been on it for hours: 16 commits in `.worktrees/task-15810-rag-first-query`,
a 369-line profile report naming the root cause, the fix, and 492 lines of
tests. Its branch was **local-only and never pushed**, and it had not edited the
task file — so the board still read `To Do`, accurately, while the work was
essentially done.

**The status field is not a lock.** It changes when someone edits the file, which
is the last thing a session does, not the first. Reading a task and finding it
open tells you nobody has *finished*; it tells you nothing about whether someone
is *working*.

**What to do.** Before starting an arc on a task id, check for a live claim as
well as a filed one — two commands, both cheap:

    ls .worktrees/ | grep -i <id>
    git branch -a | grep -i <id>

The first is the one that catches the local-only case, which is the common one:
a session mid-arc has a worktree long before it has a remote branch. `ps aux |
grep <worktree>` then tells you whether that session is still live, which decides
whether you may touch the checkout at all (you may not — see the standing rule).

**When you find one mid-flight**, the recovery is cheap if you stop early:
contribute to the existing branch rather than racing it. Here the duplicate work
had produced an independent reproduction that *refuted the second session's own
leading hypothesis* before it saw the first session's profile, so it shipped as a
corroboration commit on top of the existing tip (PR #1640) instead of a competing
fix. Independent evidence is worth keeping; a second fix for the same defect is
not.

---

## Check for an in-flight PR before designing — claiming the task does not reserve it

**TASK-595 and TASK-596, 2026-07-31 and 2026-08-01.** Both were implemented
**twice, in parallel, by different agents**, and both duplicates were discovered
only at merge time.

TASK-595 was the first. The post-mortem
(`Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md`)
proposed two guards: claim the task as `In Progress` before designing, and put
the task id in the spec filename so a `find` surfaces it.

**Both guards were applied to TASK-596. It was duplicated anyway, the very next
day, more expensively.** The spec and plan were committed to `dev` at 09:42 with
the task claimed and `task-596` in both filenames. PR #1175 — another agent's
complete implementation of the same phase — merged at 13:40. Roughly 6,900 lines
of work survived as ~1,000 lines of portable delta.

**Why the guards failed, and why one of them backfired.** A claim on the board
tells someone who *starts after reading it* to stay away. It says nothing about
work already in flight, and nobody re-reads the board mid-build. Worse, the
duplicate's branch name (`codex/task-596-model-browser-phase-1`), file paths,
rail keys and phase decomposition matched the published plan closely enough that
it had almost certainly been *worked from*. Publishing a design to a shared
branch is an invitation as much as a claim: it hands a complete blueprint to
anyone looking for work while giving the author a false sense of having
coordinated.

**What to do.** Search for in-flight work by task id — branches *and* pull
requests — before designing, again before implementing, and periodically during
any build long enough for someone else to start one. It costs one second:

```bash
gh pr list --state all --search "596" --json number,title,state,headRefName
git for-each-ref --format='%(refname:short)' refs/remotes/ | grep -i '596'
find / -name "*task-596*" -not -path "*/.git/*" 2>/dev/null   # sibling clones
```

Neither TASK-595 nor TASK-596 ever had `gh pr list --search` run against it, in
sessions spanning many hours each. That single command would have caught both.

The board is not a lock. Treat a claim as documentation, and the PR list as the
actual source of truth about who is building what.

**Third instance: TASK-15455, 2026-08-11 — same day, same task, neither session
ran the command.** The Console transcript windowing task was implemented twice
in parallel: `codex/console-transcript-windowing` (PR #1538, merged 13:26) and
`task/15455-windowed-mount` (PR #1556, closed unmerged). Both sessions worked
for hours; both wrote their own pin suites, probes, and lessons entries; neither
ran `gh pr list --search "15455"` before designing or during the build. The
duplication was found only when the second PR was opened against a dev that
already contained the first.

The cost is measurable this time: the merged-first rule stood, and the second
implementation's ~1,500 lines became a ~250-line reconciliation delta — roughly
one full task cycle spent twice, plus a third cycle to port. What survived was
only what the second session's REVIEW round had found (a prune/hydration
oscillation the merged build genuinely had, a reading-state restore gap, the
config kill switch); every line of its actual windowing mechanism was thrown
away because the merged design was equivalent or better.

Two notes for the next time this happens. First, the reconciliation is worth
doing properly rather than abandoning: the second implementation's review had
found a real infinite-churn defect in the merged one, which no amount of "we
merged first" makes untrue. Second, the porting brief that starts a
reconciliation is written by someone who has read neither implementation
closely — verify every "the incumbent is missing X" claim against the incumbent
at YOUR head before porting X. One of the four items in this reconciliation's
brief (selection into windowed-out history) was already implemented and
working; porting it blind would have duplicated a mechanism inside a PR whose
whole purpose was undoing duplication.
### Fourth instance: TASK-15457, 2026-08-12/13 — found only when the rebase would not apply

**Two complete implementations of TASK-15457 ("Library: convert per-click
whole-screen recomposes to canvas-scoped sync") existed simultaneously, and
neither session knew.** Codex's landed on `dev` as `976dbafcb`
("perf(library): scope frequent updates to canvases") on 2026-08-12 and marked
the task **Done**; the other ran 2026-08-11→13 on
`task/15457-library-recompose`, through two review rounds, and was discovered
only when the coordinator tried to merge it and dev had moved.

Both wrote a file at the SAME path, `Tests/UI/test_library_canvas_scoped_sync.py`
(441 lines vs 687), and both edited the same task file. The duplication was
invisible to every check either session ran: tests passed on both sides, the
task file on the branch still said `In Progress` because dev's `Done` was on a
commit the branch predated, and `git status` was clean.

**What made this one different from 595/596: the duplicate WON on scope, and
the survivor was the defects.** Dev's implementation converted more sites (143
→ 102 vs 152 → 132) and moved the list→editor loading views into their owning
canvases. So all eight conversion slices were dropped as superseded. What
survived was the part the other implementation lacked — four defects that
reproduced RED against dev's own code: an unmirrored `_selected_media_id` (the
media chooser opened a filtered-out item), focus escaping the canvas on every
converted site, a dropped compact-mode scroll offset, and unarmed editors on
the row→editor transition. **Roughly 1,200 lines of implementation reduced to
~190 lines of portable delta.**

**The specific check that would have caught it, and why the published one did
not.** The 595/596 lesson prescribes `gh pr list --search "<id>"` before
designing and periodically during a long build. That still would not have
fired here: `976dbafcb` reached dev as part of a batch whose PR title names
neither the task id nor the Library screen. The check that works for a task
this long-running is against **dev itself**, not the PR list:

```bash
git log origin/dev --oneline -S "task-15457" -- backlog/tasks/ | head   # task file touched?
git log origin/dev --oneline -- "Tests/**/*canvas_scoped_sync*"          # my test path claimed?
git fetch -q origin && git log --oneline $(git merge-base origin/dev HEAD)..origin/dev -- <the files I am editing>
```

Run the third one before every review round, not just at merge. A task whose
work spans days will outlive any snapshot of dev taken at its start.

**Related trap this surfaced:** a parallel implementation can carry the same
defects yours does. Do not assume the version that reached dev first is the
correct one — the four defects above shipped `Done`, and one of them (the notes
footer) *appeared* fine only because an unrelated per-refresh mechanism masked
it, which a non-discriminating test then "confirmed". Probe by disabling the
mechanism you think is responsible before claiming it is.

**Note for whoever merges this:** the third-instance block for this lesson
(TASK-15455) is on `dev` but was NOT in this branch's rebase base
(`74c8cf7043`), so this entry appends to the TASK-595/596 section instead. If
both land, renumber the instances so the sequence reads 1-2-3-4.

---

## A duplicate ID on dev may be a resurrected ghost — check for a renumbered twin BEFORE renumbering

During the TASK-2750 sweep (2026-08-06), dev carried 35 duplicated task IDs.
The obvious fix — renumber one side of each pair — was WRONG for over half of
them: 25 of the 47 colliding files were **stale pre-renumber copies of tasks
that earlier dedup sessions had already moved** (400→542, the 401 epic + nine
children→553.x, 402→561, the 506-518 STT batch→593-605, 519→869; the moves are
recorded in task-542/544/554/561/869's own notes). The ghosts came back through
merges of branches cut before those renumbers landed. Renumbering a ghost to a
fresh ID would have minted a THIRD copy of each task.

Before renumbering a duplicate, search the tree for another file with the same
title marker at a different ID (`ls backlog/tasks | grep <title-fragment>`).
Twin exists → the duplicate is a ghost: delete it and point any of its
references at the twin. No twin → renumber per the keeper rules (Done/older/
load-bearing side keeps; references are attributed per side by their context,
never rewritten wholesale — the same integer means different tasks in
different files).

---

## Do not assign zsh's special lowercase `path` variable

**TASK-3401.6 closeout, 2026-08-09.** An all-ref/all-worktree collision sweep
used `path` as the loop variable. In zsh, lowercase `path` is a special array
tied to uppercase `PATH`, so the first assignment replaced command lookup with
one worktree directory. The title scan had completed, but subsequent `git`,
`tr`, `sed`, `sort`, and `tail` calls failed with `command not found`; the
maximum-ID and worktree portions of the sweep were therefore incomplete. No
repository file changed, and the sweep was corrected and rerun before either
follow-up task was created.

**What to do.** In zsh scripts, never assign to lowercase `path`. Use a
task-specific variable such as `task_file` or `worktree_dir`. After shell-variable
setup in a multi-step Backlog script, validate command lookup (for example with
`command -v git rg sort`) before trusting later stages of the script.

---

## `backlog task <id>` and `task edit <id>` silently do nothing for a FIVE-digit id

**TASK-15463, 2026-08-11 (backlog CLI 1.44.0, the bun build on `PATH`).** The
standard opening moves — `backlog task edit 15463 -s "In Progress" -a @claude`
then `--plan "..."` — both printed `Updated task TASK-` and exited 0. Neither
touched `backlog/tasks/task-15463 - ....md`. What they actually did was create a
new file literally named **`backlog/tasks/task-task- - .md`**, with an empty
title, empty status, and `id: TASK-TASK-`, carrying a copy of the real task's
description. It reappeared on every retry.

`backlog task 15463 --plain` shows the same failure read-only: it resolves and
prints the correct *file path*, then renders `Task TASK- - ` with an empty
status and no acceptance criteria. `backlog task 4026 --plain` (four digits) on
the same checkout renders perfectly, and `backlog task list --plain` lists all
84 five-digit tasks correctly — so this is specific to addressing a single task
whose id is five digits. `TASK-15463` and `15463` behave identically.

**What to do.** For a five-digit id, edit the task file directly (status,
assignee, `## Implementation Plan`, `## Implementation Notes`, and the AC
checkboxes inside the `<!-- AC:BEGIN -->` markers) — that file is the source of
truth the board reads. After ANY `backlog task edit` on a high id, run
`git status backlog/` and delete a stray `task-task- - .md` if one appeared;
committing it would put a nameless, statusless task on the board.

---

## Markdown hard breaks can pass review but fail diff-check—or lose rendering when stripped

**TASK-694 closeout, 2026-08-12.** The ownership design used two trailing
spaces to render metadata and ADR fields on separate lines. The required
`git diff --check origin/dev...HEAD` gate correctly blocked closeout on those
spaces. Removing them made diff-check green, but also removed the Markdown
`<br>` separation; an independent quality review caught the rendering change.

**What to do.** Use bullets, blank paragraphs, or another explicit semantic
structure for governed Markdown metadata. If whitespace cleanup touches a
Markdown hard break, render-compare the result before calling the edit a
no-op.

---

## Gitignored working files die with their worktree — keep long-lived ledgers above it

**Supervisor-fleet PR 3a-1 → 3a-2, 2026-08-13.** PR 3a-1's working ledger —
the full seven-seam audit table, the decision records, every task report —
lived in the branch worktree's gitignored `.superpowers/sdd/` directory.
Deleting that worktree after the merge (routine, correct cleanup) destroyed
all of it. The distilled record survived only because it had been copied
OUT before the deletion — the committed plan, the lessons entries, the
backlog tasks 15660-15671, the PR #1557 body — but the full audit detail is
gone permanently: when PR 3a-2 needed 3a-1's seam-audit evidence days
later, it had to re-derive it by execution. PR 3a-2's ledger was therefore
placed at the repository root
(`<repo>/.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/`),
above every worktree, and that placement is what let its reports survive
the branch's own multi-agent worktree churn.

**What to do.** A gitignored file's lifetime is its directory's lifetime,
and a worktree is disposable by design — post-merge deletion is the normal
end of its life, and everything inside it that exists nowhere else is part
of what gets deleted. Working state that must outlive the branch (audit
tables, decision logs, incident reports) goes either into the repository
as a commit or into a location above every worktree. Choose at
ledger-creation time; by cleanup time it is too late.

---

## `backlog task create -p <priority>` corrupts the assigned ID

**What happened.** 2026-08-19, filing the Chunking Parity phase tasks: `backlog task
create "…" -d "…" --ac "…" -l chunking -p high` produced a task whose ID was
`TASK-HIGH.1` — the priority flag's value was parsed into the ID slot. The CLI
printed no error (`No Definition of Done items defined`, as usual), and the malformed
file landed as `task-high.1 - ….md` with `id: TASK-HIGH.1` in its frontmatter.

**What to do.** Never pass `-p`/`--priority` to `backlog task create`. Set priority by
editing the task file's `priority:` frontmatter field directly (the CLI reads it fine),
or via `backlog task edit <id> --priority high` after creation. Combined with the two
known quirks above — CLI auto-assignment ignoring remote-held IDs, and `--ac` joining
commas into one run-on criterion — the reliable filing path is: sweep remotes for the
true max ID, write the task file directly with a safe ID and a hand-authored AC block,
then verify with `backlog task <id> --plain`.

---

## In zsh, brace `"${b}:path"` — unbraced `"$b:path"` applies the `:t` modifier and the sweep lies

**What happened.** 2026-08-21, the chunking-template-parity spec's §5 version
sweep (`_CURRENT_SCHEMA_VERSION` across all refs). The loop ran
`git show "$b:tldw_chatbook/DB/Client_Media_DB_v2.py"` per remote ref and
returned the **same old version for every branch** — "clean, no collisions
anywhere." That was the bug: in zsh, `:t` after a parameter inside quotes is
a **history modifier** (`:t` = tail-of-path), so `"$b:path"` silently
rewrote the ref. `git show` then resolved something other than the intended
`<ref>:<path>` (or errored into the fallback), and the sweep reported a
uniform answer that proved nothing about any ref. The spec's sweep results
were only trusted after re-running with the braced form — which is also the
form the sweep snippets in this file already use (`"${b}:backlog/..."` in
the collision-sweep habit above, where the braces make it work).

**What to do.** Any zsh loop that builds a `<ref>:<path>` or `<var><suffix>`
string must brace the variable: `git show "${b}:path"`, `git log
"${b}..HEAD"`. If a per-ref sweep returns an identical value for every ref,
treat that as the sweep being broken, not as the board being clean — 200
refs agreeing exactly is the signature of a loop that never varied its
input. Sanity-check with one ref you know differs
(`git show "${b}:path" | grep VERSION` against two refs you expect to
disagree).
## ADR numbers collide across concurrent branches — re-verify at merge time

**What happened.** 2026-08-20, two independent sessions in the same window: the
chunking-parity branch drafted its engine ADR as 072 and found at PR time that other
branches had contended for the same number (it shipped as ADR-073); the hermes-parity
branch (PR #1832) drafted its server-offload ADR as 072 on 2026-08-19 and discovered
the collision only as a merge conflict in `backlog/decisions/README.md` — by merge
time on 2026-08-21, dev had claimed 072 (checkpoint harness), 073, 074, AND 075 from
other branches, forcing a rename to ADR-076 (file, `# ADR-NNN` header, README row,
and the owning task's plan references) mid-merge. The thesis then recurred on
itself: 076 was ALREADY claimed (library-lifecycle landed `1c567f3ae` at 14:24 the
same day, four hours before the 18:44 renumber), caught only the next day, and the
ADR renumbered again to 077 (TASK-19610) — the merge-time check must cover the
number being renamed TO, not just the one being renamed FROM.

**What to do.** ADR numbers have exactly the same collision dynamics as task IDs
(see "assign against origin/dev" above), but no CI guard. Treat the drafted number
as provisional until merge. At merge time, before resolving any README conflict:
`git ls-tree --name-only origin/dev backlog/decisions/` for the authoritative taken
list, plus a sweep of open PRs' changed files for `backlog/decisions/` claims. If
renumbering, grep the repo for `NNN-<slug>` AND both header forms — `ADR-NNN`
(hyphenated, ~90 files today) and `ADR NNN` (spaced, e.g. `# ADR 008 - ...`; a
`ADR[- ]NNN` character class covers both) — the owning task's plan section
references the ADR by number and path, and stale references in either form
mislead the next session.

## Backlog filenames must survive every supported checkout platform (TASK-21139)

**What happened.** On 2026-08-22, commit `46cb7bc1f` added TASK-21130 with `>`
in its tracked filename. Git for Windows fetched the repository, but
`actions/checkout` exited 128 before project tests ran in runs `32617893248` and
`32617893237`.

**What to do.** Content may stay expressive, but direct files in the live,
completed, and archived Backlog buckets must use Win32-compatible basenames.
The shared stdlib guard is the authoring-time source of truth because Windows
cannot run repository code before checkout succeeds.

---

## Related

- `lessons-testing-evidence.md`
- `backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md`
- `Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md`
