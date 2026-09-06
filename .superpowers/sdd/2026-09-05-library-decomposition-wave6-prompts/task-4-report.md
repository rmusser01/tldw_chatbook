# Wave-6 Task 4 — Wave close (prompts series)

Branch: `refactor/library-decomp-wave6-prompts`
Worktree: `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`
Parent (task 3 tip): `3eba0592f`
Wave-6 start commit: `e5e03846a`

| Commit | Subject |
|---|---|
| `33cffc4a4` | `docs(library): wave-6 stale-prose sweep -- comment-only, line-neutral` |
| `3cf89dbe9` | `docs(backlog): file the two cross-wave follow-ups the prompts wave surfaced` |
| `a595347f3` | `docs(library): wave-6 close -- recipe institutional memory, sweep + probe evidence` (also carries the TASK-31816 AC correction) |
| `12749306d` | `docs(sdd): wave-6 task-4 report + brief (durable evidence)` |

All hashes read out of `git rev-parse HEAD` / `git log --oneline` at the
moment each commit was made. The evidence commit's own hash could not be
written into the file that commit contains (a hash cannot be known before
the commit exists), so it is filled by a one-line follow-up commit — the
same two-commit shape every `.git-blame-ignore-revs` entry in this series
uses, and the reason §6 now names a commit message as the one place a
number's copy can never be corrected after the fact.

Every number below was verified against the
live file before it was written — this wave's recurring failure mode
(three separate incidents; §5 below).

---

## 1. Stale-doc sweep — six corrections, all line-neutral

Commit `33cffc4a4`. Every edit is comment/docstring-only and line-for-line
neutral, so no ratchet row moves and the byte-for-byte canon is untouched
(none of the six sits inside a moved body).

### 1.1 The two task-3-disclosed sites

| Site | Was | Now |
|---|---|---|
| `Tests/UI/test_library_prompts_characterization.py:48` | "``_library_prompt_detail`` and ``_library_prompt_block_state`` both back to ``None``" | "``_prompts_state.detail`` and ``_prompts_state.block_state`` both back to ``None``" |
| `tldw_chatbook/Library/library_prompts_state.py:2405` | "callers thread the screen's own ``_library_prompt_dirty`` flag through" | "callers thread the screen's own ``_prompts_state.dirty`` flag through" |

The second is the **DOMAIN** module of the basename-collision pair
(`tldw_chatbook/Library/library_prompts_state.py`), not the UI state module
— comment-only, no code, and that file is not governed by either size
ratchet (the controller ratchet globs `UI/Library_Modules/*_controller.py`).

The characterization file's SIBLING passage at `:268-269` spells the same
two flat names and was deliberately **left alone**: it immediately labels
them "two of the moved ``LibraryPromptsState`` fields", so it is
self-consistent. That is the distinction task 3 flagged and this sweep
formalized into the recipe — a bare name-count census cannot tell a
self-labeled relocated reference from a present-tense mis-attribution,
because the two spell the identical name.

### 1.2 The two task-3 forward-noted out-of-scope sites — both FIXED

The brief said "fix if one-phrase corrections, else leave with the forward
note and say why". Both qualified, and the check that settled it was not
"is the sentence short" but **"does the name still resolve where the prose
says it does":**

| Site | Was | Now | Why it is now FALSE, not merely indirect |
|---|---|---|---|
| `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:131` | "see ``LibraryScreen._update_library_prompt_meta_static``" | "see the prompts controller's ``_update_library_prompt_meta_static``" | `_update_library_prompt_meta_static` is in `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` — **no screen delegator survives**, verified by `grep` returning zero `def` on `library_screen.py` |
| `tldw_chatbook/UI/Console_Modules/prompts.py:1854` | "same way ``library_screen._save_library_prompt``'s own create path does" | "same way ``LibraryPromptsController._save_library_prompt``'s own create path does" | same — `_save_library_prompt` is also on the pruned list |

Had either name kept a delegator, the prose would have been merely
indirect and the honest call would have been to leave it. Both were pruned,
so both attributions were false as written. Wrapping was re-flowed to stay
under the files' own line width; both edits are 2-lines-in / 2-lines-out.

### 1.3 Two MORE, found by this task's own repo-wide census

Not disclosed by any prior task, and **in files no task in the series had
any reason to open** — which is the whole argument for making this a
standing wave-close census rather than a per-task file-scoped sweep:

- `tldw_chatbook/Library/library_shell_state.py:44` — "The screen
  distinguishes ... by view/selection state (``_library_prompts_view ==
  "editor"`` plus a ``prompt_id=None`` sentinel)". → `_prompts_state.view`.
- `Tests/Library/test_library_shell_state.py:371` — the same sentence
  mirrored in that module's own test. → `_prompts_state.view`.

**Method.** Tokenize every `.py` in the repo (COMMENT and STRING tokens
only, so a live attribute access cannot masquerade as prose); match all 43
deleted flat field names (read out of `LibraryPromptsState` via
`prompt_state_shim_attr`, not hand-listed) plus all 39 pruned method names
(read out of `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` by `ast`, not
hand-listed); then narrow to hits carrying a screen attribution
(`LibraryScreen`/`library_screen`/"the screen's") within ±3 lines.
**145 raw prose lines → 8 candidates → 6 real defects.**

> **A census bug worth recording, because it produced a confidently wrong
> zero.** The first run of that script reported **0 hits** — including for
> `library_export_controller.py:194`, which certainly contains one of the
> names. Cause: the skip-list (`.worktrees`, `.venv`, …) was applied to
> `path.parts`, and this worktree's own absolute path *contains*
> `.worktrees`, so **every file in the repo was skipped**. The script
> reported a clean sweep over zero files and looked exactly like a clean
> sweep over 3,000. Caught only by asserting a known-positive
> (`"_library_prompts_mutation_in_flight" in src` → `True` while the census
> said 0) — the same "prove the instrument fires before trusting a null
> result" discipline the recipe already demands of mutation testing.
> `SKIP_DIRS` must be matched against `p.relative_to(ROOT).parts`.

### 1.4 Surveyed and deliberately NOT changed

Recorded so the next sweep does not re-derive them:

- `Tests/UI/test_library_prompts_canvas.py:5416` — "so the screen's own
  pre-check, exercised here, is what actually detects staleness -- see
  ``_save_library_prompt``'s docstring". Architectural prose about the
  Library screen's surface, not a name-level attribution to a class; the
  named method still exists (on the controller) and the sentence never
  spells it `LibraryScreen._save_library_prompt`. Borderline, and left
  rather than widening the diff into a file task 3 already retargeted
  heavily.
- `tldw_chatbook/UI/Library_Modules/library_skills_state.py:261-264` —
  names prompts flat names as explicitly comparative prose ("Mirrors the
  prompts editor's own state shape"). It also carries its OWN skills-era
  staleness (`_selected_skill_name` for what is now
  `LibrarySkillsState.selected_skill_name`), which is out of this wave's
  scope.

### 1.5 The four prompts modules — swept, clean

The brief's named sweep target (UI state, controller, wiring test,
characterization) was searched for present/future-tense claims made stale
by the completed series (`will`, `task 3`, `cleanup PR`, `deferred`, `still
lives`, `currently`, `until the`). **Zero defects.** Every hit is already
past-tense and accurate — e.g. the UI state module's "this series' own
cleanup PR (task 3) DELETED that block", the controller's "(deleted at
cleanup, task 3 …)", the wiring test's "Filled in by this series' own
cleanup task". Task 3's own docstring pass did its job on its own file
scope; what it could not reach was everything OUTSIDE that scope, which is
exactly what §1.3 found.

---

## 2. Backlog filings — TASK-31815 and TASK-31816

Commit `3cf89dbe9`.

**Id sweep, run FIRST and live** (collisions have bitten this repo
ten-plus times): after `git fetch --all`, the true max task id across
**609** local + remote refs (`git ls-tree` over `backlog/tasks`,
`backlog/drafts`, `backlog/archive`, matching on the id prefix only per the
hermes-parity lesson) is **31811**; the max across every worktree working
directory is 31701. The **CLI probe** — create one throwaway task, read the
id it assigned, delete it — offered **31702**, which is **109 BELOW the
true max**. That is the ordinary collision case, caught before it attached
to anything real. Filed at **31815** and **31816**, leapfrogging the swept
max. `scripts/check_backlog_task_ids.py` green afterwards (no duplicates
across 3,317 task files, all paths Windows-compatible).

- **TASK-31815 — cross-wave Library modal-inventory repair.** 6 ACs,
  grouped per row cluster as the brief required: the skills-era
  `_present_library_skills_import_choice_if_needed` discovery blocker; the
  stale `handle_library_ingest_browse` row; the two skill-trust passphrase
  presenter rows; a concrete-modal-type check; the file green end-to-end;
  and removal of §7's standing entry.
- **TASK-31816 — `_ui_ready` census zero headroom.** 5 ACs. Framed as
  dev-owned with `06acf148f` (dev's own pre-import paydown) named as the
  precedent for the paydown direction, and the decision — stated headroom
  vs. paydown — required to be recorded in the guard's own docstring with
  its reasoning, so the next Library wave does not re-derive it.

Every AC was passed as its own repeated `--ac` flag; comma lists do not
split.

---

## 3. Recipe updates

All in one commit, per the brief. Section numbers verified against the live
file rather than assumed — §3 is "Monkeypatch-name routing", §4 is "The
transform whitelist", §6 is "Measure after final rebase", §7 is "Sweep
evidence", §8 is the subsystem-order table, §20 is the ingest series, §21 is
the prompts series.

### 3.1 §4 — the `on_<Message>` transform folded in PROPER

`§4` gained a subsection, **"The delegator-prune whitelist — THREE members,
not two"**: `@on`-decorated, `action_*`, and `on_<message-handler-name>`.
The `@on`/`action_*`/`on_<...>` triple also now appears in §4's own bullet
list, so a reader who never scrolls past the bullets still sees it.

Textual's dispatch cited from the INSTALLED source, verified by reading it
(version pinned in the text as **8.2.8**, read out of
`textual.__version__`):

- `textual/message.py:86` — `cls.handler_name = f"on_{name}"`.
- `textual/message_pump.py:743-758` — `_get_dispatch_methods`'s signature
  through its `for cls in self.__class__.__mro__:` MRO walk. (`:743` is the
  `def` line; `:758` is the MRO `for`. Counted off the file, per §6's own
  line-range lesson.)

The incident is stated, not just the rule: all six
`on_prompt_block_editor_*` handlers were marked PRUNE on a **genuine**
zero-reference count, and the deletion **would have stayed green** because
the canvas widget handles two of those messages one level down.

> **A count correction this task made to §21 while writing it.** §21 said
> the naive census's hits were "2 in `library_prompts_canvas.py`, 4 more in
> a `Tests/UI/test_prompt_block_editor.py` harness". Re-derived by
> `grep -rn "def on_prompt_block_editor_"` over `tldw_chatbook/` + `Tests/`
> + `Docs/` + `Helper_Scripts/` + `scripts/`: the harness carries **7**
> defs, **6** of which match the six moved names (the 7th,
> `apply_requested`, matches the one `on_prompt_block_editor_*` handler
> that is an EXCLUSION and never moved). True total: **8 same-named defs**,
> not 6. §21 corrected in place with the derivation shown.

### 3.2 §3 — two new census entries, deliberately NOT numbered as bypass shapes

The catalogue of numbered bypass shapes stands at **eight**; both additions
say so explicitly, because neither is a bypass.

**(a) The patch-target TABLE row** — a fourth SPELLING every census in §3
must search for. Exemplar `Tests/UI/test_library_shell.py:5146` quoted as a
code block (verified against the live file: `:5141` is the `for target,
name, key in (`, `:5146` is the `(screen, "_request_library_prompts_browse",
"load"),` row, `:5152` the consuming `monkeypatch.setattr`). The point is
mechanical: the name exists only as a bare string in a tuple, so
`monkeypatch.setattr`-anchored and `<name>\s*\(` call-shaped searches both
score it ZERO. The prompts series ran the widened census and got a real hit
that happened to be one of its 22 exclusions — verified here
(`_request_library_prompts_browse` is absent from
`_PROMPTS_CLUSTER_METHOD_NAMES`, and the controller carries only a
late-binding `@property` for it at `:748-750`, not a moved body).

**(b) The deleted FIELD names' own prose sweep** — a cleanup-PR census §3
never required. Incident: `library_export_controller.py:375`, a DIFFERENT
subsystem's file, which is why no prompts-scoped sweep could reach it. The
entry's operative content is the **three-class split** that makes the census
usable at all (self-labeled relocated reference / past-tense narrative /
present-tense mis-attribution), because classes 1 and 3 spell the identical
name and no count can separate them.

### 3.3 §3 — the isolated-baseline carve-out, RULED

**Decision: DROPPED, not bounded.** The isolated worktree is now
unconditional for every baseline comparison of any length; `git stash -u`
and `git checkout <base> -- <paths>` are no longer baseline methods.

The reasoning is recorded in full, and it turns on a MEASUREMENT rather than
on an argument. The brief offered "write it down with a hard bound (e.g.
≤5 min, no session-limit exposure, never for sweeps)" or "drop it". I
drafted the five-condition bound first, then measured what the bound was
protecting against and the bound stopped being worth having:

- uv's own share of provisioning the isolated baseline used by THIS close:
  **2.2s** (1.00s resolve / 1.01s prepare / 177ms install, 101 packages,
  warm cache), read out of the install log.
- whole-worktree disk cost: **535 MB** (`du -sh`).
- **end-to-end provisioning: 5 seconds** (3s `git worktree add` of 12,866
  files, 2s `uv venv` + `uv pip install -e ".[dev]"`), 528 MB, measured at
  this close by timing a throwaway worktree and removing it immediately.
- Decisive: **a task creates ONE baseline worktree and reuses it**, so the
  marginal cost after the first check is zero. The carve-out was buying
  nothing.

Against that, a five-condition exception is itself a judgement call in every
task's head — and the direct evidence is that an unwritten version of
exactly that judgement drifted across two consecutive tasks, three uses
between them (task 1's 2m40s overlay, task 2's ~2m20s and ~1m50s pair),
without anyone flagging it, in a wave where task 2 then hit the very session
limit the rule exists to guard against. A flat rule is cheaper to follow
than a bounded one and here it is also nearly free.

Corroborating detail found while writing this up: **task 3 went isolated for
every one of its own paired checks without being told to**, naming the
editable-finder trap as the reason (§12 of its report). The practice was
already converging on the unconditional rule; only the written text lagged,
which is the cheapest possible moment to close a gap like this.

### 3.4 §6 — the satellite-copies lesson (both incidents, stated)

§6 gained "**A re-pin has THREE places the number lives INSIDE the guard
file … plus a fourth, outside it, that can never be corrected**":

- the `_BUDGETS` row (the only one a guard reads),
- the arithmetic derivation in the comment above it,
- the narrative prose line at the top of that comment block — the wave-6
  task-2 incident: `test_screen_size_ratchet.py:503` shipped "37718" three
  lines above a derivation resolving to 37722 and eighteen above a row
  pinned at 37722, guard green throughout, caught by review;
- the **commit message** — `6fd2b753a`'s "37576 … −146" against a true
  37574/−148, deliberately unamendable because `.git-blame-ignore-revs`
  holds that literal hash.

Plus the LINE-RANGE corollary, with its incident: a "re-verification" that
replaced a correct `:520-526` with `:518-524` (a range spanning two blank
lines and missing both the fifth `_OwnerScope` entry and the closing paren),
overturned only after two independent readers went back to the live file.

While writing that entry I checked the range against the CURRENT tree and
found the tuple at `:528-535` — the prompts cleanup added a sixth
`_OwnerScope` and shifted the block by 8 lines. Both `:520-526` and
`:528-535` are correct; neither is correct without naming the revision it
was read from. That observation went into §6 as its own clause, because it
is the same failure one step further out: a line range is a number with a
silent expiry date.

### 3.5 §20 — HOW to run the `_SURFACE` check

The standing "check every dead-import candidate against `_SURFACE`
individually, every task" rule lives in §20's import-verification lesson, so
the HOW clause was appended there: **exact-name lookup, never a
subsystem-word grep**. The incident is the prompts cleanup's first check —
a case-sensitive `grep prompt` returning **zero** while all five pinned
names spell it `PROMPT`. Screaming-snake constants are exactly what a
re-export contract pins, so the lowercase grep is wrong precisely where it
matters. Recorded honestly as a loud-red risk (a debugging round), not as a
silent-regression risk.

### 3.6 §7 — five additions to the documented-reds list

Enumerated in §4 of this report alongside the battery that re-confirmed
them.

### 3.7 §8 and §21

§8's prompts row gained the pointer to §21's "Wave-6 close" subsection and
a note that the `on_<Message>` finding is now §4 doctrine. §21 gained the
whole **Wave-6 close** subsection: pin trajectory (git-derived at every
commit), verification battery, sweep evidence, probe, and six lessons.

---

## 4. Battery

All commands from this worktree, `.venv/bin/python`, `-p no:randomly`;
`timeout` unavailable, so long runs bounded with `perl -e 'alarm N; exec @ARGV'`.

### 4.1 The combined close battery

Machine load at run time: **16.90 / 17.43 / 15.99** — a busy machine, named
per wave-4 close's own lesson 3.

One invocation covering all 7 wiring suites + support-layer surface + both
size ratchets + recompose census + pre-import closure + `_ui_ready` census +
all 5 characterization files + modal-dismissal + screen-reuse:

**301 passed, 4 failed** (253.50s). Every red is on §7's documented list as
updated by this close:

| Red | Disposition |
|---|---|
| `test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[chat_screen.py]` | §7, standing |
| `test_screen_size_ratchet.py::test_task_22507_4_does_not_worsen_chat_screen_base` | §7, standing |
| `test_library_modules_size_ratchet.py::…[library_media_browse_controller.py]` | **added to §7 by this close** — three wave-6 tasks called it "the documented pre-existing row" while §7 documented only the `chat_screen.py` pair |
| `test_library_modal_dismissal.py::test_library_modal_inventory_matches_declared_edges_bidirectionally` | **added to §7 by this close**; filed as TASK-31815 |

Zero `library_screen.py`- or `library_prompts_controller.py`-scoped failures.
The `_ui_ready` census PASSED on this run — it is a flapping guard and §7 now
says so explicitly rather than letting a lucky run stand as "the" result.

### 4.2 Fresh `_measure()` — both ratchets, exact

`37574 / 1282` (screen) and `4998` (controller), against pins of `37574 /
1282` and `4998`. Zero drift, nothing to lower. This close's own commits
touch neither governed file — the six stale-prose edits are all line-neutral
and land in `Tests/`, `tldw_chatbook/Library/`, `tldw_chatbook/Widgets/
Library/` and `tldw_chatbook/UI/Console_Modules/` — so no same-commit re-pin
was needed.

### 4.3 preflight

`./scripts/preflight.sh` — all six derived-artifact checks pass, run twice
(before the backlog filings and again after every doc edit): CSS bundle + 9
per-screen stylesheets, profile-owned-path census (48/18/46), production
diagnostic inventory (581 owners), backlog task ids (3,317 files, no
duplicates, all Windows-compatible), chachanotes allowlist (105/105), index
plan pins (270/270/57).

### 4.4 The paired sweeps — scope deviation, stated

The brief called for "the full `Tests/UI` + `Tests/Library` +
`Tests/Architecture` roots (the established sweep scope)". Those three roots
are **not** this recipe's established scope and the difference is >5×:
measured here, they collect **23,898** tests where §7's actual net,
`Tests/UI -k "library"`, collects **4,477**. At ~25-30 min/side that is ~4.5
hours of paired sweeping, 15,742 of whose tests are non-Library `Tests/UI`
cases with no relationship to a wave whose entire remaining production diff
is eight comment lines — and §7 already rules the exhaustive net "CI's job,
not this recipe's per-task evidence requirement".

Resolved as **two paired sweeps** covering the brief's intent at a defensible
cost, both against an ISOLATED worktree at `e5e03846a` with its own
`uv venv` (venv verified to resolve its own tree; no stash overlay anywhere
in this task):

| | Failed | Passed | Skipped | Wall | Load at start |
|---|---|---|---|---|---|
| A — branch (`Tests/UI -k library`) | 373 (+1 error) | 4104 | — | 30:35 | 12.44 / 14.72 / 15.28 |
| A — baseline | 370 | 4103 | — | 24:13 | 6.92 / 9.03 / 13.65 |
| B — branch (full `Tests/Library` + `Tests/Architecture`) | 22 | 3652 | 5 | 3:27 | 5.61 / 7.40 / 9.37 |
| B — baseline | 22 | 3637 | 5 | 3:21 | 3.41 / 6.15 / 8.30 |

**Sweep A's halves ran under materially different load** (branch ~2× the
baseline's, 26% longer wall clock). Stated up front because it biases the raw
counts toward more branch failures — the conservative direction, but it makes
the count comparison worthless on its own, so every branch-unique name was
resolved individually.

**Both collection asymmetries reconcile exactly**, which is the first thing to
check before reading a failure diff: sweep A branch collects +4 = the 4 tests
in `test_library_prompts_characterization.py`; sweep B branch collects +15 =
the 13 tests in `test_library_prompts_wiring.py` plus the 2 parametrizations
the controller ratchet gains from the new `_BUDGETS` row.

**Sweep B: 22 failed on BOTH trees, name sets IDENTICAL** — zero
branch-unique, zero baseline-unique. 18 are the `Tests/Architecture/` set task
1 proved name-for-name at the base; 4 are in `Tests/Library/`, two already
documented and two new (added to §7).

**Sweep A: 363 shared, 10 branch-unique, 7 baseline-unique.** Not one of the
10 is a Prompts test; three of the seven BASELINE-unique names ARE Prompts
tests, i.e. the moved subsystem failed more often on the tree without the
move. All 10 resolved:

1. combined single-process re-run of all 10: **10 passed** (14.45s);
2. true isolation, one process per node, on the branch: **9 of 10 passed**;
3. true isolation, same 10, on the baseline worktree: **10 of 10 passed**.

Five of the ten are already on §7's list (wave-4 close ×2, wave-5 tasks 1/2/3);
the other five passed every re-run in both modes and are ordinary xdist noise.

### 4.5 The one that reproduced — and why it was not dismissed on its citation

`test_library_media_reader_traversal_t22207.py::test_focus_traversal_builds_
zero_bodies_for_pass_through_rows` failed its isolated branch run while
passing its isolated baseline run. It is already a documented flaky name
(§7, wave-5 task 3) and citing that would have been the easy disposition. It
is also the only result in this whole close that could have been a
regression, and a one-sample asymmetry pointing the wrong way is not
evidence of anything.

**Matched batches of 10 isolated single-node runs per tree, back to back on a
quiet machine (load ~4-6): branch 5 failed / 10, baseline 4 failed / 10.**
Indistinguishable. The failure is a wait-loop `TimeoutError` from
`test_library_media_reader_flow.py:141` ("Detail call for backing id 23 did
not start"), not a behavioural assertion, in a Media-reader test this wave's
diff does not touch.

**The process lesson is about the BASELINE half.** That name passed on the
baseline **6 times out of 6** across the first-pass isolated runs — which
reads exactly like "green on baseline, red on branch". Only a deliberate
10-run batch showed it fails 4-in-10 there too. For a timing-sensitive test a
single clean isolated run on the baseline is not a disposition; it is one
sample of a coin flip. Recorded in §7.

### 4.6 Probe — a real before/after pair, and an order-swap experiment

The isolated baseline worktree made §9's long-specified before/after pair
free, and **this is the first close in the program to actually run one** —
§16/§19/§20 each ran a single probe and compared it to wave-2's recorded
band, which cannot separate a code effect from a machine effect. The script
is byte-identical across the wave (`git diff e5e03846a HEAD --` on it is
empty).

**Round 1 (branch first) showed the branch slower on all sixteen wall-clock
measurements** — settle +35 to +74 ms, max gap +13 to +36 ms. A uniform
one-directional shift is not noise-shaped, so it was tested rather than
argued.

**Round 2 re-ran the identical pair with the tree order REVERSED**, and the
penalty moved to the baseline: the branch came back at or below it on five of
eight rows (media switch-in 493 vs 508; max gap 130 vs 173) and within a few
ms on the rest. **The penalty follows the run ORDER, not the tree** — a
first-run warm-up artifact, corroborated by its own `waitms` column (the
delta sat in IDLE-WAIT, +25-50 ms/row, while `busyms` was unchanged).

The load-independent columns are the verdict and they are exact: `recompose`
**0 on every row of all four runs**; `full-update` the identical
`2/2/2/1/1/1/1/1` on both trees and matching every close back to wave-2; and
**`nodes` identical row-for-row between trees** (119 media / 114 notes).
Round-2 branch figures sit inside wave-2's band on every row (settle 245-493
vs 264-485 — the low end FASTER than anything wave-2 recorded; max gap 42-179
vs 54-195).

The pairing also retired a loose end three closes carried on assumption: the
mount/node growth they each attributed to "ordinary Media/Notes churn on dev"
is now **measured**, because the wave-6 BASE tree already reads the higher
counts (177-179 / 119, 114 / 114).

**The probed path was NOT untouched, and that is proven mechanically rather
than asserted.** All six probed-path methods changed across the wave. Diffing
each at both ends via `ast.unparse` and applying the 43-name
flat→`_prompts_state.<field>` rewrite (mapping read from
`prompt_state_shim_attr`, not hand-listed) to the old text makes five of them
**byte-identical**: `compose_content`,
`_select_library_rail_row_after_source_admission`,
`_toggle_library_media_reader_pane`, `restore_state`,
`_persist_library_reader_preference`. The sixth, `on_screen_suspend`, carries
the one real structural change — the prompts debounce timer lifted out of the
flat-name string loop into its own explicit block, because a plain `getattr`
cannot follow a dotted path — and is not on the probe's click path at all
(the probe clicks rail modes; it never suspends the screen).

### 4.7 A sweep interruption, and why it cost nothing

The background driver running the four sequential halves was stopped mid-way
through sweep B's baseline half (killed at 82%, last write 22:52). Because
the baseline was an **isolated worktree** rather than a same-tree overlay,
nothing was corrupted and nothing had to be re-derived — the half was simply
re-run (`22 failed / 3637 passed`, 3:21) and the branch half was untouched.

This is the §3 rule this same close had just finished writing, demonstrating
itself on the very next interruption. Under the dropped carve-out — a
`git stash -u` overlay — an interruption at that moment would have left the
shared worktree in an indeterminate state with a half-finished sweep's
numbers looking perfectly plausible.

---

## 5. Concerns

1. **Sweep scope deviated from the brief's literal wording** (§4.4). The
   brief called the three full roots "the established sweep scope"; they are
   not — §7's established net is `Tests/UI -k "library"`, and the two differ
   by 23,898 vs 4,477 collected. I ran two paired sweeps instead
   (`Tests/UI -k library`, plus the FULL `Tests/Library` + `Tests/Architecture`
   roots unfiltered), which covers the widening the brief was after at ~72 min
   instead of ~4.5 h. The non-Library `Tests/UI` remainder (15,742 tests) was
   not swept. If the coordinator wants the literal reading, it is a rerun, not
   a redo — nothing else depends on it.

2. **Sweep A's two halves ran under materially different machine load**
   (branch ~2× the baseline's; 30:35 vs 24:13 wall). This biases toward
   spurious branch-unique names, which is the conservative direction, and
   every one of the 10 was resolved individually rather than by count. But
   the paired COUNTS from sweep A should not be quoted as a like-for-like
   comparison. Sweep B's halves were comparable and came back name-identical.

3. **`test_library_modal_dismissal.py` is still 1-red** and is a BLOCKED
   guard, not a failing assertion — it aborts during discovery, so it
   currently proves nothing about any subsystem's modal rows, including the
   four this wave repointed. Filed as **TASK-31815**. Recorded in §7 with the
   distinction spelled out, because five waves in a row read "1 failed / 169
   passed" as one bad row out of 170.

4. **The `_ui_ready` census still has zero headroom** and my first draft of
   **TASK-31816** offered "re-pin with stated headroom" as an option. Reading
   the guard's own docstring while writing §7 showed that ADR-097 forbids it
   (`MAX_TLDW_MODULES_AT_UI_READY` never rises; the only paths are defer, shed,
   or a ledgered owner exception). The task's ACs were corrected in the recipe
   commit. Also worth flagging for whoever picks it up: the documented wobble
   is ±1 and the observed breach is **+2**, so this is a real breach with an
   owner, not a guard needing slack.

5. **Two stale-prose defects were found by this close in files no task in the
   series had reason to open** (§1.3), which is why §3's new deleted-FIELD-name
   census is written as a standing wave-close step. The same census's first run
   returned a confident, wrong ZERO because its skip-list matched absolute
   `path.parts` and this worktree's own path contains `.worktrees`. Any future
   use of that script shape must assert a known-positive before trusting a null.

6. **Not fixed, deliberately, and named so they are not lost:**
   `Tests/UI/test_library_prompts_canvas.py:5416` ("the screen's own
   pre-check … see `_save_library_prompt`'s docstring") is architectural prose
   rather than a name-level attribution and the method still exists on the
   controller; `tldw_chatbook/UI/Library_Modules/library_skills_state.py:261-264`
   names prompts flat names as explicitly comparative prose and carries its own
   separate skills-era staleness (`_selected_skill_name` for what is now
   `LibrarySkillsState.selected_skill_name`) that belongs to whoever next
   touches the skills series.

7. **`.git-blame-ignore-revs` needed nothing from this task**, verified rather
   than assumed: all three wave-6 pure-move commits (`f59db7c94`, `d0ec95b16`,
   `6fd2b753a`) are already present and every entry in the file resolves via
   `git rev-parse --verify <h>^{commit}`. This close's own commits are
   docs-only and correctly get no entry. Wave-5 close's lesson 3 noted that the
   state-PR commits of the four series BEFORE ingest are still missing their
   entries — untouched here, still open.

8. **`progress.md` was not touched** (controller-owned), and **nothing was
   pushed**.
