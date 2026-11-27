---
id: TASK-42626
title: Run a capped sample, not the full CI, for widely imported test files
status: Done
assignee:
  - '@claude'
created_date: '2026-11-26 15:30'
updated_date: '2026-11-27 05:30'
labels:
  - ci
dependencies:
  - TASK-42521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner wants dev PRs to merge quickly after sanity checks, with the nightly and release runs as the broad gates (ADR-366). The selection still sent a PR to the full run, about 63 jobs including 42 UI shards, whenever it touched a test module imported by more than 8 test files, a shared helper under `Tests/`, or more than 20 test files of its own.

PR #3769 is the case that cost the most. Its preflight said `Shared test module (9+ importers): Tests/Chat/test_console_dispatch_recovery.py`, and its full run took 3 h 26 min, because jobs waited up to about 2 h 50 min for runners on the account pool shared with tldw_server. The one-at-a-time merge queue held every armed PR behind it for 4.5 h. A light-tier PR (#3518) went from rebase to merge in 21 minutes.

These cases should run the light tier with a capped, deterministic sample of the affected test files. Inputs that can break CI itself, the schema, and anything a `conftest.py` loads keep the full run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A changed or deleted test module imported by more than 8 test files runs the light tier with at most 8 of its importers (direct before transitive, then by path) instead of the full run
- [x] #2 A changed or deleted non-test Python file under `Tests/` samples its importing test files the same way
- [x] #3 More than 20 selected test files are cut to 20, keeping the PR's own changed test files first, instead of taking the full run
- [x] #4 The same diff and tree give the same targets whatever order the file system lists the tree in
- [x] #5 An importer or data reader the light job cannot run is left out of the sample; the PR's own changed test files of those kinds still take the full run
- [x] #6 The plan's reason says what it sampled
- [x] #7 ADR-366 records the amendment, with the cost accepted
- [x] #8 A changed or deleted path that any `conftest.py` loads, directly or through other `Tests` modules and including imports inside functions, takes the full run; no hand-kept list decides it
- [x] #9 Application code versions its schema only through `PRAGMA user_version`; a Python name `user_version` no longer sends a module broad
- [x] #10 `scripts/ci/README.md` describes the policy as it now runs
- [x] #11 Importers are resolved by module path, so a same-named module in another package or a mention in prose is not an importer
- [x] #12 Over the cap, a sample holds only dependents a PR lane runs. The full run is kept only for a laned dependent the light job cannot run when nothing laned would run in its place. With no laned dependent, the plan stays light and says so.
- [x] #13 The PR's own changed test files and the under-the-cap dependents share the 20 slots, and sampled files take at most 8 in all, handed out in turn. The reason names what was sampled, skipped or cut, including the PR's own files it left out.
- [x] #14 The light runner's wall time on the worst realistic selection under the final caps is measured against the job's 30-minute cap, alongside the hosted dev-light jobs' durations and selection sizes
- [x] #15 A helper nothing imports has the files that name it, by file name or module name, as dependents. It goes broad when a conftest-time file names it precisely. A change nothing imports or names says so.
- [x] #16 The plan does not depend on the order of the changes, every cheap reason for the full run is found before any search, and a cut or budget that leaves a blocked change with nothing laned sends the plan broad
- [x] #17 Names match as whole words, a common data file name needs its folder named, and a conftest literal must end the path. Helpers met during the search are read by name too. A blocked change's first laned dependent takes a slot ahead of the PR's own overflow.
- [x] #18 A PR never samples more than 8 files (an own-file lead does not count). A Tests/CI file reads a change only when it names it precisely. Name counts come from tracked files and include a deleted twin, and the common-name rule covers `.py` helpers without precise readers.
- [x] #19 A common name needs its folder right before it, or a folder literal in a file that names the folder above it too. One tracked-file count serves the whole plan, and the counts survive a missing git binary.
- [x] #20 A longer path before a common name's folder must go through its parent, quoted components included (on the folder's line, or the line above when the folder opens its line; a comma separates parts only in a join). The counts ignore an inherited git environment and include the PR's deletions under `Tests/` everywhere. A fixture copied by a distinctive folder name has the copying tests as readers; a conftest-time one sends the plan broad.
- [x] #21 A module in a `DB` or `db` directory goes broad only when its text is schema code (uppercase DDL, `PRAGMA user_version` or an `UPDATE db_schema_version`); every file in a `migrations` directory stays broad.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In `scripts/ci/tier_policy.py`, sort the Tests corpus and make `_importers` breadth first with depths.
2. Replace the shared-infrastructure rule with an explicit profile-harness set.
3. Sample importers per changed module and cap the total, skipping unrunnable importers.
4. Replace the four tests that pinned the broad behaviour; check each new test fails on dev's policy.
5. Append the ADR-366 amendment and update `scripts/ci/README.md`.
6. Replay #3769 and recent dev merges offline.
7. Fix round 1 (coordinator review): derive the conftest-loaded set instead of the profile-harness list; narrow `user_version` to the PRAGMA form; bring the README up to date; re-run the replays and verification.
8. Fix round 2 (independent review I1, I2, M1-M4, N1-N3): resolve imports with `ast` for both the importer search and the conftest set; count only runnable files and go broad when none can run; share the cap in turn and write the notes after the cut; kill the reviewer's mutations; measure the light runner.
9. Fix round 3 (coordinator): sample only dependents a PR lane runs, keep the full run only where it reaches what light cannot, and cap sampled files at 8 per PR; re-measure the light runner on the worst selection.
10. Fix round 4 (independent re-review I-A, M-A to M-E, nits): read unimported helpers by name; check blocked dependents after the cuts; cheap reasons first with a word index; kill the reviewer's mutations; precise conftest naming; honest timing evidence; docstrings, sorting, `Tests/CI` note and reason length.
11. Fix round 5 (re-review m1-m6, nits): whole-word names, precise helper readers, literal conftest naming, common data names, lead slots for blocked changes, name fallback during the search, corrected notes.
12. Fix round 6 (re-review CI gap, n1-n6, nits): precise-only Tests/CI readers, a capped lead budget, same-line folder rule, deleted twins, prefix-only `./` stripping, a K1 test, the common-name rule for `.py`, tracked-file counts.
13. Fix round 7 (re-review m1, m2, nits): the folder right before the name, or a folder literal with its parent folder named; tests for T4 and T8; one count source; git fails safe.
14. Fix round 8 (re-review m1, m2, nits): grandparent-only longer paths, a scrubbed git environment, the PR's deleted names in every count, a distinctive-folder fallback for fixture packages, tests for U2 and U3.
15. Fix round 9 (re-review m1, nits): deletions under `Tests/` only, quoted components before a folder, the conftest check for folder readers, folders counted at every level, tests for F3 and F4, accurate precision claims.
16. Fix round 10 (re-review nits): no synthetic `Tests/CI` literal naming a real file, a line-aware look-back for the quoted component, and a sweep table from one script at both trees.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed (`scripts/ci/tier_policy.py`):**
- **Dependents.** A changed or deleted test module or helper under `Tests/` depends on the test files that import it, directly or through other modules. A data file under `Tests/` depends on the files that read it, or that import a helper that reads it.
- **Two modes (round 3).**
  - **Under the cap:** with at most `IMPORTER_CAP` (8) runnable dependents, all of them run, laned or not, as dev's rule did.
  - **Over the cap:** the change is sampled. Only dependents a PR lane runs are eligible (whole, by node, through a lane directory, or in the UI census), up to 8, which is a subset of what the full run would run. Unlaned dependents are skipped and noted as `skipping N no PR lane runs`. The nightly runs the same lanes, so they run only at release.
  - A PR samples at most 8 files in all, handed out in turn. Its own changed test files, then the under-the-cap dependents, take the 20 slots (`LIGHT_FILE_CAP`) first. The total never exceeds 20.
- **The full run only where it reaches what light cannot (rounds 3 and 4).**
  - A change goes broad when a dependent that a PR lane (or a push or nightly job) runs is one the light job cannot run (heavy, deselected, split, push-only), and nothing laned of it runs once the slots are handed out: `Only the full run reaches the laned dependents of <path>: ...`.
  - Round 4 (M-A) runs the check after the 20-file cut and the 8-file budget.
  - Round 5 (m5): each blocked change first gets one slot for its first laned dependent, ahead even of the PR's own files. A sampled lead counts toward the 8, unless it is one of the PR's own changed files; once 8 are spent, a further blocked sampled change gets no lead and the after-the-cut check sends the plan broad, so a PR never samples more than 8 (round 6, n1). This covers the reviewer's K1 (laned dependent after unlaned ones), K2 (under-the-cap changes filling the slots) and K6 (the laned dependent is an own file past the cut). It replaces round 4's blocked-first sort. A laned-first sample sort I also tried was redundant with the lead (no test could tell, mutation z), so it is gone.
  - A `Tests/CI` dependent covers the change, and the note says `Tests/CI covers <path>`.
  - When no PR lane runs any dependent, the plan stays light and says so: `no PR lane runs a dependent of <path>; only the release run covers its N test files`.
- **Import resolution (rounds 2 and 4).**
  - `_imports(name, text)` resolves a file's imports with `ast`: `Tests.*` absolute imports, `from pkg import mod`, relative imports, bare imports of a module beside the importer, and nested imports.
  - Package `__init__.py` files count: the importer's own and every one on the way.
  - It also reads child scripts a test embeds as a string, but not docstrings (round 4). A template that does not parse whole has its `import`/`from` lines parsed one by one.
  - A bare `import mod` counts as the module beside the importer. pyproject runs pytest with `--import-mode=importlib`, so this over-approximates toward more tests; the docstring says so (round 4).
  - Paths no file has are kept, so a deletion counts. A file that does not parse counts as an importer.
  - Candidates come from the word index: only a file that has the module's name as a word can import it.
- **Helpers nothing imports (round 4, I-A).** A changed or deleted helper with no resolved importers, such as a child process run by path or with `-m`, or a file a conftest loads with `spec_from_file_location`, is read like data. `_readers` lists the files naming it, by file name or module name, as its dependents. A conftest or conftest-loaded file that names it, or a data file, precisely goes broad: `Named at conftest time by <file>: <path>`. Precisely means with its parent directory, by module name, or by file name from beside it (`_names_path`), so a common name such as a zip member `README.md` does not count (M-D). A change nothing imports or names gets the note `nothing imports or names <path>`.
- **Names (round 5, m1-m3, m6).**
  - Names match as whole words (`_word`: no letter, digit, `_` or `-` on either side), so `guard.py` is not read inside `test_focus_guard.py`, nor a folder `interop` inside `interop-synthetic-password`.
  - A `.py` helper's precise readers win over bare-name ones.
  - `_names_path` needs a quoted, directory-qualified literal that the path ends with, so `"skills/review/SKILL.md"` names only `SKILL.md` files under `skills/review/`.
  - A file whose name other tracked files under `Tests/` share (`LICENSE`, `SKILL.md`, `README.md`, the fixtures' `guard.py`) needs, when it has no precise reader, its folder right before the name (`age_v1/README.md`, `"age_v1" / "README.md"`; round 7, m1). Alternatively the folder can appear as a quoted literal of its own, such as a list of fixture folders, in a file that also names the folder above it, unless that is `Tests` or nothing (round 7). This also covers `.py` helpers (round 6, n6).
  - Round 8 (m1): a longer path before the folder must go through the grandparent, so `"skills/review/SKILL.md"` no longer reads `.../custom/review/SKILL.md` or `.../picked/review/SKILL.md`; the reviewer counted 44 such readers over 11 paths, with no tier changes.
  - Round 8 (fixture folders): when a changed helper or data file has no readers at all, the files that name its folder as a quoted literal read it, as `Tests/Plugins/test_retained_inspection.py` does with `interop_package("cursor-guard")`. This applies only to a distinctive folder (a `-` or a digit in its name, below a top-level folder, no other tracked folder of that name). Measured on the tree, across the 173 files with no readers:
    - any ancestor folder: 1,124 readers over 156 files;
    - own folder only: 551 over 137 (`"rules"` admitted 36);
    - own folder with the parent always named: 229 over 133, and it lost `cursor-guard`;
    - the distinctive form: 22 readers over 9 files (`export-golden`, `age_v1`, `cursor-guard`, `portable-openai` and the hashed upstream packages). Most are real, with a known over-match that errs toward running more tests: `Tests/Plugins/test_owned_mcp_tools.py` names `"cursor-guard"` only as a synthetic manifest name, and the `age_v1/README.md` readers use the folder's `official.age`. An earlier version of this note said "all real", which overstated it.
  - Round 9 tightens the fallback:
    - a conftest, or a file a conftest loads, that names a distinctive folder sends the plan broad (`Named at conftest time`);
    - folder names are counted at every level, so a same-named folder holding only subfolders makes the name shared;
    - tests pin the top-level rule (F3) and the `Tests/CI` exclusion (F4).
  - Round 9: a quoted component before the folder (`"skills" / "review" / "SKILL.md"`, `os.path.join("skills", "review", ...)`) must also be the grandparent. The reviewer found 4 such admissions, in `Tests/Plugins/test_coordinator.py` and `test_recovery.py`.
  - Round 10 (look-back): the quoted component is looked for on the folder's line or, when the folder opens its line, on the line above, less a trailing `#` comment. Round 9 instead searched a 200-character window. A comma separates parts only inside a `join(` or `joinpath(` call. The reviewer's probe cases now all come out as wanted:
    - `make("plugin", "review/SKILL.md")` and a list item on the next line read the file;
    - a chain with a comment in it, or with 250 spaces in it, does not.

    A match already through the grandparent (`"skills" / "custom/review/SKILL.md"`) is accepted without the check. Round 9 compared `skills` with `custom` there, which errs toward fewer tests.

    **Left as is:** a join whose arguments span lines (`os.path.join(\n "skills",\n "review", ...)`) reads like a list, so its comma is not a separator and the file counts as read. Finding the open `join(` lines above would need a bracket scanner, and this case errs toward more tests. A comment above that itself contains a quote is not stripped either; the chain then looks unbroken and the file counts as read, again toward more tests.
  - Round 10: `test_a_common_name_needs_its_folder_right_before_it` used `"Tests/README.md"`, which names the real file precisely, so the sweep said `Tests/CI covers Tests/README.md`. The pair is now `Tests/GUIDE.md` and `Tests/fixtures/sp/GUIDE.md`. The literal scan, with no prefix excluded, and the reviewer's `ci_tiered.py` agree: `Tests/CI/test_tiered_ci.py` names 16 real files. The 10 non-test ones are conftest files, conftest-loaded or conftest-named, or `Tests/Chat/hooks_v2_owned_rig.py`. The 6 test files are named on purpose, and the name fallback never runs for a test file.
  - Round 9's sweep then left `Tests/Plugins/fixtures/interop/cursor/picked/review/SKILL.md` read only by `Tests/CI/test_tiered_ci.py`, through a synthetic literal (`"picked/review/SKILL.md"`) that the real path ends with. Four more synthetic literals named real fixtures the same way (`custom/review/SKILL.md`, `skills/review/SKILL.md`, `.codex-plugin/plugin.json`, `codex/.codex-plugin/plugin.json`). Those tests now use an `audit` folder and an `acme` package, which no tracked `Tests/` path has, so `Tests/CI` no longer claims those fixtures.
  - Round 6 had accepted the folder anywhere on the line, or a bare folder literal. That admitted 98 readers through generic folders such as `review`, `package` or `Tests`, and counted `Tests/fixtures/superpowers_skills/README.md` as a reader of `Tests/README.md`. With round 7's rule, the reviewer's probe admits 428 readers. None of the 8 `superpowers_skills` fixtures loses `Tests/Skills/test_skills_import.py`: the 7 with common names keep it through the folder list plus the `superpowers_skills` parent, and `requesting-code-review/code-reviewer.md` is not a common name.
  - A deleted twin still counts as sharing (n3).
  - The counts come from `git ls-files`, taken once per plan, and the search uses the same counts (round 7, one source), so local junk does not change them.
  - The git call drops any inherited `GIT_*` environment, because a hook's `GIT_DIR` pointed it at another repository's index (round 8, m2).
  - The PR's deletions under `Tests/` join the counts, so a helper met in the search whose twin the PR deletes stays common (round 8). A deletion elsewhere, such as `Docs/old/golden_z.json`, no longer makes a `Tests/` name common (round 9, m1).
  - They fall back to the files on disk in three cases: outside a git checkout; when git lists no `Tests` files, such as a root inside another repository with `Tests` untracked; and with no git binary (`OSError`).
  - Output is decoded with `errors="replace"`.
  - A `Tests/CI` file reads a change only when it names it precisely, because its synthetic fixtures reuse real names (round 6, CI gap).
  - A conftest literal drops a leading `./` or `../` as a prefix, keeping a dot-directory such as `.codex-plugin/` (n4).
  - A helper met during the search that nothing imports is read by name too: `Tests/fixtures/longtail/CAPTURE.md` now runs `test_capture_local_gguf_guard.py`, which loads `capture_local.py` with `spec_from_file_location`.
  - With no test dependents at all, the note is `no test file depends on <path>`.
- **The search.** `_importers` runs breadth first in sorted corpus order and follows every dependent. It stops once it has IMPORTER_CAP + 1 files that count: runnable files a PR lane runs. The depth sort is gone, because BFS order already puts direct importers first. Candidates come from a word index of the Tests tree (`_token_index`), built once per plan, instead of a substring scan per step (round 4, M-B).
- **Order and cost (round 4, M-B).** The changes are sorted by path. Every cheap reason for the full run is checked across all changes before any search: `_full_reason`, conftest-loaded, the PR's own lane reasons and schema code.
- **The conftest-loaded set.** `_conftest_loaded` follows every `Tests/**/conftest.py`'s imports through the same resolver. A changed or deleted member goes broad (`Loaded by conftest.py: <path>`), and so does a data file a member reads. `PROFILE_HARNESS` is gone.
- **Schema code.** `SCHEMA_TEXT` matches `PRAGMA user_version`, not any `user_version`. All 54 occurrences in `tldw_chatbook/` (reads, writes and docstrings) are written that way. On dev the narrowing drops exactly one file, `Chat/console_dispatch_repository.py`, which has no DDL.
- **The reason.** Notes are written after the cut. Each sampled, cut or skipped module or data file gets `sampled <ran> of <found>[+] importers|readers of <path>[ a PR lane runs][, skipping N no PR lane runs][, skipping N the light job cannot run]`. When a cap applied, the reason also says `8 sampled files in all`, or `cut to 20 test files[, not running the PR's own <files>]`. Long lists are shortened: five of the PR's own dropped files, three blocked files, ten notes, then `... and N more` (round 4).

**Other files:**
- **`Tests/CI/test_tiered_ci.py`.**
  - Earlier rounds:
    - the cap and determinism, and own files first;
    - unrunnable importers skipped, and a helper sampling its importers;
    - the conftest-loaded set (synthetic and real), the PRAGMA narrowing, and `conftest.py` still broad;
    - same-named modules, prose and every import form, and child scripts;
    - unrunnable and helper importers taking no slot, a `Tests/CI` importer not counted, and breadth first;
    - data readers and reading helpers, and the cap shared in turn with notes after the cut.
  - Round 3 gives the sampling-mode trees a PR lane (`_laned`), and adds:
    - a sample holding only laned dependents, with the search going past 20 unlaned ones;
    - broad only when the full run reaches what light cannot, in both modes;
    - light with the "no PR lane" note when nothing laned depends;
    - the 8-file sample budget, with an under-the-cap module outside it;
    - a `Tests/CI` importer, with `Tests/CI` laned, not counted toward the nine.
  - Round 4 adds:
    - helpers run by path or `-m` read by the files naming them;
    - a spec-loaded conftest helper going broad (synthetic, and the real `Tests/Performance/boot_budget_ratchet.py`);
    - a common name in a conftest not sending a fixture broad, while a directory-qualified one does;
    - data read by a conftest-loaded helper going broad (N6);
    - a package `__init__.py` having the package's own files as dependents (N7, N8);
    - 8 runnable plus 1 unrunnable staying under the cap (N10);
    - the budget never passing 20 (N16);
    - a cut that removes a blocked change's laned sample going broad, with blocked changes picking first;
    - input order not mattering;
    - cheap reasons found before any search;
    - docstrings not counting as child scripts;
    - long reasons shortened.
- **ADR-366.**
  - The Status line says "amended by TASK-42521 and TASK-42626".
  - The amendment names the three 2026-11-23 bullets it replaces and says what is counted.
  - It covers the two modes, and the budget with the 1,131 / 4,707 lane counts. For timing it cites the local 20 min 14 s re-measure (the 37-minute reading is void) and the hosted dev-light jobs.
  - It also covers when the full run is kept (after the cuts), unimported helpers, the precise conftest naming rule, the import-mode rationale, the PRAGMA narrowing and the reason format.
- **`scripts/ci/README.md`:** active-policy header, event table, the selection section as the code runs it, and a note on the rollout section.

**Evidence (2026-11-26, branch `ci/42626-tier-sample` from dev 52351c4e0b):**
- **Fail on old:**
  - The 7 first-round test cases fail on dev's `tier_policy.py`.
  - The 9 round-1 cases fail on f8b476384b.
  - 7 round-2 tests, and the 2 whose reason wording changed, fail on 5c83750feb.
  - The 3 new round-3 tests, and the 6 updated for lanes or reason wording, fail on 280dc7b24f.
  - 2 round-10 tests fail on 986eb4cb24.
  - 3 round-9 tests fail on 14f4af2b54.
  - 4 round-8 tests fail on ea45684b72.
  - 3 round-7 tests fail on 646b345fec. The T4 and T8 tests pin existing behaviour.
  - 8 round-6 tests fail on 0315213e7c. A ninth (K1) pins the lead step to kill S9.
  - 6 round-5 tests fail on ff7ec61bc0. Three more pin existing behaviour, to kill R4, R5 and R10.
  - 10 round-4 tests fail on 58421663. Three more pass on it by design: they pin behaviour that already existed, to kill the reviewer's mutations N7/N8, N10 and N16.
  - The determinism test also fails when only the corpus `sorted(` is removed.
- **Mutations: all 71 killed on the final policy, in one run, each restored from a copy with a byte check.**
  - a: no runnable filter.
  - b: cap before the runnable filter.
  - b2: the search counts every file.
  - c: no `Tests/CI` exclusion.
  - d: DFS.
  - f: name-only matching.
  - g: never broad for blocked dependents.
  - h: notes before the cut.
  - i: no child-script imports.
  - j: sample unlaned files too.
  - k: no 8-file budget.
  - l: the search stops on runnable rather than laned files.
  - m: laned-only under the cap too.
  - The reviewers' N6, N7, N8, N10 and N16, and R4 (name fallback for test files), R5 (no `Tests/CI` exemption after the cut) and R10 (blocked list not shortened).
  - n: no name fallback for helpers.
  - o: conftest check by file name only.
  - p: blocked check counts the sample.
  - q: no lead slot for blocked changes.
  - r: changes not sorted.
  - t: docstrings counted as child scripts.
  - u: reason not shortened.
  - v: names matched inside longer names.
  - w: no precise-reader preference.
  - x: conftest literal matched as parent/name.
  - y: no common-name rule.
  - aa: no name fallback during the search.
  - bb: no "no test file depends" note.
  - cc: sampled leads not counted toward the 8.
  - ee: a hyphen does not end a word.
  - Round 6: S9 (lead = the first sample file, not the first laned one), CI (Tests/CI bare-name readers count), n1a (leads not capped at 8), n1b (an own-file lead counts toward the 8), n2 (folder anywhere in the file), n3 (a deleted twin not counted), n4 (`lstrip` instead of a prefix), n6 (no common-name rule for `.py` roots), the counts taken from disk rather than tracked files, and ff (no folder-literal form for common names).
  - Round 7: m1a (folder anywhere on the line), m1b (no parent folder for a folder literal), T4 (no common-name rule for helpers met in the search), T8 (no disk fallback when git lists nothing), and git (`OSError` not caught).
  - Round 8: U2 (the plan's counts not passed to the first search), U3 (no comma or paren between folder and name), m1c and m1d (longer paths), m2 (inherited git environment), n3 (the PR's deleted names not counted), and fa, fb, fc (the folder fallback, its distinctive-name rule, and its shared-name rule).
  - Round 9:
    - m1o: deletions outside `Tests/` count.
    - comp: any quoted component before the folder.
    - cf: folder readers skip the conftest check.
    - F3: top-level folders count.
    - F4: `Tests/CI` folder readers count.
    - anc: only folders that hold files count.
    - m1a, retargeted to `named_beside`.
  - Round 10:
    - Qc: a comma always separates parts.
    - Qn: a comma never does.
    - Qa: no look at the line above.
    - Qs: the line above even when the folder does not open its line.
    - Qh: the comment above kept.
    - Qw: a 200-character window on the line.
    - Qt: no short-circuit through the grandparent.
    - comp, retargeted to `_component_before`.

    A first Qw (capping the line start at 200 characters back) survived as an equivalent mutant: the line-above step then reads back to the previous newline, which restores the same text. Qw now caps the looked-at text itself, and it is killed.
  - From round 9 on, mutation runs restore the policy from a byte copy (never `git checkout`) and check the restore byte for byte.
  - x and U3 survived the first round-8 run. The grandparent rule had made x's old case unreachable, and the comma-separated reader also passed through the folder-literal clause. A uniquely named data file written by the conftest under another folder now kills x; dropping the reader's comment leaves U3 only the comma.
  - ee survived again in round 7, because the adjacency rule already rejects the near-miss line. The generic folder list now names its parent folder only inside `fixtures-old`, which a hyphen must end.
  - ee survived the first round-6 run, because the same-line rule already rejected the near-miss file. Its folder-like word now shares a line with the name.
  - For e, the depth sort was deleted; z (a laned-first sample sort) survived because the lead slot made it redundant, so the sort was removed.
- **Precision and recall,** checked on the round-2 resolver against the reviewer's ground-truth regex (extended with the `from Tests.pkg import mod` form) over all 6,135 Tests files:
  - Every file parses.
  - **Recall:** 0 import lines the regex confirms are missed.
  - **Precision:** 15 of the 7,863 resolver edges are not confirmed by the line-anchored regex. Spot checks show they are real child-script imports: quoted `"from Tests.x import y\n"` fragments and `;`-joined statements.
- **All 235 helpers on the final policy:**
  - 94 run all their dependents;
  - 67 are sampled;
  - 19 stay light with the "no PR lane" note;
  - 18 stay light with "nothing imports or names it";
  - 32 go broad as conftest-loaded;
  - 1 goes broad as named at conftest time (`Tests/Performance/boot_budget_ratchet.py`);
  - 4 go broad because only the full run reaches their laned dependents.

  None is light, silent and empty. This corrects round 3's "123 run all their dependents": that count included helpers with no dependents at all. 18 have none and now say so; others gained the files that name them as dependents.
- **The reviewer's sweep of 1,217 real single-file changes, re-run (round 4).** 0 end light with nothing related and no note (55 before). Of the reviewer's 55: 28 now carry a note, 26 run related files, and 1 goes broad.
- **The reviewer's round-5 sweep (`sweep4.py`: those 1,217 plus the 280 data files), before (ff7ec61) and after:**

| Outcome | Before (ff7ec61) | After |
|---|---:|---:|
| broad: Loaded by conftest.py | 34 | 34 |
| broad: Named at conftest time | 12 | 10 |
| broad: Only the full run reaches | 16 | 15 |
| broad: Test file | 31 | 31 |
| light: dependents run | 865 | 876 |
| light: no PR lane runs a dependent | 46 | 43 |
| light: no test file depends on it | 0 | 3 |
| light: nothing imports or names it | 152 | 164 |
| light: sampled (laned only) | 341 | 321 |
| total | 1497 | 1497 |

  The transitions:
  - 13 light: sampled (laned only) -> light: dependents run
  - 8 light: sampled (laned only) -> light: nothing imports or names it
  - 3 light: no PR lane runs a dependent -> light: no test file depends on it
  - 2 broad: Named at conftest time -> light: sampled (laned only)
  - 2 broad: Only the full run reaches -> light: nothing imports or names it
  - 2 light: dependents run -> light: nothing imports or names it
  - 1 light: dependents run -> light: no PR lane runs a dependent
  - 1 light: sampled (laned only) -> broad: Only the full run reaches
  - 1 light: no PR lane runs a dependent -> light: dependents run

  The one new broad in that round-5 run was `Tests/Backup_Recovery/fixtures/age_v1/README.md`, a false positive. An earlier version of this note called its reader real. `Tests/Packaging/test_backup_helper_distribution.py` names `.../age_v1/official.age` on one line and an unrelated `"README.md"` on another. Round 6 (n2) requires the folder and the name on the same line, so it is no longer read.
- **Sweep (the reviewer's `sweep9.py`: 1,217 single-file changes plus 280 data files), run at 14f4af2b54 and at the final tree with one script:**

| Outcome | Before (14f4af2b54) | After |
|---|---:|---:|
| broad: Loaded by conftest.py | 34 | 34 |
| broad: Named at conftest time | 10 | 10 |
| broad: Only the full run reaches | 14 | 14 |
| broad: Test file | 31 | 31 |
| light: Tests/CI covers it | 1 | 0 |
| light: dependents run | 874 | 874 |
| light: no PR lane runs a dependent | 43 | 43 |
| light: no test file depends on it | 2 | 2 |
| light: nothing imports or names it | 164 | 166 |
| light: sampled (laned only) | 324 | 323 |
| total | 1497 | 1497 |

  Round 9's table was wrong. Its "before" column came from another run, and its script counted 7 under-the-cap plans as sampled. Their reasons say, for example, `sampled 6 of 6 importers of ..., skipping 1 the light job cannot run`: every runnable dependent runs. Both columns here come from one script and one categorizer, in which "sampled" means the reason says `a PR lane runs`, as in round 8. The 14f4af2b54 column reproduces round 8's "after".

  Changes since 14f4af2b54, none of them a tier:
  - `Tests/Plugins/fixtures/interop/cursor/picked/review/SKILL.md`: sampled -> "nothing imports or names". Its old readers, `Tests/Plugins/test_coordinator.py` and `test_recovery.py`, write the native package's `"skills" / "review" / "SKILL.md"`. A real reader, `Tests/Plugins/test_cursor_adapter.py` (`interop_package(dialect)` with `"picked"`), is found at neither head; the nightly covers it.
  - `Tests/README.md`: "Tests/CI covers it" -> "nothing imports or names". Its only reader was the synthetic `"Tests/README.md"` in `Tests/CI/test_tiered_ci.py` (round 10).
  - `codex/custom/review/SKILL.md` keeps its sampled files and skips 8 fewer unlaned dependents: the same two files and their importers.
  - Round 9 dropped `Tests/Plugins/test_source_service.py` as a reader of five `.cursor-plugin/plugin.json` fixtures, and round 10 restores it. Its `("cursor", ".cursor-plugin/plugin.json")` is a tuple, and a comma now separates path parts only inside a join. That test writes its own manifest, so this is an over-match toward more tests.
  - The 9 folder-fallback rows are unchanged, so 22 readers over 9 files stand.
- **Cost.** A one-file plan takes about 2-3 s cold, most of it the one-off word index (2.2 s). The worst cold cases, the `Tests/Plugins/fixtures/**/guard.py` fixtures, take about 9-11 s; round 4's notes said 3.8 s, a figure measured with the index already built.
- **Several changes at once (round 4, M-B),** cold, reviewer's `multi3`:
  - 30, 100 and 400 of the sweep's light rows took 7.1, 8.4 and 12.5 s, against 20, 55 and 134 s before. They end broad by the after-the-cut check, which runs after every search.
  - Light-ending sets of 100 and 400 took 3.7 and 8.3 s, with a reason of about 1.1 KB.
- **Conftest naming (round 4, M-D).** Of the 280 tracked data files under `Tests/` (guard data excluded), conftest-time files named 11 precisely in round 4, against 51 by file name only. All 11 were `SKILL.md` files; an earlier version of this note said "10 ... and one other", which was wrong. Round 5 (m2) requires the literal to end the path. 9 remain, all `SKILL.md` files under a `skills/review/` directory, which `Tests/Plugins/conftest.py`'s literal really ends; they stay broad as a conservative over-match.
- **Hosted dev-light timing (round 4, M-E).** 33 successful "Dev smoke and affected contracts" jobs on PRs from 2026-11-24 to 2026-11-26 took 4.4 to 25.0 min (median 8.1). The largest selection was 12 files (90 targets, run 45435066859) in 9.9 min. The longest job was 25.0 min for 8 files (34 targets, run 45462985645). No hosted run has selected 20 files yet.
- **Verification:**
  - `Tests/CI/test_tiered_ci.py`: 155 passed.
  - `Tests/CI -n 4`: 2072 passed, 2 skipped.
  - `./scripts/preflight.sh` under a scratch HOME with `PYTHON=.venv/bin/python` (3.12.11): all derived-artifact checks passed.
  - These all pass: `ruff check tldw_chatbook --select F`; `ruff check scripts/ci Tests/CI/test_tiered_ci.py --select F`; `ruff format --check`; `Tests/lint/ruff_gate.py`.
- **PR #3769** (base 434eb71f95, head 7a453d4d36, root = a detached worktree of the head) still runs **broad**, for one reason: `Schema input: tldw_chatbook/DB/assistant_generation_state.py`. With that file dropped, it runs light: 22 targets, 18 files. The reason is `8 sampled files in all; sampled 4 of 9+ importers of Tests/Chat/test_console_dispatch_recovery.py a PR lane runs, skipping 32+ no PR lane runs; sampled 4 of 9+ importers of Tests/UI/test_console_prompt_queue.py a PR lane runs, skipping 18+ no PR lane runs, skipping 1 the light job cannot run`. The UI `test_console_dispatch_recovery.py` runs both of its importers, under the cap.
- **Last 60 first-parent dev merges** (through 5dcba48401), replayed against one tree (the #3769 head, so approximate): 26 run light under dev's policy and 35 under the final one; rounds 9 and 10 match round 8 row for row. Round 4 had 34: 9fa2bf1b1d's own files overflow the 20-file cut, which removed the laned sample of `Tests/UI/test_console_hook_review_send_freeze.py`, so round 4's after-the-cut check sent it broad. Round 5's lead slot keeps that change's laned dependent ahead of the PR's own overflow, so it is light again, with the dropped own files named. The 25 broad runs: 21 change CI, build or test configuration, 2 their own HEAVY files, 1 its own file split across lane steps, and 1 its own deselected file.
- **Light runner, re-measured (round 3).** `scripts/ci/run_light.py` ran as the dev-light job runs it: `CI_TEST_TARGETS=<json> python scripts/ci/run_light.py`, Python 3.12.11, a scratch HOME/XDG/TMPDIR, on this 18-core Mac. The selection was the largest light plan of the 60 merges under the final caps: merge 1b9e3f8a55, 68 targets, 25 files, 26 pytest processes with `Tests/CI`.
  - None of my jobs ran alongside it. Another session's pytest runs were on the machine (load average 6.1 before, 11.5 after).
  - It took **20 min 14 s wall** (1,214 s), inside the job's 30-minute cap.
  - `Tests/CI` took 3 min 1 s, 15 % of the total. The 25 selection processes took 16 min 17 s, and process start-up the remaining ~56 s.
  - Round 2's 37 min 19 s for the same merge had my mutation, recall and probe jobs running alongside (`Tests/CI` alone then took 10 min 21 s), so that figure is void.
  - Round 4 does not change the caps, and that merge's plan is still the largest of the 60.

**Observation (no fix here).** These four unlaned UI files fail on macOS on dev's code (Python 3.12.11, isolated profile), each run alone:
- `Tests/UI/test_console_rewind_restore.py`: 42 failed, 41 of them `RecoveryRequired: raw_source_selection_changed`;
- `Tests/UI/test_console_custom_capture_worker_feedback.py`: 3 failed, event-wait timeouts;
- `Tests/UI/test_console_workflow_controls.py`: 2 failed;
- `Tests/UI/test_console_workflow_navigation.py`: 1-2 failed.

No PR lane runs them, which is the round-3 reason samples avoid unlaned files.

**Follow-up.** If a hosted light job times out, lower `LIGHT_FILE_CAP`, or run `Tests/CI` with xdist in dev-light. The longest hosted dev-light job so far took 25.0 min for 8 files.

**Trade-offs:**
- The `+` in `of 9+` means the search stopped at the cap; the true count is unknown, and so is the matching unlaned count.
- A file inside a package counts as loading its `__init__.py`, as pytest imports it, so an `__init__.py` change has the package's own tests as dependents.
- Bare imports resolve only to a module beside the importer, as the review asked.
- An over-the-cap change whose laned dependents are all past the 8-file budget runs none of them (`sampled 0 of ...`). The nightly and release runs cover them.

**AC #21 (owner decision 2026-11-27): schema inputs narrowed to schema text.** `SCHEMA_DIRS` is now `{"migrations"}`. A `DB`/`db` module goes broad through `SCHEMA_TEXT` like any application module, which now also matches `UPDATE db_schema_version` (the .sql migration runner's version stamp). #3769's file list now plans light (22 files; it had gone broad on `DB/assistant_generation_state.py`). #3686 stays broad for its own reason: it changes `Tests/Backup_Recovery/test_activation_startup.py`, which only push and nightly jobs run. A replay of dev's last 60 merges is unchanged (33 light, 26 broad, 1 replay error from a file missing in this tree): none of them changed a `DB` module without schema text. Tests: `test_a_db_module_is_schema_only_when_its_text_says_so`; two table rows that named non-existent DB paths now name real migration files.
<!-- SECTION:NOTES:END -->
