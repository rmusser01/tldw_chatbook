# Wave-5 final-review fix brief — dev reconciliation merge + conditions

You are executing the fix wave for the wave-5 (ingest) final whole-branch review, which returned MERGE-READY WITH CONDITIONS. The branch itself is verified sound; every condition below is about reconciling with origin/dev, which has moved ~89 commits since the wave's base. **This is the highest-risk commit of the wave: the reviewer proved via `git merge-tree` that a naive conflict resolution ships a runtime `AttributeError` or resurrects a deleted method.** Follow the resolution spec exactly; where reality differs from the spec, stop and report BLOCKED rather than improvising.

## Environment

- Worktree: `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation` (branch `refactor/library-decomp-wave5-ingest`, HEAD `e9eca2a38`). Work ONLY here.
- Python: `.venv/bin/python` in the worktree (uv-managed, no pip; `VIRTUAL_ENV=.venv uv pip install ...` if a dep is ever missing — it shouldn't be).
- `timeout` command unavailable: use `perl -e 'alarm N; exec @ARGV' cmd args...` if you need a bound.
- UI tests: add `-p no:randomly`.
- **Do NOT push. Do NOT touch `.superpowers/sdd/.../progress.md` (controller-owned). Do NOT dispatch subagents.** Never park waiting for notifications — they do not reach you; check command output directly and keep going.

## Background you need

- The decomposition moved 56 ingest methods byte-for-byte from `tldw_chatbook/UI/Screens/library_screen.py` into `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py`, behind screen delegators. 20 ingest fields moved into `LibraryIngestState` (`library_ingest_state.py`), exposed on the screen via a programmatic shim loop.
- Screen-owned mutable state the controller reads/writes goes through **accessor bindings**: named getter/setter callables passed to the controller constructor. The worked example to mirror is `library_canvas_resync_pending_accessor` at `library_screen.py:2534-2538`.
- Dev's task-28007 (landed after our base) edited one of the moved methods and added a NEW flat screen field `_library_ingest_analyze_outcomes` (plain `__init__` attribute on the screen, NOT one of the 20 state fields) plus new handler `handle_library_ingest_clear_finished` changes, a new handler `handle_library_ingest_analyze_skipped`, and `_record_library_ingest_analyze_outcome`.
- Wiring test `Tests/Architecture/test_library_ingest_wiring.py` asserts (a) exactly 56 movers, (b) absence of pruned names incl. `_ingest_job_id_from_button` (lines ~154-163), (c) constructor arity.
- Recipe §6 (`backlog/docs/library-decomposition-recipe.md:537`): rebase/merge onto latest origin/dev immediately before each PR's final measurement; re-measure pins AFTER the merge.

## Step 1 — the merge

```
git fetch origin
git merge origin/dev   # merge commit; NEVER rebase, NEVER squash (blame-ignore references literal branch SHAs)
```

Expected conflicts: `tldw_chatbook/UI/Screens/library_screen.py` (two hunks), possibly the persistent diagnostic inventory, possibly others.

### library_screen.py — import hunk (~line 165, the `Library.library_ingest_state`/support imports block)

Take dev's side, then:
- DROP `format_ingest_progress_line` and `ingest_progress_action_signature` from the screen's imports — their sole screen consumer (`_handle_library_ingest_progress_changed`) moved to the controller, so on the screen they are dead. **Verify before dropping**: grep the post-resolution screen for each name; zero uses → drop. If dev added a NEW screen-side use, keep the import and say so in your report.
- KEEP `library_ingest_analyze_skipped_ids` (or whatever dev names its new import) — dev's new handler needs it.

### library_screen.py — method hunk (the ingest-handler region)

- `_on_ingest_job_details`: keep HEAD's delegator. Dev's side carries the full pre-move body, which calls `self._ingest_job_id_from_button` — pruned from the screen; taking dev's side reintroduces a call to a deleted name (wiring test will catch it, but resolve it right the first time).
- `handle_library_ingest_clear_finished`: keep HEAD's delegator on the screen, and **port dev's edit into the controller body** (Step 2).
- `handle_library_ingest_analyze_skipped` (dev's NEW handler): keep it on the screen as dev wrote it. The reviewer verified it calls only KEEP delegators (`_library_ingest_registry`) and screen-resident methods.
- `_record_library_ingest_analyze_outcome` (dev's NEW helper) and the new `__init__` field `_library_ingest_analyze_outcomes`: keep on the screen as dev wrote them.
- Everything else in the hunk: HEAD's shape for moved methods (delegators), dev's shape for methods that never moved. When unsure whether a method moved, check `test_library_ingest_wiring.py`'s mover list and the controller file.

### Diagnostic inventory conflict (if any)

`git checkout --theirs` on the inventory file, then READ the drift rows it names, then `python scripts/check_persistent_diagnostic_inventory.py --write` (or the invocation preflight.sh uses). Never `--write` blind.

### Any other conflict

Resolve mechanically toward dev for non-library files; report anything that needed judgment.

## Step 2 — port dev's clear-finished edit into the controller

Dev's version of `handle_library_ingest_clear_finished` gained a ~13-line block that pops `self._library_ingest_analyze_outcomes` per terminal job. The canonical body now lives in `library_ingest_controller.py`. Port that block into the controller's body **verbatim except** for how it reaches the field:

1. Add an accessor binding: constructor param `library_ingest_analyze_outcomes_accessor` on `LibraryIngestController` (arity 38 → 39), mirroring `library_canvas_resync_pending_accessor` exactly (screen passes a getter/setter pair or the same accessor shape that precedent uses — copy the precedent's shape, don't invent one).
2. In the ported block, replace `self._library_ingest_analyze_outcomes` reads/writes with the accessor, matching how `library_canvas_resync_pending_accessor` is consumed inside the controller.
3. Update every constructor site: the screen's controller construction, the `wire_bypass_ingest_controller` test helper, and the inline-consent local helper that constructs the controller (grep for `LibraryIngestController(` across `tldw_chatbook/` and `Tests/`).
4. Update the wiring test's pinned arity if it pins 38.

This is a deliberate, documented first divergence from that body's byte-for-byte original: dev edited the method after the move, so the edit follows the body. Say exactly that in the merge-commit message.

## Step 3 — post-merge census re-verification

Run the boundary-anchored 20-field census: grep the post-merge `library_screen.py` and the ingest test files for direct flat uses of each of the 20 `LibraryIngestState` field names (word-boundary anchored, `self._<name>` outside the shim loop/accessor definitions). The reviewer found zero outside the conflict hunk at 89 commits of drift; re-verify at whatever drift exists when you merge. Any new hit: retarget it through the state shim (same mechanical pattern as task 3's retargets) and list it in the report.

Also confirm dev's task-28007 tests pass unmodified (its additions in `Tests/UI/test_library_shell.py` and `Tests/UI/test_library_ingest_canvas.py`) — they exercise the edited method through the delegator, so they prove the port.

## Step 4 — re-measure both ratchet pins (in the merge commit)

- `Tests/Architecture/test_screen_size_ratchet.py` LibraryScreen row: currently `(40094, 1296)`. Dev added screen methods, so the measure RISES — that is the documented dev-merge re-measure case. Use the same comment convention prior catch-up merges used in that file's git history.
- `Tests/Architecture/test_library_modules_size_ratchet.py` ingest controller row: currently 2569; the ported block + accessor raises it. Same convention.

Run both ratchet tests green (chat_screen rows are pre-existing dev reds — ignore those two).

## Step 5 — condition/minor fixes (one follow-up commit after the merge commit)

1. **M1**: `.git-blame-ignore-revs` — the `12ba4fb13` entry says "the 3 sites this sweep still missed"; the true count is **2** (both in `Tests/UI/test_parakeet_v2_install_ui.py`). Fix the word.
2. **Blame-ignore prior-wave gap (ruled FIX-FORWARD)**: append the four missing prior-wave state-PR pure-move commit hashes — wave-2 export, wave-2 collections, wave-3 search+RAG, wave-4 skills (`87c318d57` per the reviewer — verify it). Find each via `git log --oneline` on the relevant series (the recipe's per-subsystem tables and existing blame-ignore entries for each wave's controller/cleanup commits point at the neighborhoods). **Every hash you write MUST come from `git rev-parse` output in this session — never typed from memory** (a fabricated-hash incident already happened in this program). Update the comment that currently says the gap is "not retroactively fixed here".
3. **M2**: `Tests/Architecture/test_library_ingest_wiring.py:151-153` — the prune-fraction comment compares 6-of-29 (~21%) against a range whose endpoints use total-mover denominators. Fix to the apples-to-apples figure: 6-of-56 ≈ 11% (source: task-3 report §delegator-pruning).
4. **M3**: `backlog/docs/library-decomposition-recipe.md:967` (§8 ingest row) — the parenthetical "(24-vs-27-site undercount…)" conflates the task-1 CRITICAL (2 RED tests at HEAD under no-red-ships) with the separate Important (the site undercount). Reword to match §20's correct framing: CRITICAL = the RED tests; the undercount was a distinct finding.
5. **I2 follow-up task**: file ONE backlog task: fold dev's `_library_ingest_analyze_outcomes` into `LibraryIngestState` as a 21st field (retargeting dev's screen reads) and census the two new dev-side ingest methods (`handle_library_ingest_analyze_skipped`, `_record_library_ingest_analyze_outcome`) into the decomposition tables; note the accessor added in this merge as the interim bridge. **Before filing: sweep the true max task id across origin/* AND local branches** (`git grep -ho 'task-[0-9]\+' $(git branch -a --format='%(refname:short)') -- backlog/tasks | sort -t- -k2 -n | tail`, or the ls-based equivalent across refs) — three id collisions already bit this program. Use `backlog task create` with repeated `--ac` flags (comma lists do NOT split).

## Step 6 — verification battery

All from the worktree, `.venv/bin/python -m pytest`:
1. The 6 library wiring suites (`Tests/Architecture/test_library_*_wiring.py`).
2. Both size ratchets + the recompose census guard (`Tests/UI/test_library_recompose_ratchet.py`) + support-layer surface test.
3. The 4 ingest characterization files + inline-consent tests.
4. Dev's task-28007 test additions (test_library_shell.py, test_library_ingest_canvas.py — run the files).
5. `./scripts/preflight.sh` — all green.
6. **Paired-baseline sweep** (§7 of the recipe): create an ISOLATED `git worktree` at `origin/dev` with its OWN `uv venv` (never a same-tree overlay — this exact mistake dirtied a worktree once), run the sequential xdist sweep there and on the post-merge branch, diff the failure sets. Zero branch-unique real failures required; known-flaky names from recipe §7 rerun-verified as before. Remove the throwaway worktree afterwards.

The sweep is long (~25 min/side). Run it in the foreground and wait it out in bounded chunks of other verification work; do not skip it.

## Report

Write the full report to `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/final-fix-report.md`: conflict-by-conflict resolution record (which side, what was dropped/ported, with the greps that justified each drop), the accessor wiring diff summary, new pin values with their measures, census results, hash-verification transcript for every blame-ignore entry added, the filed task id, and the full battery + sweep numbers.

Return to me ONLY: `STATUS: DONE | DONE_WITH_CONCERNS | BLOCKED`, the commit hashes you created, a one-line test summary, and concerns if any.
