# SDD ledger — plan: Docs/superpowers/plans/2026-09-03-library-decomposition-wave3-search-rag.md

Spec: Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md (on dev; binding). Recipe: backlog/docs/library-decomposition-recipe.md (mechanics; carries all foundation+wave-2 lessons).
Branch: refactor/library-decomp-wave3-search-rag (cut from origin/dev @ 155ea1564 — the commit that merged wave 2; entire prior stack is mainline).
Worktree: .worktrees/library-decomp-foundation (reused; own venv verified in prior waves — re-verify at first test run).
Wave-2 census (task-31203 basis): search 14 methods / RAG ~39; entanglement 57.1%; tracked at .superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/task-8-report.md.

## Pre-flight conflict scan

| Pair / task | Check | Finding |
|---|---|---|
| T1 vs T2-4 | Governance guard lands before new controllers are born | Consistent — ordering is the point; governance must handle move-inflation (baselining at move landings) or the T2-4 moves trip it |
| T2-4 internal | Single vs split controller decided by ownership analysis, not assumed | Escape to BLOCKED defined in plan |
| @work methods | RAG worker paths cannot move (export-series lesson) | Plan pre-flags enumeration; screen-resident + named callables |
| Plan vs rubric | Same plan-mandated duplication (shims/delegators) as prior waves | Carried in reviewer constraints |
| Backlog ids | 31204/31205 filed on wave-2; dev max now ≥31205 | T1 and any filings sweep true max first |

Standing rulings inherited (all in force): stacked-branch discipline (though this wave is directly off dev); worktree-venv; sequential paired-baseline sweeps; per-move pin lowering; monkeypatch/module-globals routing; census-before-prune; rev-parse-only hashes; verbatim comments by copy-paste; RED commit = screen untouched + pins red at parent; never-park-silently in every dispatch; dev-race convergence loop for the eventual PR.

## Task log

Task 1: implementer DONE (controller-file size governance, task-31203 AC#4) — chose
option (a) exact per-file `_BUDGETS` rows, new guard
`Tests/Architecture/test_library_modules_size_ratchet.py`, discovered by glob
(`UI/Library_Modules/*_controller.py`) so an unlisted controller fails loudly instead
of landing ungoverned (property `test_screen_size_ratchet.py`'s hand-kept dict lacks);
12 controllers pinned at exact measured line counts (699-2,023 lines, 11,130 total);
method count deliberately NOT tracked (justified: no reliable dominant-class-per-file
convention, e.g. `LibrarySkillImportCoordinator` in a `_controller.py` file, and
summing all classes would punish the canon's own helper-class pattern). Mutation-
tested all 4 directions (growth trip, anti-slack trip, unlisted-existing-file,
unlisted-new-file) plus the tolerance boundary (50 passes/51 fails); each reverted
cleanly, confirmed via empty `git diff`. Battery: new guard 25/25 passed; guard +
screen ratchet 28 passed/2 failed (both pre-existing chat_screen reds); Library
recompose census 6/6 passed; preflight clean (5/5 checks); full `Tests/Architecture`
527 passed/15 failed/1 skipped, all 15 confirmed pre-existing via `git stash`
re-run against the unmodified base. Recipe updated with new §17 (decision, reasoning,
re-pin-at-move flow, measured rows). task-31203 AC#4 checked; task stays To Do
(AC#1-3 out of scope). Full report: task-1-report.md.

Task 1: review Needs Fixes (1 Important, 1 Minor) — the `--check-ac`/`--notes`
backlog CLI call silently stripped task-31203's `## Renumbering provenance` section
(the documented lessons-backlog-hygiene.md trap: CLI strips free-form sections /
`--notes` replaces rather than appends; the diff-after-notes mitigation was skipped).
Fix round 1: restored the section verbatim from base commit `2a90fa74c`, diffed the
result against that base to confirm only `updated_date`/AC#4-tick/new Implementation
Notes differ; re-ran and captured fresh command output for the round-4 anti-slack
mutation (both the 51-over fail and 50-over boundary pass) per the Minor. No new
lesson needed (already documented). Commit:
`fix(backlog): restore task-31203 renumbering provenance stripped by the notes edit`.
Task 1: implementer DONE (commit f5e675354; option (a), 12 controllers pinned 699-2023 lines, glob self-defending, 4-direction mutation); reviewer dispatching (sonnet)
Task 1: note: dev-drift Architecture reds now at 15 (2 chat_screen + 12 Console/timer/worker/diagnostic) — baseline-health signal for repo owner at wave close
Task 1: review Needs Fixes — mechanism verified exemplary (all mutations reproduced); 1 Important: backlog --notes edit stripped task-31203 Renumbering provenance (documented CLI trap, lessons-backlog-hygiene) — fix round 1/5 dispatched
Task 1: fix round 1/5 (2 addressed — provenance restored verbatim, mutation-4 captured; commit fa07400a1)
Task 1: complete (commits 2a90fa74c..fa07400a1, review clean after 1 fix round — 12 controllers pinned, self-defending glob)
Task 2: dispatched (implementer sonnet), BASE=fa07400a1 — combined search+RAG state PR (series 1/3)
Task 2: implementer DONE post-nudge (commits 315cd4c3c RED + 77750c85d GREEN; 20 fields -> combined LibraryRagSearchState; 3 @work stay; 0 new pins claimed 14/14 covered; ratchet 43923/1316); reviewer dispatching (sonnet)
Task 2: forward note for Task 3: _patch_sibling_library_search_input + _refresh_search_rag_panel_state_widgets are instance-monkeypatched in tests — exclusion candidates
Task 2: review Needs Fixes — mechanics all verified; 1 Important: state-shape evidence overclaims (16/20 fields not 20/20; "continuation" nonexistent; 3 answer fields unread in the cited method) — fix round 1/5 dispatched (evidence correction, verified-before-written)
Task 2: process note: the reviewer briefly forked a subagent against the no-subagents rule, self-caught, treated its output as corroboration only — findings verified by the reviewer's own reads; noted, no action
Task 2: minor (deferred): report line citation drift (13886 vs actual 13802) — folded into the fix
Task 2: fix round 1/5 adjudicated (independent count 14/20 CONFIRMS the fix; all 6 consumer traces verified file:line; one-object conclusion stands on corrected evidence; commit 8efc79655)
Task 2: complete (commits fa07400a1..8efc79655, review clean after 1 fix round)
NOTE: wave-3 SDD dir currently tracks only progress.md — force-add reports/diffs at wave close per the wave-2 durable-record mandate
Task 3: dispatched (implementer sonnet), BASE=8efc79655 — combined search+RAG controller move (series 2/3)
Task 3: implementer DONE (commits b61d55987/877eeaf9a/750df2c8c/f8974b9cb; 42/50 moved single controller; 8 excluded incl. battery-caught module-globals name; screen 43923->43009; controller born-governed 1857); reviewer dispatching (opus)
Task 3: review Approved with 4 Important (2 false caller-claims in permanent docstrings; self-contradictory ratchet comment 43-vs-42; shipped-red path-census test) + 3 minors — fix round 1/5 dispatching (FRESH fixer; original implementer was killed post-completion)
Task 3: Ruling: the deterministically-red test (test_library_screen_call_sites_never_pass_scope_kwarg) gets its census retargeted IN THE FIX ROUND — a test edit outside the cleanup PR, sanctioned narrowly because no commit boundary may ship red (same-commit philosophy as the pin rule); the invariant is intact at the new location, so this is a pure path retarget with assertions preserved. — Cost if wrong: slightly blurs the cleanup-only-edits-tests line; the recipe gets a sentence recording the no-red-ships precedence.
Task 3: fix round 1/5 (6.5/7 addressed + pin conditions a-c MET; commits 544664510 + 801c5375e)
Task 3: Ruling on the finding-2 residual: the method-docstring false claim at library_rag_search_controller.py:1476 is BYTE-FOR-BYTE ORIGINAL text (screen :43006 at base) — leaving it unfixed in the fix round is canon-CORRECT, not a miss; the module docstring (new code) was the right correction surface. The moved-body docstring update goes to Task 4 cleanup (wave-2 _apply_library_row_toggle precedent). — Cost if wrong: a reader hits the stale claim for one more task; module docstring 20 lines above already corrects it.
Task 3: complete (commits 8efc79655..801c5375e, review clean after 1 fix round; residual ruled cleanup-scope)
Task 4: dispatched (implementer sonnet), BASE=801c5375e — search+RAG cleanup (series 3/3)
Task 4: implementer DONE (commits 5bea63cdc + ab8cda560; shim deleted, 35 field refs retargeted, 12/42 delegators pruned, 5 dead imports, ruled docstring fix landed; screen 43009/1316->42949/1304, controller 1890->1895); near-miss caught by own sweep: first-draft retarget of canvas_sync.py's _sync_library_canvas (self._library_rag_answer_render_key -> self._rag_search_state.answer_render_key) broke test_media_choice_and_rag_toggles_are_canvas_scoped because that branch's only callers forward the CONTROLLER as "screen" (no _rag_search_state attribute by design) — reverted, canvas_sync.py needed no change; documented as a new recipe lesson (§18). Full sequential xdist paired-baseline sweep: 350f/3931p (branch) vs 349f/3932p (baseline), 5 branch-unique — 3 xdist noise, 2 confirmed genuinely pre-existing via a second git stash -u (added to §7's documented list). Zero real regressions. task-31203 all 4 ACs met, status Done.
Task 4: complete (commits 801c5375e..ab8cda560)
Task 4: implementer DONE (commits 5bea63cdc + ab8cda560; shims out, 12/42 pruned, pins 42949/1304 + 1895; self-caught canvas_sync near-regression reverted -> recipe §18); reviewer dispatching (opus)
Task 4: review Needs Fixes — near-regression mechanism CLEARED (independent baseline); 1 Important (9/14 dead imports missed at :487-497) + 2 minors — fix round 1/5 dispatched (fresh fixer; original transcript gone)
Task 4: fix round 1/5 (3 addressed — 9 imports pruned with live neighbour preserved, counts corrected 66/11 in both records, canvas_sync guard comment; pin 42940/1304; commit a150fc766)
Task 4: complete (commits 801c5375e..a150fc766, review clean after 1 fix round)
SEARCH+RAG SERIES COMPLETE (Tasks 2-4). Task 5: dispatched (implementer sonnet), BASE=a150fc766 — wave close
Task 5: recipe/task-31203 close-out largely pre-landed by Task 4's own commits (§18 pin trajectory, §8 subsystem table, task-31203 ACs 1-3 + Done status all already present at BASE) — verified accurate against git log/ratchet comments (all match) rather than rewritten. Found and fixed 3 residual staleness spots: (1) task-31203's own Implementation Notes still quoted the PRE-fix-round numbers (42949/1304 "final", 35/9 retarget count, 5 dead imports) — corrected to the fix-round-1 final values (42940/1304, 66/11, 14 total) via direct file edit (not backlog --notes, per lessons-backlog-hygiene), diffed clean; (2) library_rag_search_state.py's module docstring still described the shim deletion in future tense ("A future controller PR... will delete") naming the wrong PR type (controller, not cleanup) — rewritten past-tense matching the export/collections template; (3) library_rag_search_controller.py's module docstring claimed `LibraryScreen` "carries" (present tense) the two-prefix shim, stale since Task 4 deleted it — fixed to past tense "installed... (deleted at cleanup, task 4)", matching export_controller.py's correct precedent (also found the SAME present-tense staleness pre-existing, unfixed, in library_conversations_controller.py:1712 from an earlier wave — out of scope for this close, left alone, not touched). Added 4 new lessons to recipe §18 (state-shape evidence-accuracy generalization, moved-docstring-correction-is-cleanup-only canon, no-red-ships reconfirmed, born-governed same-commit mechanism confirmed live) — the canvas_sync trap lesson was already present.
Task 5: fresh measurements confirmed — test_screen_size_ratchet.py/test_library_modules_size_ratchet.py both green except the 2 documented pre-existing chat_screen reds (30/2). Full Tests/Architecture: 15 failed/534 passed/1 skipped — all 15 failures identity-matched to the documented dev-drift list (2 chat_screen + 13 console/timer/worker/diagnostic), zero Library-related, zero new. Preflight: 6/6 green. Wiring+characterization+support-layer+recompose battery (11 files): 79 passed/2 failed (same chat_screen reds).
Task 5: self-caught regression -- the controller docstring fix above (item 3, "carries" -> "installed... deleted at cleanup") grew library_rag_search_controller.py by +2 lines (1895 -> 1897), drifting the governance pin I had just verified green. Caught by re-deriving the measurement independently via the ratchet's own `_measure()` function (not by re-running pytest alone, which had already gone stale before this edit landed) rather than assuming the earlier green run still held after a later edit. Re-pinned 1895 -> 1897 in test_library_modules_size_ratchet.py with a dated comment, same-commit, per §17's re-pin-at-move flow; re-ran both ratchets to confirm 30/2 restored. Exactly the wave's own "verified-before-written" discipline applied to my own work, not just the prior tasks' claims. Added as recipe §18 lesson 8.
Task 5: full sequential xdist paired-baseline sweep (Tests/UI -k "library" -n 8 --dist worksteal, branch then git stash -u to pristine a150fc766, per recipe §7): branch 357f/3924p (1323.95s), baseline 349f/3932p (1329.73s) -- baseline is an EXACT match to Task 4's own most recent baseline run, strong reproducibility signal. Diff: 347 shared, 10 branch-unique, 2 baseline-unique (both already documented). 2 of the 10 branch-unique already matched Task 4's own confirmed entries. Of the remaining 8, re-run combined single-process on branch: 4 passed (xdist noise). The other 4 (test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_navigation_and_local_back + 3 test_screen_navigation.py generic/rag-draft tests) still failed -- re-ran on a SECOND git stash -u to the same pristine a150fc766 tree, combined single-process: all 4 reproduced identically, confirmed genuinely pre-existing. One of them, ..._with_rag_draft, is literally the same name Task 4's own sweep flagged as BASELINE-unique in the OPPOSITE direction -- flip-flopping sides between independent runs is itself strong evidence of pure flakiness. Zero real regressions. All 4 added to recipe §7's documented pre-existing list. NOTE (process): my first attempt at isolating one of these was contaminated -- I ran it while the baseline sweep's 8 xdist workers were still active in the background, producing a spurious 30s mount-timeout; caught via `ps aux`, discarded, re-ran properly once the machine was idle.
Task 5: probe run (perl -e 'alarm 150; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py) landed inside the recipe's own recorded wave-2-close band: settle 264-464ms (recorded 264-485ms), max gap 56-132ms (recorded 54-195ms), recompose 0 in both, full-update 1-2 in both. Media-row mount/node counts ~5 higher than recorded (unrelated Media-screen drift since wave-2 close, not search+RAG -- Notes rows are byte-identical). No regression signal.
Task 5: complete. All items done: recipe close-out verified+corrected+8 lessons total added to §18/§7, task-31203 close-out verified+corrected (status already Done from Task 4, prose numbers fixed), stale-doc sweep (2 fixes + 1 flagged-out-of-scope), durable SDD evidence force-added (13 files), fresh measurements confirmed (+1 self-caught fix), full verification battery all green/zero-regression (ratchets, 11-file suite, full Architecture, preflight x2, full sequential xdist paired-baseline sweep, probe). Ready for commit.
