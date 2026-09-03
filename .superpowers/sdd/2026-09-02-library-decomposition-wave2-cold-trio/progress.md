# SDD ledger — plan: Docs/superpowers/plans/2026-09-02-library-decomposition-wave2-cold-trio.md

Spec: Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md (as corrected; binding). Recipe: backlog/docs/library-decomposition-recipe.md (mechanics authority).
Branch: refactor/library-decomp-wave2-cold-trio (stacked on refactor/library-decomp-0a-support-layer @ 2b20ebbb9), worktree .worktrees/library-decomp-foundation with own venv (verified in foundation run).
Baselines at wave start: library_screen.py 43,965 lines / 1,282 methods; recompose census 63=pin; export 51m/12f, collections 67m/28f, search 23m/4f.

## Pre-flight conflict scan

| Pair / task | Check | Finding |
|---|---|---|
| T1 vs T2-9 | T1 adds slack guard to recompose census; later cleanups may LOWER the census pin when pruning sites | Consistent — guard enforces lowering, which the recipe already mandates |
| T2-4, T5-7, T8-9 | Same-shape sequential series, one subsystem in flight at a time | Consistent with spec constraint |
| T5 vs T8 | `_library_collections_saved_searches*` ownership contested between collections and search | Plan explicitly assigns the decision to consumer census + recipe table; escape to BLOCKED defined |
| T8-9 | Search/RAG entanglement | Plan defines a quantitative escape (>1/3 cross-calls → STOP, controller ruling on a combined wave-3 series) |
| Plan vs rubric | No new plan-mandated rubric violations beyond the foundation's (shims/delegators duplication, already carried in reviewer constraints) | Clean |
| T1 self-consistency | "Write the failing guard first" via scratch-copy pin raise — inverted-TDD acceptable form for a guard (prove it CAN fail) | Clean |

Standing rulings inherited from the foundation run (all still in force): stacked-branch/no-push-until-finish; worktree-venv; xdist paired-baseline sweep protocol; per-move-PR pin lowering; monkeypatch-routing incl. ingest-options trio; census-before-prune; rev-parse-not-memory.

## Task log
Task 1: dispatched (implementer sonnet), BASE=68636a8fc
Task 1: implementer DONE (commit 477704580; tolerance 5, boundary-tested; 27019 Done; settings task filed); reviewer dispatching (haiku)
Task 1: complete (commits 68636a8fc..477704580, review clean — spec ✅, Approved, no findings)
Task 2: dispatched (implementer sonnet), BASE=477704580 — export series 1/3 (characterization + state)
Task 2: implementer DONE post-nudge (commits a0c8a6410 + f4e8acecf; 5 pins, 13/13 fields moved incl. origin_row_id class-attr edge case; ratchet 43965->43930; 0 branch-unique, 14 newly-documented pre-existing reds); reviewer dispatching (sonnet)
Task 2: note: monitor-parking stall recurred (3rd time) — future dispatches carry no-silent-parking up front
Task 2: forward note for export cleanup: _close_open_library_choice_strip dynamic setattr on quality_choices_visible (4th-bypass shape)
Task 2: review Approved with 1 Important (comments retyped not carried verbatim x3, self-review misreported) — fix round 1/5 dispatched; recipe-discipline point for the rehearsal series
Task 2: fix round 1/5 (1 addressed — verbatim comments restored, self-review corrected; commit 264314c5f)
Task 2: complete (commits 477704580..264314c5f, review clean after 1 fix round)
Task 3: dispatched (implementer sonnet), BASE=264314c5f — export controller move (series 2/3)
Task 3: implementer stalled (stream watchdog 600s) post-move pre-tests; work present uncommitted (controller 1307 lines + screen/test edits); resumed with verify-and-finish instructions
Task 3: implementer DONE (commits 4cc9b6109 + 5e74c64cb; 22/51 moved, 29 excluded across 3 rounds — 18 other-subsystem, 2 framework-decorator/@work-DOMNode, 9 unbound-fake-self found only by running the battery, 4 of those 9 in Tests/Library/ outside the recipe's canonical Tests/UI sweep root; ratchet 43930->43432/1282; xdist branch 333f/3901p vs baseline 332f/3902p, 5 branch-unique all confirmed xdist noise via isolated re-run, none export-related; 1 more pre-existing failure appended to recipe §7); full report at task-3-report.md
Task 3: complete (commits 264314c5f..5e74c64cb)
Task 3: implementer DONE post-stall-resume (commits 4cc9b6109 + 5e74c64cb; 22/51 moved, 29 excluded incl. 2 NEW bypass shapes (@work self-type assertion, silent Mock) + Tests/Library sweep-root lesson; ratchet 43930->43432; 5 branch-unique flakes cleared by isolated re-runs); reviewer dispatching (opus)
Task 3: complete (commits 264314c5f..5e74c64cb, review clean — spec ✅, Approved; exhaustive 51/51 verification)
Task 3: minor (deferred to Task 4): 5 named dead imports (LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP, MEDIA_QUALITY_OPTIONS, count_export_scope, default_export_name, normalize_export_destination)
Task 3: minor (deferred): cluster constants not self-defending against a 23rd export method — canon-consistent; note for a future guard improvement
Task 3: minor (deferred): ast.unparse delegator formatting — cosmetic, precedent-consistent
Task 4: dispatched (implementer sonnet), BASE=5e74c64cb — export cleanup (series 3/3)
Task 4: implementer DONE post-nudge (commits cdb43ebcc + 58118128c; shims out, 42+1 retargets, 1/22 delegators pruned, 5 imports pruned, ratchet 43413/1281, 0 branch-unique); reviewer dispatching (sonnet)
Task 4: complete (commits 5e74c64cb..58118128c, review clean — spec ✅, Approved; 3 reporting-precision minors)
Task 4: minor (deferred): pin 43413 is one above true measurement 43412 (ceiling semantics make it safe; next pin-lowering measures fresh)
Task 4: minor (deferred): _assign_library_reader_preferences_attribute now serves two subsystems under a reader-specific name — rename candidate for a docs/naming pass
EXPORT SERIES COMPLETE (Tasks 2-4). Task 5: dispatched (implementer sonnet), BASE=58118128c — collections state (series 1/3)
Task 5: implementer DONE post-nudge (commits 2ccfccbc7 + bca923b4c; 17 pins, 26 moved + 1 wiring-stay, saved-searches MOVE uncontested, ratchet 43410/1281 fresh-measured); reviewer dispatching (sonnet); NOTE field-count 28-snapshot vs 27-accounted needs reconciliation
Task 5: complete (commits 58118128c..bca923b4c, review clean — spec ✅, Approved; field-count reconciled: 27 exact, snapshot ~28 was an estimate)
Task 5: minor adopted as standing rule: wiring test commits WITH the pins commit (real RED in git history), before the state/screen edit — carried into Task 6+ dispatches
Task 5: minor (deferred to Task 7 cleanup): 4 named dead imports (CaptureCapabilities, CaptureHighlight, SavedCaptureSearch, CollectionsReaderMode)
Task 6: dispatched (implementer sonnet), BASE=bca923b4c — collections controller move (series 2/3)
Task 6: implementer DONE (commits 806cfea6f + 09d238f50 + 14ab3823b; 64/67 moved, 3 Prompts-owned exclusions, RED commit landed separately)
Task 6: controller-brief error confirmed: library_collections_browse_controller.py was deleted by dev 5dd1077df pre-base — my dispatch carried a stale foundation-era reference; implementer resolved correctly against the real tree
Task 6: reviewer dispatching (opus) — top risk: 12 branch-unique sweep failures cleared as flake (volume well above wave norm), plus concurrent-sweep amplification note
Task 6: review Needs Fixes — code exemplary (full-scale verification clean), 1 Important is RECORD-only (false clearing rationale for test_closeout_single_app_route_cycle, which DOES traverse moved code) + census overstatement + 2 unnamed test ids — fix round 1/5 dispatched (docs-only, with a baseline reproduction demanded for the one overlapping test)
Task 6: Ruling: standing RED-commit rule wording aligned with practice (2 occurrences) — the RED commit must leave the SCREEN untouched and its delegation tests failing at the parent; the controller module MAY ship in it. — Cost if wrong: slightly weaker RED purity; the structural criterion (screen untouched, tests red at parent) is the part that matters and is preserved.
Task 6: fix round 1/5 (3 addressed — overlap named + baseline reproduction run, census corrected + recipe guidance added, test ids named; commit 91feba4a7)
Task 6: complete (commits bca923b4c..91feba4a7, review clean after 1 fix round)
Task 7: dispatched (implementer sonnet), BASE=91feba4a7 — collections cleanup (series 3/3)
Task 7: implementer DONE post-nudge (commits 39a976321 + 1e466ffac; shims out, 14+49 retargets incl. Tests/Live discovery (3rd test root), 14/64 delegators pruned, 10 imports pruned, ratchet 42411/1267)
Task 7: NOTE for wave close + user: origin/dev drifted 337 commits since foundation base — landing urgency for #2315/#2316 rising; ratchet-row conflicts predicted by the final review will materialize
Task 7: reviewer dispatching (sonnet)
Task 7: complete (commits 91feba4a7..1e466ffac, review clean — spec ✅, Approved; all 7 risks verified against live repo)
Task 7: minor (deferred to wave-close docs pass): library_collections_state.py module docstring describes the deleted screen shim block (same shape as the conversations one the foundation fix wave corrected)
COLLECTIONS SERIES COMPLETE (Tasks 5-7). Task 8: dispatched (implementer sonnet), BASE=1e466ffac — search series with entanglement gate
Task 8: BLOCKED-at-gate, correctly — search/RAG cross-call census 8/14 (57.1%), conservative cut 5/11 (45.5%), both >> 1/3 threshold; entanglement structural (search submit calls _start_library_rag_query; _execute/_apply rag-search pair is one mechanism). No code touched.
Task 8: Ruling: search does NOT extract alone. Search+RAG becomes ONE combined series in wave 3, per the wave plan's pre-committed contingency; wave 2 closes with export + collections complete + the census guard. Census preserved in task-8-report and to be copied into the recipe's per-subsystem table at wave close. — Cost if wrong: search waits one wave; nothing regresses (zero code was touched).
Task 9 (search cleanup): MOOT — no search move occurred. Task 10: dispatched (implementer sonnet), BASE=1e466ffac — wave close.
Task 10: implementer DONE (docs-only: recipe §8 table completed + new §16 wave-2-close summary; stale post-cleanup docstring fix in BOTH library_export_state.py and library_collections_state.py, not just the one ledger-flagged collections file; task-27021 committed). Fresh _measure() 42411/1267 — exact match, zero drift. Full battery green (16 wiring + 14 characterization + 3p/2f ratchet incl. only the 2 pre-existing chat_screen reds + 6 recompose-census + 8 support-layer + preflight 6/6). Full xdist paired-baseline sweep run SEQUENTIALLY per Task 6/7's own forward note: branch 335f/3904p vs baseline (2b20ebbb9 overlay) 343f/3895p, 5 branch-unique, all confirmed pure noise by combined single-process re-run, zero real regressions. Probe run recorded as the FIRST-ever captured baseline (no prior run exists anywhere in the repo despite §9 calling for one around each controller-move PR — flagged, not silently reconciled). Commit: `docs(library): wave-2 close — recipe tables, stale-doc alignment, task-27021`.
WAVE 2 CLOSED. Export + collections series complete; search BLOCKED at the entanglement gate by design, deferred to wave-3's combined search+RAG series (task-27021).
Task 10: complete (commit 09a5cadff; fresh 42411/1267 exact; all suites green; sweep zero real regressions; probe first-capture recorded; brief assumptions corrected honestly)
FINAL REVIEW (wave 2): dispatching (fable), MERGE_BASE=2b20ebbb9 HEAD=09a5cadff
FINAL REVIEW (wave 2): MERGE-READY WITH CONDITIONS (fable). 0 Critical. Important #1: guard-gap note + search census only in git-ignored files -> fix wave commits the SDD dir (repo precedent) + copies the note into the recipe. Important #2: spec absent from stacked branches -> PR-ordering guidance (#2316 with/before #2315), no code. Minors: stale comments x2 modules, construction-order sentinel, controller-size governance note. Triage: all prior deferred minors RESOLVED or FINE-TO-DEFER. Ruling audit: no disputes; Task-8 BLOCKED called the wave's strongest decision. Dev drift now 381 commits, ~1325 new lines in library_screen.py on dev — rebase is textual pain, verified zero semantic overlap with moved/pruned names.
FINAL FIX WAVE (wave 2): dispatched (sonnet) — durable record, stale comments, sentinel, wave-3 notes; pin accounting pre-ruled under PR-granularity framing (net trajectory down).
