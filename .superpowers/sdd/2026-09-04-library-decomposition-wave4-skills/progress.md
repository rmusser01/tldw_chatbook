# SDD ledger — plan: Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md

Spec (dev, corrected) + recipe §1-§18 binding. Branch: refactor/library-decomp-wave4-skills-ingest — NOTE branch name says skills-ingest but scope RULING below narrows to skills only (wave 5 = ingest).
Worktree: .worktrees/library-decomp-foundation (own venv, verified prior waves). Baseline: screen 43225/1311; skills cluster 133 methods (both prefix families, raw match) / 38 init-fields; existing skills controllers (import-coordinator, browse) untouched per plan.

## Pre-flight conflict scan
| Check | Finding |
|---|---|
| T1→T2→T3 same-shape series | Consistent with four prior series |
| Existing skills controllers vs new extraction | Plan pre-rules: untouched; delegating methods are exclusion candidates |
| Two prefix families + fourth test root (Tests/Skills) | Named in every task; the wave's likeliest novel trap |
| 133-method scale | Split-controller escape defined; sequential commits if split |
| Rubric conflicts | Same plan-mandated duplication as prior waves; carried in reviewer constraints |

Ruling: wave-4 scope = skills ONLY despite the branch name (measured 133/38 is the largest series yet; ingest 78/20 deferred to wave 5 rather than rushed). — Cost if wrong: branch name slightly misleading; content correct.
Standing rulings inherited: all of waves 1-3 (per-move pin lowering, no-red-ships, canon-docstring scope, born-governed, sweep protocol, monkeypatch/module-globals routing incl. the ingest-options trio which is INGEST-scope anyway, rev-parse hashes, verbatim comments, never-park, convergence loop at PR).

## Task log
Task 1: dispatched (implementer sonnet), BASE=2372ea764 — skills state PR (series 1/3)
Task 1: DONE. RED ef289548a (wiring test + characterization pins + LibrarySkillsState module),
GREEN 87c318d57 (screen __init__ collapsed + shim block + ratchet 43225/1311 -> 43179/1311).
38 fields censused (both prefix families + `_selected_skill_name`, missed by a `startswith`-only
pass — recipe §11 trap reproduced on a third prefix shape): 36 moved (3-way shim prefix, a
genuine deviation from the plan's own "two-prefix" framing), 2 wiring-stayed
(`_library_skill_import_coordinator` per plan + `_library_skills_browse_controller`, same
capture-controller precedent). Characterization: 6 new pins covering 9 genuinely-unpressed `@on`
handlers (CSS-selector-level census, not method-name grep) across the four roots; 1 dead/
unreachable selector found and left alone; several already-characterized-via-unbound-fake
handlers left unpinned for consistency. Full battery green: wiring RED->GREEN, both ratchet
guards, controller-ratchet untouched, support-layer + all 4 prior wiring suites green (60/60),
full Architecture suite (16 pre-existing failures, all confirmed via git-stash baseline),
`-k "skill and library"` sweep (11 failures, all pre-existing/order-noise), Tests/Skills full run
(2 pre-existing failures), full sequential xdist paired-baseline sweep (branch 370f/3933p vs.
baseline 371f/3928p; 9 branch-unique, 8 resolved as noise/pre-existing on isolation+baseline
reruns, 1 — test_closeout_single_app_route_cycle — investigated in depth: same failure
signature reproduces on both trees at different observed rates, no plausible code-level
mechanism found, classified as pre-existing timing-sensitive flakiness under this session's
heavily fluctuating machine load, not a regression). preflight clean. Report:
.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/task-1-report.md.
Process note: a brief mid-investigation checkout mixup (grepped the wrong commit once) was
self-caught and corrected before any conclusion relied on it; recorded in the report's own
self-review rather than smoothed over.
Task 1: implementer DONE post-nudge (commits ef289548a RED + 87c318d57 GREEN; 36 moved/2 wiring-stay, 3-prefix shim deviation disclosed, 6 pins/9 handlers, ratchet 43179/1311; closeout-cycle flake triaged as pre-existing under load); reviewer dispatching (sonnet)
Task 1: complete (commits 2372ea764..87c318d57, review clean — spec ✅, Approved, no blocking findings; 36-shim runtime probe + independent census all held)
Task 1: minor (deferred): closeout-cycle flake triage rests on an 11-run sample (mechanism soundly ruled out; rate disparity unexplained) — a 20+-run paired quiescent sample would settle; candidate for the flake follow-up tasks
Task 2: dispatched (implementer sonnet), BASE=87c318d57 — skills controller move (series 2/3)
Task 2: implementer DONE post-nudge (commits 5ecf223d4/60857a2be/679a90d1b; 86/127 moved single controller; 41 exclusions incl. NEW bare-self-identity class; 2 mid-task regressions battery-caught + reverted; screen 41247/1311, controller 3099); reviewer dispatching (opus)
Task 2: review Needs Fixes — mechanics 86/86 verified; 1 CRITICAL (unbound `focused` on controller: getattr-default masks a dead focus-preservation fix from 8027e99f0 — the review's own bare-self sweep found the one hit the census reasoned past) + 1 Important (new bypass class absent from recipe §3, widened to unbound-attribute escapes) + 3 minors — fix round 1/5 dispatched with a mandated fail-without/pass-with covering test
Task 2: fix round 1/5 (CRITICAL + Important + 2 minors all addressed; revert-probe both directions; canon preserved — call site unchanged, binding added; pin 3131; commits bf13b133b + f472f7512)
Task 2: complete (commits 87c318d57..f472f7512, review clean after 1 fix round)
Task 2: forward note for a follow-up filing at wave close: pre-existing loading/ready race in library_skills_browse_controller.py (found building the covering test; unrelated to the move; could cause intermittent focus-restore misses against a fast skills service)
Task 3: dispatched (implementer sonnet), BASE=f472f7512 — skills cleanup (series 3/3)
Task 3: implementer DONE post-nudge (commits ed4c29d45 + 2a744c434; shims out, 130+269 retargets, 16/86 pruned, 28 imports, pins 41155/1295 + 3140; 2 self-caught methodology bugs documented; sweep under heavy load but paired-clean); reviewer dispatching (sonnet)
Task 3: review Approved on all correctness risks; 1 Important (18-vs-28 restructure count in recipe/report) + 1 minor (SkillEditorState liveness attribution) — fix round 1/5 dispatched (doc-only)
Task 3: fix round 1/5 (2 addressed — 28 verified independently in both records; SkillEditorState homes named; commit f42f75d98)
Task 3: complete (commits f472f7512..f42f75d98, review clean after 1 fix round)
SKILLS SERIES COMPLETE (Tasks 1-3). Task 4: dispatched (implementer sonnet), BASE=f42f75d98 — wave close
Task 4: DONE. Recipe §8 skills row rewritten to "complete" with the full field/method/exclusion
summary; §19 gains a "Wave-4 close" subsection: pin trajectory re-derived fresh from git log
(exact match to all three tasks' own numbers — screen 43225/1311->43179/1311->41247/1311->
41155/1295, controller born 3099->3131->3140); fresh `_measure()` on both ratchet files (exact
match, zero drift); combined wiring/characterization/support-layer/size-guard run (83p/2f, both
pre-existing chat_screen rows); full Architecture run (543p/1skip/16f, one fewer than task 2's
17 — an external git-object-availability change in test_persistent_diagnostic_inventory.py, not
this diff); `-k "skill and library"` sweep (10f/272p, exact match to task 3's baseline);
Tests/Skills full run (538p/1f, the other documented pre-existing flip to pass); preflight clean.
Full sequential xdist paired-baseline sweep, WHOLE-WAVE span (2372ea764 vs f42f75d98+close, not
just this task's own trivial doc diff, per wave-2 close's own precedent): branch 372f/3932p
(28:19) vs baseline 365f/3934p (23:18), both near the documented ~330-371 historical backdrop
under this run's own recorded ~22-47 load average; 356 shared, 9 baseline-unique (one of them
test_closeout_single_app_route_cycle, flipping direction from task 1's own observation —
corroborating TASK-31422's premise), 16 branch-unique, all resolved (12 passed on combined
re-run, 4 passed in true isolation) — zero unexplained branch-unique failures across the whole
wave. Probe run recorded with an honest machine-load caveat: every interaction came in slower
than wave-2's own band, attributed to this session's sustained 8-9+ concurrent pytest processes
(measured, not assumed) rather than a code regression, since this task's diff touches zero code
on the probed Media/Notes rail-switch path. Follow-ups filed: TASK-31421 (skills-browse
loading/ready settlement race, Task 2's own forward note) and TASK-31422 (closeout-cycle flake
20+-run paired quiescent sample, Task 1's own minor). True max task id swept at 31420 across
every remote ref + local branch + worktree; CLI auto-assign probed and confirmed stale (offered
31263); both filed by hand at 31421/31422 with two-item hand-authored AC blocks. Stale-doc sweep:
library_skills_state.py's own module docstring rewritten from future to past tense (comment-only)
to match the collections/search+RAG post-cleanup template; library_skills_controller.py's own
docstring re-checked and already correct. Durable evidence force-added (ledger + 3 reports + 5
review diffs + this report). Report: .superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/task-4-report.md.
WAVE-4 COMPLETE.
