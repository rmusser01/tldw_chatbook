# Plan: Library residue batch (backlog tasks 3223, 3800, 3801, 3021)

Branch `fix/library-residue-batch` (worktree `.worktrees/library-residue`), cut from dev `4d0232358` (our polish-batch merge is dev's tip). Two plan tasks close the Library queue's last four items. Each plan task owns its backlog task files end-to-end (In Progress + plan before code; ACs + notes + Done after).

## Global Constraints

- **Python/tests**: `.venv/bin/python` (repo root) for everything; pytest is the ONLY entry point that may import app modules. Targeted tests + `--collect-only -q` over the touched trees. "no tests ran" = failed gate. Foreground Bash only — NEVER end your turn waiting on background work.
- **TDD** where behavior changes; for test-repair items the evidence is before/after failure counts + red-on-revert where a new guard is added. Rendered-geometry assertions for layout claims.
- **CSS**: never edit the generated bundle; source tcss → regenerate via build_css.py (+check_bundle_sync). Widget DEFAULT_CSS needs no bundle step.
- **Live verification** where a task touches visible UI (socket `resT<N>lib$RANDOM`, scratch /tmp/resT<N>, users_name sdd_rest<N>; the standard recipe: cold start ~12s, palette nav, char-index clicks, capture -e for styling proof, cleanup).
- **Git**: commit per task, message ends `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; add SPECIFIC paths; never stash/push.
- **Docs**: UI-visible changes update the matching Docs/User_Guide page or stamp (`Verified against dev @ 4d0232358`).
- **Known ambient**: dev currently carries ~52 failures in `Tests/UI/test_library_shell.py` (note/ingest real-DB churn from another arc) + 1 shadow-name video-gen failure in Tests/Library — A/B against clean origin/dev before blaming your change; do NOT fix those here.

## Task 1: Test-debt trio (backlog tasks 3223, 3800, 3801)

Read the three backlog task files (`ls backlog/tasks/ | grep -E "3223|3800|3801"`); their ACs bind.
- **3223** — `test_narrow_footer_collapses_but_f1_help_stays_truthful` fails ambiently at 90 cols (Settings surface). Diagnose at HEAD: the polish batch's footer changes may have altered the expected narrow-width string; repair the test to the CURRENT footer contract (ADR-031 + the rendered-globals dedup) — contract-derived expectation, not copy-paste-what-renders; if the failure exposes a real footer bug instead, fix the bug (widget-level, blast radius checked) rather than the assertion.
- **3800** — `test_action_library_skill_back_honors_dirty_guard` hits the fixture-bypass `AttributeError` on `_library_list_entry_focus_timer`. Fix at the FIXTURE level per task-3022's established shape (construct properly or set attributes); no production code.
- **3801** — add the bundled-CSS harness test pinning the nav ghost rule's width-neutrality (the round-4 border-reflow incident). Precedent: `Tests/UI/test_mcp_inspector.py`'s `InspectorAppWithBundledCSS` / `test_disabled_action_buttons_stay_legible_with_bundled_css`. The test loads the REAL bundle (CSS_PATH tier) and asserts a ghosted nav button's width is identical to its un-ghosted width. Must be red if a box-model property is reintroduced into the bundle-tier `.nav-button-clip-ghost` rule — prove via a temporary Edit-mutation of the SOURCE tcss + regenerated bundle, then restore both and regenerate again (verify check_bundle_sync clean after).
Exit: all three tasks Done with notes; the two repaired tests pass 5× each; the new harness test passes + mutation evidence recorded.

## Task 2: Home-surface import vocabulary + first-click honesty (backlog task-3021)

Read `backlog/tasks/task-3021 - Home-surface-import-vocabulary-and-first-click-honesty-audit.md`; its 3 ACs bind.
- Strings: `Home/active_work_adapter.py` "Opening Library ingest job details." and app.py's `HomeControlResult` "This ingest job can no longer be retried." → Import vocabulary (grep for any siblings the task file doesn't list; plain `grep -rn -i 'ingest'` over `tldw_chatbook/Home/` + Home-rendered strings in app.py, each hit fixed or justified one-line — the evidenced-sweep standard).
- First-click honesty: Home's Study rows/docs ("opens Study at flashcards", Docs/User_Guide/home.md ~line 75) — verify LIVE what Home's Study suggestion actually does at HEAD (memory: a filed defect said Home's suggestion button navigates from a different HomeAction than it displays — check whether that's since fixed); make gloss/docs describe the real first-click destination (the Library staging canvas vs Study directly). Do not change navigation behavior — this is an honesty pass; if you find live navigation BUGS, file them (scan IDs against origin/dev + worktrees, leapfrog with headroom) rather than fixing here.
- "Chatbook"-as-app-name in File Notes panels stays out of scope (recorded).
Exit: task-3021 Done; changed strings inventoried; affected pages re-stamped; live screenshot-equivalents for the Home rows touched.
