# PR3299 current-dev integration and review repairs

- Incoming source: `a1fe280c15bae5c42f430fbe518d196431ebaf15`, branch `fix/task-33622.3-bare-letters-esc`.
- Authoritative integration base: `db7e04cf484b4c0b803366b00f2bbae1b5550042`, after PR3295 and PR3298. Rebased onto dev; main checkout was not edited.
- Initial cue-repair commit on this base: `618161fb9ddab7eb5a68cc69e9b78563b24d0df6`; the commit carrying this document adds the final independent-review corrections.
- Tasks: TASK-33622.3 and TASK-33622.4. Existing ADR: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`. No new ADR: routine corrections under its single-letter ownership and truthful-hints contract.

## Final behavior and preservation

The composer hides “Esc to type” while the focused Terminal captures Escape. It observes outside-control focus transitions and Terminal capture/release. Cue ownership changes use the existing measured draft refresh so a returning cue is painted at a full-width boundary; cursor blinking retains its layout-free path. Terminal input handlers are unchanged. The mounted regression records actual Terminal input messages through an inert backend, without launching a shell or provider.

Current dev's newer U report binding now follows Y's transcript-only ownership rule. Prose beginning with U from Help, status, or rail controls reaches the composer. U still invokes reports from the transcript, with its footer/F1 hints and user guide describing that scope. The report-opening implementation is unchanged.

The rebase preserves dev's received-input/CAS/run-feedback, attach/resume implementation, batched Stop classes, transcript presentation/selection helpers and archive container. The full 1,243-selector dev census is the exact prefix, followed by only the three incoming selectors (1,246 total). Incoming task IDs 37480/37621/37622/37960 have no current-dev filename or ID-field collision. Dev's unrelated TASK37620 remains intact; stale incoming menu comments now refer to 37960, while TASK37622 refers to earlier parent 33622.3 to avoid a forward task reference.

## Executed evidence

Raw receipts live under `C:/Users/GDesktop-1/.codex/worktrees/pr3299-resync/`, beside the isolated checkout. Each contains command.json, pytest.log, JUnit XML, before/after SHA256 manifests and result.json. Every listed run reported unchanged source hashes. The runner creates native Windows private 0700 config/temp roots, selects actual TLDW_CONFIG_PATH before app imports, then invokes `python -m Tests.windows_private_fixture_runner`. No scratch bootstrap plugin, participant reset, auth-guard override, normal-profile probe, or symlink workaround was used. Later receipts also hash CSS sources, generated bundles and the local build manifest; CSS was rebuilt before source freeze with no tracked stylesheet delta.

| Receipt | Selection | Result | Elapsed |
| --- | --- | --- | --- |
| r-68903920 | Final repaired ownership, hints, geometry and preservation batch | 12 passed, 1 setup failure, 0 errors/skips | 735.19 s |
| r-406bb744 | Final collapsed-Escape preservation node, diagnostics removed | 1 passed, 0 failures/errors/skips | 26.58 s |
| r-cf74ca14 | Exact-node publication diagnostic (`PYTEST_ADDOPTS=-rP`) | 1 passed, 0 failures/errors/skips | 24.64 s |
| r-9d1ffeee | Collapsed-Escape with attach readiness alone | 1 setup failure, 0 errors/skips | 29.39 s |
| r-7915a649 | U prose, U access/hints and painted cue before final repair | 3 expected failures, 0 errors/skips | 120.58 s |
| r-74208d86 | Selected adjacent preservation controls before fixture qualification | 10 passed, 4 setup failures, 0 errors/skips | 708.08 s |
| r-7f4b11e0 | All 25 original-plus-cue key-ownership cases on db7, before final U/geometry edits | 25 passed, 0 failures/errors/skips | 480.50 s |
| r-a0a3700d | Stabilized original Terminal cue regression before repair | 1 expected cue failure, 0 errors/skips | 47.67 s |
| r-d08aa9cc | Same original cue regression after initial repair | 1 passed, 0 failures/errors/skips | 49.17 s |

Final qualification is **12 passing checks plus one separate passing retry**, not a single 13-pass run. All 278 production/CSS/manifest paths remain byte-identical between r-68903920 and the final retry. Only the exact preservation fixture changed after the 12 passes. The whole key-ownership file was not rerun after the final scoped edits; its affected controls and new regressions are covered by the final batch. Earlier Ruff-only formatting of that file was verified AST-identical before adding the final regressions.

The 14-case preservation selection was narrower than three whole adjacent files: three arrival, two rail, three tab, two Terminal, one collapsed-Escape and three action/footer pins. Rail, tab, first arrival, direct Terminal keys, footer/workbench, status-line typing and Trace pins passed in that batch. Both failed arrival cases subsequently passed with their original five-second bounds unchanged. Terminal rail and collapsed-Escape initially stopped at `raw_source_selection_changed` before mounting: their app factories read the collection-bound config. Only those two nodes received the documented bootstrap_profile marker; all admission guards and behavior assertions remain.

The remaining collapsed-Escape failure occurred before any Escape: its seeded row did not appear. Attach/resume readiness alone did not solve it (r-9d1ffeee). The diagnostic then observed full sync returning False, follow-up requested, the same store and the seeded logical message present. Its raw `mounted_ids` label reads `_visible_messages`, so it proves logical projection, not DOM mount. The documented `_sync_native_console_transcript(force=True)` path for direct idle mutations published the real row and passed the original selection/Escape assertions. That flag bypasses only the display-token cache; real projection preparation and source guards remain. The final fixture retains bounded attach/resume/focus readiness and this real publication path; temporary diagnostic output is removed.

The three final-review RED failures reproduced U report dispatch from Help, absent contextual U hints, and a cue present in a two-row renderable but cropped from the actual 93x1 surface after Terminal release. Their repaired controls, both Terminal cue transitions and the ordinary focus-height control passed in r-68903920. The final painted-strip assertions remain in source.

Earlier attempts r-08263f83/r-f643739e/r-6d00285d/r-004d193c failed on an invalid queue witness or unsettled startup focus and are excluded from product RED/GREEN. r-b554ba56 reached the original cue defect; stabilized r-a0a3700d is authoritative. The initial key-ownership fixture now waits for worker attach/resume before deliberate focus changes, without altering production readiness.

The r-68903920 runner saved all artifacts, then its console echo hit a cp1252 UnicodeEncodeError on a triangle in the failure diagnostic. Counts and frozen-source status come from the saved artifacts. Later outer-runner calls use `-X utf8`; test execution is unchanged.

## Static checks, review and limits

- Census checker passed 1,246/1,246; the dev prefix was checked again after final rebase. No new duplicate method declarations relative to dev; inherited property getter/setter pairs remain.
- Targeted Ruff diagnostics match dev by code/message/source content: ChatScreen 18/18, composer_run_controls 3/3, composer 18/18, workbench test 22/22. Key-ownership, Terminal workspace and both per-node marker files are clean. Formatter checks pass for the key-ownership file, Terminal workspace and final collapsed-Escape file. Existing exact-type security checks were preserved; this is baseline parity, not globally clean lint.
- Size limits are explicitly waived. Exact pins: composer 7,139 lines; transcript 8,584; ChatScreen 24,513 lines/741 methods, with dated rationale. No behavior was removed to fit old pins.
- Self-review and targeted verification are complete. Final independent approval binds to the committed head in the companion `independent-integration-review.md` outside the checkout. The implementation subagent performed no remote push, merge or comments.
- No full suite, hosted CI, Qodo, size-gate sweep, live provider/shell or native terminal paint verification. Headless painted-strip checks are not native paint evidence.
- TASK37621 invisible selection layer, TASK37480 status-line cursor paint, TASK37622 Meetings lifecycle and TASK33622.5 new-session/Arm redesign remain bounded follow-ups.
