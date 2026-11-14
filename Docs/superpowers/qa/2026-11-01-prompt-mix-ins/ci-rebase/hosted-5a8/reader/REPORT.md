# Final reader restore repair — bounded evidence

Published base: `5a8f23eb9dfabeda25d46ed168d8ac675f483217`. Repository source/test files are frozen at the hashes in `final-frozen-proof.json`; `write_report.py` asserts all four identities still match and the final RED overlay used the same identities. No commit/index/branch/push or unrelated checkout changes were made.

## Scope and cause

Two focused mechanisms were repaired. A retained Reader's Media rail refresh no longer queues an independent generic first-row focus (`None if keep_media_reader else row0`); fresh list entry retains row0. Separately, the default synchronous recompose focus writer now arms the existing exact-target programmatic marker before `set_focus(..., scroll_visible=False)`, so its queued native DescendantFocus cannot select another media item. The common writer skips same-focused no-event calls and clears only its still-owned marker after failed/no-op focus. Later owners retain their marker. Genuine user keys remain authoritative. The generic Library focus helper is unchanged; this is not a claim that every programmatic focus path is fixed.

To preserve every existing size ceiling, only the existing restore method's executable body moved into the existing canvas_sync owner. The controller keeps its original signature and full docstring, and passes live defining globals for the original Widget/NoMatches/QueryError/grip dependencies. Reverse AST normalization proves the original gates, fallbacks and scroll=False call unchanged apart from the approved exact-marker/same-target/failure-cleanup delta. No new module, dependency, global suppression flag, alias or size allowance was added.

Sizes: Media controller5281→5246; canvas_sync1256→1314; LibraryScreen38177→38177 with only the approved argument conditional. Root owns exact ratchet rows, docs/task updates and publication. ADR required:no new ADR; routine focus repair and mechanical move within existing Library controller/canvas ownership.

## Authoritative verification

- `final-approved-old-red.log`:1 failed in9.61s at selected Media2 != Media1, after exact row event/dispatch and detail retirement. The private import overlay compiles only the exact pre-fix restore method in its original module globals; final repository source/test bytes stay untouched. All four overlay identities match final proof.
- `final-approved-green.log`: 5 passed in 34.91s. Ordinary final module run; no product callable replacement. Original three native Media/Notes journeys remain byte-identical as the entire published test file is an unchanged prefix. The original n→note→Media identity, source line, caret, scroll and loaded-row assertions/time budgets remain intact.
- New native case uses official App.run_test(message_hook=...), exact active-message-pump/target/focus matching and a queued call_later witness after the original synchronous handler. It waits for neutral owners and subsequent timer/request/detail-worker retirement; selected/loaded Media1 must survive system focus loss/restore to Media2's row. Real Down/Up/Down then checks DOM focus, selected ID, Reader selected ID, loaded ID and retirement for Media1/Media2/Media1.
- New permanent no-app facade test covers same-focused no-event, silent no-op, raised focus failure and a later owner's marker. It is one pytest node with four internal scenarios. The private10-mode self-check additionally retains missing/disabled/channel/None coverage as limited helper evidence.
- `wiring.log`:13 passed in1.51s; original controller/public wiring. This wiring run preceded helper signature formatting and restoring the approved Screen argument conditional; the final five-case module run qualifies the combined behavior. Root retains consolidated guard checks.
- `final-frozen-proof.json`: all original test/helper ASTs unchanged; original entire source byte prefix unchanged; facade signature/doc unchanged; reverse move AST true; sole approved Screen delta exact; all3new blocks formatted. Ruff vs exact published5a8:224 findings before and224 after, added0/removed0. Existing full-file lint/format debt was preserved, not represented as clean files. `git diff --check` passed.

Native deterministic contract is system-event qualification, not a reproduction of a hosted wall-clock/storage schedule. Initial resize suppression remains until real input; one normal Tab in setup clears it through the existing owner, then focus returns to the real Reader. No authority/resize/selection flag is written by the test. Early invalid-field and resize-precondition failures are harness-only, preserved in commands.json; intermediate proof/lint/RED/GREEN records are superseded by final-approved logs and final-frozen proof.

## Root handoff

Replace old fragile late-refresh census tail with:
`Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_row_restore_keeps_reader_until_user_keys`

Add permanent marker unit node:
`Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_restore_releases_only_its_failed_focus_marker`

Both are in the normally bootstrap_profile-marked module. Root chose floor1338. Final native slot is released after final-approved-green completion; no process is left running by this task. Root retains consolidated guards/performance/fresh hosted CI/review/commit/push/merge responsibility.

No full sweep, startup/performance rerun, live server test or historical cold keyboard baseline repeat was performed. Fresh hosted CI is still required before merge.
