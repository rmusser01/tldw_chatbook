### Spec Compliance

- ✅ **Spec compliant for fix round1. I1 is addressed.** Range: `e8fa9b0b97811e46c7431689320622396555b219` → `03caf0a6b4bd1b10617bf3a86360c9e904981981`. This verdict covers the requested authority-lifetime correction and new breakage in this exact delta; earlier completed reviews remain completed.
- `tldw_chatbook/Chat/console_chat_controller.py:26573–26626` now places rider generation and the awaited reseal inside the existing bridge cleanup region, immediately after admission. The existing `finally` at `:26884–26886` releases the exact admitted authority on return, exception or cancellation. There is no intervening await between successful admission and entering the protected region.
- The helper, source ownership proof, immutable replacement and content-free `not_established` exception remain unchanged. The moved block retains the existing persona, kill-switch and native-provider conditions and remains before bridge fitting/counting. Existing error propagation at `console_chat_controller.py:26849–26853` preserves the controlled trace-refusal path; existing cancellation handling at `:26805–26848` now covers this await too.
- `Tests/Chat/test_console_archive_recovery_followup.py:456–558` provides the missing native-admission regression: actual admit/release, real captured predecessor search, real helper with an unknown source for refusal, and actual Stop while the helper is held for cancellation. It checks one exact release, closed/removed/unauthenticated pending authority, no run activation, no successor gateway/capture-factory entry, no confirmation or restored target, unchanged predecessor call rows/reconstruction, and a non-exception native outcome. Refusal additionally checks the blocked status and exact sanitized error/status.
- The test's `finally` releases the barrier, waits for the held callable's completion event, gathers the native turn, then clears any exact paused preparation through the existing Cancel API and requires no dispatch-recovery owner. This is a callable-completion barrier and task gathering, not an OS executor-thread join. The appended QA describes that distinction accurately.

### Strengths

- The production fix reuses the existing owner and cleanup path instead of adding another authority registry or parallel release policy (`console_chat_controller.py:26573,26884–26886`). No source-proof relaxation or Cancel/Discard production repair was introduced.
- The new regression observes real admission and delegates to real release (`test_console_archive_recovery_followup.py:478–486`); it does not replace cleanup with a fake. Its cancellation assertion runs while the helper is still blocked (`:537–539`), proving cleanup is not accidentally supplied by later worker completion.
- The focused final selection includes both captured Direct/RAG success paths and existing typed-transform composition alongside the two new failure-lifetime cases. It checks the moved protection does not merely turn the leak into a general refusal.
- The authored QA preserves failed intermediate phases, historical hashes/logs, inherited static debt and precise evidence scope. The copied original review is explicitly historical input; it does not replace this fix disposition.

### Issues

#### Critical (Must Fix)

- None identified.

#### Important (Should Fix)

- **I1: Addressed.** The acquire→reseal→dispatch lifetime now reaches the same exact-authority finally on the formerly uncovered refusal/cancellation paths. No new Important defect identified.

#### Minor (Nice to Have)

- No new actionable Minor defect identified.

### Evidence and Limits

- Read the scoped brief, existing task-reviewer instructions, original full review, appended fix report, and all source/test/authored-QA changes in the supplied companion. Inventoried the full19-file package and inspected relevant raw RED/green/final/static/store evidence. Did not regenerate the diff or rerun any test.
- **Named surrounding-code check:** the source hunk does not include its reused handlers/finally. Read `console_chat_controller.py:26795–26895` specifically to verify that widening the try preserves TraceCallPersistenceError propagation, handles cancellation without relying on an assigned run_id, and executes the existing release. No unrelated unchanged-code investigation was performed.
- **Observed RED:** `fix1/red.log:72–73` shows the refusal's authority remaining unclosed; its cancellation case initially exits through `Cannot mark a stopped message terminal.` (`:1012`). `fix1/red2.log:74–75` directly observes the cancellation authority leak after retaining the native exception for the custody assertion. Summaries are2failed/6.70s and1failed/2.83s; neither has teardown errors. These phases are not relabeled green.
- **Observed green:** `fix1/green.log:7` records2passed/4.93s. The entire `fix1/final.log:1–11` records **5passed/21.62s**, including refusal, cancellation, captured Direct/RAG and typed composition; no warnings or teardown/network-guard errors appear. The fix did not repeat the original39-node selection or previous feature waves.
- **Observed artifacts:** `fix1/static-final.json` records reseal await26619 under cleanup try26573 and finally release26886, unchanged191 Ruff signatures, no pending formatting edits in the modified protected block, and final controller/test hashes. `fix1/verification.json` shows each of the two custody stores retaining only3 predecessor COMPLETE calls and both archived targets. Source assertions additionally verify prior reconstruction equality and no successor factory/transport entry. Root independently verified15 newly manifested artifacts,14 immutable historical artifacts, final source/test hashes and3 unchanged boundary files; no redundant regeneration was necessary here.
- **Post-test formatting:** final import/expression/assertion wrapping followed execution. The companion shows formatting-only differences for those wraps, without executable behavior changes. The final recorded hashes identify those formatted bytes; the5-pass run establishes the corresponding pre-wrap executable behavior, as explicitly disclosed.
- **Static qualification remains bounded:** final test Ruff/format logs are clean, but whole-controller Ruff retains191 baseline signatures and whole-file formatting remains non-pristine. The correction does not claim global lint/format success. Source/authored-QA scoped whitespace is clean; raw artifacts intentionally retain warnings.
- **Six exact raw-log whitespace exceptions remain:** under `Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-capture-on/`, `red.log`, `red2.log`, `red3.log`, `red4.log`, `fix1/red.log`, and `fix1/red2.log`. The first four retain14 historical trailing-space lines; the two new logs retain34 (30+4). The full artifact diff is not whitespace-clean. The ledger's byte-preservation ruling waives no source or authored-document path.
- ⚠️ **Not newly verified:** live Capture-On behavior, generic blocked-checkpoint Cancel/Discard recovery, the dictionary producer's earlier failure, historical unsupported_surface_change/provider errors, +729 FD attribution and Next Send preview/estimate behavior. The final green is automated fake-transport qualification through native runtime/capture boundaries, not a live-model walkthrough. Neither I1's fix nor this scoped review resolves those separate limits.
- **Review actions:** no tests, app/model/network requests, subagents, source/index/HEAD/task writes or broad sweeps. Only this requested review report was written.

### Assessment

**I1 disposition: Addressed. Spec verdict: Compliant within scoped fix review. Task quality: Approved.**

**Actionable finding counts: 0 Critical, 0 Important, 0 Minor.** The exact newly introduced authority-lifetime gap is closed with targeted native refusal/cancellation evidence, and no new breakage was identified in the fix delta. This approval preserves the earlier evidence boundaries and is not a new full-feature or live qualification verdict.
