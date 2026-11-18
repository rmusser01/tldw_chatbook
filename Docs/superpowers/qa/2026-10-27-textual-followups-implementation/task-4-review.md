# TASK-37034 task review

## Spec Compliance

- ✅ Spec compliant for the task-scoped diff `c375f164e6955c68c90c3d5e75a6c936023dacbc..1be8428cd7dffbd5c7c7de870454544b4306d1e3`.
- AC1: `tldw_chatbook/UI/Screens/stts_screen.py:159,190,211,431` derives the custom registration and native action availability from the same current-view/attached-owner target. Lab hints remain concatenated and unrecognized actions delegate to the parent. `UI/STTS_Window.py:1992,2119` emits owner-tagged change messages at actual replacement boundaries. The mounted test checks the painted footer child, absent bindings, and Effects Enter (`Tests/UI/test_speech_footer_scope.py:78`).
- AC2: the same journey holds the native return mount, verifies the no-target interval, releases it, requires restored hints/bindings, and proves returned rail Clear changes real TextArea content (`Tests/UI/test_speech_footer_scope.py:137`).
- AC3: the five existing mirror bindings remain nonpriority and unchanged. Availability reads ownership/current view and queries the pane; it invokes no synthesis, capture, playback or provider method (`stts_screen.py:131,190,211`). The mounted journey exercises Lab mode focus and printable `grxps` input (`test_speech_footer_scope.py:135,177`); existing shortcut/Recognition controls remain qualified by retained runs.
- AC4: both new tests use the production CSS host and actual widgets/keys. Gating wraps and forwards native lifecycle operations, rather than replacing their results or state machine (`test_speech_footer_scope.py:34,78,191`).
- Scope: all brief-listed files have corresponding changes. The additional existing STTSWindow notification seam and recomposition wrapper directly implement the same target-ownership requirement, without a shared registry, dependency, state contract or mount hook. The four legacy module profile/readiness adjustments preserve their action assertions. The capability correction matches the existing fifth OmniVoice capability and the fixture that omits its dependencies (`Tests/UI/test_stts_capability_state.py:141,179,187`). No missing or unapproved extra feature found.
- ⚠️ This is not whole-branch or merge approval. Final combined-source/current-head hosted gates, shared CI census, task closure and PR obligations remain with root. The retained local evidence is explicitly qualified; no pristine-output, passing timing gate, full-suite or live audio/device claim is made.

## Strengths

- Exact owner rejection and recomputation prevent a delayed message from replaying stale action snapshots (`stts_screen.py:218`). Before recomposition the old owner is revoked and persisted/rendered hints are cleared; the finally clause restores only a still-attached owner if native removal was refused (`stts_screen.py:171`).
- Recomposition coverage sends old-owner messages both while the old window is still attached and after replacement, then proves a working Clear on the fresh pane (`test_speech_footer_scope.py:191`). This targets the lifecycle hazard rather than merely checking a helper return value.
- Native readiness synchronization in legacy tests waits for the actual window Mount and workers; it does not increase timeouts, remove behavior assertions or introduce skips (`test_speech_shortcuts.py:101,144`; `test_speech_rail_navigation.py:102`; `test_stts_capability_state.py:169,236`).
- Scope is small in production: six changed/added screen methods and one existing window replacement method plus its message type. Documentation describes the owner and the no-work availability contract (`Docs/Architecture/speech.md:137`).

## Issues

### Critical

None (0).

### Important

None (0).

### Minor

1. Inherited output/guard hygiene remains open, not introduced by this task (1): `.test-tmp/textual-37031-37034/task4-final02/run.log:1,12,16` and the private-child logs retain the unset pytest-asyncio loop-scope warning and import-budget diagnostics. `task4-static/format-final.log:1` retains two format failures; `task4-static/source-static-final.json` records seven matching baseline E721 findings. This does not invalidate the demonstrated Speech behavior, but these receipts must not be described as pristine or all-gates-green. Preserve the qualifications and handle baseline cleanup separately; do not broaden this repair or raise allowances. The unchanged broad MRO and timing failures remain root integration qualifications, as recorded in the implementer report and actual `task4-final01/result.xml` / `task4-import-original01/result.xml`.

## Checks and concrete-risk inspections

- Read the task brief, actual task, exact binding constraints, implementer report and immutable diff. Recovered the tool-truncated middle of that diff by bounded ranges. No git commands, test execution, source/index/HEAD mutation or subagents.
- Named risk: replacement messages or the current-view watcher revive old hints. Completed the diff-cut `_mount_view` / `_mount_view_unlocked` context and inspected the unchanged watcher/owner construction at `STTS_Window.py:1799,1946` and `stts_screen.py:308,362`. Mounts serialize under the existing lock; notifications carry no old view snapshot; footer calculation reads the current owner. The watcher may run during replacement but cannot bypass the target predicate.
- Named risk: native refusal, state semantics or MRO duplicate mount changes. Inspected `UI/Screens/lab_frame.py:283,296,311,412`, `UI/Navigation/base_app_screen.py:371,452,652`, and installed Textual `widget.py:1704`. Base refusal raises before removal, so the wrapper's finally restores the original attached owner and rethrows. Lab retains its one existing mount handler/deferred body path. Native/Lab recomposition does not call save_state/restore_state; unchanged `stts_screen.py:338,352` remains the state seam. No new axis/view preservation claim is inferred.
- Named risk: revised 4/5 expectation hides a capability regression. Inspected the cut-off fixture setup (`test_stts_capability_state.py:141`) and the existing local capability declarations/probes (`UI/Lab_Modules/lab_speech_status.py:30,81,145`; `UI/Speech/speech_runtime_status.py:40`). Five capabilities already exist and OmniVoice requires onnxruntime plus tokenizers; neither is in that fixture. The new explicit missing-OmniVoice assertion is correct.
- Independently parsed actual retained `task4-red01` and `task4-recompose-red01` parent and private-child XML: each has one intended assertion failure, zero errors/skips. Original failure is absent Playground still advertised; recomposition failure is cached hint retained.
- Independently parsed `task4-final02/result.xml`: nine passed, zero failures/errors/skips. Both actual private-child XML files contain one pass, zero failures/errors/skips. All six recorded source hashes in each child independently match current native bytes (12/12 matches).
- Independently parsed qualified legacy XML: controls01 = 13/1 failures, final01 = 30/5, amended01 = 3/1, behavior01 = 21/2, import-original01 = 1/1; all have zero errors/skips. Fourteen behavior01 private children and the final01 private child pass. These confirm the report's qualification rather than promoting red batches to green. No already-answered test was rerun.
- Read the retained focused Ruff pass, format debt and final static baseline/method-identity report. Current-hosted integration gates remain unverified by this review.

## Assessment

**Task quality: Approved.**

**Counts: Critical 0; Important 0; Minor 1 inherited hygiene qualification; introduced implementation findings 0.**

The implementation aligns custom hints, native bindings and dispatch with the existing active target owner and handles the demonstrated replacement/recomposition gaps without changing unrelated Speech behavior. The mounted RED/GREEN evidence is source-qualified and the remaining baseline limitations are disclosed rather than suppressed.
