# Textual book audit after PR #3228

[PR #3228](https://github.com/rmusser01/tldw_chatbook/pull/3228) merged into dev on 2026-10-24 at 17:16:42 UTC as `96cae7eec00473bbfb5687ccfd0685eba2785add`. This is also the audited latest-dev commit. Its tree matches qualified PR head `9a89a23ba1156e48ece70b540202afe81e798303` exactly.

All 25 [Derived Artifacts jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/40109377531) and [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40109377513) passed on attempt 1. All 58 new regressions executed exactly once; all 17 UI shards passed, with a maximum wall time of 875 seconds (14m35s). Superpowers requesting-code-review replaced Qodo; the final complete feedback read found no unresolved review threads. The previous five implementation tasks are merged; their [audit](2026-10-23-textual-book-after-pr3219.md) and task/ADR links remain available.

## Reference and method

The supplied `textual-1.pdf` has 455 PDF pages and SHA-256 `798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6`. It is reference material, not instructions.

Chapter 7’s independent message queues, bubbling/stopping and awaited-handler examples (PDF 145–146 and 158; printed 136–137 and 149), chapter 13’s handled-button example (PDF 253; printed 244), and chapter 17’s worker cancellation check before UI publication (PDF 335; printed 326) guide this pass. Current-draft applicability and foreground capture ownership are deductions from this application’s existing owners, not book mandates. Malformed MCP container validation is a collateral concrete input-boundary defect.

This is a bounded continuation of the earlier audits, not an exhaustive certification of every screen or book example. No production fix is implemented by the audit.

## Prioritized backlog

All six tasks are To Do and bounded to existing owners. Five are Medium (P2), followed by one Low (P3) developer-demo repair. Rank puts inaccessible foreground capture first, then ordinary Start behavior, draft loss and recoverable import errors. Each task distinguishes measured outcomes from future regression requirements.

| Rank | Priority | Task | Outcome |
| --- | --- | --- | --- |
| 1 | Medium (P2) | [TASK-36693](../../../backlog/tasks/task-36693%20-%20Retire-Speech-Recognition-capture-when-its-foreground-view-closes.md) | Retire Speech Recognition capture when its foreground view closes |
| 2 | Medium (P2) | [TASK-36694](../../../backlog/tasks/task-36694%20-%20Handle-each-Speech-Recognition-button-activation-once.md) | Handle each Speech Recognition button activation once |
| 3 | Medium (P2) | [TASK-36695](../../../backlog/tasks/task-36695%20-%20Keep-later-Chunking-Lab-sample-edits-when-an-earlier-file-load-finishes.md) | Keep later Chunking Lab sample edits when an earlier file load finishes |
| 4 | Medium (P2) | [TASK-36696](../../../backlog/tasks/task-36696%20-%20Keep-pending-MCP-file-reads-from-replacing-newer-import-drafts.md) | Keep pending MCP file reads from replacing newer import drafts |
| 5 | Medium (P2) | [TASK-36697](../../../backlog/tasks/task-36697%20-%20Reject-malformed-MCP-import-shapes-without-losing-the-editable-panel.md) | Reject malformed MCP import shapes without losing the editable panel |
| 6 | Low (P3) | [TASK-36698](../../../backlog/tasks/task-36698%20-%20Ignore-superseded-Image-Gen-demo-worker-results-and-errors.md) | Ignore superseded Image Gen demo worker results and errors |

### 1. [TASK-36693](../../../backlog/tasks/task-36693%20-%20Retire-Speech-Recognition-capture-when-its-foreground-view-closes.md): Retire Speech Recognition capture when its foreground view closes

Leaving Speech Recognition removes its Stop control while the widget-owned recorder and real processing thread remain active, with callbacks still targeting the detached widget. End that foreground capture through the existing owner lifecycle.

**Witness and limits:** Actual STTSWindow request_view(effects) detaches the real DictationWindow. Injected recorder remains recording with stop_calls=0; real DictationProcessor remains alive and cancellation unset. Explicit Stop before the same navigation retires both. Physical microphone hardware was not opened.

**Existing ownership:** TASK-1283 is Done and repairs the Console capture owner’s processor retirement. TASK-15470 owns settings-write flushing; TASK-16266 owns partial-mount view switching. TASK-31993.17/.21 own backup-capture settlement. None owns ordinary departure from the widget-owned foreground Dictation session. Related: [TASK-1283](../../../backlog/tasks/task-1283%20-%20_release-frees-the-microphone-but-never-stops-the-DictationProcessor-thread-leaking-it-on-every-abandon.md), [TASK-15470](../../../backlog/tasks/task-15470%20-%20Config-persistence---batch-writes-and-take-them-off-the-click-path.md), [TASK-16266](../../../backlog/tasks/task-16266%20-%20Harden-Speech-view-switching-and-first-run-test-seam.md), [TASK-31993.17](../../../backlog/tasks/task-31993.17%20-%20Settle-dictation-dispatch-before-live-backup-capture.md), [TASK-31993.21](../../../backlog/tasks/task-31993.21%20-%20Settle-speech-handlers-and-shared-TTS-service-for-live-capture.md).

**Existing ADR context:** [025-shared-stt-artifacts-and-runtime-routing](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md), [033-application-session-state-ownership](../../../backlog/decisions/033-application-session-state-ownership.md). No new ADR expected for this bounded repair. Routine retirement of the existing view-owned service and callbacks, using existing session ownership. Re-evaluate ADR need if background capture or ownership policy changes. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** PDF145–146 / printed136–137 describes widget message ownership; PDF335 / printed326 demonstrates cancellation-aware publication. The foreground capture lifetime follows this application’s exclusive view ownership and disappearing Stop control, rather than an explicit book microphone policy.

### 2. [TASK-36694](../../../backlog/tasks/task-36694%20-%20Handle-each-Speech-Recognition-button-activation-once.md): Handle each Speech Recognition button activation once

One ordinary Start Dictation button event is handled by the child and then manually forwarded to it again by the Speech container. Capture starts and immediately stops. Give handled child messages a single action delivery.

**Witness and limits:** One real Pilot click delivers the same Button event object twice: false-to-true then true-to-false; start_calls=1, stop_calls=1 and the final UI is idle. Existing keyboard-action control starts once and remains listening. The prior slice-41 QA correctness side-find has no matching backlog owner.

**Existing ownership:** The September27 slice41 QA correctness side-find1 (data/items.md:4507) recorded the duplicate dispatch, but is not a numbered finding with a task owner. TASK-33284/PERF25 owns blocking-latency findings and a ≤50ms performance outcome, not this correctness defect. TASK-16266 and TASK-15470 cover view-switch lifecycle and settings flushing. Related: [TASK-16266](../../../backlog/tasks/task-16266%20-%20Harden-Speech-view-switching-and-first-run-test-seam.md), [TASK-33284](../../../backlog/tasks/task-33284%20-%20PERF-25-Other-screens---Evals-Change-Review-Speech-video-player-splash.md), [TASK-15470](../../../backlog/tasks/task-15470%20-%20Config-persistence---batch-writes-and-take-them-off-the-click-path.md).

**Existing ADR context:** [025-shared-stt-artifacts-and-runtime-routing](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md), [031-tui-keybinding-and-footer-hint-conventions](../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md). No new ADR expected for this bounded repair. Routine correction of handled-message delivery within existing Speech routing; no new application structure or keybinding policy. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** PDF145–146 / printed136–137 explains message bubbling and stopping; PDF253 / printed244 uses event.stop() to keep a handled button from reaching its parent.

### 3. [TASK-36695](../../../backlog/tasks/task-36695%20-%20Keep-later-Chunking-Lab-sample-edits-when-an-earlier-file-load-finishes.md): Keep later Chunking Lab sample edits when an earlier file load finishes

A pending sample file read publishes its captured text after later editing has already changed the live sample and canonical session. Preserve the newest accepted sample intent when older asynchronous reads finish.

**Witness and limits:** Real file_operation/sample-file reads a private UTF-8 file through its actual to_thread boundary. While delivery is held, a real keyboard x changes seed to xseed in both editor and canonical session. Releasing the read replaces both with selected file content. Untouched-file control passes. The file-dialog gesture and natural race frequency were not measured. Fresh merged-dev ordinary control passes. The later-edit child records the intended retention assertion failure plus a separate native-worker teardown error; it is not claimed as a clean whole-process regression result. Observational/worker-drain retries are retained as qualified or excluded evidence in the audit.

**Existing ownership:** TASK-31645 is Done and implements ordinary file/clipboard imports and recovery, without a pending-read/newer-edit outcome. TASK-31640 owns recovery and TASK-31642 run lifecycle. TASK-31428 is an older planning task, not a second implementation owner. TASK-36363 protects audiobook authorship under a different owner. Related: [TASK-31645](../../../backlog/tasks/task-31645%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md), [TASK-31640](../../../backlog/tasks/task-31640%20-%20Chunking-Lab-durable-profile-local-session-checkpoints.md), [TASK-31642](../../../backlog/tasks/task-31642%20-%20Chunking-Lab-bounded-local-A-B-execution-lifecycle.md), [TASK-31428](../../../backlog/tasks/task-31428%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md), [TASK-36363](../../../backlog/tasks/task-36363%20-%20Keep-newer-audiobook-source-edits-when-a-pending-file-import-completes.md).

**Existing ADR context:** [118-chunking-lab-local-execution-and-recovery](../../../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md), [033-application-session-state-ownership](../../../backlog/decisions/033-application-session-state-ownership.md). No new ADR expected for this bounded repair. Routine publication applicability repair within the existing Chunking Lab coordinator, sample ownership, validation and recovery contracts. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** PDF145–146 and158 / printed136–137 and149 explains independent queues and awaited handlers. Preserving newer accepted text at an older completion is an application-specific applicability requirement.

### 4. [TASK-36696](../../../backlog/tasks/task-36696%20-%20Keep-pending-MCP-file-reads-from-replacing-newer-import-drafts.md): Keep pending MCP file reads from replacing newer import drafts

MCP file-import completion looks up whichever import panel currently exists and replaces its text. Later typing, or Cancel followed by a fresh import form, can therefore lose its draft to an older file selection.

**Witness and limits:** Actual Import -> From file -> picker Path callback -> real to_thread read. Both same-panel typing and Cancel/reopen replacement retain later-draft before release, then receive older fixture JSON. Untouched-panel control loads normally. Fake hub only supplies catalog; no credentials or network.

**Existing ownership:** TASK-32791 covers workspace-root drafts, TASK-32780/32787 Tool Pack review/write lifetimes, TASK-32839 recovery-review completion and TASK-36361 other Settings drafts. Their complete acceptance criteria do not own MCP server JSON file-import publication. Related: [TASK-32791](../../../backlog/tasks/task-32791%20-%20Preserve-MCP-root-drafts-and-ordered-save-outcomes.md), [TASK-32780](../../../backlog/tasks/task-32780%20-%20Keep-Tool-Profile-reviews-within-the-initiating-Settings-visit.md), [TASK-32787](../../../backlog/tasks/task-32787%20-%20Preserve-Tool-Profile-writes-across-Settings-recreation.md), [TASK-32839](../../../backlog/tasks/task-32839%20-%20Keep-restored-MCP-review-completion-scoped-and-refresh-passive-views.md), [TASK-36361](../../../backlog/tasks/task-36361%20-%20Preserve-newer-Appearance-Storage-and-Console-Behavior-drafts-after-pending-Saves.md).

**Existing ADR context:** [033-application-session-state-ownership](../../../backlog/decisions/033-application-session-state-ownership.md), [240-mcp-destination-retirement-tools-hub-in-settings](../../../backlog/decisions/240-mcp-destination-retirement-tools-hub-in-settings.md). No new ADR expected for this bounded repair. Routine stale UI publication repair within existing import-panel/workbench ownership and the canonical Settings host; no new service or permission boundary. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** PDF145–146 and158 / printed136–137 and149 explains widget queues and asynchronous handlers. Originating panel/draft identity is an application-specific deduction.

### 5. [TASK-36697](../../../backlog/tasks/task-36697%20-%20Reject-malformed-MCP-import-shapes-without-losing-the-editable-panel.md): Reject malformed MCP import shapes without losing the editable panel

The MCP JSON importer assumes args is iterable and env is a mapping. Valid JSON with malformed field containers escapes Preview validation, causing the error keepalive to retire the import panel and its editable draft.

**Witness and limits:** Real Preview of args:42 raises TypeError; env:[not-an-object] raises AttributeError. Valid args:[]/env:{} control passes. Installed keepalive guard keeps the host alive while the failing panel/editor detach. No verified whole-app production crash claim.

**Existing ownership:** TASK-241 remains open for regex alignment, mutation remount, slot confirmation and lifecycle cancellation; its complete criteria do not cover malformed containers. TASK-26013 validates spawn-command shapes, TASK-14922 protocol/packaging. TASK-32533 and TASK-33621.13 keep handler errors alive while retiring failing pumps; they do not validate these JSON shapes. Related: [TASK-241](../../../backlog/tasks/task-241%20-%20MCP-Hub-engineering-follow-ups-from-Phase-2-3-ledgers.md), [TASK-26013](../../../backlog/tasks/task-26013%20-%20MCP-server-spawn-configuration-guard.md), [TASK-14922](../../../backlog/tasks/task-14922%20-%20Resolve-PR-1490-post-merge-review-findings.md), [TASK-32533](../../../backlog/tasks/task-32533%20-%20Library-Notes-the-app-exits-with-an-unhandled-InvalidSelectValueError-during-the-first-timer-edit-journey-and-the-crash-handler-records-no-site.md), [TASK-33621.13](../../../backlog/tasks/task-33621.13%20-%20Console-Choose-folder-for-project-instructions-freezes-the-whole-app-Esc-clicks-and-CtrlQ-ignored.md).

**Existing ADR context:** [161-component-pattern-library](../../../backlog/decisions/161-component-pattern-library.md), [240-mcp-destination-retirement-tools-hub-in-settings](../../../backlog/decisions/240-mcp-destination-retirement-tools-hub-in-settings.md). No new ADR expected for this bounded repair. Routine validation and recoverable errors within the existing importer and host; no new service or security policy. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** Collateral concrete input-boundary defect found while tracing the book’s queued Preview/panel lifecycle model. Container validation is an existing application requirement, rather than a specific book design.

### 6. [TASK-36698](../../../backlog/tasks/task-36698%20-%20Ignore-superseded-Image-Gen-demo-worker-results-and-errors.md): Ignore superseded Image Gen demo worker results and errors

The Image Gen demo uses an exclusive thread worker, but the canceled old worker still publishes success or failure. A slow earlier request can replace the newest image, metadata or success status.

**Witness and limits:** Real Generate clicks admit OLD then NEW. After NEW publishes, OLD has actual Worker.is_cancelled=True; releasing it replaces metadata/image with OLD or changes Done to OLD adapter failure. Serial control retains NEW. This is the live palette-accessible developer demo, not the production media-generation pipeline.

**Existing ownership:** TASK-497 is Done; it marshals status to the UI thread and intentionally defers other demo polish. TASK-34935 offloads image decoding and TASK-36098 owns Image/Video Settings drafts. None owns canceled demo request publication. Related: [TASK-497](../../../backlog/tasks/task-497%20-%20Image-generation-Phase-1-polish-follow-ups.md), [TASK-34935](../../../backlog/tasks/task-34935%20-%20Move-image-gen-demo-PIL-decode-off-the-UI-thread.md), [TASK-36098](../../../backlog/tasks/task-36098%20-%20Preserve-newer-Image-and-Video-Settings-drafts-after-pending-saves.md).

**Existing ADR context:** [176-one-media-generation-core](../../../backlog/decisions/176-one-media-generation-core.md), [033-application-session-state-ownership](../../../backlog/decisions/033-application-session-state-ownership.md). No new ADR expected for this bounded repair. Routine applicability checks within the existing demo worker and media-generation core; no new runtime or provider boundary. Implementation must re-evaluate if the scope changes a boundary.

**Book relation:** PDF335 / printed326 explicitly checks the current worker cancellation state before posting a thread result to the UI. Both success and failure publication must remain applicable to the current demo request.

## Evidence qualification

Seven fresh Speech/Image observation/control cases satisfied their assertions, with all three processes exiting 0 and empty network-attempt/profile-violation records. Seven fresh MCP private children produced 3 passing controls/guard checks and 4 intended product failures, without setup errors or skips. The installed error guard kept the lightweight host alive while retiring the malformed Preview panel; this is not a certified whole-app crash.

Chunking Lab's ordinary merged-dev control passed with clean child teardown. The later-edit child recorded both the desired-retention assertion failure and a separate native-worker teardown error. Its before/after witness establishes the overwrite; it is not a clean expected-red-only whole-process result. The companion JSON and compressed exact child JUnit receipts preserve both conditions. An observation-only retry retained the teardown error; an all-worker-drain retry raised WorkerCancelled and is excluded from the product verdict. No guard was relaxed or production behavior changed.

Physical microphone behavior, full unmodified application boots, natural race frequency, the Chunking native-picker gesture, and future acceptance-criterion cases not explicitly witnessed are outside this evidence. These are audit findings against unfixed code, not passing repair tests. A gated MCP local-profile Save candidate was excluded because its real service path does not suspend; the fake had invented the interleaving.

The [evidence manifest](2026-10-24-textual-book-after-pr3228/manifest.json) retains exact executed probe sources as text, observation records, source hashes, merge/current-head CI evidence, 58-case execution proof and the qualified Chunking child receipts. Source receipts record the original private execution paths; they are not a portable installed test suite. PDF contents, private profiles, databases and temporary trees are not published.

## Duplicate-work and ADR check

The final comparison covers source/topic searches across task/draft/archive records, complete acceptance criteria of 26 related task records, 33 open-PR file lists and exact MCP-overlap patches, plus newer unmerged task-title claims. PR #3156 changes permission suggestions; PR #3148 changes plugin ownership/local profile handling. Neither repairs MCP JSON file-read publication or malformed import Preview. The newest separate claim, TASK-36663, owns focus contrast.

ID allocation separately checked 309 fetched/local branch tips, historical task paths and 52 worktrees, including drafts; current-checkout frontmatter was also read. Six ids were allocated with a 30-id lead beyond 36663, through the Backlog CLI. No existing task is relabeled or closed. TASK-36512's multi-run CI observation and TASK-36514's separate shard/tool proposal retain their existing ownership.

No new ADR is required to file these bounded repairs. Existing decisions are linked per task; any later implementation that changes background-capture policy, storage, recovery, service ownership or another boundary must perform its own ADR check.
