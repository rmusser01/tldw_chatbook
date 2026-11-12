# Library Media highlight CI investigation

Date: 2026-10-28. Investigation only; no tracked product/test/task files changed, no commits, rebases, pushes, or guard changes. Superpowers systematic-debugging was used. The local App slot is released after the two targeted runs below.

## Diagnosis

The test's final highlight assertion can run inside the legitimate asynchronous replacement of Media list children. Its preceding condition proves that editing ended and the Reader DOM has the saved title; it does not prove the sibling Items canvas has published its replacement rows. Native Textual 8.2.8 removes the old children before awaiting the new mount, inside `Widget.batch()`. During that await, a query of the canvas can return zero row Buttons, while the pure/canvas selection state and Reader action target still identify `local:media:3`.

A controlled hold at that exact native mount boundary reproduced every relevant CI precondition, including the final `[]` result, and the cue returned after release without a product change. Rendering was batched throughout the empty-DOM interval. The last render-eligible compositor frame retained `▸ Meeting notes` and its `loaded` line; the next published frame showed `▸ Meeting notes (edited)`. This supports a test observation gap. It does not demonstrate a user-visible selection gap or a persisted target-identity defect.

The original CI execution was not instrumented, so attributing that particular execution to this interval is an inference from the matching assertion, source order, and controlled native witness. No natural frequency is claimed.

## Failure and original contract

- Head: `012ca47c39a16c10aa9b9b0a2298287945e22a89`; dev baseline: `798f8fca75a7acba56da5f523329efa52c4a78f0`.
- CI run `40744955216`, UI shard 11, job `123016160566`: `Tests/UI/test_library_media_actions_target_opened_item.py::test_read_later_and_edit_save_target_the_opened_item_not_row_1`, line 250, `[] != ['local:media:3']`. Save target 3, saved title, Reader title, and loaded identity passed earlier assertions. Shard result: 1 failed, 168 passed, 2 warnings, 729.01 seconds; this was an assertion failure, not a timeout.
- Preserved CI log: `.test-tmp/textual-37031-37034/current-head-012ca47c-watch/logs/123016160566.log`.
- [TASK-34000.18](../../../../backlog/tasks/task-34000.18%20-%20Library-Media-Read-later-trash-and-Edit-metadata-act-on-the-item-opened-by-link-not-row-1.md) AC1 requires the opened item's list highlight to stay there **after the list refresh lands**. AC2 requires Reader-scoped actions to act on the shown item. AC4 requires the Reader to keep that item. The controlled witness preserves these contracts.
- [ADR-084](../../../../backlog/decisions/084-library-media-reader-ia.md) distinguishes selected and loaded identities, rejects stale detail completion, and keeps the permanent Reader beside Items. [ADR-055](../../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md) provides the existing common mutation/interlock seam referenced by the save path; this investigation does not change it.
- [Library architecture](../../../../Docs/Architecture/library-shell.md), [design language](../../../../backlog/docs/design-language.md), [component patterns](../../../../backlog/docs/component-patterns.md), and the testing/live-verification lessons were consulted. The testing lesson “Settled data and settled workers can still leave the authoring surface stale” specifically requires distinguishing state, mount, and paint boundaries.

## Source trace

1. `library_screen.py:33740` resolves the edit target with `_library_media_reader_target_id()`. `:33819` writes through the service with the numeric backing id. `:33861` clears editing, then `:33865` awaits `_refresh_library_media_detail(media_id)`.
2. Detail completion stores and settles the Reader identity at `library_screen.py:19907–19918`, then `:19986` queues `_recompose_library_media_detail_if_unrendered` on the next callback. Reader synchronization and the sibling Items publication run through separate pumps.
3. The save's `finally`, `library_screen.py:33867`, calls `_complete_library_media_mutation`. That method reconciles retained items and starts a new page request at `:17020` plus facets at `:17021`.
4. `library_media_browse_controller.py:338–349` marks loading and requests a canvas sync before starting the page worker. `_apply`, `:478–499`, stores the accepted result and clears `loading` **before** calling its view synchronization. Thus even `loading=False` and `applied_scope == requested_scope` are not a DOM-publication receipt.
5. `_sync_library_media_browse_state`, `library_screen.py:17250–17251`, builds/adopts the correct cue, then `:17355` invokes `_sync_library_canvas(..., 'media')`. `LibraryMediaCanvas.sync_state`, `library_media_canvas.py:613–628`, stores that display model and schedules `refresh(recompose=True)`.
6. Installed native `textual/widget.py:1704–1716` runs `async with self.batch()`, removes existing children, and then awaits `mount_all(compose_nodes)`. `batch`, `:4415–4430`, holds the widget lock and `App.batch_update`. `textual/app.py:3832` refuses display while `_batch_count` is nonzero, and `Screen._on_timer_update` also checks that batch count.
7. The test condition at lines 243–248 observes only `not editing` and the Reader title; line 250 immediately queries row Buttons. By comparison, this same file's `_open_by_link` already waits for the accepted page **and** five mounted rows before making its opening assertions.
8. `Widgets/Library/library_canvas_sync.py` documents the same independent-pump ordering issue and provides callbacks after native recompose finishes. No additional publication system is needed to repair the test.

## Reproduction and receipts

Both commands used the required launcher, fresh native 0700 run/profile/temp directories, the normal profile/path guards, real Textual App journeys, and targeted selection only. There were no setup/teardown errors or `raw_source_selection_changed` refusals. Each verification receipt lists one actual JUnit report and no private child reports.

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 '.test-tmp/textual-37031-37034/run-targeted-external-profile.py' 'media-highlight-original-20261028-01' 'Tests/UI/test_library_media_actions_target_opened_item.py::test_read_later_and_edit_save_target_the_opened_item_not_row_1'

& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 '.test-tmp/textual-37031-37034/run-targeted-external-profile.py' 'media-highlight-held-20261028-01' '.test-tmp/textual-37031-37034/media-highlight-probe.py.txt::test_native_recompose_observation_gap'
```

| Run | Result | Actual report SHA256 | Native profile root |
| --- | --- | --- | --- |
| `media-highlight-original-20261028-01` | Exact unchanged node: 1 passed, 0 failures/errors/skips; 62.20s session, 25.93s call | `76155c205efc980bc3fc33da13b2238ad356e17380aaf6a83dca4bb08fa83d22` | `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-c09e1884cec24a3a8208395e060b2769` |
| `media-highlight-held-20261028-01` | Diagnostic: 1 passed, 0 failures/errors/skips; 51.32s session, 31.20s call | `b1978a3d6bfcb08965e7a92a06003dfbd07fdadaf7b21089bf386a355e10edaf` | `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-cf1a6d65eadb4dc5b90798a649aaaf9c` |

Actual XML files are `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui/.test-tmp/textual-37031-37034/<run-label>/result.xml`; `command.json`, `run.log`, and `verification.json` remain beside them. Native 0700 directories require native/escalated reads; they were not weakened for convenience. The diagnostic is a passing observation test, not an expected-red product regression.

The probe executes the exact original journey source, inserting only an observer binding and replacing the final cue assertion with the boundary observation/release check. It holds the real native `Widget.mount_all` await only for the Media canvas after the real fake service has recorded an update. It releases inside the App context before teardown. It does not replace the save, list search, pure builder, action routing, native recompose/removal, or profile guards.

At the hold, [observation-01.json](media-highlight-observation.json) records:

| Observation | Value |
| --- | --- |
| DOM row ids / highlighted ids | `[]` / `[]` |
| Reader title / editing | `Meeting notes (edited)` / `False` |
| page loading / applied equals requested | `False` / `True` |
| selected / loaded / Reader action target | `local:media:3` for all three |
| pure-state / canvas-state highlighted ids | `['local:media:3']` for both |
| native App batch count | `1` |
| render-eligible frame count before / during hold | `66` / `66` |
| Read later service calls | save 3, remove 3 |
| Edit Save service call | media id 3, title `Meeting notes (edited)` |
| original final assertion would pass | `False` |
| highlighted ids after native release | `['local:media:3']` |
| selected / loaded after release | `local:media:3` for both |
| render-eligible frame count after release | `70` |

Frame evidence: [before hold](media-highlight-frame-before.json), SHA256 `716587daea44be6cb7f5c3db5c616ba33ee2543dc74e5fd8f5a0b00bb282d1f0`; [after release](media-highlight-frame-after.json), SHA256 `63f5d23d035015397f8671ae0b0f7a1ed975e905832a2518b19bc226e0599595`. The observer captures compositor content passed to the App display seam only while batch count is zero. This is a headless `run_test` witness of render eligibility and batching, not a claim of a physical terminal capture. Native Textual suppresses screen updates during the held batch.

Full parsed XML/path/frame/probe hashes: [media-highlight-diagnostic-receipts.json](media-highlight-diagnostic-receipts.json). Frozen probe source: [test_media_highlight_probe.py](media-highlight-probe.py.txt), SHA256 `d8f839294bcdbbb0f70a128cc7a49163eee230475f8ffe32beb9ace7302e2455`.

## Baseline comparison and source custody

[media-highlight-source-receipts.json](media-highlight-source-receipts.json) records raw working SHA256 and HEAD/dev Git blob ids for ten files covering the exact journey/harness, screen, media/browse controllers, canvas sync, canvas/mixin and pure media/Reader state. Every HEAD blob equals dev `798f8fca75` and every tracked working diff is empty. This is a pre-existing timing gap in these sources; there is no evidence that the audit introduced the identity defect described by the original task. Changed shard scheduling can expose an existing timing interval without changing the product contract.

Representative raw working source SHA256:

- failing test: `c143202062cab5e7381f96b0fbb5367b162d2fef5db54854237f46ab195bd5ee`
- shared harness: `24396eeb7956b84ce37c63944fc188d5919747fef64c208cc1fa2fc6256218a8`
- Library screen: `abc900fa3a5fe315ed1b571037745b4368cf747e9a3d90369ea72f0b42eb9a86`
- Media controller: `8c969a66d373f3d24a6f8f4fb0a1cd60553edddee72d6665cf1dd92f81db19a2`
- browse controller: `edcd715b209afc64436cba8b73771f76db35095d8afe7b992874453bb7509d69`

## Minimal recommendation

Use a test-only publication condition before the existing final cue assertion. Extend the post-save wait to require the accepted page and its current replacement rows to be present/attached (five fixture rows, including the opened row's edited title), along with the existing Reader title/editing conditions. Keep the final `highlighted_ids == [canonical]` assertion separate from the wait so that a populated list which genuinely loses its cue still fails. Do not merely wait on controller flags: the controlled witness already had `loading=False` and matching scopes while no row Buttons existed. Do not add an arbitrary pause.

Qualify the proposed test change with this same native hold: prove its publication predicate remains false during the empty-DOM interval, then true after release, while the unchanged final cue assertion passes. Add a negative control that publishes the rows but removes the selected cue; publication should complete and the final cue assertion must fail. This preserves the original identity/reader assertions and discriminates a broken cue from legitimate asynchronous replacement. A short targeted run of this node and its immediate file neighbors is sufficient; no full suite is authorized.

ADR required: no. ADR path: N/A (existing ADR-084/055 apply). Reason: a routine test synchronization correction preserves existing product boundaries and TASK-34000.18's accepted behavior. Root should allocate/update any needed narrow Backlog task before tracked implementation per repository policy. No task status or acceptance criteria were changed during this investigation.
