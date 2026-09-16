# Dev 00240 semantic merge review

Read-only review of ours `65a5c152b31e77f33506fceac5ec5921b61ef040`, incoming `00240d0f1f6401726093df4bed2609d87db69d58`, preview tree `270f5c2ab1f30641c4b48e03e4721dda5347eeff`. No repository edits, merges, app tests, network operations, or UAT fixture access.

## Required Notes conflict resolution

Retain the entire upstream `NotesSyncRuntimeOwner.binding_labels` method. Put `@producer_call` directly above both this new method and the existing `compare_conflict`. In `binding_labels`, replace its sole `await asyncio.to_thread(self._store.get_root, root_id)` with `await self._maintenance_offload(self._store.get_root, root_id)`.

The conflict is at preview lines 2372–2453: our side contains the decorator belonging to `compare_conflict`, while upstream inserts a new method immediately before that method. Simply retaining the upstream block loses the existing comparison lifetime. Applying the decorator only to the new method also loses it. The new saved-root read additionally needs the existing cancellation-safe worker tracking.

This follows the neighboring `conflict_labels` contract. `producer_call` retains the accepted async call across awaits and refuses new intake after close. `_maintenance_offload` shields and retains the actual worker in `_maintenance_pending` even if its waiter is cancelled; `_maintenance_drain` waits for both producer calls and pending workers before closing the store cache. A plain `to_thread` waiter can disappear while its native store access continues. Do not replace these protections with `_admit_task` alone.

Keep upstream's exact setup/saved-review lookup, root/state/direction/plan equality, before/after authority checks, bounded typed label validation, observation release, and `_finish_task` cleanup. No new schema, namespace, cache, or lifetime mechanism is necessary.

An in-memory candidate with only this resolution parses. All 17 pre-existing decorated runtime methods remain decorated, `binding_labels` is the sole added decorated method, and `compare_conflict` has an identical AST to ours. Receipt: `/private/tmp/uat-dev-00240-merge-structural.json`.

## Auto-merge assessment

No additional concrete semantic conflict found in the inspected auto-merges.

- Notes keyword creation flows consistently through `NotesSyncExecutionRequest` → executor → `NotesScopeSyncAuthority.create` → `NotesScopeService.create_note_for_sync`. The new keyword parameter defaults to empty, recovery metadata decodes absent keywords as empty, and native local note creation/readback continues through the existing owner. The device-store schema is unchanged. Preserve upstream title/keyword bounds and its intentionally in-memory Obsidian setting behavior; this review does not expand that upstream feature.
- `binding_labels` reads its fresh observation bundle and existing logical folder through the established Notes service. The controller treats unavailable labels as display degradation, but execution retains its existing reviewed authority checks. The new complete async operation must remain visible to backup settlement as above.
- Library changes retain dirty-save refusal and the post-await entry-current check. New target labels and dirty-veto notification do not clear dirty state, bypass save, or grant an alternate destination. Settings changes use checked Select assignment for stale/missing choices while keeping actual save behavior. Wizard changes reuse radio glyph constants; the existing backup entry is retained.
- App upstream exception handling intentionally keeps a failed widget/screen message pump from terminating the whole ordinary app; worker, application-loop, compositor/driver paths retain fatal handling. Backup restart, shutdown, admission and capture code are unchanged. This is a changed production error policy, so headless full-app success alone does not verify it: the upstream dedicated keepalive tests must run. A failed panel may remain unavailable; that is documented upstream behavior, not evidence of a backup authority exemption.
- New persistent diagnostics add bounded identifier fields for raising/site module/function, widget class/id, and numeric line fields, without forwarding exception messages or file paths. Regenerate the diagnostic inventory from the resolved source, including new Notes label failure metadata and app exception-site calls. Selecting either conflicting generated inventory would leave it stale.

All 26 changed product Python files parse when the Notes conflict is resolved in memory. The entire `Backup_Recovery` subtree, native `windows_files.py`, and `notes_device_state_store.py` are byte-identical to ours in the preview.

## Focused verification after resolution

1. Run `Tests/Backup_Recovery/test_merged_service_producer_maintenance.py`. Add or extend its existing Notes lifecycle cases specifically for `binding_labels`: closed admission rejects the call; an admitted label read prevents drain; cancellation while saved-root `get_root` runs retains the worker until native completion. Preserve `compare_conflict` coverage. Existing `test_cancelled_notes_command_keeps_native_worker_owned` is the right fixture pattern.
2. Run upstream `Tests/Notes/test_notes_sync_conflict_runtime.py`, `test_notes_sync_runtime.py`, `test_notes_sync_reconciler.py`, `test_notes_sync_executor.py`, and `test_notes_sync_cutover.py` as practical scoped groups, emphasizing binding labels, stale reviews, Obsidian frontmatter, and resumed create keyword behavior.
3. Run the new `Tests/ProductionApp/test_app_unhandled_exception_keepalive.py` plus `Tests/Utils/test_persist_event.py`; regenerate and verify production diagnostic inventory. Then targeted changed Library review/dirty-veto, Settings Select, and Wizard cases.
4. Use existing fresh installed first-note capture, recovery handoff, and mounted Library capture for aggregate integration. Avoid classifying a late per-test HOME/config rebinding failure as merge behavior: the prior dev698 representative baseline already demonstrated that fixture limitation. Record any such failure separately and use the existing qualified fresh-process infrastructure rather than weakening admission or globally opting tests into an incompatible fixture.

This is a source-resolution recommendation, not an all-tests or installed-platform acceptance claim. No test suite was run during this review, as requested.
