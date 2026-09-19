# Dev f10a merge semantic review

Read-only review: ours `d4e1d0dff80fbabe92e3cc2f59dba7c2d10d599c`, incoming `f10a3fa36471ade188a002dc119bf19db54c4d83`, preview `ba1c39127b131a4169cf57e364b6b5239478943a`. No repository edits, merge, app tests, remote operations, or UAT fixture access.

## Notes resolution required

The runtime conflict is another insertion immediately before `compare_conflict` (preview approximately 2536–2608). Preserve the original `@producer_call` immediately above `compare_conflict`. Retain upstream's new `RuntimeReceiptLabel`, `RuntimeWriteReceipt`, adapter receipt projection, `_read_receipt_labels`, and public `write_receipts` behavior. Preserve the previously resolved `binding_labels` decorator and tracked store read.

The receipt feature needs the following narrow lifetime integration before accepting the merge:

1. Put `@producer_call` on public `write_receipts`. Its whole accepted call must remain visible to global backup maintenance, including its first read and empty-history return.
2. Register the entire public request with the existing `_require_cutover` / `_register_task` / `finally: _finish_task` pattern before its first store read. Do NOT use `_admit_task`: a normally paused root must still expose completed receipts. Global backup maintenance and root pause are different gates. Upstream currently registers only inside `_read_receipt_labels`; an empty history never reaches that registration, and nonempty histories have an initial unregistered store read. `settle()` and ordinary shutdown inspect `_active_tasks`, not `ProducerLifetime`.
3. Move that registration responsibility out of the private helper when registering the whole public call. Nested same-task registrations use a set, not a reference count; an inner `_finish_task` removes the outer membership. The private helper is called only from `write_receipts` in this tree. Keep its cutover/root identity proof if desired; do not grant a new public bypass.
4. Route owner `list_completed_operations` and `get_root` through `_maintenance_offload`, as neighboring native store reads do. Preserve `limit` validation, keyword passing, order, label mapping, and return values.
5. Protect the NEW adapter `asyncio.to_thread(self._store.list_bindings, root.root_id)` too. A decorator on the awaiting owner alone does not retain this thread after cancellation. A small explicit private callback seam is sufficient: pass `offload=self._maintenance_offload` into `build_receipt_labels`, update the private protocol/production adapter and narrow test adapters, and use it only for this store read. Do not replace it with synchronous event-loop IO, a generic new task framework, or an owner-wide rewrite of unrelated existing adapter reads.

Items 2 and 5 are semantic integration requirements beyond deleting conflict markers. The exact implementation should receive focused cancellation/shutdown review before merge; this report is not approval of an unimplemented callback seam. Existing `_maintenance_pending` behavior and store native close protections must remain intact. In particular, test cancellation followed by shutdown as well as maintenance; do not assume backup-drain coverage alone establishes ordinary shutdown safety.

## Upstream behavior to preserve

- `observe_root` now uses `_root_signatures.setdefault` instead of assigning every observation. `changed_root_ids` remains the component that advances the watcher baseline. This prevents manual Check from consuming a disk-change signal before the automatic writer sees it. Keep this change: the signatures are watcher hints, not native admission/identity qualifications. Existing fresh discovery and per-item reuse validation remain unchanged.
- `NotesDeviceStateStore.list_completed_operations` is a bounded parameterized SELECT through the existing `transaction()` API, returning new in-memory records. No schema, table, migration version, physical storage type, path selection, connection registration, or native cache implementation changes in the auto-merge.
- Receipt label projection tolerates only the existing typed `note_missing` condition; other authority failures propagate. It reads existing bindings and note owners, not a new namespace. Preserve root identity checks and the distinction between completed receipts and reviewed future-write labels.
- Library adds action-specific footer labels and calls `refresh_receipts` on opening sync roots. The controller retains actual runtime status/action and overlays only failure display labels, avoiding accidental re-enabling of unsupported controls. Receipt read errors are metadata-only and per-root; the list remains open. There is no change to backup shutdown/unsaved-work checks or native owner declarations.

## Other conflicts

Regenerate `Docs/security/production-diagnostic-inventory.json` against the fully resolved source. New controller/screen receipt failure logs and root-refusal diagnostics must be included; do not choose either generated side by count.

Only the lessons conflict block was inspected (preview lines 14327–14373). Our side of that block is empty; incoming adds two independent lessons about producer-driven assertions and duplicate dict keys. Retain both incoming additions while preserving all auto-merged existing lessons. No need to inspect or rewrite the whole document.

## Minimum useful verification

- Extend `Tests/Backup_Recovery/test_merged_service_producer_maintenance.py` with receipt closed/admitted/resume cases, including a normally paused root that remains readable after global maintenance resumes.
- Parameterize native worker cancellation at EACH new store boundary: `list_completed_operations`, helper `get_root`, adapter `list_bindings`. Block the actual native read after connection acquisition, cancel the waiter, and prove maintenance cannot retire it early; after release, prove normal close/resume. Also cover ordinary shutdown while the initial/empty-history read is in flight, plus cancelled-read shutdown safety.
- Run existing `Tests/Notes/test_notes_device_state_store.py` completed-operation ordering/bounds and unchanged schema tests; `Tests/Notes/test_notes_sync_runtime.py` receipt naming, paused receipt, watcher/manual-check behavior, and existing shutdown/cancellation cases. Run the existing producer-maintenance file after the resolution.
- Run the upstream controller/root-state/canvas tests (`Tests/UI/Library_Modules/test_library_notes_sync_controller.py`, `Tests/Library/test_library_notes_lasting_sync_state.py`, `Tests/Widgets/Library/test_library_notes_sync_roots_canvas.py`) and targeted new roots journey tests using qualified fixture infrastructure. Do not confuse pre-body raw_source_selection_changed fixture failures with merge behavior.
- Regenerate/verify the diagnostic inventory; obtain fresh installed Library capture/normal Notes sync and recovery-handoff evidence separately, without concurrent app suites during keyboard UAT.

No app tests were run, and no aggregate/native-platform acceptance is claimed. The new receipt-worker lifetime is the concrete remaining merge integration issue; no storage qualification relaxation is needed.
