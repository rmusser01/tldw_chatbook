# Final f10a Notes merge resolution review

APPROVED for the reviewed receipt lifetime and shutdown integration. No actionable source finding. This is a bounded integration verdict, not aggregate merge or Windows acceptance.

Reviewed working source against checkpoint `f54a5a630c` and incoming dev `f10a3fa36471ade188a002dc119bf19db54c4d83`. No repository edits or app-booting tests performed.

## Source findings

`write_receipts` now enters the existing producer lifetime and registers the entire request before the first store read, including empty histories. It checks cutover once at entry without requiring the root to be unpaused. The private projection helper does not remove outer task membership or recheck admission after an already-accepted request has begun. Root identity remains checked.

All three new store reads use the owner's existing tracked/shielded offload mechanism: completed operations, root record, and adapter binding rows. The adapter's required typed callback is passed explicitly by the owner; it does not introduce another registry, lifetime framework, connection closer, or namespace.

Ordinary shutdown closes intake and joins existing tasks as before, then waits for the existing `_maintenance_pending` set before adapter or store retirement. This closes the demonstrated cancelled-waiter gap. It does not force-close a still-running worker, bypass foreign-owner checks, change native deadlines, or interrupt an accepted receipt halfway through its stages. Gathering exceptions retains existing offload error-consumption behavior; the original caller still receives its failure/cancellation.

AST comparison confirms that methods differing from BOTH parents are exactly the private protocol/production adapter receipt signatures, `_read_receipt_labels`, `write_receipts`, and `_shutdown_once`. All other methods match one parent. Existing `compare_conflict` and previously resolved `binding_labels` are AST-identical to ours, retaining their producer decorators. Upstream watcher baseline behavior is retained.

## Independent evidence

`Tests/Backup_Recovery/test_merged_service_producer_maintenance.py`: **36 passed in 14.11s**, exit 0. Log `/private/tmp/uat-dev-f10a-resolution-independent.log`; explicit isolated basetemp `/private/tmp/uat-dev-f10a-independent-fixtures`.

The 16 receipt read combinations cover empty/nonempty initial reads, root and binding reads, with/without cancellation, under both maintenance and ordinary shutdown. Each waits at an actual native store connection, proves closure cannot finish early, explicitly proves the worker connection survives until release, then proves retirement closes it. The two paused-root cases preserve ordinary receipt access, refuse global backup pause, and restore access after resume. Existing foreign-borrower refusal cases remain green. No receipt-content claims are made from fake note persistence; production receipt projection and real store/native maintenance are exercised.

The provisional source RED log `/private/tmp/uat-dev-f10a-receipts-red.log` records 16 failures / 2 passes in 5.11s. The passed provisional cases are retained, not relabelled RED.

Additional upstream selection: **2 passed / 2 failed in 1.99s** at `/private/tmp/uat-dev-f10a-upstream-independent.log`. Completed-operation bounds/order and pinned historical receipt preservation pass. The real-root watcher and receipt tests fail at initial `local_notes.note['content'] == 'new'` (actual `old`), before their new feature assertions; setup emits `raw_source_selection_changed` throughout sensitive-path qualification. No baseline-equivalence claim is made from this run. These two tests do not provide upstream functional acceptance; root's qualified installed/native integration remains necessary. No guard or fixture was changed to make them pass.

## Frozen hashes

- `tldw_chatbook/Notes/notes_sync_runtime.py`: `54a4567a32b55a27fa0d366f9b80d5b91bc9b84f453dcf86738a750f192c8b8e`
- `Tests/Backup_Recovery/test_merged_service_producer_maintenance.py`: `2056c7cea505fc8deba94c51eab8e6c32abad6a54be4501e9eb4118130d23dc4`

Machine receipt: `/private/tmp/uat-dev-f10a-resolution-review-hashes.json`. Static checks and regenerated diagnostic inventory verification remain root-owned; this review did not modify or independently regenerate them.

Final test hash acknowledgment: the sole subsequent edit is a B106 suppression comment on the synthetic journal observation ID. Removing that exact comment reproduces the independently tested file SHA-256; no behavior changed or rerun was needed. Root separately reports final 36 passed in 13.31s and zero-new static findings.
