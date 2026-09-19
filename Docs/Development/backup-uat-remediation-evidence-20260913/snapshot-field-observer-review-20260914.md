# Snapshot changed-field observer independent review

Approved after the primitive-reason correction. No remaining actionable findings in the two-file diagnostic scope against `29baddc035`.

The initial helper compared `ValueError.args` before validating an exact string payload, which invoked a custom equality callback. A private probe reproduced this (1 failed, 1 passed; `/private/tmp/uat-snapshot-fields-private.log`). The final helper checks built-in exception/string types and argument count before equality; the author also promoted an observer-level invalid-payload regression. No product files were changed by this review.

The extraction uses only retained traceback/local snapshots. It performs no stat, open, SQLite, or other native reads. Its output is limited to fixed main/WAL member labels, three fixed comparison phases, and five field names; no paths, database content, exception text, identifiers, or numeric field values are emitted. Trace traversal remains bounded to 64 frames and records to the existing eight-record limit. Windows native `WindowsOS._stat_handle` returns the exact stdlib `os.stat_result` type accepted here.

Requiring the innermost traceback frame to match the original snapshot code excludes helper failures. In the actual copy loop, a WAL opening mismatch is established before considering retained main-copy locals. Growth uses the current reset count; successful previous iterations cannot leave unequal observed pairs. Absent WAL, reuse, and equal retained final-state comparisons yield no attribution. This is intentionally incomplete attribution, not a claim to identify a writer or every failure phase.

Independent final verification: **25 passed, 40 deselected in 3.43 seconds** in `/private/tmp/uat-snapshot-fields-final-independent.log`: 23 selected repository cases plus two private probes. The six native main/WAL races cover opening state, growth during copying, and post-copy metadata changes; they assert private-copy cleanup. Existing and extended tests retain original exception object/terminal traceback, successful return, observer restoration, cancellation and metadata-failure precedence. Private probes additionally check invalid-payload equality is never invoked and valid retained locals do not attribute helper/equal-final-state failures.

No new deadlines, guards, native operations, archive behavior, or cleanup behavior are introduced. Native Windows verification remains pending; these macOS results do not identify which field changed in the historical Windows failure.

Exact SHA-256:

- `Tests/Backup_Recovery/thread_diagnostics.py`: `1194abc1b4351c53bb2910e1d05f5cab34b94d8f5f7d26e59553a0359b4767cb`
- `Tests/Backup_Recovery/test_thread_diagnostics.py`: `55d303f1ce6493662dc5dfea2e9cddfe4b4b9a8abc0f48947bfff206e3627683`
