# Console attention connection lifetime — 2026-10-07

The original `ConsoleRuntime.recompute_console_attention` callback reads uncached durable marks on the calling thread. Pending decisions can invoke it on a worker. That original read opened a CharactersRAGDB handle and returned without closing it; later runtime/fixture shutdown could not retire a connection owned by another thread.

A native regression observed the actual original reader's connection and storage lease before its return. The new worker handle remained physically open and registered in both ownership registries. A real pre-existing transaction remained valid, as required. Fixture cleanup occurred only after recording those results; the regression did not replace SQLite or storage admission.

The correction enters the existing `operation_owned_connection` scope for the stock file-backed marks service inside the original attention lock. It spans the original marks and outcome reads. Custom and in-memory services retain their prior path; existing caller handles and transactions remain borrowed. AST reversal confirms that every original query, notification and publication statement is unchanged. Existing ADR-126 ownership rules apply; no new schema or runtime contract is added.

All three native ownership cases (new handle, borrowed transaction, query failure) pass. Forty-three original attention behavior cases also pass. The first original case stopped during fixture setup because its collection-bound configuration lacked the existing `bootstrap_profile` declaration. Only that marker was added; its body is unchanged and its rerun remains pending at this checkpoint. The test-created Runtime now receives its normal disposal so unrelated policy work also retires.

The 57.437-second contained run preserved source/HEAD and exited normally, with zero forced retirement, observer overflow or identity lookup races. Production and new regression lint pass, the new test is formatted, and the original attention body is preserved. The 47-case targeted set is included in Windows/macOS/Linux CI.

The actual original pending-close journey still needs a post-fix run. Prior Linux/macOS diagnostics show the same kind of open worker handle, but this isolated result alone does not prove the complete journey is repaired. Whole startup/Send performance remains open.
