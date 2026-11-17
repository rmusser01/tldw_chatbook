# Accepted original process-turn test extension

Artifact-only incremental proposal for `Tests/Chat/test_console_execution_backend_process.py`, based on SHA256 `39d316b182224a64e30c1ebfe0902fd27eb69d39a94030ea63b92cac34584dd7`. No live edit or execution. Existing ADR-126 and Task62 accepted-process-turn contract apply.

The existing projection-reference/pre-ACK cancellation node gains a real loopback OpenAI HTTP/SSE endpoint and a second, explicitly ACKed turn. The fixture derives the original HTTP handler/server conventions from `test_console_refused_stream_retry.py`, runs only inside the actual private-profile child, and closes/joins listener and request owners. The original provider selection must resolve to this loopback base before any send. Requests are witnessed, with zero requests before and after first-turn cancellation and before second-turn ACK. Provider/gateway/control implementations are not replaced.

Original independent process test source is unchanged. The first turn's submit/plan/cancel assertions are source-identical except the additional zero-provider-request assertion. Its existing `_until` default bound, observer/watchdog rules, stderr diagnostics, and original 30-second shutdown remain unchanged. The second turn also uses `_until` without increasing its bound. No speed assertion is added; separate parent monotonic plan and terminal diagnostics are recorded.

Integration prerequisites were explicitly confirmed with the central owner and are not yet treated as released runtime APIs:

- After `await client.read_projection(sid, incarnation)`, `client.projection_view.session(sid)` supplies the current original row/settings/draft revision.
- `client.pressed_inputs` supplies detached current input facts from that same retired backend publication. Its initial authored token may be `None`; after public `UpdateInputs`, the actual tuple must be published.

The new UpdateInputs supplies a test-owned parent authored identity `[2, 1]`. All backend settings/attachment/prefill/draft revisions come from actual publication/results, never inferred increments. The test rereads after UpdateInputs, confirms the actual authored identity and accepted revision, then passes the published pressed-input values verbatim to ReceiveTurn. CommitPlanAck uses the exact retained event's request, turn, ordinal, and complete fingerprint; the potentially-committed flag is asserted immediately after transmission is queued.

After positive original turn retirement, the test requires an actual streamed HTTP request containing the submitted user draft, completed original USER/assistant rows, original persisted IDs matching the commit fingerprint, and a completed run. After the original contained process shuts down, it reopens the original ChaChaNotes repository on the creator thread and verifies the exact saved conversation/two-message content, roles, parent link, and completed assistant generation state. New DB cleanup preserves a preceding verification failure and records a secondary cleanup error.

The first-plan path was failing before this extension (a7). This proposal neither repairs nor qualifies that path. Central integration must first correct it and implement the two publication APIs. Missing APIs or unsupported original outcomes fail visibly; no fake records, guessed counters, empty-frame fallback, or synthetic retirement is supplied.

Thirteen source dependencies and the exact baseline are saved and pinned. No Python/import, lint, native launch, tests, or extra component matrix ran. Central owns merging the incremental diff, static validation, and actual execution.

Manifest SHA256: `b8bd3466a5cf23af2f197a0654bb0629bce4cd039dd9cbec6a39a8650cce6d29`.

Proposed test SHA256: `3834dc8dce9f2b13a5e40fdea85a2cb45203b4fac32ce0505e8b5ede181e809f`.
