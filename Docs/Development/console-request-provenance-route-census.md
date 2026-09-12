# Console request-provenance route census

TASK-23113.3 owns provider-neutral preparation provenance. TASK-23113.4 owns
adapter-entry sealing and execution-time verification. The frozen executable
census is `CONSOLE_REQUEST_ROUTE_CENSUS` and `CONSOLE_GATEWAY_CALLSITE_CENSUS`
in `Chat/console_trace_provenance.py`; its AST test is intentionally
bidirectional so a newly added or removed direct gateway call cannot silently
escape classification.

Conversation-owned logical routes are fresh send, retry, continue,
regenerate, edit/resend, direct prefill, agent first call, agent tool loop,
citation repair, manual summary, impersonation, automatic compaction, and the
llama fallback. Agent-first and tool-loop calls require actor/chain metadata at
the later execution-sealing boundary. Automatic compaction is conversation
owned even though its auxiliary result has no visible assistant owner.

The explicitly excluded preparation owners are Console side chat, visual
evaluation, and prompt improvement. Persona preview and character generation
are also classified as excluded because the AST census finds their use of the
same gateway API outside a Console conversation.

Fresh requests record either a RAG source descriptor or
`fresh_rag_not_selected`. Retry/continue/regenerate/edit requests do not
silently imply reuse and record `retry_rag_not_replayed` when absent. Agent
wakes record `agent_wake_rag_skipped` when their fresh-wake path skips RAG.

Session-rendered system framing is header-owned by the later slice and is not
duplicated as a history-surface item. RAG and project material remain
provider-only surface intent for later ownership; preparation descriptors
record only source, frozen capture policy, omissions, and structural
transforms. Descriptors never carry semantic text, canonical-message hashes,
provider wire payloads, authority, or tool permission.

No schema migration is part of this slice. Current shadow request writes are
covered here; exhaustive viewer/export ownership remains TASK-23113.6.

The Slice A caller boundary is `admit_preparation_trace_provenance(...)`. It owns
`trace_provenance_admission_transaction(database)` and invokes
`admit_message_provenance(cursor, coordinator, message_ids)` inside it. Revision
work and commit failures both roll back before becoming a content-free manual
pause or autonomous `TraceProvenancePersistenceError`; neither retains the
source exception. The dedicated manual pause exposes Retry, explicit one-shot
Send without capture, or Cancel. Send without capture calls
`admit_one_shot_capture_off` with a new preparation ID and attempt ID. It
returns a fresh `READY` preparation whose execution context carries the new
attempt and whose `one_shot_capture_off` flag requires rebuilding from wholly
Capture-Off inputs; it cannot reuse the failed Capture-On aggregate or policy.
Capture-on callers then supply the frozen run policy and all four parallel
descriptor sequences to `build_console_request`; supplying only some fails
closed with `TraceProvenanceAlignmentError`. The explicit run policy means a
saved-revision-only request can later attach provider thinking, continuation,
or compaction intent without depending on an unrelated system/RAG artifact.
The TASK-23113.4 shadow builder
will wire this API at the controller/gateway dispatch boundary after native
rows have been resolved to durable message IDs. Until that sealing slice,
default preparation remains wholly Capture Off—never a partial Capture On.

## Final provider-value boundary (TASK-23113.3 Slice B)

For a generic provider, the final semantic boundary is the exact keyword
mapping passed by `ConsoleProviderGateway` to `Chat_Functions.chat_api_call`.
The separately verified provider projection is limited to Chatbook-owned
behavior: `PROVIDER_PARAM_MAP`, provider defaults exposed by the dispatcher,
and project-instruction marker preservation for Anthropic/Google or stripping
for every other registered handler. Adapter-owned HTTP internals are not part
of this boundary.

`API_CALL_HANDLERS`, `PROVIDER_PARAM_MAP`, the sensitive auxiliary endpoint
audit, and the settings execution-provider aliases are pinned as one
bidirectional key set. The documented exception is `llama_cpp` and
`local_llamacpp`: Console conversation sends use the direct gateway transport,
so Slice B additionally verifies the literal OpenAI-compatible JSON payload.
Their generic handler entries remain audited because non-Console callers can
still dispatch through `chat_api_call`.

Capture On independently reconstructs the semantic keyword mapping and first
compares it with the exact runtime mapping ephemerally in memory, including
the credential value and bounded credential-decision category. Only an exact
raw structural match proceeds to credential filtering; the credential value
is then discarded and never enters the bundle. There is no lossy-redaction
comparison, digest, retained mismatch value, or diagnostic representation.
The immutable shadow bundle contains only sanitized values, bounded
credential-decision categories, bounded redaction metadata, and content-free
typed omission on failure. A mismatch or sanitizer/sink failure blocks the
Capture-On shadow path before adapter entry. The llama stream-to-completion
fallback repeats this check against its second literal endpoint and JSON body
immediately before that HTTP dispatch.

The pre-existing Exchange inspector remains a separate legacy sink while the
normalized writer is disabled. Its generic and llama request, response, and
tool values pass through the same mandatory credential sanitizer before an
in-memory capture is formed, and capture serialization applies a final
fail-closed gate before `message_exchanges.capture_blob` is written. Safe and
Full still retain ordinary trace content; neither retains credentials. This
slice does not persist the new shadow bundle or enable a normalized writer;
later slices own repository artifacts, headers, surfaces, and dispatch
lifecycle.

Capture Off bypasses shadow construction and preserves the existing final
kwargs and wire payload. `complete_auxiliary()` remains explicitly Capture Off
unless a future API supplies an explicit trace context. Provider-internal
transport retries and the llama stream-to-completion fallback add only bounded
structural overlay/default metadata to the new shadow model; TASK-23113.4 owns
normalized execution-attempt identity and lifecycle. Existing legacy Exchange
rows are unchanged in lifecycle semantics. Canonical transcript storage is
outside this provider-shadow boundary.
