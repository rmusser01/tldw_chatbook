# Speculative voice durable-owner inventory

This is the qualification inventory for ADR-098's rule that an obsolete voice
attempt may leave content-free counters, but may not retain transcript, response,
audio, tool, citation, approval, replay, or diagnostic content. The Task 12A probe
uses a distinct user and assistant poison sentinel, cancels the attempt before the
effect barrier, and searches every owner below. Each probe also performs a control
write with a different sentinel so an empty or disconnected search cannot pass.

| Owner ID | Concrete owner and backing surface | Writer/admission seam | Qualification probe |
|---|---|---|---|
| `console_messages` | `CharactersRAGDB`: `conversations`, `messages`, `console_trace_semantic_revisions`, and every related ChaChaNotes table | `ChatPersistenceService.commit_completed_voice_pair`; ordinary accepted-turn persistence | Search every non-FTS SQLite table/value in the isolated ChaChaNotes file; prove a control message is found. |
| `terminal_marks` | `CharactersRAGDB.conversation_local_marks` | `ConversationLocalMarksService.set_console_terminal_with_cursor` inside the winning-pair transaction | Search mark keys and timestamps; prove a control mark is found. |
| `trace_capture` | ADR-097 `console_trace_*` tables, `message_exchanges`, and `ConsoleProviderGateway`'s provisional voice registry | `ConsoleVoiceTracePromotionGateway.promote` and winning-only exchange import | Search SQLite and live provisional envelopes; prove a control provisional envelope and capture row are found. |
| `provider_usage` | `VoiceAttemptSnapshot.usage_payloads` while provisional; winning assistant `messages.usage_json` and imported trace usage events when durable | `VoiceAttempt` signal collection followed by winning-pair/trace promotion | Inspect the cancelled lifecycle and all SQLite values; prove a control usage row is found. Content-free discarded-attempt totals are allowed. |
| `tools_approvals` | `ConsoleChatController`/`ConsoleAgentBridge` run state, parked approval maps, dispatch checkpoints, and trace tool events | `AttemptToolRequested` effect barrier, then ordinary accepted-turn submission | Assert no ordinary handoff, tool execution, pending approval, checkpoint, or tool event; prove injected control state is visible to the probe. |
| `citations` | `rag_citation_traces`, `rag_message_trace_owners`, answer/source rows, and trace citation references in ChaChaNotes | ordinary accepted-turn citation creation; speculative dispatch sets `requires_citation_creation` and stops at the effect barrier | Search every SQLite table and any staged citation collection; prove a control citation row is found. |
| `notifications` | Console terminal activity/notification callbacks and `NotificationPresentationStore`; no speculative-attempt-specific durable writer is permitted | accepted terminal outcome and app notification dispatch | Inspect captured notification payloads/presentation records; prove a control notification is found. |
| `replay_trajectory` | `messages.provider_continuation_json`, `messages.thinking_blocks_json`, `message_trajectory_metadata`, ADR-097 trace events, and live trajectory/run snapshots | accepted ordinary run or winning trace import | Search SQLite plus live replay/trajectory snapshots; prove a control trajectory entry is found. |
| `chatbook_exports` | Conversation/chatbook, exchange, and trajectory serializers; these are projections over store/DB state and must not become independent attempt owners | export functions only, after durable/read authority | Serialize the isolated store through each applicable projection and search output; prove a control saved message appears. |
| `temporary_chat` | `ConsoleChatStore` in-memory session/message graph and temporary-chat save-later promotion | `publish_temporary_voice_pair` only for a winner; `promote_ephemeral_session` later serializes the visible graph | Search live messages and a save-later projection; prove a control temporary message is found. |
| `persistent_files` | configured private rotating application log and `persistent_diagnostics` metadata-only sink | `Logging_Config` handlers; `persist_event` is the only Chatbook persistent diagnostic admission | Search every isolated log generation; prove a schema-admitted control diagnostic is found. |
| `runtime_logs` | stdlib handlers, Loguru sinks, terminal output, and queued `RichLogHandler` records | standard logging/Loguru routing | Capture every handler/sink buffer and stderr/stdout; prove a control log is found. |
| `in_app_logs` | Logs-screen `RichLog` buffer and any share/copy-log serialization derived from it | `RichLogHandler` queue and the Logs-screen share action | Search the buffered and serialized text; prove a control log line is found. |
| `exception_diagnostics` | task exceptions, event-loop exception handler, crash/traceback rendering, and shutdown diagnostics | content-free error-class events; exception bodies from provider/TTS work must not be surfaced or persisted | Capture exception representations, traceback text, and loop-handler contexts; prove a control exception is found while cancelled poison is absent. |

## Interpretation

- The provider and TTS necessarily receive provisional content. Their remote retention
  policies are outside local durable-owner qualification; local response objects and
  tasks are fenced and released.
- PCM, rolling STT windows, and rendered references are memory-only resources owned by
  the view-scoped session. The probe verifies they are closed but never writes a
  control audio payload to disk.
- Metrics may retain bounded counts, timings, provider/model identifiers, AEC state,
  and cancellation outcomes. They must not retain transcript, response, PCM, paths,
  credentials, or exception text.
- A temporary chat is capture-ineligible for speculative dispatch. Saving it later
  cannot retroactively manufacture a trace for an earlier provisional call.
