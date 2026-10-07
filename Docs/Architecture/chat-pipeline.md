# Chat pipeline: streaming, transforms, swipes, and attachments

This document describes the Console's chat mechanics below the send lifecycle: provider streaming, per-send transforms (skills, chat dictionaries, world info), branching and swipes, tool-call presentation, attachments and vision, and the failure paths. For turn orchestration and approvals see [console.md](./console.md); for the agent loop see [agent-runtime.md](./agent-runtime.md).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Transcript widget | `Widgets/Console/console_transcript.py` | `ConsoleTranscript`, `ConsoleMarkdownMessage`, `ConsoleToolDiffRow`, `ConsoleRoleplayMarkdown` |
| Composer | `Widgets/Console/console_composer_bar.py` | `ConsoleComposerBar`, `DraftChanged`, `ConsoleDraftStash` |
| Provider gateway | `Chat/console_provider_gateway.py` | `stream_chat()` — the async streaming seam; `ProviderToolCalls`, `ProviderThinkingDelta`, `normalize_provider_response()` |
| Turn engine | `Chat/console_chat_controller.py` | `_stream_assistant_response()`, `regenerate_message()`, `edit_and_resend_message()`, `retry_message()`, `_apply_chat_dictionaries()`, `_apply_world_info()` |
| Store | `Chat/console_chat_store.py` | `append_stream_chunk()`, `create_sibling()`, `siblings_at()`, `set_active_leaf()`, `add_variant()` |
| Attachments | `Chat/attachment_core.py` | `PendingAttachment`, `vision_block_reason()`, `max_history_images()`, `image_content_parts()` |
| File extraction | `Utils/file_handlers.py` | `ProcessedFile`, the `FileHandlerRegistry` (image/text/code/data/pdf/document/ebook/plaintext-db/default handlers) |
| Tool activity | `Chat/console_tool_activity.py` | `ConsoleToolActivity` — tool chips in the transcript |
| Message actions | `Chat/console_message_actions.py` | `ConsoleMessageActionService.dispatch()` (regenerate, variant browse, …) |

## Streaming flow

One `stream_chat()` invocation = one provider call. The gateway runs the provider handler in a worker thread, bridges results into an async queue, and yields text chunks, thinking deltas, or a terminal `ProviderToolCalls`. The controller's streaming loop routes:

- **thinking deltas** → `store.replace_message_thinking`;
- **text chunks** → `store.append_stream_chunk` (buffers, stamps trajectory first-token time, runs the character-emote parser inline, and drops late chunks silently if the message was already stopped);
- **usage** → attached at completion;
- **completion** → thinking settled, run state COMPLETED, message-completed subscribers fire, trajectory sidecar flushed.

The transcript re-renders via the Console's 0.2 s poll with O(delta) diffing (see [console.md](./console.md)). Non-streaming responses are normalized by `normalize_provider_response` — a full mapping is yielded as a single chunk, and synthetic fallback copy (no content / unsupported shape) is flagged so usage and trace never mislabel it as provider output.

`Chat/Chat_Functions.chat()` remains only for **non-streaming** callers (CCP, media analysis, library, evals); its worker wrapper raises `ValueError` on `streaming=True` — streaming is exclusively owned by the Console gateway.

## Per-send transforms

Before dispatch, the ephemeral final user message of the provider payload passes, in order:

1. **Skill substitution** (`/`-prefixed flows),
2. **Chat dictionaries** (`_apply_chat_dictionaries`, constants max 500 tokens, strategy "sorted_evenly"),
3. **World info** (`_apply_world_info`, deliberately after dictionaries), gated by `[character_chat] enable_world_info`.

All three mutate only the ephemeral provider payload — **never the stored transcript** — and swallow non-cancellation exceptions (returning the payload unchanged). See [character-chat.md](./character-chat.md) for the libraries behind 2–3.

## Branching and swipes

- **Regenerate** forks a **sibling node** under the same parent (`store.create_sibling`): the anchor and its old tail stay stored but drop off the active path; the new sibling streams as a fresh node. Validation happens before any tree mutation. A failed regenerate leaves the new sibling as a retryable `failed` node and reactivates the original branch.
- **Swipe navigation** (`variant-previous`/`variant-next`) resolves `store.siblings_at` (works for off-path rows), computes the target, and sets the active leaf to the deepest leaf under the target sibling — swiping back into a mid-conversation branch resumes at its branch tip, not the fork point. The header shows "name (i/n)" when a message has siblings.
- **Durable structure**: `messages.parent_message_id` (tree) plus `conversations.active_leaf_message_id` (a local-only pointer). Whole-conversation copies go through `ChatPersistenceService.fork_console_conversation_bundle` with fork fences.
- **Image variants** are a separate per-message variant set (`ConsoleVariantSet`) — see [media-generation.md](./media-generation.md).

## Tool calls and presentation

When the agent runtime is enabled, sends route through `_run_agent_reply` (the agent bridge loop); otherwise the plain provider path can still surface native tool calls as a terminal `ProviderToolCalls` from `stream_chat`. Live tool activity renders as chips (`ConsoleToolActivity`) and TOOL marker rows with diffs (`ConsoleToolDiffRow`); approvals render as `ChatApprovalCard`s resolved through the controller. Character sessions never take the agent path (`force_plain` for `assistant_kind == "character"` — roleplay has no tool calls).

## Attachments and vision

`PendingAttachment` carries bytes + mime with an inline/attachment insert mode. Caps: 100 MiB per attachment, 10 MiB per image (`[chat.images]` config: `supported_formats`, `max_size_mb`, `resize_max_dimension` — default 2048). SVG is dropped from formats when `cairosvg` is absent. Image parts are built OpenAI-style (`image_content_parts`); vision capability is gated per model by `model_capabilities.is_vision_capable` (see [llm-providers.md](./llm-providers.md)), and `max_history_images` bounds how many past images ride along per history mode (`send_all` / `send_last_user_image` / `tag_past` / `ignore_past`). Text extraction runs through the `FileHandlerRegistry` (pdf, documents, ebooks, code, data files, plaintext databases), and RAG capture/turn-scoped library preparation is wired per session (see [rag.md](./rag.md)).

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Provider validation stalls | 30 s hard deadline → synthetic not-ready stand-in with visible copy |
| Worker/provider error | Re-raised as `ChatProviderError` with the captured HTTP status; surfaced as BLOCKED/FAILED rows + toasts |
| Provider returns no/odd content | Synthetic fallback copy, flagged so usage/trace stay honest |
| Stop during stream | Cooperative cancel; partial usage attached; thinking settled "stopped"; late chunks dropped silently |
| Empty stream | Message marked failed; retryable via `retry_message` |
| Provider-continuation conflict | Send blocked with a recovery-required notice (retry or discard) |
| Stream stall | Tracked by `Chat/stream_stall_watchdog.py` (`StallTracker`, session stall records) |
| Mandatory overflow request | `ChatBadRequestError` raised before dispatch |

## Verified gotchas

1. UI updates are poll-based — there is no streaming event class; the store is the single source of truth the poll reads.
2. Dictionaries/world info mutate the ephemeral payload only; the stored transcript keeps what the user typed.
3. `Chat_Functions.chat()` is non-streaming-only by contract.
4. Late chunks after Stop are dropped silently by design, not a bug.
5. Character/roleplay sessions bypass the agent bridge entirely.
6. `save_chat_history` in `Chat_Functions.py` is a JSON file export to a temp dir — durable persistence is `commit_durable_turn` through `ChatPersistenceService`.

## Related docs

- [console.md](./console.md) — send lifecycle, approvals, run states
- [llm-providers.md](./llm-providers.md) — the provider handlers under `stream_chat`
- [character-chat.md](./character-chat.md) — dictionaries, world info, roleplay identity
