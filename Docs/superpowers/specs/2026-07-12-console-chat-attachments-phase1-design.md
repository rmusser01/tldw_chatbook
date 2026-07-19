# Console Chat Native Attachments & Image Support — Phase 1 Design

- **Date:** 2026-07-12
- **Status:** Approved (design review complete; verified against code)
- **Scope anchor:** Native Console attach flow with full legacy file-type parity, image send to vision providers, DB persistence, placeholder chip rendering in the transcript.

## Why

The Console chat's Attach button is currently **entirely non-functional**: it delegates to `_get_active_chat_session()`, which returns `None` unconditionally in the current build (`#console-chat-tabs` is never composed; `_ensure_chat_window()` has no callers), so every attach attempt notifies "No active Console chat session is available." The legacy bridge it relies on is dead code — verified 2026-07-12, no Console session is legacy-backed at runtime. Meanwhile the legacy chat has a complete attachment pipeline (images + docs/code/data, validation, resize, vision gating, DB columns, provider conversion) that the Console cannot reach. This design gives the Console its own first-class attachment flow while extracting the legacy logic into a shared, UI-agnostic core (Approach C — chosen over a Console-only service to deduplicate logic across both chats). Removing the dead bridge delegation in `_handle_console_attach_context` is part of the change; broader dead-code cleanup (`_ensure_chat_window`, legacy `_get_tab_container` branches) is not.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| Attach scope | Full legacy parity: images as attachments; docs/code/data text-inlined; all `file_handler_registry` types |
| Transcript display | Placeholder chip (`🖼 name.png · 240 KB`) + Save Image action; inline pixel/TGP rendering is a fast-follow |
| Non-vision model + pending image | Send **blocked** with visible reason (not legacy warn-and-drop) |
| Image history | All session images resent as multimodal parts when model is vision-capable, capped by the model's `max_images` (most recent N kept). Capability data often omits `max_images` (pattern-matched models return only `{"vision": True}`), so the cap is `capabilities.get("max_images", 10)` — the 10 default lives as a named constant |
| Text-file inlining | Collapsed draft segment (`📄 name · size`) reusing composer paste-collapse machinery |
| Architecture | C: shared attachment core consumed by both legacy and Console chats |

## Architecture

### New shared core — `tldw_chatbook/Chat/attachment_core.py`

UI-agnostic, no Textual imports. Async functions — the underlying processors `ChatImageHandler.process_image_file` and `file_handler_registry.process_file` are already async with zero Textual dependencies. Both consumers keep their existing call patterns: the Console awaits from its async workers; the legacy handler already drives these async processors via `asyncio.run()` inside a sync method under `run_worker(thread=True)` (with a direct-await fallback when unmounted), so the core slots in without changing legacy worker mechanics.

- `PendingAttachment` dataclass: `file_path`, `display_name`, `file_type` (image/document/code/data/ebook), `insert_mode` (`"attachment"` | `"inline"`), `data: bytes | None` + `mime_type` (attachment mode), `text_content` (inline mode), original/processed sizes, warnings.
- `process_attachment_path(path, settings) -> PendingAttachment`: wraps `is_safe_path` validation, size caps (100 MB general, 10 MB image), type dispatch via existing `file_handler_registry`, image pipeline via existing `ChatImageHandler` (validate → PIL resize to `resize_max_dimension` → re-encode). Format allowlist and caps read from `[chat.images]` — the same source the file-picker filters use, so they cannot drift.
- `vision_block_reason(provider, model) -> str | None`: wraps `model_capabilities.is_vision_capable`, produces the user-facing reason string so both UIs gate identically. Models with unknown capabilities (e.g. runtime-discovered, unverified — which already carry the separate generic "Capabilities unknown" readiness label) are treated as non-vision, the safe default. The blocked copy therefore references the `[model_capabilities.models]` config override so users running local vision models (llama.cpp llava/qwen-VL) or unverified discovered models have a documented escape hatch.

### Consumer 1 — legacy `ChatAttachmentHandler` (refactor, zero behavior change)

Becomes a thin UI adapter: file picker, call core, set existing `pending_image`/`pending_attachment` reactives, update the 📎 indicator. Warn-and-drop send semantics are **retained** in legacy. The existing image test suites are the regression gate and must pass unedited (baseline verified green on 2026-07-12: 81 passed, 1 skipped across the six image suites).

### Consumer 2 — Console adapter

- **State:** `ConsoleChatSession.pending_attachment: PendingAttachment | None` in `ConsoleChatStore` — per-session, so each Console session tab keeps its own draft attachment. Store gains `set_pending_attachment` / `clear_pending_attachment`; re-attach replaces (single attachment per message).
- **Model:** `ConsoleChatMessage` gains `image_data: bytes | None`, `image_mime_type: str | None`, `attachment_label: str | None`. Dataclass is mutable and copied via `dataclasses.replace` throughout — new fields are transparent to existing code.
- **Screen wiring:** `_handle_console_attach_context` (already wired to composer button, control-bar action, and empty-state button) stops bridging to legacy; pushes `enhanced_file_picker.FileOpen` with filters derived from `[chat.images].supported_formats` + registry types, processes via core in an async worker, then routes by `insert_mode`.
- **Composer:** `_DraftSegment` gains optional `label`; `_segment_display_text` renders it; new `insert_file_segment(text, label)` API. (`draft_text()` already returns full underlying text — collapse is display-only — so the send path needs no change for inlined files.) Attachment-mode results set store pending + a composer indicator (`📎 name`, clear affordance).

## Data flow

**Attach:** button/action → `FileOpen` → async worker → `process_attachment_path` → inline ⇒ `insert_file_segment(text, "📄 name · size")`; attachment ⇒ `store.set_pending_attachment` + indicator. Errors (unsafe path, oversized, unsupported, parse failure) → warning toast, state unchanged.

**Send:**
- `_console_send_blocked_reason` gains: pending image + non-vision model ⇒ blocked with reason. Rendered through the existing composer blocked machinery (`#console-send-disabled-reason`, tooltip, CSS classes); readiness is recomputed on every settings apply, so the block auto-clears when the user switches to a vision model (verified: `_active_console_settings_readiness` is rebuilt per call).
- Draft validation relaxed: empty text is sendable when an attachment is pending (image-only messages).
- `submit_draft` moves pending attachment onto the user `ConsoleChatMessage` and clears pending.
- `_provider_messages_for_session` (the single string-typed chokepoint, `console_chat_controller.py:632`): emits content-parts lists (`{"type":"text",...}` + `{"type":"image_url","image_url":{"url":"data:<mime>;base64,..."}}`) for session image messages when vision-capable — most recent `max_images` (from `get_model_capabilities`) if the session exceeds the cap — plain strings otherwise. Its empty-content skip is amended to keep image-only messages. Type hints widen to `dict[str, Any]`; `_ensure_user_continuation_instruction` appends to the text part.
- Gateway: **no changes needed** — both the generic `chat_api_call` bridge and the dedicated llama.cpp path pass message content verbatim into JSON (verified; list-content serializes correctly). Provider-side converters (Anthropic/Gemini/Moonshot/OpenAI-passthrough) already handle `image_url` parts.

**Persist:** store passes `message.image_data`/`image_mime_type` in `_persist_new_message` **and** `_persist_existing_message` (today both hardcode `None`, and `ChaChaNotes_DB.update_message(image_data=None)` NULLs the columns — so without this, editing a message's text would erase its image). `append_message` gains image parameters. `ChatPersistenceService` and the DB schema already accept image fields; no DB changes.

**Resume:** `_console_messages_from_conversation_tree` (chat_screen.py:1956-1979) extended to read `image_data`/`image_mime_type` from rows and to keep image-only rows (today it reads only `content` and drops empty-content rows — images would vanish on resume).

**Display:** transcript row for an image message renders a chip line via `_message_render_text` + row-signature update (cheap; no heavy rendering, safe under the 0.2 s reconcile timer). "Save Image" message action, gated on image presence: service conditional tuple + `dispatch` branch, chat_screen button-id prefix + handler, transcript renders automatically. Writes to `[chat.images].save_location` using the legacy save-image naming; bytes come from the in-memory message, DB via `persisted_message_id` as fallback.

## Serialization & ephemerality

Console screen state round-trips through explicit allowlists into an **in-memory** app dict (never disk/config; verified against task-150 and the rail-state serializer):

- A **pending** (unsent) attachment is deliberately NOT serialized: it is lost if the user navigates away from Console before sending. Accepted for Phase 1.
- **Sent** messages: `image_mime_type` + `attachment_label` (metadata only, never raw bytes) are added to `_serialize_console_message`/`_restore_console_message` so chips render immediately on screen return; bytes rehydrate from the DB via the resume path when needed.
- Sync v2 message enqueue passes scalar/content fields only — unaffected, but note the fidelity gap: image messages sync as text-only until Sync v2 attachment upload lands (out of scope here; task-57 territory).

## Error handling summary

| Failure | Behavior |
|---|---|
| Unsafe/oversized/unsupported file | Warning toast; pending state unchanged |
| Processing error in worker | Toast + logged with context; state unchanged |
| Pending image + non-vision model | Send blocked, visible reason, auto-clears on model change |
| Provider rejects request | Existing controller `_block` path (system message + run-state blocked) |
| DB persist failure | Existing store persist error handling; message stays in memory |

## Testing

1. **Shared core unit tests** (new): path-safety rejection, size caps, registry dispatch, image resize/re-encode, `vision_block_reason`; reuse fixtures/patterns from `test_chat_image_events.py` + Hypothesis property tests.
2. **Legacy regression gate** (existing, unedited): `Tests/UI/test_chat_image_attachment.py`, `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py` + `_properties.py`, `Tests/unit/test_chat_image_unit.py`, `Tests/DB/test_chat_image_db_compatibility.py`, `Tests/Widgets/test_chat_message_enhanced.py`. Baseline verified green pre-design. Needing to edit these files is a design smell — stop and examine.
3. **Console unit tests** (extend `Tests/Chat/test_console_chat_store.py`, `test_console_chat_controller.py`): per-session pending isolation, replace-on-reattach, image fields through `append_message` + both persist paths (explicit "edit does not wipe image" test), resume hydration incl. image-only rows, parts-list building (vision vs not), `max_images` cap, image-only send, continuation instruction on parts content, blocked-send statuses.
4. **Mounted UI tests** (extend `Tests/UI/test_console_native_chat_flow.py`, `test_console_native_transcript.py`; `app.run_test()` per harness constraints): composer `insert_file_segment` label display + full `draft_text()`, attach indicator + clear, blocked-send visuals appearing/clearing on model change, transcript chip row + signature diffing, Save Image action visibility + dispatch.

Plus the standing **visual review gate**: textual-serve screenshot capture of composer indicator, blocked state, and transcript chip against the Console style anchor before merge. CI is verified locally (checks intentionally cancelled remotely).

## Out of scope (follow-up tasks to file)

- Inline pixel/TGP transcript rendering with Toggle View (agreed fast-follow).
- Clipboard image paste; drag-and-drop.
- Multiple attachments per message (single, replace-on-reattach).
- Pending attachment surviving navigation away from Console.
- Sync v2 attachment upload (task-57 territory).
- Chatbook export carrying images (Save Chatbook will omit them; task-19 adjacent).
- Any behavior change to legacy chat's warn-and-drop semantics.
- Config knob for image-history policy (fixed all-session-images + `max_images` cap; note: legacy `image_history_mode` is an unwired dead parameter, not a real config surface).

## Key file touch list

| File | Change |
|---|---|
| `Chat/attachment_core.py` | **New** — shared core |
| `UI/Chat_Modules/chat_attachment_handler.py` | Refactor to consume core (behavior-identical) |
| `Chat/console_chat_models.py` | `ConsoleChatMessage` image fields |
| `Chat/console_chat_store.py` | Pending attachment state; image-aware persists; `append_message` params |
| `Chat/console_chat_controller.py` | Parts-list builder, image-only send, `max_images` cap, blocked statuses |
| `Widgets/Console/console_composer_bar.py` | `_DraftSegment.label`, `insert_file_segment`, attach indicator, blocked copy |
| `Widgets/Console/console_transcript.py` | Chip line + row signature |
| `Chat/console_message_actions.py` | Save Image action |
| `UI/Screens/chat_screen.py` | Native attach wiring, vision block in readiness, resume hydration fix, Save Image handler, message serialization metadata |
