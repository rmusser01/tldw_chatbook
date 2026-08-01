# Temporary Conversations — Design

**Date:** 2026-07-31
**Status:** Approved (design); implementation not started

## Purpose

Let a user hold a Console conversation that is never written to local storage.
The chat lives only in memory: it disappears when its tab closes or the app
restarts, and it leaves no conversation row, no message rows, and no derived
files behind.

The promise the UI makes is **"not saved locally"** — nothing weaker, nothing
stronger. What a provider does with the prompt after it leaves the machine is
explicitly out of scope, and no copy may imply otherwise.

## Scope

In scope:

- No `conversations` or `messages` rows for the chat.
- No derived local artifacts from the chat: no generated image files, no
  chatbook exports, no RAG/embedding indexing of its content.

Out of scope (deliberate, user-confirmed):

- In-process traces. The running app may hold the chat in memory caches,
  rewind buffers, and the screen-state snapshot; only durability is promised.
- Provider-side retention. No warning or refusal based on the provider.

## Architecture

### The flag

`ConsoleChatSession` gains `ephemeral: bool = False`, set when the session is
created and changed only by the explicit promote action. It sits with the
other identity fields (`runtime_backend`, `assistant_kind`, `character_id`).

### The gate

`ConsoleChatStore.persist_session_if_needed` returns `None` immediately when
`session.ephemeral` is true, before the persistence adapter is consulted. Its
documented contract widens from "`None` when no persistence adapter is
configured" to "…or when the session is temporary", and the docstring is
updated to say so.

This single gate is the entire durability mechanism, and it works because that
function is how a session acquires its `persisted_conversation_id`. Message
flushes call it to obtain that id, then return early when it yields `None`
(`console_chat_store.py:2578`), and other durable writes are gated on the
session already owning a persisted conversation. With no id, those paths no-op
along a branch they already take today whenever no persistence adapter is
configured.

**Why not guard each write site.** 43 sites consult `self.persistence`.
Guarding all of them makes the guarantee depend on having found every one, and
a missed site writes silently. Gating the id means a missed site fails toward
*not writing*. The failure direction is the point.

**Second door.** `restore_persisted_session` assigns
`persisted_conversation_id` directly, bypassing the gate. It is unreachable
for ephemeral sessions, which are never restored from the database, but the
implementation adds a guard so a later change cannot quietly open it.

### Data flow

Messages append with `persist=False` and live in the store's in-memory list.
The transcript, rewind, variants, regeneration, and edit-resend all read from
memory and behave identically to a normal chat. Provider calls are unchanged:
context is assembled from in-memory messages, so the send path never needs to
know the chat is temporary.

### Death

Closing the tab drops the session and its messages. An app restart takes
everything, because the screen-state store is memory-only by design
(`UI/Navigation/screen_state_store.py`: "Memory-only ownership for cross-visit
screen snapshots"). No cleanup code is required, because nothing reached disk.

### Promotion

"Save this chat" runs in **one transaction**: clear `ephemeral` (which is what
opens the gate), mint the conversation via `persist_session_if_needed`, then
flush each message with `persist_message_if_needed`. That function is
idempotent and no-ops only without an adapter or on an already-persisted row,
so it flushes every unpersisted message — no deferral arming required.

On failure the transaction rolls back **and `ephemeral` is restored to true**,
so a failed save leaves a still-temporary chat rather than one that silently
starts persisting on the next send. A half-saved conversation is never left in
history.

## UX surface

### Entry

A new action, `action_new_temporary_console_tab`, creating a session exactly as
`action_new_console_tab` does but with `ephemeral=True`. Reachable by:

- the command palette entry "New temporary chat" (guaranteed path);
- a "New temporary chat" button in the tab strip beside the existing
  `#console-new-chat-tab` button, in `console_session_surface.py`;
- `Alt+T`, chosen for consistency with the existing Alt family (`Alt+M` model,
  `Alt+W` workspace) and its mnemonic link to `Ctrl+T` (new tab). **The binding
  must be verified live in a real terminal before it is committed to**; if it
  does not reach the app, the palette and tab-strip paths stand alone.

The existing `Ctrl+T` / "New tab" path keeps its current meaning.

### Marking

- The tab title renders as `Temporary · <title>`. **Presentation only** —
  derived at render time from the flag, never written into `session.title`,
  so promotion cannot save a conversation literally titled "Temporary · …"
  and renaming does not fight the marker.
- The shell chip strip shows a `Temporary — not saved` chip beside
  Provider/Model/Character, present only in temporary chats (following the
  retrieval-scope chip, which hides entirely when it has nothing to say).

### Promotion controls

Two entry points into one handler:

- the composer menu gains "Save this chat", shown only in temporary chats;
- the `Temporary` chip is an action chip — click or Enter/Space offers the
  same save, reusing the `ConsoleModelChip`/`ConsoleAssistantChip` pattern.

After promotion the marker and chip disappear and a toast confirms the chat is
saved.

### Blocked actions

Disabled with a stated reason, never hidden, matching the convention used for
Generate Caption:

| Action | Reason shown |
| --- | --- |
| Generate Image | writes a PNG to disk — not available in a temporary chat |
| Save Chatbook | exports a file — not available in a temporary chat |

RAG indexing of the chat's own content is skipped silently: it has no user
control to disable, so there is nothing to label. RAG **retrieval** stays
fully available, because reading the index stores nothing.

Attachments stay allowed. Staging holds bytes in memory and references the
file the user already has; `attachment_core` performs no copy, so no new
artifact appears.

## Risks and mitigations

**The flag must round-trip through `save_state`/`restore_state`.**
`save_state` serializes sessions as an explicit field list. If `ephemeral` is
omitted, navigating Console → another screen → Console restores the session
without it, silently converting a temporary chat into a persisting one, and
the next send writes it to the database. The field is added to both the
serializer and the restore path, pinned by a round-trip test. This is the
highest-severity risk in the design: the path that makes a temporary chat
survive screen navigation is the same path that would drop its guarantee.

**Sink enumeration, not the gate, is the residual risk.** The gate covers the
store rigorously. "No derived local artifacts" rests on the blocked-action
list being complete. Implementation begins with an explicit audit: does
attaching ingest into the Media DB? do other export or generation paths write
files? Findings extend the table above rather than being assumed absent.

## Testing

**The proof test.** Run a complete temporary conversation against a real
in-memory SQLite database — send, receive, regenerate, rewind, edit-resend —
then assert the `conversations` and `messages` row counts are unchanged from
before the chat began.

**This test is vacuous without its control.** A harness with
`persistence=None` passes it trivially. The same test must show a *normal*
chat in the *same* harness writing rows. Without that control the test proves
nothing, and flag-assertions ("`ephemeral` is True") would pass even against a
completely broken gate.

Supporting tests:

- `ephemeral` survives a `save_state`/`restore_state` round trip.
- Promotion writes exactly the on-screen messages, in order, and is
  idempotent when triggered twice.
- Promotion failure rolls back: no partial conversation, chat stays temporary.
- Blocked actions report disabled-with-reason in a temporary chat and stay
  enabled in a normal one.
- RAG retrieval works in a temporary chat while indexing is skipped.
- Closing the tab leaves no rows.
- The tab marker is not present in `session.title`.

## Edge cases

- **Screen navigation is not "reload."** Leaving Console and returning keeps
  the chat alive — intended, and the reason the round-trip test above exists.
- **Promotion mid-stream** is allowed; an in-flight reply persists on
  completion through the normal path, the gate being open by then.
- **Failed and send-blocked messages** already append with `persist=False` and
  need no special handling.
- **The conversation switcher** will not list temporary chats among persisted
  conversations (no row exists); the in-memory tab switcher will, correctly,
  because the tab is open.
- **Auto-title updates** call through persistence and no-op without a
  conversation id.

## Decided: no save prompt on close or quit

Closing a tab discards silently, and so does quitting the app with temporary
chats open. The marker and the chip are the protection against doing it by
reflex; a confirmation dialog would contradict the feature's whole premise,
nag on every quit, and train the user to dismiss it. Adding a prompt later is
a small change if real use shows people losing work they wanted.

## Sink audit (task 1)

Ran the four searches below and read every hit, then traced the Console
message-action row and composer menu in `chat_screen.py` by hand (the two
known sinks — Generate Image and Save Chatbook — live there, and reading the
surrounding dispatch table surfaced siblings the greps alone did not).

```
grep -rn "generate-image\|save-chatbook\|save_chatbook" tldw_chatbook/UI/Screens/chat_screen.py
grep -rn "def .*write\|open(.*[\"']w\|\.write_bytes\|\.write_text\|save_image\|export" tldw_chatbook/Chat/ tldw_chatbook/Widgets/Console/
grep -rn "index_conversation\|index_message\|embed\|add_to_index" tldw_chatbook/RAG_Search/ | grep -i "conversation\|chat"
grep -rn "media_db\|Client_Media_DB" tldw_chatbook/Chat/attachment_core.py tldw_chatbook/UI/Screens/chat_screen.py
```

The second and fourth searches are scoped to `tldw_chatbook/Chat/` and
`tldw_chatbook/Widgets/Console/`, which does not include
`tldw_chatbook/UI/Screens/chat_screen.py` itself — that is where the actual
`write_bytes`/DB-insert calls for the message-action row live, so they do
not appear as direct grep hits. They were found by reading the dispatch
table the first search's `save-chatbook` hit sits in
(`chat_screen.py:1664`, `WorkbenchActionRequested` handling) and following
its sibling action ids in the selected-message action row
(`console_message_actions.py`, `_parse_console_message_action_button_id`).
That row turned out to hide five more local-write sinks the design's known
list did not name — this is exactly the residual risk the design called
out ("Sink enumeration, not the gate, is the residual risk").

| Sink | Reachable from a temporary chat? | Writes what | Decision |
| --- | --- | --- | --- |
| Generate Image (`generate-image`, composer menu → `/generate-image`) | Yes — composes from the in-memory draft/session | PNG file to the configured image save location | blocked |
| Save Chatbook (`save-chatbook`, Workbench button, `chat_screen.py:1664`, `handle_console_save_chatbook:16230`) | Yes — Workbench action, no persisted conversation required to invoke | Chatbook export file | blocked |
| Save Image (`save-image`, message-action row, `_save_console_message_image`, `chat_screen.py:16669-16800`) | Yes — reads the message's in-memory attachment bytes first, only falls back to a DB fetch when `persisted_message_id` is set (never true for a temporary chat) | Image bytes (`target.write_bytes`) to the configured save location | blocked (new) |
| Save as Note (`save-as-note`, `_save_console_message_as_note`, `chat_screen.py:16807-16857`) | Yes — reads the message from the in-memory store only | Row in the local Notes database via `notes_scope_service.save_note` | blocked (new) |
| Save as Media (`save-as-media`, `_save_console_message_as_media`, `chat_screen.py:16859-16912`) | Yes — same, in-memory message content only | Row in the local Media DB via `media_db.add_media_with_keywords` | blocked (new) |
| Save as Prompt (`save-as-prompt`, `_save_console_message_as_prompt`, `chat_screen.py:16913-16986`) | Yes — same | Row in the local Prompts DB via `prompts_db.add_prompt` | blocked (new) |
| Save as Chatbook, per-message (`save-as-chatbook`, `_save_console_message_as_chatbook`, `chat_screen.py:16987-17073`) | Yes — same; distinct from the Workbench-level Save Chatbook button | Chatbook artifact file via `local_chatbook_service.create_chatbook` | blocked (new) |
| Save context snapshot (`save-context`, `#console-context-save` in `ConsoleContextModal`, `console_context_modal.py:287-302`, opened by `action_view_chat_context`, Ctrl+Shift+P) | Yes — dumps the live transcript and next-send payload of the active session, temporary or not | JSON file to `~/Downloads/chatbook_context_*.json` | blocked (new) |
| RAG indexing of the chat's own content (`conversation_index_entry`/`conversation_document`, `ingestion_indexing.py:586-631`) | No — requires `conversation.get("id")`; a temporary chat has no `conversations` row for the indexer to read | Nothing, for a temporary chat | no-op (needs a conversation id) |
| Legacy RAG-context/citation sidecar (`ChatConversationService.record_message_rag_context` / `_save_rag_context_store`, `chat_conversation_service.py:322-330,745-788`) | No — requires an already-persisted message row (`db.get_message_by_id`), and is not wired into `console_chat_controller.py` or `console_chat_store.py` at all (zero references) | JSON sidecar file, but only for non-Console flows | no-op (not reachable from Console) |
| Generate Caption (`generate-caption`, composer menu, `_insert_console_caption_prompt`) | Yes | Nothing — appends a pre-canned prompt string to the in-memory draft | allowed (no write) |
| Narrate Entire Conversation (`narrate-conversation`, composer menu) | Yes | Nothing — feature is unimplemented; the handler only shows an information toast | allowed (no write) |
| Impersonate (`impersonate`, composer menu, `_run_console_impersonate`) | Yes | Nothing — drafts the user's next message via a model call and inserts it into the composer | allowed (no write) |
| Attachment staging (`attachment_core.py`: `process_attachment_path`, `process_attachment_bytes`, `load_processed_file`) | Yes | Nothing — zero `write`/`save` call sites in the file; bytes are staged in memory and referenced from the original path | allowed (no write) |

Six new rows extend `EPHEMERAL_BLOCKED_ACTIONS`: `save-image`,
`save-as-note`, `save-as-media`, `save-as-prompt`, `save-as-chatbook`, and
`save-context`. The blocked-action table in the UX-surface section above
predates this audit and only lists the original two; the full, current list
is `tldw_chatbook.Chat.console_ephemeral.EPHEMERAL_BLOCKED_ACTIONS`, and
task 9's UI wiring is scoped to all eight entries, not just the original
two.
