# Image Generation — Console Card with Variants (Phase 2a)

**Status:** Design approved, ready for implementation planning
**Date:** 2026-07-23
**Program:** In-chat image generation (Phase 1 backend foundation merged via PR #800)
**This spec covers:** P2a — `/generate-image` in the Console chat, producing a persistent assistant message rendered as a "render details" card with working variant navigation. TTS, Style-preset picker UI, prompt-from-context, and the character canvas are later slices (§10).

---

## 1. Background

Phase 1 shipped the multi-provider engine (`tldw_chatbook/Image_Generation/`: 6 backends, registry, `worker.build_request`/`run_generation`, `listing`) proven only by a throwaway command-palette demo panel. P2a brings generation into the Console conversation itself, mirroring tldw_server's webui pattern (image generation as an assistant message) and the reference card layout: image beside a details block (Style / Source / Seed / Prompt / Negative), variant `◀ n/N ▶` navigation, regenerate.

Recon of current `dev` (see file:line anchors throughout) established: a real slash-command framework exists; the transcript renders per-message sub-rows with an image-row pattern to mirror for a card row; `messages` has **no metadata column**; Console has **no image-variant** support (text-variant machinery only) and no TTS. Schema is at **v24** (re-verify at build time — the version number is a known collision point between concurrent branches).

## 2. User-visible behavior

- `/generate-image <prompt>` in the Console composer generates image(s) and appends an **assistant message** to the conversation. A **leading `:backend` token in the args** overrides the backend: `/generate-image :swarmui a red dragon`. (Verified constraint: `ConsoleCommandRegistry.parse` exact-matches the leading token — `console_command_grammar.py:151` `self._commands.get(word.lower())` — so a `/generate-image:backend` command-name suffix would miss the registry and fall into the skill-fallback/unknown path. The override therefore lives in the args, handled by `parse_generate_image_args`; helper style mirrors `parse_prefill_args`, `Chat/console_prefill.py:65`.)
- The initial generation produces `[image_generation].default_batch` variants (**default 1**).
- The message renders as a bordered **Image Generation card**: selected variant image (existing pixels/graphics render modes) + "Render details" block showing **Style** (template name, or "Custom" for raw prompts in P2a), **Source** (backend name, e.g. `swarmui`), **Seed**, **Prompt**, **Negative**, plus model/size when known.
- **`◀ n/N ▶`** navigates variants (in-card, ephemeral view state); **Keep** marks the displayed variant as the message's canonical image everywhere (export, sync, non-card views — durable). **Regenerate (♻)** on a generation message **appends one new variant** and browses to it (it is NOT auto-kept; press Keep to make it canonical) — it does not create an LLM text sibling.
- Variants are capped at `[image_generation].max_variants_per_message` (**default 8**); at cap, regenerate refuses with a clear status line.
- Generation runs off the UI loop; a status line shows progress; failures surface as a visible error (message stays usable). While a generation for a message is in flight, regenerate on that message is refused (in-flight guard — NOT `exclusive=True`, which would cancel the running generation).
- The message's text `content` is a short marker (`[image] <prompt-excerpt>`) so history, exports, and FTS remain sensible.
- Works in a persisted conversation; images + metadata survive app restart and re-render from the DB.

## 3. Trigger wiring (slash command)

Follow the `/prefill` template exactly:
1. `ConsoleCommand` + constants in `Chat/console_command_grammar.py` (register in `default_console_registry()`, `console_command_grammar.py:166` — registering as a real command keeps it out of the `/skill-name` fallback).
2. Entry in `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` (`UI/Screens/chat_screen.py:11172`).
3. Coroutine in `dispatch_map` (`chat_screen.py:11205`).
4. `async def _console_command_generate_image(self, parse)` — structural template `_console_command_prefill` (`chat_screen.py:11476`). Note command dispatch runs **before** the send-readiness gate (`chat_screen.py:11032`), which is correct for generation.
5. New pure helper `parse_generate_image_args(args) -> (backend | None, prompt)` with its own test table.

The handler resolves the backend (explicit `:backend` → else `default_backend`), refuses with a clear message when unresolvable/not configured (`listing` knows `is_configured`), then dispatches the generation worker.

## 4. Generation flow & concurrency

- Build requests via `worker.build_request` (Phase-1 engine; `run_generation` enforces validation). A batch of N = N sequential `run_generation` calls in one worker (adapters are blocking; per-variant seed recorded from the request/backend response).
- Run on a thread worker with a distinct group (e.g. `group="imagegen-console"`, `exclusive=False`) + an **in-flight set keyed by message id** (and one for "new command in this conversation") to prevent double-triggering without cancelling running work.
- On completion (marshalled back to the UI loop): append/update the assistant message, persist, register card specs, scroll. On failure: **append only on first success** — if the initial command fails outright, surface the error as a transient status/system line and create no assistant message (no orphan rows). A failed *append* (regenerate) leaves the existing message untouched and reports the error.

## 5. Storage (schema v24 → v25) — the load-bearing design

### 5.1 Where things live

- **Variant image bytes** → the existing authoritative attachments contract: positions `0..N-1` (`ChatPersistenceService.create_message`/`update_message_content` split contract, `chat_persistence_service.py:439-472`; position 0 on the `messages` row, ≥1 in `message_attachments`).
- **Per-variant generation metadata** → **new sidecar table** (v19 `message_attachments` precedent — no `messages` sync/FTS trigger changes):

```sql
CREATE TABLE IF NOT EXISTS message_generation_metadata(
  message_id      TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position        INTEGER NOT NULL CHECK (position >= 0),
  prompt          TEXT    NOT NULL,
  negative_prompt TEXT    NOT NULL DEFAULT '',
  backend         TEXT    NOT NULL,
  model           TEXT,
  seed            INTEGER,
  style           TEXT,             -- template id/name; NULL/"custom" for raw prompts
  params_json     TEXT    NOT NULL DEFAULT '{}',  -- width/height/steps/cfg/sampler/format…
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, position)
);
CREATE INDEX IF NOT EXISTS idx_msg_gen_meta_message ON message_generation_metadata(message_id);
```

- **Selected variant** needs no column: **the selected/kept variant IS position 0** (§5.2). A message "is a generation message" ⇔ it has sidecar rows.

### 5.2 Keep semantics (decision: kept = canonical everywhere)

The store's documented invariant (`console_chat_store.py:688-692`): scalar `image_data`/`image_mime_type` are strictly a mirror of `attachments[0]`, and **every attachments mutation flows through `_set_message_attachments`**, which re-bases positions from 0. Therefore keep is implemented as a **reorder**: move the kept variant to index 0 of the attachments tuple, persist via the authoritative attachments-list contract, and re-key the sidecar rows to the new positions atomically. No side-channel scalar writes.

**Verified seams (and one required extension):**
- Persistence already supports the rewrite: `update_message_content(attachments=[...])` (`chat_persistence_service.py:292`) is an authoritative full-position rewrite — position 0 → messages row, ≥1 → `message_attachments`, row update + table rewrite in one transaction.
- **The console store has no public API to mutate an existing message's attachments** (`_set_message_attachments` is a private append/sibling-time helper; the public attachment APIs are composer staging). P2a adds a store method (e.g. `update_message_attachments(session_id, message_id, attachments, *, persist)`) that flows through `_set_message_attachments` and persists via the contract above, passing the unchanged `content`.
- **Atomicity requires extending the persistence seam:** `update_message_content` opens its own transaction, so the sidecar re-key must **join that transaction** — a generation-aware update path (e.g. an optional `generation_metadata` rows argument on the persistence call, or a sibling method sharing the transaction) rather than a second transaction. Two separate transactions (attachments then sidecar) would leave metadata misaligned with images on a crash between them; the spec requires the single-transaction form.

In-card `◀ ▶` **browsing does not reorder** — it's view state (selected index in the card spec, screen-state only, like image view modes). Only **Keep** commits the reorder. On reload, the card shows position 0 (the kept/canonical variant) — deliberate: browsing is ephemeral, keeping is durable.

### 5.3 Migration

Exact v23→v24 pattern (`ChaChaNotes_DB.py:2489` SQL const, `:3861` runner shape, register in `migration_steps` `:4020-4041`, bump `_CURRENT_SCHEMA_VERSION` `:160`, mirror DDL to `DB/migrations/`). **Re-verify at build time that v25 is still unclaimed** (concurrent branches; check all refs, not just dev).

### 5.4 Conscious exclusions

- **Sync:** the sidecar table gets NO sync-trigger integration in P2a (v19 attachments + v24 `active_leaf_message_id` precedent: deliberate exclusion, documented in the migration comment).
- **FTS:** generation prompts are NOT indexed (the `image_data` precedent); the message `content` marker text is FTS-indexed normally, which gives basic searchability.

## 6. Card renderer

Mirror the image-row pattern end to end:
- New row kind `"generation-card"` in `_TranscriptRow` (`console_transcript.py:205`), a per-message spec map + `set_generation_card_specs(...)` setter (mirror `set_image_specs`, `:547`), row emission in `_transcript_rows()` (`:804`), widget construction in `_build_row_widget()` (`:950`), and a **signature entry** in `_reconcile_rows` (`:887`) covering selected index + variant count + view mode so changes re-render.
- **For generation messages the standard image row is suppressed** (the card renders the image itself — no double render). Concretely: generation messages DO carry `image_data` (the position-0 mirror), so `_build_console_image_specs`/`_recent_console_image_messages` (`chat_screen.py:3100-3135`) must **exclude messages that have card specs** — both to avoid the double render and so card messages don't consume slots in the 16-message inline-image window (`IMAGE_CACHE_MAX_ENTRIES`). The message-body row still renders the `[image] …` content text.
- The card widget is a mounted child widget (so it can hold Rich renderables — the message body `Content` cannot, `console_transcript.py:173-187`): bordered panel titled "Image Generation", selected variant image (reuse `ConsoleImageRenderCache` decode + pixels/graphics modes and the existing off-thread prep pattern, `chat_screen.py:3135`) beside the render-details block, with `◀ n/N ▶`, Keep, and collapse controls. Image view-mode cycling ("View" action) continues to work for the card's image.
- Variant bytes for non-selected variants are **not** decoded until navigated to (cache does this naturally; only the selected variant enters the render cache).
- Screen side: `_build_generation_card_specs()` from store messages + sidecar metadata (loaded with the conversation; sidecar read API on the DB layer), handed to the transcript alongside image specs (`chat_screen.py:10578` region).

## 7. Actions

- `ConsoleMessageActionService.available_actions()` (`console_message_actions.py:101`): for generation messages (sidecar present), regenerate `♻` stays visible (assistant-gated already) but dispatch **branches to append-variant**; `< >` variant actions reuse the existing button plumbing but drive the **card's** selected index for generation messages (not the text-variant `ConsoleVariantSet`). **Verified gating constraint:** `< >` visibility and dispatch are currently keyed on `sibling_count`/`sibling_index` (`console_message_actions.py:107, 287-290`) — text siblings, which generation messages don't have. The action-service surface therefore gains an explicit generation-variant input (e.g. a `generation_variant_count`/browsed-index pair passed alongside the message) gating `< >` and enabling boundary checks for card messages; don't overload `sibling_count`. New **Keep** action (visible only on generation messages when the browsed variant ≠ position 0). "View"/"Save Image" continue to work (Save saves the *browsed* variant).
- The action row remains select-only (existing model). Making it persistent like the reference is explicitly out of scope (would change every message's UX — separate discussion).

## 8. Config additions

`[image_generation]` gains `default_batch = 1` and `max_variants_per_message = 8` (config template + `_GLOBAL_KEYS` in `Image_Generation/config.py`'s flattening loader + dataclass fields + `DEFAULT_*` constants; loader tests extended).

## 9. Testing

- **Migration/DB:** v24→v25 applies + idempotent guard; sidecar CRUD; variant append; keep-reorder re-keys sidecar atomically (attachments and sidecar agree after reorder); cascade delete.
- **Parse:** `parse_generate_image_args` table (plain, `:backend`, empty, whitespace).
- **Flow (mocked `run_generation`):** command → assistant message appended with attachments + sidecar rows persisted; batch N; regenerate appends + selects; cap refusal; in-flight guard; failure leaves no orphan message; backend-unconfigured refusal.
- **Renderer:** card row emitted for generation messages and image row suppressed; details block contents; `◀ ▶` updates selected variant + signature; keep commits position-0; reload renders from DB.
- **Live TUI verify** (tmux recipe) against a real local backend for the end-to-end proof.

## 10. Out of scope (later slices)

- **P2b:** TTS 🔊 on messages (reuse `tldw_chatbook/TTS/` backends); Style-preset picker + `generation_templates.py` wiring beyond the display field; prompt-from-context (`extract_context_from_messages`); a Generate modal.
- **P3:** character canvas (auto-prompt from appearance, per-mood reaction images).
- Sync integration for the sidecar table; FTS for prompts; persistent per-message action rows; deleting the orphaned `Media_Creation` SwarmUI chain (separate cleanup task, still pending).

## 11. Locked decisions

Variant model = config `default_batch` (default 1) + regenerate-appends. Keep = canonical-everywhere, implemented as attachments reorder through the store invariant (§5.2). Card = one combined bordered widget (image + details side-by-side). Storage = attachments 0..N-1 + `message_generation_metadata` sidecar keyed `(message_id, position)`; selected = position 0; no sync/FTS integration in P2a. Cap = 8 default. Browsing ephemeral, keep durable.
