# Character chat, personas, and buddies

This document describes the roleplay stack: character card import (PNG/WebP/JPEG containers, V2/V3 JSON), the single card→prompt composer, world books and the world-info engine, chat dictionaries, the persona/character/buddy identity split, and expression/emote playback.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Character library core | `Character_Chat/Character_Chat_Lib.py` | card loading/parsing, `compose_character_card_template()`, exports, legacy conversation CRUD |
| Multi-format detection | `Character_Chat/character_card_formats.py` | `CharacterCardFormatDetector`, `detect_and_parse_character_card()` (Agnai / CharacterAI / KoboldAI / TextGen / generic) |
| World books | `Character_Chat/world_book_manager.py` | `WorldBookManager` — CRUD, conversation association, character attach, import/export |
| World info engine | `Character_Chat/world_info_processor.py`, `world_info_resolver.py` | `WorldInfoProcessor` (scan depth, token budget, recursive scanning), `apply_world_info_to_message()` |
| Lorebook import | `Character_Chat/world_book_import.py` | `normalize_world_book_import`, `character_book_to_world_book_block` |
| Chat dictionaries | `Character_Chat/Chat_Dictionary_Lib.py` | `ChatDictionary`, `apply_strategy`, `apply_active_chatdicts_to_text` |
| Persona store | `Character_Chat/local_character_persona_service.py` | `LocalCharacterPersonaService` — personas in an atomic JSON store |
| Roleplay identity | `Chat/console_roleplay_identity.py` | `expand_character_template()`, `effective_user_display_name()`, `ConsolePresentationContext` |
| Emotes/expressions | `Character_Chat/emote_directives.py`, `Chat/character_expression_playback.py` | `CharacterEmoteEvent` (stream parser), `prepare_expression()` |
| Buddies | `Persona_Buddy/` (`controller.py`), `Character_Chat/buddy_conversion.py` | `PersonaBuddyController`, `convert_buddy` |
| Management UI | `UI/Screens/personas_screen.py` | `PersonasScreen` — characters, personas, dictionaries, behavior profiles |
| Conversation navigation | `Character_Chat/character_conversation_navigation.py` | `CharacterConversationCursor` (ADR-120 semantic search pickers) |

## Card import (dataflow)

1. **Entry**: the Personas screen opens a file via an app-owned import worker (so import survives screen unmount); the Console also offers a character switcher.
2. **Container decode** (`extract_json_from_image_file`):
   - PNG: card JSON in `tEXt`/`zTXt`/`iTXt` chunks under keys `chara` / `ccv3`. SillyTavern writes these chunks **after IDAT**, so a forced `img_obj.load()` re-probe is required; decode is pixel-bounded against decode bombs.
   - WebP/JPEG: base64 card JSON in the EXIF `UserComment` tag.
3. **JSON parse** (`import_character_card_from_json_string`): V2 detection via `spec == "chara_card_v2"`, `spec_version` starting `"2."`, or a heuristic `data` node with `name`. Structural validation is advisory — parsing continues leniently; only `name` is strictly required. The field map normalizes to the DB schema (`first_mes` → `first_message`, `mes_example` → `message_example`, …). Fallback chain: V1 parser → the multi-format detector, with an explicit reject when the detector's "Unknown" placeholder is the only name (prevents merging nameless cards).
4. **V3**: there is no dedicated v3 parser module (the empty `ccv3_parser.py` placeholder was deleted, task-34894). V3 support is implicit: `ccv3` chunks are read, `spec == "chara_card_v3"` is accepted, and V3 bodies ride the lenient V2 reader.
5. **Embedded lorebook**: a V2 `character_book` (top-level or legacy `extensions.character_book`) converts into a managed character world-book snapshot; the legacy key is dropped only when at least one entry imported.
6. **Persist**: `Chat_Functions.save_character` upserts by name with `expected_version` optimistic concurrency, normalizes alias keys, decodes base64 `image` to bytes.

## The single card→prompt composer

`compose_character_card_template()` is the **only** card-to-prompt joiner (shared by Console seeding and the character-probe eval engine so they cannot drift). Fixed order: `system_prompt` verbatim, `Personality:`, `Description:`, `Scenario:`, `Example dialogue:`, `post_history_instructions` verbatim — joined by blank lines, with presence checks so whitespace-only fields vanish.

Placeholder expansion (`replace_placeholders`) handles `{{char}}`, `{{user}}`, `{{random_user}}`, `<USER>`, `<CHAR>`, and the aliases `{{character}}`/`{{persona}}` — **both aliases resolve to the character's name, never the user's**. Macros are resolved before text reaches session settings so they never leak verbatim into provider payloads, but stored templates keep macros unresolved for trusted provenance (ADR-046). A card with no prompt-bearing text at all seeds the Console-owned fallback `"Stay in character."`; the seeded greeting comes from `first_message`.

## Character vs persona vs buddy

- **Character** — the AI-side card: multi-field, stored in the ChaChaNotes SQLite tables, session key `assistant_kind="character"` + `character_id`; seeds both system prompt and greeting; drives emotes/expressions and roleplay transcript styling; sessions take the plain provider path (no tools).
- **Persona** — the user-profile side of the identity split (ADR-037): no multi-field join and no greeting; contributes its `system_prompt` verbatim. Stored in a JSON persona store file (not the character tables); sessions carry `persona_memory_mode` (read-only / read-write).
- **Buddy** — a persona-attached companion with visuals, speech, and independent conversations (ADR-139); sprite conversion via `buddy_conversion`; a character+buddy is bundled (Pixel Migu, ADR-122) with Petdex publication support (ADR-146).

All three are managed in the Personas screen.

## Search activation

The pure `UI/Persona_Modules/roleplay_list_window_state.py` owns the existing
single `WindowIntent`; `roleplay_frame_state.py` re-exports its public types and
helpers with identical object identity. Search Enter owes the first match and
returns immediately. A second Enter while matching rows are pending marks that
same intent for one activation; repeats coalesce. Actual current row publication
uses the stateless `persona_search_completion.py` helper to land the cursor and
post a deferred selection. The same owner retains that intent until the screen
accepts its exact published generation, row and cursor; newer user/context
intent, failure or empty results retires it. Thus Ctrl+F, query, Enter, Enter
opens the first match without a handler wait or third Enter, while no pending
activation can open an old row. Ordinary row selection still uses its existing
dirty-editor and selection guards.

## World books and world info

`WorldBookManager` owns world-book CRUD, conversation association, and character attachment; imports normalize lorebook JSON into the managed block shape. At send time, `apply_world_info_to_message` runs the `WorldInfoProcessor`: entries are scanned against recent history with per-book `scan_depth`, `token_budget`, and `recursive_scanning`; matching entries are sorted by `insertion_order` and injected into the ephemeral provider payload (never the stored transcript — see [chat-pipeline.md](./chat-pipeline.md)). Disabled-entry diagnostics surface candidate near-misses.

Lore Settings Save admits one screen-owned worker and releases the screen message
queue before its threaded manager operation finishes. The operation captures its
manager, book identity/version, payload, and exact selected record/panel. Repeated
Save cannot cancel or duplicate an admitted write; cancellation drains an active
native write before reopening admission. Completion adopts the committed version
only for that original selection and panel, retains newer raw controls, and labels
those later edits as unsaved. A retired selection receives feedback naming the
captured book without changing its replacement. This uses the existing Lore
transition/dirty-state ownership; it does not add entry/Options draft domains.
If Save refresh supersedes a submitted Lore search, the existing first-match
intent stays owed until current matching rows publish. Generation, query, mode
and attachment are rechecked after the row-update await. Background Save marks
the loaded book without moving the search cursor or creating load-through intent.

## Chat dictionaries

Dictionaries map trigger phrases to replacement text. `apply_active_chatdicts_to_text` applies active dictionaries to the ephemeral final user message under a strategy (`apply_strategy`; Console constants: max 500 tokens, "sorted_evenly"). Parsing supports user markdown dictionary files; the applier never raises on failure — the payload goes through unchanged. Console dictionary settings are currently hard-coded at the screen; world info is gated by `[character_chat] enable_world_info` (default on).

The mounted Dictionary Settings form acknowledges only the raw controls submitted by Save and adopts the actual committed optimistic-lock version. Controls edited while that Save is pending remain visible and unsaved for the next explicit Save; the existing New, leave, and quit guards continue to detect those settings drafts.

## Expressions and emotes

Cards can drive live expression playback: `emote_directives.CharacterEmoteEvent` parses emote directives **inline during streaming** (inside `append_stream_chunk`, with a fail-closed parser reset), and `character_expression_playback.prepare_expression` selects the sprite/mood for display. Durable emote metadata follows ADR-075.

## Local portrait performance attachments


`Character_Chat/portrait_compositing.py` provides the ADR-243 composition core:
strict source-pixel rectangles, canonical tagged RGBA base identity, inward masks,
one-time raw-edit feathering, binary finished-plane assembly and invariant-checked
component recovery. Original alpha and outside-mask pixels are preserved exactly;
PNG encoding changes do not change prepared pixel identity. It uses the existing
shared decode reservations and bounded still normalizer with explicit image closure.

`PortraitCompositionCandidate` is a disposable immutable snapshot of the recipe,
region revision, finished components and currently available frames. Draft replacement
is authoritative, not a merge; a new rest-only candidate cannot inherit old slots.
Composed publication requires all three components and six roles. Successful manual
imports switch to ordinary six-role readiness; failed edits preserve lineage.
These core contracts and the settings repository are implemented in TASK-36551/36552.
`UI/Persona_Modules/portrait_regions.py` implements source-bound preparation and
independent region Save; `PortraitRegionEditor` contains numeric/two-corner selection,
centered mosaic mapping and a coalesced static source/crop/mask preview. It shares
compositor decode reservations, returns only Rich text and rejects retired render work.
The controller retires the exact modal/sample, reuses canonical settings, recovers
only invariant-checked components and installs a rest-only candidate on changed Save.
Apply preserves candidate provenance; manual imports still drop lineage on success.
Reference preparation is optional: a saved source that exceeds its decode/PNG budget
still opens ordinary manual frames and permits import/Apply/Remove. Only region
selection and component generation are disabled; original limits are unchanged.
`portrait_composition_generation.py` now replaces the historical six-edit path with
three shared-reference requests and one-component dependent retries. Every fresh batch
resets to rest-only before dispatch; candidate identity, source, settings revision and
config fence late work. Manual/composed readiness uses `draft.can_apply`. The actual
`PortraitCompositionReview` consumes immutable original frame PNGs for a labeled 2×3
matrix and common native eye/mouth crops, supports keyboard/numeric pan, and retires
exact displayed owners even beneath another modal. While either authoring dialog is
open, the existing context poll coalesces one owned off-thread source/settings-revision
check. Persisted changes retire the draft/dialogs; a read failure closes the review but
keeps draft frames. Captured epochs/handles reject queued or late old checks, own Save
is fenced, and screen close drains native validation before releasing its owner.
Its output raster is capped at
1 megapixel and native rendering introduces no fitted-thumbnail resampling. Ordinary
Console animation and speech presentation ownership are unchanged.


`Character_Chat/portrait_performance.py` normalizes six aligned still poses
(`rest`, `small`, `open` and their `blink_` variants) into bounded PNGs. Work shares
the existing avatar preparation budget and releases decoded images on failure.
Canonical source identities use the actual resolved expression, source digest and
immutable asset or legacy-image IDs; current and historical resolution of one
asset have the same identity.

Schema 83 adds `character_portrait_performances` and
`character_portrait_performance_frames`. `CharactersRAGDB` Apply/get/Remove APIs
publish all six poses atomically, revalidate source selection and expected revision,
and retain increasing inactive revision tombstones after Remove. Savepoints also
preserve replacement atomicity inside caller-owned transactions. A source change
leaves the old record available for explicit revision-checked replacement; consumers
must match its source before rendering it. Soft deletion preserves rows, while
permanent character deletion cascades through the parent and frames.

Schema 84 (ADR-243) adds `character_portrait_regions`, one source-bound recipe per
character/expression, and nullable `composition_recipe_json` on the performance parent.
`DB/portrait_region_settings.py` validates bounded canonical recipe JSON and prepared
RGBA identity, independently saves regions under source/settings CAS, and atomically
publishes optional provenance with six validated poses under both revisions. Idempotent
current-revision Save leaves region and performance revisions unchanged. Manual Apply
and Remove clear applied provenance; Remove retains regions. Valid unchanged stored
roles are decoded/content-checked and retained byte-for-byte under publication CAS.
New roles still pass the existing normalization contract. Unknown/invalid applied
recipes read as absent, leaving ordinary six-frame artwork usable; component recovery
requires the compositor's complete pixel invariant check. No raw provider edit or
finished-component sidecar is stored. Regions follow character soft-delete/restore and
permanent-delete cascades. The installed recovery catalog retains historical schemas
and has the fixed v82→v83→v84 portrait steps after the v81→v82 Canvas step; native full-profile restore preserves both contracts.

These attachments are profile-local (ADR-242/243). Native full-profile SQLite backup
and recovery schema admission include them; original artwork, character cards,
Actor Packs, Chatbooks and sync formats are unchanged. See [ADR-242](../../backlog/decisions/242-console-radio-portrait-performance.md).

`prepare_performance` reuses existing per-image decode reservations to prepare six
aligned fitted frames off-thread. Small artwork is fitted once, keeping idle offset
in canvas pixels. Current/next pose images share the same 64 MiB active/in-flight
budget; the existing independent shared-decode cap is unchanged. Failures close
partial allocations and retain the selected poster rather than a different talking
face. No pose decode, SQLite lookup or rail remount occurs on the frame timer.

`PortraitPerformanceClock` keeps blink and idle timing independent of mouth pose.
It reads packaged scalar motion tokens once during preparation: blink spacing
3–7 s, hold `$ds-duration-fast`, and zero whole-image idle offset inside
a fixed canvas (TASK-36699). Blinking supplies idle animation; the retained
six-second cycle/helper has zero default amplitude. Hidden time does not advance it. `CharacterExpressionAvatar`
paints at most 30 fps, rejects stale speech and rechecks mouth qualification after
slow rendering so an old loud frame cannot reopen after silence. Unmount waits
for decode/render workers before releasing their input images.

The Console matches the exact local profile/session/character owner and current
canonical source before using a performance. Active revision, inactive tombstone
or explicit absence participates in spec identity. Forced refresh workers have
separate generations, so unchanged artwork keeps its mounted animation while late
replaced work cannot paint. Manual selection wins; owned older-message replay uses
its valid immutable mood and restores the current mood afterward. Missing historical
art never borrows neutral talking frames. Static, Reduce motion, animations-off,
reactions-off and avatar visibility preserve their existing rules. The image viewer
copies the current fitted canvas without acquiring a timer.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Card container has no metadata | Error listing the accepted locations (`chara`/`ccv3` chunk, EXIF UserComment) |
| Oversized/decode-bomb image | Bounded decode refuses |
| V2 validation failures | Advisory warnings; lenient parse continues |
| Dictionary/world-info applier error | Payload passes through unchanged (never raises into the send) |
| Seeding failure on session create | Tolerated; session still created |
| Card name conflict on import | Optimistic-concurrency conflict surfaces; returns None rather than clobbering |

## Governing decisions

ADR-037 (roleplay assistant identity / persona-user profile separation), ADR-046 (display identity and template provenance), ADR-075 (durable character emote metadata), ADR-120 (conversation navigation + local semantic search), ADR-122/139/146 (bundled buddy, independent buddy bindings, Petdex), ADR-144 (expression playback), ADR-149 (persona session identity), ADR-152 (roleplay mode keys). Feature docs: `Docs/Features/World-Lore-Books-Documented.md`, `Docs/Features/ChatDictionaries-Documented.md`, `Docs/User_Guide/roleplay-chat-dictionaries/`.

## Verified gotchas

1. There is no v3 parser module; V3 rides the lenient V2 path.
2. PNG card chunks live after IDAT; Pillow needs the forced `load()` re-probe or embedded cards look absent.
3. `{{persona}}` is a character-side alias (the AI's name), not the user persona.
4. Character sessions bypass the agent bridge (`force_plain`) — no tool calls in roleplay.
5. Macros are expanded before persistence boundaries but stored templates keep them raw (trusted provenance, ADR-046).

`UI/Persona_Modules/portrait_performance.py` owns one disposable authoring draft.
The inline `PortraitPerformanceEditor` emits explicit actions through PersonasScreen;
it has no DB or provider authority. Preparation captures the actual local DB/service,
editor session, canonical persisted source and local authority ID. Native file work
normalizes a bounded still off-thread and recaptures the source before staging;
Apply and Remove use the storage CAS and emit Console invalidation after commit.
The existing Personas context poll discards obsolete drafts without another timer.
Worker draining retains actual publication results through cancellation, and closes
only operation-owned thread-local DB connections. Dirty artwork/bindings require
explicit Save/Apply; unrelated text is never saved by this flow. Preview uses the
same bounded avatar renderer/timer, with explicit offline pose/cycle/idle modes.

`PortraitRadio` projects the current owned snapshot through the character controller
and the existing avatar frame clock/Console sync; it owns no timer or audio service.
Its fixed status/meter rows update without layout invalidation. Fresh qualified PCM
levels fill the bar; stale/unqualified timing hides it with visibility, retaining the
row. Measured compact layout alone removes that row. The existing rail allocator
subtracts frame borders and identification/status rows before contain fitting.
Character captions and workbench labels render as plain text, including card names
that resemble Rich markup. Source styles use design tokens and generated CSS.
