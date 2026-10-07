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
4. **V3**: there is no dedicated v3 parser module (`ccv3_parser.py` is an empty placeholder). V3 support is implicit: `ccv3` chunks are read, `spec == "chara_card_v3"` is accepted, and V3 bodies ride the lenient V2 reader.
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

## World books and world info

`WorldBookManager` owns world-book CRUD, conversation association, and character attachment; imports normalize lorebook JSON into the managed block shape. At send time, `apply_world_info_to_message` runs the `WorldInfoProcessor`: entries are scanned against recent history with per-book `scan_depth`, `token_budget`, and `recursive_scanning`; matching entries are sorted by `insertion_order` and injected into the ephemeral provider payload (never the stored transcript — see [chat-pipeline.md](./chat-pipeline.md)). Disabled-entry diagnostics surface candidate near-misses.

## Chat dictionaries

Dictionaries map trigger phrases to replacement text. `apply_active_chatdicts_to_text` applies active dictionaries to the ephemeral final user message under a strategy (`apply_strategy`; Console constants: max 500 tokens, "sorted_evenly"). Parsing supports user markdown dictionary files; the applier never raises on failure — the payload goes through unchanged. Console dictionary settings are currently hard-coded at the screen; world info is gated by `[character_chat] enable_world_info` (default on).

## Expressions and emotes

Cards can drive live expression playback: `emote_directives.CharacterEmoteEvent` parses emote directives **inline during streaming** (inside `append_stream_chunk`, with a fail-closed parser reset), and `character_expression_playback.prepare_expression` selects the sprite/mood for display. Durable emote metadata follows ADR-075.

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

1. `ccv3_parser.py` is an empty placeholder — do not cite it as the v3 parser; V3 rides the lenient V2 path.
2. PNG card chunks live after IDAT; Pillow needs the forced `load()` re-probe or embedded cards look absent.
3. `{{persona}}` is a character-side alias (the AI's name), not the user persona.
4. Character sessions bypass the agent bridge (`force_plain`) — no tool calls in roleplay.
5. Macros are expanded before persistence boundaries but stored templates keep them raw (trusted provenance, ADR-046).
