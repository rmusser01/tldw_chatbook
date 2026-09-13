# Server-Compatible Streaming Character Emotes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match the pinned server character-emote protocol so directives drive live Console portraits without entering visible or durable assistant text, while bounded immutable metadata restores the final historical expression.

**Architecture:** Add pure parser/projection and pinned mood-heuristic modules as compatibility authorities. At the async dispatch choke point, capture one immutable actor/active-pack snapshot off-thread, revalidate owning-session authority, use that same snapshot for prompt inventory and store capture, then sanitize every streamed or citation-replaced body before it becomes Console content. The store owns sanitized text, a monotonic content-free live-event feed, and atomic terminal metadata finalization; existing Visual Identity resolution owns immutable asset bytes/display fallback, while `MessageMetadata` remains the local-only durable envelope. No database schema change, server transport, sync payload, Persona Buddy control, or historical beat replay is introduced.

**Tech Stack:** Python 3.11+, Textual 8, SQLite/FTS5, frozen dataclasses, pytest, Ruff, Black-format checks, existing Visual Identity repository/resolver, and the project virtualenv at `../../.venv` from the isolated worktree.

---

## Source, Boundaries, And ADR Check

- Task: `backlog/tasks/task-19060 - Match-server-streaming-emotes-and-persistence.md`
- Approved design: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Governing ADR: `backlog/decisions/075-durable-character-emote-metadata.md`
- Existing Visual Identity ADR: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`
- Pinned upstream repository state: `rmusser01/tldw_server@385afa951922c8a9dc2002c675bb6cad65e4ac23`
- Frozen upstream authorities: `tldw_Server_API/app/core/Character_Chat/emote_directives.py`, `apps/packages/ui/src/utils/character-emotes.ts`, and `apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json` at that commit.

ADR required: no

ADR path: `backlog/decisions/075-durable-character-emote-metadata.md`

Reason: ADR-075 already decides parser semantics, safe prompt projection, live/manual precedence, the local-only metadata envelope, immutable final-expression restore, and the explicit exclusions. The implementation extends existing `messages.metadata_json` data only, so no schema or migration decision is added.

## File Structure

- Create `tldw_chatbook/Character_Chat/emote_directives.py`: pure normalization, UTF-16 accounting, one-shot sanitizer, bounded streaming parser, prompt-state projection, and fixed diagnostic categories.
- Create `tldw_chatbook/Character_Chat/character_mood.py`: exact pure Python port of the pinned server WebUI mood classifier, confidence clamp/rounding, and bounded topic extraction.
- Create `Tests/fixtures/character_emote_directives.json`: reviewed frozen cross-language vectors copied from the pinned upstream corpus and augmented only with named compatibility edge cases required by ADR-075.
- Create `Tests/Character_Chat/test_emote_directives.py`: one-shot, streaming, chunk partition, CRLF, fence, inline, cap, duplicate, cancellation, long-line, final-line, prompt-projection, and mutation tests.
- Create `Tests/Character_Chat/test_character_mood.py`: frozen server heuristic vectors, preceding-user context, punctuation scoring, confidence/rounding, topic bounds, and explicit-bypass authority tests.
- Modify `tldw_chatbook/Chat/message_metadata.py`: add closed bounded emote event/visual metadata value objects and fail-soft JSON decoding inside the existing local-only envelope.
- Modify `Tests/Chat/test_message_metadata.py`: constructor validation, JSON round trip, malformed-load, privacy, and bound mutation tests.
- Modify `Tests/DB/test_chachanotes_message_metadata_migration.py`: real SQLite existing-schema persistence/reload proof for the new JSON fields; explicitly prove no schema bump is required.
- Modify `tldw_chatbook/Chat/console_chat_store.py`: per-message parser capture, sanitized append and citation-body replacement, monotonic content-free event feed, complete/stop/fail/variant atomic terminal handling, ephemeral explicit-state reads, and final metadata attachment before persistence.
- Create `Tests/Chat/test_console_character_emotes.py`: store-level streaming/non-streaming/cancellation/persistence/search-export-sink/variant/history tests at the real append and terminal seams.
- Modify `tldw_chatbook/Chat/console_chat_controller.py`: character-only capture arming, active-pack prompt inventory, provider-visible-text-only boundary, explicit-over-heuristic finalization, and fail-soft resolution.
- Modify `Tests/Chat/test_console_chat_controller.py`: direct character streaming, one-chunk completion, provider control/tool exclusion, failure, retry, and prompt tests.
- Modify `tldw_chatbook/Chat/console_expression_state.py`: precedence-aware operational/explicit/historical state derivation from store snapshots.
- Modify `tldw_chatbook/Character_Chat/visual_identity.py`: allow safe explicit projected slugs to resolve against the active immutable pack without weakening operational aliases or manual precedence.
- Modify `tldw_chatbook/UI/Console_Modules/character.py`: preserve the last painted/base portrait when an accepted explicit state has no asset and consume the new expression state without retaining raw directives.
- Modify `Tests/UI/test_console_character_avatar.py`: sequential live events, manual suppression, missing asset, resolver failure, terminal final state, and history restore with no beat replay.
- Modify `Tests/Architecture/test_persona_buddy_boundary.py` and relevant privacy/diagnostic architecture tests only if their existing scopes need the new module path added.
- Modify the TASK-19060 file throughout execution: status, checklist, verification evidence, and implementation notes.

## Global Invariants

- Parser input is assistant-visible text only. Reasoning, tools, citations, control payloads, Buddy state, and raw provider events never call the parser.
- Any line matching standalone `Emote:` syntax is removed outside fences even when invalid, duplicated, or over cap. Inline and fenced text remain byte-for-byte visible apart from ordinary stream concatenation.
- The streaming buffer is bounded to a possible `Emote:`/fence prefix or an incomplete directive line; ordinary prose is released in the same push without waiting for a newline.
- Cancellation drops the incomplete candidate; successful completion flushes an unterminated final line.
- Offsets count UTF-16 code units in sanitized visible text, are nondecreasing, and never exceed sanitized length.
- Durable metadata contains bounded scalar identities and at most five `{state, at_char}` records. It never contains assistant text, directives, prompts, provider payloads, paths, bytes, display labels, or manual overrides.
- Character-only behavior is armed by the owning session, never the globally active tab. Generic chats and tool rows retain current behavior.
- One immutable run snapshot contains the revalidated owning session/actor identity, active pack/version, ordered assets, safe prompt states, and exact slug-to-asset mapping. Prompt projection, live capture, fallback, and durable identity all consume that same snapshot; no second repository read may silently change authority mid-run.
- Every accepted event receives a monotonic process-local sequence and remains available through a content-free per-session feed until its consumer advances a cursor. Same-chunk events and events accepted immediately before terminal completion therefore cannot collapse into a latest-state poll.
- Manual override affects display only. It never stops parsing, event ordering, final mood, or immutable metadata capture.
- A missing explicit asset keeps the current/base visual, records a fixed fallback category, and suppresses the heuristic.
- History restores only the final immutable identity. It does not replay event beats or physically garbage-collect pack versions.
- All failure logging uses fixed categories and safe identifiers only.
- A parser exception switches the message to an eventless fail-closed sanitizer that continues classifying/stripping standalone control lines for the rest of that message. It never disables sanitization or passes later raw chunks through.
- The store is the sole terminal metadata owner. Every complete/stopped/failed/variant/citation terminal calls one internal atomic finalizer before persistence; controller code never races a second metadata write.

### Task 1: Freeze The Pinned Parser Contract

**Files:**
- Create: `Tests/fixtures/character_emote_directives.json`
- Create: `Tests/Character_Chat/test_emote_directives.py`
- Create: `tldw_chatbook/Character_Chat/emote_directives.py`
- Modify: `backlog/tasks/task-19060 - Match-server-streaming-emotes-and-persistence.md`

- [ ] **Step 1: Add the upstream fixture and one-shot born-red tests**

Copy the six pinned JSON vectors exactly, then add named cases for CRLF preservation after stripped control lines, emoji-before-event UTF-16 offset, 40/41-character states, five/six accepted events, nonconsecutive duplicate acceptance, and an unterminated final directive. Test `parse_character_emote_directives()` against every vector.

- [ ] **Step 2: Run the one-shot test and verify a meaningful RED**

Run: `../../.venv/bin/python -m pytest -q Tests/Character_Chat/test_emote_directives.py -k 'one_shot or frozen'`

Expected: collection fails because `tldw_chatbook.Character_Chat.emote_directives` does not exist; after creating an empty module, behavior assertions fail rather than only import collection.

- [ ] **Step 3: Implement the minimal one-shot authority**

Implement constants `EMOTE_EVENT_LIMIT = 5`, `EMOTE_PROMPT_STATE_LIMIT = 25`, the exact safe-state regex, frozen `CharacterEmoteEvent`/parse-result values, `normalize_character_emote_state`, UTF-16 length via `encode('utf-16-le', errors='surrogatepass')`, fence recognition, and the pinned line parser. Preserve `\r` as part of the line until `.strip()` classification and preserve original visible separators.

- [ ] **Step 4: Prove one-shot GREEN and mutation guards**

Run the focused test, then temporarily mutate the event cap, duplicate guard, fence toggle, and UTF-16 function one at a time to confirm the named tests fail; restore each mutation immediately.

- [ ] **Step 5: Add streaming partition and cancellation born-red tests**

For every frozen input, feed all single split points plus representative one-character and adversarial chunk partitions. Assert concatenated `visible_text` plus `flush()` equals one-shot clean text/events. Add a 100,000-character ordinary line test proving all but the bounded control prefix is released during `push`, and cancellation tests proving incomplete `E`, `Emote:`, and fence candidates emit/persist nothing.

- [ ] **Step 6: Implement the bounded streaming state machine**

Port the pinned TypeScript `maybe`/`ordinary` line-mode behavior, but cap possible-control buffering: only retain whitespace plus exact case-insensitive prefixes for ````` `` and `Emote:` and, after `Emote:`, at most the grammar's maximum candidate plus bounded surrounding whitespace. Once a candidate cannot be valid, classify the eventual standalone control line for stripping without retaining unbounded text; never delay ordinary prose. Provide distinct `flush()` and `cancel()` terminal operations.

- [ ] **Step 7: Run parser GREEN and commit**

Run: `../../.venv/bin/python -m pytest -q Tests/Character_Chat/test_emote_directives.py`

Run: `git diff --check`

Commit: `test/feat: add pinned character emote parser`

### Task 2: Port The Pinned Heuristic And Project Safe Prompt States

**Files:**
- Modify: `Tests/Character_Chat/test_emote_directives.py`
- Modify: `tldw_chatbook/Character_Chat/emote_directives.py`
- Create: `Tests/Character_Chat/test_character_mood.py`
- Create: `tldw_chatbook/Character_Chat/character_mood.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`

- [ ] **Step 1: Write projection RED tests**

Build ordered asset mappings containing canonical keys, `custom:<token>` keys, invalid keys, labels that disagree with keys, aliases, collisions, and non-round-tripping values. Assert only exact round-tripping slugs survive, first stored order wins, the first 25 are shown, and the suffix is exactly ` (+N more)`.

- [ ] **Step 2: Implement pure projection and formatting**

Add `project_character_emote_states(assets)` and `append_character_emote_prompt_instruction(system_prompt, states)`. Derive from `expression_key` only; for `custom:` strip the prefix and require `normalize_expression_key(slug)` to equal the original key; omit any slug mapped by more than one canonical key.

- [ ] **Step 3: Freeze and port the exact mood heuristic**

Transcribe the pinned `apps/packages/ui/src/utils/character-mood.ts` classifier authority: the eight labels, ordered regex groups, question/exclamation fractional scores, stable score tie order, `0.85` neutral threshold, `[0.35, 0.98]` clamp, neutral `0.72` cap, JavaScript-compatible two-decimal rounding, stopword list, first-winner topic counts, and 40-character topic bound. Tests cover empty input, the upstream excited/neutral examples, each label, punctuation, ties, preceding-user influence, topic selection, and bounded scalar output. The classifier accepts sanitized assistant text plus the immediately preceding visible user turn only and logs neither.

- [ ] **Step 4: Run heuristic RED then GREEN**

Run: `../../.venv/bin/python -m pytest -q Tests/Character_Chat/test_character_mood.py`

Expected RED: module missing, then vector assertions fail against a stub. Expected GREEN: all frozen vectors pass. Mutate the threshold, preceding-user inclusion, score order, and rounding individually to prove the tests detect drift.

- [ ] **Step 5: Write controller prompt/snapshot RED tests**

Use a character session with a real/in-memory active Visual Identity graph and a generic session. Assert only the character provider system row gains the instruction, current active-version asset order is used on every dispatch, labels never appear, and stored session prompt remains unchanged. Pause the off-thread snapshot read, change the active actor/session and activate another version, then release it: stale authority must be rejected/retried rather than mixing old inventory with new identities.

- [ ] **Step 6: Capture one off-thread immutable run snapshot and compose just in time**

At `_stream_assistant_response_inner`, before payload dispatch or capture arming, snapshot the owning session's identity revision and actor, read only the active graph's ordered metadata through `VisualIdentityRepository` via `asyncio.to_thread`, and revalidate session/actor/identity revision after the await. Derive safe prompt states and the exact slug-to-immutable-asset mapping once. If the payload has a leading system row, append the instruction there; otherwise insert a new leading system row containing the instruction, matching the pinned helper's empty-base behavior. Never mutate stored settings. Pass this same snapshot into store capture. On stale authority, rebuild once or stop with the existing session-closed/context-changed result; on repository failure, use a content-free no-pack snapshot and fixed fallback. Test local and server-only character sessions with empty stored prompts/no active pack, plus generic sessions that must not gain a row. Never read the repository synchronously in `_resolved_system_prompt` or log prompt/card content.

- [ ] **Step 7: Prove GREEN, authority isolation, and commit**

Run the parser prompt, mood, and controller prompt/snapshot subsets. Mutate the character-session guard, post-await authority fence, shared-snapshot handoff, and asset-key source to prove isolation, TOCTOU, and label-exclusion tests fail.

Commit: `feat: add pinned character emote prompt and mood authority`

### Task 3: Define Bounded Durable Emote Metadata

**Files:**
- Modify: `Tests/Chat/test_message_metadata.py`
- Modify: `Tests/DB/test_chachanotes_message_metadata_migration.py`
- Modify: `tldw_chatbook/Chat/message_metadata.py`

- [ ] **Step 1: Write strict-construction RED tests**

Specify frozen `CharacterEmoteEventMetadata` and `CharacterEmoteMetadata` values. Cover max-five events, safe normalized states, nonnegative/nondecreasing/bounded offsets against supplied sanitized UTF-16 length, final explicit state equals `mood_label`, bounded scalar lengths, closed fallback categories, and immutable profile-local integer identities.

- [ ] **Step 2: Write fail-soft durable-load and privacy RED tests**

Assert malformed nested objects, booleans as integers, over-cap arrays, descending/out-of-range offsets, unknown categories, paths, raw text/directive-like extra keys, and server-shaped IDs discard only the emote subrecord while preserving unrelated valid message metadata. Assert serialized keys contain no forbidden content fields.

- [ ] **Step 3: Implement nested metadata values and JSON integration**

Add an optional `character_emote` field to `MessageMetadata`. Serialize stable scalar fields and event dicts explicitly. Decode through narrow helpers, validate against `sanitized_utf16_length`, and treat unknown future top-level keys as today while rejecting malformed emote records without preventing conversation load.

- [ ] **Step 4: Prove real SQLite round trip without a schema bump**

Persist an assistant row through the existing `metadata_json` column, close/reopen the real SQLite DB, hydrate it, and assert exact emote metadata. Run the existing v30-to-v31 migration test with the new reader and assert the schema version remains unchanged.

- [ ] **Step 5: Run GREEN, mutation proof, and commit**

Run: `../../.venv/bin/python -m pytest -q Tests/Chat/test_message_metadata.py Tests/DB/test_chachanotes_message_metadata_migration.py`

Mutate event bounds, offset ordering, and fail-soft nested handling to prove tests fail.

Commit: `feat: persist bounded character emote metadata`

### Task 4: Sanitize At The Shared Console Stream Seam

**Files:**
- Create: `Tests/Chat/test_console_character_emotes.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`

- [ ] **Step 1: Write store-seam RED tests**

Arm only an assistant message owned by a character session. Feed chunks through `append_stream_chunk` and assert every intermediate message snapshot contains only released visible text, accepted events are observable in order, and generic/tool/system rows remain untouched. Cover one-chunk non-streaming behavior through the same seam.

- [ ] **Step 2: Add per-message capture state and monotonic event feed**

Store a private capture keyed by native message id containing parser, the shared immutable run snapshot, accepted events, last explicit state, base display state, and fixed fallback. Add `begin_character_emote_capture`, fail-soft read accessors, and a per-session `(sequence, message_id, state)` feed with `events_after(cursor)`. Sequence assignment happens for each accepted event in parser order, including multiple events returned by one push. Feed entries survive terminal capture finalization until bounded session/message lifecycle cleanup, so completion-before-next-poll is observable.

- [ ] **Step 3: Transform before the existing buffer append and fail closed atomically**

In `append_stream_chunk`, run only armed assistant chunks through the parser, append only `visible_text`, and advance speech/payload timing exactly once per actual visible change while publishing content-free event tokens in acceptance order. Make `push()` exception-atomic: compute against a cloned/local parser state and commit state/output/events only after the whole push succeeds. On failure, classify it and reprocess the entire uncommitted chunk through an eventless fail-closed sanitizer initialized from the saved pre-push fence/line/offset state; retain that sanitizer for all later chunks. Tests inject faults after ordinary prose before and after a possible directive prefix and assert safe prose is neither lost nor duplicated, while valid/invalid directives in the faulting and later chunks are stripped.

- [ ] **Step 4: Reset and reparse citation-selected bodies**

Write RED tests around `replace_deferred_terminal_body()`: a repaired replacement containing multiple valid/invalid directives must replace provisional visible content, events, offsets, mood, and immutable final identity as one unit and atomically publish its newly accepted events to the monotonic feed in parsed order; canceled citation selection keeps the provisional sanitized capture; a replacement failure leaves the prior safe body/capture/feed intact. Implement a capture reset plus one-shot reparse before assigning `message.content`/stream buffers, then publish replacement events only after the whole replacement commits. Previously displayed provisional events remain historical live observations but never remain in replacement durable metadata. Never retain stale provisional metadata events or accept unsanitized `selected_body`.

- [ ] **Step 5: Write and implement one atomic terminal RED/GREEN behavior**

Create one private `_finalize_character_emote_capture(message, outcome=...)` used internally by `mark_message_complete`, `mark_message_stopped`, `mark_message_failed`, and `finalize_variant_stream` before any materialization/persistence. Successful complete/variant/citation selection calls `flush`; only those successful outcomes obtain the immediately preceding user text and run the pinned heuristic when no explicit event exists. Stop/fail calls `cancel`, never invokes the heuristic, and persists only already accepted explicit events plus bounded fixed outcome/fallback facts. Terminal display is outcome-specific regardless of accepted events: failed always resolves through existing operational `error`, stopped always resolves through existing operational `idle`/stop policy; durable accepted final identity is for later history restore, not the live failed/stopped portrait. Every outcome validates offsets against final sanitized UTF-16 length and attaches `MessageMetadata` once. A first provider chunk on retry atomically performs the existing lazy `prepare_message_retry` reset and begins a fresh capture; a zero-chunk/error retry preserves the prior failed answer/capture. Citation replacement reparses before this same terminal owner runs.

- [ ] **Step 6: Prove every downstream text sink is clean**

Using the real persistence adapter, assert rendered store snapshots, database content, provider-history rebuild, search results, export source rows, and copied variants contain sanitized text only. Test valid, invalid, duplicate, and over-cap directives. Do not special-case downstream consumers; their proof should pass because the store content is already clean.

- [ ] **Step 7: Run GREEN, terminal/citation/failure mutation proof, and commit**

Run: `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_character_emotes.py Tests/Chat/test_console_chat_store.py -k 'emote or citation or stream'`

Expected GREEN: all selected tests pass. Mutate the arm guard, pre-buffer transform order, citation reset/reparse, fail-closed replacement, terminal owner call, completion flush, and cancellation discard one at a time to prove failures.

Commit: `feat: sanitize character emotes at stream storage seam`

### Task 5: Arm Character Runs And Preserve Provider Boundaries

**Files:**
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Architecture/test_persona_buddy_boundary.py`

- [ ] **Step 1: Write controller integration RED tests**

Exercise submit, continue, retry, regenerate, and a whole-response single chunk for character sessions. Assert the owning session—not active tab—controls arming. Interleave fake reasoning, tool-call arguments/results, citations, usage, and control events and prove only visible content deltas reach the parser.

- [ ] **Step 2: Arm from the shared snapshot without breaking lazy retry reset**

Consume Task 2's already-revalidated run snapshot; do not query the repository again. New turns and variants arm after their normal initialization but before prefill/provider text. Preserve retry's current lazy semantics: do not clear/arm the failed row before provider output exists; on the first nonempty provider chunk, atomically call `prepare_message_retry` and arm the fresh capture before parsing that same chunk. A zero-chunk or pre-first-chunk error retains the prior failed answer and metadata byte-for-byte. Server-only characters use a no-local-actor snapshot so parsing and mood/events still work without copying server IDs into local visual identity fields.

- [ ] **Step 3: Delegate all finalization to the store owner**

Controller terminal paths call the existing store terminal APIs only. Tests spy on the pinned heuristic to prove it is never invoked when explicit events exist, invoked exactly once with sanitized assistant text plus the preceding user turn for a successful no-event completion, and never invoked for no-event stopped/failed outcomes. Retry tests prove zero-chunk/error preservation and first-chunk atomic reset/arm. Resolver/heuristic failures record fixed fallback categories and never change sanitized content.

- [ ] **Step 4: Prove provider/tool/Buddy isolation and commit**

Run controller and Persona Buddy boundary tests. Mutate force-plain/character ownership, tool exclusion, and explicit-over-heuristic guards to prove authority tests fail.

Commit: `feat: integrate character emotes with provider replies`

### Task 6: Drive Live Portraits And Historical Final Restore

**Files:**
- Modify: `Tests/UI/test_console_character_avatar.py`
- Modify: `Tests/Character_Chat/test_visual_identity_resolution.py`
- Modify: `tldw_chatbook/Chat/console_expression_state.py`
- Modify: `tldw_chatbook/Character_Chat/visual_identity.py`
- Modify: `tldw_chatbook/UI/Console_Modules/character.py`

- [ ] **Step 1: Apply Impeccable UI guidance before editing the portrait flow**

Run the required Impeccable context command once for the Console target, read the relevant playbook and craft-floor references, and keep the existing compact rail visual design. This task changes state behavior, not layout or visual direction.

- [ ] **Step 2: Write precedence and event-feed RED tests**

Assert: manual override wins display; pending uses thinking; streaming uses speaking until the first event; every subsequent accepted event advances explicit state in order; complete with explicit retains final expression; complete without explicit uses heuristic; failed always uses operational error and stopped always uses the existing idle/stop policy even after accepted explicit events; React-off stays idle. Feed tests deliver two events in one chunk and complete before the next UI tick, then assert both monotonic sequences are consumed in order. A repaired selected body with multiple directives publishes its new events in order after atomic replacement. Manual display suppression still advances the feed cursor and must not alter captured events or durable metadata.

- [ ] **Step 3: Extend safe explicit resolution**

Map a projected slug back to the exact canonical expression key from the capture snapshot. Preserve existing operational mappings and manual normalization. Require active live resolution to match the captured actor; on missing/corrupt assets return a fixed fallback without replacing the current/base painted spec.

- [ ] **Step 4: Restore historical immutable identity only**

When the latest completed assistant message has valid emote metadata, resolve its stored pack/version/expression/asset identity directly against retained local rows and verified bytes. If unavailable, expose deterministic fallback state and retain base portrait. Ignore the historical `emote_events` sequence for rendering; use final identity only.

- [ ] **Step 5: Consume every live event by cursor and prove stale-result fencing**

Have `ConsoleCharacterController` retain a per-session event cursor and drain `events_after(cursor)` in sequence before its normal operational refresh. Await or authority-fence each resolution so a delayed previous decode cannot overwrite a later event/manual selection/session switch; cursor advancement is content-free and monotonic. Use Textual Pilot/store spies to assert two same-chunk events each request their state in order, completion before poll loses neither, manual override suppresses paints but not cursor/event persistence, missing assets keep the prior/base image, and hydration never schedules historical beat transitions.

- [ ] **Step 6: Run Impeccable detector and focused GREEN**

After final UI edits, run the Impeccable detector once and address applicable findings. Run avatar, reaction picker, expression resolver, and emote integration tests.

Commit: `feat: restore final character emote portraits`

### Task 7: Verification, Governance, And Task Completion

**Files:**
- Modify: `backlog/tasks/task-19060 - Match-server-streaming-emotes-and-persistence.md`
- Modify only if evidence warrants: `backlog/docs/lessons-testing-evidence.md` or another existing lessons file.

- [ ] **Step 1: Re-run import provenance in the assigned worktree**

Run: `../../.venv/bin/python -m pytest -q Tests/test_probe_import_provenance.py`

Record the imported package path and worktree branch in TASK-19060 notes.

Run app-importing evidence with isolated roots created under `mktemp -d`, setting `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and `TLDW_CONFIG_PATH` before Python imports Chatbook. Expected: all created config/data files remain below the disposable root and the imported package path resolves inside `.worktrees/task-19060-streaming-emotes`.

- [ ] **Step 2: Run the complete touched component surface**

Run the parser fixture suite; message metadata and real SQLite migration/reload suites; Console store/controller emote suites; visual identity repository/resolver suites; avatar/reaction picker suites; search/export focused tests; Persona Buddy boundary; privacy/diagnostic and architecture tests touched by the change.

- [ ] **Step 3: Run static and formatting gates**

Run Ruff on every touched Python file, Black `--check` on touched clean files, `python -m compileall` on touched packages, `git diff --check`, and the repository's task/governance validators. Do not claim an unrelated full-suite run.

- [ ] **Step 4: Audit durable and diagnostic payloads**

Inventory every new log category and serialized key. Use `rg` to prove no raw directive, assistant content, prompt, provider payload, local path, server ID, display label, manual override, Buddy state, or bytes can reach character-emote metadata or diagnostics.

- [ ] **Step 5: Self-review against all eight acceptance criteria**

Review `git diff origin/dev...HEAD`, map each criterion to tests and production seams, and fix any gap through a fresh RED/GREEN cycle. Confirm no physical version GC, sync, transport, or beat replay was added.

- [ ] **Step 6: Complete task hygiene**

Check every acceptance criterion, add concise Implementation Notes with approach/files/trade-offs/evidence, document the ADR decision, record any justified plan deviation, and set status to Done only after every gate passes. Add a lessons entry only if the work produced a concrete reusable incident.

- [ ] **Step 7: Final commit**

Run `git status --short`, stage only TASK-19060 files, and commit the final evidence/docs update with `docs: complete task 19060 evidence`.
