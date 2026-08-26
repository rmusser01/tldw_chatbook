# Console Thinking Exchange, Privacy, and Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve supported model thinking and replay policy in conversation formats meant for later import, exclude it from human-readable and derivative answer surfaces, and prove the complete feature across real persistence owners and Console lifecycle boundaries.

**Architecture:** Reuse the canonical envelope parser/dumper at every exchange boundary. Importable conversation JSON and Chatbook V2 carry a structured thinking object plus conversation policy and sensitivity warning; each importer validates the complete conversation before its existing per-conversation transaction. Human-readable text/Markdown, diagnostic trajectory views, search, titles, summaries, speech, and answer-copy stay on explicit visible-content projections. Joined tests decode the real main DB, sync outbox, archives, and imports rather than trusting one table.

**Tech Stack:** Python 3.11+, JSON/ZIP, SQLite, existing Chatbook and conversation exporters/importers, pytest, Textual Pilot for joined verification.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md`

**Task:** `backlog/tasks/task-18932.4 - Complete-thinking-exchange-privacy-and-integration.md`

## Global Constraints

- TASK-18932.1, TASK-18932.2, and TASK-18932.3 must be complete.
- Classify by purpose, not filename extension. Conversation JSON/Chatbook V2 that restores durable conversations includes thinking; text/Markdown and read-only diagnostic trajectory/trace exports omit it by default.
- Do not add session-only superseded Console variants to a format that does not already own them. Chatbook V2 preserves its existing complete DB graph; selected-message JSON preserves only selected durable rows; trajectory preserves only the variants its current caller explicitly supplies and still excludes thinking because it is a read-only diagnostic view, not a conversation restore format.
- Export only supported canonical envelope version 1. An opaque newer durable version blocks round-trip export with an upgrade message rather than being down-converted or exposed.
- Import validates envelope, policy, provenance, role ownership, bounds, and aggregate bytes for all messages in one conversation before opening that conversation's transaction. One invalid conversation does not partially import; unrelated conversations keep existing importer isolation.
- Sensitivity warning is present when either displayable thinking or ADR-063 private continuation exists. Proprietary evidence alone is text-free but still part of the conversation record; include it in the warning inventory as thinking evidence.
- Human-readable output never includes displayable thinking or `Proprietary thinking obfuscated - not available`, even if the global display setting is On.
- Search/FTS/title/summary/log/error/usage/speech and answer-copy consume visible answer content. Do not solve exclusion with a fragile post-render string scrubber.
- Privacy tests decode every default durable owner and use mutation controls: deliberately remove each filter/evidence condition in a local test fixture and prove the assertion would fail.
- Use isolated temporary profiles/databases for joined migration and live verification. Ask before any full suite.

---

### Task 1: Round-trip supported thinking in conversation JSON and Chatbook V2

**Files:**
- Modify: `tldw_chatbook/Chat/thinking_blocks.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_models.py` only if manifest metadata has a declared sensitivity inventory
- Modify: `Tests/Chat/test_provider_continuation_privacy.py`
- Create: `Tests/Chat/test_thinking_conversation_exchange.py`
- Modify: `Tests/Character_Chat/test_character_file_operations.py`
- Create: `Tests/Chatbooks/test_chatbook_thinking_round_trip.py`
- Modify: `Tests/Chatbooks/test_chatbook_creator.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`
- Modify: `Tests/Chatbooks/test_chatbook_integration.py`

**Interfaces consumed:** canonical thinking parse/dump/read; conversation policy normalizer; existing private continuation exchange patterns.

**Interfaces produced:** one structured exchange projection, import preflight, shared sensitivity warning.

- [ ] **Step 1: Write failing selected-conversation JSON tests.** Export/import a selected assistant row with displayable and proprietary blocks plus Include policy; assert exact text/status/provenance/policy restore. Cover no thinking, Auto/NULL, an unknown bounded string policy normalizing to Auto with a content-free warning, non-string/oversized policy rejection, and private continuation plus thinking in one record.

```python
THINKING_EXPORT_WARNING = (
    "This conversation export contains model thinking or private provider "
    "continuation. Treat it as sensitive conversation data."
)

def thinking_envelope_to_exchange(raw_json: object) -> dict[str, object] | None:
    result = read_thinking_blocks_json(raw_json)
    if result.opaque_json is not None:
        raise ThinkingEnvelopeVersionError(
            "Upgrade Chatbook before exporting this conversation's thinking data."
        )
    canonical = dump_thinking_blocks_json(result.envelope)
    return cast(dict[str, object], json.loads(canonical)) if canonical else None
```

- [ ] **Step 2: Run exchange tests and confirm fields are absent.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_conversation_exchange.py Tests/Character_Chat/test_character_file_operations.py -k thinking -q`

Expected: FAIL.

- [ ] **Step 3: Add the structured exchange shape to importable JSON.** At conversation level use `thinking_history_policy` with normalized stored value (`auto`, `include`, `exclude`). At assistant message level use `thinking_blocks` containing the parsed versioned object, not a JSON-encoded string. Omit the key when NULL. Add `sensitive_data_warning` only when thinking or continuation is present.

- [ ] **Step 4: Preflight before conversation mutation.** Parse every message's `thinking_blocks` by canonical `json.dumps(..., separators=(",", ":"), ensure_ascii=False)` then `parse_thinking_blocks_json`. Reject non-assistant ownership, malformed/unsupported versions, aggregate bound overflow, invalid provenance, text-bearing proprietary blocks, and non-string or oversized policy values. Normalize an unknown bounded policy string to Auto with a content-free warning. Stage canonical JSON and normalized policy before any `add_conversation`/`add_message` call.

- [ ] **Step 5: Extend Chatbook V2 graph export.** Add `thinking_blocks_json` and `thinking_history_policy` to the explicit DB projections. Export each supported assistant envelope under `_thinking` as the structured object, parallel to but separate from `_private`. Include `contains_model_thinking` and the shared warning in manifest/conversation metadata when any displayable or proprietary evidence exists.

```python
if thinking_payload is not None:
    message_data["_thinking"] = thinking_payload
```

Do not place displayable text in Chatbook README, citation Markdown report, manifest titles, or log messages.

- [ ] **Step 6: Extend Chatbook V2 validation/import transaction.** `_validate_v2_conversation_graph` validates all `_thinking` values and aggregate UTF-8 bytes during the same full-graph pass that validates private continuation and links. It returns staged canonical JSON. The existing outer per-conversation transaction inserts the conversation policy and every message envelope; any failure rolls back that conversation.

- [ ] **Step 7: Add negative/import-isolation tests.** Cover unsupported version, wrong role, proprietary text, excess block count/text/aggregate bytes, non-string/oversized policy, unknown bounded policy fallback, graph error after a valid envelope, and two-conversation archive where one invalid conversation does not partially write while the valid one follows existing importer behavior. Assert warnings/logs contain no canary.

- [ ] **Step 8: Run JSON/Chatbook round-trip suites.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_conversation_exchange.py Tests/Chat/test_provider_continuation_privacy.py Tests/Character_Chat/test_character_file_operations.py Tests/Chatbooks/test_chatbook_thinking_round_trip.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_integration.py -q`

Expected: PASS.

- [ ] **Step 9: Commit portability.**

```bash
git add tldw_chatbook/Chat/thinking_blocks.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chatbooks/chatbook_creator.py tldw_chatbook/Chatbooks/chatbook_importer.py tldw_chatbook/Chatbooks/chatbook_models.py Tests/Chat/test_thinking_conversation_exchange.py Tests/Chat/test_provider_continuation_privacy.py Tests/Character_Chat/test_character_file_operations.py Tests/Chatbooks/test_chatbook_thinking_round_trip.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_integration.py
git commit -m "feat: round-trip Console thinking in conversation exports"
```

---

### Task 2: Lock human-readable and derivative surfaces to visible answers

**Files:**
- Modify: `tldw_chatbook/Chat/document_generator.py` only if an explicit visible-content helper is needed
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `tldw_chatbook/Chat/trajectory_import.py` only if the validator must explicitly reject unexpected thinking in the diagnostic format
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `Tests/Chat/test_provider_continuation_privacy.py`
- Create: `Tests/Chat/test_thinking_privacy_surfaces.py`
- Modify: `Tests/Chat/test_trajectory_export.py`
- Modify: `Tests/Chat/test_trajectory_import.py`
- Modify: `Tests/Chat/test_assistant_generation_state_roundtrip.py`
- Modify: `Tests/UI/test_console_thinking_disclosures.py`

**Interfaces consumed:** visible assistant `content`, thinking sidecar, existing answer-copy/speech/title/summary paths.

**Interfaces produced:** explicit answer-only behavior and decoded privacy inventory.

- [ ] **Step 1: Write failing privacy matrix tests.** Seed three canaries: visible answer, displayable thinking, and raw proprietary continuation. Assert each surface's allowed set:

| Surface | Visible answer | Displayable thinking | Raw proprietary | Proprietary notice |
| --- | --- | --- | --- | --- |
| Main message row | yes | separate field | continuation only when required | no |
| Sync v2 payload | yes | separate field | continuation only when required | no |
| Importable JSON/Chatbook | yes | yes + warning | private continuation + warning | no |
| Human text/Markdown/document | yes | no | no | no |
| Trajectory/trace diagnostic | yes | no | no | no |
| FTS/search/title/summary/log/error/usage/speech/answer-copy | yes where applicable | no | no | no |
| Thinking disclosure | no answer concatenation | yes when enabled | no | app copy only for evidence |

- [ ] **Step 2: Run the privacy tests and record every leak/absence failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chat/test_provider_continuation_privacy.py -q`

Expected: FAIL until all new owners/exports are classified.

- [ ] **Step 3: Keep human-readable exporters answer-only by construction.** `export_conversation_to_text`, Character/Persona Markdown, `DocumentGenerator`, transcript text export, and ordinary answer clipboard read only sender/content/timestamp/status as they do today. Add explicit field projections or helper names where ambiguity exists; never attach thinking then redact it later.

```python
def visible_conversation_message(message: Mapping[str, object]) -> dict[str, object]:
    return {
        "sender": message.get("sender", ""),
        "content": message.get("content", ""),
        "timestamp": message.get("timestamp", ""),
    }
```

Use a helper only where two or more exporters currently duplicate a projection; otherwise retain the direct fields per Ponytail.

- [ ] **Step 4: Keep diagnostic trajectory/trace formats thinking-free.** Do not add `thinking_blocks_json` to `_MESSAGE_KEYS`, variant projections, trace events, privacy inventory, or imported snapshots. Validator tests reject an attempted top-level/message thinking extension for version 1 rather than silently treating it as a round-trip conversation format. Document that these are read-only diagnostic/collaboration views.

- [ ] **Step 5: Audit live derivative callers.** Title helper, compaction summary, run summary, citation/side chat, usage ticker, TTS sentence sequencer, answer-copy, and logging receive `message.content` or prepared visible messages only. Add assertions at the nearest shared boundary. Inspector/copy for a selected Thinking disclosure is the one explicit full-thinking action and resolves via its block sidecar.

- [ ] **Step 6: Decode durable owners.** In one test, execute the real local flow and inspect:

  - ChaChaNotes `messages` and `conversations` rows;
  - trigger-authored `sync_log` record;
  - Sync v2 outbox envelope/receipt database;
  - exported selected-conversation JSON;
  - unzipped Chatbook V2 conversation JSON and manifest;
  - imported target rows;
  - human text/Markdown/document/trajectory files;
  - captured logs and raised error strings.

Assert displayable text appears only in permitted separate fields/importable formats and proprietary text only in ADR-063 continuation owners. Assert the exact application notice is never durable.

- [ ] **Step 7: Add mutation controls.** Build one invalid fixture that places proprietary text in `_thinking`, one capability-only response with no event, and one human-readable exporter deliberately handed a mapping containing thinking. Tests must fail if parsers accept proprietary text, if capability fabricates evidence, or if exporters iterate all mapping values.

- [ ] **Step 8: Run privacy and derivative suites.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_trajectory_export.py Tests/Chat/test_trajectory_import.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/UI/test_console_thinking_disclosures.py -q`

Expected: PASS.

- [ ] **Step 9: Commit privacy boundaries.**

```bash
git add tldw_chatbook/Chat/document_generator.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Chat/trajectory_import.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_trajectory_export.py Tests/Chat/test_trajectory_import.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/UI/test_console_thinking_disclosures.py
git commit -m "test: lock model thinking to permitted surfaces"
```

---

### Task 3: Prove backend refusal, lifecycle integration, and user documentation

**Files:**
- Create: `Tests/Integration/test_console_thinking_end_to_end.py`
- Modify: `Tests/Chat/test_console_thinking_persistence.py`
- Modify: `Tests/Chat/test_console_thinking_history.py`
- Modify: `Tests/Sync_Interop/test_console_thinking_sync.py`
- Modify: `Tests/UI/test_console_thinking_disclosures.py`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/console/context-and-rag.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`

**Interfaces consumed:** complete feature seams from all children.

**Interfaces produced:** joined evidence and user-facing behavior contract.

- [ ] **Step 1: Write a failing joined lifecycle test.** With real temporary ChaChaNotes and Sync v2 stores plus injected provider events:

  1. create a conversation and stream displayable thinking + answer;
  2. assert live expansion then terminal auto-collapse;
  3. restart/hydrate and assert historical collapsed restore;
  4. switch policy Include and prepare a compatible local request;
  5. assert exact replay appears once in counted/dispatched payload;
  6. export Chatbook/JSON, import to a second DB, and repeat hydration;
  7. edit/delete and assert generation provenance clears/removes;
  8. repeat stop, handled failure, proprietary evidence, and no-evidence turns.

- [ ] **Step 2: Run the joined test and confirm any missing integration seams.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Integration/test_console_thinking_end_to_end.py -q`

Expected: FAIL until every child joins cleanly.

- [ ] **Step 3: Add the backend pre-provider refusal integration case.** Use a persistent fake reporting no/old thinking round-trip version and a resolution with adapter-owned displayable/proprietary disposition. Invoke the real controller send path and assert the provider spy remains untouched, the user receives upgrade-oriented content-free copy, no synthetic assistant evidence is persisted, and the draft/send remains recoverable. Repeat v1, ephemeral, and ignored-disposition controls to prove they dispatch.

- [ ] **Step 4: Add conflict and opaque-version joined cases.** Import/sync unsupported version rejects before mutation. A locally existing opaque newer version survives feedback/title/unrelated content writes byte-for-byte, is not rendered/replayed, and blocks regenerate/edit/generation replacement. Whole-record sync conflict never combines answer from one side with thinking/continuation from the other.

- [ ] **Step 5: Write concise user documentation.** Document:

  - Thinking means actual adapter-reported evidence, not inferred chain-of-thought;
  - displayable blocks versus exact `Thinking · unavailable` notice;
  - expanded-live then one-time auto-collapse and manual behavior;
  - default-on `Show model thinking` as presentation-only;
  - Auto/Include/Exclude and effective Required history policy, including Save as default for new conversations;
  - local llama.cpp/vLLM compatibility and adapter-specific exact replay;
  - persistent backend upgrade refusal;
  - importable export sensitivity and human-readable omission;
  - Planning as safe session-only activity, distinct from model thinking.

Do not promise access to hidden chain-of-thought or imply capability proves a turn produced reasoning.

- [ ] **Step 6: Run the complete joined targeted suite.** This is the feature's broadest allowed local gate and is still targeted, not the repository full suite.

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  Tests/DB/test_chachanotes_console_thinking_migration.py \
  Tests/Chat/test_thinking_blocks.py \
  Tests/Chat/test_console_thinking_persistence.py \
  Tests/Sync_Interop/test_console_thinking_sync.py \
  Tests/Chat/test_llamacpp_think_splitter.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_thinking_capture.py \
  Tests/Chat/test_console_thinking_history.py \
  Tests/Chat/test_console_prepared_request.py \
  Tests/UI/test_console_thinking_disclosures.py \
  Tests/UI/test_settings_context_memory_controls.py \
  Tests/UI/test_console_context_controls.py \
  Tests/Chat/test_thinking_conversation_exchange.py \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py \
  Tests/Chat/test_thinking_privacy_surfaces.py \
  Tests/Integration/test_console_thinking_end_to_end.py -q
```

Expected: PASS.

- [ ] **Step 7: Run derived/static checks.**

```bash
.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
.venv/bin/python Scripts/check_persistent_diagnostic_inventory.py
.venv/bin/python -m ruff format --check tldw_chatbook/Chat/thinking_blocks.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chatbooks/chatbook_creator.py tldw_chatbook/Chatbooks/chatbook_importer.py tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Chat/trajectory_import.py Tests/Chat/test_thinking_conversation_exchange.py Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chatbooks/test_chatbook_thinking_round_trip.py Tests/Integration/test_console_thinking_end_to_end.py
.venv/bin/python -m ruff check tldw_chatbook/Chat/thinking_blocks.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chatbooks/chatbook_creator.py tldw_chatbook/Chatbooks/chatbook_importer.py tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Chat/trajectory_import.py Tests/Chat/test_thinking_conversation_exchange.py Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chatbooks/test_chatbook_thinking_round_trip.py Tests/Integration/test_console_thinking_end_to_end.py
git diff --check
```

If the persistent diagnostic inventory changes because a new content-free error is intentionally added, inspect the derived diff, regenerate with the script's documented write flag, and rerun the check; reject unrelated drift.

- [ ] **Step 8: Perform isolated live Console verification.** Run with a scratch data directory and deterministic local thinking/non-thinking fixtures. Wait for authoritative session state, disclosure DOM, and painted output. Record:

  - displayable live expanded then collapsed at answer boundary;
  - proprietary evidence exact copy with no raw text;
  - capable no-evidence turn with no row;
  - setting Off/On immediate behavior;
  - Auto/Include/Exclude/Required policy presentation;
  - restart restore and supported replay;
  - unsupported backend refusal before provider contact.

- [ ] **Step 9: Commit integration/docs and close child/parent only with evidence.**

```bash
git add Tests/Integration/test_console_thinking_end_to_end.py Tests/Chat/test_console_thinking_persistence.py Tests/Chat/test_console_thinking_history.py Tests/Sync_Interop/test_console_thinking_sync.py Tests/UI/test_console_thinking_disclosures.py Docs/User_Guide/console/chat-basics.md Docs/User_Guide/console/context-and-rag.md Docs/User_Guide/console/agent-runs-and-tools.md Docs/User_Guide/settings.md tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md
git commit -m "docs: complete Console thinking integration"
```

Update TASK-18932.4 ACs and Implementation Notes, then set it `Done`. Apply `superpowers:verification-before-completion`; only then update the parent TASK-18932 ACs/notes and set the parent `Done`. A full repository suite remains optional and requires explicit user opt-in.
