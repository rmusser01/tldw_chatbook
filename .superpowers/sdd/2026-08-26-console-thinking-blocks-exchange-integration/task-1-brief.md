# Task 1 brief — Importable conversation portability

## Outcome

Preserve canonical version-1 Thinking envelopes and normalized conversation replay policy in ordinary selected-conversation JSON and Chatbook V2. Validate every thinking owner and the policy for one conversation before mutating it. Warn when an importable export contains model thinking or ADR-063 private continuation.

## Required behavior

- Reuse `read_thinking_blocks_json`, `parse_thinking_blocks_json`, and `dump_thinking_blocks_json`; do not invent a second envelope model.
- Ordinary JSON includes `thinking_history_policy` at conversation level and structured `thinking_blocks` on assistant messages. It remains active-path/selected-row only.
- Chatbook V2 includes `thinking_history_policy` at conversation level and structured `_thinking` on every graph message the format already owns. Do not add new session-only variants.
- Export only supported canonical V1. Opaque future durable versions block round-trip export with upgrade-oriented content-free copy.
- Import preflight validates all envelopes, role ownership, provenance, bounds, aggregate UTF-8 bytes, and policy before `add_conversation` or `add_message` for that conversation.
- Unknown bounded string policy normalizes to Auto with a content-free warning. Non-string or oversized policy rejects that conversation.
- A malformed/unsupported envelope excludes the entire conversation without a partial envelope or policy. Unrelated conversations preserve existing importer isolation.
- Shared export warning: `This conversation export contains model thinking or private provider continuation. Treat it as sensitive conversation data.`
- Proprietary evidence remains structurally text-free. Never persist or display the application placeholder through exchange.
- Add tests first and show RED before implementation. Run targeted tests only.

## Likely files

- `tldw_chatbook/Chat/thinking_blocks.py`
- `tldw_chatbook/Chat/Chat_Functions.py`
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- `tldw_chatbook/Chatbooks/chatbook_creator.py`
- `tldw_chatbook/Chatbooks/chatbook_importer.py`
- `tldw_chatbook/Chatbooks/chatbook_models.py` only if its declared metadata shape requires it
- `Tests/Chat/test_thinking_conversation_exchange.py`
- `Tests/Chat/test_provider_continuation_privacy.py`
- `Tests/Character_Chat/test_character_file_operations.py`
- `Tests/Chatbooks/test_chatbook_thinking_round_trip.py`
- nearest existing Chatbook creator/importer/integration tests as required

## Required verification

Run the focused new tests first, then:

`PYTHONPATH=. ../../.venv/bin/python -m pytest -q Tests/Chat/test_thinking_conversation_exchange.py Tests/Chat/test_provider_continuation_privacy.py Tests/Character_Chat/test_character_file_operations.py Tests/Chatbooks/test_chatbook_thinking_round_trip.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_integration.py`

Run scoped Ruff and `git diff --check`. Self-review against ADR-090 and commit the completed slice. Report DONE, DONE_WITH_CONCERNS, NEEDS_CONTEXT, or BLOCKED with commit hash and exact evidence.
