# Task 2 brief — Human-readable and derivative privacy

## Outcome

Human-readable and answer-oriented surfaces omit displayable thinking, raw proprietary
continuation, and the proprietary application notice by construction. Importable JSON
and Chatbook remain the explicit sensitive exceptions completed in Task 1. Diagnostic
trajectory/trace formats stay answer-only and must not become an accidental restore
format.

## Known seams

- `ConsoleTranscript.to_plain_text()` currently walks all Assistant-turn activities;
  filter `ConsoleThinkingActivityRef` activities from the export while retaining tools,
  Planning, answers, and explicit selected-Thinking Copy/Inspector behavior.
- `trajectory_export._MESSAGE_KEYS` already excludes thinking. Its V1 validator permits
  additive fields; reject reserved `_thinking`, `thinking_blocks`, and
  `thinking_blocks_json` extensions specifically without breaking general ADR-067
  additive-field compatibility.
- `export_conversation_to_text` and `document_generator` already project visible
  sender/content/timestamp; verify, do not broaden or add post-render scrubbing.
- FTS/search indexes `messages.content`; title derives from user draft; summaries,
  usage, speech, and answer-copy use `message.content`. Add nearest-boundary tests and
  production edits only for proven leaks.
- The explicit selected-Thinking disclosure copy/Inspector path remains allowed.

## Required tests

- Create `Tests/Chat/test_thinking_privacy_surfaces.py` with distinct visible-answer,
  displayable-thinking, raw-private, and proprietary-notice canaries.
- Prove human text/Markdown/document/transcript and trajectory outputs contain only the
  allowed visible projection.
- Prove real FTS/search, title, summary, usage, speech, answer-copy, logs, and errors do
  not consume thinking/private values.
- Decode real main DB, sync payload/outbox, importable JSON, Chatbook, imported rows,
  and human-readable/trajectory outputs; thinking appears only in permitted separate
  fields and private continuation only under ADR-063. The application notice is never
  durable.
- Include mutation/negative controls: proprietary text inside `_thinking` rejects,
  capability-only/no-event fabricates nothing, and a human exporter handed a mapping
  containing thinking still reads explicit visible fields only.
- Preserve trajectory's general additive compatibility while rejecting reserved
  thinking fields at message/top-level/variant locations that could imply a restore
  contract.

## Required verification

`PYTHONPATH=. ../../.venv/bin/python -m pytest -q Tests/Chat/test_thinking_privacy_surfaces.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_trajectory_export.py Tests/Chat/test_trajectory_import.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/UI/test_console_thinking_disclosures.py`

Run scoped Ruff and `git diff --check`. Follow RED/GREEN TDD, self-review against
ADR-090 and ADR-067, update the SDD ledger and create `task-2-report.md`, then commit.
