# Task 2 report — human-readable and derivative privacy surfaces

## Outcome

Task 2 keeps human-readable and answer-oriented derivatives answer-only by
construction. `ConsoleTranscript.to_plain_text()` now ignores typed
`ConsoleThinkingActivityRef` entries before activity rendering while retaining
Planning activities, tools, assistant answers, selection guides, and the existing
pruning/full-history projection. Explicit selected-Thinking Copy and Inspector remain
unchanged.

Trajectory V1 remains a diagnostic, answer-only format. Its shared validator now
rejects `_thinking`, `thinking_blocks`, and `thinking_blocks_json` at the top-level,
message, variant-set, and mapping variant-value boundaries. Rejections identify only
the field location and never echo its value. Unrelated additive fields remain valid,
preserving ADR-067 compatibility.

The existing Character Chat text/Markdown and document-context exporters already
iterate explicit visible fields. Their production code did not need changes. The
selected-conversation JSON and Chatbook V2 formats from Task 1 remain the explicit
importable sensitive exceptions and retain their approved warning and round-trip
contract.

## RED and GREEN evidence

- Initial focused RED: 14 failed, 1 passed. All 12 reserved trajectory mutations and
  the shared import-validator mutation were accepted. The transcript regression,
  after correcting its activity-order fixture, failed genuinely with
  `AttributeError` when `to_plain_text()` reached a `ConsoleThinkingActivityRef`.
  The unrelated additive-field compatibility control was already green.
- Minimal-fix focused GREEN: 15 passed, with one pre-existing dependency warning.
- Privacy-inventory GREEN: 5 passed.
- Required Task 2 gate:
  `PYTHONPATH=. ../../.venv/bin/python -m pytest -q`
  `Tests/Chat/test_thinking_privacy_surfaces.py`
  `Tests/Chat/test_provider_continuation_privacy.py`
  `Tests/Chat/test_trajectory_export.py`
  `Tests/Chat/test_trajectory_import.py`
  `Tests/Chat/test_assistant_generation_state_roundtrip.py`
  `Tests/UI/test_console_thinking_disclosures.py`
  completed with **99 passed** and one pre-existing `RequestsDependencyWarning`.
- Scoped Ruff check, formatting checks for the newly formatted scoped files, and
  `git diff --check` passed. The large transcript module retains its established
  surrounding formatting so the production diff remains two intentional lines.

After the successful pytest result, pytest also reported ambient cleanup warnings for
inaccessible pre-existing `garbage-*` temporary directories. The test command exited
zero; these paths and warnings are unrelated to Task 2.

### Spec-review fix round 1

The review found two evidence-fixture gaps; neither reproduced a production leak.

- Transcript RED: the strengthened test failed at
  `assistant.provider_continuation is not None`. The prior fixture had displayable and
  proprietary thinking blocks but no ADR-063 checkpoint, so its raw-continuation
  omission assertion could not prove ownership. Attaching a canonical complete
  Moonshot checkpoint with a distinct raw canary made the same real
  `ConsoleChatMessage`/`ConsoleTranscript.to_plain_text()` path pass while retaining
  answer, Planning, and tool output.
- Diagnostic RED: the expanded inventory ran 6 passed and 1 failed. The malformed
  export and import log checks were already content-free with all three canaries; the
  failing Chatbook mutation's input-presence assertion showed that fixture still
  lacked the exact application notice. Placing the raw continuation and exact notice
  together in the invalid proprietary-thinking text completed the mutation without
  changing production behavior.
- Focused GREEN: the transcript regression passed 1/1 and the privacy inventory
  passed 7/7.
- Final six-file gate: **101 passed** with one pre-existing
  `RequestsDependencyWarning`; the same ambient temporary-directory cleanup warnings
  followed the successful result.

Representative malformed-boundary coverage is now explicit: a human JSON exporter
receives a malformed mapping containing canonical displayable thinking, canonical raw
continuation, and the exact application notice and emits only a content-free warning;
a malformed JSON import stream contains all three and emits only safe operation/source
and `JSONDecodeError` context; and Chatbook graph validation receives all three and
raises only its generic error. This is intentionally not a claim that every diagnostic
boundary carries all three canaries; trajectory location tests retain their separate
reserved-field value canary.

## Privacy inventory

Four distinct canaries prove ownership: a visible answer, displayable thinking, raw
ADR-063 provider continuation, and the application-only proprietary notice.

| Surface / owner | Visible answer | Displayable thinking | Raw continuation | Application notice |
| --- | --- | --- | --- | --- |
| Main message row | `content` | `thinking_blocks_json` only | `provider_continuation_json` only | absent |
| Main DB `sync_log` projection | `content` | separate thinking field | separate continuation field | absent |
| Encrypted Sync V2 outbox payload | `content` | separate thinking field | separate continuation field | absent |
| Selected-conversation JSON | visible history content | approved `thinking_blocks` exception | approved `_private` exception | absent |
| Imported selected-JSON row | `content` | restored separate field | restored separate field | absent |
| Chatbook manifest/archive | visible message content | approved `_thinking` exception | approved `_private` exception | absent |
| Imported Chatbook row | `content` | restored separate field | restored separate field | absent |
| Text, Markdown, document context | present | absent | absent | absent |
| Console plain transcript | present | absent | absent | absent |
| Trajectory memory/file output | present | absent | absent | absent |
| FTS/search | answer searchable | not searchable | not searchable | not searchable |
| Title, summary, usage, speech, answer-copy, repr | visible/safe metadata only | absent | absent | absent |
| Logs and errors | safe operation/location context only | no value echo | no value echo | absent |

The required companion privacy suite exercises additional malformed import/export
logs and error paths, FTS, render/repr/copy, and document context with private
canaries. The new inventory exercises the three-canary representative malformed
boundaries above and decodes the real durable DB, sync, importable, archive, and
imported-row owners instead of relying only on serialized aggregate searches.

## Mutation and negative controls

- Proprietary raw text inserted into Chatbook `_thinking` is rejected with a
  content-free error.
- Each reserved trajectory field is rejected at every supported contract location,
  and its value is absent from the exception.
- General non-thinking additive trajectory metadata remains accepted.
- A capability-only `ThinkingCapture` with no thinking event fabricates no envelope.
- A human document exporter handed a mapping containing thinking and continuation
  sidecars still projects only role, timestamp, and visible content.

## ADR self-review

- **ADR-090:** thinking remains assistant-owned structured state, never appended to
  answer text. Human-readable and diagnostic projections omit it by construction;
  importable exceptions remain explicit and warned. The proprietary UI notice is not
  persisted. Explicit selected-Thinking disclosure actions remain allowed.
- **ADR-063:** opaque/private continuation remains a distinct owner and never appears
  under thinking or answer-oriented derivatives.
- **ADR-067:** only the three thinking-reserved names are denied at restore-contract
  locations; arbitrary unrelated additive fields remain compatible.
- **ADR required:** no. This task directly implements ADR-090 and preserves ADR-063
  and ADR-067 rather than introducing a new architectural decision.

## Files

- `tldw_chatbook/Widgets/Console/console_transcript.py`
- `tldw_chatbook/Chat/trajectory_export.py`
- `Tests/Chat/test_thinking_privacy_surfaces.py`
- `Tests/Chat/test_trajectory_export.py`
- `Tests/Chat/test_trajectory_import.py`
- `Tests/UI/test_console_thinking_disclosures.py`

Task 3 remains out of scope.
