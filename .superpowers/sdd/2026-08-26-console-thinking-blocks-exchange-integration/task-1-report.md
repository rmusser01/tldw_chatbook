# Task 1 report — Importable conversation portability

## Outcome

Selected-conversation JSON and Chatbook V2 now round-trip supported canonical V1
thinking envelopes and normalized conversation replay policy without expanding either
format's existing message ownership. Importers validate and stage a conversation's
policy and all assistant thinking envelopes before the first conversation/message
write.

## RED evidence

- `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_thinking_conversation_exchange.py -q`
  — **11 failed**. The absent exchange fields, warnings, safe future-version refusal,
  policy fallback/rejection, and preflight/round-trip behavior all failed.
- `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chatbooks/test_chatbook_thinking_round_trip.py -q`
  — **10 failed**. The graph projection, policy, metadata/README warning, future-version
  refusal, staged validation, aggregate bound, and isolation behavior were absent.
- The first compatibility run after implementation was **148 passed, 1 failed**. The
  failure exposed an existing selected-JSON projection mismatch: an assistant state
  was synthesized from a checkpoint even when the source row did not carry the field.
  Restricting that optional projection to source-owned state restored the established
  selected JSON shape; the nearest assistant-state tests remained green.

The worktree-local `.venv` did not contain pytest, so all genuine test evidence used
the repository's main Python 3.12 development environment with `PYTHONPATH=.`.

## Implementation decisions

- Reused `read_thinking_blocks_json`, `parse_thinking_blocks_json`, and
  `dump_thinking_blocks_json` behind small exchange/preflight helpers in the canonical
  thinking module. No parallel envelope type or validator was introduced.
- Exported only canonical V1 structured objects. An opaque future durable version
  blocks export with `Upgrade Chatbook before exporting this conversation's thinking
  data.`; malformed data receives content-free validation copy.
- Selected JSON reads `thinking_history_policy` from the caller-owned conversation DB,
  includes structured assistant `thinking_blocks`, omits the key for NULL, and leaves
  selected/active row choice to its existing caller. `save_chat_history` now passes its
  `db_instance` to content generation.
- Character Chat validates raw policy type/size, role ownership, canonical envelope
  structure/provenance/bounds, and aggregate canonical UTF-8 bytes before entering its
  existing transaction. Unknown bounded policy strings become `auto` with the
  content-free warning `Unknown thinking history policy was reset to Auto.`
- Chatbook V2 adds `thinking_blocks_json` to the existing complete graph query and
  emits `_thinking` on each supported assistant owner. Import graph validation stages
  canonical JSON and aggregate bytes; policy validation completes before conflict
  lookup and the existing per-conversation transaction.
- Kept existing provider-continuation warning fields for compatibility and added the
  exact shared warning to conversation JSON, Chatbook conversation metadata, manifest
  metadata, and README when either thinking evidence or ADR-063 continuation exists:
  `This conversation export contains model thinking or private provider continuation.
  Treat it as sensitive conversation data.`
- Did not change text, Markdown, trajectory, search, summary, speech, or other
  human-readable/derivative surfaces; those remain Task 2.

## GREEN and static evidence

- Required targeted run plus nearest reachable assistant-state and provider private
  round-trip coverage: **165 passed, 1 dependency warning in 23.66s**.
- Continuation-only exact shared-warning controls: **56 passed, 1 dependency warning
  in 6.46s**.
- Scoped Ruff over all changed production and new focused test files: **All checks
  passed**.
- `git diff --check`: **passed**.

The sole warning is the environment's existing RequestsDependencyWarning about the
installed urllib3/chardet/charset-normalizer combination; it is unrelated to this
slice.

## ADR-090 self-review

- Thinking remains a nullable, versioned assistant-generation owner separate from
  visible `content`, local metadata, and ADR-063 continuation.
- Proprietary evidence remains structurally text-free; neither raw values nor the UI
  placeholder enter exchange metadata, warnings, errors, or README content.
- Conversation replay policy round-trips as `auto`, `include`, or `exclude`; legacy
  absence reads as `auto`.
- Importable formats preserve their existing row/graph ownership and reject unknown
  envelope versions before mutation. Unrelated Chatbook conversations retain existing
  per-conversation isolation.
- The warning treats displayable reasoning and private continuation as sensitive
  conversation data without adding a second encryption or ownership system.
- Human-readable and answer-oriented surfaces were not broadened.

ADR required: no new ADR.

ADR path: `backlog/decisions/090-console-thinking-block-ownership-and-replay.md`

Reason: Task 1 directly implements ADR-090's accepted importable exchange boundary and
retains ADR-063's separate continuation ownership; it introduces no new architecture
decision.
