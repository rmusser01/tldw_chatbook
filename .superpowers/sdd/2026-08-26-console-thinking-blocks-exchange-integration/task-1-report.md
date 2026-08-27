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

## Independent reviews

Spec review APPROVED the final range through `406c78bec` after two bounded fix rounds.
The closed gaps were explicit-null envelope rejection, tombstone ownership, fail-
closed policy owner resolution, and empty-string unknown-policy warning. The final
review gate passed 102 focused tests with a clean diff.

Code-quality review APPROVED the same final range with no correctness, privacy,
preflight-ordering, compatibility, maintainability, test-quality, or YAGNI finding.
Canonical UTF-8 accounting, content-free failures, deterministic exchange, and whole-
conversation validation before mutation were all confirmed.

## Spec-review fix round 1

Four boundary issues were addressed with focused regression tests before production
changes.

### RED

- The combined selected-conversation and Chatbook review run collected 39 tests and
  failed **11**: lookup failure did not fail closed, empty policy did not warn, present
  null envelopes bypassed both importers, deleted thinking imported, and the Chatbook
  empty-policy warning was absent.
- The direct tombstone test was refined to place a deleted assistant off the active
  path; it then failed independently because `_thinking` was accepted. This avoids a
  false positive from the existing deleted-active-leaf graph rule.
- The durable-row control was green before production changes, proving
  `soft_delete_message` already clears `thinking_blocks_json`; the defect was confined
  to accepting an invalid external graph.

### Fixes and decisions

- Import envelope checks now distinguish key absence from a present null value.
  `thinking_blocks: null` and `_thinking: null` are malformed, and wrong-role null is
  rejected before conversation mutation.
- V2 graph validation rejects `_thinking` whenever `deleted` is true. The importer
  never constructs a tombstone that retains an envelope.
- Selected JSON resolves the caller-owned conversation record once, using it for both
  title and replay policy. Lookup exceptions raise the content-free error
  `Conversation metadata is unavailable for export.` rather than normalizing an
  unknown owner to Auto. Controls cover both Include and Exclude.
- Policy preflight reserves silent Auto for missing/null. The present bounded string
  `""` follows the ordinary unknown-string path: Auto plus the existing content-free
  warning.

### GREEN and static evidence

- Focused review regression/control run: **43 passed, 1 unrelated dependency
  warning in 5.35s**.
- Task 1 targeted suites plus canonical thinking-policy and Console policy lifecycle
  tests: **221 passed, 1 unrelated dependency warning in 28.54s**.
- Scoped Ruff: **All checks passed**.
- Changed production modules compiled with `py_compile`.
- `git diff --check`: **passed**.

The review fixes do not change the ADR decision: they harden ADR-090's deletion,
ownership, replay-policy, and fail-before-mutation requirements without creating a new
exchange model or broadening human-readable surfaces.

## Spec-review fix round 2

Selected-conversation export now requires resolution of any claimed durable owner.

### RED

- Focused selected-exchange run: **2 failed, 24 passed**. A non-null conversation ID
  without `db_instance`, and a supplied DB whose lookup returned no conversation, both
  completed export with policy Auto.
- The ownerless control (`conversation_id=None`) stayed green, confirming the intended
  legacy boundary before production changed.

### Fix

- `generate_chat_history_content` interprets `conversation_id is not None` as a durable
  ownership claim. It requires a DB instance and a resolved conversation record before
  deriving title or policy. Missing DB, lookup exception, and missing record all fail
  with `Conversation metadata is unavailable for export.` and do not expose lookup
  details.
- `conversation_id=None` remains the explicit ownerless/legacy path and exports Auto.
- Two existing privacy fixtures were brought onto the contract: the durable-owner
  projection supplies a DB stub, while a malformed-history logging test uses the
  ownerless path because it does not exercise conversation persistence.

### GREEN and static evidence

- Focused exchange suite: **26 passed, 1 unrelated dependency warning in 4.15s**.
- First expanded compatibility run: **222 passed, 2 failed**; both failures were the
  stale owner fixtures described above.
- Final Task 1 plus nearest policy suites: **224 passed, 1 unrelated dependency warning
  in 30.37s**.
- Scoped Ruff: **All checks passed**.
- Changed production module compiled with `py_compile`.
- `git diff --check`: **passed**.

Round 2 remains a direct hardening of ADR-090's importable ownership and replay-policy
contract; no new ADR or exchange shape is required.
