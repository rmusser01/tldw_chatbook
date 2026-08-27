# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-thinking-blocks-exchange-integration.md

Setup: isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-thinking-blocks`, branch `codex/console-thinking-blocks`.

Dependencies: TASK-18932.1 complete at `9906b4d4bd`; TASK-18932.2 complete at `7ec7fbc9dc`; TASK-18932.3 complete at `402ec260b4`. Existing ADR-090 governs; no new ADR is required.

Execution: serial subagent-driven development with RED/GREEN evidence, spec review, then code-quality review for each task. Full-suite verification remains excluded unless the user opts in.

Task 1: round-trip supported thinking and replay policy through selected-conversation JSON and Chatbook V2, with whole-conversation preflight and shared sensitivity warnings.

Task 1 initial implementation: commit `3cda0c166a`. Selected JSON RED was 11
failures; Chatbook V2 RED was 10 failures. The final implementer gate was 165 passed
with scoped Ruff, `py_compile`, and `git diff --check` clean.

Task 1 spec-review fix round 1: commit `94db09482c`. Present-null envelopes now
reject in both formats, deleted V2 rows cannot carry thinking, raised metadata lookups
fail closed, and empty-string policy warns while normalizing to Auto. RED was 11
focused failures; GREEN was 43 focused and 221 Task-1/policy cases.

Task 1 spec-review fix round 2: commit `406c78bec`. Any non-null selected-conversation
owner now requires a DB and resolved conversation record; only an explicitly ownerless
export keeps legacy Auto. RED was 2 failures; GREEN was 26 focused and 224 Task-1/
policy cases.

Task 1 final spec review: APPROVED at `406c78bec`. The reviewer reran 102 focused
portability/continuation cases and confirmed canonical V1-only exchange, role/tombstone
ownership, raw policy validation, aggregate preflight, warning privacy, and per-
conversation isolation. `git diff --check` and worktree state were clean.

Task 1 code-quality review: APPROVED at `406c78bec`. The reviewer found no correctness,
privacy, preflight-ordering, compatibility, maintainability, test-quality, or
over-engineering issue. The same 102 focused tests, scoped Ruff, and diff check passed.

Task 2: lock human-readable transcript/text/document and diagnostic trajectory output
to visible answers, then prove derivative privacy through real FTS/search, title,
summary, usage, speech, answer-copy, logging, and durable-owner controls.

Task 1 RED (2026-08-26): `Tests/Chat/test_thinking_conversation_exchange.py`
failed 11/11 and `Tests/Chatbooks/test_chatbook_thinking_round_trip.py` failed
10/10 before production changes. Failures showed absent policy/thinking projections,
missing shared warnings, unsupported-version export not refusing, and imports that did
not yet restore or preflight thinking data.

Task 1 GREEN (2026-08-26): the required portability, provider-continuation,
Character Chat, Chatbook creator/importer/integration, and nearest reachable
assistant-generation-state suites passed: 165 passed. A continuation-only warning
control subsequently passed with its companion privacy suite: 56 passed. Scoped Ruff
and `git diff --check` passed. The dependency environment reports one pre-existing
RequestsDependencyWarning; no test failures remain.

Task 1 implementation: selected-conversation JSON now projects normalized
conversation policy and canonical structured assistant thinking while preserving its
caller-supplied selected/active rows. `save_chat_history` passes the caller-owned DB
instance through to content generation. Character Chat performs complete policy and
thinking staging before its existing conversation transaction. Chatbook V2 projects
thinking for every message graph owner it already exports and stages policy plus all
envelopes before its existing per-conversation transaction. Both formats use the
shared sensitivity warning for thinking or ADR-063 continuation, reject opaque future
envelope versions with upgrade-oriented content-free copy, and preserve unrelated
conversation isolation.

Task 1 ADR check: no new ADR. The implementation follows ADR-090's dedicated
assistant-owned thinking envelope, normalized conversation replay policy, importable
round-trip and sensitivity boundary while retaining ADR-063 continuation as a
separate `_private` owner. Human-readable/derivative surfaces remain Task 2.

Task 1 spec-review fix round 1 (2026-08-26): four boundary findings were verified and
fixed red-first. The combined selected-JSON/Chatbook regression run was RED with 11
failures; an off-active-path tombstone refinement independently failed before the
fix. Present-null `thinking_blocks`/`_thinking` values now reject by key presence,
including wrong-role null. Chatbook V2 rejects thinking on deleted graph rows, while a
durable soft-delete control proves the normal DB path clears the envelope. Selected
JSON resolves DB-owned metadata once and fails with content-free copy when lookup
fails instead of silently exporting Auto. Explicit empty-string policy is now an
unknown bounded string that warns and normalizes to Auto; missing/null remains silent
Auto. Focused review tests passed 43/43; the Task 1 suites plus nearest canonical and
Console policy lifecycle tests passed 221/221. Scoped Ruff, production `py_compile`,
and `git diff --check` passed; the existing RequestsDependencyWarning remains the only
warning.

Task 1 spec-review fix round 2 (2026-08-26): selected JSON now treats every non-null
`conversation_id` as a claimed durable owner. RED showed 2 failures: a missing DB
instance and a DB lookup returning no record both silently exported Auto. The export
now requires the DB instance and resolved conversation or raises the same content-free
metadata-unavailable error; only `conversation_id=None` retains the legacy ownerless
Auto path. Focused exchange tests passed 26/26. The first expanded run correctly
identified two existing privacy-test fixtures that claimed an owner without supplying
its DB (222 passed, 2 failed); those fixtures now either provide the owner or use the
ownerless contract. The final Task 1 plus policy suites passed 224/224 with only the
existing RequestsDependencyWarning. Scoped Ruff, production `py_compile`, and
`git diff --check` passed.
