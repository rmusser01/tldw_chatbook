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

Task 2 implementation (2026-08-26): plain Console transcripts preserve answers,
Planning, and tools while the real grouping projection excludes thinking references
by construction. Trajectory V1's shared validator rejects the three
thinking-reserved field names at top-level,
message, variant-set, and mapping variant-value locations without echoing values;
unrelated ADR-067 additive fields remain accepted. Existing text, Markdown, document,
FTS, title, summary, usage, speech, copy, log, and error projections were verified as
visible/safe-data-only, so no broader production changes were made. The initial run
was 14 failed and 1 passed: 13 failures were genuine trajectory validation RED, while
the transcript `AttributeError` was later identified as a synthetic-harness artifact;
focused GREEN was 15 passed.
The new real-owner privacy inventory passed 5/5, and the required six-file gate passed
99/99 with only the pre-existing RequestsDependencyWarning. See `task-2-report.md` for
the decoded durable-owner matrix and ADR-090/063/067 self-review. No new ADR is
required; Task 3 remains pending.

Task 2 spec-review fix round 1 (2026-08-26): review identified evidence-fixture gaps,
not product leaks, so production code remains unchanged. The transcript fixture failed
because its assistant owner had no ADR-063 continuation; it now carries a canonical
Moonshot checkpoint with a distinct raw canary. That round still used a synthetic
monkeypatched grouping result; quality review round 2 replaced it with the real owner
path. The diagnostic inventory RED was 6 passed/1 failed because the
Chatbook malformed-error fixture lacked the exact notice; the new malformed export
and import log cases already passed with displayable, raw, and notice canaries. After
completing that fixture, focused GREEN was 7/7 plus the transcript 1/1. The required
six-file gate passed 101/101 with the pre-existing RequestsDependencyWarning; scoped
Ruff/format and diff checks are recorded in `task-2-report.md`. Coverage is stated as
representative rather than falsely claiming every boundary carries all three canaries.
No new ADR is required; Task 3 remains pending.

Task 2 quality-review fix round 2 (2026-08-26): review proved the transcript's typed
thinking guard unreachable because real `ConsoleAssistantTurn.activities` contains
only tool-role `ConsoleChatMessage` instances; thinking refs belong to a separate
interactive rendering/selection projection. The dead guard and synthetic monkeypatch
were removed. A real `set_messages()`/grouping regression now owns displayable,
proprietary, and ADR-063 continuation evidence, verifies the group contains only the
Planning and tool rows, and proves plain text retains those rows and the answer while
omitting all private canaries and the application notice. It passed immediately after
the guard was removed, so this is safe-by-construction verification, not product RED.
Existing testing-evidence lessons already cover fake call-site contracts and harness
bypasses, so no duplicate lesson was added. Trajectory production behavior remains
unchanged. Focused real-path verification passed 1/1, the required six-file gate
passed 101/101, and the Task 1 selected-JSON/Chatbook exchange suites passed 46/46;
scoped Ruff, format, `py_compile`, and diff checks passed. Task 3 remains pending.

Task 2 final spec review: APPROVED at `fad81f6d68`. The reviewer confirmed the
real `set_messages()`/grouping path retains the answer, Planning, and tool rows while
omitting displayable thinking, ADR-063 continuation, and the exact application notice.
The required Task 2 gate passed 101/101 and the Task 1 exchange regression passed
46/46; scoped static checks and the clean worktree were confirmed.

Task 2 final code-quality review: APPROVED at `fad81f6d68`. The original P1 is
resolved: no unreachable transcript guard or synthetic owner remains. The trajectory
validator stays the only Task 2 production change, and its narrow ADR-067-compatible
contract remains approved. The reviewer found no remaining correctness, privacy,
maintainability, test-quality, or YAGNI issue.

Task 3 (2026-08-27): dispatched joined lifecycle/backend-refusal integration and user
documentation. Root retains ownership of the isolated live Console verification and
Backlog closeout after independent specification and code-quality reviews.

Task 3 implementation (2026-08-27): joined lifecycle evidence found two genuine
production seams. Persistent thinking compatibility was checked only after durable
turn acceptance; it now refuses immediately after provider resolution, removes only
the transient optimistic echo, restores pre-send conversation identity/title, keeps
the draft, and contacts no provider. Direct delete, subtree delete, and descendant
tombstoning after content edit cleared thinking but retained the same generation's
provider continuation; all three now clear both fields after the complete prior
generation's sync base hash is captured. No remote adapter or new envelope was added.

Task 3 joined GREEN: the new tracked-lowercase
`Tests/integration/test_console_thinking_end_to_end.py` passed 18/18. The ownership
fix plus nearest DB regressions passed 4/4. Existing opaque future-envelope and
whole-record no-splice sync controls passed 2/2. The required broad targeted gate
passed **1,275**, skipped 2 loopback-listener permission controls, and reported 2
environment warnings in 132.46s. CSS bundle sync, scoped Ruff format/check, relevant
`py_compile`, and `git diff --check` passed. All five required user guides were
updated without promising hidden chain-of-thought. See `task-3-report.md`.

Task 3 diagnostic residual: the plan's `Scripts/` and `Tests/Integration/` casing is
stale; Git tracks lowercase `scripts/` and `Tests/integration/`. The actual lowercase
diagnostic inventory guard reports aggregate drift already present from Task 1 and
formatter rewrites since pin `995036264207f4249fce880c6d288c7a369beb0e`. Its
statement review found no Task 3 logger/sink change, and the tool has no selective
write mode, so the shared inventory was not regenerated. Root retains adjudication,
independent reviews, isolated live verification, and Backlog closeout.
