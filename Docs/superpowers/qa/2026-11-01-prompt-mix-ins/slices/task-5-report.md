# Task 5 implementation report

Status: implemented, self-reviewed; TASK-38540 remains In Progress for controller's independent review. BASE: `19dc3dbb43e5e623d77efc91741b4c49835295fe`. Worktree: `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`. ADR: `backlog/decisions/286-console-prompt-mix-ins.md`; no new ADR. No subagents, push, merge, branch creation or full-suite run.

## Implementation

- Literal Before/user/After composition preserves whitespace, braces and command syntax. Multipart adds text blocks around the original sequence without changing original references. Exact native current-user identity selects the anchor, including unrelated trailing user-role hook rows. A semantic attempt marker preserves the active boundary and prevents wrapping again; serialization strips marker and native identity from provider wire values.
- Frozen supplied/admitted inputs remain authoritative. Original Retry and accepted stopped-empty Resend use the operational service even for an empty original snapshot, after source deletion and capture-off restart. Regenerate, edit-and-resend and explicit Continue capture current choices before supplying complete configuration. Raw corrupt/half/unsupported markers refuse before adapter entry. Restart retry eligibility honors durable failed generation state even when hydrated display status is complete.
- Initial native acceptance contributes the exact attempt in its existing transaction. Replay writes before dispatch; new empty regeneration assistants use existing create_message semantic-sidecar/savepoint ownership with the attempt contribution atomically. Worker I/O uses existing operation-owned DB connections; live native identities publish only after successful transaction and exact session/incarnation/binding checks. Cancellation drains owned writes before publication. Failure preserves draft/recovery and does not enter the adapter; a failed regeneration contribution creates no durable empty assistant.
- Temporary state owns full MixinAttemptWrite batches keyed by attempt ID, retaining repeated same-row Retry identities/anchors/links. Promotion writes all in order via the existing exact-ID mapping. Only validated operational marker pairs are omitted from detached insertion metadata until contributions publish the latest marker; live/generic metadata stays intact. Forced latest-contribution rollback leaves all in-memory batches and no durable attempts.
- Same-row Retry now allows only exact predecessor lineage while keeping old records immutable. The **schema object changes**: unreleased v82 `console_prompt_mixin_message_marker_preserve` trigger admits validated predecessor/successor marker transition; its exact recovery catalog is updated. Schema version stays 82, by controller ruling for this unreleased branch. Arbitrary metadata updates still retain valid markers. Deleting only newest record refuses Retry even with valid predecessor records present.
- Contribution fingerprint uses bounded SHA-256 over every determining frozen field (identity, anchors, predecessor, source identity, version, name, placement, exact body), supporting the existing 20k body limit without exposing bodies in repr or diagnostics. No provider-request archive or duplicate request storage introduced.
- Trace provenance identifies `prompt_mixins` and pins the clean user revision; diagnostic credential/PII policy continues to apply while operational exact bodies remain independent. Existing clean-source restoration accepts a body-free `replacement_attempt_id` only after checking current valid record+marker, no newer same-assistant successor, same conversation/source user, latest complete selected assistant revision, actual next-user parent, and no pending dispatch. Existing origin/terminal/policy/range/descriptor checks remain. Ordinary response-link equality is unchanged. Negative controls cover missing/corrupt/forged/stale/foreign record, wrong parent/sibling/revision and pending generation/dispatch.
- Future history and canonical stored messages stay clean through Send→Retry→Regenerate→next Send with capture On/Off. Prepared accounting includes literal bodies and treats current input as mandatory for overflow. Context preview uses the same source/currentness fences with `publish_facts=False`, so no live preference reconciliation/last-use publication or save scheduling occurs. Normal sends retain publication. No caching or Workbench/UI expansion.

## Evidence

All commands use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` from this worktree, `-p no:cacheprovider`, and private basetemp. Runs overlap; counts below are not additive.

1. Red evidence: `/tmp/task5-red.log` pure composition missing-function failures; `/tmp/task5-retry-red.log` real provider adapter missing body before integration.
2. Final focused command:
   `python -m pytest Tests/Chat/test_console_prompt_mixin_composition.py Tests/Chat/test_console_prompt_mixin_retry.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_sqlite_validation.py::test_installed_reference_does_not_materialize_catalog_per_statement -q --tb=line -p no:cacheprovider --basetemp=/tmp/task5-final-focused`
   **75 passed in 41.88s**, log `/tmp/task5-final-focused.log`. Includes actual ConsoleAgentBridge→ConsoleProviderGateway→adapter tool loop in all capture On/Off × streaming/nonstreaming combinations, plain/character streaming/nonstreaming, restart empty/nonempty, exact latest marker discriminator, preview no-save, redaction versus exact operational body, before-dispatch rollback, repeated temporary retry/promotion/source-purge/rollback, trigger preservation and exact current recovery schema equality.
3. Existing integration owners command:
   `python -m pytest Tests/Chat/test_console_prepared_request.py Tests/Chat/test_console_trace_current_turn_transforms.py Tests/Chat/test_console_regenerate_branching.py Tests/Chat/test_console_edit_resend.py Tests/Chat/test_console_turn_resend.py Tests/Chat/test_console_prompt_mixin_admission.py -q --tb=line -p no:cacheprovider --basetemp=/tmp/task5-final-owners`
   **201 passed, 11 failed in 111.17s**, log `/tmp/task5-final-owners.log`. Failures listed below; all 11 reproduced on exact BASE source with identical final harness (11 failed in 4.99s, `/tmp/task5-final-base-attribution.log`). This is not a fully green owner sweep.
4. Earlier owner attribution (not final unique counts): unqualified initial sweep 205 passed/37 failed, mostly missing profile admission; corrected A/B harness 118 passed/15 failed on both BASE and current. Four callback signature failures were fixture drift, fixed in committed fixture. Final 11 are same baseline behaviors: failed/stopped transformed trace closure; failed-regeneration selected-leaf assertions; narrow fake persistence lacking generation projection; stale skill-refusal callback; attachment/vision settings and refused-echo fixture assumptions. No recovery admission bypass or production fallback was added to silence them.
5. Static: new test modules Ruff check and format pass. Changed-line Ruff audit has zero findings (existing touched files still contain 449 pre-existing/off-change findings; no claim whole-file lint is green). Undefined-name checks F821/F823 over changed Python files pass. `git diff --check` passes. Changed product ranges were Ruff-formatted; no unrelated mass formatting.

### Reproducible BASE qualification

`/tmp/task5-baseline` was produced by `git archive 19dc3dbb43e5e623d77efc91741b4c49835295fe tldw_chatbook Tests pyproject.toml | tar -x -C /tmp/task5-baseline`. Copy only final committed `Tests/Chat/test_console_trace_current_turn_transforms.py`, `test_console_regenerate_branching.py`, and `test_console_edit_resend.py` into the matching baseline test paths, keeping BASE product sources. These changes declare sanctioned bootstrap_profile admission for native-owner tests and update already-stale dictionary/tamper callbacks to the actual frozen-context signatures. The ordinary dictionary tests use the real enabled setting; no product gate is forced. Run the node IDs below with the shared Python from that BASE directory, `-q --tb=line -p no:cacheprovider --basetemp=/tmp/task5-final-base-attribution`. The exact node list also lives at `/tmp/task5-baseline-owner-nodeids.txt`. Committed fixtures preserve the harness; no invisible collection hook is needed.

```
FAILED Tests/Chat/test_console_trace_current_turn_transforms.py::test_next_send_after_failed_transformed_call[error]
FAILED Tests/Chat/test_console_trace_current_turn_transforms.py::test_next_send_after_failed_transformed_call[stopped]
FAILED Tests/Chat/test_console_regenerate_branching.py::test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor
FAILED Tests/Chat/test_console_regenerate_branching.py::test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail
FAILED Tests/Chat/test_console_regenerate_branching.py::test_regenerate_persists_new_sibling_when_store_has_persistence
FAILED Tests/Chat/test_console_edit_resend.py::test_edit_and_resend_forks_user_sibling_and_streams_reply
FAILED Tests/Chat/test_console_edit_resend.py::test_edit_and_resend_skill_refusal_leaves_no_pending_node
FAILED Tests/Chat/test_console_edit_resend.py::test_edit_and_resend_carries_anchor_attachments
FAILED Tests/Chat/test_console_turn_resend.py::test_refused_echo_resend_consumes_its_recovery_and_the_composer_copy
FAILED Tests/Chat/test_console_turn_resend.py::test_resend_applies_the_vision_gate_to_the_turns_attachments
FAILED Tests/Chat/test_console_turn_resend.py::test_refused_echo_resend_sends_its_own_restaged_attachments_once
```

## Review and limits

Self-review checked exact anchors and active boundaries, empty/no-op semantics, frozen action routing, write/rollback/publication ownership, retry lineage and marker retention, clean canonical history, trace witness refusal and body privacy. Architecture updates are in Docs/Architecture/console.md and chat-pipeline.md. No new general reusable lesson was added; the discovered integration traps are documented here and pinned by tests.

Not run: full suite, live provider/network/UI session, caching behavior, Workbench UI. Multipart preservation is tested at pure/prepared owner rather than a live multimodal provider. Exact operational capture credential masking is exercised; PII uses the unchanged common policy path rather than a new detector-specific test. The 11 BASE failures remain outside this task and are not claimed fixed. Independent review is still required before marking Done.

Commit: `a125761794b4e4450b00e089711a221d27e08bea` — `feat: compose per-send mix-ins with exact retry history`.

Post-commit worktree status: clean. Git emitted existing repository GC/unreachable-object maintenance warnings; no cleanup performed.


## Evidence-only static qualification after commit

At controller request, compared exact BASE and HEAD per-file `(rule code, message)` multiplicities across all **21 touched Python files**, using the same explicit configuration for both runs:

`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --no-cache --config /Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/pyproject.toml --output-format=json <touched Python paths>`

BASE existing files were restored byte-for-byte into the existing `/tmp/task5-baseline` archive using `git show 19dc3dbb43e5e623d77efc91741b4c49835295fe:<path>`; this removed the earlier test-only A/B harness adjustments for this static comparison. Newly added files have zero BASE diagnostics. HEAD remains `a125761794b4e4450b00e089711a221d27e08bea`. No product mutation or tests were run for this qualification.

**BASE 448 diagnostics; HEAD 449; one introduced; none removed.** The introduced diagnostic is `SIM102` in `tldw_chatbook/DB/console_prompt_mixins.py` at line 200: “Use a single `if` statement instead of nested `if` statements.” Its reported start lies just outside the changed lines, so the earlier changed-line-only audit did not detect it. This supersedes any inference that the earlier audit proved no new whole-file diagnostics. Sent to controller for the review fix round; deliberately left unchanged while read-only review runs.

| File | BASE | HEAD |
| --- | ---: | ---: |
| `Tests/Chat/test_console_edit_resend.py` | 6 | 6 |
| `Tests/Chat/test_console_prompt_mixin_admission.py` | 0 | 0 |
| `Tests/Chat/test_console_prompt_mixin_composition.py` | 0 | 0 |
| `Tests/Chat/test_console_prompt_mixin_retry.py` | 0 | 0 |
| `Tests/Chat/test_console_regenerate_branching.py` | 2 | 2 |
| `Tests/Chat/test_console_trace_current_turn_transforms.py` | 0 | 0 |
| `Tests/DB/test_console_prompt_mixins.py` | 0 | 0 |
| `tldw_chatbook/Chat/chat_persistence_service.py` | 65 | 65 |
| `tldw_chatbook/Chat/console_chat_controller.py` | 180 | 180 |
| `tldw_chatbook/Chat/console_chat_store.py` | 114 | 114 |
| `tldw_chatbook/Chat/console_native_commit.py` | 2 | 2 |
| `tldw_chatbook/Chat/console_prepared_request.py` | 0 | 0 |
| `tldw_chatbook/Chat/console_prompt_mixin_service.py` | 0 | 0 |
| `tldw_chatbook/Chat/console_prompt_mixins.py` | 0 | 0 |
| `tldw_chatbook/Chat/console_trace_final_values.py` | 35 | 35 |
| `tldw_chatbook/Chat/console_trace_provenance.py` | 7 | 7 |
| `tldw_chatbook/Chat/console_trace_runtime.py` | 2 | 2 |
| `tldw_chatbook/Chat/console_trace_service.py` | 33 | 33 |
| `tldw_chatbook/Chat/console_turn_resend.py` | 1 | 1 |
| `tldw_chatbook/DB/console_prompt_mixins.py` | 0 | 1 |
| `tldw_chatbook/DB/recovery_core_schema.py` | 1 | 1 |

Exact JSON evidence retained at `/tmp/task5-ruff-exact-base-head.json`, `/tmp/task5-ruff-exact-base.json`, and `/tmp/task5-ruff-exact-head.json`. The summary JSON contains all compared paths and multiplicity differences.


## Fix round 1 plan (before schema edits)

Reviewed full task-5-review.md. I1: shared synthetic continuation anchor across all replay actions, with capture-Off restart controls. I2: controller-approved three-column operational extension (`replaces_attempt_id`, `result_revision_id`, `trace_call_id`) and guarded set-once result/trace seals; exact checkpoint binding of original trace, bounded chain to origin call, edit rejection, fork authority clearing and no invented temporary proof. Existing ADR286 amended and task plan updated before schema edits. Also fix SIM102 and test cancellation drain/no dispatch. No broad owner rerun authorized or planned.


## Fix round 1 completion

I1 fixed by making `_ensure_user_continuation_instruction` return the exact synthetic anchor derived from the already-owned attempt identity; Retry, Regenerate, Continue and edit routes pass it through the common composition owner. Canonical historical user text remains unchanged. Restart tests select the exact restored failed continuation response from all nodes, as the default reopened active branch may be the prior successful response.

I2 fixed with controller-approved existing-table extension. `MixinAttemptWrite.replaces_attempt_id` is an explicit accepted replacement relationship (Retry, Regenerate, accepted-original stopped-empty Resend); ordinary new Continue remains unrelated. `trace_call_id` binds at FRESH/AGENT_FIRST reservation through the exact checkpoint in that same transaction; tool-loop calls do not overwrite it. `result_revision_id` seals a complete replacement in the existing generation settlement callback after semantic revision creation. Immutable guards enforce insert-unsealed, set-once proof and exact ownership; FK deletion permits clearing only when the referenced proof has actually disappeared, retaining operational snapshots. The validator requires a bounded cycle-free replacement chain to the exact origin call and exact sealed result revision, in addition to the prior current-owner, marker, source, parent, no-dispatch, origin/terminal/policy/range/descriptor checks. Ordinary response-link equality is not bypassed by an initial marker, and edited results cannot inherit seals.

The actual unreleased-v82 schema-object amendments are three nullable columns/FKs on `console_prompt_mixin_attempts` plus its insert/update guards; the exact current recovery catalog was updated. No new table, body copy, or route identity contract. Fork drops all three witness-authority fields while retaining original retry records; temporary promotion retains snapshot/retry/replacement history but creates no historical trace/result seals (temporary Capture On has no durable origin). A new durable send establishes proof normally. No released-v82 upgrade compatibility is claimed.

Cancellation regression now pauses the real persistence worker, cancels controller Retry, proves no early publication/task completion, releases the worker, then proves committed record publication precedes CancelledError and no second adapter call occurred.

### Exact targeted evidence (overlapping counts, not additive)

All commands use shared Python `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` from the managed worktree and `-p no:cacheprovider`:

- `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k 'continue_replay or cancel_during_attempt' -q --tb=short --basetemp=/tmp/task5-fix1-anchor`: initial 3 passed/2 failed because the restart fixture looked only on the default selected branch. The exact saved failed response was present in all nodes; the fixture now selects it explicitly.
- `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k continue_replay -q --tb=short --basetemp=/tmp/task5-fix1-anchor2`: 4 passed, 36 deselected in 4.85s.
- `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py::test_retry_original_regenerate_current_and_clean_history -q --tb=short --basetemp=/tmp/task5-fix1-proof1`: 2 passed in 3.51s (genuine replacement restoration On/Off).
- `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k 'replacement_trace_proof or marker_cannot or result_seal' Tests/Backup_Recovery/test_sqlite_validation.py::test_installed_reference_does_not_materialize_catalog_per_statement -q --tb=short --basetemp=/tmp/task5-fix1-proof2`: 16 passed, 31 deselected in 16.20s. The -k expression deselected the catalog test; catalog proof comes from the explicit runs below. Includes original missing/unequal links, edited first and edited genuine replacement, immutable seals, forced terminal rollback, fork authority clearing and real diagnostic GC retaining operational retry bodies.
- `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k 'promoted_history or real_agent_tool_loop or replacement_trace_proof' -q --tb=short --basetemp=/tmp/task5-fix1-proof3`: 18 passed, 32 deselected in 20.34s. Includes wrong origin, missing seal and cyclic relation refusal; actual agent first-call association across tool loops with capture/streaming matrix; edited temporary history promotion → first durable captured send → captureOff exact Retry after close/reopen and exact result seal.
- `-m pytest Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_sqlite_validation.py::test_installed_reference_does_not_materialize_catalog_per_statement -q --tb=short --basetemp=/tmp/task5-fix1-db`: 34 passed in 9.03s, including exact installed/recovery catalog equality.
- **Final:** `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py Tests/Chat/test_console_prompt_mixin_values.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_sqlite_validation.py::test_installed_reference_does_not_materialize_catalog_per_statement -q --tb=short --basetemp=/tmp/task5-fix1-final`: **95 passed in 58.33s**. Log `/tmp/task5-fix1-final.log`. This includes all above controller/operational controls and cancellation; no broad owner suite was rerun.

### Static and review qualification

Exact BASE 19dc3dbb43e5e623d77efc91741b4c49835295fe versus current per-file `(code,message)` multiplicities over all Task5-touched Python files, with the same explicit worktree pyproject config and `ruff check --no-cache --output-format=json`: **448 versus 448, no introduced or removed diagnostic multiplicities**. The reported SIM102 is fixed. Evidence `/tmp/task5-fix1-ruff-differential.json`; original exact BASE files remain in `/tmp/task5-baseline`. Existing unrelated warnings remain, so this is differential qualification rather than whole-repository lint-green. New/changed test module Ruff check and format are clean; changed product ranges were formatted; `git diff --check` clean.

Self-review rechecked action identity, immutable lineage, seal/write rollback, no proof after edits, captureOff behavior, temporary/fork boundaries and diagnostic GC. Live provider/UI/full-suite/actual backup deployment remain unrun; restore qualification is exact schema/recovery validation and real file close/reopen. Prior 201/11 baseline owner evidence is unchanged and not counted as this round's verification. Task remains In Progress for controller re-review.

Fix round 1 commit: `8c551fe41fdff3c637a5cedcbcfc5e81314bf22c` (`fix: bind mix-in replay and trace replacement ownership`).


## Fix round 2 — current-choice recovery without old proof

Review BASE `8c551fe41fdff3c637a5cedcbcfc5e81314bf22c`; R1 confirmed: unconditional Regenerate replacement lineage made current choices depend on the unavailable original snapshot. The controller now qualifies this optional link with the existing exact operational replay reader before creating the sibling. An unavailable original leaves replacement lineage unset; current capture and the mandatory new attempt transaction still govern dispatch. Exact Retry, DB writer predecessor guards, trace validators and schema are unchanged. No original snapshot is repaired or fabricated. Existing ADR286 applies; task plan was extended before code and Console architecture documents the recovery boundary.

### Focused evidence (runs overlap)

All commands below use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` from the managed worktree, `-p no:cacheprovider`, and private basetemps. Native-owner module profile admission is committed.

- Regression command: `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k 'regenerate_current_choices_recovers_unavailable_original or regenerate_unavailable_original_still_requires_database_write' -q --tb=short -p no:cacheprovider --basetemp=/tmp/task5-fix2-red` → **7 failed, 50 deselected in 8.41s**, log `/tmp/task5-fix2-red.log`. Six missing-attempt/missing-items/corrupt-ordinal × capture On/Off cases reproduced no Regenerate adapter entry. The seventh pinned that optional original read is attempted and mandatory write still refuses on database errors.
- Same command after product fix with `--basetemp=/tmp/task5-fix2-green` → **3 failed, 4 passed, 50 deselected in 7.41s**, log `/tmp/task5-fix2-green.log`. All six recovery cases reached the adapter with current B. Three capture-On cases exposed an incorrect test expectation that Regenerate receives a fresh trace origin binding. Corrected the test to assert actual capture mode via the original FRESH binding and that recovered Regenerate has no trace-origin binding, matching the existing first-call-only contract; no product adjustment was made for this test correction.
- Final covering command: `-m pytest Tests/Chat/test_console_prompt_mixin_retry.py -k 'regenerate_current_choices_recovers_unavailable_original or regenerate_unavailable_original_still_requires_database_write or retry_original_regenerate_current or original_retry_survives or replacement_trace_proof or marker_cannot or regenerate_attempt_failure or continue_replay' -q --tb=short -p no:cacheprovider --basetemp=/tmp/task5-fix2-cover` → **35 passed, 22 deselected in 34.89s**, log `/tmp/task5-fix2-cover.log`. Covers the seven new cases, exact Retry/current Regenerate/clean history journey, deleted-source/restart Retry refusal, strict replacement proof negatives including edited results, atomic placeholder write rollback, and synthetic Continue anchors. The new recovery cases assert original library source deleted, valid distinct current B sent, original still unavailable, and no retry/replacement/result proof authority on the new attempt. Injected read+write OperationalError prevents new operational commit and adapter entry.
- `-m ruff check --no-cache Tests/Chat/test_console_prompt_mixin_retry.py` → all checks passed; `-m ruff format --no-cache Tests/Chat/test_console_prompt_mixin_retry.py` formatted the new tests. Final format check and `git diff --check` clean.
- Exact fix-BASE versus current per-file `(code,message)` Ruff multiplicities across both touched Python files (`console_chat_controller.py`, `test_console_prompt_mixin_retry.py`), same explicit worktree `pyproject.toml`, `ruff check --no-cache --output-format=json`: **180/180, no introduced or removed diagnostic multiplicities**. Evidence `/tmp/task5-fix2-ruff-differential.json`; exact original files `/tmp/task5-fix2-lint-base`. No schema/DB source changed in this round.

Self-review checked optional proof is decided before sibling creation and accepted write, valid original lineage still uses unchanged strict DB/trace validation, and failed new persistence still prevents provider entry. No broad 95-test rerun, full suite, live provider/UI or deployment validation. Prior round evidence remains separately qualified above. Task remains In Progress for controller re-review.

Fix round 2 commit: `98c32189b6014791ab8e9f613b8114af82f7ea69` (`fix: recover current mix-ins without unavailable retry proof`).
