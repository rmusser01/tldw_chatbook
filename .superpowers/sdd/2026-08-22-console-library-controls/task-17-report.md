# Task 17 report — qualify and close TASK-19900.3

Date: 2026-08-23

## Outcome and scope

Task 17 made no production or test-behavior change. It qualified the completed
Tasks 12–16 delivery at the exact approved head
`c13acd90f310e4d4f91a5745de7371bda823b6ac`, reconciled all 22 delivery
acceptance criteria against named production-path tests, and closed only the
backlog/report/ledger records after the evidence was green.

ADR required: no.

ADR paths:

- `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`
- `backlog/decisions/079-console-library-conversation-authority.md`

Reason: Task 17 only verifies the already-approved ADR-079 Console Library
authority/recovery design and ADR-063 exclusive post-handoff continuation
ownership. It adds no behavior, schema, persistence authority, or architectural
boundary.

No app, user profile database, network, full repository suite, Task 18+, or push
was used.

## Provenance and approved head

```text
git status --short --branch
## docs/console-rag-ux-design...origin/docs/console-rag-ux-design [ahead 304, behind 1]

git rev-parse HEAD
c13acd90f310e4d4f91a5745de7371bda823b6ac

git log -1 --format='%H %s'
c13acd90f310e4d4f91a5745de7371bda823b6ac fix(console): harden continuation recovery

../../.venv/bin/python -m pytest -q --tb=short Tests/test_probe_import_provenance.py
1 passed, 1 warning in 0.39s
```

The branch tracking delta was not reconciled because the parent supplied the
exact approved local head and explicitly prohibited network/push work. The
worktree itself was clean before the Task 17 plan was appended.

## Exact targeted delivery battery

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no \
  Tests/Chat/test_console_turn_preparation.py \
  Tests/Chat/test_library_preparation.py \
  Tests/Chat/test_console_automatic_library_preparation.py \
  Tests/Chat/test_console_durable_turn_acceptance.py \
  Tests/Chat/test_console_first_send_atomicity.py \
  Tests/Chat/test_console_dispatch_recovery.py \
  Tests/Chat/test_console_dispatch_queue_recovery.py \
  Tests/UI/test_console_dispatch_recovery.py \
  Tests/Chat/test_console_dispatch_continuation_handoff.py \
  Tests/Chat/test_console_assistant_generation_history.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_prompt_queue_coordinator.py \
  Tests/Chatbooks/test_provider_continuation_roundtrip.py \
  Tests/Sync_Interop/test_chat_outbox_producer.py \
  Tests/Sync_Interop/test_envelope_builder.py \
  Tests/Sync_Interop/test_envelope_applier.py \
  Tests/Sync_Interop/test_provider_continuation_reconciliation.py
716 passed, 1 warning in 43.53s
```

The warning is the inherited Requests dependency-version warning.

## Three-times fault qualification

The same three exact nodes were run in three independent pytest processes:

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no \
  'Tests/Chat/test_console_durable_turn_fix_round2.py::test_real_persistence_retry_reuses_exact_staged_message_owners[commit-automatic-queued]' \
  'Tests/Chat/test_console_dispatch_recovery_fix_round2.py::test_explicit_retry_resumes_every_unfinished_postcommit_effect_before_provider[checkpoint_transition]' \
  Tests/Chat/test_console_dispatch_recovery_fix_round1.py::test_queued_settlement_failure_hydrates_exact_fence_before_return

repetition 1: 3 passed, 1 warning in 0.88s
repetition 2: 3 passed, 1 warning in 0.87s
repetition 3: 3 passed, 1 warning in 0.86s
```

The nodes prove distinct invariants:

1. The actual SQLite COMMIT fault leaves zero accepted rows, then Retry uses the
   same captured conversation/USER/assistant IDs and finishes with exactly those
   two message rows—no duplicate conversation or turn owner.
2. The injected checkpoint-transition failure returns accepted recovery with
   `provider_started=False`, zero gateway calls, and the exact owner; explicit
   Retry completes the unfinished effects and only then invokes the provider
   from `dispatch_started`.
3. The injected terminal-settlement checkpoint DELETE failure returns with the
   exact queued recovery fence hydrated; only the accepted entry was submitted,
   the later entry remains in the paused queue, and the run is BLOCKED.

## Fresh review/fix and migration companions

Task 17's written battery predates the later Task14–16 review-fix files, so
their narrow official gates were refreshed rather than relying on old reports:

```text
# Task 14 durability and four fix rounds
73 passed, 1 warning in 38.94s

# Task 15 recovery and four fix rounds, including mounted UI
101 passed, 1 warning in 24.60s

# Task 16 independent-review ratchets
14 passed, 1 warning in 3.45s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no \
  Tests/DB/test_chachanotes_console_library_policy_migration.py \
  Tests/DB/test_chachanotes_console_library_migration_seed_openers.py
30 passed, 4 warnings in 7.11s
```

The migration gate constructs real historical/current temporary databases and
AST-inspects production openers. It does not open the profile database. Its
warnings are the inherited Requests warning plus three existing invalid-escape
SyntaxWarnings in unrelated source files inspected by the opener audit.

## Acceptance-criterion reconciliation

All named tests below were part of the fresh 716 delivery battery unless a
fresh companion count is explicitly identified.

| AC | Evidence | Result |
| --- | --- | --- |
| 1 | `test_evidence_success_uses_exact_draft_fixed_categories_and_exact_bundle`; queued/manual origin coverage in `test_console_prompt_queue_coordinator.py` | Met: executed draft and fixed Notes/Media/Conversations authority are pinned independently of manual filters. |
| 2 | `test_active_scope_uses_exact_note_media_allowlist_and_excludes_conversations` | Met: current scope narrows Note/Media IDs and removes Conversations. |
| 3 | `test_explicit_staged_evidence_skips_duplicate_automatic_retrieval`; `test_never_and_nonordinary_text_skip_automatic_retrieval` | Met: explicit evidence and ineligible kinds do not spend or stage automatic retrieval twice. |
| 4 | `test_manual_cancel_preserves_staged_state_and_removes_only_transient_echo`; `test_controller_cancel_removes_exact_owner_and_sidecars_without_touching_staged_inputs`; pre-provider marker tests | Met: preparation is cancellable before provider entry and restores exact staged state. |
| 5 | `test_pause_actions_are_the_exact_frozen_data_matrix`; `test_failure_and_timeout_pause_with_bounded_error`; `test_manual_recovery_continues_same_frozen_send_without_second_submit` | Met: Retry/Bypass/Cancel remain distinct, tested, and standing policy is unchanged. |
| 6 | `test_success_injects_the_sealed_bundle_into_the_same_dispatched_request`; `test_zero_matches_readies_with_one_bounded_contribution`; Task12 payload-deny tests | Met: success uses the exact request; zero/bypass disclosure is bounded and contains no query/source identity. |
| 7 | `test_generic_trajectory_sidecar_cannot_displace_or_duplicate_user_anchor`; default/full export equality; inert import test | Met: the preparation sidecar owns disclosure without taking over generic trajectory ownership. |
| 8 | `test_store_preparation_cas_is_exact_and_survives_controller_replacement`; racing-action, destination-change, close, and shutdown tests | Met: one store-owned CAS authority survives navigation and refuses changed destination/repeated actions. |
| 9 | parametrized `test_every_new_conversation_write_or_commit_failure_rolls_back_exactly_and_retries_once`; `test_precommit_failure_keeps_input_and_never_calls_provider`; repeated COMMIT node | Met: identity/title and all durable owners publish only after one atomic commit and Retry does not duplicate them. |
| 10 | persistence pause action matrix; Automatic/Never cases in Task14's fresh 73-test gate; Never skip test | Met: persistence exposes only Retry/Cancel, while bypass remains retrieval-failure-only and Never has no Library-preparation status. |
| 11 | accepted/dispatch-started loader tests; exact-owner Retry; no-auto-replay; mounted recovery tests in fresh 101 gate | Met: one empty assistant/checkpoint owner drives explicit recovery without a second USER/assistant. |
| 12 | strict checkpoint codec/privacy assertions in fresh Task14 gate; unreconstructable Retry reason test; source scan below | Met: checkpoints retain no content/credential request payload, are not projected, and delete only through durable settlement/handoff. |
| 13 | terminal-stale-checkpoint loader case; invalid-pair quarantine; atomic Discard and success/failure/cancel settlement fault cases in fresh 101 gate | Met: terminal/Discard update plus checkpoint deletion is atomic and reconciliation is deterministic/fail-closed. |
| 14 | terminal settlement tests; Cartesian shared history predicate; empty closed-state literal renderer; fresh Task16 mounted five-state matrix | Met: complete/stopped/failed/discarded state persists and renders/history-filters under exact versions/deletion guards. |
| 15 | `test_cartesian_history_predicate_and_console_provider_builder_agree`; `test_active_continuation_is_sidecar_only_and_never_an_ordinary_blank_item` | Met: accepted/dispatch-started are excluded and active continuation enters only through validated ADR-063 projection. |
| 16 | first-batch handoff ordering; local statement/COMMIT failure; post-commit Sync-v2 failure; exact version/deletion guards | Met: local handoff commits and deletes dispatch recovery before tools; ADR-063 is then the sole owner. |
| 17 | legacy normalization/rebind, rollback, conflict, Discard, and dual-owner precedence tests | Met: valid legacy continuation normalizes lazily, actions enable only after rebound proof, and stale ownership fails closed. |
| 18 | Sync-v1/v2 create/update/delete/undelete, source/envelope/applier gates; Chatbook graph/private round-trip; inert remote loader/UI copy; `test_regenerate_message_streams_into_new_sibling_node` | Met: portable state projects compatibly, local checkpoint never does, remote unresolved state is inert, and ordinary Retry/regenerate forks a sibling. |
| 19 | queued failure/reclaim tests; `test_precommit_cancel_releases_only_exact_claim_to_pending`; exact queue authority tests | Met: queued preparation freezes one entry/authority, pauses later work, acknowledges once, and only precommit Cancel returns it pending. |
| 20 | postcommit queue hydration; no-auto-resume; exactly-once settlement/drain; repeated queued settlement-fault node | Met: accepted queue work never returns pending and later work remains fenced until Retry/Discard settles the owner. |
| 21 | ephemeral store-only/no-row, promotion-block, in-memory settlement, lifecycle tests; fresh mounted/store fixes | Met: promotion contributions use the transaction seam, unresolved state blocks before writes, and ephemeral recovery is runtime-only. |
| 22 | complete 716 delivery battery, fresh 73/101/14 companions, and all three repeated fault nodes | Met: precommit, postcommit, dispatch, settlement, handoff, sync/export/import, restart, promotion, and manual/queued recovery boundaries are covered without silent provider fallthrough. |

No criterion depends solely on a report claim: each maps to executable tests or
the explicit source/privacy scans below. All 22 therefore qualify for checking.

## Backlog closeout verification

```text
backlog task edit 19900.3 -s Done
Updated task TASK-19900.3

backlog task 19900.3 --plain
Task TASK-19900.3 - Make automatic Console Library retrieval a truthful send gate
Status: Done
Acceptance Criteria: #1 through #22 all [x]

rg -c '^- \[x\]' <task-file>
22

rg -n '^status:' <task-file>
4:status: Done

find backlog/tasks -maxdepth 1 -name 'task-task- - *.md' -print
<empty>
```

The CLI added its standard section markers, criterion numbers, updated date, and
frontmatter ordering while preserving the task text. The known high-ID ghost
file failure did not occur for the dotted child ID.

## Static, privacy, and source qualification

The final post-documentation outputs are recorded during closeout:

```text
../../.venv/bin/python -m ruff check <Task12–16 production modules and targeted/fix tests>
All checks passed!

git diff --check
<empty>

rg -n '_CURRENT_SCHEMA_VERSION = ' tldw_chatbook/DB/ChaChaNotes_DB.py
503:    _CURRENT_SCHEMA_VERSION = 45  # Device-local Console Library policy and dispatch recovery.

rg -n 'console_dispatch_checkpoints' tldw_chatbook/Sync_Interop tldw_chatbook/Chatbooks tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Character_Chat
<empty>

rg -n 'logger\\.(debug|info|warning|error|exception|critical).*?(content|prompt|evidence|provider_continuation_json|api[_-]?key)' <delivery production files>
<empty>
```

No prompt, Library query/source identity, evidence body, continuation body,
tool payload, credential, provider secret, or checkpoint payload was added to a
log or Task17 metadata. Task17 itself changes documentation only.

## Files changed

- `backlog/tasks/task-19900.3 - Make-automatic-Console-Library-retrieval-a-truthful-send-gate.md`
- `.superpowers/sdd/2026-08-22-console-library-controls/task-17-report.md`
- `.superpowers/sdd/2026-08-22-console-library-controls/progress.md`

No production or test file changed.

## Qualifications and self-review

- The inherited Requests dependency warning remains. The isolated opener audit
  also surfaces three unrelated invalid-escape SyntaxWarnings; neither affects
  the qualified Console delivery behavior.
- Task16's earlier non-gating broad Markdown-widget run documented five stale
  header-location assertions. Task17's actual production-hierarchy review
  ratchets pass 14/14, and Task17 did not rewrite unrelated legacy assertions.
- The branch is one commit behind its remote tracking ref, but the parent
  explicitly approved the exact local head and prohibited network/push work.
- No new incident arose that supports a general lesson; existing migration,
  import-provenance, lifecycle, mounted-recovery, and Backlog CLI lessons were
  followed instead of adding folklore.
- Self-review found no production/test change, no Task18+ work, no migration or
  profile access, and no unchecked acceptance criterion after the evidence
  reconciliation.
