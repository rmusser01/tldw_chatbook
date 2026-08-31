# TASK-24727 Chatbook first-link implementation report

## Status

Independent-review remediation is implemented, targeted-test clean, and through
scoped static/security verification. The remediation commit is pending below;
controller cross-repository review and final backlog closure remain. This report
does not mark TASK-24727 Done.

## Commit

Implementation commit: `2043974607a9d30a32b3a6bfa754b01a64f39c35`
(`feat(personal-context): add reviewed first-link sync`).

Independent-review remediation commit: `a86e19828c`
(`fix(personal-context): verify first-link convergence`).

Structured bootstrap attention integration commit: `f42c173c55`
(`fix(personal-context): surface bootstrap attention`), consuming tldw_server
contract commit `a92e12110d`.

Final contract-quality remediation commit: this report's implementation commit
(`fix(personal-context): harden first-link recovery`).

## RED evidence

All feature work began with focused failing tests. Principal RED observations:

- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/tldw_api/test_personal_context_sync_client.py Tests/Personal_Context/test_profile_reconciliation.py Tests/Sync_Interop/test_personal_context_first_link.py Tests/UI/test_personal_context_link_modal.py`
  - 3 collection errors: missing typed API exports, reconciliation module, and
    Settings modal.
- Focused transport/custody run: 3 failures for missing bootstrap/completion
  methods and generic-enrollment rejection.
- Focused reviewed-repository run: 2 failures for missing canonical apply and
  integrity rebaseline.
- Focused coordinator run: 1 collection error for the missing link service.
- Focused canonical Settings runs: 1 panel-launcher failure and 1 app-launcher
  failure.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/Personal_Context/test_profile_reconciliation.py::test_reviewed_apply_adopts_server_identity_and_rebaselines_every_artifact Tests/Sync_Interop/test_personal_context_dispatcher.py::test_dispatcher_is_fail_closed_until_first_link_completion`
  - 2 failures: proposal/noncanonical heads were not retained and ordinary
    dispatch was not gated.
- Focused staged-recovery run: 1 `TypeError` for the absent recovery-key
  constructor seam.
- Focused activation-pending run: 1 collection error for the absent explicit
  activation-pending failure type.
- Focused recovery-resume run: 1 `AttributeError` for the absent post-activation
  resume method.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_personal_context_first_link.py::test_retry_can_replace_an_unapproved_attention_snapshot Tests/Sync_Interop/test_personal_context_first_link.py::test_new_review_cannot_overwrite_an_apply_in_progress`
  - 2 failures: attention retry was rejected while an in-progress apply could
    be overwritten.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_personal_context_first_link.py::test_precommit_interruption_discards_only_the_uncommitted_staged_key`
  - 1 `AttributeError` for the absent pre-commit interruption cleanup seam.
- Independent-review edge consolidation:
  `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_personal_context_first_link.py Tests/Sync_Interop/test_personal_context_first_link_sync.py Tests/Personal_Context/test_profile_reconciliation.py Tests/UI/test_personal_context_link_modal.py`
  - 12 intended failures across durable convergence, terminal recovery,
    restart/freeze lifecycle, exact workspace outcomes, device-only privacy,
    stale destination cleanup, and bounded journal transfer.
- Negotiated batch/public push-gate focus:
  `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py::test_personal_context_bootstrap_registers_wrapping_key_without_generic_enroll Tests/Sync_Interop/test_server_sync_service.py::test_public_push_rejects_personal_context_before_transport_dispatch Tests/Sync_Interop/test_server_sync_service.py::test_private_first_link_push_uses_reviewed_transport_path Tests/Sync_Interop/test_personal_context_first_link.py::test_plan_seeds_normal_sync_profile_without_generic_enrollment Tests/Sync_Interop/test_personal_context_first_link_sync.py::test_special_cycle_drains_101_entries_in_negotiated_push_batches`
  - 4 intended failures: bootstrap discarded `max_batch_size`, public PC push
  dispatched, the private push path was missing, and planning did not persist
  the negotiated limit. The existing 101-entry batch behavior remained green.
- Final independent-review RED runs:
  - 4 cursor-contract failures showed bootstrap receipt cursors seeding both
    push/pull and overwriting ordinary Sync cursor state.
  - 3 key-boundary failures left unwrap/stage/apply errors in `applying` or let
    cleanup errors mask the original failure.
  - 2 reconciliation failures showed an adopted bound workspace returning as a
    decision and a retained Undo artifact keeping provisional identity fields.
  - 2 readiness failures showed schema/quota capability mismatches preventing
    the authenticated typed bootstrap-attention endpoint from being called.
  - 1 restart-recovery failure showed the absence of a locked-profile freeze
    release fallback; the production dispatcher crash regression then pinned
    exact-plan replay between the profile and Sync databases.

## GREEN evidence

Final focused verification used `PYTEST_DEBUG_TEMPROOT=/private/tmp/task3b-pytest`
to isolate pytest cleanup from unrelated pre-existing temporary directories.

- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q --disable-warnings Tests/Personal_Context/test_repository.py Tests/Personal_Context/test_service.py Tests/Personal_Context/test_runtime_policy.py Tests/Personal_Context/test_profile_sync_outbox.py Tests/Personal_Context/test_profile_reconciliation.py Tests/Personal_Context/test_profile_link_key_custody.py`
  - 108 passed, 1 warning.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q --disable-warnings Tests/Sync_Interop/test_personal_context_first_link.py Tests/Sync_Interop/test_personal_context_dispatcher.py Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_personal_context_adapter.py Tests/Sync_Interop/test_personal_context_capabilities.py Tests/Sync_Interop/test_local_first_sync_service.py -k 'personal_context or profile or link'`
  - 63 passed, 57 deselected, 1 warning.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q --disable-warnings Tests/tldw_api/test_personal_context_sync_client.py Tests/tldw_api/test_sync_client.py`
  - 16 passed, 1 warning.
- `PYTHONPATH=.:packages/tldw_profile_core/src ../../.venv/bin/python -m pytest -q --disable-warnings Tests/UI/test_personal_context_link_modal.py Tests/UI/test_settings_personal_context.py::test_my_profile_is_registered_in_data_privacy_and_settings_contracts Tests/UI/test_settings_personal_context.py::test_link_action_is_exposed_only_on_canonical_profile_panel`
  - 4 passed, 2 warnings.
- Focused first-link state/recovery file after the final fixes:
  - 9 passed, 1 warning.
- Ruff over every touched Python source and test:
  - `All checks passed!`
- Python `compileall -q` over touched packages/files:
  - exit 0.
- `../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py`
  - authoritative component bundle and all four generated widget/screen CSS
    artifacts reproduce exactly.
- `git diff --check`
  - exit 0.
- Latest focused gate/capability regression:
  - 6 passed, 1 dependency warning.
- Latest complete touched-scope verification (Personal Context, Sync transport,
  typed API, Settings modal, and production composition):
  - 285 passed, 2 dependency warnings in 12.75 seconds.
- Latest Ruff run over all changed Python source and tests:
  - `All checks passed!`
- Python `compileall -q` over the touched production packages/files:
  - exit 0.
- CSS source reproduction:
  - authoritative component bundle and all four generated widget/screen CSS
    artifacts reproduce exactly.
- `git diff --check` and `git diff --check 0e6b38204d`:
  - both exit 0.
- Bandit over the touched production files:
  - high-severity gate (`-lll`) exits 0 with no finding;
  - full report: 0 high, 13 medium, 38 low. The medium findings are Bandit's
  existing string-SQL heuristic over repository-allowlisted table/column names
  and parameter-placeholder construction; low findings are existing asserts,
  exception cleanup, and app subprocess/random patterns. No new unreviewed
  high-severity issue is present.
- Final remediation verification:
  - Personal Context repository/service/reconciliation/custody group: 122 passed.
  - Sync first-link/dispatcher/service/adapter/capability group: 105 passed,
    57 deselected.
  - Typed API and canonical Settings/modal/app-flow group: 95 passed.
  - Focused all-changed contract group: 116 passed.
  - Ruff: `All checks passed!`; compileall exited 0; CSS source reproduction
    matched all generated artifacts; Bandit `-lll` exited 0; both diff-hygiene
    checks exited 0.

Warnings were the repository environment's existing `requests` dependency
version warning and pytest cleanup warnings for unrelated stale temporary
directories. No product-test failure remained.

## Changed files

- Planning: `IMPLEMENTATION_PLAN_personal_context_first_link.md`
- Personal Context production: `bootstrap.py`, `key_protector.py`,
  `link_key_custody.py`, `link_service.py`, `reconciliation.py`,
  `repository.py`, `service.py`
- Sync production: `local_first_sync_service.py`,
  `personal_context_dispatcher.py`, `server_sync_service.py`,
  `sync_state_repository.py`
- Typed API: `tldw_api/__init__.py`, `tldw_api/client.py`,
  `tldw_api/sync_schemas.py`
- Canonical Settings/UI: `settings_screen.py`, `personal_context_panel.py`,
  `personal_context_link_modal.py`, `app.py`
- Styling: `css/components/_profile_interview.tcss`, generated
  `css/tldw_cli_modular.tcss`
- Tests: `test_profile_link_key_custody.py`,
  `test_profile_reconciliation.py`, `test_personal_context_first_link.py`,
  `test_personal_context_dispatcher.py`, `test_server_sync_service.py`,
  `test_personal_context_link_modal.py`,
  `test_settings_personal_context.py`,
  `test_personal_context_sync_client.py`

## Implementation summary

- Added strict typed bootstrap/completion schemas and exact authenticated client
  routes; generic dataset enrollment now rejects every Personal Context domain.
- Added secure RSA-OAEP-SHA256 device wrapping and exact-binding staged
  integrity-key custody with verified keyring production providers and explicit
  in-memory test providers. There is no plaintext durable fallback.
- Added content-free reconciliation plans covering canonical profile adoption,
  exact IDs/versions, collisions, lineage/version decisions, quotas, schema,
  purge generation, global mapping, peer-local workspace decisions, unlinked
  remote workspaces, and device-only exclusions.
- Added one reviewed canonical apply path. It keeps the local encryption key,
  adopts the server profile identity, activates the authenticated server
  integrity key, re-tags every retained encrypted object/artifact under a new
  key version, and preserves accepted canonical IDs and versions.
- Added exact stale-plan detection so local edits racing with planning are never
  overwritten; the user returns to review. SQLite convergence runs in one write
  transaction.
- Added a durable, exact-binding link state machine. Ordinary Personal Context
  dispatcher and local-first push/pull paths remain fail closed until local
  rebaseline, exact server completion, and persisted `complete` state.
- Added a dedicated first-link convergence path that drains every reviewed journal
  page and pushes bounded, lineage-ordered batches using the negotiated server
  `max_batch_size`; it confirms exact heads using an include-own pull before
  persisting the ordinary-Sync cursor.
- Public Personal Context push and pull now reject before transport dispatch.
  Private first-link and exact-complete wrappers are reached only after their
  callers validate the matching durable binding.
- Added explicit recovery for both sides of the database/keyring boundary:
  staged-key authenticated activation after a committed rebaseline, safe
  cleanup/review after a pre-commit interruption, and idempotent exact-cursor
  completion after local convergence.
- Added the protected canonical F9 Settings modal with content-free counts,
  explicit collision/version/workspace choices, unlinked-workspace disclosure,
  cancel, retry, attention, disabled approval, and bounded return data.
- Separated immutable bootstrap review receipts from ordinary transport cursors;
  exact-dataset existing cursors are retained, while absent/mismatched cursors
  trigger full first-link pull without overwriting unrelated Sync state.
- Closed the reconciliation write boundary around source dispatch as well as
  destination apply, with exact-plan production crash replay and no ungated
  fallback path.
- Made every provisional key/apply failure and locked restart recovery converge
  on content-free attention plus exact freeze release, while preserving the
  original operational error when secure cleanup also fails.
- Preserved bound canonical workspace identities across replanning, rebound
  valid retained Undo payload identities, and allowed only schema/quota
  readiness facts through to authenticated typed bootstrap attention.

## Self-review

- Confirmed Chatbook retains and mutates the same canonical profile objects
  returned by the server; no mirror or translated authoritative profile was
  introduced.
- Confirmed plan/cancel paths make no completion or canonical Personal Context
  content upload call and do not modify canonical profile/key custody. Bootstrap
  does reserve the approved content-free server control-plane device/dataset/
  authority/key scaffolding; cancellation releases local freeze/staging and leaves
  both canonical content replicas unchanged.
- Confirmed remote workspace scopes remain without local bindings and therefore
  unavailable to agents until explicitly mapped.
- Confirmed staged keys, wrapped blobs, plaintext keys, record/proposal bodies,
  and manifests are not logged or included in content-safe state/error strings.
- Confirmed state replacement permits a fresh unapproved retry but cannot
  replace `applying` or `local_rebaseline_complete` work.
- Confirmed keyring/database activation is described and implemented as a
  resumable two-store transition, not falsely claimed atomic.
- Confirmed the new action is wired only into canonical F9 Settings and uses
  the incumbent Textual layout/copy conventions without decorative redesign.

## Known limitations and skips

- No full repository suite was run, per the explicit task and repository
  instruction to use targeted verification only.
- No live server/keyring/TUI session was run in this child slice. Server contract
  behavior is pinned with typed client and production-shaped service tests
  against the contract from server commit `a92e12110d`; secure custody is tested
  with injected providers.
- Production linking requires a verified secure OS keyring and advertised
  compatible server capabilities. Missing custody or capabilities fail closed.
- Independent cross-repository review remains controller-owned. The backlog task
  status and final acceptance-criteria closure were not changed here.
- The full Bandit report exits nonzero for 13 medium and 38 low findings already
  present in the scanned full files; the explicit high-severity gate passes. The
  report is retained rather than suppressing unrelated repository findings.

## Independent-review remediation (2026-08-30)

Status: implemented locally; controller re-review pending.

Additional RED evidence:

- Convergence/exact state: 3 intended failures (missing first-link-sync injection
  and confirmed-cursor persistence), then 3 passed.
- Existing Sync merge/binding: 2 intended failures (cursor/capability clobber and
  silent dataset/device replacement), then 4 focused cases passed.
- Content-safe rebind: missing schema-directed transformer, then 2 focused cases
  passed.
- Workspace identity: 2 intended failures for `unlinked` rejection and provisional
  `new` reuse, then 2 passed.
- Canonical journal/materializer lineage: local-only two-version history initially
  emitted only the head; after remediation it passes. Same-ID merge and losing
  remote tombstone cases both pass.
- Secure dataset staging custody: missing persistent storage-key seam, then the
  focused custody/convergence set passed (3 passed).
- Dedicated delta confirmation: staged push plus include-own confirming pull passes.

Additional implementation summary:

- Added durable reconciling/bootstrap-head/expected-head/confirmed-cursor state and
  exact normal-Sync binding validation.
- Added a dedicated first-link Sync cycle and production lazy/restart wiring.
- Added distinct secure dataset staging-key custody with strict restart load.
- Replaced recursive string substitution with schema-directed identity rebinding.
- Rebuilt the first-link journal from reviewed canonical winners, retained local
  lineage, merge versions, and tombstones.
- Added explicit unlinked/random-new/one-to-one workspace handling and same-write-
  transaction snapshot/binding validation.
- Added a durable review-time profile freeze, expired-review restart recovery,
  terminal attention classification, exception-safe complete cleanup, and exact
  stale Personal Context destination cleanup on fresh apply and applying-state
  crash recovery.
- Added exact preallocated `new` scope IDs and preapproval mapping-collision
  prevention, device-only same-ID privacy protection, bounded 101-entry transfer,
  and negotiated batch capability propagation from bootstrap through convergence.
- Split public and private Personal Context push/pull transports and proved
  ordinary post-complete LocalFirst uses only the exact-complete wrappers.

The remediation commit and final aggregate verification counts are recorded by the
controller after this report update. Full-suite execution remains intentionally
skipped under repository policy.

## Structured bootstrap attention integration (2026-08-30)

Status: implemented and focused-verification clean in `f42c173c55`; controller
cross-repository review remains pending and TASK-24727 remains In Progress.

RED evidence:

- Strict schema focus initially failed 3 cases because the bootstrap error
  response and discriminated attention models did not exist; after strict
  parsing and semantic validation, 7 focused schema cases passed.
- Real-httpx client focus initially failed 6 cases because 409 responses had no
  typed attention exception or malformed-body fallback; all 6 passed after the
  client boundary was implemented.
- Link-service focus initially failed 3 cases because typed attention was not
  mapped into a content-free Settings boundary; all 3 passed after the new link
  exception was added without creating review/freeze state.
- Modal focus initially failed 4 cases because there was no exact blocked
  attention surface; all 4 passed after exact rows, disabled approval,
  retry, and cancel were added.
- Public API export focus failed 1 case because the discriminated attention
  alias was not exported; it passed after the lazy package export was added.

GREEN evidence:

- `PYTHONPATH=.:packages/tldw_profile_core/src PYTEST_DEBUG_TEMPROOT=/private/tmp/task3b-attention-structured ../../.venv/bin/python -m pytest -q --disable-warnings Tests/tldw_api/test_personal_context_sync_client.py Tests/Sync_Interop/test_personal_context_first_link.py Tests/Sync_Interop/test_server_sync_service.py`
  - 86 passed, 1 dependency warning in 1.88 seconds.
- `PYTHONPATH=.:packages/tldw_profile_core/src PYTEST_DEBUG_TEMPROOT=/private/tmp/task3b-attention-structured ../../.venv/bin/python -m pytest -q --disable-warnings Tests/UI/test_personal_context_link_modal.py Tests/UI/test_personal_context_link_app_flow.py`
  - 12 passed, 1 dependency warning in 3.25 seconds.
- `PYTHONPATH=.:packages/tldw_profile_core/src PYTEST_DEBUG_TEMPROOT=/private/tmp/task3b-attention-structured ../../.venv/bin/python -m pytest -q --disable-warnings Tests/UI/test_settings_personal_context.py`
  - 52 passed, 2 dependency warnings in 43.48 seconds.
- Ruff over every changed Python source and test: `All checks passed!`.
- Python compilation over changed production/test modules: exit 0.
- CSS authoritative-source reproduction: all bundles reproduce exactly.
- High-severity Bandit gate over changed production modules: exit 0 with no
  finding.
- Impeccable detector over the canonical modal/app UI diff: no finding.
- `git diff --check` and range diff from `73f0a0bf75`: exit 0.

Changed files:

- Typed API: `tldw_api/sync_schemas.py`, `tldw_api/client.py`,
  `tldw_api/exceptions.py`, and lazy exports in `tldw_api/__init__.py`.
- Link/UI production: `Personal_Context/link_service.py`, canonical
  `personal_context_link_modal.py`, and `app.py`.
- Tests: typed client/schema, first-link boundary, canonical modal, and new
  production app-flow coverage.
- Documentation: implementation plan, cross-repository rollout plan, backlog
  task notes, and this report.

Implementation summary and self-review:

- Only a fully validated discriminated attention object crosses the client and
  link-service boundary. Error-code/kind mismatches, extra fields, coercion,
  inconsistent quota deficits, compatible schema claims, and equal purge
  generations fail closed.
- Canonical Settings shows exact safe schema bounds, required/server quotas and
  deficits, or expected/current purge generations. Approval is disabled and the
  owning worker handles retry/cancel without creating or orphaning link state.
- Malformed/unstructured 409 responses never enter the modal and produce only
  the existing generic content-safe notification. Raw server messages/bodies are
  not logged or displayed.
- Successful bootstrap and the previously reviewed convergence path are
  unchanged. No new storage, sync identity, or profile-content representation
  was introduced.

Known limitations/skips:

- No full repository sweep or live external server/keyring session was run,
  per repository policy and the assigned focused scope.
- Cross-repository behavior is contract-pinned to tldw_server `a92e12110d` and
  covered with real-httpx and production-handler tests; final independent
  controller review remains pending.
