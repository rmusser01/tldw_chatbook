# Native source readiness for exact Personal Context evidence

Date: 2026-09-26
Task: [TASK-25907.13](../tasks/task-25907.13%20-%20Audit-native-source-readiness-for-exact-memory-evidence.md)
Inspected baseline: `1dab946be8`; native Python 3.12.11, existing isolated worktree.
Status: audit complete and reviewed; no source adapter shipped.

## Finding

Existing message revisions supply stable lineage, not an unconditional archive
of original message text. Notes expose the mutable current head. A first exact
source adapter should therefore qualify currently authorized text; historical
text must remain unavailable unless a separately governed exact representation
actually survives. This is a recommendation, not approval to add an adapter.

The Muse diagram's verified-claim index and inspectable evidence are useful
ideas. Its file layout and background jobs do not establish these source-owner
guarantees. Existing Personal Context, conversation and Notes owners remain the
places to implement them; this audit creates no parallel memory store.

## Source matrix

| Candidate | Existing guarantee | Limit for exact memory evidence |
| --- | --- | --- |
| Current `messages.content` | The nondeleted message getter reads the current canonical row; semantic mutations coordinate revision lineage. | Exact text is possible only after trusted owner/visibility checks, version binding and publication recheck. The row is mutable; its existence does not establish conversation visibility or profile/workspace permission. |
| Semantic revision metadata | Opaque revision IDs, source conversation/message, predecessor, sequence, normalized role and a live locator. | Metadata is digest-free and body-free. The live locator is retired on semantic replacement; identity alone cannot recover old text. |
| Retired semantic revision under a trace policy | Reachable policies can retain a sanitized envelope or a content-free omission. | Credential/PII filtering can change the representation. No reachable policy can mean no retained body. A historical projection cannot be labelled the original exact `messages.content` or reused as unrestricted memory evidence. |
| Current Notes row | Current content, title, optimistic version and deletion state; version probes use one read snapshot. | No historical-body getter in the inspected Notes path. A title-only edit advances the row version. Content is stored as-is; titles are stripped. Choose one representation rather than joining fields. |
| `sync_log` | Bounded frontier proofs: live messages retain current/previous versions; Notes retain the current version. Content-free deletion proofs are treated separately. | This is a synchronization log, not a historical evidence archive. Pruning and deletion can remove old bodies. |
| Message-owned citation snapshot | Governed snapshot text, storage mode, lineage and transformation metadata; separate identity and capability policy. | Snapshot text may be a transformed retrieved chunk rather than original source text. Its message owner, retention, masking and namespace do not become Personal Context authority by copying IDs. |

Inspected owners:

- [Message creation and initial metadata](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L13237), [semantic coordination](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L13515), [current message getter](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L14154).
- [Semantic projection](../../tldw_chatbook/Chat/console_semantic_revision.py#L54), [mutation transaction](../../tldw_chatbook/Chat/console_semantic_revision.py#L644), [policy materialization](../../tldw_chatbook/Chat/console_semantic_revision.py#L859), [locator retirement](../../tldw_chatbook/Chat/console_semantic_revision.py#L1012).
- [Notes current getter/version probe](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L17603), [Notes mutation](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L18947), [sync-log retention](../../tldw_chatbook/DB/ChaChaNotes_DB.py#L20973).
- [Citation snapshot shape](../../tldw_chatbook/Chat/citation_trace_models.py#L612), [citation identity namespaces](../../tldw_chatbook/Chat/citation_trace_identity.py#L73), [strict native locator authorization](../../tldw_chatbook/Chat/citation_source_locators.py#L872).

These are source inspection findings, not qualification of an unimplemented
Personal Context resolver. The database inspected here supports schema 73;
older schema numbers in general project guidance do not describe this checkout.

## Requirements the first adapter must preserve

1. **Bind authority before reading a body.** A supplied message ID, expected
   conversation ID, path, client ID or `role=user` is not a trusted grant.
   `project_semantic_revision_provider_message` verifies conversation membership;
   it does not implement Personal Context profile/workspace/purpose authorization.
   Its internal envelope query does not filter soft deletion. Use explicit current
   message and conversation visibility checks in the future adapter.
2. **Resolve one representation and version.** Prefer exact stored text in
   `messages.content` initially. A semantic revision covers the larger envelope,
   including attachments and sidecars; provider projection can compose multimodal
   parts. Never apply original offsets to that composed or sanitized output.
   Bind the representation SHA and span SHA from the completed
   [digest primitive](personal-context-exact-span-digests.md), plus owner/version
   identity. An edit outside the span still invalidates the representation.
3. **Recheck before publication.** Read source identity/version/text under a
   coherent owner transaction, then fence source mutation, parent visibility,
   scope changes, profile lock and revocation before returning a quote or metadata.
   The semantic graph epoch can detect visibility/ownership changes but is not
   a replacement for current authority. No failed check triggers another-store
   lookup, path search or historical trace fallback.
4. **Separate meanings.** Exact digest equality establishes text identity.
   Trusted capture origin establishes source role; persisted `user` role alone
   cannot distinguish an import or quotation from a direct user statement.
   Semantic support, user approval and temporal validity remain separate.
5. **Preserve lifecycle ownership.** Notes and sync logs must not retain extra
   bodies merely to make an evidence lookup work. A captured excerpt would be a
   new governed derivative requiring the accepted forgetting and disclosure
   controls, not a workaround for missing history. Trace retention after a
   canonical hard delete does not mean that a memory reader may access it.

The V1 [Notes/chat locator payloads](../../tldw_chatbook/Chat/citation_source_locators.py#L378)
contain item/chunk/message identities but no exact content version, representation
or span digests. `resolver_payload_version` identifies the locator format, not
source content. Their current authority checks are useful precedent; extending
those contracts or bridging their namespaces requires its own scoped design.

## Recommended sequence

1. Specify a read-only current-message adapter with runtime-owned authority,
   nondeleted parent/message checks, strict version/span verification and
   disposable results. Test authorized success alongside denied, imported,
   deleted-parent, changed-source, Unicode and publication-race cases. Add no
   source persistence, provider send or automatic refresh in that slice.
2. Qualify Notes current text separately; declare mutable-head or explicit
   captured-representation semantics honestly. A row version is not proof that
   the owner retains the former body. Do not advertise historical note revision
   resolution through the current getter.
3. Integrate canonical V2 bindings and any retained excerpt only after the
   separate compatibility, authorization, dependency-forgetting, suppression and
   destination/purpose controls have shipped and been verified. Background
   consolidation remains gated by those controls.

ADR required: no new ADR for this audit and test-only fixture corrections.
ADR paths: [ADR-185](../decisions/185-versioned-profile-evidence-and-temporal-claims.md),
[ADR-097](../decisions/097-console-reference-backed-semantic-trace-ledger.md),
[ADR-024](../decisions/024-rag-citation-provenance-and-source-resolution.md).
Reason: document existing boundaries and exercise their existing guards. No new
storage, retention, authority interface or runtime policy is selected here.
Future implementations must also retain [ADR-186](../decisions/186-dependency-aware-personal-context-forgetting.md)
and [ADR-187](../decisions/187-personal-context-provider-disclosure-authority.md).

## Native verification and fixture repair

The initial targeted run had **110 passes / one failure**: a retention test
issued raw message deletion after `add_message` had attached a semantic revision.
SQLite correctly rejected it. The unchanged guard is in
[the migration](../../tldw_chatbook/DB/migrations/chachanotes_v56_to_v57_semantic_mutation_guard.sql#L46).
Repair used the public semantic coordinator under the caller's transaction,
retained the sync-log absence assertions, and added positive pre-delete and
canonical-row absence controls. The repaired test passed on its own.

The complete affected retention module then exposed the analogous raw
conversation cascade (**26 passes / one failure**). Its legacy cascade now uses
an actual v54 database reopened through current migrations, asserting that the
message has no live semantic revision before deletion. A separate current-schema
test asserts unauthorized tracked-message cascade rejection and rollback of
conversation, message content and sync frontier. No trigger, production
permission or private authorization bypass was changed. Only these deletion
fixtures and one existing tuple's formatting changed in the test module.

The expanded selection passed **132 distinct cases**: 26 revision coordinator,
73 locator, 28 retention, one masked retired-revision projection, one Notes
version and three citation-observation authority/concurrency cases. The final
receipt after static cleanup passed the same 132 cases in 22.85 seconds;
repeated runs are not additional cases.

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest \
  Tests/Chat/test_console_semantic_revision_coordinator.py \
  Tests/Chat/test_citation_source_locators.py \
  Tests/Chat/test_console_trace_pii_masks.py::test_retired_revision_remains_available_through_masked_trace_artifact \
  Tests/DB/test_chachanotes_sync_log_retention.py \
  Tests/ChaChaNotesDB/test_chachanotes_db.py::TestNotesAndKeywords::test_add_and_update_note \
  Tests/Chat/test_citation_source_observations.py::test_observation_validates_namespace_inventory_trace_policy_and_authorization \
  Tests/Chat/test_citation_source_observations.py::test_observation_read_intersects_stored_capabilities_with_current_authorization \
  Tests/Chat/test_citation_source_observations.py::test_observation_read_serializes_with_concurrent_revocation \
  -q --tb=short --show-capture=no \
  --basetemp=/private/tmp/memory-source-audit-clean-20260926 \
  --junitxml=/private/tmp/memory-source-audit-clean-20260926.xml
.venv/bin/python -m ruff check --no-cache Tests/DB/test_chachanotes_sync_log_retention.py
.venv/bin/python -m ruff format --no-cache --check Tests/DB/test_chachanotes_sync_log_retention.py
```

The existing `Tests/conftest.py` redirected config/data and keyring; owner tests
used in-memory or temporary SQLite. Whole-file Ruff and formatting passed after
one context-manager lint correction and formatting cleanup. No real profile,
app launch, provider, network, full application sweep or companion-server test
was used. Existing synthetic authorization/concurrency checks qualify their
current owners; they do not qualify a future memory publication path, transcript
capture authenticity, complete forgetting or device-only provider disclosure.

## Diagnostic run receipts

Each command ran from the isolated worktree with native Python and the same
synthetic `Tests/conftest.py` isolation. The diagnostic runs were:

| Run | Result | JUnit receipt |
| --- | --- | --- |
| Initial selection below | 110 passed / one obsolete raw-message-delete failure, 16.62 s | `/private/tmp/memory-source-audit-20260926.xml` |
| Repaired message case below | One passed, 1.26 s | `/private/tmp/memory-source-audit-repair-20260926.xml` |
| Complete retention module below | 26 passed / one obsolete raw-conversation-cascade failure, 10.22 s | `/private/tmp/memory-source-retention-module-20260926.xml` |
| Expanded selection before static cleanup | 132 passed, 23.15 s | `/private/tmp/memory-source-audit-final-20260926.xml` |
| Final expanded command above | 132 passed, 22.85 s | `/private/tmp/memory-source-audit-clean-20260926.xml` |

Exact initial command:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest \
  Tests/Chat/test_console_semantic_revision_coordinator.py \
  Tests/Chat/test_citation_source_locators.py \
  Tests/Chat/test_console_trace_pii_masks.py::test_retired_revision_remains_available_through_masked_trace_artifact \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_soft_deleting_a_message_removes_its_body_from_sync_log \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_soft_deleting_a_note_removes_its_body_from_sync_log \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_hard_deleting_a_message_removes_its_body_from_sync_log \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_editing_a_message_does_not_accumulate_old_bodies \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_a_live_message_keeps_only_the_frontier_its_readers_need \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_editing_a_note_keeps_only_the_current_version \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_prune_sync_log_is_idempotent \
  Tests/ChaChaNotesDB/test_chachanotes_db.py::TestNotesAndKeywords::test_add_and_update_note \
  Tests/Chat/test_citation_source_observations.py::test_observation_validates_namespace_inventory_trace_policy_and_authorization \
  Tests/Chat/test_citation_source_observations.py::test_observation_read_intersects_stored_capabilities_with_current_authorization \
  Tests/Chat/test_citation_source_observations.py::test_observation_read_serializes_with_concurrent_revocation \
  -q --basetemp=/private/tmp/memory-source-audit-20260926 \
  --junitxml=/private/tmp/memory-source-audit-20260926.xml
```

Exact message-recovery and affected-module commands:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest \
  Tests/DB/test_chachanotes_sync_log_retention.py::test_hard_deleting_a_message_removes_its_body_from_sync_log \
  -q --tb=short --show-capture=no \
  --basetemp=/private/tmp/memory-source-audit-repair-20260926 \
  --junitxml=/private/tmp/memory-source-audit-repair-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest \
  Tests/DB/test_chachanotes_sync_log_retention.py \
  -q --tb=short --show-capture=no \
  --basetemp=/private/tmp/memory-source-retention-module-20260926 \
  --junitxml=/private/tmp/memory-source-retention-module-20260926.xml
```

The first expanded run used the exact final expanded command above with
`--basetemp=/private/tmp/memory-source-audit-final-20260926` and
`--junitxml=/private/tmp/memory-source-audit-final-20260926.xml` substituted
for its two clean-receipt flags. Receipts are temporary local verification
artifacts; the recorded failure history is not a claim that rerunning corrected
fixtures reproduces their earlier failures.

## Review and closeout

A bounded independent read-only review checked the source matrix, guard,
coordinator path, migrated legacy cascade and current rollback test. It found no
Critical or Important issue. One Minor reproducibility gap was resolved by the
exact diagnostic commands and receipt table above. The reviewer inspected source;
it did not run tests or independently validate JUnit receipts. Native receipt,
link, ownership and tracker verification belong to the local coordinator.

All eight task criteria are checked and Implementation Notes are recorded;
Backlog CLI set the intended TASK-25907.13 file Done. Scoped checks passed
14 unique family IDs, 74 local links, 15 source-line anchors and the final
132 distinct JUnit cases. Original task files/criteria, runtime/schema/core
fixtures, historical evidence and unrelated test AST are preserved. The testing
lesson is append-only; independent TASK-25907.10 and its roadmap suffix retain
their original bytes.

Allocation was checked against 565 available refs and 79 worktrees before
filing, then 566 refs and 81 worktrees at closeout. Only the canonical new task
path was found. No fetch or open-PR query was made; integration must recheck.
The five owned paths are report, task, roadmap prefix, affected test module and
testing lesson. Keep the local branch/worktree; no push, PR or merge.
