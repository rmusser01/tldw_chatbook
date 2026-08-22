# Task 7 implementation report

## Outcome

Task 7 integrates per-session Library policy holders with one shared runtime/store
coordinator, atomic first persistence, and rollback-safe temporary-session
promotion. It implements only the narrow unresolved-operation promotion guard;
it does not add Task 8+ preparation or dispatch behavior.

ADR check: no new ADR was required. This is the direct implementation of
[ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md),
status Accepted.

## Implementation

- Added immutable `ConsoleStagedConversationIdentity`; conversation/title and
  persisted message identities are staged and published only after successful
  transaction exit.
- New sessions capture current config defaults once. Untouched captured defaults
  remain pristine; explicit policy edits dirty the session. Restored conversations
  with no policy row remain write-free Never/Blocked until explicit save.
- Registered policy holders on create/restore and unregistered them on close.
  Runtime construction supplies exactly one coordinator to the Store, and committed
  saves publish only to same-process sibling holders.
- Added atomic service operations for conversation+policy first persistence and
  full temporary promotion. Promotion writes policy, parent-first message lineage,
  attachments, active leaf, context summary, and generic sidecar contributions in
  one `BEGIN IMMEDIATE` transaction through the restricted Task-6 writer and its
  shared trajectory-sequence allocator. Contributions never receive a raw cursor.
- Added a narrow unresolved-operation guard before any promotion write.
- Closed Delivery-1 database verification fallout: legacy v4–v44 tests now reopen
  through one explicit sanitized seed helper; v45 schema-version, table, and index
  inventories include the final objects. The production missing-seed guard was not
  relaxed.

## TDD and verification evidence

### Required RED/GREEN

- Exact three-file RED before production:
  `pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_chat_store_atomic_promotion.py -q`
  — **10 failed, 283 passed**. Five failures covered lifecycle/default capture,
  holder/coordinator, restore, and first policy persistence; five covered immutable
  staging, rollback-safe promotion, contribution atomicity, and the unresolved guard.
- First implementation attempt: **4 failed, 289 passed**. Remaining categories were
  promotion compatibility and post-commit project-context behavior.
- Final exact three-file GREEN: **294 passed, 1 warning** in 16.56 seconds.
- Failure injection covers conversation and policy writes plus contribution writes.
  It compares pre/post ephemeral identity, title, policy flags/snapshot, scope,
  message IDs/parents, and attachments; failed attempts leave zero conversations,
  retry creates exactly one, and identity publication observes `in_transaction ==
  False`.

### Named foundation battery

The exact Task-7 Step-7 eleven-file battery completed with **203 passed, 4
warnings** in 26.67 seconds. It covers Library policy contracts, assistant
generation state, production migration openers, v45 migration/atomicity,
policy/checkpoint repositories, coordinator publication, restricted transaction
contributions, and the two new Store suites.

### Controller database subtree sweep

- Initial complete `Tests/DB/ Tests/ChaChaNotesDB/` run: **100 failed, 1370
  passed, 1 skipped, 4 errors**. Root categories were missing mandatory v45 seeds
  at legacy reopen boundaries, stale v45 version/table/index inventories, and six
  media-owner privacy/exception failures.
- After the shared legacy opener and inventory fixes, the older-migration affected
  group passed **246/246**, with two special v42/v37 fixture cases passing **12/12**.
- Final complete subtree run: **6 failed, 1468 passed, 1 skipped, 4 warnings** in
  114.08 seconds. All six remaining failures are the media parameterizations in
  `Tests/DB/test_core_sqlite_owner_privacy.py`: four unsafe-namespace variants and
  two exception-cause-preservation variants.
- Baseline proof: the exact six node IDs were run in a detached worktree at
  pre-Delivery-1 commit `815d3aec9` and produced the identical **6 failed, 1
  warning** result. The relevant media implementation and test have no diff from
  that commit through Task 7. These are verified unrelated baseline failures; the
  complete database sweep is therefore deliberately not described as fully green.

### Static and supplemental checks

- Scoped Ruff over every modified Python source/test: **all checks passed**.
- `git diff --check`: passed.
- Supplemental runtime/session controller group: **51 passed, 1 pre-existing
  failure**. The failing shutdown fixture omits the newer
  `notes_sync_runtime_owner` and times out before entering its mocked Console
  shutdown; neither `app.py` nor that test is changed by Task 7.
- No full repository suite, profile database, network push, or real user database
  was used.

## Self-review

- Confirmed no policy/checkpoint fields enter sync, Chatbook export, metadata, FTS,
  or generic trajectory projection.
- Confirmed first persistence/promotion mutate live identity, policy, messages, and
  scope only after the persistence service returns from its transaction.
- Confirmed all promotion contributions share one revoked restricted writer and no
  raw cursor crosses the contribution boundary.
- Confirmed unresolved-operation rejection occurs before identity allocation or DB
  work and does not implement future preparation/dispatch semantics.
- Confirmed existing unrelated worktree changes were not overwritten; no secrets,
  profile paths, debug logging, deprecated Settings surfaces, or Task 8+ behavior
  were added.

## Fix round 1 — review correction

### Design correction

Review found that WorkspaceDB membership had been written from inside the
ChaChaNotes promotion call. The two connections cannot share rollback, and a
real two-database policy-failure probe left a dangling membership. The spec,
plan, ADR-079, Task-7 brief, and Backlog notes were amended before production
code: `conversations.workspace_id` is now durable authority, workspace targets
are validated before the Chat transaction, and registry membership is an
idempotent post-commit projection. Projection failure keeps the committed
identity, records pending work, and reconciles from the durable row on retry or
restart without minting a second UUID/membership.

The same review corrected three lifecycle boundaries. Runtime/Store now owns a
live future-session defaults provider used by every `Store.create_session`
entrypoint; rollback and state replacement unregister/rebind holders; and sync
restore registers a fail-closed placeholder without reading policy. Production
resume and launch hydration await coordinator `to_thread` loading, with
generation checks and activation only after hydration. No Task-8 preparation or
dispatch behavior was introduced.

### RED and failure injection

- Exact fix-round three-file RED before production: **10 failed, 19 passed**.
  Categories were missing async hydration/production await, live defaults,
  rollback unregister, restore-state rebind, precommit workspace orphaning,
  postcommit projection recovery, and restart reconciliation.
- First post-implementation attempt: **29 passed**. After the complete review
  matrix and runtime probes were added, the fix-specific three files finished
  **35 passed, 1 warning**.
- The promotion matrix injects at conversation, policy, both message indices,
  position-0 attachment storage, extra-attachment sidecar, active leaf, context
  summary, and both generic contributions. Each failure preserves the full
  captured in-memory identity/policy/message/attachment/scope/active-leaf/
  summary/unresolved-operation state, leaves all five durable bundle tables at
  zero rows, and retries to exactly one conversation, two messages, one policy,
  one sidecar attachment, and two contributions.
- Real ChatDB + WorkspaceDB cases prove policy/message/attachment/contribution
  failure creates zero membership; postcommit membership failure leaves one
  committed conversation and pending projection; retry and crash/restart each
  converge to one UUID and one membership.
- First persistence observes the policy insert while `in_transaction == True`
  and the session still owns its pre-call ID/title. Durable success verifies
  position 0 in the message row and the extra attachment sidecar separately.

### Final verification

- Exact Task-7 Store battery: **316 passed, 1 warning**.
- Exact named foundation battery: **225 passed, 4 warnings**.
- Chat persistence service: **71 passed, 1 warning**.
- Stale-version-comment owners and the real v45 exact pin: **12 passed, 1
  warning**.
- Targeted runtime/hydration/workspace/UI group: **94 passed, 2 failed**. Both
  failures reproduce unchanged at exact pre-fix commit `51ace1b3b`:
  `test_app_fences_console_then_drains_buddy_before_profile_teardown` times out
  because its fixture lacks `notes_sync_runtime_owner`, and
  `test_a_launch_built_controller_is_not_sticky_when_console_opens` reaches its
  vacuity guard with `configured_model is None`. Neither is fix-induced; the
  temporary detached baseline worktree was removed.
- Scoped Ruff and `git diff --check`: passed. Per the fix-round ruling, the full
  DB subtrees were not rerun because no schema or migration fixture changed;
  the six previously proven unrelated media baseline failures remain visible in
  the original report above.

### Fix-round self-review

- Confirmed no WorkspaceDB write occurs before the ChaChaNotes bundle commits,
  and projection failure cannot revert a committed session to ephemeral.
- Confirmed every production `store.create_session` call funnels through the
  runtime-installed provider; changing Settings affects later sessions only.
- Confirmed restored policy reads occur through coordinator `asyncio.to_thread`,
  the loop remains responsive, stale generations do not publish old authority,
  and production does not activate the restored session while placeholder
  authority remains.
- Confirmed holder rollback/close/state-replacement cleanup, overlapping-ID
  rebinding, stale-holder non-publication, retry idempotence, restricted shared
  transaction writer use, and the narrow unresolved-operation guard.
- Added the cross-database rollback incident to
  `backlog/docs/lessons-testing-evidence.md`; no deprecated Settings surface,
  profile database, full-suite run, push, or Task-8+ behavior was added.

## Fix round 2 — mandatory atomic promotion adapter

### Review correction and RED

Review found one remaining compatibility escape: an instance-shadowed
`persist_session_if_needed` could divert temporary promotion away from the
production atomic bundle path. The valid exact three-file RED was **2 failed,
316 passed**. A real ChatDB + WorkspaceDB regression proved the shadow was
called, and a second regression proved an adapter without the atomic promotion
operation did not refuse before writes.

The Store now has one promotion path. Any configured persistence adapter must
provide `promote_console_conversation_bundle`; otherwise promotion fails closed
before identity allocation or durable writes. The legacy conversation/message/
scope partial-write branch and its instance-shadow condition were deleted.
Fakes that support promotion now implement the atomic adapter explicitly.

The first-persistence timing regression now intercepts the actual
`publish_committed_identity` call. Its injected policy failure publishes
nothing and leaves zero conversations; retry creates exactly one conversation,
observes the old live ID/title immediately before publication with
`connection.in_transaction is False`, and observes the committed ID/title only
after publication. The real two-database shadow regression confirms the shadow
is never called and a later atomic policy failure leaves zero Chat bundle rows,
zero workspace memberships, and no holder binding mutation.

### Final verification

- Exact Task-7 Store battery: **318 passed, 1 warning**.
- Exact eleven-file foundation battery: **227 passed, 4 warnings**.
- Focused ephemeral/project-context/persistence/hydration/workspace/version
  group: **150 passed, 1 warning**.
- The directly edited ephemeral promotion module reran after lint-only cleanup:
  **17 passed, 1 warning**.
- Scoped Ruff over every modified Python file and `git diff --check`: passed.
- Per the review ruling, the unchanged runtime/UI baseline cases and complete DB
  subtrees were not rerun. Their exact previously verified baseline failures
  remain recorded above; this round changed no schema or migration fixtures.

### Documentation and self-review

- Corrected the full v44 migration module reference so the v45 test owns the
  exact current-version pin, and documented `create_conversation` workspace
  handling as validation only with membership as a separate post-commit
  projection.
- Confirmed no production route, instance attribute, contribution shape, or
  fake can select a non-atomic promotion path; adapter absence is observable
  before any write.
- Confirmed project-context post-commit behavior and the durable
  `workspace_id` projection model remain intact without weakening ChatDB bundle
  atomicity. No new reusable incident beyond the already documented boundary
  lessons was found, so no additional lesson entry was added.
- No full suite, database subtree sweep, profile database, push, deprecated
  Settings edit, or Task-8+ behavior was introduced.
