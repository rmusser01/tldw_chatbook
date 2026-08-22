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
