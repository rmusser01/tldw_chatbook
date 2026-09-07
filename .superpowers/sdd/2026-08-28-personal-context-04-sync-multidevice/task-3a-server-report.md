# Task 3a server bootstrap report

## Status

Implemented the server-owned Personal Context canonical bootstrap slice. TASK-13148
remains **In Progress** for controller-owned final verification and closure.

Server commit: `bf794d753c feat(sync): bootstrap canonical personal context profile`.

## TDD evidence

The bootstrap test file was created before production edits.

### RED

Command run before implementation with the repository shell baseline:

```text
python3 -m pytest --confcutdir=tldw_Server_API/tests/Sync \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Output:

```text
Python 3.9.6
ImportError: cannot import name 'UTC' from 'datetime'
packages/tldw_profile_core/src/tldw_profile_core/canonical.py:5
```

This was a collection RED caused by the baseline interpreter being Python 3.9
while `tldw_profile_core` requires Python 3.11. It prevented observing the
intended missing-bootstrap behavioral RED before implementation. Once the
isolated 3.11 runner was supplied, iterative RED runs exposed and corrected the
test support signature, a missing Sync-domain import, and the test helper's
override merge. No repository dependencies were changed.

### GREEN

Command:

```text
PYTHONPATH=/tmp/tldw-pc-sync-yaml:. \
UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache \
uv run --no-project --python /Users/macbook-dev/.local/bin/python3.11 \
  --with pytest --with fastapi --with httpx --with psutil --with loguru \
  --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 \
  pytest --confcutdir=tldw_Server_API/tests/Sync \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Output: `9 passed, 5 warnings in 0.80s`.

Focused affected regressions:

```text
... pytest [bootstrap, Personal Context adapter/materializer/transport,
profile bootstrap, service, store, repository, service] -q -k 'not postgres'
```

Output: `443 passed, 9 deselected, 5 warnings in 21.75s`.

The two deselected Store tests need the unavailable `pg_database_config`
PostgreSQL fixture. A broader first attempt also showed model and endpoint test
collection needs uncached `asyncpg` and `cachetools`; those tests were not
changed by this slice.

The directly affected factory regression command completed with
`14 passed, 5 warnings in 0.42s`.

## Implementation

- `SyncV2ProfileManager.bootstrap_personal_context()` creates/reads canonical
  Personal Context state through `PersonalContextService`, binds only opaque
  metadata in the Sync dataset, returns one version cursor, and validates
  user/device/authority/schema/quota/purge/capability boundaries.
- Device delivery uses a `SyncKeyRecord` with `wrapped_for="device"`; plaintext
  integrity keys do not enter response metadata, durable Sync data, or logs.
- `complete_personal_context_link()` performs the narrow cursor-checked state
  transition. `SyncV2Service.push()` rejects Personal Context envelopes until
  that state is complete.
- The factory supplies the existing per-user Personal Context service resolver.
  The device wrapping callback is deliberately injected; absent custody fails
  closed rather than inventing a second key path.

## Changed files

- `tldw_Server_API/app/core/Sync/v2/profile.py`
- `tldw_Server_API/app/core/Sync/v2/service.py`
- `tldw_Server_API/app/core/Sync/v2/factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`
- `IMPLEMENTATION_PLAN_personal_context_bootstrap.md`
- `backlog/tasks/task-13148 - Bootstrap-Personal-Context-canonical-profile.md`

The transport fixture is explicitly post-reconciliation (`link_state: complete`)
so its established transport/CAS tests continue to exercise the admitted path.

## Static verification and self-review

- Python 3.11 `py_compile` passed for all changed production and focused test
  files.
- `git diff --check` passed.
- Self-review confirmed canonical reads stay in `PersonalContextService`, Sync
  stores only opaque binding/key-record state, device-only records/proposals are
  excluded, and no cross-database transaction is claimed.
- Ruff and Bandit are not installed in the isolated interpreter. Attempts to
  provision each with `uv run --no-project --with ruff` and `--with bandit`
  failed because package-index DNS is blocked; no dependency files were changed.

## Concerns / follow-up

- The production composition currently has no concrete device-key wrapping
  provider; the new callback fails closed until the authenticated device custody
  provider is wired by the transport/key-enrollment work.
- Bootstrap retries a manifest-version-stable canonical read, but canonical and
  Sync persistence remain separate stores because the design intentionally has
  no cross-database transaction.
- Controller verification should run the PostgreSQL-specific Store tests and
  endpoint/model suites in a fully provisioned environment, then run Ruff and
  Bandit before closing TASK-13148.

## Review round 1 remediation

Commit pending this report update adds the following reviewer-directed hardening:

- The factory now injects RSA-OAEP-SHA256 wrapping using the registered device's
  `personal_context_wrapping_public_key` capability; a generated private key
  regression decrypts the returned ciphertext to prove it is not the previous
  hash placeholder.
- `authority_id` is selected by the service/factory and no longer comes from the
  typed HTTP bootstrap request. Typed `/personal-context/bootstrap` and
  `/personal-context/complete` routes carry only authenticated user context and
  registered-device identifiers.
- Generic `/datasets/enroll` rejects all Personal Context domains and the
  `personal_context` metadata namespace. The completion gate stores a
  device-specific receipt carrying profile, integrity-key ID, purge generation,
  and bootstrap cursor; another registered device remains blocked.

Evidence after these changes:

```text
pytest test_sync_v2_personal_context_bootstrap.py \
  test_sync_v2_personal_context_transport.py test_sync_v2_factory.py -q
27 passed, 5 warnings in 0.81s
```

Python 3.11 `py_compile` passed for factory/profile/service, the Sync endpoint
and schema modules, and the bootstrap test. `git diff --check` passed.

Remaining review limitation: receipt persistence currently uses the existing
server-owned dataset metadata update, not a dedicated Sync-DB receipt table with
compare-and-set semantics. The canonical snapshot still relies on manifest
version retry rather than a newly introduced single Personalization DB snapshot
method. These must be resolved before the reviewer can consider TASK-13148 ready.

## Review round 1 completion

The two outstanding architecture findings are now implemented in the next
server commit:

- `PersonalContextRepository.sync_bootstrap_snapshot()` reads the manifest,
  bounded complete scope/record/proposal head sets, and integrity-key identity
  in one Personalization transaction. The service filters device-only records
  and proposed records, keeps tombstone/expired lifecycle heads, derives the
  cursor there, and rejects more than the explicit 1,000-head bound.
- `SyncV2Store` persists one receipt row keyed by authenticated user, dataset,
  device, profile, integrity-key ID, purge generation, and bootstrap cursor in
  an atomic Sync DB transaction. Push checks that exact current receipt; it no
  longer trusts dataset metadata or a dataset-global completion flag.

Final targeted command:

```text
pytest test_sync_v2_personal_context_bootstrap.py \
  test_sync_v2_personal_context_transport.py \
  test_personal_context_service.py -q
33 passed, 5 warnings in 2.04s
```

Python 3.11 compilation and `git diff --check` passed. The earlier paragraph
describing metadata-only receipts and manifest retry is superseded by this
section; no cross-database transaction is claimed.

## Review round 2 completion

- Fresh canonical bootstrap now calls `ensure_sync_profile()` before the single
  repository snapshot read; concurrent canonical creation resolves through the
  canonical service conflict path.
- Completion now compares the exact `sync_bootstrap_snapshot()` cursor used for
  bootstrap, so lifecycle heads and full bounded proposal coverage cannot cause
  a false stale cursor.
- The device-receipt table is created in both Sync database schema definitions;
  receipt upsert verifies the current server-owned dataset profile/key/purge
  binding in the same Sync transaction. Runtime DDL and dataset metadata rewrite
  were removed.
- Endpoint error conversion now returns stable Personal Context reason codes and
  non-500 typed HTTP statuses.

Final focused evidence: `33 passed, 5 warnings in 1.05s`; Python 3.11 compile
and `git diff --check` passed.

## Review round 2 follow-up: wrapping-key rotation

- Key-record reuse is now fenced by the SHA-256 fingerprint of the current
  registered RSA public key. A changed registered key revokes the superseded
  device-wrapped key record and produces a fresh RSA-OAEP-SHA256 wrapper with a
  fingerprint-qualified record identifier.
- Completion no longer rewrites dataset metadata after the receipt CAS; the
  receipt table remains the sole completion source.
- `_safe_sync_v2_http_error` maps every `PersonalContextBootstrapError` reason
  code to a stable, content-free response before the generic `SyncStoreError`
  path.

RED evidence during TDD:

```text
test_bootstrap_rewraps_after_registered_public_key_rotation
AttributeError: 'SyncV2Store' object has no attribute 'revoke_key_record'
```

GREEN evidence:

```text
PYTHONPATH=/tmp/tldw-pc-sync-yaml:. /tmp/tldw-pc-sync-uv-cache/archive-v0/aBIxc7vhsrTCWOyBjLA2V/bin/pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
12 passed, 5 warnings in 0.78s
```

`python3.11 -m py_compile` for the changed factory/profile/service/store/DB/
endpoint modules and `git diff --check` also passed. The isolated test runner
lacks `asyncpg`, so authenticated FastAPI endpoint collection could not be run
there; the endpoint mapping is compile-checked and the exact collection error
was `ModuleNotFoundError: No module named 'asyncpg'`.

## Receipt CAS schema follow-up

- Removed remaining request-time receipt-table DDL and delete/insert sequence.
  Completion now reads and validates the current server-owned dataset binding in
  the same transaction before the portable conflict-upsert. This fails closed on
  a stale profile/key/purge binding.
- Added deterministic SQLite/PostgreSQL DDL and CAS-source contract coverage:
  both canonical schema strings must define the receipt primary key, while the
  store source must contain the portable `ON CONFLICT` update and no runtime
  DDL/delete path.

Focused result: `13 passed, 5 warnings in 0.92s`; changed DB/store modules
compiled and `git diff --check` passed. An attempted expanded `uv run` for API
coverage could not resolve cached dependency metadata and attempted DNS access
for `aiosqlite`; the pre-provisioned direct runner still lacks `asyncpg`.

## Replacement-worker endpoint verification

Replacement-worker commit: `598c88339b test(sync): cover personal context endpoints`.
TASK-13148 remains **In Progress** for controller-owned re-review and closure.

The endpoint suite now covers every stable, content-free reason code mapped by
`_safe_sync_v2_http_error`: bootstrap/device/authority-invalid/capability/
schema/quota/key-custody/snapshot-unavailable/snapshot-unstable/purge-stale/
link-unavailable/cursor-stale/authority-mismatch. Each parameterized response
asserts its HTTP status and typed `detail.error_code`, permits only the stable
`error_code`/`message` shape, and rejects profile plaintext, integrity-key IDs,
ciphertext, and wrapped-key material.

The real authenticated endpoint test uses `sync_v2_service_for_user("101")`
with only authenticated-user and constructed-service dependency overrides. It
registers a generated RSA public key, omits `authority_id` from the bootstrap
request, decrypts the returned RSA-OAEP-SHA256 integrity-key wrapper using that
device's private key, verifies canonical manifest creation, proves stale
completion is rejected, completes the link, verifies the receipt, and admits a
signed `personal_context.manifest` through `/push`. It also executes a real
missing-device bootstrap request before registration.

### RED

Before the minimal production correction, the following exact focused command
collected the real FastAPI test and failed its fresh-account bootstrap:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -k personal_context
```

Result: `1 failed, 14 passed, 66 deselected`. The bootstrap response was HTTP
503 with `personal_context_snapshot_unavailable` rather than 200. The exact
underlying exception was `ValueError: timestamp precision must not exceed
milliseconds` from `packages/tldw_profile_core/src/tldw_profile_core/canonical.py:29`.
`PersonalContextService.create_profile()` obtains one `now = self._now()` and
writes it to both `ProfileManifest.created_at`/`updated_at` and
`ProfileScope.created_at`/`updated_at`; the default `datetime.now(UTC)` supplied
arbitrary microseconds. The correction rounds at `_now()` to milliseconds. This
is the canonical clock boundary used by profile creation and mutations; fixing
only Sync snapshot serialization would leave invalid canonical timestamps in
storage and miss non-Sync mutations.

### GREEN

Focused endpoint coverage after the `_now()` correction:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -k personal_context
```

Result: `15 passed, 66 deselected, 4 warnings in 2.53s`.

Targeted full endpoint module:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
```

Result: `81 passed, 41 warnings in 6.08s`.

Personal Context bootstrap regression module:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Result: `13 passed, 3 warnings in 1.10s`.

The following static checks both passed before the commit:

```text
/Users/macbook-dev/.local/bin/python3.11 -m py_compile tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/app/core/Personalization/personal_context_service.py
git diff --check
```

The additional `uv --with` packages were ephemeral collection dependencies for
the authenticated server import graph; no dependency or lock files changed.

## Review round 3: cursor and receipt-CAS remediation

Server commit: `41f602657f fix(sync): bind personal context cursors to keys`.
TASK-13148 remains **In Progress** for controller-owned re-review and closure.

- `PersonalContextService.sync_bootstrap_snapshot()` now hashes
  `integrity_key_id` with manifest/purge/head values. A canonical integrity-key
  generation change therefore invalidates an otherwise identical bootstrap
  cursor.
- Completion requires the opaque Sync dataset binding key ID to equal the
  current canonical snapshot key ID before the receipt CAS. The receipt is
  written with the canonical snapshot key ID, never stale metadata.
- The exact Store CAS outcome `personal_context_link_binding_stale` becomes the
  stable content-free `PersonalContextBootstrapError` reason of the same name;
  the typed endpoint maps it to HTTP 409. Completion reason-code coverage
  includes it.
- Bootstrap now requires `sync_bootstrap_snapshot()` and fails closed as
  `personal_context_snapshot_unavailable` when a legacy/separate-read service
  is supplied. The obsolete separate-read cursor/retry helpers were deleted.

### RED evidence

The key-only snapshot transition was initially unchanged, proving the cursor
omitted key identity:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Personalization/test_personal_context_service.py -q -k integrity_key_changes
```

Result: `1 failed, 20 deselected`; both cursors were exactly
`personal-context-bootstrap-v1:5d62ce...` despite `integrity_key_id` changing
from v1 to v2.

The missing transactional seam initially produced
`personal_context_key_custody_unavailable` instead of the required
`personal_context_snapshot_unavailable`. The real endpoint stale-binding run
initially produced HTTP 204 for the pre-CAS binding mismatch and HTTP 500
`sync_store_error` for the newly mapped reason-code case:

```text
... test_sync_v2_endpoints.py -q -k 'stale_integrity_binding or personal_context_link_binding_stale'
2 failed, 81 deselected
```

To prove the Store boundary rather than mock an error, the deterministic test
transitions the real dataset binding immediately before delegating to the real
`complete_personal_context_link_receipt()` CAS. With only the profile-boundary
translation temporarily removed, its exact endpoint response was HTTP 500:

```text
{"detail":{"error_code":"sync_store_error","message":"Internal sync storage error while processing request."}}
```

### GREEN evidence

Focused Personal Context service/bootstrap/endpoint coverage:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Personalization/test_personal_context_service.py tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -k 'personal_context or integrity_key_changes or transactional_canonical_snapshot'
```

Result: `55 passed, 66 deselected, 4 warnings in 5.51s`.

Full focused bootstrap module:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Result: `15 passed, 3 warnings in 0.79s`.

Full endpoint module:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
```

Result: `85 passed, 41 warnings in 7.88s`.

Python 3.11 compilation passed for the changed service/profile/endpoint and
test modules; `git diff --check` passed before commit. No dependency or lock
files changed.

## Quality remediation: protected enrollment, durable receipt locking, and outage visibility

Replacement-worker quality commit: `4286a90fe8 fix(sync): protect personal context receipt state`. TASK-13148
remains **In Progress** for controller-owned re-review and closure.

- Generic Sync dataset re-enrollment now preserves existing server-owned
  Personal Context domains and the opaque `personal_context` metadata binding.
  An older client can update its ordinary metadata without orphaning receipts
  or key records; an attempted generic binding overwrite is likewise ignored.
  The Profile Manager uses a narrow server-authoritative Store method for its
  own canonical binding transitions.
- Receipt completion now belongs to `SyncDatabase`. PostgreSQL locks the owned
  dataset row with `FOR UPDATE` in the same transaction before validating the
  binding and upserting the receipt. SQLite continues to use its transactional
  write serialization. The regression executes a PostgreSQL backend contract
  that proves the lock statement and upsert share one transaction object and
  occur in that order; it also models a binding transition observed under the
  lock and proves no stale receipt is written. This is not a claim of live
  PostgreSQL integration coverage.
- Receipt lookup no longer converts every storage failure to `False`. The DB
  error now crosses the established Sync error boundary, and endpoint coverage
  verifies a content-free `sync_store_error` response rather than a false
  reconciliation/link-incomplete hint.
- The unbounded process-local per-user bootstrap lock registry was removed;
  durable canonical and Sync transactions now own correctness.

### RED evidence

The new protection/CAS tests initially exposed the missing behavior:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -k 'generic_reenrollment_preserves_server_owned_personal_context_state or personal_context_receipt_locks_binding or personal_context_receipt_rejects_transition or reenrollment_preserves_personal_context'
```

Result: `4 failed, 196 deselected`. Generic re-enrollment had erased
`metadata["personal_context"]`; the new DB receipt method did not yet exist;
and direct generic re-enrollment had removed the reserved domains. A subsequent
facade-boundary RED was
`test_sync_store_facade_does_not_embed_sql_statements`: receipt lookup still
embedded SQL in `SyncV2Store`; lookup was then moved to `SyncDatabase`.

### GREEN evidence

Final focused Personal Context coverage:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -k 'personal_context or generic_reenrollment or reenrollment_preserves_personal_context or receipt_storage_failure'
```

Result: `40 passed, 247 deselected, 5 warnings in 5.65s`.

Facade/receipt-failure follow-up:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -k 'sync_store_facade_does_not_embed_sql_statements or personal_context_receipt or reenrollment_preserves_personal_context or receipt_lookup_surfaces_storage_failure or personal_context_push_surfaces_receipt_storage_failure'
```

Result: `6 passed, 265 deselected, 5 warnings in 1.85s`.

Targeted modules:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Result: `16 passed, 3 warnings in 0.83s`.

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
```

Result: `86 passed, 42 warnings in 8.54s`.

The direct Store module run reached `183 passed, 3 warnings` and two collection
errors only because the selected `--confcutdir` environment does not provide
the unrelated `pg_database_config` fixture used by two pre-existing PostgreSQL
concurrency tests. The clean non-PostgreSQL slice was `174 passed, 11
deselected, 3 warnings in 2.13s`; the new PostgreSQL receipt behavior is
covered by the executed backend transaction contract above.

Python 3.11 `py_compile` passed for all six changed production/test files;
`git diff --check` passed. No dependency or lock files changed.

## Concurrency remediation: RSA key-record winner and narrow binding merge

Replacement-worker concurrency commit: `491a545f71 fix(sync): serialize personal context bootstrap state`. TASK-13148
remains **In Progress** for controller-owned re-review and closure.

- Concurrent real RSA-OAEP wrappers now handle the durable key-record
  idempotency race. After an insert conflict, bootstrap refetches only the
  deterministic record ID and accepts it only when owner, dataset, device,
  purpose, `wrapped_for`, rewrap state, revocation state, encryption policy,
  integrity-key ID, and current wrapping-key fingerprint all match. The durable
  winner’s randomized ciphertext is returned, so both callers receive a
  decryptable wrapper of the same canonical key; nonmatching rows fail closed
  as key-custody unavailable.
- Personal Context binding now uses a dedicated locked DB mutation rather than
  a stale whole-dataset enrollment rewrite. It validates the pre-read
  profile/authority fence against the locked row, preserves all ordinary
  domains/metadata from that row, merges the required Personal Context domains,
  and writes only domain/metadata/updated-at fields. PostgreSQL uses the
  existing owner-row `FOR UPDATE` helper; SQLite retains transaction
  serialization.

### RED evidence

The new production-RSA race test forces both threads past the empty key-record
read using a barrier before invoking the real RSA-OAEP wrapper:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q -k concurrent_real_rsa
```

Result: `1 failed, 16 deselected`. The losing bootstrap raised
`SyncIdempotencyConflictError: Sync key record ID was reused with different key
material`, because RSA-OAEP generated a different ciphertext for the same
deterministic record ID.

The binding interleaving regression then committed an ordinary metadata/domain
update after `_bind_personal_context_dataset()` had read the dataset and before
the old whole-row update. It failed with `KeyError: 'ordinary_update'`, proving
the stale binding copy erased the concurrent update:

```text
... test_sync_v2_personal_context_bootstrap.py -q -k 'concurrent_real_rsa or binding_preserves_update'
```

Result: `2 failed, 16 deselected`.

### GREEN evidence

The concurrent RSA winner, deterministic ordinary-update interleaving,
profile/authority mismatch, and executed PostgreSQL lock/merge contract pass:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -k 'concurrent_real_rsa or binding_preserves_update or binding_rejects_profile or postgres_personal_context_binding_locks'
```

Result: `4 passed, 201 deselected, 3 warnings in 0.78s`.

Focused module checks:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Result: `19 passed, 3 warnings in 1.00s`.

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
```

Result: `86 passed, 42 warnings in 9.23s`.

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -k 'not postgres'
```

Result: `174 passed, 12 deselected, 3 warnings in 2.96s`. The PostgreSQL
database fixture remains unavailable under this focused `--confcutdir`; the
new PostgreSQL behavior is therefore an executed backend transaction contract,
not a live database claim.

Python 3.11 `py_compile` passed for all changed production/test files and
`git diff --check` passed. No dependency or lock files changed.

## Binding-CAS remediation: complete expected state and durable winner validation

Replacement-worker binding-CAS commit: `c6ae23aab4 fix(sync): fence personal context binding transitions`. TASK-13148
remains **In Progress** for controller-owned re-review and closure.

- The locked Personal Context binding mutation now receives the entire pre-read
  opaque binding (or explicit absence): profile, authority, integrity-key ID,
  purge generation, and link state. It returns idempotently when the locked row
  already equals the desired binding. Otherwise the locked row must exactly
  match the expected binding before transition; any changed key, purge, link
  state, profile, or authority fails as a stale binding and cannot overwrite the
  newer state or its receipt.
- Generic dataset enrollment no longer has a Personal Context preservation
  flag or a `False` escape path. Existing reserved domains/metadata are always
  retained, while the dedicated locked binding method remains the only
  authoritative updater.
- `SyncV2Service.bootstrap_personal_context()` no longer exposes its ignored
  `authority_id` argument; authority remains server-configured.
- The RSA conflict-winner regression suite now parameterizes every semantic
  fence: owner, dataset, device, purpose, wrapped-for mode, rewrap status,
  revocation, encryption policy, integrity-key ID, and current wrapping-key
  fingerprint. Each mismatch fails closed as key custody unavailable without
  returning foreign wrapper ciphertext.

### RED evidence

Before the full-state CAS change, the focused runner reproduced both reviewer
races and the two minor API escapes:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q -k 'concurrent_absent or stale_personal_context_binding or accepts_no_client_authority or no_personal_context_escape'
```

Result: `4 failed, 19 deselected`. Two barrier-synchronized callers that had
both pre-read no binding yielded one success and one
`personal_context_authority_mismatch`. A stale v1/0 caller then rewrote a
committed v2/2 binding, so the expected stale error was not raised. The service
signature still exposed `authority_id`, and generic enrollment still exposed
`preserve_personal_context`.

### GREEN evidence

The SQLite absent-winner race, stale generation/receipt protection, the ten
RSA semantic-mismatch cases, and the two removed API-surface checks pass:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q -k 'concurrent_absent or stale_personal_context_binding or accepts_no_client_authority or no_personal_context_escape or binding_rejects_profile or conflicting_rsa_key_record_winner_mismatch'
```

Result: `15 passed, 18 deselected, 3 warnings in 0.72s`.

The executed PostgreSQL backend contracts cover the original locked merge plus
the new one-update identical-winner path and stale-state rejection:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -k 'postgres_personal_context_binding'
```

Result: `2 passed, 185 deselected, 3 warnings in 0.35s`.

Affected module checks:

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -q
```

Result: `33 passed, 3 warnings in 1.21s`.

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q
```

Result: `86 passed, 42 warnings in 8.79s`.

```text
PYTHONPATH=. UV_CACHE_DIR=/tmp/tldw-pc-sync-uv-cache uv run --offline --no-project --python /Users/macbook-dev/.local/bin/python3.11 --with pytest --with pytest-asyncio --with fastapi --with httpx --with psutil --with loguru --with pydantic-settings --with aiosqlite --with cryptography --with rfc8785 --with pyyaml --with chardet --with defusedxml --with email-validator --with python-dotenv --with rich --with asyncpg --with cachetools --with sqlglot --with 'python-jose[cryptography]' --with apscheduler --with redis --with argon2-cffi pytest --confcutdir=tldw_Server_API/tests/Sync tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -k 'not postgres'
```

Result: `174 passed, 12 deselected, 3 warnings in 2.48s`. Live PostgreSQL
remains unavailable under the focused configuration; its behavior is represented
by executed backend transaction contracts rather than a live-database claim.

Python 3.11 compilation and `git diff --check` passed. No dependency or lock
files changed.

## Post-closure contract correction: non-mutating review and exact attention

Two cross-repository contract gaps were corrected after the original closure:

- An absent-profile bootstrap now reserves only a random profile identity and
  wrapped server key custody. The returned manifest/global-scope objects are a
  deterministic transient review snapshot; no canonical object version/head is
  persisted until the authenticated device submits the exact reviewed cursor to
  `/api/v1/sync/personal-context/complete`. Completion materializes the exact
  planned profile/scope IDs, version IDs, timestamps, purge generation, and
  cursor before writing the existing device receipt. Repeated plans, concurrent
  plans, concurrent completions, cancellation-by-omission, explicit server-side
  creation after cancellation, and already-linked profiles are idempotent.
- Schema, quota, and purge-generation incompatibilities retain their existing
  HTTP 409 statuses and stable reason codes, but `detail.attention` is now a
  discriminated content-free object. It reports exact required/server schema
  bounds, required/available/insufficient quotas, or expected/current purge
  generations. No canonical manifest, scope, record, proposal, wrapped key, or
  ciphertext is included in these errors.

RED evidence: seven selected service/bootstrap/endpoint tests failed against
the prior implementation because planning persisted the profile, no planning or
post-review materialization boundary existed, and all three errors lacked
attention facts.

GREEN evidence after the correction:

- `163 passed, 42 warnings` across the Personal Context repository/service,
  bootstrap, and authenticated Sync endpoint modules.
- Ruff reported `All checks passed` for every touched Python file.
- Bandit exited 0 for all touched production modules; Python 3.11 compilation
  and `git diff --check` exited 0.

ADR-002 still governs this correction: content-free reservation stays outside
canonical Personal Context content authority, and only the Personalization
service materializes reviewed canonical objects. No new ADR was required. The
full repository suite and a live PostgreSQL run were not performed; neither is
needed for the SQLite-only Personalization reservation transaction, while the
unchanged Sync binding/receipt PostgreSQL contracts retain their prior evidence.

## Final transport-watermark and successful-quota correction

Final server contract commit:
`6455ab08cb12ec239c53b7b9180b1cc1ea5f8375`.

Successful bootstrap now includes `sync_transport_cursor`, a separate signed
private-pull token. The existing `cursor` remains the semantic canonical review,
receipt, and completion identity and is not accepted by private pull. The new
token is bound to the authenticated dataset, registered device, negotiated
adapter version set, and every Personal Context domain/version stream.

Bootstrap enrolls only content-free Personal Context transport domain control
state before taking the boundary. It then holds the Sync dataset-row lock while
capturing the accepted per-stream watermarks and reading the canonical snapshot.
This is an ordering fence, not a cross-database transaction: the required
application ordering is canonical commit followed by Sync envelope append, and
relevant append paths take that same dataset-row lock. Consequently an append
that starts after boundary capture receives a greater sequence and is delivered
for reconciliation. The durable device key record associates the boundary with
the exact semantic cursor, so retry/reissue of the same reviewed plan may refresh
signing timestamps but cannot advance watermarks. The signed cursor lives for 30
days with five minutes of bounded clock skew.

Successful `quotas` now includes every syntactically valid requested unknown
zero-minimum quota at effective value `0`. A positive unknown requirement still
returns the typed content-free quota incompatibility with available value `0`.

TDD RED initially showed the zero quota omitted and both core/HTTP bootstrap
responses lacking the transport cursor. GREEN evidence includes `41 passed` for
the complete bootstrap module and `93 passed` for the complete authenticated
Sync endpoint module. Added regressions cover retained multi-revision history,
post-boundary delivery, unsigned semantic-cursor rejection, narrowed-scope
rejection, 29-day review, expiry/skew, stable retry watermarks, deterministic
SQLite capture/append interleaving, and an executed PostgreSQL dataset-lock
contract. Live PostgreSQL remains unavailable and is not claimed.
