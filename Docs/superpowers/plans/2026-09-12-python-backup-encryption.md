# Python backup encryption implementation plan

> Use subagent-driven-development for the bounded implementation units and their
> reviews. The user's explicit removal of Go supersedes the original Go plan.

Goal: replace the backup Go implementation with Python while preserving encrypted
archive compatibility and the existing backup/recovery behavior.

Spec: ../specs/2026-09-12-python-backup-encryption-design.md. TASK-32561.
Architecture: a package-owned Python streaming worker using existing PyCryptodome
primitives; the current parent transport retains cancellation and atomic output.
Tech stack: Python>=3.11, existing pycryptodomex, stdlib pipes/files, pytest.

ADR required: no new ADR; existing ADR-126 applies with the user's Python correction.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: the user superseded its language/delivery choice; the archive format,
process boundary and recovery contracts remain intact. The linked correction spec
records that decision.

## Global constraints

- Preserve `crypto.transform(source, target, *, password, decrypt, cancel,
  input_limit, output_limit, space_check, expected_input_sha256)` and age v1 files.
- No Go/Rust toolchain or separate compiled helper; no new crypto dependency.
- Password1–4096 raw bytes; header64KiB; independent input/output2TiB;
  write scrypt18/read factors1–18; one KDF/application; bounded64KiB chunks.
- No extra recipients/plugins/features, automatic plaintext fallback, or platform
  qualification widening. No real user data/keychain/model/network access in tests.
- Targeted tests only; preserve failed receipts; do not conceal platform failures.

## Task 1: Python streaming worker

Goal: independent compatible worker, before changing application transport.
Success criteria: decrypt fixed prior age archives; produce canonical authenticated
files; reject malformed input before expensive work; bounded binary stream I/O.
Tests: new `Tests/Backup_Recovery/test_python_age.py` and fixed age fixtures.
Status: Complete. Worker63 tests pass; independent review approved after closed-pipe
shutdown regression correction. Evidence: /private/tmp/task32495-python-worker-report.md
and /private/tmp/task32495-python-worker-review.md.

Create `tldw_chatbook/Backup_Recovery/age_worker.py`, a directly executable isolated
Python file with no application imports. Preserve stdin four-byte big-endian
password length then password then container bytes, with modes info/encrypt/decrypt.
Info reports protocol2, implementation python, format age-v1. Primitive keys/nonce
sizes and STREAM final-block semantics follow the linked age specification.
Use PyCryptodome primitives, canonical base64/header parsing, exact bounded reads and
writes, independent input/output counters and fixed sanitized failure codes.
Verify fixture interoperability, plaintext sizes0/1/65535/65536/65537/multiple chunks,
short reads/writes, binary password, invalid header/MAC/tag/truncation/trailing data
and excessive KDF factor. Establish behavioral RED before implementation, then GREEN.

## Task 2: Parent transport and behavioral tests

Goal: existing public transform uses the Python worker without Go resources.
Success criteria: original cancellation/backpressure/limits/publication semantics
pass through the actual Python child; callers use unchanged API shapes.
Tests: `test_crypto.py`, worker tests and directly affected capability tests.
Status: Complete. Crypto42 tests and directly affected release gate pass; independent
review approved. Evidence: /private/tmp/task32495-python-transport-report.md and
/private/tmp/task32495-python-transport-review.md.

Modify `crypto.py` to execute `[sys.executable, '-I', worker_path, mode]`; replace the
Go tuple/binary manifest check with fixed Python worker/dependency/protocol checks.
Retain source/destination checks, pipe cleanup, digest checks and the serialized
lock. Adapt `Tests/Backup_Recovery/conftest.py` and `test_crypto.py` to the Python
worker and fixed independent interoperability fixtures. Do not manufacture packaged
product qualification through helper overrides. Review and run scoped Bandit/Ruff.

## Task 3: Python-only delivery and fixture integration

Goal: source, editable, sdist and wheel encryption work without a Go installation.
Success criteria: built packages contain the Python worker; no backup Go executable,
Go build hook, helper target selection or Go-specific runtime manifest remains.
Tests: `Tests/Packaging/test_backup_helper_distribution.py`, existing installed
package fixture consumers and packaging checker cases.
Status: Complete. Packaging11 tests pass; scoped review approved after correcting
fresh-cache provisioning and Windows uv discovery. Evidence:
/private/tmp/task32495-python-packaging-report.md and
/private/tmp/task32495-python-packaging-review.md.

Remove exclusively Go-owned `Packaging/backup_age` build assets, setuptools cmdclass
entries and Go-only manifest declarations. Update `Packaging/check_manifest.py`,
distribution tests and `Tests/Backup_Recovery/native_package.py` to ordinary Python
wheels with preserved installation hashes. Update the two existing qualification
workflows and relevant packaging/user docs for the Python backend. Preserve native
filesystem evidence and exact platform gates. No workflow dispatch or new runners.

## Task 4: Targeted macOS/Linux verification and PR update

Goal: demonstrate the final Python code and accurately document remaining limits.
Success criteria: compatibility and package tests pass without Go; actual final
macOS encrypted F9/rollback paths use Python; Linux crypto outcomes are recorded.
Tests: final changed crypto/package modules, affected release gates/archive paths,
three F9 modes, combined replacement/later rollback; Linux targeted crypto/package.
Status: Complete. All three installed F9 modes and combined replacement/later
rollback passed. Linux worker/crypto/packaging passed 116 tests without skips.
Final review approved; known baseline failure and Linux filesystem limitations
recorded in Docs/Development/backup-python-verification-2026-09-12.md and PR2642.

Run serial native tests in private locations with NullKeyring/network guard.
Record exact source/package identities; distinguish test fixture faults from native
Linux publication/F_FULLFSYNC gaps. Independent review of final change and evidence;
scoped production Bandit and static checks. Update TASK-32561, TASK-32560 and PR2642
with the final implementation and actual results. Commit only task-owned files.
