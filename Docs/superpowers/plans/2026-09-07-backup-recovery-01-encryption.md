# Encryption helper delivery Implementation Plan

2026-09-12 correction: this plan's Go implementation and delivery steps are
historical. Follow the user-requested [Python encryption replacement](2026-09-12-python-backup-encryption.md)
for the current backend; archive format and backup scope are unchanged.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An independently qualified, packaged age helper with bounded secret transport.

**Architecture:** Implement the approved Backup_Recovery service through existing storage owners, with fixed startup admission and private immutable staging. Keep archive parsing separate from normal application startup, and expose only qualified operations through thin user views.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite/FTS5, Pydantic, stdlib ZIP64/file primitives, and a bundled helper built from official age Go library.

**Spec:** [Approved complete local backup and restore](../specs/2026-09-07-complete-local-backup-restore-design.md), revision 4.

## Global Constraints

- Python ≥3.11; retain the repository's current Textual 8.x dependency contract.
- “All identified Chatbook-owned local profiles and durable data are included by default, including configured database locations outside the default directory.”
- “External folders and model files are explicit options. Server-owned data has a separate recovery boundary and is not captured by this feature.”
- “Managed credentials are excluded by default. Including exportable credentials requires a password-encrypted archive.”
- “Partial archives support extraction and isolated recovery of validated dependency groups; they cannot replace current installation data in v1.”
- “Use .tldw-backup.zip for plaintext and .tldw-backup.zip.age for encrypted output.”
- “v1 reader defaults: 100,000 members; 16 MiB manifest; 1 TiB expanded payload; 256 GiB per member; 1,024 UTF-8 bytes per path.”
- “Default encrypted/plain input-container and decrypted-container budgets are each 2 TiB, independently enforced as actual bytes stream”; outer header 64 KiB; one KDF with 256 MiB working-memory budget.
- “No rebuild starts automatically, including a local rebuild.” Restored execution and reconnection require durable local owner review.
- “Run targeted checks only; a full suite needs separate user authorization.” Runtime tests listed here are required evidence, not results already obtained.
- “Do not expose Complete backup or destructive replacement before its full contract is qualified.” Keep per-operation/platform capability failures visible.
- ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: direct implementation of the approved storage, credential, archive, and recovery contract; reuse ADR-126 with ADR-029/030/036/059/060.

---

[Delivery order and cross-plan contracts](2026-09-07-complete-local-backup-restore.md).

## Execution discipline

Read the approved spec and the matching Backlog task before implementation. Move only
the current task to In Progress, then copy its steps into that task's Implementation
Plan before touching production code. Tasks still To Do have design references and
acceptance criteria, not premature implementation notes. Use an isolated worktree at
execution time; this planning checkout contains unrelated staged/unstaged work.
Activate the project Python ≥3.11 environment before the commands below, for example
`source .venv/bin/activate`; the system `python3` on this host is older and must not
be used for application-test evidence.

The interface code fences below specify signatures, not stub implementations to ship.
Invariant fragments belong inside the named implementation and do not replace the
behavioral steps. All Python types use stdlib dataclasses/pathlib/threading/typing;
untrusted serialized data is validated with strict Pydantic models at the boundary.
`StorageItem`, `Inventory`, `SchemaPolicy`, and `OwnerAdapter` are defined in task 3;
`ArchiveLimits`/`SealedArchive` in task 12; `CaptureResult` in task 15;
`RestorePlan` in task 17; `Journal` in task 18. Import these definitions, do not
create lookalike cross-module types. Relative file paths below are repository-relative.

Capture options are validated at the service boundary: `external_roots: tuple[Path, ...]`,
`model_ids: tuple[str, ...]`, `temporary_media: bool`, `diagnostics: bool`,
`credential_mode: Literal["exclude", "include", "rollback"]`, `encrypted: bool`,
`allow_partial: bool`, and `limits: ArchiveLimits`. Unknown keys are rejected. Missing
optional collections/booleans mean empty/false; credentials default to exclude.
The backup UI cannot select rollback mode; only the local replacement executor can.
Inventory status vocabulary is included, included_directory, intentionally_excluded,
unused, intentionally_deleted, unavailable, unsupported, and missing_required. Only
validated owner tombstones may produce intentionally_deleted.

Frozen tuples carry snapshot data. A scope digest includes selected profile/config
selectors, owner/root identities, shared groups, dependencies, exclusions, and budgets;
it excludes ordinary row counts and changing record contents. Capture payload digests
and target fingerprints serve separate purposes and must not be substituted for it.
String status/error codes in tests are fixed sanitized codes, never raw exception text.
Owner-private method names in invariant fragments are local implementation details,
not undeclared public services. Define and test them in the same task if retained.

Every test example is the first focused red case, followed by the listed behavioral
matrix. Import failure may start a new module's TDD loop, but establish a behavioral
red failure after skeleton importability before claiming regression evidence. Use
real temporary SQLite/files/processes, owner APIs, and subprocess synchronization;
mock only external keyrings/network/process-effect sentinels where explicitly stated.
All app-importing tests live under Tests/ and its isolation fixture. Never read the
developer's real config, keychain, models, or databases to construct fixtures.

Run the named test file after each small behavior change, then the task's listed
guards once. Run changed Python through `ruff check --select E9,F63,F7,F82` and
new focused modules through `ruff format --check`; use `gofmt -l` and `go vet` for
Go changes. Provision these development tools in the isolated execution environment
if absent; do not reformat large unrelated existing modules. Record baseline lint
failures separately with a clean-HEAD reproduction. Run `git diff --check` before
the scoped commit. New SQLite/private writers update their existing inventories;
diagnostic changes run the production diagnostic inventory guard. No full suite or
collection sweep is implied by these targeted commands.

For each task: review the diff, add implementation notes with the ADR and actual
commands/outcomes, update documentation, then check criteria and mark Done only
after evidence passes. The five-digit Backlog CLI bug is documented in
`backlog/docs/lessons-backlog-hygiene.md`; verify the resulting file, use its documented
direct-file fallback if necessary, and never let a malformed CLI file reach a commit.
Commit only that task's changed files with the provided subject. Existing staged
changes in another checkout are not part of this work.

## File structure

- Task 1 owns `Packaging/backup_age/go.mod`, `Packaging/backup_age/go.sum`, `Packaging/backup_age/main.go`, `Packaging/backup_age/main_test.go`, `tldw_chatbook/Backup_Recovery/__init__.py`, `tldw_chatbook/Backup_Recovery/crypto.py`.
- Task 2 owns `Packaging/backup_age/build_helper.py`, `Packaging/backup_age/wheel_commands.py`, `Packaging/backup_age/qualification.json`, `Packaging/backup_age/README.md`, `tldw_chatbook/Backup_Recovery/helper_manifest.json`, `.github/workflows/backup-helper-qualification.yml`.

<a id="task-1"></a>
## Task 1: Qualify bounded age helper protocol

**Backlog:** [TASK-31984](../../../backlog/tasks/task-31984%20-%20Qualify-bounded-age-helper-protocol.md) — To Do.

**Dependencies:** Approved design only.

**Files and ownership:**

- Create: `Packaging/backup_age/go.mod`
- Create: `Packaging/backup_age/go.sum`
- Create: `Packaging/backup_age/main.go`
- Create: `Packaging/backup_age/main_test.go`
- Create: `tldw_chatbook/Backup_Recovery/__init__.py`
- Create: `tldw_chatbook/Backup_Recovery/crypto.py`
- Create: `Tests/Backup_Recovery/test_crypto.py`
- Create: `Tests/Backup_Recovery/conftest.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# crypto.py: secret-bearing arguments are never included in repr/errors/logging.
def transform(source: Path, target: Path, *, password: bytes,
              decrypt: bool, cancel: Event) -> None: ...
def helper_capability() -> tuple[bool, str]: ...
def _package_resource_root() -> Path: ...
```

- [x] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_crypto.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_encrypted_stream_round_trip(tmp_path, helper_resource_root, monkeypatch):
    from threading import Event
    from tldw_chatbook.Backup_Recovery.crypto import transform
    from tldw_chatbook.Backup_Recovery import crypto
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source, encrypted, restored = (tmp_path / n for n in ("in", "sealed", "out"))
    source.write_bytes(b"synthetic recovery bytes" * 10000)
    transform(source, encrypted, password=b"test-only passphrase", decrypt=False, cancel=Event())
    transform(encrypted, restored, password=b"test-only passphrase", decrypt=True, cancel=Event())
    assert restored.read_bytes() == source.read_bytes()
    assert b"synthetic recovery bytes" not in encrypted.read_bytes()
```

- [x] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_crypto.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [x] **Step 3:** Pin filippo.io/age v1.3.2 for the qualification build and commit go.sum; verify the upstream tag/checksums and build-tool compatibility before accepting the pin. Record the actually tested Go toolchain, licenses, and module graph. Do not use main/latest during release builds.

- [x] **Step 4:** Define _package_resource_root() -> Path in crypto.py and a helper_resource_root pytest fixture in Tests/Backup_Recovery/conftest.py. The fixture builds this task's real Go helper with subprocess.run(["go", "build", "-o", str(binary), "."], cwd=helper_source, check=True), writes its protocol/version/platform/digest manifest into a private temporary resource root, and returns that Path. It needs the qualification Go toolchain and does not skip missing required tools. Monkeypatch only resource resolution, never encryption. Production resolution remains package-owned and unavailable when unbuilt; the next task qualifies actual distribution delivery.

- [x] **Step 5:** Define helper protocol v1: argv contains only encrypt/decrypt/info; stdin is a four-byte big-endian password length, 1–4096 password bytes, then streamed input. Password length is validated before allocation. stdout contains only transformed bytes; stderr contains bounded fixed error codes, never underlying error text or input. No files or executable paths supplied by an archive are opened by the helper.

- [x] **Step 6:** Use age.NewScryptRecipient with SetWorkFactor(18), and NewScryptIdentity with SetMaxWorkFactor(18). A bounded 64 KiB header gate admits only a single scrypt stanza before age performs derivation. Drain the authenticated reader through EOF before success; close the encrypting writer and check its error. Serialize helper jobs so only one derivation runs.

- [x] **Step 7:** Implement concurrent stdin pumping and stdout draining with bounded buffers to avoid pipe deadlock; consume bounded stderr concurrently. On cancellation kill and reap the child, remove only operation-owned unpublished output, and return a sanitized cancellation result. Reject empty passwords and prohibit argv/environment/password-file fallbacks.

- [x] **Step 8:** Add wrong-password, truncated-final-chunk, excessive work factor, malformed/oversized header, multi-recipient, large-stream cancellation, child crash, and pipe-backpressure tests. Inspect argv/environment/error output for synthetic secret sentinels. Interoperate in both directions with a separately built official age command.

**Implementation invariant:** preserve this control flow while implementing the steps.

```go
// In the encrypt/decrypt branches after bounded header admission:
recipient, err := age.NewScryptRecipient(password)
if err != nil { return err }
recipient.SetWorkFactor(18)
identity, err := age.NewScryptIdentity(password)
if err != nil { return err }
identity.SetMaxWorkFactor(18)
// Map returned errors to fixed codes at main; never print err itself.
```

- [x] **Step 9:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_crypto.py -q
go -C Packaging/backup_age test ./...
go -C Packaging/backup_age vet ./...
gofmt -l Packaging/backup_age
```

- [x] **Step 10:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [x] **Step 11:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): qualify bounded age helper protocol`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Streaming encrypted round trips interoperate with official age and reject incomplete authentication.
- Header, KDF, memory, password transport, cancellation, and child cleanup limits are demonstrated with synthetic data.
- Unqualified or absent helpers report unavailable before password collection or maintenance; no plaintext fallback occurs.


<a id="task-2"></a>
## Task 2: Package and qualify the backup encryption helper

**Backlog:** [TASK-31985](../../../backlog/tasks/task-31985%20-%20Package-and-qualify-the-backup-encryption-helper.md) — To Do.

**Dependencies:** TASK-31984.

**Files and ownership:**

- Create: `Packaging/backup_age/build_helper.py`
- Create: `Packaging/backup_age/wheel_commands.py`
- Create: `Packaging/backup_age/qualification.json`
- Create: `Packaging/backup_age/README.md`
- Create: `tldw_chatbook/Backup_Recovery/helper_manifest.json`
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `Packaging/build_dist.sh`
- Modify: `Packaging/check_manifest.py`
- Create: `Tests/Packaging/test_backup_helper_distribution.py`
- Create: `.github/workflows/backup-helper-qualification.yml`
- Modify: `tldw_chatbook/Backup_Recovery/crypto.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# build_helper.py: builds only from the pinned module graph.
def build_helper(goos: str, goarch: str, destination: Path) -> Path: ...
# crypto.py: resolves only package resources covered by helper_manifest.json.
def helper_capability() -> tuple[bool, str]: ...
```

- [x] **Step 1:** Add this first regression to `Tests/Packaging/test_backup_helper_distribution.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_checkout_without_helper_is_unavailable(monkeypatch, tmp_path):
    from tldw_chatbook.Backup_Recovery import crypto
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: tmp_path)
    available, reason = crypto.helper_capability()
    assert available is False
    assert reason == "helper_unavailable"
```

- [x] **Step 2:** Run `python -m pytest Tests/Packaging/test_backup_helper_distribution.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [x] **Step 3:** Use the existing internal _package_resource_root() -> Path resolver from crypto.py; verify the installed resource path, expected digest, executable type, protocol version, and platform tuple before invocation. Never search PATH or download at runtime.

- [x] **Step 4:** Build candidate tuples darwin/arm64, darwin/amd64, linux/amd64, linux/arm64, windows/amd64. These are qualification targets, not advertised support. Native runners must record OS minimum, filesystem, Python 3.11/3.12/3.13, pipe behavior, signature/package trust, and helper integrity results; unsupported tuples remain unavailable.

- [x] **Step 5:** Use platform-tagged wheels containing only the corresponding helper and its license/version/digest inventory; a native binary must not enter a py3-none-any wheel. Extend the existing setuptools build command and package-data inventory explicitly. Source distributions include helper source and pinned module files. Document explicit contributor Go build for editable/source checkouts; missing binaries do not trigger automatic compilation.

- [x] **Step 6:** Build a wheel and install it into an isolated environment with Go absent, network disabled during runtime, and an empty PATH except Python requirements. Run the real helper round trip, malformed digest/version, missing helper, and upgrade interoperability cases. Test the sdist and documented editable workflow separately.

- [x] **Step 7:** Record reproducible build inputs and output digests in qualification.json, with packaging ownership in Packaging/backup_age/README.md. A failed platform gate stays unadvertised; an integration-wide failure requires an ADR amendment before encrypted replacement work proceeds.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Inside helper_capability(), using the package-owned manifest entry:
if not executable.is_file():
    return False, "helper_unavailable"
if sha256(executable.read_bytes()).hexdigest() != expected_digest:
    return False, "helper_integrity_mismatch"
# Identity/permission/version/platform checks also precede execution.
```

- [x] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Packaging/test_backup_helper_distribution.py -q
git diff --check
```

- [x] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [x] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): package and qualify the backup encryption helper`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Qualified wheels work without Go, runtime downloads, or PATH helper substitution.
- Source/editable installation and unsupported-platform behavior are explicit and tested.
- Native platform evidence, pinned dependencies, integrity/version checks, and upgrade interoperability accompany each advertised tuple.

## Upstream qualification inputs

The starting pin is age v1.3.2. Its tagged module declares Go 1.25.0 and a release
toolchain directive; record the compatible toolchain actually tested, do not infer
availability from a source declaration. Its scrypt API permits explicit work-factor
limits, which the helper must set before processing untrusted input.
Sources: [tagged module](https://raw.githubusercontent.com/FiloSottile/age/v1.3.2/go.mod),
[tagged scrypt implementation](https://raw.githubusercontent.com/FiloSottile/age/v1.3.2/scrypt.go),
[official API](https://pkg.go.dev/filippo.io/age).
