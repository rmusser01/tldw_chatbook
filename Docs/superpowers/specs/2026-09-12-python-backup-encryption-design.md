# Python backup encryption correction

User decision, 2026-09-12: Go was an oversight in the original backup plan and must
be removed. This correction supersedes the Go implementation/build/delivery portions
of the revision-4 design and ADR-126. The backup feature scope and archive format
remain unchanged. Tracked by TASK-32561; Linux baseline testing is TASK-32560.

## Required behavior

Use the existing `pycryptodomex` project dependency through its Python APIs for
scrypt, HKDF/HMAC-SHA256 and ChaCha20-Poly1305. Implement only the existing age v1
single-scrypt passphrase envelope, as specified by C2SP; add no recipients, plugins,
new archive format or encryption algorithms. Retain `.tldw-backup.zip.age` and the
existing `crypto.transform` signature. Prior encrypted archives and rollback copies
must remain readable.

Retain a Python subprocess for transforms so cancellation can terminate a running
KDF and reap all pipe workers. Execute a fixed package-owned Python entry using the
current interpreter in isolated mode. Passwords remain length-prefixed bytes on
stdin, never command arguments, environment variables or files. The worker imports
no application configuration, owner services or network clients. No executable
lookup, Go/Rust toolchain, downloaded worker, or new native executable is introduced.
PyCryptodome's existing native implementation is an ordinary project dependency.

Keep the existing 1–4096-byte password range, one serialized KDF per application,
scrypt work factor18 for writes (256 MiB working memory), factors1–18 for reads,
64 KiB outer-header budget, and independent 2 TiB input/output stream budgets.
Parse the complete canonical single-scrypt header before KDF allocation. Authenticate
the header and every payload chunk, including the final-chunk marker; reject wrong
passwords, corruption, truncation, trailing bytes, unsupported recipients and
excessive KDF work. Data must stream in bounded chunks, including short pipe reads
and writes. Parent output remains private and is published without replacement only
after authenticated EOF and successful worker completion. Preserve cancellation,
space checks, input digest checks and cleanup behavior.

## Delivery and availability

Package the Python worker as ordinary application source in wheels and sdists.
Remove Go helper sources/build hooks/target selection/manifests/notices and Go CI
steps that belong exclusively to this backup implementation. Keep third-party
notices only for code actually distributed. Existing native filesystem evidence is
separate and must remain enforced; Python encryption availability does not qualify
Linux publication, admission, capture or replacement.

Keep existing callers working while replacing Go-specific capability assumptions.
The internal `helper_capability()` name may remain as a compatibility entry point
returning the existing `(bool, sanitized_reason)` shape; it now checks the packaged
Python worker/dependency/protocol. Source/editable installs can supply Python
encryption without a prebuilt architecture-specific binary. Do not claim that old
Go receipts qualify the new worker. Final product evidence must exercise the new
Python backend and source/package provenance.

Preserve worker-byte integrity verification: the parent checks the fixed Python
worker's SHA-256 against a parent-owned constant before execution. A worker change
updates that constant in the same reviewed change. This retains the previous
tamper/corruption refusal without introducing a replacement delivery manifest or
platform inventory. The application package remains the trust boundary.

## Verification

Use fixed existing age ciphertext and expected plaintext/password fixtures for
independent compatibility; no Go toolchain is required to run tests. Preserve
behavioral coverage of the current crypto tests while replacing implementation-
specific helper assertions. Add chunk-boundary, canonical-header, invalid-tag,
final-chunk, binary-password and short-I/O cases using established primitives.
Build/install wheel, sdist and editable forms without Go on PATH and exercise actual
encryption/decryption. Retain installed-file preservation assertions. Run the
affected archive/credential/recovery/F9 checks against the Python backend on macOS,
and targeted crypto/package checks on the provided Linux machine. Report Linux
filesystem failures separately; do not change those primitives in this correction.

Sources: [age v1 specification](https://github.com/C2SP/C2SP/blob/main/age.md),
[PyCryptodome KDF API](https://pycryptodome.readthedocs.io/en/latest/src/protocol/kdf.html),
[ChaCha20-Poly1305 API](https://pycryptodome.readthedocs.io/en/latest/src/cipher/chacha20_poly1305.html).
