# Recovery age helper packaging and qualification

This directory implements protocol v1 under [ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md).
Source, editable, and `py3-none-any` installations explicitly report
`helper_unavailable`. A release maintainer may build a native helper wheel only for a
cell marked `qualified` in `qualification.json`. Do not copy a PATH executable into
place or fall back to plaintext.

## Protocol and resource contract

The executable accepts exactly one argument: `info`, `encrypt`, or `decrypt`.
`info` needs no input/password and emits at most 1 KiB of nonsecret JSON:
`protocol` (integer 1), `helper_version` (`"1"`), `age_version` (`"v1.3.2"`),
`os` (Go GOOS), and `arch` (Go GOARCH).

For transforms, stdin contains a four-byte big-endian password length, 1–4096
password bytes, then the input byte stream. Only transformed data goes to stdout.
Errors on stderr are fixed codes: `protocol_error`, `invalid_header`, `byte_limit`,
`transform_failed`, or `helper_failed`. Secrets are never arguments, environment
values, or request files. There is no network/plugin/recipient-extension role.

The writer uses the official age v1 single-scrypt passphrase envelope with work
factor 18. Before decryption, a 64 KiB header gate admits exactly the canonical
single scrypt stanza, validates salt/body/MAC encoding, and refuses work factors
outside 1–18 before derivation. The identity also caps work factor at 18. That
factor uses 256 MiB of scrypt working memory; runtime overhead is additional.
Both helper input and output independently count actual streamed bytes against
2 TiB. Input length excludes the private password prefix. Authentication must
succeed through EOF; encryption success requires successful writer finalization.

Python resolves only its package's fixed `Backup_Recovery/_age/backup-age` path
(`backup-age.exe` on Windows). The adjacent package-owned
`Backup_Recovery/helper_manifest.json` inventories every candidate tuple and is the
single authoritative installed digest/identity contract. Native wheel construction
changes only its selected tuple from `unavailable` to `qualified` and copies the
matching binary and notices into `_age`. Manifest data is strict, bounded to 16 KiB,
and rejects unknown fields. The binary must be a regular executable of at most
32 MiB with matching platform, digest, permission, and runtime `info`. No manifest
value selects an executable path. Package installation is the trust boundary; this
is integrity verification, not protection against a compromised local installer.

Python serializes transforms per application process. Three bounded pipe workers
avoid stdin/stdout/stderr backpressure, with a minimal child environment. Cancel
kills/reaps the child and joins its pipe workers, including while waiting for the
job lock. Targets are new files: private mode-0600 temporary output is published
with an atomic no-replacement link only after clean authenticated completion.
Existing/racing destinations survive. Failure removes only the operation's temp
file. A cleanup I/O failure reports `cleanup_failed` and releases the job lock;
its private leftover may require owner cleanup. No secure-erasure claim is made.
Decrypt callers must choose an owner-private staging directory and must not parse
ZIP until `transform` returns successfully. Caller-owned paths pass the shared
path validator; regular input is opened without following the final symlink on
qualified POSIX hosts.

## Qualification and dependency ownership

Verified upstream pin: `filippo.io/age v1.3.2`, official tag commit
`b74dce4cdbe35b5e5f66c06d9612b72f89028758`.
Module checksum: `h1:r6RSZLFSMm6rzKepZ7ZAYkKCu14f3/Me8c7uKYh7C8c=`.
Go module checksum: `h1:TH/Yr2sSRhCKbaH4XPxpUV0Us8Gv6txYUpiZQWz8Evk=`.
The checksum service verified these during qualification; `go.sum` is committed.

Native package qualification passed on macOS arm64, macOS 26.5.2 (25F84), APFS,
Python 3.12.11, and Go 1.26.2 with
`GOTOOLCHAIN=local`. Upstream declares Go 1.25.0 and a Go 1.27.0 release-toolchain
preference; both this helper and the independent official CLI compiled and ran
using the explicitly selected Go 1.26.2. This module requires Go 1.26.0. This is
the only advertised native cell. Python 3.11 and 3.13 on this tuple, and every
darwin/amd64, linux, and Windows cell, remain explicitly unavailable until actual
native evidence is recorded. The macOS wheel targets macOS 12.0 or newer, matching
its recorded `LC_BUILD_VERSION`.

The official CLI is a pinned qualification-only Go `tool` dependency; it is not
shipped or invoked by production. Tests compile both executables into private
temporary directories. The application never builds/downloads one during backup.
Contributors must explicitly provision Go and the pinned module cache; missing
required tools fail tests rather than skip them.

To exercise an editable/source checkout, provision the pinned modules first and then
invoke the build API explicitly. This writes only the requested contributor artifact;
it does not make the checkout advertise installed encryption capability:

```sh
export GOTOOLCHAIN=local
export GOMODCACHE=/private/path/to/go-mod-cache
export GOCACHE=/private/path/to/go-build-cache
export GOPATH=/private/path/to/go-path
go -C Packaging/backup_age mod download all
python -c 'from pathlib import Path; from Packaging.backup_age.build_helper import build_helper; build_helper("darwin", "arm64", Path("/private/tmp/backup-age"))'
/private/tmp/backup-age info
```

Build distributions only from a clean committed tree so generated `build/lib` copies
cannot enter repository-wide source scans. `Packaging/build_dist.sh` archives `HEAD`
into an external temporary tree. Omit the target for a source distribution plus a
pure wheel. Select the one currently qualified native wheel explicitly:

```sh
Packaging/build_dist.sh
TLDW_BACKUP_HELPER_TARGET=darwin/arm64 Packaging/build_dist.sh
```

The native wheel build is offline with respect to Go dependencies (`GOPROXY=off`,
`GOSUMDB=off`, pinned module graph, `-mod=readonly`) and fails if its reproducible
helper digest differs from `qualification.json`. Runtime never finds, downloads, or
builds a helper. Qualification installed the wheel into a fresh environment with Go
absent from `PATH`; the application and native child ran under inherited macOS
process sandbox network denial. A real network probe failed with `EPERM` before the
round trip and two-direction source/package interoperability checks ran.

From the repository root, with a development Python environment and Go on PATH:

```sh
export GOTOOLCHAIN=local
python -m pytest Tests/Backup_Recovery/test_crypto.py -q
go -C Packaging/backup_age test ./...
go -C Packaging/backup_age vet ./...
gofmt -l Packaging/backup_age
go -C Packaging/backup_age mod verify
```

Use private `GOCACHE`, `GOMODCACHE`, and `GOPATH` directories when qualifying in an
isolated worktree. The current PTY/permission/RSS qualification tests target POSIX;
Windows pipe and private-file delivery needs independent qualification.

Selected module graph (`go list -m all`):

```text
c2sp.org/CCTV/age v0.0.0-20260829155415-4448f2097b2d
filippo.io/age v1.3.2
filippo.io/edwards25519 v1.2.0
filippo.io/hpke v0.4.0
filippo.io/nistec v0.0.4
github.com/rogpeppe/go-internal v1.16.0
golang.org/x/crypto v0.55.0
golang.org/x/net v0.57.0
golang.org/x/sys v0.47.0
golang.org/x/term v0.45.0
golang.org/x/text v0.41.0
golang.org/x/tools v0.49.0
```

The linked helper dependency packages on this host are age, hpke, and x/crypto,
plus the Go standard library. Their inspected upstream LICENSE files use the
BSD 3-Clause license; the Go toolchain/runtime does too. The module graph also
contains qualification-only CLI and upstream-test dependencies. Binary delivery
must carry applicable notices and recheck the linked graph for every release
platform. Release maintainers own pin changes, reproducible packaging, integrity
metadata, license notices, and rerunning both-direction interoperability tests.
