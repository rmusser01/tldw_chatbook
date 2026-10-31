# ADR-254: Windows portable WezTerm spike

Status: Accepted for an explicitly requested proof of concept only
Date: 2026-10-28
Related task: TASK-37205

## Decision

Build an experimental Windows x64 ZIP containing an official, checksum-pinned
WezTerm portable release, CPython's embeddable distribution, and an installed
Chatbook wheel with its dependencies. A small standard-library launcher starts
the existing CLI in a dedicated WezTerm process. No Textual UI or terminal engine
is rewritten. This is a spike, not a supported or signed release channel.

All default application state belongs to a sibling `Data` directory. The
launcher derives paths from its own file, supplies a process-local HOME,
USERPROFILE, APPDATA, LOCALAPPDATA, cache and temporary directories, selects a
private `TLDW_CONFIG_PATH`, and retains existing profile defaults beneath that
portable home. No absolute bundle path is written into a seed configuration.
It drops inherited Chatbook profile selectors and Python import overrides.
The parent environment and installed Chatbook profile are not modified.
The launcher disables OS keyring integration for this independent profile;
portable credentials use Chatbook's existing config/encryption behavior.
No credentials or existing user data are copied into an artifact.

Preserve ADR-029's private path and Windows ACL checks and ADR-032's immutable
installed assets. Refuse redirected/reparse-point Data paths rather than
following them. Do not weaken checks to accommodate removable filesystems.
Portable does not mean anonymous, encrypted by default, or isolated from the
host OS. Explicitly selected external workspace/file paths remain external.

The empty distribution contains no Data directory. Relocation is tested only
after every Chatbook process has exited; arbitrary user-configured absolute
paths, OS credential migration and ACL ownership across accounts/machines
are outside the spike's guarantee. The launcher never silently falls back to
the installed profile when portable storage is unavailable.

## Alternatives

- Bundle Ghostty: its desktop GUI does not support Windows.
- Build a libghostty shell: adds terminal rendering/input ownership unrelated
  to the installation proof requested by the user.
- Freeze with Nuitka immediately: adds compiler/build debugging before the
  portable-data and terminal boundary are proven. An embedded interpreter
  also preserves existing `sys.executable -m ...` subprocess contracts.
- Redirect config only: leaves home-based caches, temporary files and legacy
  defaults pointed at the installed profile.

## Verification

Run focused launcher/archive tests, build from a wheel, execute the bundled
interpreter outside the checkout, write/reopen real SQLite data after copying
the bundle to a different path, and inspect a native WezTerm launch. Record
runtime versions, archive hashes, dependency provenance and limitations in
the spike README and task notes. A same-machine relocation does not prove
another Windows account, another machine, FAT/exFAT, or every optional feature.

## Spike findings

The real Notes database and SQLite probe persist across process restarts, and
Chatbook renders its first-run UI in the bundled WezTerm. Copying a used profile
to a new path fails with `Recovery required: recovery_scope_uncertain`. Existing
recovery records deliberately pin path/inode identities; they must not be
deleted, patched or bypassed by a packaging launcher. The launcher therefore
records its first location and gives an explicit refusal before opening a moved
profile. Safe relocation remains an unmet qualification criterion and requires a
supported recovery ownership transition beyond this spike.

The stock WezTerm binary also uses Windows' known-folder API for its own IPC and
runtime files, ignoring HOME/USERPROFILE for those files. Chatbook's data is
contained; zero host footprint is not established. No terminal fork or filesystem
API interception is introduced to conceal this limitation.
