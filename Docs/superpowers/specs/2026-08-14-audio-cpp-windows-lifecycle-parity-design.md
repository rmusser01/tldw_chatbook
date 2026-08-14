# audio.cpp Windows Lifecycle and Clone Parity — Design

Status: Approved

Date: 2026-08-14

Task: [TASK-13208](../../../backlog/tasks/task-13208%20-%20Add-Windows-parity-for-guided-audio.cpp-lifecycle-and-cloning.md)

Governing decisions:

- [ADR-023: TTS adapter registry and audio.cpp runtime](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-029: Local private data boundary](../../../backlog/decisions/029-local-private-data-boundary.md)
- [ADR-050: Generated audio.cpp setup ownership](../../../backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md)
- [ADR-051: Private TTS clone-reference assets](../../../backlog/decisions/051-private-tts-clone-reference-assets.md)

Parent design:

- [audio.cpp Guided Model Setup and Clone Profiles](2026-08-09-audio-cpp-guided-model-setup-design.md)

## Purpose

Complete Windows parity for the existing Guided audio.cpp lifecycle and clone
reference flow without adding a second process manager, scanner, profile owner,
or configuration source.

The supported product baseline is Windows 10 or later on x86 or x64 with
Python 3.12 or later. ARM64 is outside this task. Platform code must remain
pointer-width neutral so ARM64 can be qualified later without a lifecycle
rewrite, but no ARM64 support or compatibility label is implied here.

## Scope

This task covers:

- Guided binary detection and manual path admission;
- Windows drive, UNC, Unicode, and long-path handling;
- no-follow package scanning with junction/reparse rejection;
- owner-private generated launch artifacts and clone materializations;
- exact native child-process ownership, shutdown, and handle closure;
- Windows-aware CPU and evidenced accelerated backend selection;
- Settings and Speech Lab truth, recovery, privacy, and focus parity;
- hermetic Windows tests; and
- a provisioned Windows CPU real-process and audible UAT handoff.

It does not distribute or install `audiocpp_server`, claim ownership of process
descendants, support daemonizing builds, add a general process-tree manager,
or convert every existing Chatbook private path to native Windows ACLs.

## Decision summary

Chatbook will preserve the existing cross-platform lifecycle and add the
smallest native Windows capabilities it lacks:

1. a focused stdlib Windows filesystem capability for no-reparse handles,
   stable identity, protected DACLs, ownership locks, and exact cleanup;
2. an explicit close operation for the exact native subprocess transport
   created by the existing supervisor;
3. Windows-backed scanner file opens and directory pins rather than the current
   fail-closed `O_NOFOLLOW` absence;
4. Windows CPU/backend evidence in the existing recipe selection model; and
5. a parameterized PowerShell UAT runner that consumes user-provisioned inputs.

No new runtime dependency is added. Native calls use the Python standard
library's `ctypes`, `msvcrt`, and existing `asyncio` Proactor subprocess support.

## Architecture

### One lifecycle owner

`AudioCppSupervisor` remains the sole process owner. Guided launch continues to
produce one `AudioCppManagedLaunchConfig` and one generated-artifact owner.
Clone references continue to enter through the existing profile admission and
materializer boundaries.

Windows capabilities are injected below those owners. UI code does not call
Win32 APIs, inspect handles, infer ACLs, or maintain cleanup registries.

### Native Windows filesystem capability

One internal capability provides:

- conversion of an admitted absolute drive or UNC path to a wide Win32 path;
- native opens with `FILE_FLAG_OPEN_REPARSE_POINT` and directory backup
  semantics;
- non-inheritable handles and explicit handle closure;
- stable volume/file identity from the opened object;
- reparse-attribute inspection before admission;
- exclusive nonblocking ownership locks with `LockFileEx`;
- protected DACL installation and verification; and
- deletion disposition applied to the exact retained handle.

Long-path support is explicit. Internally generated Win32 paths use the wide
extended namespace only after ordinary absolute path validation. User-provided
device namespaces and drive-relative paths such as `C:relative` are rejected.
Unicode components are preserved. Public failures never echo the full path.

The capability is native only on Windows. Injectable host-independent doubles
cover failure and race matrices on other platforms; real Windows tests cover
the actual calls.

### ACL posture

Generated launch directories/files and clone materialization
directories/files receive a protected DACL with inheritance removed. Allowed
trustees are limited to:

- the current process token's user SID;
- LocalSystem; and
- Builtin Administrators.

The owning user receives the access needed by the exact artifact lifecycle;
system and administrators retain administrative access. Directory ACEs inherit
only to owned children. Every installation is re-read and verified. A null or
invalid DACL, an unexpected allow trustee, an unrecognized reparse point, or a
native error fails the operation closed.

The user-facing statement is truthful: the artifact is protected for the
Windows account, local system/administrators retain access, the content is
plaintext, and filesystem ACLs are not encryption or forensic erasure.

This posture applies only to TASK-13208's generated audio.cpp launch artifacts
and clone materializations. Other ADR-029 Windows paths remain unverified until
separately implemented.

### Exact cleanup

Cleanup owns retained native handles and recognized random names. It deletes
only the exact opened object through handle-directed deletion after verifying:

- object type and stable identity;
- expected protected ACL posture;
- exact recognized owner/asset names;
- no unexpected directory entries; and
- no live ownership lock.

Unknown directories, additional entries, reparse points, substituted names,
identity changes, and ambiguous cleanup failures are preserved and reported as
bounded cleanup failures. Age is never ownership evidence.

### Package scanner

The scanner keeps its existing bounded queue, recipe matching, cancellation,
and partial-result vocabulary. On Windows it pins each directory with a native
no-reparse handle that denies rename/delete while enumerating, and opens each
allowlisted metadata file through the native capability before reading.

The scanner rejects nested symlinks, junctions, mount-point reparses, other
reparse points, special files, and identity changes. A disclosed top-level link
may still resolve through the existing review flow, but its canonical target is
reopened and pinned before becoming accepted identity.

Drive roots, UNC roots, Unicode paths, paths beyond `MAX_PATH`, case-insensitive
collisions, and reparse substitution receive real Windows coverage. The
scanner never recursively searches a drive or user profile.

### Binary admission

Detection remains explicit and side-effect free:

- retain the configured candidate;
- use `shutil.which` with Windows `PATHEXT` behavior for
  `audiocpp_server.exe`;
- inspect only separately reviewed conventional candidates if the upstream
  distribution documents them; and
- keep manual file selection available.

No recursive disk search, execution, version probe, PATH rewrite, or silent
ambiguous selection occurs during detection or Save. Admission rejects
relative, drive-relative, directory, reparse, and non-executable selections.

### Process creation and ownership

The supervisor keeps direct `asyncio.create_subprocess_exec` creation with the
exact argument vector and `shell=False`. Windows uses the default Proactor
subprocess implementation. No process group, shell, job object, process-tree
kill, listener adoption, or descendant claim is introduced.

The owned-process record gains one explicit native-transport close operation.
The launch task is retained across cancellation so a process created while
Start/Restart/App Close is racing cannot escape before ownership publication.
After the exact child exits, the supervisor joins its one waiter, output drains,
monitor, probes, clients, hooks, generated artifact, and clone materializations,
then closes the exact process transport/handle once.

Graceful termination and force termination apply only to that exact process.
The same outer shutdown deadline governs startup, stop, and application close.
A foreground budget expiry may report retained cleanup, but `wait_closed()`
cannot claim completion while a process handle or generation resource remains.

Native Windows tests inspect the handle before and after settlement and prove
it becomes invalid only after exact exit/join. Builds that daemonize or break
away from the owned process are unsupported and receive no clean-shutdown
claim.

### Backend evidence

Backend selection accepts Windows plus normalized x86/x64 architecture names.
CPU is the baseline candidate. Accelerated backends are selectable only when
the exact recipe, host architecture, and binary evidence admit them.

Static declarations begin as `Expected` or `Untested`. Only a provisioned real
tuple may become `Verified`. Auto fallback remains limited to the existing
allowlisted backend-unavailable failures and cannot start until the exact prior
child and resources are fully cleaned.

x86 lifecycle code is supported, but a recipe/backend tuple remains
unverified until a compatible 32-bit server and package complete the same real
gate. No x64 result is projected onto x86.

### Settings and Speech Lab

The existing saved/applied/process and package-evidence states remain
authoritative. Windows adds no second preferences or runtime copy.

The UI must:

- display actual ACL posture rather than POSIX modes;
- preserve keyboard/focus and immutable-action behavior;
- retain last valid playable WAV on later failure;
- keep sample, clone, profile-save, restart, and shutdown recovery unchanged;
- expose stable phase-specific Windows failures without raw native messages;
  and
- avoid claiming Running, Verified, private, or cleaned before the matching
  evidence settles.

## Error and privacy contract

Win32 error numbers, exception messages, SIDs, full executable/model/runtime
paths, environment values, transcript, reference bytes, prompt text, and raw
child output do not enter public errors, persistent logs, or exception graphs.

Stable phases extend existing codes rather than introducing path-bearing
Windows variants: binary unavailable, package changed, ACL unavailable,
generated artifact unavailable, startup failed, cleanup failed, and clone
materialization unavailable.

Control-flow exceptions preserve their family after retained cleanup. Ordinary
cleanup failure remains owned and retryable through the existing app lifecycle.

## Verification

### Hermetic tests

The Windows CI matrix covers:

- Win32 capability constants, pointer-width behavior, cleanup, and sanitization;
- protected-DACL success, unexpected trustees, invalid/null DACL, and failures;
- drive, UNC, Unicode, long-path, case, reserved-name, and device-namespace
  validation;
- top-level link review and nested junction/reparse rejection;
- substitution during enumeration/open and exact handle identity;
- generated JSON and clone materialization create/validate/cleanup matrices;
- exact process wait/terminate/force/handle-close ordering;
- start/restart/crash/cancel/app-close races under one deadline;
- CPU/accelerated selection and allowlisted fallback; and
- mounted Settings/Speech Lab truth, privacy, focus, and recovery.

Normal CI uses local fixtures and a real short-lived helper process. It does not
download audio.cpp or model packages.

### Provisioned Windows UAT

The repository provides one parameterized PowerShell entry point. The operator
supplies the compatible server binary, local and/or Model Library package
roots, clone reference inputs, and exact expected identities. The runner:

1. creates disposable config, data, profile, model, and runtime roots;
2. verifies Guided Save performs no launch;
3. verifies generated JSON acceptance, health, catalog, and exact model IDs;
4. synthesizes and structurally validates one text WAV;
5. synthesizes and structurally validates one clone-reference WAV;
6. exercises local and Model Library package paths;
7. performs restart, cancellation, crash, and final app shutdown checks;
8. proves no owned child, handle, task, client, endpoint, generated artifact,
   or clone materialization remains; and
9. requires a human audible-playback confirmation.

The script emits only sanitized evidence. It does not bake in a specific
machine version, username, path, or device. TASK-13208 remains In Progress
until this provisioned gate passes.

## Alternatives rejected

### Add `pywin32`

Rejected because the required primitives are narrow, stdlib `ctypes` is
already used for no-reparse Windows reads in this repository, and a new
platform dependency would widen packaging and installation risk.

### Introduce Windows Job Objects

Rejected because TASK-13208 explicitly owns only the exact created process and
must not claim arbitrary descendants. A job object would establish a different
process-tree ownership contract.

### Rewrite POSIX and Windows behind a new general lifecycle framework

Rejected because the existing POSIX lifecycle is heavily verified. The task
needs a Windows capability seam, not a replacement supervisor, scanner, or
private-storage architecture.

### Keep Windows privacy `Unverified`

Rejected because clone references and generated configuration are plaintext
private artifacts and the task explicitly requires an implemented truthful ACL
posture before enabling them.

## Rollout and rollback

- External and user-provided JSON sources remain unchanged.
- POSIX behavior remains on its current implementation.
- If native Windows capability initialization or verification fails, Guided
  launch and clone materialization fail closed with bounded recovery.
- Disabling Windows Guided/clone admission leaves saved configurations and
  profile references visible; it does not delete them or silently retarget.
- Rollback never deletes user-selected models, unknown runtime directories, or
  unrecognized reparse objects.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md` amendment

Reason: TASK-13208 changes the Windows security posture for two classes of
private plaintext artifacts. ADR-023, ADR-050, and ADR-051 already fix the
process, generated-configuration, and clone owners; the narrow ADR-029
amendment defines the newly verified Windows ACL boundary without implying
that every other Windows private path is now protected.

## ARM64 follow-up

The native handle, ACL, path, scanner, and supervisor design is architecture
neutral and uses pointer-sized Win32 types. A later ARM64 task should therefore
need little lifecycle code. Its main work is:

- admitting `windows/arm64` architecture normalization;
- proving compatible audio.cpp binary and backend availability;
- adding exact recipe/backend evidence; and
- running the full provisioned ARM64 real-process and audible gate.

Until that evidence exists, ARM64 remains unsupported rather than inheriting
x64 or x86 labels.
