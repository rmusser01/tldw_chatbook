# Terminal dependency qualification

These scripts implement the content-free pre-product qualification gate required
by ADR-099. They do not import Chatbook product code and do not add project
dependencies.

## Prepare an isolated row

Use a unique disposable directory and the exact host interpreter being claimed:

```bash
ROW_DIR=$(mktemp -d /tmp/tldw-task-22512-macos-arm64-py311.XXXXXX)
python3.11 scripts/terminal_qualification/common.py prepare-row \
  --row-id macos-arm64-py311 \
  --row-dir "$ROW_DIR" \
  --requirement pyte==0.8.2 \
  --requirement "wcwidth>=0.2.14,<1" \
  --json-out "$ROW_DIR/artifacts.json"
```

`prepare-row` creates a venv, downloads every named requirement and dependency,
hashes the exact files, installs only those files with `--no-index`, then
rehashes every artifact. `artifacts.json` binds each artifact's filename, size,
pre-install hash, post-install hash, license facts, and the installed
distribution's primary-file and `RECORD` hashes. It refuses a row directory
that already contains qualification state.

## Reproduce the Linux ARM64 Docker row

The recorded Linux row used `ubuntu:24.04` with immutable local image ID
`sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
The image tag is not treated as immutable: this recipe refuses to run if the
current tag no longer resolves to that exact image. A new container has a new
container ID and new timestamps, so its raw JSON is a comparable reproduction,
not a byte-identical copy of the recorded row.

Run from the qualification worktree root on an ARM64 Docker host:

```bash
TASK22512_LINUX_ROOT=$(mktemp -d /tmp/tldw-task-22512-linux-arm64-py312.XXXXXX)
TASK22512_WORKTREE=$PWD
TASK22512_IMAGE_ID=sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea
test "$(docker image inspect ubuntu:24.04 --format '{{.Id}}')" = "$TASK22512_IMAGE_ID"
test "$(docker image inspect ubuntu:24.04 --format '{{.Os}}/{{.Architecture}}')" = "linux/arm64"
docker run --platform linux/arm64 --rm \
  --mount type=bind,src="$TASK22512_WORKTREE",dst=/worktree,readonly \
  --mount type=bind,src="$TASK22512_LINUX_ROOT",dst=/qualification \
  --workdir /worktree \
  "$TASK22512_IMAGE_ID" \
  bash -lc '
    set -eu
    test "$(uname -m)" = aarch64
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip zsh vim less procps
    TASK22512_CONTAINER_ID=$(hostname)
    python3 -B scripts/terminal_qualification/common.py prepare-row --row-id linux-arm64-py312 --row-dir /qualification/row --requirement pyte==0.8.2 --requirement "wcwidth>=0.2.14,<1" --json-out /qualification/row/artifacts.json --runtime-kind docker --runtime-image ubuntu:24.04 --runtime-image-id sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea --runtime-container-id "$TASK22512_CONTAINER_ID"
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell default --json-out /qualification/row/env-default.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell bash --json-out /qualification/row/env-bash.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell zsh --json-out /qualification/row/env-zsh.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/pyte_probe.py --artifact-manifest /qualification/row/artifacts.json --json-out /qualification/row/pyte.json
    if /qualification/row/venv/bin/python -B scripts/terminal_qualification/pywinpty_probe.py --artifact-manifest /qualification/row/artifacts.json --json-out /qualification/row/pywinpty.json; then exit 1; else test "$?" -eq 1; fi
  '
../../.venv/bin/python -B scripts/terminal_qualification/common.py collect-row \
  --row-dir "$TASK22512_LINUX_ROOT/row" \
  --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw \
  --replace
```

The pywinpty command must exit 1 after writing
`UNSUPPORTED_FAIL_CLOSED`; a zero exit on Linux is a qualification failure.
The container installs capture tools only inside the disposable row runtime and
does not modify project dependency manifests.

## Run probes

Always run probes with the prepared row interpreter. A POSIX row is exactly
`artifacts`, `environment-default`, `environment-bash`, `environment-zsh`,
`pyte`, and `pywinpty`:

```bash
"$ROW_DIR/venv/bin/python" scripts/terminal_qualification/environment_probe.py \
  --shell default --json-out "$ROW_DIR/env-default.json"
"$ROW_DIR/venv/bin/python" scripts/terminal_qualification/environment_probe.py \
  --shell bash --json-out "$ROW_DIR/env-bash.json"
"$ROW_DIR/venv/bin/python" scripts/terminal_qualification/environment_probe.py \
  --shell zsh --json-out "$ROW_DIR/env-zsh.json"
"$ROW_DIR/venv/bin/python" scripts/terminal_qualification/pyte_probe.py \
  --artifact-manifest "$ROW_DIR/artifacts.json" \
  --json-out "$ROW_DIR/pyte.json"
"$ROW_DIR/venv/bin/python" scripts/terminal_qualification/pywinpty_probe.py \
  --artifact-manifest "$ROW_DIR/artifacts.json" \
  --json-out "$ROW_DIR/winpty.json"
```

A native Windows row instead consists of exactly `artifacts`,
`environment-default`, `environment-powershell`, `environment-cmd`, `pyte`,
and `pywinpty`:

```powershell
& "$ROW_DIR\venv\Scripts\python.exe" -B scripts\terminal_qualification\environment_probe.py `
  --shell default --json-out "$ROW_DIR\environment-default.json"
& "$ROW_DIR\venv\Scripts\python.exe" -B scripts\terminal_qualification\environment_probe.py `
  --shell powershell --json-out "$ROW_DIR\environment-powershell.json"
& "$ROW_DIR\venv\Scripts\python.exe" -B scripts\terminal_qualification\environment_probe.py `
  --shell cmd --json-out "$ROW_DIR\environment-cmd.json"
& "$ROW_DIR\venv\Scripts\python.exe" -B scripts\terminal_qualification\pyte_probe.py `
  --artifact-manifest "$ROW_DIR\artifacts.json" --json-out "$ROW_DIR\pyte.json"
& "$ROW_DIR\venv\Scripts\python.exe" -B scripts\terminal_qualification\pywinpty_probe.py `
  --artifact-manifest "$ROW_DIR\artifacts.json" --json-out "$ROW_DIR\pywinpty.json"
```

The Windows `default` probe uses a code-owned policy: `pwsh.exe` first,
Windows PowerShell (`powershell.exe`) second, then the validated `COMSPEC` CMD
path. It records the selected family. The named PowerShell and CMD probes are
still both mandatory members of the Windows generation; the default probe is
not an alias for either output file.

`environment_probe.py` runs only the explicitly selected shell from a validated,
scrubbed environment. On POSIX it first exercises the real account shell's
normal startup path, waiting for interactive readiness before sending the probe
command. A separate controlled temporary-home profile then proves profile
execution, intentional sensitive-key repopulation, and command discovery.

The native Windows path preserves normal startup semantics through a real,
disposable local account and profile. The supervisor creates that account and
profile, then starts only a waiting Python bootstrap under the disposable
identity with `CreateProcessWithLogonW(LOGON_WITH_PROFILE)`. It assigns the
bootstrap to the kill-on-close Job, independently verifies exact Job
membership, process identity, loaded profile path, and disposable user hive,
and only then sends the release token. The bootstrap verifies its own Job
membership again before writing profile fixtures or spawning the shell.

PowerShell launches with `-NoLogo`, not `-NoProfile` or `-NonInteractive`, and
discovers profile/module fixtures from the disposable account's normal
Documents and home paths. CMD launches with `/Q`, not `/D`, and its bootstrap
writes `Command Processor\AutoRun` only through `RegOpenCurrentUser` after
verifying that the running identity is the disposable account. The interactive
user's `HKEY_CURRENT_USER` is never opened for write. Setting `USERPROFILE`
alone is not considered isolation. Account/profile cleanup is supervisor-owned
and runs in `finally`, including after probe failure or crash. If account
creation, profile creation/loading, alternate-user launch, Job admission, or
identity/profile verification is unavailable, collection fails closed before
the real shell starts and performs no interactive-HKCU mutation.

All bounded command output uses reader threads over pipes with one combined
retained-byte ceiling plus a fixed 8-KiB read chunk. Exceeding the ceiling is
recorded as `output_overflowed`, immediately terminates the owned process
group/Job, and fails the environment or formatter operation. `TemporaryFile`
is used for neither command-output capture nor overflow buffering.

These Windows paths have host-independent behavioral/source coverage but have
not run on a native Windows host. An unavailable optional shell is recorded as
unavailable; another shell is never substituted. No native Windows PASS is
claimed.

`pyte_probe.py` reopens the manifest and exact wheel, rechecks all three artifact
hash fields and the installed distribution facts, and binds those facts into
the parser result. Its full-screen row requires one available real editor,
pager, and monitor, a clean bounded exit, and a class-specific interactive
marker for each class. Fixture count is zero for that row. Captures are parsed
in memory and discarded, and bounded cleanup reaps the whole process group.

`pywinpty_probe.py` owns the native Windows boundary. A non-Windows host writes
every mandatory Windows row as `UNSUPPORTED_FAIL_CLOSED` and exits nonzero. The
native path performs package/platform/API identity first, uses only low-level
`winpty.PTY` with `Backend.ConPTY`, and creates a non-inheritable kill-on-close
Job before process admission.

The host-independent harness tests enforce the intended native contracts: each
output credit is bytes returned by a real `PTY.read(blocking=True)`, capped at
64 KiB, with only one unacknowledged chunk and an explicit acknowledgement. A
dedicated fourth terminal is drained with nonblocking credited reads until its
startup output is acknowledged and the terminal is quiet. Real read, write,
resize, and cancellation calls must then all enter. Naturally synchronous
write, resize, and cancellation calls must complete before handoff; only the
known blocking read may remain unresolved. Entered, completed-at-handoff, and
completed-post-close outcomes are recorded separately. Priority terminal close
must terminate that read boundedly; synchronous calls are not required to
remain unresolved or return after close.

Post-exit reads drain until `iseof()` or a one-second cutoff and verify the
complete multi-buffer digest separately from EOF. Normal cleanup retains a
waitable handle for every stable known Job member before closing the Job,
requires positive and equal expected, retained, and `WAIT_OBJECT_0` counts,
then reaps the controller process. Candidate observations are committed only
after this cleanup succeeds. A retention/open/wait/reap exception, timeout,
`WAIT_FAILED`, unknown wait result, false all-waited fact, or partial count
invalidates the candidate instead of publishing an otherwise passing result.
The four-session RSS delta includes controller, worker/IPC, and helpers while
excluding exactly the four fixture workloads. Cleanup and all observations are
bounded and fail closed. These tests use fakes and static guards where
necessary: native execution of all mandatory Windows rows remains unproven,
and no native Windows PASS is claimed.

The application-crash fixture has three levels: supervisor, a separate
app/controller process, and its admitted worker. The app/controller alone owns
the non-inheritable kill-on-close Job handle and actually aborts. Before that
abort, the supervisor pre-opens only `SYNCHRONIZE` handles for the stable known
app/worker descendant set; it never owns or duplicates the Job handle. Closing
the sole Job handle through app abort must make every retained wait handle
return `WAIT_OBJECT_0`; timeout, wait failure, or an unknown result fails
closed. Terminal-child crash facts are recorded separately and cannot satisfy
`app_crash_observed`. This topology has host-independent behavioral and source
coverage only; native Windows execution remains unproven.

Every probe accepts `--json-out PATH`, refuses an existing output by default,
and allows replacement only when `--replace` is explicit. A failed mandatory row
returns nonzero.

## Collect content-free JSON

```bash
../../.venv/bin/python -B scripts/terminal_qualification/common.py collect-row \
  --row-dir "$ROW_DIR" \
  --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw
```

That command is for a row ID with no retained destination. Add `--replace` only
when intentionally replacing an existing complete validated generation; the
collector otherwise refuses to overwrite it.

`collect-row` parses every source JSON before publishing anything. It applies
the current exact schema and semantic checks, scans all strings, requires one
row/platform/runtime identity across exactly the complete platform-specific
six-probe sibling set named above, rejects missing, extra, or duplicate probes
and source generation or
collection metadata, and then injects one new 32-hex generation ID and the
exact collection command into every sibling.

Publication is recoverable and fail closed; it does not claim impossible
universal multi-file atomicity. Before replacing a row, the collector validates
the complete previous generation. A stale pre-marker row must be exactly six
canonical files from one legacy generation with one row/platform/runtime
identity. The collector copies those exact files into a recovery directory,
creates and validates their hash manifest, and writes a pending-transaction
marker before replacing any visible sibling. It then replaces all six staged,
flushed files and commits `.current-generation` last. The current-generation
validator accepts a row only when no pending marker exists and the committed
manifest names and hashes exactly those six files with one generation identity.

A catchable exception during replacement restores the complete previous bytes
and marker state and removes transaction residue. Abrupt process death leaves
the pending marker, so the mixed/partial visible set cannot validate as current;
recovery restores the complete previous generation. A restored legacy set
remains deliberately unaccepted as current until a later successful collection
commits a new `.current-generation` manifest. `collect-row` never copies wheels,
virtual environments, shell output, terminal output, or profile files.

## Formatter ratchet

`format_ratchet.py snapshot` resolves `--base` once, materializes each base blob
in a temporary directory, measures formatter debt with the repository Ruff, and
stores the immutable commit SHA plus source and normalized-diff hashes.

```bash
../../.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --head HEAD --baseline Docs/superpowers/reviews/evidence/task-22512/format-baseline.json
```

`verify --head HEAD` first remeasures every immutable base fact, including each
source hash, normalized formatter-diff hash, debt count, red-path set, and the
recorded Ruff version, so baseline tampering or tool drift fails closed. It then
measures committed `HEAD`, reads zero-context changed-line ranges, and rejects
either changed-line formatter overlap or growth in normalized formatter debt.
Omit `--head HEAD` only when an intentional local diagnostic must include
unstaged working-tree changes; the Task 1 gate uses the exact command above.

## Privacy and evidence rules

Raw evidence is checked against strict shape-specific allowlists. Each probe and
each result-row ID has its own exact keys and field types, so a field valid for a
different probe or row, an unknown key, a wrong type or placement, or a nested
command object fails closed. Strings are accepted only in named metadata fields,
command argv, or environment-key-name lists, and every string is scanned for
secret-shaped material before collection. This includes genuinely allowed
fields: credential assignments, JWT-shaped or Authorization/Bearer values in a
license field, and token values or GitHub-style token forms in argv are rejected
even though their containing fields are otherwise permitted.

Raw evidence may contain only versions, hashes, wheel metadata, platform facts,
timestamps, elapsed time, RSS/peak measurements, statuses, booleans, counts,
environment key names, and exact argv/cwd metadata. Do not add environment
values, profile content, terminal output, captured stdout/stderr, command
output, credentials, tokens, or secrets. `collect-row` validates every sibling
JSON before copying any file, so one invalid file prevents a partial row from
being admitted.

Forbidden payloads include environment values, profile content, terminal output,
and secrets.

Keep each OS/Python/architecture row in a fresh disposable environment. Never
retarget the repository's shared editable install. Packaging metadata, mocks,
Wine, and upstream CI are not substitutes for required native behavior.
