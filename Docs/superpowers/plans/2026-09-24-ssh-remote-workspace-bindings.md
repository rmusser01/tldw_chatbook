# SSH Remote Workspace Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ssh-filesystem` workspace bindings: full `fs_*` parity against a remote Linux root over a transient SSH worker, with graceful per-binding degradation and automatic recovery.

**Architecture:** Reuse the one-shot pinned-worker architecture over SSH stdio. A stdlib-only worker bundle (built at commit time) is piped to `ssh <target> python3 -I -c '<bootstrap>'` per call; the worker pins the remote root and executes ops server-side. Status is an in-memory cache bucketed by the protocol's existing `admitted` marker; composition reads cache only.

**Tech Stack:** Python ≥3.12 (app), stdlib-only worker bundle (floor 3.10), OpenSSH client (ControlMaster), SQLite (existing tables), Textual UI, pytest + Hypothesis.

**Spec:** `Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md` (revision 5). The spec travels with this plan — executors read both. Where this plan and the spec disagree, the spec wins and the plan gets fixed.

## Global Constraints

- **Branch:** cut from `origin/dev` (not the drafting branch). Re-confirm ADR number 181 is free there; adjust all references if not.
- **No new runtime dependencies.** The worker bundle is stdlib-only, Python floor 3.10, compiled under a real 3.10 in CI (`uv python install 3.10`).
- **Auth delegated:** BatchMode always; no credentials in app, keyring, or binding metadata, ever.
- **Hot-path rule:** run composition reads the in-memory status cache only — never spawns a process, never touches the network.
- **Exit codes (fixed):** 75 = worker watchdog; 76 = bootstrap version gate ("python ≥ 3.10 required"); 127 = interpreter missing; 2 = worker framed failure; 255 = ssh-level.
- **Bootstrap payload:** the exact version-gated string in the spec's Transport section; charset `letters, digits, space, _ . , ( ) " ; : < > = + % \ [ ]`.
- **Admitted-marker bucketing:** no marker before failure → transport/BLOCKED-eligible; marker seen then anything → typed op error, status unchanged. Pin failure → identity-stale + re-probe, not BLOCKED.
- **No schema migration:** `workspace_runtime_bindings.binding_kind` is text; SSH bindings are rows with `binding_kind='ssh-filesystem'`.
- **Testing:** targeted pytest runs only (repo rule — ask before full sweeps). Live run against a real server before any Done.
- **ADR-181 authored and committed before any Phase 0 code.**

---

## File Structure

New files (package `tldw_chatbook/` unless noted):

- `Tools/workspace_wire_decode.py` — stdlib-only request/response decoder (worker side of Phase 0 split).
- `Tools/remote_binding_locator.py` — locator parsing, per-component charsets, ssh argv builder.
- `Tools/build_remote_worker_bundle.py` — commit-time bundle generator.
- `Tools/remote_worker_bundle.py` — committed generated artifact (do not hand-edit).
- `Tools/remote_sensitive_paths.py` — remote-home-relative denylist (data only).
- `Tools/remote_root_types.py` — `LocalRoot | RemoteRoot` union descriptors.
- `Tools/remote_workspace_transport.py` — `SshMasterManager`, `RemoteWorkspaceTransport` (spawn, magic, admitted watch, deadlines, taxonomy).
- `Tools/remote_workspace_executor.py` — `RemoteWorkspaceToolExecutor` (`execute(tool, args, intent)`).
- `Tools/remote_binding_status.py` — status cache, recovery-probe scheduler, identity staleness.
- `backlog/decisions/181-ssh-remote-workspace-bindings.md` — ADR (Task 1).

Modified files: `Tools/workspace_tool_protocol.py`, `Tools/workspace_tool_worker.py`, `Workspaces/models.py`, `Workspaces/registry_service.py`, `Tools/workspace_tool_executor.py`, `Tools/workspace_file_roots.py`, `Agents/local_tool_provider.py`, `Agents/virtual_cli_provider.py` (check-only), `Agents/project_instruction_resolver.py`, `Chat/console_chat_controller.py`, `UI/Screens/settings_screen.py`, `UI/Console_Modules/session.py`, `UI/Console_Modules/workspace.py`, `config.py`, `app.py` (shutdown hook), `Docs/User_Guide/console/context-and-rag.md`, repo `AGENTS.md`.

Test files created under `Tests/` mirroring package paths (listed per task).

---

### Task 1: ADR-181, backlog task, branch setup

**Files:**
- Create: `backlog/decisions/181-ssh-remote-workspace-bindings.md`
- Create: backlog task via CLI

**Interfaces:**
- Produces: ADR path referenced by every later commit message; backlog task id `task-<N>` recorded in the plan execution notes.

- [ ] **Step 1: Cut the implementation branch from dev**

```bash
git fetch origin dev
git switch -c feat/ssh-remote-workspace-bindings origin/dev
grep -l "^# ADR-181" backlog/decisions/*.md || echo "181 free"
```

If 181 is taken, pick the next free number `X` and use it consistently from here on.

- [ ] **Step 2: Write ADR-181**

Content: Decision (six bullets mirroring the spec's Decision summary: SSH_FILESYSTEM kind; transient stdlib worker over ssh stdio with exact bootstrap; admitted-marker bucketing + two-tier watchdog; status cache with optimistic cold start, transport-only learning, debounced recovery; LocalRoot|RemoteRoot split; executor-owned ControlMaster). Context (the Problem section). Alternatives table copied from the spec. Links: the spec, ADR-005/028/032/069/101/102/174/175 by filename.

```bash
backlog task create "SSH remote workspace bindings" -d "Implement Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md" --ac "Full fs parity on ssh-filesystem bindings,Availability guarantee both directions,Nothing left behind on server,No credentials stored" -l console,workspaces
backlog task edit <id> -a @robert -s "In Progress"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/181-ssh-remote-workspace-bindings.md
git commit -m "docs: ADR-181 — SSH remote workspace bindings (spec 2026-09-24)"
```

---

### Task 2: Stdlib wire decoder (Phase 0a)

**Files:**
- Modify: `Tools/workspace_tool_protocol.py` (extract wire field constants; keep pydantic models unchanged in behavior)
- Create: `Tools/workspace_wire_decode.py`
- Test: `Tests/Tools/test_workspace_wire_decode.py`

**Interfaces:**
- Consumes: `WorkspaceToolRequest.from_bytes` (existing, unchanged).
- Produces: `decode_request(raw: bytes) -> dict` and `decode_response(raw: bytes) -> dict`, both raising `WireDecodeError(reason: str)`. Field names identical to the pydantic models.

- [ ] **Step 1: Write the failing conformance-style unit tests**

```python
# Tests/Tools/test_workspace_wire_decode.py
import json
import pytest
from tldw_chatbook.Tools.workspace_wire_decode import decode_request, WireDecodeError

def _valid_request() -> bytes:
    return json.dumps({
        "version": 1, "op": "fs_read", "request_id": "r1",
        "root_locator": "/tmp/w", "root_identity": ["/tmp/w", 1, 2, 16877],
        "ancestor_identities": [["/", 1, 1, 16877]],
        "args": {"path": "a.txt"}, "timeout_seconds": 300,
        "sensitive_exclusions": [],
    }).encode()

def test_accepts_valid_request():
    assert decode_request(_valid_request())["op"] == "fs_read"

@pytest.mark.parametrize("mangle", [
    lambda b: b.replace(b'"op"', b'"op2"', 1),            # unknown field (rename arg->op2 keeps known op) -> actually unknown field
    lambda b: b.replace(b'"version": 1', b'"version": 2'),# wrong version
    lambda b: b'{"version":1,"version":1}',               # duplicate keys
    lambda b: b.replace(b'"op"', b'"op"', 1),             # placeholder replaced below
])
def test_rejects_malformed(mangle):
    with pytest.raises(WireDecodeError):
        decode_request(mangle(_valid_request()))
```

Adjust the parametrize list per the corpus in Task 3; the exact valid-request shape comes from `WorkspaceToolRequest`'s pydantic fields — read `workspace_tool_protocol.py` first and mirror every required field.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest Tests/Tools/test_workspace_wire_decode.py -x -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the decoder**

```python
# Tools/workspace_wire_decode.py
"""Stdlib-only wire decoder for the pinned worker. Must accept/reject
identically to the parent's pydantic models (conformance corpus test)."""
import json

class WireDecodeError(ValueError):
    pass

_MAX_BYTES = 10 * 1024 * 1024
_KNOWN_FIELDS = {  # mirror WorkspaceToolRequest.model_fields
    "version", "request_id", "op", "root_locator", "root_identity",
    "ancestor_identities", "args", "timeout_seconds", "sensitive_exclusions",
}

def _no_dupes(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise WireDecodeError(f"duplicate key: {k}")
        seen.add(k)
    return dict(pairs)

def _no_constants(x):
    raise WireDecodeError(f"non-JSON constant: {x}")

def decode_request(raw: bytes) -> dict:
    if len(raw) > _MAX_BYTES:
        raise WireDecodeError("oversize")
    try:
        text = raw.decode("utf-8")  # strict: invalid UTF-8 raises
        doc = json.loads(text, object_pairs_hook=_no_dupes,
                         parse_constant=_no_constants)
    except WireDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireDecodeError(str(exc)) from exc
    if not isinstance(doc, dict):
        raise WireDecodeError("not an object")
    unknown = set(doc) - _KNOWN_FIELDS
    if unknown:
        raise WireDecodeError(f"unknown fields: {sorted(unknown)}")
    if doc.get("version") != 1 or isinstance(doc.get("version"), bool):
        raise WireDecodeError("wrong version")
    _validate_types(doc)
    return doc

def _validate_types(doc: dict) -> None:
    # strict bool-vs-int and per-field shapes mirroring pydantic strict mode;
    # fill from WorkspaceToolRequest.model_fields — each field gets an
    # explicit isinstance check with bool excluded where int is expected.
    ...
```

Complete `_validate_types` with one explicit check per field (the pydantic model is the reference — read it, mirror it). In `workspace_tool_protocol.py`, export the shared constants (`WIRE_VERSION = 1`, field-name tuple) and import them here so the two sides cannot drift structurally; **do not** import pydantic or loguru in this module.

- [ ] **Step 4: Run tests, commit**

```bash
python3 -m pytest Tests/Tools/test_workspace_wire_decode.py -q
git add Tools/workspace_wire_decode.py Tools/workspace_tool_protocol.py Tests/Tools/test_workspace_wire_decode.py
git commit -m "feat: stdlib wire decoder for pinned worker (Phase 0a)"
```

---

### Task 3: Decoder conformance corpus (Phase 0b)

**Files:**
- Test: `Tests/Tools/test_wire_conformance.py`

**Interfaces:**
- Consumes: `decode_request` (Task 2), `WorkspaceToolRequest.from_bytes` (existing).
- Produces: `CORPUS: list[tuple[str, bytes]]` — shared valid+malformed corpus used by later drift guards.

- [ ] **Step 1: Write the corpus test**

```python
# Tests/Tools/test_wire_conformance.py
"""The one-strict-JSON acceptance contract (TASK-32855) must hold on BOTH
decoders: identical accept/reject on the same corpus."""
import json
import pytest
from tldw_chatbook.Tools.workspace_tool_protocol import (
    WorkspaceProtocolError, WorkspaceToolRequest,
)
from tldw_chatbook.Tools.workspace_wire_decode import (
    WireDecodeError, decode_request,
)

def _base() -> dict:
    # Build from the pydantic model itself so the corpus can never go stale:
    req = WorkspaceToolRequest(...valid minimal args...)  # mirror test_workspace_tool_protocol.py fixtures
    return json.loads(req.to_bytes().decode())

CORPUS: list[tuple[str, bytes]] = []
def _add(name, doc): CORPUS.append((name, json.dumps(doc).encode()))
# valid
_add("valid", _base())
# malformed — each targets a TASK-32855 rule
d = _base(); d["version"] = 2; _add("wrong-version", d)
d = _base(); d["timeout_seconds"] = True; _add("bool-for-int", d)
CORPUS.append(("duplicate-key", b'{"version":1,"version":1,"op":"fs_read"}'))
CORPUS.append(("nan", b'{"version":1,"op":"fs_read","args":NaN}'))
CORPUS.append(("invalid-utf8", b'{"version":1,"op":"\xff\xfe"}'))
d = _base(); d["surprise"] = 1; _add("unknown-field", d)
CORPUS.append(("oversize", b'{"version":1,"op":"' + b"a" * (11*1024*1024) + b'"}'))
CORPUS.append(("truncated", b'{"version":1,"op":'))

@pytest.mark.parametrize("name,raw", CORPUS, ids=[n for n, _ in CORPUS])
def test_acceptance_is_identical(name, raw):
    pyd_ok, wire_ok = True, True
    try: WorkspaceToolRequest.from_bytes(raw)
    except (WorkspaceProtocolError, ValueError): pyd_ok = False
    try: decode_request(raw)
    except WireDecodeError: wire_ok = False
    assert pyd_ok == wire_ok, f"{name}: pydantic={pyd_ok} wire={wire_ok}"
```

- [ ] **Step 2: Run until green (fix the decoder, never pydantic)**

Run: `python3 -m pytest Tests/Tools/test_wire_conformance.py -q` — iterate on `workspace_wire_decode.py` only. This is the known-hard part of the split (spec: "matching pydantic's strict-type behaviour"); budget real time for bool-vs-int and unknown-field edge cases.

- [ ] **Step 3: Commit**

```bash
git add Tests/Tools/test_wire_conformance.py Tools/workspace_wire_decode.py
git commit -m "test: wire decoder conformance corpus — identical accept/reject (Phase 0b)"
```

---

### Task 4: Worker import closure (Phase 0c)

**Files:**
- Modify: `Tools/workspace_tool_worker.py` (use `decode_request`; drop pydantic import)
- Test: `Tests/Tools/test_worker_import_closure.py`

**Interfaces:**
- Consumes: `decode_request` (Task 2).
- Produces: a worker whose import closure is stdlib-only — precondition for the bundle (Task 8).

- [ ] **Step 1: Write the closure test**

```python
# Tests/Tools/test_worker_import_closure.py
import subprocess, sys, textwrap

BLOCKER = "import sys;mods=__import__('sys').modules;" \
          "bad=[m for m in mods if m.split('.')[0] in {" \
          "'pydantic','loguru','tldw_chatbook','httpx','rich','textual'}" \
          " and m.split('.')[0]!='tldw_chatbook.Tools.workspace_wire_decode'];" \
          "sys.exit(1 if any(bad) else 0)"

def test_worker_closure_is_stdlib_only():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, ".")
        import tldw_chatbook.Tools.workspace_tool_worker as w
        {BLOCKER}
    """)
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
```

The point: importing `workspace_tool_worker` must not pull pydantic/loguru/app config. (Exact blocker set: run once, read the failure list, freeze it in the test.)

- [ ] **Step 2: Run — expected FAIL listing pydantic via the protocol import**

- [ ] **Step 3: Migrate the worker**

In `workspace_tool_worker.py`: replace `WorkspaceToolRequest.from_bytes` with `decode_request` from `Tools/workspace_wire_decode.py` (same field access; the decoder returns the validated dict — access via `request["op"]` or wrap in a thin namespace). If `local_tool_impls`/`sensitive_paths` import loguru/config transitively, move the needed pure functions into modules with stdlib-only imports (extract, do not copy) until the closure test passes. Response encoding already stdlib (`_emit` uses json).

- [ ] **Step 4: Run worker test suites green, commit**

```bash
python3 -m pytest Tests/Tools/test_worker_import_closure.py Tests/Tools/test_workspace_tool_worker.py Tests/Tools/test_workspace_tool_protocol.py Tests/Tools/test_workspace_tool_executor.py -q
git add -A Tools/ Tests/Tools/
git commit -m "refactor: pinned worker import closure is stdlib-only (Phase 0c)"
```

---

### Task 5: Locator parsing, charsets, argv builder (Phase 1b)

**Files:**
- Create: `Tools/remote_binding_locator.py`
- Test: `Tests/Tools/test_remote_binding_locator.py`

**Interfaces:**
- Produces: `RemoteLocator(user, host, port, ipv6, path)` frozen dataclass; `parse_remote_locator(raw: str) -> RemoteLocator` (raises `RemoteLocatorError`); `locator_string(loc) -> str`; `build_ssh_argv(loc, options: Sequence[str], command: Sequence[str]) -> list[str]` — always `[*options, ("-p", str(port))?, ("-l", user)?, "--", host, *command]`, IPv6 brackets stripped.

- [ ] **Step 1: Failing tests**

```python
# Tests/Tools/test_remote_binding_locator.py
import pytest
from tldw_chatbook.Tools.remote_binding_locator import (
    RemoteLocatorError, build_ssh_argv, parse_remote_locator,
)

@pytest.mark.parametrize("raw,user,host,port", [
    ("ssh://devbox/srv/app", None, "devbox", None),
    ("ssh://me@devbox:2222/srv/app", "me", "devbox", 2222),
    ("ssh://[2001:db8::1]:2222/srv/app", None, "2001:db8::1", 2222),
])
def test_parse_ok(raw, user, host, port):
    loc = parse_remote_locator(raw)
    assert (loc.user, loc.host, loc.port) == (user, host, port)

@pytest.mark.parametrize("raw", [
    "ssh://devbox",                # no absolute path
    "ssh://devbox/~/x",            # ~ not allowed
    "ssh://-F./evil/srv/app",      # leading dash host
    "ssh://me$@devbox/srv/app",    # shell metachar
    "ssh://devbox:70000/srv/app",  # bad port
    "ssh://[::1]:22/srv/app extra",# trailing junk
])
def test_parse_rejects(raw):
    with pytest.raises(RemoteLocatorError):
        parse_remote_locator(raw)

def test_argv_components_never_mix():
    loc = parse_remote_locator("ssh://me@[2001:db8::1]:2222/srv/app")
    argv = build_ssh_argv(loc, ["-o", "BatchMode=yes"], ["python3", "-I"])
    assert argv == ["-o", "BatchMode=yes", "-p", "2222", "-l", "me",
                    "--", "2001:db8::1", "python3", "-I"]
```

- [ ] **Step 2: Run, verify failure. Step 3: Implement**

```python
# Tools/remote_binding_locator.py
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

class RemoteLocatorError(ValueError): ...

_USER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")
_V6_RE = re.compile(r"^[A-Za-z0-9:]+$")

@dataclass(frozen=True)
class RemoteLocator:
    user: str | None
    host: str
    port: int | None
    ipv6: bool
    path: PurePosixPath

def parse_remote_locator(raw: str) -> RemoteLocator:
    ...  # split scheme/authority/path; bracket-parse IPv6; per-component
         # charset checks (leading '-' rejected by the regexes' anchoring on
         # [A-Za-z0-9] first char); path must be absolute, no '..', no '~'
def build_ssh_argv(loc, options, command):
    argv = list(options)
    if loc.port is not None: argv += ["-p", str(loc.port)]
    if loc.user is not None: argv += ["-l", loc.user]
    return argv + ["--", loc.host, *command]
```

- [ ] **Step 4: Run green, commit** `feat: ssh locator parsing and argv builder (Phase 1b)`

---

### Task 6: ssh -G canonicalization + fingerprint (Phase 1c)

**Files:**
- Modify: `Tools/remote_binding_locator.py` (+`canonicalize_locator(loc) -> CanonicalTarget{hostname, port, user}` via `ssh -G`)
- Test: `Tests/Tools/test_remote_locator_canonicalize.py`

**Interfaces:**
- Produces: `canonicalize_locator(loc: RemoteLocator, *, ssh_bin: str = "ssh") -> CanonicalTarget` — runs `ssh -G [argv parts] -- host`, parses `hostname`, `port`, `user` from output, lowercases hostname. Never called at dispatch.

- [ ] **Step 1: Failing test with a fake ssh binary**

Write a fake `ssh` shell script into `tmp_path` printing fixed `hostname devbox.example.com\nport 2222\nuser me\n`; assert `canonicalize_locator` parses it, uses `-p/-l/--` argv form, and lowercases. Add a test that a failing `ssh -G` raises `RemoteLocatorError`.

- [ ] **Step 2: Run fail → Step 3: Implement** via `subprocess.run([ssh_bin, *parts_no_command, "-G"], capture_output=True, timeout=10)` — note `ssh -G` takes no command; strip it from the argv before appending `-G`. Wait — order: `ssh -G -p 2222 -l me -- devbox` (G before target; -G ignores command). Adjust builder or call shape accordingly and pin it in the test.

- [ ] **Step 4: Run green, commit** `feat: ssh -G canonical locator resolution (Phase 1c)`

---

### Task 7: Registry — SSH binding kind (Phase 1a)

**Files:**
- Modify: `Workspaces/models.py` (+`SSH_FILESYSTEM = "ssh-filesystem"` in `RuntimeBindingKind`), `Workspaces/registry_service.py`
- Test: `Tests/Workspaces/test_ssh_binding_registry.py`

**Interfaces:**
- Produces: `add_ssh_binding(workspace_id, raw_locator, *, allow_write=False) -> WorkspaceRuntimeBinding` (stores canonical fingerprint in `metadata["canonical_locator"]`, `access`, `python`; kind `ssh-filesystem`; default-workspace rejected; same-host overlap via `find_root_binding_conflict` against canonical host+path only), `list_ssh_bindings(workspace_id) -> list[...]` (status NOT recomputed here — the status cache owns it).

- [ ] **Step 1: Failing tests** — add on named workspace ok; default workspace rejected; duplicate/nested same-host rejected (`ssh://me@dev:2222/srv/app` vs `ssh://dev/srv/app/sub` — same canonical host); different hosts overlap allowed; no credentials persisted (metadata scrub test).
- [ ] **Step 2: Run fail. Step 3: Implement** — mirror `add_folder_binding` structure; canonicalize via Task 6; store fingerprint = sha256 of `canonical_locator + path`.
- [ ] **Step 4: Run green; also run `Tests/Workspaces/ -q` targeted; commit** `feat: ssh-filesystem runtime binding kind in registry (Phase 1a)`

---

### Task 8: Remote denylist + bundle builder + drift guard (Phase 1d)

**Files:**
- Create: `Tools/remote_sensitive_paths.py` (denylist data: `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.netrc` — relative to remote home)
- Create: `Tools/build_remote_worker_bundle.py`, `Tools/remote_worker_bundle.py` (generated)
- Test: `Tests/Tools/test_remote_worker_bundle.py`

**Interfaces:**
- Produces: `python -m tldw_chatbook.Tools.build_remote_worker_bundle` regenerates the bundle (concatenating: wire decode, root pin, local_tool_impls core, dispatch, remote denylist, IO adapter with magic strip + watchdog hooks). `Tools/remote_worker_bundle.py` exposes `main(stream)` taking the buffered stdin.

- [ ] **Step 1: Failing tests** — (a) drift guard: run the builder into a temp file, `assert result == Path("Tools/remote_worker_bundle.py").read_text()`; (b) charset: bundle contains only stdlib imports (`ast.walk` for Import nodes, assert each root is stdlib per `sys.stdlib_module_names`); (c) 3.10 syntax floor: `ast.parse` under a real 3.10 if `uv python find 3.10` succeeds, else skip with a loud marker (CI installs it).
- [ ] **Step 2: Run fail → Step 3: Implement** the builder as a source-transform script: read the module sources via `importlib` + `inspect.getsource` for the fixed module list, emit `main(stream)` that: reads size-prefixed... (bundle does NOT need the size prefix — the bootstrap already consumed N bytes; `main` receives the stream positioned after the bundle). IO adapter: `RESPONSE_MAGIC = b"TLDWWRK1"` (8 bytes; spec said 16 — use 16: `b"TLDW-REMOTE-WORKER-1"`), strip-before-parse helper `split_magic(raw)` used on responses; watchdog arming function `arm_watchdog(budget_seconds, temp_registry)` implementing Timer + `signal.alarm(budget + 2)` default-action; temp registry list with register/unregister.
- [ ] **Step 4: Run green, commit** `feat: remote worker bundle builder, drift guard, 3.10 floor (Phase 1d)`

---

### Task 9: Loopback executor + ping (Phase 1e)

**Files:**
- Create: `Tools/remote_workspace_executor.py` (loopback mode now; transport plugged in Task 11)
- Test: `Tests/Tools/test_remote_executor_loopback.py`

**Interfaces:**
- Produces: `RemoteWorkspaceToolExecutor.execute(tool: str, args: dict, intent: str) -> WorkspaceToolResponse`; `ping()` returning `{"identity_chain": [[path, dev, ino, mode], ...], "canonical_path": str, "python_version": "3.x.y", "bundle_sha256": "…"}`; `run_bundle_loopback(root: Path, request: dict) -> dict` (spawns `python3 -I -c <bootstrap-with-N>` with the compressed bundle + request JSON on stdin — the loopback harness used by tests and, later, wrapped by ssh).

- [ ] **Step 1: Failing tests** — loopback against a tmp dir: `fs_list` returns entries; `fs_read` returns content + sha256 + size (new response fields — extend wire response schema and both decoders, conformance corpus re-run); `ping` returns a chain covering root and ancestors; magic strip works when stdout is prefixed with `b"noise\nwith newlines\n" + magic`; request identity from a previous ping pins successfully; a stale identity → no admitted marker → error.
- [ ] **Step 2: Run fail → Step 3: Implement** executor + bundle `ping` op + identity-chain capture in the bundle (walk root→`/` with `os.stat`, emit list).
- [ ] **Step 4: Run green (incl. Tasks 2–3 suites), commit** `feat: loopback remote executor, ping with full identity chain (Phase 1e)`

---

### Task 10: SshMasterManager (Phase 2a)

**Files:**
- Create: `Tools/remote_workspace_transport.py`
- Modify: `config.py` (+`[console_ssh]`: `control_persist="10m"`, `enable_multiplexing=true`, `connect_timeout_s=3`, `max_concurrent_calls=8`), `app.py` (shutdown: `ssh -O exit` for all masters)
- Test: `Tests/Tools/test_ssh_master_manager.py`

**Interfaces:**
- Produces: `SshMasterManager(ssh_bin="ssh")` with `control_path_for(loc) -> Path` (short `0700` dir under the app-state dir, `%C` key, total path < 104 bytes — assert in test), `ensure_master(loc) -> None` (`ssh -MNf -o ControlPersist=<cfg> -o ServerAliveInterval=15 -o ServerAliveCountMax=2 ...` under a `per-host threading.Lock`, `start_new_session=True`), `restart_if_dead(loc)` (failure-triggered only), `close_all()`; `client_options(loc) -> list[str]` (BatchMode, ConnectTimeout, `ControlMaster=no`, ControlPath).

- [ ] **Step 1: Failing tests with fake ssh** — master started once under contention (10 threads → 1 `-MNf` invocation, counted via fake script log); dead-socket detection triggers exactly one restart on next `ensure_master`; `close_all` invokes `-O exit`; socket path length < 104 and dir mode 0700; `enable_multiplexing=false` → no master, no ControlPath options.
- [ ] **Step 2: Run fail → Step 3: Implement** → **Step 4: Run green, commit** `feat: ssh ControlMaster lifecycle manager (Phase 2a)`

---

### Task 11: RemoteWorkspaceTransport call path (Phase 2b)

**Files:**
- Modify: `Tools/remote_workspace_transport.py` (+`RemoteWorkspaceTransport.call`)
- Test: `Tests/Tools/test_remote_transport_call.py`

**Interfaces:**
- Consumes: `SshMasterManager` (Task 10), `build_ssh_argv` (Task 5), bundle bytes (Task 8).
- Produces: `RemoteWorkspaceTransport.call(loc, request_bytes, *, budget: float) -> RemoteCallResult(admitted: bool, response: bytes | None, failure: TransportFailure | None)` where `TransportFailure(kind, exit_code, reason)` with kinds: `UNREACHABLE`, `INTERPRETER_MISSING`, `PYTHON_TOO_OLD`, `WORKER_FAILED_TO_START`, `OP_TIMEOUT`, `REMOTE_OP_FAILED`, `STDOUT_NOISE`, `MUX_ERROR`. Deadlines: marker-arrival = spawn + budget + grace(5s); completion = `admitted_at + budget + grace`. Exit 76 parses found version from stderr line `tldw-worker:python3.10+:found:X.Y`.

- [ ] **Step 1: Failing tests with fake ssh** covering every taxonomy row: no-marker 255 → UNREACHABLE; 127 → INTERPRETER_MISSING; exit 76 + stderr line → PYTHON_TOO_OLD("3.9.7"); exit 1 no marker → WORKER_FAILED_TO_START; admitted then exit 75 → OP_TIMEOUT; admitted then 255 → REMOTE_OP_FAILED; 5KB noise no magic → STDOUT_NOISE; newline-containing noise before magic → response parsed, no injected lines.
- [ ] **Step 2: Run fail → Step 3: Implement** — spawn via `subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, start_new_session=True)`; write `len(bundle)` bootstrap (exact spec string with N literal) — the executor builds `[ssh_bin, *client_options, *argv_parts, python, "-I", "-c", BOOTSTRAP]`; feed compressed bundle + request + `stdin.close()`; read stdout with magic scan (raw bytes, cap 4KB pre-magic); watch for admitted line; enforce the two deadlines with `selectors`/timed reads; kill process group on deadline.
- [ ] **Step 4: Run green, commit** `feat: ssh transport call path with admitted-marker bucketing (Phase 2b)`

---

### Task 12: Worker watchdog tiers (Phase 2c)

**Files:**
- Modify: `Tools/build_remote_worker_bundle.py` (+watchdog, temp registry; regenerate bundle)
- Test: `Tests/Tools/test_remote_watchdog.py`

**Interfaces:**
- Consumes: `timeout_seconds` in the request (executor now sends remaining budget = `WORKSPACE_HELPER_TIMEOUT_SECONDS - elapsed_at_spawn`).
- Produces: bundle arms `threading.Timer(budget, _watchdog)` + `signal.alarm(budget + 2)`; `_watchdog` unlinks registered temp paths, writes stderr marker `tldw-worker-watchdog`, `os._exit(75)`; local_tool_impls atomic writes register/unregister temp paths.

- [ ] **Step 1: Failing tests** — loopback with budget 1s and an `fs_grep` op whose pattern is `(a+)+$` against a 60KB line of `a`s (GIL starvation): process exits ≤ 3s (alarm), no orphan (poll pid), temp file created by a concurrent write registration is unlinked when the Timer tier fires (budget 1s, op sleeps 2s — use a test-only `sleep` op).
- [ ] **Step 2: Run fail → Step 3: Implement + regenerate bundle → Step 4: Run green (drift guard too), commit** `feat: two-tier worker watchdog with temp cleanup (Phase 2c)`

---

### Task 13: Status cache + recovery (Phase 2d)

**Files:**
- Create: `Tools/remote_binding_status.py`
- Test: `Tests/Tools/test_remote_binding_status.py`

**Interfaces:**
- Produces: `RemoteBindingStatusCache` — `status(binding_id) -> BindingStatus` (optimistic READY when unseen), `record_transport_failure(binding_id, kind, reason)` → BLOCKED(reason), `record_pin_failure(binding_id)` → STALE_IDENTITY + probe schedule, `record_success(binding_id, identity_chain)` → READY(chain), `probe_scheduled(host_key) -> bool` + `mark_probe_dispatched(host_key)` (debounce 30s), `identity_for(binding_id)`. Pure in-memory; thread-safe.

- [ ] **Step 1: Failing tests** — optimistic default; transport failure flips; OP_TIMEOUT does NOT flip; pin failure → STALE_IDENTITY not BLOCKED; success after stale re-captures chain; debounce: two schedules within 30s → one probe; different hosts → independent.
- [ ] **Step 2–4: implement, green, commit** `feat: remote binding status cache with debounced recovery (Phase 2d)`

---

### Task 14: Executor wiring + fake-ssh integration suite (Phase 2e)

**Files:**
- Modify: `Tools/remote_workspace_executor.py` (transport mode: master manager, per-host semaphore `max_concurrent_calls`, budget computation, status-cache recording)
- Test: `Tests/Tools/test_remote_executor_ssh.py` (fake `ssh` end-to-end)

**Interfaces:**
- Produces: the final `RemoteWorkspaceToolExecutor(loc, cache, masters, cfg)` used by Task 16.

- [ ] **Step 1: Failing integration tests** — happy path `fs_read` via fake ssh (script echoes admitted line + response JSON prefixed with magic after optional noise); catastrophic regex → OP_TIMEOUT + status unchanged + remote child gone (fake script records child pid); host down → BLOCKED + next call after probe-success re-admitted; master restart after dead socket; concurrent 10 calls → ≤ `max_concurrent_calls` concurrent ssh invocations (fake counts).
- [ ] **Step 2–4: implement, green, commit** `feat: ssh executor wiring + fake-ssh integration suite (Phase 2e)`

---

### Task 15: LocalRoot | RemoteRoot + remote request builder (Phase 3a)

**Files:**
- Create: `Tools/remote_root_types.py` (`LocalRoot(path: Path)`, `RemoteRoot(alias, canonical_locator, root: PurePosixPath)` — union type `AdmittedRoot`)
- Modify: `Agents/local_tool_provider.py:389-417` (`RunAdmittedWorkspaceRoot.root: AdmittedRoot`), `Tools/workspace_tool_executor.py::_build_request` (dispatch on root type), `Tools/workspace_file_roots.py::_binding_matches_frozen_authority` (LocalRoot only)
- Test: `Tests/Tools/test_admitted_root_types.py`, extend `Tests/Agents/test_local_tool_provider.py`

**Interfaces:**
- Produces: type boundary — every `root: Path` consumer either handles `RemoteRoot` or raises `TypeError("remote root reached laptop-disk path")`. Remote `_build_request`: lexical normalization only (reject `..`, hidden paths), identity from `cache.identity_for(binding_id)`.

- [ ] **Step 1: Failing tests** — passing a `RemoteRoot` into an un-migrated consumer raises TypeError (not silent laptop IO); remote request builder produces a request whose `root_identity` equals the cached chain; `..` and hidden paths rejected lexically.
- [ ] **Step 2–4: implement the union + migrate `_build_request` + the two guards, targeted suites green (`Tests/Tools/test_workspace_tool_executor.py Tests/Agents/test_local_tool_provider.py`), commit** `feat: LocalRoot|RemoteRoot type boundary (Phase 3a)`

---

### Task 16: CAS from worker responses ×4 (Phase 3b)

**Files:**
- Modify: `Agents/local_tool_provider.py` — `_record_fs_read_observation`, `_fs_write_guard_injection`, `_stale_targets_for`, `_update_ledger_after_write` (~2356), `_stale_write_refusal` (~2248); bundle: `fs_read`/write responses gain `sha256`, `size` fields (Task 9 added fs_read's).
- Test: extend `Tests/Agents/test_local_tool_provider.py`; new `Tests/Tools/test_wrong_file_hazard.py`

**Interfaces:**
- Produces: all five sites take stamps from worker-reported `sha256`/`size` when the root is `RemoteRoot`; `_hash_file` is LocalRoot-only.

- [ ] **Step 1: Failing tests** — **named wrong-file hazard test**: remote root `/tmp/w` (PurePosix), laptop `/tmp/w` exists with different content; `fs_read` via fake-ssh returns remote bytes/hash; assert ledger stamp == worker sha256, laptop file never opened (monkeypatch `builtins.open` / `pathlib` to blow up on that path); refusal message "now" value from worker hash.
- [ ] **Step 2–4: migrate each of the five sites (same pattern: `if isinstance(root, RemoteRoot): stamp = worker_values else: _hash_file(...)`), green, commit** `feat: CAS stamps from worker responses, wrong-file hazard closed (Phase 3b)`

---

### Task 17: Exclusions, redaction, admission (Phase 3c)

**Files:**
- Modify: `Chat/console_chat_controller.py` (`_exclusion_paths_provider`, `_project_instruction_excluded_dirs`, `_validate_project_instruction_binding`, `_workspace_binding_authority_is_current`), `Workspaces/registry_service.py` (`add_binding_exclusion` accepts ssh bindings, stores raw relative), `Agents/local_tool_provider.py` (`redaction_root`), `Agents/virtual_cli_provider.py` (record conclusion at site), `Tools/workspace_tool_executor.py` (`_call_context` — dev:148)
- Test: extend `Tests/Chat/test_console_project_instructions.py`, `Tests/Workspaces/test_folder_binding_validator.py`

**Interfaces:**
- Produces: exclusions ride the request as relative strings (worker matches them — bundle gains the matcher, corpus re-run); remote admission = registry read + cache status (+ sync ping at first selection); remote `_call_context.is_file` via executor stat; redaction against `RemoteRoot` paths.

- [ ] **Step 1: Failing tests** — macOS symlink hazard: remote root `/var/www/site` (laptop `/var` → `/private/var`), exclusions still serialized and matched worker-side (loopback); ssh binding selectable as working folder when cache READY; BLOCKED selected binding → degraded warning, no forced re-selection; virtual CLI: assert or document no-remote-root conclusion in a comment+test.
- [ ] **Step 2–4: implement, targeted Chat/Workspaces suites green, commit** `feat: worker-side exclusions, remote admission, redaction (Phase 3c)`

---

### Task 18: Run composition + availability tests (Phase 4a)

**Files:**
- Modify: `Chat/console_chat_controller.py::capture_run_admitted_workspace_roots`, `Tools/workspace_file_roots.py::allowed_file_roots`/`workspace_context_note`
- Test: `Tests/Chat/test_console_ssh_composition.py`

**Interfaces:**
- Produces: SSH bindings (cache READY) admitted with `RemoteWorkspaceToolExecutor`; BLOCKED/MISSING excluded + note line; recovery probe scheduled at composition (never awaited); git_* on remote alias → call-time typed error.

- [ ] **Step 1: Failing tests — the three named availability regression tests** (spec Testing #6): (a) remote BLOCKED + local READY → composes, local advertised, remote excluded, note mentions it; (b) op timeout → typed error, next send admits; (c) BLOCKED + probe success → re-admitted next send, no user action. Plus: no subprocess spawned during composition (monkeypatch subprocess — hot-path rule), URI-in-path rejection message teaches `root_alias`.
- [ ] **Step 2–4: implement, green, commit** `feat: ssh roots in run composition, availability both directions (Phase 4a)`

---

### Task 19: AGENTS.md remote reads (Phase 4b)

**Files:**
- Modify: `Agents/project_instruction_resolver.py` (IO seam: `read_candidate` strategy local-fd vs executor bounded-read; `BindingRootIdentity` from worker chain)
- Test: extend `Tests/Agents/test_project_instruction_resolver.py`

**Interfaces:**
- Consumes: executor bounded-read op (bytes + stat in one call — added to bundle in Task 9's pattern).
- Produces: `resolve_startup`/`resolve_targets` over a `RemoteRoot` via executor; all caps/ledger/consent unchanged; unreachable-remote prep failure → content-free warning, proceed (ADR-069 posture).

- [ ] **Step 1: Failing tests** — remote AGENTS.override.md precedence; byte-cap enforcement via loopback executor; unreachable → warning + proceed; lazy nested scopes activate before remote ops.
- [ ] **Step 2–4: implement, green (incl. `Tests/Chat/test_console_agent_project_instructions.py`), commit** `feat: project instructions from remote bindings (Phase 4b)`

---

### Task 20: UI surfaces (Phase 5a)

**Files:**
- Modify: `UI/Screens/settings_screen.py` (add-SSH form in Workspaces pane; status column `unreachable`/`missing on host`, async probe refresh via `@work(thread=True)`), `UI/Console_Modules/session.py` (picker), `UI/Console_Modules/workspace.py` (switcher chips)
- Test: `Tests/UI/test_settings_ssh_bindings.py`

**Interfaces:**
- Produces: "Add SSH folder…" form (target, path, ro/rw, interpreter override); advisory add-time probe (save always allowed with reason shown); bindings listed with live status chips. Design tokens per ADR-150 (`$ds-*` only — no new literals; governance test must stay green).

- [ ] **Step 1: Failing tests** — form renders, submits `add_ssh_binding`; advisory probe failure still saves with status reason; status column states; UI never blocks on probe (probe runs in worker thread).
- [ ] **Step 2–4: implement, run `Tests/UI/test_design_token_governance.py` + new tests, commit** `feat: SSH binding UI in Settings and Console pickers (Phase 5a)`

---

### Task 21: Docs, startup check, live verification (Phase 5b)

**Files:**
- Modify: `config.py` or app startup (clear message if no `ssh` binary — feature disabled), `Docs/User_Guide/console/context-and-rag.md`, repo `AGENTS.md` (Special Systems bullet)
- Test: `Tests/test_ssh_feature_availability.py`

- [ ] **Step 1: Startup check test** — `shutil.which("ssh")` None → feature flag off with message; present → on.
- [ ] **Step 2: Implement + docs** — user-guide section (add binding, prerequisites, ro/rw, degradation behavior); AGENTS.md bullet: SSH bindings reuse the Console file authority model; remote is fs_*-only, family B local-only.
- [ ] **Step 3: Live verification** (per `backlog/docs/lessons-live-verification.md`) — run the app against the user's real server: add binding, fs_read/fs_write/fs_grep, kill network mid-run, watch recovery on next send. Record evidence in the task Implementation Notes.
- [ ] **Step 4: Backlog hygiene + commit**

```bash
backlog task edit <id> --notes "Live verification passed: <evidence>" -s Done
git add -A && git commit -m "docs: SSH remote workspace bindings user guide + availability check (Phase 5b)"
```

---

## Self-Review

**Spec coverage:** Requirements → Tasks: fs parity (9, 14–16), AGENTS.md (19), availability ×3 directions (13, 18), nothing-left-behind (12), no credentials (5, 7, 10), python floor (8, 11). Decision-summary prerequisites: protocol split (2–4), root type (15), timeout split (11–13). Transport: master lifecycle (10), bootstrap/taxonomy (11), watchdog (12), recovery (13), concurrency (14). Relocations: request building (15), CAS ×4 + refusal (16), exclusions/redaction/_call_context/admission (17). UX/config (20, 21). ADR-first (1). All six phases present.

**Known deliberate abbreviations:** Tasks 2, 8, 11 mark three places where the implementer must read the referenced source first and mirror it exactly (`_validate_types` field checks; bundle module list; valid-request fixture) — the conformance, drift, and taxonomy tests are the gate that catches wrong mirrors. Tests named in the spec (wrong-file hazard, catastrophic regex, availability ×3, starvation) all appear as explicit test steps.

**Type consistency:** `RemoteLocator`/`build_ssh_argv` (5→6→10→11), `decode_request` (2→3→4→8), `RemoteCallResult`/`TransportFailure` (11→13→14), `AdmittedRoot` union (15→16→17→18), `RemoteBindingStatusCache` (13→17→18).
