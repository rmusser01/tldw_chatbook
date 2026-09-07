# Reference-File Reachability (`skill_file` runtime tool) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let agents read a skill's bundled reference files on demand: a `skill_file(skill_name, path)` runtime tool available to model-invoked skill forks and to native-tools Console turns driven by `$skill` invocations.

**Architecture:** One trust-gated, contained read seam in Skills_Interop; `skill_file` becomes the fourth RUNTIME tool (schema pinned via `runtime_schemas`, name-branch dispatch in `run_agent_loop`, authorization via a per-run mutable `SkillFileBindings` seeded from the turn's `$skill` names and extended when `SkillRunner` spawns a skill child). A "Bundled files" block is rendered only where the tool's existence is certain (bridge `run_reply` + `SkillRunner.run`).

**Tech Stack:** Python 3.11+, pytest. No new dependencies.

**Design doc:** `Docs/superpowers/specs/2026-07-23-skills-reference-file-reachability-design.md` (governs on any ambiguity).

## Global Constraints

- **Read seam order (verbatim from spec §1):** `_enforce("skills.read_file.launch.local")` → `_require_trusted_skill(skill_name)` (per-READ trust re-verification; raises `SkillTrustBlockedError`) → `validate_supporting_file_path(relative_path)` → contained read via the `_read_text_preserving_newlines` discipline → binary = clean refusal string, never raises → cap **100,000 chars** with `truncated: True` + trailing marker.
- **Policy registry entry is REQUIRED** (`_resource("skills.read_file", actions=(LAUNCH,))` in the `server_skills` capability block) — the engine FAILS CLOSED on unknown action ids.
- **Runtime-tool pattern:** schema pinned into `runtime_schemas` when bindings non-empty (active from step 1, never disclosure-gated); dispatch via a name-branch in `run_agent_loop` BEFORE `deps.invoke_tool`; authorization = membership in `SkillFileBindings.authorized`, NOT `config.allowed_tools`. `agent_runtime.py` must NOT import Skills_Interop.
- **Bindings:** per-run, mutable; seeded from `turn_skill_bindings` (leading-resolved + embedded-SPLICED names only — never blocked/fork-literal mentions); `_BridgeSkillRunner.run` adds the spawned skill's name BEFORE `spawn`. Own bundle always readable — independent of frontmatter `allowed-tools`.
- **"Bundled files" block** renders ONLY bridge-side (`run_reply`, trailing user message of its OWN `agent_messages` copy) and in `SkillRunner.run` (spawned body) — never at substitution time, never in the stored transcript, never on plain sends.
- **Collision:** `SKILL_FILE_TOOL_NAME` joins `RUNTIME_TOOL_NAMES` (agent_models.py:33) — the existing consumers (`_non_colliding_skill_entries`, bridge composition) then exclude a skill literally named `skill_file` automatically.
- **Sync everywhere:** the agent runtime runs on a worker thread with no event loop; the reader closure wraps the async scope-service call with `asyncio.run(...)` exactly like `BuiltinToolProvider.invoke` / `_BridgeSkillRunner.run` do.
- **Tests:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q`; Skills/UI suites need `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`. Known baselines: `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_through_untouched` (pre-existing); `Tests/Skills/test_skills_library_flow.py::test_skill_editor_canvas_scrolls_trust_panel_into_view` (flaky — re-run in isolation, don't chase).
- **Commit hygiene:** `git add` ONLY named files — NEVER `git add -A` (tracked scratch `.superpowers/sdd/progress.md` must stay out; SUBAGENTS MUST NOT `git checkout` it).
- Line anchors below were verified on this branch at `fbb17d81d` — re-grep before editing; they shift.

## File Structure

| File | Change |
|------|--------|
| `tldw_chatbook/runtime_policy/registry.py` (~:1019) | `_resource("skills.read_file", actions=(LAUNCH,))` |
| `tldw_chatbook/Skills_Interop/local_skills_service.py` | `read_skill_file`; `execute_skill` gains `reference_files` |
| `tldw_chatbook/Skills_Interop/skills_scope_service.py` | local-only `read_skill_file` passthrough |
| `tldw_chatbook/tldw_api/skills_schemas.py` | `SkillExecutionResult.reference_files` (additive) |
| `tldw_chatbook/Agents/agent_models.py` | `SKILL_FILE_TOOL_NAME`, join `RUNTIME_TOOL_NAMES`, `SkillFileBindings` |
| `tldw_chatbook/Agents/tool_catalog.py` | `SKILL_FILE_TOOL_SCHEMA` |
| `tldw_chatbook/Agents/agent_service.py` | ctor kwarg, runtime_schemas pin, reader closure into LoopDeps |
| `tldw_chatbook/Agents/agent_runtime.py` | `LoopDeps.read_skill_file` field + name-branch dispatch |
| `tldw_chatbook/Chat/console_agent_bridge.py` | bindings per run_reply; SkillRunner grant + block; bridge-side block; `turn_skill_bindings` param |
| `tldw_chatbook/Chat/console_chat_controller.py` | 4-tuple widening + funnel threading |
| Tests | `Tests/Skills/test_read_skill_file.py`, `Tests/Agents/test_skill_file_runtime_tool.py`, extensions to `Tests/Chat/test_console_skill_substitution.py` + `Tests/Agents/test_skill_tool_spawn.py` |

---

### Task 1: Policy entry + the read seam

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py` (the `server_skills` capability block, `_resource("skills.execute", ...)` is at ~:1019); `tldw_chatbook/Skills_Interop/local_skills_service.py`; `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Test: `Tests/Skills/test_read_skill_file.py` (create)

**Interfaces:**
- Produces: `LocalSkillsService.read_skill_file(skill_name: str, relative_path: str) -> dict` returning `{"content": str, "truncated": bool, "size": int}` (sync-callable via `await` — it is `async def` for service-interface symmetry but does no awaiting I/O); `SkillsScopeService.read_skill_file(skill_name, relative_path, *, mode=None)` — local-only, `ValueError("skill_file reads are local-only")` for server mode; constant `SKILL_FILE_READ_CAP_CHARS = 100_000` in `local_skills_service.py`.

- [ ] **Step 1: Write the failing tests** (create `Tests/Skills/test_read_skill_file.py`)

```python
import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import (
    SKILL_FILE_READ_CAP_CHARS,
    LocalSkillsService,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _svc(tmp_path):
    return LocalSkillsService(store_dir=tmp_path)


async def _make_skill(svc, name="demo"):
    await svc.create_skill(name=name, content=f"---\nname: {name}\n---\nbody\n")
    d = svc._skill_dir(name)
    (d / "references").mkdir(parents=True, exist_ok=True)
    (d / "references" / "api.md").write_text("# api docs\n", encoding="utf-8")
    (d / "assets").mkdir(exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")
    return d


@pytest.mark.asyncio
async def test_read_happy_path_nested(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    out = await svc.read_skill_file("demo", "references/api.md")
    assert out == {"content": "# api docs\n", "truncated": False, "size": len("# api docs\n")}


@pytest.mark.asyncio
async def test_read_traversal_and_bad_paths_rejected(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    for bad in ("../escape.md", "/abs.md", "refs/SKILL.md"):
        with pytest.raises(ValueError):
            await svc.read_skill_file("demo", bad)


@pytest.mark.asyncio
async def test_read_binary_returns_refusal_not_bytes(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    out = await svc.read_skill_file("demo", "assets/logo.png")
    assert out["truncated"] is False
    assert "binary file" in out["content"]
    assert "\x89" not in out["content"]


@pytest.mark.asyncio
async def test_read_truncates_over_cap(tmp_path):
    svc = _svc(tmp_path)
    d = await _make_skill(svc)
    (d / "references" / "big.md").write_text("x" * (SKILL_FILE_READ_CAP_CHARS + 500), encoding="utf-8")
    out = await svc.read_skill_file("demo", "references/big.md")
    assert out["truncated"] is True
    assert len(out["content"]) < SKILL_FILE_READ_CAP_CHARS + 200  # cap + marker line
    assert "truncated" in out["content"].rsplit("\n", 1)[-1]


@pytest.mark.asyncio
async def test_read_untrusted_raises_blocked(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    await _make_skill(svc)

    def _deny(name):
        raise SkillTrustBlockedError(
            skill_name=name, reason_code="skill_modified", trust_status="quarantined_modified"
        )

    monkeypatch.setattr(svc, "_require_trusted_skill", _deny)
    with pytest.raises(SkillTrustBlockedError):
        await svc.read_skill_file("demo", "references/api.md")


@pytest.mark.asyncio
async def test_read_missing_file_clean_error(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    with pytest.raises(ValueError, match="local_skill_file_not_found"):
        await svc.read_skill_file("demo", "references/nope.md")


@pytest.mark.asyncio
async def test_scope_service_server_mode_rejected(tmp_path):
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    scope = SkillsScopeService(local_service=_svc(tmp_path), server_service=None)
    with pytest.raises(ValueError, match="local-only"):
        await scope.read_skill_file("demo", "references/api.md", mode="server")


def test_policy_action_id_registered():
    # The engine denies unknown ids (fail-closed) — pin that the new id exists.
    from tldw_chatbook.runtime_policy.registry import build_default_registry  # adjust to the module's actual builder name after reading it

    registry = build_default_registry()
    assert "skills.read_file.launch.local" in registry
```

(For the policy test: READ `runtime_policy/registry.py` first and use its actual public builder/lookup — the assertion contract is "the concrete action id `skills.read_file.launch.local` is registered"; adapt the access idiom to whatever `skills.execute.launch.local` is reachable through, e.g. the same structure existing policy tests in `Tests/` use — grep `skills.execute.launch` in `Tests/` and mirror.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_read_skill_file.py -q`
Expected: FAIL — `ImportError: cannot import name 'SKILL_FILE_READ_CAP_CHARS'`.

- [ ] **Step 3: Implement**

`registry.py` — inside the `server_skills` capability's `resources=(...)` tuple, after the `skills.execute` line:

```python
            _resource("skills.read_file", actions=(LAUNCH,)),
```

`local_skills_service.py` — add near the other module constants: `SKILL_FILE_READ_CAP_CHARS = 100_000`, then the method (place near `execute_skill`):

```python
    async def read_skill_file(
        self, skill_name: str, relative_path: str
    ) -> dict[str, Any]:
        """Read one bundled supporting file of a trusted skill, contained + capped.

        The runtime `skill_file` tool's single backing seam. Order is
        load-bearing: policy gate, per-READ trust re-verification (a skill
        revoked mid-run stops being readable immediately), path validation,
        then the same containment discipline `_read_text_preserving_newlines`
        already applies to the skill body.

        Args:
            skill_name: Canonical skill name.
            relative_path: POSIX relative path within the skill's bundle.

        Returns:
            ``{"content", "truncated", "size"}``; a binary file yields a
            clean refusal string as ``content`` (never bytes, never raises).

        Raises:
            SkillTrustBlockedError: Skill not currently trusted.
            ValueError: Bad path, unknown skill, or missing file
                (``local_skill_file_not_found:...``).
        """
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import validate_supporting_file_path

        self._enforce("skills.read_file.launch.local")
        self._require_trusted_skill(skill_name)
        validate_supporting_file_path(relative_path)
        skill_dir = self._skill_dir(skill_name)
        if not skill_dir.is_dir():
            raise ValueError(f"local_skill_not_found:{skill_name}")
        path = skill_dir / PurePosixPath(relative_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"local_skill_file_not_found:{relative_path}")
        raw_size = path.stat().st_size
        try:
            text = self._read_text_preserving_newlines(path, base_dir=skill_dir)
        except UnicodeDecodeError:
            return {
                "content": f"binary file — {raw_size} bytes; not readable as text",
                "truncated": False,
                "size": raw_size,
            }
        if "\x00" in text:
            return {
                "content": f"binary file — {raw_size} bytes; not readable as text",
                "truncated": False,
                "size": raw_size,
            }
        if len(text) > SKILL_FILE_READ_CAP_CHARS:
            text = (
                text[:SKILL_FILE_READ_CAP_CHARS]
                + f"\n[truncated — file is {raw_size} bytes; showing first {SKILL_FILE_READ_CAP_CHARS} characters]"
            )
            return {"content": text, "truncated": True, "size": raw_size}
        return {"content": text, "truncated": False, "size": raw_size}
```

(`PurePosixPath` is already imported in this module; verify.) `skills_scope_service.py` — bespoke local-only dispatch mirroring `count_skills` (:174-200):

```python
    async def read_skill_file(
        self,
        skill_name: str,
        relative_path: str,
        *,
        mode: SkillsBackend | str | None = None,
    ) -> dict[str, Any]:
        """Read a bundled file of a LOCAL trusted skill (runtime skill_file seam).

        Local-only by design: the server backend has no read_skill_file and
        every runtime skill path is already hardcoded local. Rejecting server
        mode here beats surfacing a raw AttributeError from _call.
        """
        normalized_mode = self._normalize_mode(mode) if mode is not None else SkillsBackend.LOCAL
        if normalized_mode is not SkillsBackend.LOCAL:
            raise ValueError("skill_file reads are local-only")
        service = self._require_service(SkillsBackend.LOCAL)
        self._enforce_policy("skills.read_file.launch.local")
        return await self._maybe_await(service.read_skill_file(skill_name, relative_path))
```

(Match `_normalize_mode`/`_require_service`/`_enforce_policy`/`_maybe_await` exactly to the file's existing idioms — read `count_skills` first.)

- [ ] **Step 4: Run to verify pass**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_read_skill_file.py Tests/Skills/ -q`
Expected: new file PASS; full Tests/Skills green (modulo known flaky).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/registry.py tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/Skills_Interop/skills_scope_service.py Tests/Skills/test_read_skill_file.py
git commit -m "feat(skills): trust-gated contained read_skill_file seam + policy entry"
```

---

### Task 2: `reference_files` metadata on execute_skill

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py` (`execute_skill`, ~:1197); `tldw_chatbook/tldw_api/skills_schemas.py` (`SkillExecutionResult`, :137-145)
- Test: `Tests/Skills/test_read_skill_file.py` (extend)

**Interfaces:**
- Produces: `execute_skill` result dict gains `reference_files: list[{"path": str, "size": int, "is_text": bool}] | None` (from `_read_bundle_manifest`, dropping the `executable` field — not needed for reads). `SkillExecutionResult` gains the matching optional field.

- [ ] **Step 1: Failing test** (append)

```python
@pytest.mark.asyncio
async def test_execute_skill_carries_reference_files(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    svc.allow_untrusted_without_trust_service = True  # match existing execute tests' trust setup; READ the file's existing execute_skill tests and mirror their trust arrangement instead if different
    result = await svc.execute_skill("demo")
    refs = {r["path"]: r for r in (result.get("reference_files") or [])}
    assert refs["references/api.md"]["is_text"] is True
    assert refs["assets/logo.png"]["is_text"] is False
```

(READ the existing `execute_skill` tests first — `grep -rn "execute_skill" Tests/Skills/` — and mirror their trust setup exactly; the assertion contract is the field's presence and shape.)

- [ ] **Step 2: RED** — field absent. **Step 3: Implement**: in `execute_skill`, after building the result fields, add `reference_files=` derived from `self._read_bundle_manifest(self._skill_dir(skill["name"]))`, mapping each entry to `{"path", "size", "is_text"}` (or `None` when the manifest is `None`); add `reference_files: list[dict] | None = None` to `SkillExecutionResult` (it has `extra="allow"`, but declare explicitly). **Step 4: GREEN** + `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv-path pytest Tests/Skills/ Tests/Chat/test_console_skill_substitution.py -q` (substitution consumers must be unaffected). **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/tldw_api/skills_schemas.py Tests/Skills/test_read_skill_file.py
git commit -m "feat(skills): execute_skill carries reference_files bundle metadata"
```

---

### Task 3: Runtime core — bindings, schema pin, dispatch branch

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (:30-33 names block + new dataclass); `tldw_chatbook/Agents/tool_catalog.py` (schemas near :28-58); `tldw_chatbook/Agents/agent_service.py` (ctor :162-175, runtime_schemas :340-344, LoopDeps build :547-561); `tldw_chatbook/Agents/agent_runtime.py` (LoopDeps :199-230, dispatch :416-521)
- Test: `Tests/Agents/test_skill_file_runtime_tool.py` (create)

**Interfaces:**
- Consumes: nothing from Skills_Interop directly (layering: the reader is an injected callable).
- Produces: `SKILL_FILE_TOOL_NAME = "skill_file"` (in `RUNTIME_TOOL_NAMES`); `SkillFileBindings` (mutable dataclass: `authorized: set[str]`, `reader: Callable[[str, str], dict] | None`); `SKILL_FILE_TOOL_SCHEMA` (ToolSchema; params: `skill_name` str required, `path` str required); `AgentService(..., skill_file_bindings: SkillFileBindings | None = None)`; `LoopDeps.read_skill_file: Callable[[str, str], ToolResult] | None = None`; loop branch: when `call.name == SKILL_FILE_TOOL_NAME` and `deps.read_skill_file` is set → dispatch to it (bindings/authorization enforced INSIDE the service-built closure, not the loop).

- [ ] **Step 1: Failing tests** (create; model the harness on `Tests/Agents/test_skill_tool_spawn.py` — READ it first and reuse its fake chat_call/registry scaffolding idioms)

```python
from tldw_chatbook.Agents.agent_models import (
    RUNTIME_TOOL_NAMES,
    SKILL_FILE_TOOL_NAME,
    SkillFileBindings,
)


def test_skill_file_is_a_runtime_tool_name():
    assert SKILL_FILE_TOOL_NAME == "skill_file"
    assert SKILL_FILE_TOOL_NAME in RUNTIME_TOOL_NAMES  # collision exclusion rides existing consumers


def test_bindings_object_shape():
    b = SkillFileBindings(authorized=set(), reader=None)
    b.authorized.add("demo")
    assert "demo" in b.authorized
```

Plus two loop-level tests (using the file's established fake-model scaffolding): (a) with `skill_file_bindings=SkillFileBindings(authorized={"demo"}, reader=<fake returning {"content": "REF", "truncated": False, "size": 3}>)` and a scripted model turn calling `skill_file(skill_name="demo", path="references/api.md")` → the tool result contains `REF`, and the provider call's schema list INCLUDES the `skill_file` schema on the FIRST turn; (b) same but calling `skill_name="other"` (not authorized) → `ToolResult(ok=False)` with a refusal mentioning `other`, and with `skill_file_bindings=None` → the schema is NOT offered and a call to the name falls through to normal unknown-tool handling. Write these concretely against the harness you find; the assertions above are the contract.

- [ ] **Step 2: RED** (ImportError). **Step 3: Implement**:

`agent_models.py` (names block):

```python
SKILL_FILE_TOOL_NAME = "skill_file"
RUNTIME_TOOL_NAMES = frozenset({SPAWN_TOOL_NAME, FIND_TOOLS_NAME, LOAD_TOOLS_NAME, SKILL_FILE_TOOL_NAME})


@dataclass
class SkillFileBindings:
    """Per-run authorization + reader for the skill_file runtime tool.

    Mutable by design: seeded with the turn's $skill names; SkillRunner adds
    each spawned skill's name before spawn so a skill can always read its own
    bundle. Authorization lives here, never in config.allowed_tools.
    """

    authorized: set[str]
    reader: Callable[[str, str], dict] | None = None
```

`tool_catalog.py` (beside the other runtime schemas):

```python
SKILL_FILE_TOOL_SCHEMA = ToolSchema(
    id="runtime:skill_file",
    name=SKILL_FILE_TOOL_NAME,
    description=(
        "Read a bundled reference file of a skill active in this run. "
        "Args: skill_name (the skill whose bundle to read), path (relative "
        "POSIX path, e.g. references/api.md). Text files only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["skill_name", "path"],
    },
)
```

`agent_service.py`: ctor gains `skill_file_bindings: SkillFileBindings | None = None` stored as `self.skill_file_bindings` (keyword-only, beside `skill_runner`). In `_run_one`'s runtime_schemas block (:340-344):

```python
        if self.skill_file_bindings is not None and self.skill_file_bindings.authorized:
            runtime_schemas.append(SKILL_FILE_TOOL_SCHEMA)
```

Build the reader closure beside `invoke_tool` and thread into LoopDeps:

```python
        def read_skill_file_tool(skill_name: str, path: str) -> ToolResult:
            bindings = self.skill_file_bindings
            if bindings is None or skill_name not in bindings.authorized:
                return ToolResult(ok=False, error=f"skill_file: '{skill_name}' is not active in this run")
            if bindings.reader is None:
                return ToolResult(ok=False, error="skill_file: no reader configured")
            try:
                out = bindings.reader(skill_name, path)
            except Exception as exc:  # SkillTrustBlockedError, ValueError, OSError
                return ToolResult(ok=False, error=f"skill_file: {exc}")
            return ToolResult(ok=True, content=str(out.get("content", "")))
```

`agent_runtime.py`: `LoopDeps` gains `read_skill_file: Callable[..., ToolResult] | None = None`; in the dispatch chain (before the final `else` at ~:519), add:

```python
                elif call.name == SKILL_FILE_TOOL_NAME and deps.read_skill_file is not None:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.read_skill_file(
                        str(call.args.get("skill_name", "")), str(call.args.get("path", ""))
                    )
```

then let it flow into the existing shared tool-result handling exactly as the `deps.invoke_tool` branch does (READ the surrounding code; mirror how `result` is recorded/fed back — do not duplicate logic). Import `SKILL_FILE_TOOL_NAME` from agent_models (agent_runtime already imports from there).

- [ ] **Step 4: GREEN** + regression `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ -q` (all pass — RUNTIME_TOOL_NAMES growth may affect collision tests; if a pre-existing test enumerates the frozenset's exact members, update it — intended). **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_skill_file_runtime_tool.py
git commit -m "feat(agents): skill_file fourth runtime tool with per-run bindings"
```

---

### Task 4: Bridge fork side — bindings wiring, own-bundle grant, spawned-body block

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (run_reply AgentService ctor ~:992-1001; `_BridgeSkillRunner` ~:727-772)
- Test: `Tests/Agents/test_skill_tool_spawn.py` (extend) or the bridge's existing test file (READ both, put tests where the existing SkillRunner tests live)

**Interfaces:**
- Consumes: `SkillFileBindings`, `SKILL_FILE_TOOL_NAME` (Task 3); `SkillsScopeService.read_skill_file` (Task 1); `reference_files` on execute result (Task 2).
- Produces: `run_reply` constructs one `SkillFileBindings` per run — `reader` wraps the scope service (`lambda name, path: asyncio.run(self._skills_service.read_skill_file(name, path, mode="local"))`), `authorized` seeded empty for now (Task 5 seeds turn bindings) — and passes it to BOTH `AgentService(skill_file_bindings=...)` and `_BridgeSkillRunner(..., skill_file_bindings=...)`. `_BridgeSkillRunner.run`: after a successful `execute_skill`, `self._skill_file_bindings.authorized.add(name)` (when bindings present), and append the "Bundled files" block to `rendered` before `spawn` when `reference_files` non-empty:

```python
        refs = result.get("reference_files") if isinstance(result, Mapping) else None
        if refs and self._skill_file_bindings is not None:
            rows = ", ".join(
                f"{r['path']} ({r['size']} bytes{'' if r.get('is_text', True) else ', binary'})"
                for r in refs
            )
            rendered = f"{rendered}\n\nBundled files (readable via skill_file): {rows}"
```

- [ ] **Step 1: Failing tests**: (a) SkillRunner.run with a fake skills service whose execute result carries `reference_files` → the spawned task text contains `Bundled files (readable via skill_file): references/api.md` AND the bindings' `authorized` now contains the skill name; (b) with `reference_files=None` → body unchanged, name still authorized; (c) bindings=None (non-bridge construction) → byte-identical legacy behavior (no block, no crash). Model on the file's existing `_BridgeSkillRunner`/spawn tests.
- [ ] **Step 2: RED** → **Step 3: implement** (ctor param with default `None` for compat; wire in run_reply) → **Step 4: GREEN** + `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv pytest Tests/Agents/ Tests/Chat/ -q` (baseline failures only). **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py <the test file(s) you extended>
git commit -m "feat(skills): fork skill children read their own bundle via skill_file"
```

---

### Task 5: Turn bindings — controller 4-tuple + bridge seeding + bridge-side block

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_apply_skill_substitution` ~:1790-2012 — 7 returns; call sites :470/:1150/:1214/:1312; funnel `_stream_assistant_response` ~:2349 / `_run_agent_reply` ~:2550 / run_reply call ~:2634); `tldw_chatbook/Chat/console_agent_bridge.py` (`run_reply` signature ~:832-846 + bindings seeding + block append)
- Test: `Tests/Chat/test_console_skill_substitution.py` (extend)

**Interfaces:**
- Consumes: bindings object from Task 4's run_reply wiring; `reference_files` on the execute results the substitution already holds (Task 2).
- Produces (NORMATIVE): `_apply_skill_substitution` widens to a **5-tuple** `(messages, refuse, notes, skill_bindings, skill_bundle_block)`:
  - `skill_bindings: tuple[str, ...]` — the leading-RESOLVED name (inline AND fork outcomes; not on refuse) plus embedded names that actually SPLICED (never blocked-literal, never fork-literal).
  - `skill_bundle_block: str` — the fully-rendered "Bundled files" block for all bound skills with non-empty `reference_files` (Task 4's row format, one combined block headed `Bundled files (readable via skill_file):`), or `""`. Built at substitution as **pure string work from the execute results already in hand** (no re-execution, no extra service calls) but **NEVER inserted into `messages` here** — only `run_reply` appends it, so plain sends drop it unused and the invariant (block only where the tool exists) holds.
  - All 7 returns updated; all 4 call-site unpacks widened; `_stream_assistant_response` and `_run_agent_reply` gain `skill_bindings: tuple[str, ...] = ()` and `skill_bundle_block: str = ""` kwargs threaded to `run_reply(turn_skill_bindings=..., turn_bundle_block=...)`.
  - `run_reply`: seeds `bindings.authorized.update(turn_skill_bindings)`; when `turn_bundle_block` is non-empty, appends `\n\n{turn_bundle_block}` to the LAST `role=="user"` entry of its OWN copied `agent_messages` list before `run_turn`.

- [ ] **Step 1: Failing tests** (extend substitution tests; construct as the file does):

```python
@pytest.mark.asyncio
async def test_leading_mention_returns_binding_and_block():
    controller, skills = ...  # file's existing construction; execute result must carry reference_files
    messages = [{"role": "user", "content": "$code-review look"}]
    out, refuse, notes, bindings, block = await controller._apply_skill_substitution(messages)
    assert bindings == ("code-review",)
    assert "Bundled files (readable via skill_file):" in block
    assert all("Bundled files" not in m.get("content", "") for m in out)  # NEVER in messages here


@pytest.mark.asyncio
async def test_embedded_spliced_binds_but_blocked_does_not():
    ...  # one spliced + one blocked mention -> bindings == (spliced_name,)


@pytest.mark.asyncio
async def test_plain_text_no_bindings():
    ...  # no $ -> bindings == () and block == ""
```

(Also update EVERY existing unpack in the file — mechanical 3→5 widening; assertions unchanged. And the `test_console_chat_controller.py` monkeypatch fake that returns the tuple.)

- [ ] **Step 2: RED** → **Step 3: implement** (7 returns; block built from execute results' `reference_files` with the Task 4 row format, one combined block headed `Bundled files (readable via skill_file):`; fork-literal/blocked mentions excluded from bindings AND block) → thread through funnel (2 signatures + 4 call sites + run_reply params; run_reply: `if self._skill_file_bindings...` — seed `bindings.authorized.update(turn_skill_bindings)`; `if turn_bundle_block:` append `\n\n{turn_bundle_block}` to the LAST `role=="user"` entry of its own copied `agent_messages` list before `run_turn`).
- [ ] **Step 4: GREEN** + full: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv pytest Tests/Chat/ Tests/Agents/ -q` (baselines only). **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_skill_substitution.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(skills): turn \$-bindings authorize skill_file; bridge-side bundled-files block"
```

---

### Task 6: E2E + full regression

**Files:**
- Test: `Tests/Agents/test_skill_file_runtime_tool.py` (extend with the e2e), full suites

- [ ] **Step 1: E2E test** — real `LocalSkillsService` (tmp store) with a trusted skill carrying `references/api.md`; a scripted model that (turn 1) calls `skill_file(skill_name=..., path="references/api.md")` and (turn 2) finishes; run through `AgentService` with bindings authorized for that skill and reader = the real scope-service read — assert the tool result fed back contains the file text and the run completes. (Compose from Task 1's service + Task 3's harness; trust via the same arrangement Tests/Skills uses — file-marker keyring fallback.)
- [ ] **Step 2: Full regression:**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ Tests/Agents/ Tests/Chat/ Tests/UI/test_console_skill_commands.py -q`
Expected: 0 failed modulo the two known baselines.

- [ ] **Step 3: Commit**

```bash
git add Tests/Agents/test_skill_file_runtime_tool.py
git commit -m "test(skills): e2e — fork skill reads its own reference file on demand"
```

---

## Notes for the executor

- The spec governs. §2's registry/disclosure rationale is why this is a RUNTIME tool — do not "simplify" to a provider registration.
- Task 5 contains a normative RESOLUTION paragraph (5-tuple; block rendered at substitution as a STRING, inserted only by run_reply) — it supersedes the spec's looser "metadata fetched at that point" wording; the spec's invariant (block only where the tool exists; never stored transcript; never plain sends) is unchanged.
- `agent_runtime.py` must not import Skills_Interop; the reader is injected.
- Re-grep every anchor; files shift.
