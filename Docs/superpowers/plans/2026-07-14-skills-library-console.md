# Skills — Library home + Console/agent invocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Library ▸ Skills (the standalone Skills tab retires behind a `skills → library` route alias) and make trusted skills usable from the agent-capable Console on both surfaces the SKILL.md format encodes — users trigger them (`/skill-name [args]` / `/skills <name> [args]`) and the agent discovers/calls them mid-run as tools. Both are thin adapters over shipped machinery. Spec (the contract — every requirement below traces to a section): `Docs/superpowers/specs/2026-07-14-skills-library-console-design.md`.

**Architecture:** Phase 1 adds a skills source to the Library shell (rail row + count seam, list/detail canvases, import) through `skills_scope_service`/`local_skills_service`/`local_skill_trust_service` — never raw index files from UI; the list renders BOTH populations the scope service returns (trusted `available_skills` + `blocked_skills`). Phase 2 registers a pure skills resolver on `console_command_grammar`'s fallback hook plus a `/skills` command, a skill picker, a provider-payload substitution rule (raw command stored; rendered body substituted for the final user message at build time only), a `SkillToolProvider` for the agent catalog, and a per-run spawn-wired skill executor — and lands the two deferred catalog fixes (loop-side dedupe, per-run owner-map cache) with regression tests, since skills are the first realistic >8-tool catalog.

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0, dataclasses, pydantic, SQLite/JSON index, pytest, `asyncio.run`/`run_until_complete` on worker threads.

## Global Constraints

- **Worktree:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime` (branch `claude/skills-spec` @ `origin/dev` 9463a8a2, post-#629 — the agent-capable Console is shipped). ALL paths below are relative to it; ALL file writes go inside it. Do **not** write to the main checkout `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs` — a prior plan-writer made that mistake.
- **Tests run ONLY via the venv python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` → `$PY -m pytest …`, run from the worktree root (the venv installs this package `-e`, so it imports the worktree). System `python3` is 3.9 and breaks collection.
- **No `AppTest`** (does not exist in this Textual version). UI tests use `async with app.run_test() as pilot:` (real `TldwCli`). Keep UI tests minimal; most coverage is pure/service-level.
- **CSS discipline:** any new styled id/class is added to the SOURCE `tldw_chatbook/css/components/_agentic_terminal.tcss`, the bundle is regenerated with `$PY tldw_chatbook/css/build_css.py`, and a pin test asserts the selector is present in BOTH the source and `tldw_chatbook/css/tldw_cli_modular.tcss` (dual pin). Widget `DEFAULT_CSS` must parse standalone (no `$ds-*` without local fallbacks). Modal reuse (`SkillTrustPassphraseModal`) already carries its own `DEFAULT_CSS`.
- **Markup-escape-where-markup-on discipline (#629):** every skill-derived string (name, description, argument_hint, marker copy, list rows, picker rows, refuse copy) is `rich.markup.escape`d ONLY where it renders into a markup-enabled Button label or `markup`-default Static. Where the consumer renders `markup=False` (rail Statics, transcript TOOL markers via `_message_render_text`/`Content.assemble`, and every `markup=False` field Static/Input) the text is passed RAW — escaping there leaves literal backslashes (`fetch [docs]` → `fetch \[docs]`). Mirror the shipped `format_agent_step_marker`/`_summarize` decisions exactly.
- **Pure-module rules:** `tldw_chatbook/Library/library_skills_state.py`, `tldw_chatbook/Chat/console_skill_resolver.py`, and the new pure helpers in `Agents/tool_catalog.py`/`Agents/agent_runtime.py` import stdlib + dataclasses + (state module only) the pydantic/DB exception types they classify — NO Textual, app, DB IO, or network. `agent_runtime.py` stays pure (stdlib + `agent_models`); `agent_stream.py` stays pure. Only `agent_service.py`, `console_agent_bridge.py`, the canvases, and `chat_screen.py`/`console_chat_controller.py` touch IO/UI.
- **Skills-as-tools route through the spawn path, never plain `invoke()`** (spec Review-correction 1): budgets/cancel/DB lineage. `SkillToolProvider.invoke()` raises by design; the run-scoped executor renders via `execute_skill` and calls the run's spawn machinery.
- **Render-vs-persist** (spec Review-correction 2): the raw `/skill-name …` command is what is stored and displayed as the user message; the rendered body substitutes into the provider payload for that turn only; retries re-resolve + re-trust-check + re-render.
- **Skill `model` override deferred** (spec Review-correction 3): ignored in v1; the editor shows a "not applied in v1" hint when a skill declares one.
- **`context: fork` = clean-context primary run** (spec Review-correction 4): the fork payload is the leading session-system message (if any) + the rendered turn only, no conversation history; `inline` = history + rendered turn.
- **Built-ins always shadow skills** (spec Review-correction 5): registration order — `/prompt`, `/system`, `/skills` and the builtin tool names win; the Library editor warns on a shadowing name but never blocks save.
- **Trust gates twice** (spec Security model): catalog/resolve time (untrusted skills are absent from `available_skills`, so invisible to the resolver and the `SkillToolProvider`) AND execution time (`execute_skill`'s `_require_trusted_skill` + `_verify_exact_skill_content` re-verify the content hash on every render). Never bypass `execute_skill` for rendering.
- **`allowed_tools` intersects, never grants:** a skill-driven turn's `AgentConfig.allowed_tools = intersect_skill_tools(skill.allowed_tools, builtins)`; a normal run's allow-list = builtins ∪ eligible skill names, computed fresh at run start (else the disclosed-AND-allowed gate refuses every skill tool).
- **Refuse-copy exact strings (do NOT paraphrase when implementing; these are asserted verbatim in tests):**
  - Untrusted refuse (Console): `Skill "{name}" isn't trusted ({reason}) — review and approve it in Library ▸ Skills before running it.`
  - Empty `/skills` list row: `No skills yet — create them in Library ▸ Skills.`
  - Skill-driving marker row (TOOL role, raw): `skill {name} → driving this turn`
  - Save-marks-needs-review warning (editor): `Saving marks this skill "needs review" — re-approve it in the trust panel after saving.`
  - Shadow-name warning (editor): `Name shadows a built-in command/tool ("{name}") — it will not be invocable as /{name} or as an agent tool.`
- **Config reads at interaction time** go through live services / `tldw_chatbook.config.load_settings()` — no boot-snapshot reads.
- **Backlog IDs:** if you mint backlog tasks for this work, create them in an `origin/dev` worktree and re-verify against `origin/dev` (this checkout has previously minted colliding IDs). Do not reference future task IDs.
- Commit after every task. Messages `feat(skills): …` / `feat(console): …` / `refactor(agents): …`, ending with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

### Shipped API (verified — target these exact shapes)

- `Skills_Interop/skills_scope_service.py` — `SkillsScopeService` (app attr `self.app_instance.skills_scope_service`): async `list_skills(*, mode="local", **kw)`, `get_context(*, mode="local")`, `get_skill(name, *, mode="local")`, `create_skill(*, mode="local", **kw)`, `update_skill(name, *, mode="local", **kw)`, `delete_skill(name, *, mode="local", **kw)`, `import_skill(*, mode="local", **kw)`, `import_skill_file(bytes, *, mode="local", **kw)`, `execute_skill(name, *, mode="local", **kw)`. `mode=None` defaults to SERVER — always pass `mode="local"`. `get_context` normalizes `available_skills` + `blocked_skills` (each gets `record_id`/`backend`).
- `Skills_Interop/local_skills_service.py` — `LocalSkillsService.get_context()` returns `{available_skills:[summary…], context_text, blocked_skills:[summary…]}`. Each summary: `name, description, argument_hint, user_invocable, disable_model_invocation, context, agent_skill_name, validation_status, validation_errors, trust_status, trust_reason_code, trust_blocked, trust_changed_files, trust_manifest_generation, trust_last_verified_at`. `get_skill` adds full `SkillResponse`: `id, content, supporting_files, allowed_tools, model, directory_path, created_at, last_modified, version`. `create_skill(*, name, content, supporting_files=None, trust_approved=False)`; `update_skill(name, *, content=None, supporting_files=None, expected_version=None, trust_approved=False)`; `delete_skill(name, *, expected_version=None)`; `import_skill(*, content, name=None, supporting_files=None, overwrite=False, trust_approved=False)`; `execute_skill(name, *, args=None)` → dump of `SkillExecutionResult(skill_name, rendered_prompt, allowed_tools, model_override, execution_mode, fork_output)`. `execute_skill` calls `_require_trusted_skill` + `_verify_exact_skill_content` (raises `SkillTrustBlockedError`) before rendering `body.strip().replace("{{args}}", args or "")`.
- `Skills_Interop/skill_trust_service.py` — `SkillTrustService` (app attr `self.app_instance.local_skill_trust_service`): `unlock_with_passphrase(passphrase, *, salt=None)`, `bootstrap_trust(passphrase=None, *, salt=None)`, `status_for_skill(name) -> SkillTrustStatus`, `overall_status()`, `capture_review(name) -> {review_id, changed_files, current_files, current_fingerprints, ...}`, `discard_review(review_id)`, `trust_reviewed_snapshot(review_id)` (approve; requires unlocked keys + generation match). **There is NO `revoke`/`untrust` method** on the service or store — see Out-of-scope.
- `Skills_Interop/skill_trust_models.py` — `SkillTrustStatus(skill_name, trust_status, trust_reason_code, trust_blocked, changed_files, manifest_generation, last_verified_at)`; `SkillTrustBlockedError(skill_name, reason_code, trust_status, changed_files)`; status constants `TRUST_STATUS_TRUSTED="trusted"`, quarantined/locked/uninitialized variants.
- `tldw_api/skills_schemas.py` — `_normalize_skill_name(v)` (lowercase, `^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$`, no `--`; raises ValueError otherwise), `SKILL_NAME_PATTERN`, `SUPPORTING_FILE_NAME_PATTERN`, `MAX_SUPPORTING_FILES_COUNT=20`, `MAX_SUPPORTING_FILE_BYTES=500000`; `SkillBase`/`SkillCreate`/`SkillUpdate`/`SkillExecuteRequest(args max_length=10000)`/`SkillExecutionResult`.
- `UI/Screens/skills_screen.py` — `SkillTrustPassphraseModal(ModalScreen[str | None])(*, confirm_bootstrap: bool)` (dismisses the passphrase string or None; ids `#skill-trust-passphrase-input`/`-submit`/`-cancel`). REUSE it — do not fork.
- `Agents/tool_catalog.py` — `ToolProvider` Protocol (`list_catalog`/`load_schema`/`invoke`); `BuiltinToolProvider` (tools `calculator`, `get_current_datetime`); `ToolCatalogRegistry` (`register_provider`, `list_catalog`, `find`, `load_schema`, `resolve_name`, `invoke_by_name`, `_owner_and_id` with the `TODO(task-201)` re-listing marker @138); `initial_disclosure(registry, budget) -> (schemas, offer_find_load)` (direct-disclose when `len(catalog) <= DIRECT_DISCLOSE_THRESHOLD=8`).
- `Agents/agent_models.py` — `AgentConfig(model, system_prompt, allowed_tools=(), budget=RunBudget())`; `RunBudget(max_steps=8, max_wall_seconds=240, max_subagents=2, max_active_tools=8, max_subagent_result_chars=4000)`; `ToolCatalogEntry(id, name, one_line_description, source)`; `ToolSchema(id, name, description, parameters)`; `ToolResult(ok, content="", error="")`; `SPAWN_TOOL_NAME`, `FIND_TOOLS_NAME`, `LOAD_TOOLS_NAME`, `RUNTIME_TOOL_NAMES`, `RUN_DONE`, `clamp_child_budget` (zeros child `max_subagents`).
- `Agents/agent_runtime.py` — `run_agent_loop(config, initial_messages, active_schemas, deps)`; `LoopDeps(call_model, invoke_tool, spawn, find_tools, load_schemas, should_cancel, clock, on_step=…)`; skill tool calls take the `else` branch (@342) → `deps.invoke_tool(call)`; LOAD_TOOLS room-slice @332–334; `LOOP_DETECTION_N=3`.
- `Agents/agent_service.py` — `AgentService(db, registry, chat_call=None, clock=time.monotonic, on_step=None)`; `_run_one` @104 (spawn closure @171, `_make_invoke_tool` @83, `deps=LoopDeps(...)` @198); `run_turn(*, conversation_id, messages, config, api_endpoint, should_cancel, supersede_run_id)` @222; `SUBAGENT_SYSTEM_PROMPT`.
- `Chat/console_command_grammar.py` — `ConsoleCommandRegistry` (`register`, `register_fallback_resolver(Callable[[str,str], CommandParse | None])`, `parse`, `available_names`); `CommandParse(kind, name="", args="")`; `KIND_COMMAND/KIND_FALLBACK/KIND_UNKNOWN/KIND_NOT_COMMAND`; `default_console_registry()` (registers `/prompt`, `/system`).
- `Chat/console_agent_bridge.py` — `ConsoleAgentBridge(*, agent_runs_db, store, provider_gateway, registry=None, clock=time.monotonic)`; `__init__` builds registry+`_allowed_tools` once (@327–332); `run_reply(*, conversation_id, session_id, resolution, assistant_message_id, model, session_system_prompt, agent_messages, should_cancel, supersede_previous=False) -> RunOutcome` (AgentConfig site @342–346); `format_agent_step_marker`, `compose_agent_system_prompt`, `_append_marker` (raw), `_summarize` (raw).
- `Chat/console_chat_controller.py` — `_leading_system_message()` @919, `_provider_messages_for_session(session_id, *, before_message_id=None)` @937, `_provider_messages_through_message` @952, `_provider_message_payloads(...)` @966; send builds @206/@508; `_run_agent_reply` @749; `_stream_assistant_response` @622.
- `UI/Screens/chat_screen.py` — grammar imports @45–54; `self._console_command_registry = default_console_registry()` @1185; send action parse @7238–7279; `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` @7281; `_console_unknown_command_hint` @7287; `_dispatch_console_command` @7294; `_console_command_insert_prompt` @7319; `_console_command_apply_system` @7521; `_submit_console_native_draft` @7132; `_console_send_blocked_reason` @7194; `_ensure_console_agent_bridge` @2078; controller construction @2228/@2284; `_open_console_prompt_picker_for_insert` @7415.
- `Widgets/Console/console_prompt_picker_modal.py` — `ConsolePromptPickerModal(ModalScreen[Optional[Mapping]])(*, mode, initial_query, prompt_search)` — the pattern to mirror for the skill picker (ids `console-prompt-picker-*`, debounced FTS search, `↑/↓/Enter/Esc`).
- Library mirror: `Library/library_prompts_state.py`, `Widgets/Library/library_prompts_canvas.py` (`LibraryPromptsListCanvas`), `Library/library_shell_state.py` (`LIBRARY_ROW_BROWSE_*` @9–17, `browse_rows` builder @126–176), `UI/Screens/library_screen.py` (rail-row → row map @375, `_list_local_source_snapshot` @1777, prompts count/page seam @1801/@1884/@1888, canvas render @2970, `.library-prompt-row` press @6249, dirty flag `_library_prompt_dirty` @657, `flush_pending_work` @1091).
- `UI/Navigation/screen_registry.py` — `_SCREEN_ROUTES["skills"]` @68; `_SCREEN_ALIASES` @106 (`"notes": "library"` @109, `"prompts": "library"` @114); `resolve_screen_target` @139.
- `app.py` — skills service wiring @3623–3642 (`local_skill_trust_service`, `local_skills_service`, `skills_scope_service`).

---

## Phase 1 — Library ▸ Skills

### Task 1: `count_skills` + context seam + rail row `Skills (N)`

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py` (add `count_skills`), `tldw_chatbook/Skills_Interop/local_skills_service.py` (add `count_skills`)
- Modify: `tldw_chatbook/Library/library_shell_state.py` (Browse row `Skills (N)` after Prompts; constant `LIBRARY_ROW_BROWSE_SKILLS = "browse-skills"`), `tldw_chatbook/UI/Screens/library_screen.py` (snapshot fetch + rail-row → row map)
- Test: `Tests/Skills/test_skills_count_seam.py` (new), extend `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: `LocalSkillsService.count_skills() -> int` (async; `len(get_context available)+len(blocked)` — i.e. all managed skills incl. needs-review); `SkillsScopeService.count_skills(*, mode="local") -> int` (async passthrough, action id `skills.context.list.local`); rail row id `LIBRARY_ROW_BROWSE_SKILLS`; snapshot key `"skills"` in `_local_source_records` carrying `(count, context_payload)` where `context_payload` = the normalized `get_context` dict (`available_skills`+`blocked_skills`).
- Consumes: `LocalSkillsService.get_context()` (already returns both populations).

- [ ] **Step 1: Failing seam test** `Tests/Skills/test_skills_count_seam.py`:

```python
import pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_count_skills_counts_managed_skills(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path, allow_untrusted_without_trust_service=True)
    assert await svc.count_skills() == 0
    await svc.create_skill(
        name="code-review",
        content="---\nname: code-review\ndescription: Review code\n---\nDo it.")
    await svc.create_skill(
        name="summarize",
        content="---\nname: summarize\ndescription: Summarize text\n---\nGo.")
    assert await svc.count_skills() == 2
```

- [ ] **Step 2:** `$PY -m pytest Tests/Skills/test_skills_count_seam.py -v` → FAIL (`count_skills` missing).
- [ ] **Step 3:** Implement `LocalSkillsService.count_skills` (Google docstring): `ctx = await self.get_context(); return len(ctx.get("available_skills") or []) + len(ctx.get("blocked_skills") or [])`. Implement `SkillsScopeService.count_skills` mirroring `list_skills`'s `_call` shape but returning the int — because `_normalize_response` only mutates dict/list envelopes, route it directly like `delete_skill` does: normalize mode, `_require_service`, `_enforce_policy(self._source_action_id("skills.context.list.server", mode))`, `result = await self._maybe_await(service.count_skills(**kwargs))`, return `int(result)`.
- [ ] **Step 4:** Test passes; commit `feat(skills): count_skills seam`.
- [ ] **Step 5: Failing UI test** in `Tests/UI/test_library_shell.py`: mount the Library screen with a fake `skills_scope_service` exposing async `count_skills`/`get_context` on the app → assert the rail renders a row whose label contains `Skills (2)` and whose id is `LIBRARY_ROW_BROWSE_SKILLS`. Copy the existing `Prompts (N)` rail test in that file verbatim and rename (`grep -n "Prompts (" Tests/UI/test_library_shell.py`).
- [ ] **Step 6:** Add `LIBRARY_ROW_BROWSE_SKILLS` in `library_shell_state.py` (@9–17 block) and a Browse row after the Prompts row (@158–166) titled `"Skills"` with `count`-suffix rendering identical to Prompts. In `library_screen.py`: add `"skills": LIBRARY_ROW_BROWSE_SKILLS` to the rail-row→row map (@375 area). In `_list_local_source_snapshot` (@1777) add — mirroring the `count_prompts`/`prompts_page` optional-call pattern (@1884/@1888) — a `skills` optional call fetching `skills_scope_service.get_context(mode="local")` via `_run_library_service_call(..., isolate_in_worker=True)`; store `_local_source_records["skills"] = (count, context_payload)`, count = available+blocked len (degrade to `(None, {"available_skills": [], "blocked_skills": []})` on failure, matching the prompts `(None, ())` degrade). Row click sets `_library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS` and (until Task 3) shows the empty-canvas path so the row is inert-but-selectable.
- [ ] **Step 7:** UI test passes; `$PY -m pytest Tests/UI/test_library_shell.py -q` stays green (baseline row count). Commit `feat(skills): Library rail row Skills (N) + snapshot seam`.

### Task 2: Pure state builders `library_skills_state.py`

**Files:**
- Create: `tldw_chatbook/Library/library_skills_state.py`
- Test: `Tests/Library/test_library_skills_state.py`

**Interfaces (all pure, dataclasses frozen; mirror `library_prompts_state.py`):**
- `SkillListRow(name: str, secondary: str, trust_glyph: str, blocked: bool)` — `name` raw (canvas escapes at render); `trust_glyph` = `"✓"` (trusted) or `"⚠"` (needs review); `secondary` = the flags line `"user · agent"` variants + optional description, `blocked` = `trust_blocked`.
- `build_skills_list_state(context_payload, *, query, sort) -> SkillsListState` with `SkillsListState(rows: tuple[SkillListRow, ...], count: int, sort: str)` — renders BOTH `available_skills` (trusted) and `blocked_skills` (needs-review); `query` filters case-insensitively over name+description; `sort in {"name","status"}` (`"status"` = needs-review first, then name).
- `skill_flags_line(user_invocable: bool, disable_model_invocation: bool) -> str` — `"user · agent"` when both, `"user"` / `"agent"` when one, `"not invocable"` when neither (user=user_invocable, agent=not disable_model_invocation).
- `SkillEditorState(name, description, argument_hint, allowed_tools_csv, user_invocable, disable_model_invocation, context, model, body, supporting_files: tuple[tuple[str,int], ...], version, trust_status, trust_blocked, trust_changed_files: tuple[str, ...])` + `build_skill_editor_state(detail: Mapping) -> SkillEditorState` (maps `get_skill` output; `content` split into frontmatter/body via the same `---\n…\n---` grammar `local_skills_service._parse_front_matter` uses; supporting_files → sorted `(name, byte_len)` pairs).
- `compose_skill_markdown(editor_state, *, body: str) -> str` — assemble a SKILL.md string (`---\n<yaml frontmatter>\n---\n<body>`) the service's `_parse_front_matter`+`_metadata_from_content` round-trip (only emit keys the editor owns: `name`, `description`, `argument_hint`, `allowed_tools` (list), `user_invocable`, `disable_model_invocation`, `context`, and `model` only when set). Use `yaml.safe_dump(sort_keys=False)`.
- `skill_name_shadows_builtin(name: str) -> str | None` — returns the shadowed builtin name when `name` (normalized) ∈ `{"calculator","get_current_datetime","spawn_subagent","find_tools","load_tools","prompt","system","skills"}`, else None.
- `save_marks_needs_review(trust_status: str, trust_blocked: bool) -> bool` — `True` when the open skill is currently trusted (`trust_status == "trusted"` and not blocked); saving changes its content hash → it will re-quarantine.
- `classify_skill_save_error(result: Any, message: str, exc: Exception | None) -> str` → `"exists" | "version-conflict" | "invalid-name" | "trust-blocked" | "ok" | "error"` (`local_skill_exists:` → exists; `local_skill_version_conflict:` → version-conflict; `ValueError` from `_normalize_skill_name` / "must contain only lowercase" → invalid-name; `SkillTrustBlockedError` → trust-blocked; a dict result with a `name` and no exc → ok).

- [ ] **Step 1: Failing tests** (`Tests/Library/test_library_skills_state.py`, ~14 focused tests with literal expected values):

```python
from tldw_chatbook.Library.library_skills_state import (
    build_skill_editor_state, build_skills_list_state, classify_skill_save_error,
    compose_skill_markdown, save_marks_needs_review, skill_flags_line,
    skill_name_shadows_builtin,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _ctx(available=(), blocked=()):
    return {"available_skills": list(available), "blocked_skills": list(blocked)}


def _summary(name, **over):
    base = {"name": name, "description": f"{name} desc", "argument_hint": None,
            "user_invocable": True, "disable_model_invocation": False,
            "trust_status": "trusted", "trust_blocked": False}
    base.update(over)
    return base


def test_list_renders_trusted_and_blocked_with_glyphs():
    state = build_skills_list_state(
        _ctx(available=[_summary("alpha")],
             blocked=[_summary("bravo", trust_status="quarantined_modified", trust_blocked=True)]),
        query="", sort="name")
    by_name = {r.name: r for r in state.rows}
    assert by_name["alpha"].trust_glyph == "✓" and by_name["alpha"].blocked is False
    assert by_name["bravo"].trust_glyph == "⚠" and by_name["bravo"].blocked is True
    assert state.count == 2


def test_status_sort_puts_needs_review_first():
    state = build_skills_list_state(
        _ctx(available=[_summary("zeta")],
             blocked=[_summary("aardvark", trust_blocked=True)]),
        query="", sort="status")
    assert [r.name for r in state.rows] == ["aardvark", "zeta"]


def test_query_matches_name_and_description():
    state = build_skills_list_state(
        _ctx(available=[_summary("code-review", description="Review pull requests"),
                        _summary("summarize", description="Shorten text")]),
        query="pull", sort="name")
    assert [r.name for r in state.rows] == ["code-review"]


def test_flags_line_variants():
    assert skill_flags_line(True, False) == "user · agent"
    assert skill_flags_line(True, True) == "user"
    assert skill_flags_line(False, False) == "agent"
    assert skill_flags_line(False, True) == "not invocable"


def test_shadow_predicate():
    assert skill_name_shadows_builtin("calculator") == "calculator"
    assert skill_name_shadows_builtin("skills") == "skills"
    assert skill_name_shadows_builtin("code-review") is None


def test_save_marks_needs_review_only_when_currently_trusted():
    assert save_marks_needs_review("trusted", False) is True
    assert save_marks_needs_review("quarantined_modified", True) is False


def test_editor_state_splits_frontmatter_and_body():
    detail = {"name": "code-review", "description": "Review code",
              "argument_hint": "[path]", "allowed_tools": ["calculator"],
              "user_invocable": True, "disable_model_invocation": False,
              "context": "inline", "model": None, "version": 3,
              "trust_status": "trusted", "trust_blocked": False,
              "supporting_files": {"notes.md": "hello"},
              "content": "---\nname: code-review\ndescription: Review code\n---\nReview {{args}} now."}
    state = build_skill_editor_state(detail)
    assert state.name == "code-review" and state.argument_hint == "[path]"
    assert state.allowed_tools_csv == "calculator"
    assert state.body.strip() == "Review {{args}} now."
    assert state.supporting_files == (("notes.md", 5),)
    assert state.version == 3


def test_compose_roundtrips_through_frontmatter_grammar():
    detail = {"name": "code-review", "description": "Review code", "argument_hint": None,
              "allowed_tools": None, "user_invocable": True, "disable_model_invocation": False,
              "context": "fork", "model": None, "version": 1, "trust_status": "trusted",
              "trust_blocked": False, "supporting_files": None,
              "content": "---\nname: code-review\ndescription: Review code\n---\nBody here."}
    state = build_skill_editor_state(detail)
    text = compose_skill_markdown(state, body="New body {{args}}")
    assert text.startswith("---\n") and "name: code-review" in text
    assert text.rstrip().endswith("New body {{args}}")


def test_classify_outcomes():
    assert classify_skill_save_error(None, "local_skill_exists:x", None) == "exists"
    assert classify_skill_save_error(None, "local_skill_version_conflict:x", None) == "version-conflict"
    assert classify_skill_save_error(None, "", SkillTrustBlockedError(
        skill_name="x", reason_code="skill_modified", trust_status="quarantined_modified")) == "trust-blocked"
    assert classify_skill_save_error({"name": "x"}, "", None) == "ok"
```

- [ ] **Step 2:** FAIL run. **Step 3:** Implement (~180 lines). Reuse `format_console_relative_age` from `Workspaces.conversation_browser_state` if a modified-age is wanted in `secondary`; reuse the frontmatter split by importing `LocalSkillsService._parse_front_matter` (it's a `@staticmethod` — call `LocalSkillsService._parse_front_matter(content)`), keeping the state module free of Textual. **Step 4:** PASS run. **Step 5:** Commit `feat(skills): pure list/editor state builders + shadow/needs-review predicates`.

### Task 3: List canvas + screen wiring

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (list part)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (canvas render when `_library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS`; sort/filter handlers), `tldw_chatbook/css/components/_agentic_terminal.tcss` (only if new list classes need styling) + regenerate
- Test: `Tests/UI/test_library_skills_canvas.py` (new)

**Interfaces:**
- Consumes: Task 1 snapshot `_local_source_records["skills"]` + Task 2 `build_skills_list_state`.
- Produces: widget `LibrarySkillsListCanvas(state: SkillsListState)` with row Button ids `library-skill-row-<name>` (skill names are unique + name-shaped, safe as id suffixes) and class `library-skill-row`; the row label = `f"{glyph} {name}"` (escaped) with the flags/description as a secondary Static; blocked rows carry class `library-skill-row-blocked` (dim, still selectable — the trust panel needs them visible). Toolbar ids `library-skills-sort`, `library-skills-import`; filter Input `library-skills-filter`; empty-state `library-skills-empty` copy `No skills yet — create them in Library ▸ Skills.` Screen handler `handle_library_skill_row(name)` opens the editor (Task 4).

- [ ] **Step 1:** Copy `Widgets/Library/library_prompts_canvas.py`'s list-compose structure (header count line, filter Input, one `Horizontal` toolbar row, escaped row Buttons) as the starting file; rename all ids/classes skills-specific. Structural template copy — NOT shared code.
- [ ] **Step 2: Failing UI test:** mount the canvas in a bare test App with a 2-row state (one trusted `✓`, one blocked `⚠`) → assert 2 row buttons, the blocked row has class `library-skill-row-blocked`, a bracket-name renders literally (escape proof: seed a name is impossible — names are name-shaped — so assert the escape on a `description` containing `[x]` shown in the secondary Static instead), and the empty state renders its exact copy when rows are empty.
- [ ] **Step 3:** Implement; wire `library_screen.py`: render `LibrarySkillsListCanvas(build_skills_list_state(context_payload, query=…, sort=…))` when the skills row is selected (mirror the prompts canvas render @2970). Sort button cycles `name ↔ status`; filter (Enter) rebuilds state. Fresh state from `_local_source_records["skills"]`.
- [ ] **Step 4:** Tests pass; `test_library_shell.py` green. If new CSS classes were added, regenerate the bundle + add a dual-pin assertion. **Step 5:** Commit `feat(skills): list canvas (trusted + needs-review populations)`.

### Task 4: Detail editor + trust panel + save + shadow warning + delete

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (editor part), `tldw_chatbook/UI/Screens/library_screen.py` (open/save/delete/trust handlers; `_library_skill_dirty` flag participating in `flush_pending_work`), `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate)
- Test: extend `Tests/UI/test_library_skills_canvas.py`, `Tests/Skills/test_skills_library_flow.py` (new, real services)

**Interfaces:**
- Consumes: `skills_scope_service` CRUD (`get_skill`/`create_skill`/`update_skill`/`delete_skill`, `mode="local"`); `local_skill_trust_service` (`capture_review`/`trust_reviewed_snapshot`/`unlock_with_passphrase`/`status_for_skill`); the shipped `SkillTrustPassphraseModal`; Task 2 state + `compose_skill_markdown` + `classify_skill_save_error` + `skill_name_shadows_builtin` + `save_marks_needs_review`.
- Produces: editor field ids `library-skill-name`, `-description`, `-argument-hint`, `-allowed-tools`, `-user-invocable` (Checkbox/Switch), `-disable-model` (Checkbox/Switch), `-context` (Select inline/fork), `-model` (Input + hint Static `library-skill-model-hint` copy `Not applied in v1.`), `-body` (TextArea), a read-only supporting-files list `library-skill-supporting`, and a trust panel `library-skill-trust-panel` (state line, changed-files list, buttons `library-skill-trust-unlock`, `library-skill-trust-review`, `library-skill-trust-approve`); action ids `library-skill-save`, `-delete`, `-back`; a warnings Static `library-skill-warnings`; save-status Static `library-skill-save-status`. Screen methods `handle_library_skill_row(name)`, `_save_library_skill()`; dirty flag `_library_skill_dirty` mirrors the prompts branch inside `flush_pending_work` (@1091) exactly.

- [ ] **Step 1: Failing tests** (write all before implementing):
  - Real-service flow (`Tests/Skills/test_skills_library_flow.py`, `allow_untrusted_without_trust_service=True` service): open row → editor shows name/description/argument_hint/context/body populated from a seeded skill; edit body → Save → `update_skill` bumps version; the save-status Static shows `Saved.`
  - Shadow warning: type name `calculator` in the editor → the warnings Static shows the exact shadow copy `Name shadows a built-in command/tool ("calculator") — it will not be invocable as /calculator or as an agent tool.` and Save is NOT blocked.
  - Save-marks-needs-review: open a TRUSTED skill (real `SkillTrustService` bootstrapped) → the editor shows `Saving marks this skill "needs review" — re-approve it in the trust panel after saving.` before saving; after Save, re-list shows that skill in `blocked_skills` (trust re-quarantined).
  - Trust panel approve: a needs-review skill → `Review changes` (capture_review) populates the changed-files list → `Approve` prompts the passphrase modal (fake `push_screen_wait` returning a passphrase) → `unlock_with_passphrase` + `trust_reviewed_snapshot` → re-list shows it in `available_skills`.
  - Delete → back to list, count decremented.
  - Dirty nav-away: `flush_pending_work` returns the same veto shape the prompts branch uses (read the prompts flush test first; mirror the exact contract).
- [ ] **Step 2:** FAIL run. **Step 3:** Implement editor compose + handlers:
  - **Open** (`handle_library_skill_row`): `detail = await skills_scope_service.get_skill(name, mode="local")` → `build_skill_editor_state(detail)` → render editor. Blocked rows open with the trust panel primed (changed_files from `detail["trust_changed_files"]`).
  - **Save** (`_save_library_skill`): gather fields → `content = compose_skill_markdown(state, body=…)`; for a new skill `create_skill(name=…, content=content, mode="local")`, for an existing one `update_skill(name, content=content, expected_version=state.version, mode="local")`; classify with `classify_skill_save_error`; on `ok` refresh snapshot + status `Saved.`; on `version-conflict` show a conflict bar (Reload); on `exists`/`invalid-name`/`trust-blocked` show the mapped copy. Compute the shadow warning live from `skill_name_shadows_builtin(name)` and the needs-review warning from `save_marks_needs_review(state.trust_status, state.trust_blocked)`; both are non-blocking Statics.
  - **Trust panel:** `Unlock` → passphrase modal → `unlock_with_passphrase(passphrase)` (mirror `skills_screen._handle_skill_trust_passphrase_action`); `Review changes` → `capture_review(name)` stores the active review + renders `changed_files`; `Approve` → passphrase modal → `unlock_with_passphrase` then `trust_reviewed_snapshot(review_id)`; refresh snapshot after each. Route every trust call through an `await`/`asyncio.to_thread` wrapper matching `skills_screen._call_skill_trust_service` (sync methods offloaded).
  - **Delete:** `delete_skill(name, expected_version=state.version, mode="local")` (single dim button, matching the notes/prompts delete affordance) → back to list + refresh.
  - Dirty tracking mirrors `_library_prompt_dirty` (@657 + `flush_pending_work`). Add CSS for the trust panel + dim blocked/needs-review lines to `_agentic_terminal.tcss`; regenerate; dual-pin one selector. **Step 4:** PASS + `test_library_shell.py` green. **Step 5:** Commit `feat(skills): SKILL.md editor + trust panel (passphrase reuse) + save/delete`.

### Task 5: Import + `skills` tab retirement + route alias

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (import path row), `tldw_chatbook/UI/Screens/library_screen.py` (import flow; `_open_library_item_by_id` skills case if the Library uses one), `tldw_chatbook/UI/Navigation/screen_registry.py` (alias `"skills": "library"`; remove the `"skills"` entry from `_SCREEN_ROUTES` @68), `tldw_chatbook/UI/Navigation/shell_destinations.py` if a nav-context selecting `LIBRARY_ROW_BROWSE_SKILLS` is threaded (mirror the `notes`/`prompts` alias precedent exactly — `grep -n '"prompts"' tldw_chatbook/UI/Navigation/*.py`)
- Test: `Tests/Skills/test_skills_import.py` (new), extend `Tests/UI/test_screen_navigation.py`

**Interfaces:**
- Produces: import row (path Input `library-skills-import-path`, `Browse…`, `Import`, `Cancel` — mirror the prompts import ids @6003–6051) validating via the same `validate_path_simple` the ingest form uses; imported skills land trust-pending (`import_skill_file`/`import_skill` with `trust_approved=False`) with the review panel primed; outcome line `1 imported · re-review it in the trust panel`. Route alias `skills → library` resolving to `LibraryScreen` with the skills rail row selected.
- Consumes: `skills_scope_service.import_skill_file(bytes, mode="local", filename=…)` / `import_skill(content=…, name=…, mode="local")`.

- [ ] **Step 1:** `grep -rn "SkillsScreen\|screen_registry.*skills\|\"skills\"" tldw_chatbook Tests | grep -v skills_screen.py` — confirm the only remaining hard references to the `skills` ROUTE are the registry entry + alias + nav callers; if a live nav caller depends on the `SkillsScreen` class directly, STOP and report rather than deleting (the class file may stay for the reused modal even after the route retires — do NOT delete `skills_screen.py`; only retire the route).
- [ ] **Step 2: Failing tests:**
  - Import (real service): write a temp `SKILL.md` (valid frontmatter) → run the import flow → `get_context` shows the new skill in `blocked_skills` (trust-pending) → outcome line asserted.
  - Nav alias (`test_screen_navigation.py`): `NavigateToScreen("skills")` (or the shell's resolve helper) lands on `LibraryScreen`; mirror the `notes`/`prompts` alias test (`grep -n '"prompts"' Tests/UI/test_screen_navigation.py`).
- [ ] **Step 3:** Implement import (path input + worker + per-file outcome) + the alias/retirement. `$PY -c "import tldw_chatbook.app"` clean. **Step 4:** Navigation + skills suites green. **Step 5:** Commit `feat(skills): import (trust-pending) + retire skills tab behind library alias`.

### Task 6: Phase-1 gate — suites + live captures (STOP for user approval)

- [ ] **Step 1:** `$PY -m pytest Tests/Skills Tests/Library Tests/UI/test_library_shell.py Tests/UI/test_library_skills_canvas.py Tests/UI/test_screen_navigation.py Tests/UI/test_destination_shells.py -q` → all green (no NEW failures vs the ~33-failure UI baseline).
- [ ] **Step 2:** Serve from the worktree (textual-serve recipe: scratchpad `serve_qa.py`, playwright bundled chromium 2050×1240, route-abort `https://**` only, gate on `body.-first-byte`). Seed ≥4 real skills into the profile's local skills store via `LocalSkillsService.create_skill`/`import_skill` (one trusted via a real `bootstrap_trust`, one needs-review by editing after approval, one with an `allowed_tools` list, one with `context: fork`). Capture: rail `Skills (4)`, list showing `✓`/`⚠` glyphs + flags line, the SKILL.md editor with the model "not applied in v1" hint, the trust panel review→approve via the passphrase modal, the save-marks-needs-review warning, the shadow-name warning, the import outcome line, and the `skills` alias landing on Library ▸ Skills. Save to `Docs/superpowers/qa/library-skills-2026-07/` with a README per QA convention.
- [ ] **Step 3:** Commit QA evidence. **STOP — present captures for user approval before Phase 2. No PR.**

---

## Phase 2 — Console: resolver, `/skills`, substitution, agent-tool

### Task 7: Pure skills resolver `console_skill_resolver.py`

**Files:**
- Create: `tldw_chatbook/Chat/console_skill_resolver.py`
- Test: `Tests/Chat/test_console_skill_resolver.py`

**Interfaces (pure — no Textual/app/IO):**

```python
@dataclass(frozen=True)
class SkillCommandCandidate:
    name: str
    description: str = ""

@dataclass(frozen=True)
class SkillResolution:
    kind: str                 # "resolved" | "ambiguous" | "none"
    name: str = ""            # canonical skill name when resolved
    matches: tuple[str, ...] = ()

SKILL_ARGS_MAX = 4000
SKILLS_LIST_COMMAND_NAME = "skills"
SKILL_UNTRUSTED_REFUSE = 'Skill "{name}" isn\'t trusted ({reason}) — review and approve it in Library ▸ Skills before running it.'
SKILLS_EMPTY_LIST_ROW = "No skills yet — create them in Library ▸ Skills."

def resolve_skill_command(word, args, candidates) -> SkillResolution: ...
def cap_skill_args(args: str) -> str: ...                 # trim to SKILL_ARGS_MAX chars
def format_skills_list(candidates) -> str: ...            # transcript system-row text; empty → SKILLS_EMPTY_LIST_ROW
def make_skill_fallback_resolver(candidates_getter) -> Callable[[str, str], "CommandParse | None"]:
    # returns KIND_FALLBACK(name=word, args=cap_skill_args(rest)) when resolve is "resolved"/"ambiguous"; None otherwise
```

`resolve_skill_command` rules: exact case-insensitive name match → `resolved`; else unique case-insensitive name-PREFIX match → `resolved` (canonical name); else 2+ prefix matches → `ambiguous(matches=…)`; else `none`. Args are pre-capped by callers. The fallback resolver claims (returns a `CommandParse`) ONLY when the word plausibly matches a cached skill (so unknown words still fall through to the unknown-command hint); the async dispatch re-resolves authoritatively.

- [ ] **Step 1: Failing tests** (literal cases):

```python
from tldw_chatbook.Chat.console_command_grammar import KIND_FALLBACK
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILLS_EMPTY_LIST_ROW, SkillCommandCandidate, cap_skill_args, format_skills_list,
    make_skill_fallback_resolver, resolve_skill_command,
)


def _cands(*names):
    return tuple(SkillCommandCandidate(n, f"{n} desc") for n in names)


def test_exact_case_insensitive():
    r = resolve_skill_command("Code-Review", "x", _cands("code-review", "summarize"))
    assert (r.kind, r.name) == ("resolved", "code-review")


def test_unique_prefix():
    r = resolve_skill_command("summ", "", _cands("summarize", "code-review"))
    assert (r.kind, r.name) == ("resolved", "summarize")


def test_ambiguous_prefix():
    r = resolve_skill_command("s", "", _cands("summarize", "scan"))
    assert r.kind == "ambiguous" and set(r.matches) == {"summarize", "scan"}


def test_no_match():
    assert resolve_skill_command("zzz", "", _cands("summarize")).kind == "none"


def test_exact_wins_over_prefix():
    r = resolve_skill_command("scan", "", _cands("scan", "scanner"))
    assert (r.kind, r.name) == ("resolved", "scan")


def test_cap_args():
    assert len(cap_skill_args("x" * 10000)) == 4000


def test_format_list_empty():
    assert format_skills_list(()) == SKILLS_EMPTY_LIST_ROW


def test_format_list_lines_include_name_and_desc():
    text = format_skills_list(_cands("summarize"))
    assert "summarize" in text and "summarize desc" in text


def test_fallback_claims_matching_word_only():
    resolver = make_skill_fallback_resolver(lambda: _cands("summarize"))
    claimed = resolver("summ", "the doc")
    assert claimed is not None and claimed.kind == KIND_FALLBACK and claimed.name == "summ"
    assert resolver("unknownword", "x") is None
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement (~90 lines, imports only `console_command_grammar` types + stdlib). **Step 4:** PASS. **Step 5:** Commit `feat(console): pure skills command resolver`.

### Task 8: Skill picker modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_skill_picker_modal.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (picker classes if not reusing the prompt picker's) + regenerate
- Test: `Tests/UI/test_console_skill_picker.py`

**Interfaces:**
- Produces: `ConsoleSkillPickerModal(*, initial_query: str, skill_search: Callable[[str], Awaitable[list[Mapping]]])` — dismisses with `Mapping | None` (the chosen skill summary: at least `{"name","description"}`). Filter Input id `console-skill-picker-filter`, row Buttons `console-skill-picker-row-<name>`, empty line `No skills yet — create them in Library ▸ Skills.` `skill_search` returns trusted user-invocable skills matching the query (bounded ≤ 25), called fresh per filter change (debounce 200ms). Keyboard-first (↑/↓/Enter/Esc), mirroring `ConsolePromptPickerModal` exactly (copy its structure).
- Consumes: a scope-service-backed async `skill_search` closure (Task 9 supplies it from `get_context` filtered client-side by query — `get_context` has no server filter, so filter the returned `available_skills` in the closure).

- [ ] **Step 1: Failing tests:** type-to-filter calls `skill_search` with the query; Enter on the highlighted row dismisses with that record; Esc dismisses None; empty store shows the exact empty line; a description containing `[x]` renders literally (escape proof) in the row's markup-enabled label. Use a fake async `skill_search`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (copy `console_prompt_picker_modal.py`; rename ids/classes; drop the `apply-system` mode — skills have one mode). **Step 4:** PASS. **Step 5:** Commit `feat(console): skill picker modal`.

### Task 9: Register `/skills` + fallback resolver + dispatch (list / run / refuse)

**Files:**
- Modify: `tldw_chatbook/Chat/console_command_grammar.py` (add `SKILLS_COMMAND_NAME="skills"`, `SKILLS_COMMAND_ARGUMENT_HINT="[name] [args]"`, `SKILLS_COMMAND_HANDLER_ID="skills"`, register in `default_console_registry`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (register the fallback resolver on `self._console_command_registry` after construction; extend `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` + `_dispatch_console_command`; add `_console_command_skills`, `_console_command_run_skill`, `_append_skill_refuse_row`, a cached skill-name snapshot + refresh hook; update `_console_unknown_command_hint` to derive from `available_names()`)
- Test: `Tests/Chat/test_console_command_grammar.py` (extend), `Tests/UI/test_console_skill_commands.py` (new)

**Interfaces:**
- Consumes: Task 7 resolver, `skills_scope_service.get_context`/`execute_skill`, `_submit_console_native_draft` (the raw-command send), Task 8 picker.
- Produces:
  - `/skills` REGISTERED command: bare `/skills` → a transcript system row from `format_skills_list(candidates)` (persist=False, via `_append_native_console_system_message`); `/skills <name> [args]` → run.
  - Bare `/skill-name [args]` fallback: dispatch re-resolves (fresh `get_context`), then runs.
  - **Run semantics (spec §Slash surface):** resolve against user-invocable trusted skills; untrusted/edited/absent → `_append_skill_refuse_row` with `SKILL_UNTRUSTED_REFUSE` (draft preserved, nothing runs); resolved-single → SEND THE RAW COMMAND as the user turn via `_submit_console_native_draft(raw_command)` (respecting the send-blocked/readiness gate tail @7260–7279) — the substitution rule (Task 10) renders it at payload build; ambiguous/0-with-args → open the skill picker prefilled. On resolved-single also append the marker row `skill {name} → driving this turn` (TOOL role, raw) via `_append_marker`-style store append AFTER the user turn is queued.
  - `_dispatch_console_command`: `KIND_FALLBACK` → `_console_command_run_skill(parse.name, parse.args)`; `KIND_COMMAND` name `"skills"` → `_console_command_skills(parse)` (existing `/prompt`/`/system` unchanged).

- [ ] **Step 1: Failing tests:**
  - Grammar (`test_console_command_grammar.py`): `default_console_registry().available_names()` includes `"skills"`; `parse("/skills code-review go")` → `KIND_COMMAND, name="skills", args="code-review go"`; with a fallback resolver registered that claims `summ`, `parse("/summ the doc")` → `KIND_FALLBACK, name="summ"`.
  - UI (`test_console_skill_commands.py`, real App + fake `skills_scope_service` on the app): bare `/skills` with two trusted skills → a system row listing both names; `/skills unknownskill` → the refuse row with the exact `SKILL_UNTRUSTED_REFUSE` copy (name `unknownskill`); `/code-review fix it` on a trusted skill → `_submit_console_native_draft` spy called with the raw `"/code-review fix it"` text AND a TOOL marker row `skill code-review → driving this turn` present; an untrusted skill (present but `trust_blocked`) → refuse row, submit spy NOT called.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. For the fallback resolver's cached candidate snapshot: refresh it when the Console screen mounts/activates and after a skills-affecting event (a lazy `self._console_skill_candidates: tuple[SkillCommandCandidate, ...] = ()` refreshed by an async `_refresh_console_skill_candidates()` worker calling `get_context(mode="local")`, filtered to `user_invocable and not trust_blocked`). The fallback resolver closes over `lambda: self._console_skill_candidates`. The async run/dispatch re-fetches `get_context` fresh for the authoritative trust decision (snapshot may be stale). Update `_console_unknown_command_hint` to `f"Unknown command /{name} — available: {', '.join('/' + n for n in self._console_command_registry.available_names())}. Press Enter again to send as text."` **Step 4:** PASS + `Tests/UI/test_console_command_composer.py`, `test_console_native_chat_flow.py` green. **Step 5:** Commit `feat(console): /skills command + bare /skill-name fallback + refuse copy`.

### Task 10: Provider-payload substitution rule (final-user-message-only)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (add async `_apply_skill_substitution(provider_messages) -> tuple[list[dict], str | None]`; call it in the submit / regenerate / continue send paths after building `provider_messages`; a `skills_service`/`skill_candidates_getter` injected via the ctor like `agent_bridge`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (pass the substitution deps into the controller ctor @2228/@2284)
- Test: `Tests/Chat/test_console_skill_substitution.py` (new)

**Interfaces:**
- Consumes: Task 7 `resolve_skill_command`; `skills_scope_service.get_context` (fresh candidates) + `execute_skill` (render, re-trust, re-verify, `{{args}}`).
- Produces: `ConsoleChatController(..., skills_service=None, skill_substitution_enabled=True)`. `_apply_skill_substitution(messages)`:
  1. No skills_service → return `(messages, None)` (no-op).
  2. Find the FINAL message with `role == "user"` (the triggering message). Parse its content with `console_command_grammar`-style tokenization + `resolve_skill_command` over fresh `get_context` user-invocable-trusted candidates. Not a resolvable skill command → `(messages, None)`.
  3. Resolved → `result = await skills_service.execute_skill(name, mode="local", args=capped_args)`. On `SkillTrustBlockedError` (edited-since-approval at retry) → return `(messages, SKILL_UNTRUSTED_REFUSE.format(name=name, reason=exc.reason_code))` — the caller appends the refuse row and aborts the turn.
  4. Replace that final user message's content with `result["rendered_prompt"]`. If `result["execution_mode"] == "fork"` → drop every message before it EXCEPT a leading `role == "system"` message (clean context = session system prompt + rendered turn only). Return `(new_messages, None)`.
- Wired at every provider-payload build site: `submit_draft` (@206), `regenerate_message` (@508), and the continue/retry builder — one rule, so retries re-render + re-trust (spec §Substitution rule). The send methods do `provider_messages, refuse = await self._apply_skill_substitution(provider_messages)`; if `refuse` is set they append the refuse system row and return a not-accepted result without running.

- [ ] **Step 1: Failing tests** (fake `skills_service` with async `get_context` + `execute_skill`):

```python
import pytest
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


class _Skills:
    def __init__(self, mode="inline", raise_trust=False):
        self._mode = mode
        self._raise = raise_trust

    async def get_context(self, *, mode="local"):
        return {"available_skills": [{"name": "code-review", "description": "d",
                                      "user_invocable": True, "trust_blocked": False}],
                "blocked_skills": []}

    async def execute_skill(self, name, *, mode="local", args=None):
        if self._raise:
            raise SkillTrustBlockedError(skill_name=name, reason_code="skill_modified",
                                         trust_status="quarantined_modified")
        return {"skill_name": name, "rendered_prompt": f"RENDERED[{args}]",
                "allowed_tools": None, "execution_mode": self._mode, "fork_output": None}


def _controller(skills):
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=object(),
                                 provider="llama_cpp", model="m",
                                 skills_service=skills), store


@pytest.mark.asyncio
async def test_inline_substitutes_final_user_message_only():
    controller, _store = _controller(_Skills("inline"))
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "/code-review fix it"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert refuse is None
    assert out[-1] == {"role": "user", "content": "RENDERED[fix it]"}
    assert out[1] == {"role": "user", "content": "earlier"}    # history preserved


@pytest.mark.asyncio
async def test_fork_drops_history_keeps_system():
    controller, _store = _controller(_Skills("fork"))
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier"},
            {"role": "user", "content": "/code-review go"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert refuse is None
    assert out == [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "RENDERED[go]"}]


@pytest.mark.asyncio
async def test_non_skill_final_message_unchanged():
    controller, _store = _controller(_Skills("inline"))
    msgs = [{"role": "user", "content": "just a question"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert out == msgs and refuse is None


@pytest.mark.asyncio
async def test_edited_skill_refuses_at_build():
    controller, _store = _controller(_Skills(raise_trust=True))
    msgs = [{"role": "user", "content": "/code-review go"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert out == msgs
    assert refuse == ('Skill "code-review" isn\'t trusted (skill_modified) — '
                      "review and approve it in Library ▸ Skills before running it.")
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement `_apply_skill_substitution` + ctor deps; wire the three send sites (each: build `provider_messages`; `provider_messages, refuse = await self._apply_skill_substitution(provider_messages)`; if `refuse`: append refuse row via the same system-row mechanism the agent-error rows use + return not-accepted). Import `resolve_skill_command`/`cap_skill_args`/`SKILL_UNTRUSTED_REFUSE` from `console_skill_resolver` and the tokenizer split from `console_command_grammar`. **Step 4:** PASS + `Tests/Chat -k "console and (controller or substitution)"` green. **Step 5:** Commit `feat(console): render-fresh skill substitution rule for the triggering turn`.

### Task 11: `SkillToolProvider` + `intersect_skill_tools` helper

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (add `SkillToolProvider` + `intersect_skill_tools`)
- Test: `Tests/Agents/test_skill_tool_provider.py`

**Interfaces:**
- Produces:
  - `intersect_skill_tools(skill_allowed_tools: list[str] | None, builtin_names: Iterable[str]) -> tuple[str, ...]` — `None` → all builtins (skill did not narrow); a list → `[t for t in list if t in builtins]` (never grants; order = builtin order). Pure.
  - `SkillToolProvider(entries: list[Mapping])` — built from a per-run snapshot of trusted, model-invocable skill summaries (`{"name","description","argument_hint"}`). `list_catalog()` = `ToolCatalogEntry(id=f"skill:{name}", name=name, one_line_description=description, source="skill")`; `load_schema(tool_id)` = single-string schema `ToolSchema(id, name, description, parameters={"type":"object","properties":{"args":{"type":"string","description": argument_hint or description}},"required":[]})`; `invoke(tool_id, args)` RAISES `RuntimeError("SkillToolProvider.invoke must not be called; skills route through the run-scoped spawn executor")` — the guard for future misuse outside a run context (spec Architecture bullet).

- [ ] **Step 1: Failing tests:**

```python
import pytest
from tldw_chatbook.Agents.tool_catalog import SkillToolProvider, intersect_skill_tools


def test_intersect_none_is_all_builtins():
    assert intersect_skill_tools(None, ["calculator", "get_current_datetime"]) == (
        "calculator", "get_current_datetime")


def test_intersect_narrows_never_grants():
    assert intersect_skill_tools(["calculator", "nonexistent"],
                                 ["calculator", "get_current_datetime"]) == ("calculator",)


def test_provider_catalog_and_schema():
    prov = SkillToolProvider([
        {"name": "code-review", "description": "Review code", "argument_hint": "[path]"}])
    entry = prov.list_catalog()[0]
    assert (entry.id, entry.name, entry.source) == ("skill:code-review", "code-review", "skill")
    schema = prov.load_schema("skill:code-review")
    assert schema.name == "code-review"
    assert schema.parameters["properties"]["args"]["type"] == "string"
    assert schema.parameters["properties"]["args"]["description"] == "[path]"


def test_invoke_raises_by_design():
    prov = SkillToolProvider([{"name": "x", "description": "d", "argument_hint": None}])
    with pytest.raises(RuntimeError):
        prov.invoke("skill:x", {"args": "y"})
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement (~50 lines). **Step 4:** PASS + `Tests/Agents -q` green. **Step 5:** Commit `feat(agents): SkillToolProvider (catalog/schema; invoke raises) + intersect helper`.

### Task 12: Per-run spawn-wired skill executor + run allow-list at the bridge

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`AgentService(..., skill_runner=None)`; `spawn` gains an `allowed_tools` override; skill-aware `invoke_tool` with its own `max_subagents`-bounded counter)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (per-run registry = builtins + a fresh `SkillToolProvider` snapshot; per-run allow-list = builtins ∪ eligible skill names ∪ spawn; a `_BridgeSkillRunner` wrapping `execute_skill` + `intersect_skill_tools`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (pass the skills service into `_ensure_console_agent_bridge`)
- Test: `Tests/Agents/test_skill_tool_spawn.py` (new), `Tests/Chat/test_console_agent_bridge.py` (extend)

**Interfaces:**
- Produces:
  - `agent_service`: a `SkillRunner` protocol (`is_skill_tool(name) -> bool`; `run(name, args, spawn) -> ToolResult` where `spawn(task, *, allowed_tools)` is the run's spawn). In `_run_one`, `spawn` becomes `def spawn(spawn_task, *, allowed_tools=None)` — child `allowed_tools = allowed_tools if allowed_tools is not None else tuple(n for n in config.allowed_tools if n != SPAWN_TOOL_NAME)` (default preserves the shipped behavior; `deps.spawn` still calls `spawn(task)`). `invoke_tool` (built after `spawn`): if `skill_runner and skill_runner.is_skill_tool(call.name)` → gate (`call.name in config.allowed_tools and in disclosed_names`, else "Tool not permitted"), then a service-side `max_subagents` check (a `_run_one`-local `skill_spawns` counter → `"sub-agent budget exhausted"` when exhausted), then `skill_runner.run(call.name, str(call.args.get("args","")), spawn)` (which renders → `spawn(rendered, allowed_tools=intersected)`); else the shipped builtin path.
  - `console_agent_bridge`: `ConsoleAgentBridge(..., skills_service=None)`. `run_reply` builds a per-run registry (`ToolCatalogRegistry()` + `BuiltinToolProvider()` + `SkillToolProvider(snapshot)`) and per-run allow-list (`builtin names + skill names + SPAWN_TOOL_NAME`) when `skills_service` is present, from a fresh `asyncio.run(skills_service.get_context(mode="local"))` filtered to `not trust_blocked and not disable_model_invocation`; else the shipped shared registry/`_allowed_tools`. `_BridgeSkillRunner.run` renders via `asyncio.run(skills_service.execute_skill(name, mode="local", args=args))`, catches `SkillTrustBlockedError` → `ToolResult(ok=False, error=refuse-copy)`, computes `allowed_tools = intersect_skill_tools(result["allowed_tools"], builtin_names)`, and calls `spawn(result["rendered_prompt"], allowed_tools=allowed_tools)`.
- Depth/skills-can't-call-skills: the child runs with `intersect_skill_tools(...)` (builtins only — a skill's tools can never include another skill name, since the allow-list excludes skill names) and `clamp_child_budget` (max_subagents=0). So a skill sub-agent sees no spawn tool and no skill tools even under find/load (both gates filter by the child's builtin-only allow-list). Documented: `spawn_subagent` and skill-tool sub-agents keep independent `max_subagents` counters in v1 (both bounded, depth-1) — an accepted simplification.

- [ ] **Step 1: Failing test** `Tests/Agents/test_skill_tool_spawn.py` (real `AgentService` + `AgentRunsDB`, scripted `chat_call`, a fake `SkillRunner`):

```python
import json
import pytest
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, SPAWN_TOOL_NAME, RUN_DONE, ToolResult
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _FakeSkillRunner:
    def __init__(self):
        self.spawned_with = None

    def is_skill_tool(self, name):
        return name == "code-review"

    def run(self, name, args, spawn):
        self.spawned_with = args
        return spawn(f"RENDERED[{args}]", allowed_tools=("calculator",))


def test_skill_tool_routes_through_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry(); reg.register_provider(BuiltinToolProvider())
    script = [
        {"choices": [{"message": {"content": _fence("code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "child answer"}}]},   # sub-agent turn
        {"choices": [{"message": {"content": "Done reviewing."}}]},  # primary final
    ]
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0), skill_runner=runner)
    _run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "review"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget()),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert runner.spawned_with == "the diff"
    assert db.count_subagent_runs("c1") == 1          # skill ran as a budget-counted sub-agent


def test_skill_tool_respects_subagent_budget(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry(); reg.register_provider(BuiltinToolProvider())
    # Two skill calls with max_subagents=1: the second must be refused.
    script = [
        {"choices": [{"message": {"content": _fence("code-review", {"args": "a"})}}]},
        {"choices": [{"message": {"content": "child a"}}]},
        {"choices": [{"message": {"content": _fence("code-review", {"args": "b"})}}]},
        {"choices": [{"message": {"content": "child b never"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0),
                           skill_runner=_FakeSkillRunner())
    _r, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_subagents=1, max_steps=12)),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1          # second skill spawn refused by budget
```

Plus extend `Tests/Chat/test_console_agent_bridge.py`: with a fake `skills_service` (async `get_context` returning one trusted model-invocable skill + async `execute_skill` rendering it), a scripted gateway whose first turn is `_fence("code-review", {"args":"x"})`, assert `db.count_subagent_runs(conversation_id) == 1` and a spawn/tool TOOL marker exists. Also assert the per-run allow-list contains `"code-review"` (drive via a run and inspect the persisted primary run's `budget`/steps, or expose a tiny `bridge._compose_run_allowed_tools(context)` pure helper and unit-test it directly).

- [ ] **Step 2:** FAIL. **Step 3:** Implement. In `agent_service._run_one`: give `spawn` the `allowed_tools` keyword (default preserves current behavior); build the skill-aware `invoke_tool` AFTER `spawn` with a `skill_spawns` counter; pass it to `LoopDeps(invoke_tool=invoke_tool, spawn=spawn, ...)` (replace `self._make_invoke_tool(...)`). In the bridge: add `skills_service` param; extract `_compose_run_registry_and_allowed(context) -> (registry, allowed_tools, builtin_names)` and `_BridgeSkillRunner`; build them per-run in `run_reply` and pass `skill_runner` into `AgentService(...)`. In `chat_screen._ensure_console_agent_bridge` (@2078) pass `skills_service=getattr(self.app_instance, "skills_scope_service", None)`. **Step 4:** PASS + `Tests/Agents Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py` green (no-skills path byte-identical). **Step 5:** Commit `feat(console): per-run skill catalog + spawn-wired skill-tool executor`.

### Task 13: Catalog fixes — loop-side dedupe + per-run owner-map cache (with regressions)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (LOAD_TOOLS branch: dedupe `loaded` against active names before slicing — F1-b)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (`ToolCatalogRegistry` owner-map cache + `reset_catalog_cache()`; replace the `TODO(task-201)` re-listing in `_owner_and_id` @137)
- Modify: `tldw_chatbook/Agents/agent_service.py` (`run_turn` calls `self.registry.reset_catalog_cache()` at the start — per-run scope)
- Test: `Tests/Agents/test_tool_catalog_owner_cache.py` (new), `Tests/Agents/test_agent_loop_load_dedupe.py` (new)

**Interfaces:**
- Produces:
  - `agent_runtime.run_agent_loop` LOAD_TOOLS branch (@326–341): before `room = budget.max_active_tools - len(active)`, filter `loaded = [s for s in loaded if s.name not in {a.name for a in active}]`; if the filter empties `loaded`, treat as "no room" (`ToolResult(ok=True, content="no room")`) rather than the "No valid tools found" error (which is for the all-invalid case) — a re-load of an already-active tool is a no-op, not an error. Keeps the loop's `active` free of duplicate names even if a provider returns an already-active schema (F1-b desync, plan-a-final-review).
  - `tool_catalog.ToolCatalogRegistry`: `self._owner_cache: dict[str, ToolProvider] = {}`; `_owner_and_id` consults/populates the cache; `reset_catalog_cache()` clears it. The cache is scoped PER RUN via `AgentService.run_turn`'s reset — the catalog is listed fresh at run start, so skill CRUD between runs is always picked up and no cross-run invalidation signal is needed (spec §Catalog scale; retires the `TODO(task-201)` marker). Comment: this is the fix MCP (task-201) also needs — a network-backed provider must not re-`list_catalog()` per lookup.

- [ ] **Step 1: Failing tests:**

```python
# Tests/Agents/test_tool_catalog_owner_cache.py
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolSchema, ToolResult


class _CountingProvider:
    def __init__(self):
        self.list_calls = 0

    def list_catalog(self):
        self.list_calls += 1
        return [ToolCatalogEntry(id="p:foo", name="foo", one_line_description="d", source="p")]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name="foo", description="d", parameters={})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="x")


def test_owner_lookup_is_cached_within_a_run():
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)
    reg.load_schema("p:foo")
    calls_after_first = prov.list_calls
    reg.load_schema("p:foo")
    reg.invoke_by_name("foo", {})
    # The owner map is cached: no fresh list_catalog() re-listing per lookup.
    assert prov.list_calls <= calls_after_first + 1  # resolve_name may list once; owner is cached


def test_reset_catalog_cache_picks_up_new_run():
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)
    reg.load_schema("p:foo")
    reg.reset_catalog_cache()
    before = prov.list_calls
    reg.load_schema("p:foo")
    assert prov.list_calls > before  # cache cleared → re-listed for the new run
```

```python
# Tests/Agents/test_agent_loop_load_dedupe.py
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, ToolSchema
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop, FENCE_OPEN
import json


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


def test_load_tools_never_duplicates_an_active_schema():
    active = [ToolSchema(id="p:foo", name="foo", description="d", parameters={})]
    turns = iter([
        type("M", (), {"text": _fence("load_tools", {"ids": ["p:foo"]}), "tool_calls": ()})(),
        type("M", (), {"text": "done", "tool_calls": ()})(),
    ])
    seen_active_sizes = []

    def call_model(messages, active_schemas):
        seen_active_sizes.append(len(active_schemas))
        return next(turns)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda call: None,
        spawn=lambda task, **k: None,
        find_tools=lambda q: [],
        load_schemas=lambda ids: [ToolSchema(id="p:foo", name="foo", description="d", parameters={})],
        should_cancel=lambda: False,
        clock=lambda: 0.0)
    outcome = run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("foo", "load_tools"),
                    budget=RunBudget(max_active_tools=8)),
        [{"role": "user", "content": "hi"}], active, deps)
    # The already-active "foo" schema is not appended a second time.
    assert seen_active_sizes[-1] == 1
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement both fixes + the `run_turn` reset call. **Step 4:** PASS + the FULL `Tests/Agents -q` suite green (the loop/catalog changes must not regress any shipped agent test). **Step 5:** Commit `refactor(agents): loop-side load dedupe + per-run owner-map cache`.

### Task 14: Phase-2 live gate — served captures + suite sweep (STOP)

**Files:** none shipped (captures are artifacts). Use the established served-capture recipe (`textual-serve` @ 2050×1240; the `cap.py` https-only route-abort + body-first-byte recipe).

- [ ] **Step 1: Broad suite sweep**
```
$PY -m pytest Tests/Skills Tests/Agents Tests/Chat/test_console_skill_resolver.py Tests/Chat/test_console_skill_substitution.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_command_grammar.py Tests/UI/test_console_skill_commands.py Tests/UI/test_console_skill_picker.py Tests/UI/test_console_command_composer.py Tests/UI/test_library_skills_canvas.py -q
```
Expected: ALL PASS. Then the broad regression: `$PY -m pytest Tests/UI Tests/Chat Tests/Agents -q` — confirm NO new failures vs the ~33-failure UI baseline.
- [ ] **Step 2: Live served captures on llama.cpp (2050×1240)** — point the Console at a tool-capable local model with `[console] agent_runtime = true`. **Seed >8 skills** into the local store (so progressive disclosure actually engages), at least one trusted `context: inline`, one trusted `context: fork`, one edited-since-approval (untrusted). Drive:
  1. `/skill-name args` on a trusted `inline` skill → the raw command persists as the user turn, a `skill … → driving this turn` marker, and a real reply that used the rendered body (verify the on-wire payload's final user message = the rendered body, not the raw command).
  2. `/skill-name` on a trusted `fork` skill → clean-context reply (no history bleed).
  3. `/skill-name` on the edited skill → the exact `SKILL_UNTRUSTED_REFUSE` transcript row; nothing runs.
  4. A question that makes the agent discover + call a skill mid-run (>8 skills → `find_tools`/`load_tools` engage) → a skill TOOL marker + a budget-counted, `parent_run_id`-linked sub-agent run + the `[1 Sub-Agents]` badge.
  5. The catalog-fix regressions (dedupe + cache) confirmed green under the served run's logs (no duplicate-active desync, no per-lookup re-list storm).
  Save under `Docs/superpowers/qa/console-skills-2026-07/` with a README (request-capture method stated).
- [ ] **Step 3: STOP for user approval.** Do NOT open a PR. Present captures + suite results; request explicit sign-off (program approval-gate convention). Note any live-model degradations (e.g. the model ignoring fence-first → graceful plain-answer fallback) for the follow-up list.

---

## Out of scope (named follow-ups)

- **Skill `model` override** (deferred — needs a one-turn provider/model override path; the editor only shows a "not applied in v1" hint).
- **Trust REVOKE** — the shipped `SkillTrustService`/`SkillTrustStore` expose approve (`trust_reviewed_snapshot`) + diff (`capture_review`) but **no revoke/untrust primitive**; the Library trust panel ships unlock + review + approve only. A revoke button awaits a `SkillTrustService.revoke_skill` API (backlog) — do NOT fabricate one.
- **Skills-calling-skills** (a skill sub-agent's allow-list is builtins-only by construction; depth stays 1).
- **Supporting-file editing in the detail canvas** (read-only names + sizes in v1; add/remove via import).
- **Server-side skills surfaces** beyond what `skills_scope_service` already routes (`mode="server"`).
- **Skill marketplace / discovery.**
- ~~**Unified sub-agent budget** across `spawn_subagent` and skill-tool spawns (two independent `max_subagents` counters in v1; both bounded, depth-1).~~ **Stale — actually shipped.** Task 12 review Finding 2 folded this in during implementation: `agent_service.py`'s `_run_one` uses a single `sub_agent_spawns` counter shared by both the native `spawn_subagent` path and the skill-tool path (both call the same `spawn` closure), so `RunBudget.max_subagents` bounds the COMBINED count, not two independent counters. Regression-locked by `Tests/Agents/test_skill_tool_spawn.py::test_combined_budget_native_spawn_then_skill_call`/`test_combined_budget_skill_call_then_native_spawn`.

---

## Self-review

**Spec coverage (every section walked):**
- §Purpose / §Decisions "Both invocation surfaces, flag-gated" — slash surface (Tasks 7–10) gated by `user_invocable` (candidate filter); agent-tool surface (Tasks 11–12) gated by `disable_model_invocation` (SkillToolProvider snapshot filter). ✔
- §Decisions "Home: Library ▸ Skills … skills → library route alias" — Tasks 1–5 (rail row, canvases, import, alias). ✔
- §Decisions "Trust UX: refuse + point to Library" — refuse copy constant (Task 7), Console refuse rows (Tasks 9/10), approval only in Library (Task 4). ✔
- §Review corrections 1–5 — spawn-path (Task 12), render-vs-persist (Tasks 9/10), model deferred (Task 4 hint + Out-of-scope), fork=clean-context (Task 10), built-ins shadow + editor warning (Tasks 2/4/9). ✔
- §Architecture units — `library_skills_state.py` (Task 2), `library_skills_canvas.py` (Tasks 3/4), `console_skill_resolver.py` (Task 7), `SkillToolProvider` + per-run spawn wiring (Tasks 11/12), Console integration via `execute_skill` + narrowed AgentConfig (Tasks 9/10/12). ✔
- §Invocation semantics (slash 1–5) — resolve trusted-only, `execute_skill` render, per-turn AgentConfig intersect, raw-command transcript + marker, one substitution rule for sends+retries. Tasks 9/10/12. ✔
- §Invocation semantics (agent-tool) — catalog visibility + disclosure + args-string invoke → run-scoped sub-agent, no skills-calling-skills. Tasks 11/12. ✔
- §Security model — trust gates twice (candidate/snapshot filter + `execute_skill` re-verify), intersect never grants, run allow-list = builtins ∪ eligible skills fresh at run start, escaping discipline. Tasks 9/10/11/12 + Global Constraints. ✔
- §Catalog scale — progressive disclosure engages (>8 skills), both catalog fixes land WITH regressions. Task 13 + Task 14 live gate seeds >8. ✔
- §Error handling — empty `/skills` list row copy, render-failure refuse rows (draft preserved), sub-agent failure feeds parent (shipped `spawn` result capping), import per-file trust-pending. Tasks 7/9/10/5. ✔
- §Testing (pure / service / UI / live gate) — pure (Tasks 2/7/11/13), service (Tasks 1/4/12), UI (Tasks 3/4/8/9), live gate STOP (Tasks 6/14). ✔
- §Out of scope — restated with the revoke reality (no shipped API). ✔

**Placeholder scan:** every task carries Files/Interfaces + failing test + implementation direction + run/commit; no `TODO`/`...`/`TBD` left in shipped code paths (the one `TODO(task-201)` marker is explicitly REMOVED in Task 13). Pure modules (Tasks 2/7/11/13) have complete test code; UI/service tasks (3/4/5/8/9/10/12) have complete pure/service test code + precise UI assertions + exact file:line anchors.

**Type consistency vs the REAL code read:** `execute_skill` return keys (`rendered_prompt`/`allowed_tools`/`execution_mode`) and its `SkillTrustBlockedError` raise (Task 10/12) match `local_skills_service.py`; `get_context` `available_skills`/`blocked_skills` shape (Tasks 1/2/9/12) matches; `AgentConfig(model, system_prompt, allowed_tools, budget)` + `spawn` closure + `_make_invoke_tool` gate + `clamp_child_budget` (Task 12) match `agent_service.py`/`agent_models.py`; LOAD_TOOLS room-slice (Task 13) matches `agent_runtime.py` @332; `ToolCatalogRegistry._owner_and_id`/`initial_disclosure`/`ToolCatalogEntry`/`ToolSchema` (Tasks 11/13) match `tool_catalog.py`; `CommandParse`/`register_fallback_resolver`/`KIND_FALLBACK`/`available_names` (Tasks 7/9) match `console_command_grammar.py`; `_provider_messages_for_session`/`_leading_system_message` role-`"system"` leading message (Task 10) match `console_chat_controller.py`; `SkillTrustPassphraseModal(*, confirm_bootstrap)` + `capture_review`/`trust_reviewed_snapshot`/`unlock_with_passphrase` (Task 4) match `skills_screen.py`/`skill_trust_service.py`; `LIBRARY_ROW_BROWSE_*` + `_list_local_source_snapshot` optional-call pattern (Task 1) match `library_shell_state.py`/`library_screen.py`; `_SCREEN_ALIASES` precedent (Task 5) matches `screen_registry.py`. Builtin tool names verified as `calculator` + `get_current_datetime` (Task 2 shadow set). No `revoke` method exists — Task 4 ships approve/review/unlock only; revoke is Out-of-scope.

**Executor notes:** `chat_screen.py`/`console_chat_controller.py` are high-churn — rebase onto `origin/dev` before Phase 2 and re-run `Tests/Chat/test_console_agent_swap.py` + `Tests/UI/test_console_command_composer.py` first; any new failure there is yours. Read each real method before editing — the plan pins behavior + shapes, not unverified line numbers (anchors are from 9463a8a2 and may drift).
