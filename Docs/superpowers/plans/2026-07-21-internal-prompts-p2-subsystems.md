# Internal Prompts P2 — Agents / Summarization / Doc-Gen / Subscriptions Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the remaining 19 curated internal prompts onto the P1 `Internal_Prompts` registry: agents (3), summarization (3), document generation (6), subscriptions (7) — plus file the P1 follow-up backlog tasks.

**Architecture:** Same pattern P1 shipped (merged as PR #741): per-subsystem `*_prompts.py` spec modules registered in `tldw_chatbook/Internal_Prompts/__init__.py`; call sites resolve via `get_internal_prompt`/`render_internal_prompt`; golden-parity tests embed the original literals; one transport-boundary integration test per subsystem. Spec authority: `Docs/superpowers/specs/2026-07-21-internal-prompts-settings-page-design.md` §1-§6. Reference implementations to mirror: `tldw_chatbook/Internal_Prompts/websearch_prompts.py`, `Tests/Internal_Prompts/test_websearch_prompt_parity.py`, `Tests/Web_Scraping/test_websearch_internal_prompts.py`.

**Tech Stack:** Python ≥3.11, pytest; existing registry API only — `PromptSpec`, `register`, `CATALOG`, `get_internal_prompt(prompt_id)`, `render_internal_prompt(prompt_id, **values)`, `safe_substitute(text, **values)`.

## Global Constraints

- **Worktree:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/internal-prompts-p1`, branch `feat/internal-prompts-p2-subsystems` (off dev e74fea705). Verify `git rev-parse --abbrev-ref HEAD` before any write. venv-only pytest: `.venv/bin/python -m pytest` from worktree root. No `timeout` command. Never broad-kill pytest.
- **Byte-identical defaults**: copy prompt literals VERBATIM (indentation, blank lines, trailing spaces, `<s>`/`{{ .Prompt }}` cruft, typos). Parity tests enforce; fix copies, never assertions.
- **Never `.format()`-family on resolved text.** Zero-placeholder prompts fetched via `get_internal_prompt` may be concatenated. Templates render via `render_internal_prompt`.
- **Prompt spec modules import only `from .catalog import PromptSpec, register`.** `Tests/Internal_Prompts/test_import_hygiene.py` (already merged) will fail the whole branch if any new prompt module drags `tldw_chatbook.config` (or any heavy consumer module) into package import — do not import consumer modules from spec modules.
- **Programmatic channels keep winning** (spec §1): explicit caller `system_message`/`custom_prompt`/options values beat the registry; the registry replaces only hardcoded-default branches.
- **Prompt IDs frozen once merged:** `agents.*`, `summarization.*`, `document_generation.*`, `subscriptions.*` exactly as this plan names them.
- **`.superpowers/sdd/progress.md` is a TRACKED repo file** — NEVER stage or commit it; if a rebase demands cleanliness, copy-aside → `git checkout --` → restore.
- Commit per task; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; stage only named files.

**Scout facts (verified 2026-07-21 at dev e74fea705; line numbers may drift a little — re-anchor by quoted text):** all source spans cited per task below come from a read-only scout of the exact call sites; the three traps to respect: (1) `console_agent_bridge._is_subagent()` (415-422) prefix-matches `SUBAGENT_SYSTEM_PROMPT` — an identity contract; (2) `Local_Summarization_Lib.py:656-660` rebinds a LOCAL `summarizer_prompt = "Please summarize the following text:"` in `summarize_with_oobabooga` — a DIFFERENT prompt, excluded from this migration; (3) document-generation's shipped TOML prompt strings (config.py 2824-2837) are RICHER than the inline fallback dicts in `document_generator.py:70-95` — the TOML text is canonical.

---

### Task 1: Agents prompt specs + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/agents_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py` (one import line after `rag_reranker_prompts`)
- Test: `Tests/Internal_Prompts/test_agents_prompt_parity.py`

**Interfaces:**
- Consumes: `PromptSpec`, `register` (P1); `render_internal_prompt`, `get_internal_prompt` for tests.
- Produces ids: `agents.subagent_system`, `agents.console_agent_operating`, `agents.tool_protocol` — Task 2 consumes.

**Sources (copy verbatim):**
- `agents.subagent_system` ← `tldw_chatbook/Agents/agent_service.py:49-52` — the parenthesized concatenated literal ("You are a focused sub-agent. …"). Zero placeholders. contract_note: "The leading text is an identity contract: console_agent_bridge detects sub-agent turns by prefix-matching this prompt. Rewording the opening changes detection; the runtime also matches the shipped default as a fallback."
- `agents.console_agent_operating` ← `tldw_chatbook/Chat/console_agent_bridge.py:57-63` ("You are a capable assistant with optional tools. …"). Zero placeholders. contract_note: "References the fenced tool protocol and spawn_subagent; keep consistent with agents.tool_protocol."
- `agents.tool_protocol` ← the STATIC scaffold of `render_tool_protocol` at `tldw_chatbook/Agents/agent_runtime.py:183-193`, converted to a template with tokens `{tool_list}`, `{fence_open}`, `{fence_close}`:

```python
default=(
    "You can call tools. Available tools:\n"
    "{tool_list}\n\n"
    "To call a tool, your reply MUST START with the fence as its first "
    "content — no prose before it:\n"
    '{fence_open}\n{"name": "<tool name>", "arguments": {...}}\n'
    "{fence_close}\n"
    "One tool call per reply. After you receive the tool result, either "
    "call another tool the same way or answer the user directly. If no "
    "tool is needed, just answer directly."
),
required_placeholders=("tool_list", "fence_open", "fence_close"),
```

Note the conversion: the source is an f-string, so its `{{"name"...}}` doubled braces become SINGLE literal braces here (same rule as the P1 reranker templates), and `{FENCE_OPEN}`/`{_FENCE_CLOSE}` become the tokens. contract_note: "Fence markers are injected by code from agent_runtime.FENCE_OPEN/_FENCE_CLOSE and are parsed by the tool-call parser — the {fence_open}/{fence_close}/{tool_list} tokens are required. The empty-tools case renders no protocol at all (code-side)."
- All three: `subsystem="agents"`, `applies="live"` (default), `used_in` naming the module/function, no `legacy_config_path`.

- [ ] **Step 1: Write the module + wire the import** (mirror `rag_reranker_prompts.py` structure; add `from . import agents_prompts  # noqa: F401  (registers specs on import)` to `__init__.py` after the reranker line)

- [ ] **Step 2: Write the parity tests**

```python
# Tests/Internal_Prompts/test_agents_prompt_parity.py
"""Registry defaults must match the agent-runtime literals byte-for-byte;
the tool-protocol template must render exactly what render_tool_protocol's
static scaffold produced pre-migration."""

from tldw_chatbook.Internal_Prompts import CATALOG, render_internal_prompt


def test_subagent_system_matches_source_constant():
    from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT

    assert CATALOG["agents.subagent_system"].default == SUBAGENT_SYSTEM_PROMPT


def test_console_agent_operating_matches_source_constant():
    from tldw_chatbook.Chat.console_agent_bridge import (
        CONSOLE_AGENT_OPERATING_PROMPT,
    )

    assert (
        CATALOG["agents.console_agent_operating"].default
        == CONSOLE_AGENT_OPERATING_PROMPT
    )


def test_tool_protocol_template_renders_original_scaffold():
    # Pre-migration expected output, built exactly as agent_runtime.py:183-193
    # does today (copy the f-string expression verbatim with sample values).
    from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN, _FENCE_CLOSE

    tool_list = '{\n  "name": "demo",\n  "description": "d",\n  "parameters": {}\n}'
    expected = (
        "You can call tools. Available tools:\n"
        f"{tool_list}\n\n"
        "To call a tool, your reply MUST START with the fence as its first "
        "content — no prose before it:\n"
        f'{FENCE_OPEN}\n{{"name": "<tool name>", "arguments": {{...}}}}\n'
        f"{_FENCE_CLOSE}\n"
        "One tool call per reply. After you receive the tool result, either "
        "call another tool the same way or answer the user directly. If no "
        "tool is needed, just answer directly."
    )
    assert render_internal_prompt(
        "agents.tool_protocol",
        tool_list=tool_list,
        fence_open=FENCE_OPEN,
        fence_close=_FENCE_CLOSE,
    ) == expected
```

Note: the first two tests import from heavy consumer modules INSIDE the test functions (not module top-level) — keep it that way so collection stays light. These parity tests compare against the LIVE constants; Task 2 will redefine those constants as re-exports of the catalog defaults, keeping these tests meaningful as wiring checks (the byte-fidelity guarantee is this task's pre-migration green run).

- [ ] **Step 3: Run** `.venv/bin/python -m pytest Tests/Internal_Prompts/ -v` — all pass (parity + no duplicate ids + import hygiene still green).

- [ ] **Step 4: Commit** `feat(internal-prompts): agents prompt specs with parity tests` (module, `__init__.py`, test file only).

---

### Task 2: Agents migration (spawn site, compose, protocol, detection)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Test: `Tests/Agents/test_agents_internal_prompts.py`

**Interfaces:**
- Consumes: `agents.*` ids (Task 1), `get_internal_prompt`, `render_internal_prompt`, `scratch_config` fixture (re-export pattern — create `Tests/Agents/conftest.py` addition if none exists: `from Tests.Internal_Prompts.conftest import scratch_config  # noqa: F401`; if `Tests/Agents/conftest.py` exists, ADD the line).
- Produces: agents runtime reads all three prompts from the registry; module constants preserved as catalog-default re-exports (existing imports/tests keep working).

- [ ] **Step 1: `agent_service.py`**

1. Redefine the constant as the catalog default (keeps `console_agent_bridge` and existing tests importing it valid, and pins "shipped default" semantics):

```python
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Internal_Prompts.catalog import CATALOG

SUBAGENT_SYSTEM_PROMPT = CATALOG["agents.subagent_system"].default
```

(Replace the literal at 49-52. Import placement: with the other local imports at top — `Internal_Prompts` is lightweight by construction.)

2. At the spawn site (~409), resolve live: replace `system_prompt=SUBAGENT_SYSTEM_PROMPT` with `system_prompt=get_internal_prompt("agents.subagent_system")`.

3. At the protocol site (~197), replace `protocol_text = render_tool_protocol(schemas)` — NO change here; the change is inside `render_tool_protocol` (Step 3). Do not touch the memoization or the `f"{config.system_prompt}\n\n{protocol_text}"` composition.

- [ ] **Step 2: `console_agent_bridge.py`**

1. Redefine the constant the same way:

```python
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Internal_Prompts.catalog import CATALOG

CONSOLE_AGENT_OPERATING_PROMPT = CATALOG["agents.console_agent_operating"].default
```

2. `compose_agent_system_prompt` (~102-116): replace both uses of the constant with a local `operating = get_internal_prompt("agents.console_agent_operating")` resolved at call time (blank session → return `operating`; else `f"{session_prompt}\n\n{operating}"`). The existing test `compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT` stays green when no override is set.

3. `_is_subagent` (~415-422): the prefix check must survive overrides. Replace the single `startswith(SUBAGENT_SYSTEM_PROMPT)` with:

```python
        resolved = get_internal_prompt("agents.subagent_system")
        return content.startswith(resolved) or content.startswith(
            SUBAGENT_SYSTEM_PROMPT
        )
```

(Dual match: current-resolved OR shipped default — covers an override applied between spawn and detection in either direction. Preserve the surrounding function shape and the explanatory comment at ~287; extend that comment to mention the registry.)

- [ ] **Step 3: `agent_runtime.py` — `render_tool_protocol`**

Keep the signature, the empty-schemas early return, and the `tool_list` JSON assembly (170-182) unchanged. Replace ONLY the return expression (183-193) with:

```python
    from tldw_chatbook.Internal_Prompts import render_internal_prompt

    return render_internal_prompt(
        "agents.tool_protocol",
        tool_list=tool_list,
        fence_open=FENCE_OPEN,
        fence_close=_FENCE_CLOSE,
    )
```

Import inside the function on purpose: `agent_runtime` is a pure module today and P1's import-hygiene philosophy is to keep prompt plumbing out of module import paths that don't need it. (Module-level would also pass the hygiene test; function-level is the more conservative choice — note it in the report.)

- [ ] **Step 4: Write the integration/behavior tests**

```python
# Tests/Agents/test_agents_internal_prompts.py
"""Registry overrides must reach the agent runtime; identity contracts must
survive overrides. Real code paths; no accessor mocks."""

pytest_plugins_note = "scratch_config arrives via Tests/Agents/conftest.py re-export"

from tldw_chatbook.Internal_Prompts import get_internal_prompt


def test_tool_protocol_override_reaches_render(scratch_config):
    from tldw_chatbook.Agents.agent_runtime import (
        FENCE_OPEN,
        _FENCE_CLOSE,
        render_tool_protocol,
    )
    from tldw_chatbook.Agents.agent_models import ToolSchema

    scratch_config(
        "[internal_prompts.agents]\n"
        'tool_protocol = "TOOLS: {tool_list} OPEN {fence_open} CLOSE {fence_close}"\n'
    )
    out = render_tool_protocol(
        [ToolSchema(name="t", description="d", parameters={})]
    )
    assert out.startswith("TOOLS: ")
    assert FENCE_OPEN in out and _FENCE_CLOSE in out
    assert '"name": "t"' in out


def test_tool_protocol_empty_schemas_still_empty(scratch_config):
    from tldw_chatbook.Agents.agent_runtime import render_tool_protocol

    scratch_config(
        '[internal_prompts.agents]\ntool_protocol = "{tool_list}{fence_open}{fence_close}"\n'
    )
    assert render_tool_protocol([]) == ""


def test_compose_uses_override_and_default(scratch_config):
    from tldw_chatbook.Chat.console_agent_bridge import (
        CONSOLE_AGENT_OPERATING_PROMPT,
        compose_agent_system_prompt,
    )

    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT
    scratch_config(
        '[internal_prompts.agents]\nconsole_agent_operating = "CUSTOM OPERATING"\n'
    )
    assert compose_agent_system_prompt("") == "CUSTOM OPERATING"
    assert compose_agent_system_prompt("S") == "S\n\nCUSTOM OPERATING"


def test_is_subagent_detects_overridden_and_shipped_prefix(scratch_config):
    from tldw_chatbook.Chat import console_agent_bridge as bridge

    shipped = bridge.SUBAGENT_SYSTEM_PROMPT
    scratch_config(
        '[internal_prompts.agents]\nsubagent_system = "CUSTOM SUBAGENT RULES"\n'
    )
    resolved = get_internal_prompt("agents.subagent_system")
    assert resolved == "CUSTOM SUBAGENT RULES"
    # Detection accepts BOTH the live-resolved and the shipped prefix.
    assert bridge._is_subagent(_history_with_system(resolved)) is True
    assert bridge._is_subagent(_history_with_system(shipped)) is True
```

The `_history_with_system` helper and `_is_subagent`'s exact input shape must be written against the real function signature (read `_is_subagent` at ~415-422 and its call sites; adapt the helper — likely a list of message dicts with a leading system role — and document the shape in the report). ToolSchema import path likewise: verify against `agent_runtime.py`'s own import.

- [ ] **Step 5: Run** the new file, then `.venv/bin/python -m pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py Tests/Internal_Prompts/ -q`. Expected: all pass (agents suites were green at base; `Tests/Chat/` broadly has known unrelated failures — only the named bridge file must be green).

- [ ] **Step 6: Grep guard** — `grep -n '"You are a focused sub-agent\|You are a capable assistant with optional tools\|You can call tools. Available tools' tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Agents/agent_runtime.py` → only the catalog-re-export lines (no inline literals remain).

- [ ] **Step 7: Commit** `feat(internal-prompts): agent runtime prompts via registry (dual-prefix subagent detection)`.

---

### Task 3: Summarization prompt specs + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/summarization_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py`
- Test: `Tests/Internal_Prompts/test_summarization_prompt_parity.py`

**Interfaces:** produces ids `summarization.analyze_default_system`, `summarization.local_summarizer_template`, `summarization.rolling_summarize_system`.

**Sources (copy verbatim):**
- `analyze_default_system` ← `LLM_Calls/Summarization_General_Lib.py:528-550` (the bulleted-notes-specialist literal assigned when `system_message is None`; includes its embedded triple-backtick block). Zero placeholders. contract_note: "'Based on the content between backticks' refers to the embedded ``` block — keep them together."
- `local_summarizer_template` ← `LLM_Calls/Local_Summarization_Lib.py:39-56` module constant, INCLUDING the `<s>`/`</s>` sentinels and trailing `{{ .Prompt }}` cruft (ships unchanged; cleanup is a filed behavior-change follow-up). Zero placeholders (call sites concatenate). EXPLICITLY EXCLUDED: the local shadow at ~656-660 in `summarize_with_oobabooga` ("Please summarize the following text:") — different prompt, stays untouched.
- `rolling_summarize_system` ← the literal default `"Rewrite this text in summarized form."` (Chunk_Lib.py ~271 and ~682). Zero placeholders. `legacy_config_path="chunking_config.summarize_system_prompt"` — NOTE for the spec module docstring: this key has NO entry in the shipped default TOML, so `_shipped_default_for` returns None and any user-set value counts as customized — which is exactly right.

- [ ] **Step 1: module + import wire** (same pattern; `subsystem="summarization"`).
- [ ] **Step 2: parity tests** — for the two verbatim moves, embed the ORIGINAL literal in the test (copied from source pre-migration) and assert equality with `CATALOG[id].default`; for `rolling_summarize_system` assert `CATALOG[...].default == "Rewrite this text in summarized form."`. Also assert `"{{ .Prompt }}" in CATALOG["summarization.local_summarizer_template"].default` (pins the cruft-ships-unchanged decision).
- [ ] **Step 3: Run** `Tests/Internal_Prompts/ -v` → green. **Step 4: Commit** `feat(internal-prompts): summarization prompt specs with parity tests`.

---

### Task 4: Summarization migration

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py` (~526-550)
- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py` (constant + 5 sites: ~226, ~404, ~863, ~1596, ~1846 — NOT ~656)
- Modify: `tldw_chatbook/Chunking/Chunk_Lib.py` (~268-272 and ~680-682)
- Test: `Tests/LLM_Calls/` does not exist — create `Tests/Internal_Prompts/test_summarization_migration.py` (keeps the net-new tests next to the fixture)

**Interfaces:** consumes Task 3 ids + `get_internal_prompt`; produces registry-resolved summarization defaults with all caller-override channels intact.

- [ ] **Step 1: `analyze()` default** — replace the 528-550 literal assignment with:

```python
    if system_message is None:
        system_message = get_internal_prompt("summarization.analyze_default_system")
```

(add the `from tldw_chatbook.Internal_Prompts import get_internal_prompt` import at top with local imports; delete the literal block). The `is None` guard is the programmatic channel — unchanged.

- [ ] **Step 2: `Local_Summarization_Lib`** — delete the module constant `summarizer_prompt` (39-56). At each of the 5 consuming sites, the pattern `xxx = f"{summarizer_prompt}\n\n...{text}"` becomes:

```python
            xxx = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n...{text}"
```

(keep each site's exact separator — `\n\n` at 226, `\n\n\n\n` at the others — and its `custom_prompt is None` guard). Do NOT touch `summarize_with_oobabooga` (the ~656 local shadow). Add the import at top.

- [ ] **Step 3: `Chunk_Lib`** — both sites: replace the whole `get_cli_setting("chunking_config", "summarize_system_prompt", "...")` expression (in the import-time options dict) with `get_internal_prompt("summarization.rolling_summarize_system")` — the resolver now owns the `[chunking_config]` key as the legacy tier, so a new `[internal_prompts.summarization]` override correctly outranks it (spec §1 precedence). Same replacement for the inline default at the `_get_option("summarize_system_prompt", ...)` call (~680-682): the options-dict/caller channel still wins via `_get_option`'s lookup order. Note in the report: the options dict is built at import time, so for THAT path overrides apply on next process start (`applies` stays "live" for the spec since `_get_option`'s default re-resolves per call — verify which path actually feeds `_rolling_summarize` and document it).

- [ ] **Step 4: Tests**

```python
# Tests/Internal_Prompts/test_summarization_migration.py
"""Overrides must reach the summarization payloads; caller channels win.
Fakes only at the LLM dispatch/transport seams."""

from tldw_chatbook.Internal_Prompts import get_internal_prompt


def test_analyze_default_system_uses_registry(scratch_config, monkeypatch):
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl

    scratch_config(
        '[internal_prompts.summarization]\nanalyze_default_system = "CUSTOM ANALYZE SYSTEM"\n'
    )
    captured = {}

    def fake_dispatch(*args, **kwargs):
        captured["kwargs"] = kwargs
        captured["args"] = args
        return "ok"

    # Find the internal dispatch seam analyze() calls (scout: routes into
    # _dispatch_to_api / provider function). Monkeypatch THAT name in sgl's
    # namespace; assert the system message it receives is the override.
    monkeypatch.setattr(sgl, "_dispatch_to_api", fake_dispatch)
    sgl.analyze(
        input_data="text", custom_prompt_arg="p", api_name="openai",
        api_key=None, temp=0.3, system_message=None, streaming=False,
    )
    assert "CUSTOM ANALYZE SYSTEM" in str(captured)


def test_analyze_caller_system_message_wins(scratch_config, monkeypatch):
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl

    scratch_config(
        '[internal_prompts.summarization]\nanalyze_default_system = "REGISTRY"\n'
    )
    captured = {}

    def fake_dispatch(*args, **kwargs):
        captured["all"] = (args, kwargs)
        return "ok"

    monkeypatch.setattr(sgl, "_dispatch_to_api", fake_dispatch)
    sgl.analyze(
        input_data="text", custom_prompt_arg="p", api_name="openai",
        api_key=None, temp=0.3, system_message="CALLER", streaming=False,
    )
    assert "CALLER" in str(captured) and "REGISTRY" not in str(captured)


def test_rolling_summarize_system_override_reaches_llm_payload(scratch_config):
    from tldw_chatbook.Chunking.Chunk_Lib import Chunker  # adapt per module API

    scratch_config(
        '[internal_prompts.summarization]\nrolling_summarize_system = "CUSTOM ROLLING"\n'
    )
    captured = []

    def fake_llm(payload_or_kwargs, **kw):
        captured.append((payload_or_kwargs, kw))
        return "summary"

    # Call the real _rolling_summarize seam with the injected step function
    # (scout: _rolling_summarize takes llm_summarize_step_func and builds
    # payload {"system_message": final_system_prompt, ...} at ~1658).
    # Adapt the invocation to the real signature; assert:
    assert any("CUSTOM ROLLING" in str(c) for c in captured)
```

The exact `analyze()` signature and the `_dispatch_to_api` seam name, plus `_rolling_summarize`'s invocation shape, MUST be adapted from the real code (read the functions first); the assertions and the fake-at-transport-only rule are fixed. Local_Summarization sites: cover ONE representative (e.g. `summarize_with_custom_openai`) by monkeypatching its HTTP/`requests` seam or the lowest internal call and asserting the payload contains the registry text with no override AND the override text with one — if the function's structure makes that disproportionate, assert via `get_internal_prompt("summarization.local_summarizer_template")` identity at the call-site level and say so in the report (the parity test already pins the text).

- [ ] **Step 5: Run** new tests + `Tests/Internal_Prompts/ Tests/Chunking/ -q`. **Step 6: grep guard** — the bulleted-notes first line and `"Rewrite this text in summarized form."` appear nowhere in the three migrated modules. **Step 7: Commit** `feat(internal-prompts): summarization defaults via registry`.

---

### Task 5: Document-generation prompt specs + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/document_generation_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py`
- Test: `Tests/Internal_Prompts/test_document_generation_prompt_parity.py`

**Interfaces:** produces ids `document_generation.timeline_system` / `timeline_user` / `study_guide_system` / `study_guide_user` / `briefing_system` / `briefing_user`.

**Sources:**
- The 3 system prompts ← `Chat/document_generator.py:219/317/415` one-line literals, verbatim. Zero placeholders.
- The 3 user prompts: **canonical default = the SHIPPED TOML strings** at config.py 2824-2837 (timeline 2825, study_guide 2830, briefing 2835 — the richer "… Include key dates, events, …" versions), NOT the shorter inline dicts at document_generator.py:70-95. Rationale (put in the module docstring): first-run config writes the TOML, so the richer text is what every real user runs; the inline dicts only fire when the config table is entirely absent. `legacy_config_path="prompts.document_generation.<type>.prompt"` each — the differs-from-shipped rule then treats a user-customized TOML value as an override and an untouched one as default. Zero placeholders (context is concatenated by code).

- [ ] **Step 1: module + import wire.** **Step 2: parity tests** — system prompts equal source literals (embed originals); user prompts equal the TOML values: assert `CATALOG[id].default == DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"][t]["prompt"]` (import config INSIDE the test function — never in the spec module). **Step 3: run package suite. Step 4: Commit** `feat(internal-prompts): document-generation prompt specs (TOML-canonical defaults)`.

---

### Task 6: Document-generation migration

**Files:**
- Modify: `tldw_chatbook/Chat/document_generator.py`
- Test: `Tests/Internal_Prompts/test_document_generation_migration.py`

- [ ] **Step 1:** In each generator (`generate_timeline`/`generate_study_guide`/`generate_briefing`): replace the hardcoded `system_prompt = "..."` with `system_prompt = get_internal_prompt("document_generation.<type>_system")`; replace `self.<type>_config['prompt']` in the user-prompt f-string with `get_internal_prompt("document_generation.<type>_user")` (the resolver's legacy tier now reads the same `[prompts.document_generation.*]` table, so customized TOML keeps winning and the `__init__` config dicts remain ONLY for temperature/max_tokens — do not touch those reads at ~232-233/330-331/428-429, and leave `self.*_config` in place for them; drop only the `['prompt']` usage).
- [ ] **Step 2: Tests** — instantiate `DocumentGenerator` (check its `__init__` deps; if heavy, monkeypatch the DB/client args per its real signature), monkeypatch the provider dispatch inside `_call_llm` (find the actual outbound call around ~507 where messages are assembled; patch THAT boundary, not `_call_llm` itself, so message assembly runs real). Three cases: (a) no override → shipped TOML user text + hardcoded system text in messages; (b) `[internal_prompts.document_generation]` override → override text reaches messages; (c) customized legacy `[prompts.document_generation.timeline].prompt` in scratch config → that text wins over shipped default (legacy tier proof). Use `scratch_config`; note `DocumentGenerator.__init__` reads config → construct AFTER `scratch_config(...)` writes.
- [ ] **Step 3: run; grep guard** for the three system-prompt literals; **Commit** `feat(internal-prompts): document generation renders via registry`.

---

### Task 7: Subscriptions prompt specs + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/subscriptions_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py`
- Test: `Tests/Internal_Prompts/test_subscriptions_prompt_parity.py`

**Interfaces:** produces ids `subscriptions.analysis_system`, `subscriptions.feed_analysis`, `subscriptions.url_change_analysis`, `subscriptions.podcast_analysis`, `subscriptions.generic_analysis`, `subscriptions.recursive_summarizer_system`, `subscriptions.briefing`.

**Sources (`Subscriptions/content_processor.py`, `recursive_summarizer.py`, `briefing_generator.py`):**
- `analysis_system` ← content_processor.py:272 one-liner. Zero placeholders.
- The four per-type prompts ← `_build_analysis_prompt` branches (343-359, 362-374, 377-389, 393-405). The f-string interpolations become named tokens — precompute in code, tokens per prompt:
  - `feed_analysis`: `{name}`, `{title}`, `{url}`, `{published}`, `{content}`
  - `url_change_analysis`: `{url}`, `{change_percentage}`, `{content}` (the `:.1f` format spec moves INTO the code: compute `change_percentage = f"{item.get('change_percentage', 0) * 100:.1f}"` before rendering)
  - `podcast_analysis`: `{name}`, `{title}`, `{published}`, `{content}`
  - `generic_analysis`: `{name}`, `{title}`, `{type}`, `{content}`
  All text otherwise verbatim; contract_note on each: "The `processing_options.analysis_prompt` per-subscription override (code-side .replace channel) outranks the registry."
- `recursive_summarizer_system` ← recursive_summarizer.py:455-462 (the static `_get_system_prompt` literal ONLY — the 422-449 style/format assembly stays code). Zero placeholders.
- `briefing` ← briefing_generator.py:312-322 f-string; token `{content_summary}`. contract_note: "Output parsed by _parse_llm_sections: the four section labels (Executive Summary / Key Insights / Trending Topics / Recommended Actions) are matched by substring — keep them." required_placeholders=("content_summary",). (The system role at :328 is NOT migrated in P2 — it's a one-liner shared-shape fallback; keep scope to the brief's 7. List it in the module docstring as deliberately deferred.)

- [ ] **Step 1: module + wire. Step 2: parity tests** — embed each ORIGINAL f-string with sample values (locals named to match tokens after the code-side precompute) and assert `render_internal_prompt(id, **tokens) == original`; for zero-placeholder ones compare `CATALOG[id].default` with the embedded literal. **Step 3: run package suite. Step 4: Commit** `feat(internal-prompts): subscriptions prompt specs with parity tests`.

---

### Task 8: Subscriptions migration

**Files:**
- Modify: `tldw_chatbook/Subscriptions/content_processor.py`, `recursive_summarizer.py`, `briefing_generator.py`
- Test: `Tests/Internal_Prompts/test_subscriptions_migration.py`

- [ ] **Step 1: `content_processor.py`** — `analyze_content` system role → `get_internal_prompt("subscriptions.analysis_system")`. `_build_analysis_prompt`: keep the 326-340 custom-`analysis_prompt` early-return EXACTLY as is; in each type branch, precompute the token values (same expressions the f-strings use today, including slices `content[:5000]`/`content[:3000]` and the `:.1f` precompute) then `return render_internal_prompt("subscriptions.<id>", **tokens)`.
- [ ] **Step 2: `recursive_summarizer.py`** — `_get_system_prompt` returns `get_internal_prompt("subscriptions.recursive_summarizer_system")`. The 422-449 assembly untouched.
- [ ] **Step 3: `briefing_generator.py`** — keep the `custom_prompt` branch (309-310) untouched; the else branch becomes `prompt = render_internal_prompt("subscriptions.briefing", content_summary=content_summary)`. System role at :328 untouched (deferred).
- [ ] **Step 4: Tests** — monkeypatch `chat_api_call` in each module's namespace (async in briefing_generator — use an async fake): (a) `ContentSummarizer.analyze_content` + `_build_analysis_prompt` for one feed item: no-override default text reaches payload; override reaches payload; per-subscription `processing_options.analysis_prompt` still wins over a registry override (three-way precedence proof); (b) briefing `_generate_sections_with_llm`: override reaches payload, and `_parse_llm_sections` still parses a canned four-section response (contract intact); (c) recursive summarizer system prompt override reaches its payload. Construct the real objects with minimal args per their `__init__` signatures (read first; fake only DB/network deps).
- [ ] **Step 5: run new tests + `Tests/Subscriptions/ Tests/Internal_Prompts/ -q`; grep guard for the migrated literals; Commit** `feat(internal-prompts): subscriptions prompts via registry`.

---

### Task 9: Follow-up backlog tasks + verification sweep

**Files:** `backlog/tasks/` (new task files via backlog CLI), no source changes unless the sweep finds a branch-caused regression.

- [ ] **Step 1: Backlog tasks.** Determine the next free IDs against CURRENT origin/dev (`git fetch origin dev && ls backlog/tasks | sort -V | tail -5`, then `backlog task list --plain | tail`) — memory rule: six past ID collisions, re-verify at creation AND at merge time. Create with repeated `--ac` flags (comma syntax merges into one AC — known CLI trap):
  1. "Internal prompts: warn once on wrong-typed override values" — `_extract_text` silently ignores non-str/non-dict TOML values; add a warn-once. (m1)
  2. "Internal prompts: transport-boundary tests for websearch sites 2-4" — relevance eval / result summarization / answer synthesis lack live integration tests; a wrong kwarg fails silently. (m3)
  3. "RAG reranker: remove unreachable literal fallback in _call_llm_impl (~reranker.py:168)" — dead arm now that `__init__` always populates config.
  4. "Internal prompts: dead-key hygiene after full migration" — remove unconsumed `prompts_strings` loader, `CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT`, and the stale `[Prompts]` stub comments once P2/P3 land.
  5. "Local summarizer: drop the trailing `</s> {{ .Prompt }}` Ollama cruft (behavior change, needs own review)".
  Commit as `chore(backlog): internal-prompts follow-up tasks (P1 review riders)`.
- [ ] **Step 2: Sweep.** `.venv/bin/python -m pytest Tests/Internal_Prompts/ Tests/Agents/ Tests/Chunking/ Tests/Subscriptions/ Tests/Chat/test_console_agent_bridge.py Tests/Web_Scraping/ Tests/RAG/ -q` — expected: green except the RAG baseline 4 (config-loading, corrupted-metadata, chromadb-mock, shared-service). Cold-import guard runs as part of the package suite (`test_import_hygiene.py`) — it now also proves none of the four NEW spec modules leak config.
- [ ] **Step 3:** If clean: no commit. Hand off per superpowers:finishing-a-development-branch (PR to dev; merge only on user direction).

---

## Self-review (performed at plan-writing time)

- **Spec coverage (P2 slice):** §3 agents/summarization/document_generation/subscriptions rows → Tasks 1-8 with the spec's counts (3/3/6/7); §1 programmatic-precedence preserved at every site (analyze `is None` guard, custom_prompt channels, `_get_option`, per-subscription analysis_prompt); §6 golden parity + per-subsystem transport tests → Tasks 1,3,5,7 + 2,4,6,8. Follow-up filing → Task 9.
- **Known judgment calls encoded:** doc-gen TOML-canonical defaults; sub-agent dual-prefix detection; oobabooga shadow excluded; briefing system role deferred; tool-protocol tokens with code-injected fences.
- **Placeholder scan:** test skeletons that require adapting to real signatures say so explicitly with the invariant assertions fixed — deliberate implementer-verification points, not TBDs, matching the P1 pattern that worked.
- **Type consistency:** ids and API names match P1's merged module exactly.
