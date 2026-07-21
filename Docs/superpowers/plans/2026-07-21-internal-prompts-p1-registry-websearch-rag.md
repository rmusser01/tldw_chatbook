# Internal Prompts P1 — Registry + Web-Search + RAG Reranker Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `Internal_Prompts` registry (catalog + never-raises resolver) and migrate the web-search pipeline (4 prompts, fixing the dead `[Prompts]` config keys) and the RAG rerankers (6 prompts, removing the raw-`.format()` crash vector) onto it.

**Architecture:** A pure-data catalog of `PromptSpec` entries plus a resolver that resolves user override → customized-legacy-key → shipped default and renders via declared-token-only substitution (never raises for user-caused problems). Call sites stop owning prompt literals and render through the resolver. Spec: `Docs/superpowers/specs/2026-07-21-internal-prompts-settings-page-design.md` (P1 covers §1, §2, and the `websearch`/`rag_reranker` rows of §3; UI is P3).

**Tech Stack:** Python ≥3.11, pytest, loguru; config I/O via existing `tldw_chatbook/config.py` helpers only.

## Global Constraints

- **Work in an isolated worktree** off `origin/dev` (multiple agents mutate this checkout): branch `feat/internal-prompts-p1-registry`. Use the superpowers:using-git-worktrees skill.
- **venv-only pytest**: run tests with the project venv's `python -m pytest` from the worktree root (never bare `pytest`; never `python -m` from another cwd).
- **Byte-identical defaults**: every migrated default must render exactly the text the old literal produced. Copy prompt text VERBATIM — indentation, blank lines, leading/trailing newlines, typos and all. Parity tests enforce this; never "clean up" prompt text.
- **Never `.format()`-family calls on resolved/override text.** Rendering is `safe_substitute` (declared-token replacement) only. Plain concatenation of zero-placeholder prompts is fine.
- **`catalog.py` and the per-subsystem prompt modules must not import `tldw_chatbook.config`** (directly or transitively). The resolver imports config helpers lazily inside functions.
- **Prompt IDs are frozen public config API** once merged: `websearch.*`, `rag_reranker.*` exactly as written here.
- **Do not touch** `.superpowers/`, `backlog/`, or other sessions' files; stage only files this plan names. Commit after every task with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- The `timeout` shell command is not available in this environment.

---

### Task 1: Catalog module (`PromptSpec`, `CATALOG`, `register`)

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/__init__.py`
- Create: `tldw_chatbook/Internal_Prompts/catalog.py`
- Create: `Tests/Internal_Prompts/__init__.py` (empty)
- Test: `Tests/Internal_Prompts/test_catalog.py`

**Interfaces:**
- Consumes: nothing (pure data).
- Produces: `PromptSpec` frozen dataclass (fields exactly as below); `CATALOG: dict[str, PromptSpec]`; `register(spec) -> PromptSpec` (raises `ValueError` on duplicate id or id/subsystem mismatch). Later tasks import these as `from tldw_chatbook.Internal_Prompts.catalog import PromptSpec, CATALOG, register`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Internal_Prompts/test_catalog.py
import pytest

from tldw_chatbook.Internal_Prompts.catalog import CATALOG, PromptSpec, register


def _spec(**overrides):
    base = dict(
        id="demo.example",
        subsystem="demo",
        title="Example",
        description="An example prompt.",
        used_in="tests",
        default="Hello {name}",
        required_placeholders=("name",),
    )
    base.update(overrides)
    return PromptSpec(**base)


def test_register_adds_to_catalog():
    spec = _spec()
    try:
        assert register(spec) is spec
        assert CATALOG["demo.example"] is spec
    finally:
        CATALOG.pop("demo.example", None)


def test_register_rejects_duplicate_id():
    spec = _spec()
    try:
        register(spec)
        with pytest.raises(ValueError, match="Duplicate"):
            register(_spec())
    finally:
        CATALOG.pop("demo.example", None)


def test_register_rejects_id_subsystem_mismatch():
    with pytest.raises(ValueError, match="subsystem"):
        register(_spec(id="other.example"))


def test_spec_is_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.default = "changed"


def test_spec_defaults():
    spec = _spec()
    assert spec.optional_placeholders == ()
    assert spec.contract_note is None
    assert spec.legacy_config_path is None
    assert spec.applies == "live"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Tests/Internal_Prompts/test_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Internal_Prompts'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Internal_Prompts/catalog.py
"""Declarative catalog of internal/system prompts.

Pure data. This module (and the per-subsystem prompt modules that call
``register``) must not import ``tldw_chatbook.config`` directly or
transitively — the resolver does config lookups lazily. Prompt ids are
public config API and frozen once shipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """One internal prompt: identity, shipped default, and edit contract."""

    id: str
    subsystem: str
    title: str
    description: str
    used_in: str
    default: str
    required_placeholders: tuple[str, ...] = ()
    optional_placeholders: tuple[str, ...] = ()
    contract_note: str | None = None
    legacy_config_path: str | None = None
    applies: str = "live"


CATALOG: dict[str, PromptSpec] = {}


def register(spec: PromptSpec) -> PromptSpec:
    if spec.id in CATALOG:
        raise ValueError(f"Duplicate internal prompt id: {spec.id!r}")
    if not spec.id.startswith(spec.subsystem + "."):
        raise ValueError(
            f"Prompt id {spec.id!r} must start with its subsystem {spec.subsystem!r} + '.'"
        )
    CATALOG[spec.id] = spec
    return spec
```

```python
# tldw_chatbook/Internal_Prompts/__init__.py
"""Internal/system prompt registry.

Import from this package (not submodules) so subsystem prompt modules are
registered: ``from tldw_chatbook.Internal_Prompts import get_internal_prompt``.
"""

from .catalog import CATALOG, PromptSpec, register

__all__ = ["CATALOG", "PromptSpec", "register"]
```

(Resolver and prompt modules are appended to `__init__.py` in later tasks.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Tests/Internal_Prompts/test_catalog.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Internal_Prompts/__init__.py tldw_chatbook/Internal_Prompts/catalog.py Tests/Internal_Prompts/__init__.py Tests/Internal_Prompts/test_catalog.py
git commit -m "feat(internal-prompts): PromptSpec catalog with duplicate/mismatch guards"
```

---

### Task 2: Resolver (`safe_substitute`, precedence, legacy rule, warn-once)

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/resolver.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py`
- Create: `Tests/Internal_Prompts/conftest.py`
- Test: `Tests/Internal_Prompts/test_resolver.py`

**Interfaces:**
- Consumes: `CATALOG`, `PromptSpec` from Task 1; `tldw_chatbook.config.get_cli_setting(section, key, default)` and `tldw_chatbook.config.DEFAULT_CONFIG_FROM_TOML` (lazy imports).
- Produces (later tasks and P2/P3 rely on these exact signatures):
  - `safe_substitute(text: str, **values) -> str` — replaces only exact `{name}` tokens for the given kwargs (`str(value)`), leaves all other braces untouched, never raises.
  - `get_internal_prompt(prompt_id: str) -> str` — resolved raw template. Raises `KeyError` for unknown id (programmer error); never raises for bad config values.
  - `render_internal_prompt(prompt_id: str, **values) -> str` — `safe_substitute(get_internal_prompt(id), **values)`.
  - Test fixture `scratch_config` (writes a throwaway config file and reloads the config cache).

- [ ] **Step 1: Write the scratch-config fixture**

`TLDW_CONFIG_PATH` (config.py:50) points the app at an alternate config file.

```python
# Tests/Internal_Prompts/conftest.py
import pytest


@pytest.fixture
def scratch_config(tmp_path, monkeypatch):
    """Point the app at a throwaway config file. Yields write(toml_text)."""
    from tldw_chatbook import config
    from tldw_chatbook.Internal_Prompts import resolver

    config_file = tmp_path / "config.toml"

    def write(toml_text: str) -> None:
        config_file.write_text(toml_text, encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
        config.load_settings(force_reload=True)

    resolver._warned_ids.clear()
    write("")
    yield write
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    resolver._warned_ids.clear()
    config.load_settings(force_reload=True)
```

Note: if `get_cli_setting` values do not change after `write(...)` in Step 4's
run, read `get_cli_setting` (config.py:3856) to find which cache it consults
and reset that cache in `write` as well — the fixture must guarantee
read-after-write. Fix the fixture, not the tests.

- [ ] **Step 2: Write the failing tests**

```python
# Tests/Internal_Prompts/test_resolver.py
import pytest

from tldw_chatbook.Internal_Prompts.catalog import CATALOG, PromptSpec, register
from tldw_chatbook.Internal_Prompts.resolver import (
    get_internal_prompt,
    render_internal_prompt,
    safe_substitute,
)


@pytest.fixture
def demo_spec():
    spec = register(
        PromptSpec(
            id="demo.greeting",
            subsystem="demo",
            title="Greeting",
            description="Test prompt.",
            used_in="tests",
            default="Hello {name}, JSON stays: {\"k\": 1}",
            required_placeholders=("name",),
            legacy_config_path="Prompts.demo_legacy",
        )
    )
    yield spec
    CATALOG.pop("demo.greeting", None)


# --- safe_substitute -------------------------------------------------------

def test_substitute_replaces_declared_tokens_only():
    out = safe_substitute("A {x} B {y} C {z}", x=1, y="two")
    assert out == "A 1 B two C {z}"


def test_substitute_leaves_json_and_ollama_braces_alone():
    text = 'Return {"score": 0.5} and {{ .Prompt }} end {q}'
    assert safe_substitute(text, q="Q") == 'Return {"score": 0.5} and {{ .Prompt }} end Q'


def test_substitute_never_raises_on_stray_braces():
    assert safe_substitute("{unclosed {weird}} {q}", q="ok") == "{unclosed {weird}} ok"


# --- precedence ------------------------------------------------------------

def test_default_when_no_config(demo_spec, scratch_config):
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_override_table_wins(demo_spec, scratch_config):
    scratch_config(
        '[internal_prompts.demo.greeting]\ntext = "Hi {name}!"\nbaseline = "abc"\n'
    )
    assert get_internal_prompt("demo.greeting") == "Hi {name}!"


def test_override_plain_string_accepted(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = "Yo {name}"\n')
    assert get_internal_prompt("demo.greeting") == "Yo {name}"


def test_empty_override_means_no_override(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = ""\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_invalid_override_falls_back_to_default(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = "no placeholder here"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_override_beats_legacy(demo_spec, scratch_config):
    scratch_config(
        '[internal_prompts.demo]\ngreeting = "Override {name}"\n'
        '[Prompts]\ndemo_legacy = "Legacy {name}"\n'
    )
    assert get_internal_prompt("demo.greeting") == "Override {name}"


# --- legacy tier -----------------------------------------------------------

def test_customized_legacy_honored(demo_spec, scratch_config):
    scratch_config('[Prompts]\ndemo_legacy = "Legacy {name}"\n')
    assert get_internal_prompt("demo.greeting") == "Legacy {name}"


def test_legacy_equal_to_shipped_stub_ignored(demo_spec, scratch_config, monkeypatch):
    from tldw_chatbook import config as config_mod

    monkeypatch.setitem(
        config_mod.DEFAULT_CONFIG_FROM_TOML.setdefault("Prompts", {}),
        "demo_legacy",
        "the shipped stub {name}",
    )
    scratch_config('[Prompts]\ndemo_legacy = "the shipped stub {name}"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_invalid_legacy_falls_back(demo_spec, scratch_config):
    scratch_config('[Prompts]\ndemo_legacy = "customized but tokenless"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


# --- misc ------------------------------------------------------------------

def test_unknown_id_raises_keyerror(scratch_config):
    with pytest.raises(KeyError):
        get_internal_prompt("nope.nothing")


def test_render(demo_spec, scratch_config):
    out = render_internal_prompt("demo.greeting", name="Ada")
    assert out == 'Hello Ada, JSON stays: {"k": 1}'


def test_warn_once(demo_spec, scratch_config, caplog):
    scratch_config('[internal_prompts.demo]\ngreeting = "tokenless"\n')
    get_internal_prompt("demo.greeting")
    get_internal_prompt("demo.greeting")
    warnings = [r for r in caplog.records if "demo.greeting" in r.getMessage()]
    assert len(warnings) == 1
```

Note on `test_warn_once`: loguru does not propagate to stdlib logging by
default. If `caplog` captures nothing, add this fixture to
`Tests/Internal_Prompts/conftest.py` and keep the assertion:

```python
@pytest.fixture(autouse=True)
def _loguru_to_caplog(caplog):
    import logging
    from loguru import logger as loguru_logger

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}")
    yield
    loguru_logger.remove(handler_id)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest Tests/Internal_Prompts/test_resolver.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_internal_prompt'` (or ModuleNotFoundError for `resolver`)

- [ ] **Step 4: Write the implementation**

```python
# tldw_chatbook/Internal_Prompts/resolver.py
"""Resolution + rendering for internal prompts.

Precedence: user override table -> customized legacy config key -> shipped
default. Never raises for user-caused problems (bad override text falls back
to the default with a once-per-prompt warning). Unknown prompt ids raise
KeyError — that is a programmer error the test suite catches.

Config helpers are imported lazily to keep this package off the cold-start
import chain and out of config.py import cycles.
"""

from __future__ import annotations

from loguru import logger

from .catalog import CATALOG, PromptSpec

_warned_ids: set[str] = set()


def safe_substitute(text: str, **values: object) -> str:
    """Replace only the exact ``{name}`` tokens named in ``values``.

    All other braces (JSON examples, Ollama ``{{ .Prompt }}`` cruft, stray
    user typos) pass through untouched. Cannot raise.
    """
    for name, value in values.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def get_internal_prompt(prompt_id: str) -> str:
    spec = CATALOG[prompt_id]

    override = _extract_text(_config_value("internal_prompts." + prompt_id))
    if override is not None:
        if _has_required_placeholders(override, spec):
            return override
        _warn_once(prompt_id, "override is missing a required placeholder")

    if spec.legacy_config_path:
        legacy = _extract_text(_config_value(spec.legacy_config_path))
        if legacy is not None and legacy != _shipped_default_for(
            spec.legacy_config_path
        ):
            if _has_required_placeholders(legacy, spec):
                return legacy
            _warn_once(
                prompt_id,
                f"legacy value at [{spec.legacy_config_path}] is missing a "
                "required placeholder",
            )

    return spec.default


def render_internal_prompt(prompt_id: str, **values: object) -> str:
    return safe_substitute(get_internal_prompt(prompt_id), **values)


def _has_required_placeholders(text: str, spec: PromptSpec) -> bool:
    return all("{" + name + "}" in text for name in spec.required_placeholders)


def _extract_text(raw: object) -> str | None:
    """Normalize a config value: {text, baseline} table or plain string."""
    if isinstance(raw, dict):
        raw = raw.get("text")
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


def _config_value(dotted_path: str) -> object:
    from tldw_chatbook.config import get_cli_setting  # lazy on purpose

    section, _, key = dotted_path.rpartition(".")
    if not section:
        return None
    return get_cli_setting(section, key, None)


def _shipped_default_for(dotted_path: str) -> object:
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML  # lazy on purpose

    node: object = DEFAULT_CONFIG_FROM_TOML
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _warn_once(prompt_id: str, message: str) -> None:
    if prompt_id in _warned_ids:
        return
    _warned_ids.add(prompt_id)
    logger.warning(
        f"internal_prompts: {message} (prompt id: {prompt_id}); "
        "using shipped default"
    )
```

Update the package init (full new content):

```python
# tldw_chatbook/Internal_Prompts/__init__.py
"""Internal/system prompt registry.

Import from this package (not submodules) so subsystem prompt modules are
registered: ``from tldw_chatbook.Internal_Prompts import get_internal_prompt``.
"""

from .catalog import CATALOG, PromptSpec, register
from .resolver import get_internal_prompt, render_internal_prompt, safe_substitute

__all__ = [
    "CATALOG",
    "PromptSpec",
    "register",
    "get_internal_prompt",
    "render_internal_prompt",
    "safe_substitute",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest Tests/Internal_Prompts/ -v`
Expected: all pass (catalog + resolver). If override reads return stale values, fix the `scratch_config` fixture per its note.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Internal_Prompts/resolver.py tldw_chatbook/Internal_Prompts/__init__.py Tests/Internal_Prompts/conftest.py Tests/Internal_Prompts/test_resolver.py
git commit -m "feat(internal-prompts): resolver with override/legacy precedence and safe substitution"
```

---

### Task 3: Web-search prompt specs (verbatim moves) + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/websearch_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py` (add one import line)
- Test: `Tests/Internal_Prompts/test_websearch_prompt_parity.py`

**Interfaces:**
- Consumes: `register`, `PromptSpec` (Task 1); `render_internal_prompt` (Task 2).
- Produces: catalog ids `websearch.sub_question_generation`, `websearch.result_relevance_eval`, `websearch.result_summarization`, `websearch.answer_synthesis` — Task 4 renders these.

**Source of truth for the moves** — `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (line numbers at plan time; re-locate by the quoted anchors if drifted):

| id suffix | source | kind | tokens (verbatim in source) |
|---|---|---|---|
| `sub_question_generation` | `analyze_question`, f-string ~645–675, first line `You are an AI assistant that helps generate search queries.`, last line `Original query: {original_query}` | f-string → drop `f` | `{original_query}` |
| `result_relevance_eval` | `search_result_relevance`, f-string ~809–825, first line starts `Given the following search results for the user's question:`, last line `Reasoning: [Your reasoning for the selections]` | f-string → drop `f` | `{original_question}`, `{sub_questions}`, `{content}` |
| `result_summarization` | `search_result_relevance`, plain string ~789–800, first line starts `Summarize the following text` | already a template — copy as-is | `{question}`, `{content}` |
| `answer_synthesis` | `aggregate_results`, rf-string ~1001–1082, first line starts `INITIAL_QUERY: Here are some sources`, last line `The user's query is: {question}` | rf-string → drop `f`, KEEP `r` prefix (LaTeX backslashes) | `{concatenated_texts}`, `{current_date}`, `{question}` |

The move is mechanical because each f-string's interpolations are already
plain local-variable names — dropping the `f` (keeping `r` where present)
yields the template text unchanged. Copy each literal byte-for-byte
including its leading/trailing newlines and per-line indentation.

- [ ] **Step 1: Create the module with the four specs**

```python
# tldw_chatbook/Internal_Prompts/websearch_prompts.py
"""Web-search pipeline prompt specs. Defaults moved verbatim from
Web_Scraping/WebSearch_APIs.py — do not edit prompt text here without a
behavior-change review; parity tests compare against the original literals."""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="websearch.sub_question_generation",
        subsystem="websearch",
        title="Sub-question generation",
        description=(
            "Generates alternative/expanded search queries for the user's "
            "question before hitting search engines."
        ),
        used_in="Web search pipeline (analyze_question in WebSearch_APIs.py)",
        default="""<COPY the ~645-675 literal here verbatim, f prefix dropped>""",
        required_placeholders=("original_query",),
        contract_note=(
            "The model must return a JSON array of query strings (a "
            '{"sub_questions": [...]} object also parses); quoted strings '
            "are regex-extracted as a fallback."
        ),
        legacy_config_path="Prompts.sub_question_generation_prompt",
    )
)

register(
    PromptSpec(
        id="websearch.result_relevance_eval",
        subsystem="websearch",
        title="Result relevance evaluation",
        description="Judges whether one search result is relevant to the question.",
        used_in="Web search pipeline (search_result_relevance in WebSearch_APIs.py)",
        default="""<COPY the ~809-825 literal here verbatim, f prefix dropped>""",
        required_placeholders=("original_question", "sub_questions", "content"),
        contract_note=(
            "Downstream regex requires the exact lines 'Selected Answer: "
            "True|False' and 'Reasoning: ...' in the model output."
        ),
        legacy_config_path="Prompts.search_result_relevance_eval_prompt",
    )
)

register(
    PromptSpec(
        id="websearch.result_summarization",
        subsystem="websearch",
        title="Result summarization",
        description="Summarizes a scraped page relative to the user's question.",
        used_in="Web search pipeline (search_result_relevance in WebSearch_APIs.py)",
        default="""<COPY the ~789-800 literal here verbatim (already a template)>""",
        required_placeholders=("question", "content"),
    )
)

register(
    PromptSpec(
        id="websearch.answer_synthesis",
        subsystem="websearch",
        title="Answer synthesis",
        description=(
            "Perplexity-style cited answer synthesis over the collected "
            "search-result summaries."
        ),
        used_in="Web search pipeline (aggregate_results in WebSearch_APIs.py)",
        default=r"""<COPY the ~1001-1082 literal here verbatim, f dropped, r kept>""",
        required_placeholders=("concatenated_texts", "current_date", "question"),
        contract_note=(
            "Citation format [n], markdown/LaTeX formatting rules, and "
            "query-type sections are load-bearing instructions."
        ),
        legacy_config_path="Prompts.analyze_search_results_prompt",
    )
)
```

The four `<COPY ...>` markers are move instructions, not content to type:
replace each with the exact source literal. Everything else above is final.

Add to `tldw_chatbook/Internal_Prompts/__init__.py`, directly after the
`from .catalog import ...` line:

```python
from . import websearch_prompts  # noqa: F401  (registers specs on import)
```

- [ ] **Step 2: Write the parity tests**

Each test embeds the ORIGINAL literal (copied from WebSearch_APIs.py as an
f-string, exactly as it appears there today) with sample locals, and asserts
the new render path produces identical bytes. This is the guard that the
move was verbatim.

```python
# Tests/Internal_Prompts/test_websearch_prompt_parity.py
"""Golden parity: registry defaults render byte-identical text to the
original WebSearch_APIs.py literals. The f-strings below are verbatim copies
of the pre-migration source."""

from tldw_chatbook.Internal_Prompts import render_internal_prompt


def test_sub_question_generation_parity():
    original_query = "How does climate change affect biodiversity?"
    expected = f"""<PASTE the original ~645-675 f-string body here verbatim>"""
    assert render_internal_prompt(
        "websearch.sub_question_generation", original_query=original_query
    ) == expected


def test_result_relevance_eval_parity():
    original_question = "What is quantum computing?"
    sub_questions = ["basics of quantum computing", "qubits explained"]
    content = 'Snippet with {braces} and "quotes" to prove safety.'
    expected = f"""<PASTE the original ~809-825 f-string body here verbatim>"""
    assert render_internal_prompt(
        "websearch.result_relevance_eval",
        original_question=original_question,
        sub_questions=sub_questions,
        content=content,
    ) == expected


def test_result_summarization_parity():
    question = "What is quantum computing?"
    content = "Long scraped text with a stray { brace."
    original_template = """<PASTE the original ~789-800 plain string verbatim>"""
    expected = original_template.format(question=question, content=content)
    assert render_internal_prompt(
        "websearch.result_summarization", question=question, content=content
    ) == expected


def test_answer_synthesis_parity():
    concatenated_texts = "1. Source one summary\n2. Source two summary"
    current_date = "2026-07-21"
    question = "Compare Python and JavaScript"
    expected = rf"""<PASTE the original ~1001-1082 rf-string body here verbatim>"""
    assert render_internal_prompt(
        "websearch.answer_synthesis",
        concatenated_texts=concatenated_texts,
        current_date=current_date,
        question=question,
    ) == expected
```

Notes for the implementer:
- `<PASTE ...>` markers are copy instructions; the sample values are final.
- In `test_result_relevance_eval_parity` the original interpolates the local
  names `original_question`, `sub_questions`, `content` — the test locals are
  named to match, so the pasted f-string evaluates as the original did
  (f-string `{sub_questions}` renders `str(list)`, and `safe_substitute`
  does `str(value)` — identical).
- `test_result_summarization_parity` intentionally uses `.format` on the
  ORIGINAL template (that is what the old code did at line ~883); the sample
  `content` has no `{}` pairs that `.format` would reject.

- [ ] **Step 3: Run the parity tests**

Run: `python -m pytest Tests/Internal_Prompts/test_websearch_prompt_parity.py -v`
Expected: 4 passed IF both copies were verbatim. Any assertion diff means one
of the two copies is not byte-identical to the source — fix the copy, never
the assertion. (There is no meaningful red state for a pure-move task; the
parity test IS the verification.)

- [ ] **Step 4: Verify catalog integrity across the suite**

Run: `python -m pytest Tests/Internal_Prompts/ -v`
Expected: all pass (no duplicate-id collisions from double import).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Internal_Prompts/websearch_prompts.py tldw_chatbook/Internal_Prompts/__init__.py Tests/Internal_Prompts/test_websearch_prompt_parity.py
git commit -m "feat(internal-prompts): websearch prompt specs with golden parity tests"
```

---

### Task 4: Migrate WebSearch_APIs call sites + integration test

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (four literals → resolver calls)
- Test: `Tests/Web_Scraping/test_websearch_internal_prompts.py`

**Interfaces:**
- Consumes: `render_internal_prompt` (Task 2), the four `websearch.*` ids (Task 3), `scratch_config` fixture pattern (Task 2 — copy the fixture into this test file's conftest or import it as shown below).
- Produces: WebSearch pipeline reads all four prompts from the registry; no prompt literals remain in `WebSearch_APIs.py`.

- [ ] **Step 1: Add the import**

At the top of `WebSearch_APIs.py` with the other local imports:

```python
from tldw_chatbook.Internal_Prompts import render_internal_prompt
```

- [ ] **Step 2: Replace the four literals**

1. In `analyze_question` (~645): delete the `sub_question_generation_prompt = f"""..."""` block; replace with:

```python
    sub_question_generation_prompt = render_internal_prompt(
        "websearch.sub_question_generation", original_query=original_query
    )
```

2. In `search_result_relevance` (~789): delete the `summarization_prompt = """..."""` block (the loop below uses the registry directly — see change 4).

3. In `search_result_relevance` (~809): delete the `eval_prompt = f"""..."""` block; replace with:

```python
        eval_prompt = render_internal_prompt(
            "websearch.result_relevance_eval",
            original_question=original_question,
            sub_questions=sub_questions,
            content=content,
        )
```

4. In `search_result_relevance` (~883): replace

```python
                        summary_prompt = summarization_prompt.format(
                            question=original_question,
                            content=scraped_content["content"],
                        )
```

with:

```python
                        summary_prompt = render_internal_prompt(
                            "websearch.result_summarization",
                            question=original_question,
                            content=scraped_content["content"],
                        )
```

(This removes a raw-`.format()` call on a template — the crash vector the
spec bans.)

5. In `aggregate_results` (~1001): delete the `analyze_search_results_prompt_2 = rf"""..."""` block; replace with:

```python
    analyze_search_results_prompt_2 = render_internal_prompt(
        "websearch.answer_synthesis",
        concatenated_texts=concatenated_texts,
        current_date=current_date,
        question=question,
    )
```

Keep every surrounding line (retry loops, payload assembly) unchanged. If a
local name in a replacement does not exist at that point in the function,
stop and re-read the enclosing function — the names above were verified at
plan time.

- [ ] **Step 3: Write the integration test (transport boundary only)**

```python
# Tests/Web_Scraping/test_websearch_internal_prompts.py
"""Overrides in config.toml must change the text handed to the LLM transport.
Fakes live ONLY at chat_api_call — the pipeline code runs real."""

import pytest

pytest_plugins = ["Tests.Internal_Prompts.conftest"]


def test_sub_question_prompt_override_reaches_transport(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    scratch_config(
        "[internal_prompts.websearch]\n"
        'sub_question_generation = "CUSTOM SUBQ PROMPT for: {original_query}"\n'
    )

    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        return '{"sub_questions": ["a", "b"]}'

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    result = WebSearch_APIs.analyze_question("what is love", api_endpoint="openai")

    content = captured["payload"][0]["content"]
    assert "CUSTOM SUBQ PROMPT for: what is love" in content
    assert result["sub_questions"] == ["a", "b"]


def test_default_used_when_no_override(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        return '{"sub_questions": ["a"]}'

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    WebSearch_APIs.analyze_question("what is love", api_endpoint="openai")

    content = captured["payload"][0]["content"]
    assert "You are an AI assistant that helps generate search queries" in content
    assert "Original query: what is love" in content
```

Adjust `analyze_question`'s return-shape assertion only if its actual dict
key differs — verify against the function body, and assert on whatever key
carries the sub-questions.

- [ ] **Step 4: Run the new tests**

Run: `python -m pytest Tests/Web_Scraping/test_websearch_internal_prompts.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the existing web-scraping + registry suites**

Run: `python -m pytest Tests/Web_Scraping/ Tests/Internal_Prompts/ -v`
Expected: all pass (pre-existing failures unrelated to prompts: note them in the commit message rather than fixing).

- [ ] **Step 6: Verify no prompt literals remain**

Run: `grep -n "You are an AI assistant that helps generate\|Summarize the following text in a concise way\|INITIAL_QUERY: Here are some sources\|Given the following search results for the user" tldw_chatbook/Web_Scraping/WebSearch_APIs.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Web_Scraping/WebSearch_APIs.py Tests/Web_Scraping/test_websearch_internal_prompts.py
git commit -m "feat(internal-prompts): websearch pipeline renders via registry (fixes dead [Prompts] keys)"
```

---

### Task 5: Reranker prompt specs + parity tests

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/rag_reranker_prompts.py`
- Modify: `tldw_chatbook/Internal_Prompts/__init__.py` (add one import line)
- Test: `Tests/Internal_Prompts/test_reranker_prompt_parity.py`

**Interfaces:**
- Consumes: `register`, `PromptSpec` (Task 1); `safe_substitute` (Task 2).
- Produces: catalog ids `rag_reranker.pointwise_system`, `rag_reranker.pointwise_template`, `rag_reranker.pairwise_system`, `rag_reranker.pairwise_template`, `rag_reranker.listwise_system`, `rag_reranker.listwise_template` — Task 6 injects these.

**Brace conversion rule (this task's one subtlety):** the source templates in
`RAG_Search/reranker.py` escape literal JSON braces as `{{`/`}}` because they
go through `.format()`. Catalog defaults store the FINAL single-brace form
(`{{"score"` → `{"score"`), because `safe_substitute` does not treat braces
specially. The parity tests prove old `.format()` output == new
`safe_substitute` output. System prompts have no placeholders and copy
verbatim.

- [ ] **Step 1: Create the module — all six defaults inline (final form)**

```python
# tldw_chatbook/Internal_Prompts/rag_reranker_prompts.py
"""LLM-reranker prompt specs. Defaults moved from RAG_Search/reranker.py
__init__ fallbacks; templates converted from .format ({{) to final
single-brace form. Parity tests compare rendered output against the
original .format results. Rerankers snapshot config at construction, so
overrides apply on the next search (applies="next search")."""

from .catalog import PromptSpec, register

_APPLIES = "next search"

register(
    PromptSpec(
        id="rag_reranker.pointwise_system",
        subsystem="rag_reranker",
        title="Pointwise reranker — system",
        description="System prompt for scoring one search result at a time.",
        used_in="RAG search LLM reranking (PointwiseReranker)",
        default="""You are a search result relevance evaluator. 
Your task is to score how relevant a search result is to a given query.
Return only a JSON object with a 'score' field (0.0 to 1.0) and optionally a 'reasoning' field.
Higher scores indicate better relevance.""",
        contract_note="Model must return JSON with a numeric 'score' field.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pointwise_template",
        subsystem="rag_reranker",
        title="Pointwise reranker — scoring template",
        description="Per-result scoring prompt.",
        used_in="RAG search LLM reranking (PointwiseReranker)",
        default="""Query: {query}

Search Result:
Title: {title}
Content: {content}

How relevant is this search result to the query? Consider:
1. Direct answer to the query
2. Topical relevance
3. Information quality
4. Completeness

Return JSON: {"score": 0.0-1.0{reasoning}}""",
        required_placeholders=("query", "title", "content", "reasoning"),
        contract_note=(
            "Output is parsed as JSON with a numeric 'score'; {reasoning} "
            "expands to the optional reasoning JSON fragment."
        ),
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pairwise_system",
        subsystem="rag_reranker",
        title="Pairwise reranker — system",
        description="System prompt for choosing the better of two results.",
        used_in="RAG search LLM reranking (PairwiseReranker)",
        default="""You are a search result comparator.
Given a query and two search results, determine which one is more relevant.
Return only a JSON object with 'choice' (1 or 2) and optionally 'reasoning'.""",
        contract_note="Model must return JSON with 'choice' of 1 or 2.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pairwise_template",
        subsystem="rag_reranker",
        title="Pairwise reranker — comparison template",
        description="Two-result comparison prompt.",
        used_in="RAG search LLM reranking (PairwiseReranker)",
        default="""Query: {query}

Result 1:
Title: {title1}
Content: {content1}

Result 2:
Title: {title2}
Content: {content2}

Which result better answers the query? Consider relevance, accuracy, and completeness.

Return JSON: {"choice": 1 or 2{reasoning}}""",
        required_placeholders=(
            "query",
            "title1",
            "content1",
            "title2",
            "content2",
            "reasoning",
        ),
        contract_note="Output is parsed as JSON with 'choice' of 1 or 2.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.listwise_system",
        subsystem="rag_reranker",
        title="Listwise reranker — system",
        description="System prompt for reordering a result list.",
        used_in="RAG search LLM reranking (ListwiseReranker)",
        default="""You are a search result ranker.
Given a query and a list of search results, reorder them by relevance.
Return a JSON object with 'ranking' as an array of result indices in order of relevance.""",
        contract_note="Model must return JSON with a 'ranking' index array.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.listwise_template",
        subsystem="rag_reranker",
        title="Listwise reranker — ranking template",
        description="Whole-list reordering prompt.",
        used_in="RAG search LLM reranking (ListwiseReranker)",
        default="""Query: {query}

Search Results:
{results_list}

Reorder these results by relevance to the query (most relevant first).
Return the indices in order.

Return JSON: {"ranking": [indices in order]{reasoning}}""",
        required_placeholders=("query", "results_list", "reasoning"),
        contract_note="Output is parsed as JSON with a 'ranking' index array.",
        applies=_APPLIES,
    )
)
```

Verify each `default` against the source before committing: system prompts
byte-identical to reranker.py lines ~218-221 / ~443-445 / ~571-573
(including the trailing space after "evaluator. " on the first pointwise
line); templates identical to ~225-237 / ~448-460 / ~576-584 except
`{{`→`{` and `}}`→`}`.

Add to `tldw_chatbook/Internal_Prompts/__init__.py` after the
`websearch_prompts` import:

```python
from . import rag_reranker_prompts  # noqa: F401  (registers specs on import)
```

- [ ] **Step 2: Write the parity tests**

```python
# Tests/Internal_Prompts/test_reranker_prompt_parity.py
"""Old .format() rendering vs new safe_substitute rendering must be
byte-identical. The double-braced templates below are verbatim copies of the
pre-migration reranker.py literals."""

from tldw_chatbook.Internal_Prompts import CATALOG, safe_substitute

ORIGINAL_POINTWISE_TEMPLATE = """<PASTE reranker.py ~225-237 verbatim (double braces intact)>"""
ORIGINAL_PAIRWISE_TEMPLATE = """<PASTE reranker.py ~448-460 verbatim>"""
ORIGINAL_LISTWISE_TEMPLATE = """<PASTE reranker.py ~576-584 verbatim>"""


def test_pointwise_template_parity():
    values = dict(
        query="what is rust",
        title="Rust book",
        content="Rust is a systems language {with braces}",
        reasoning=', "reasoning": "explanation"',
    )
    assert safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default, **values
    ) == ORIGINAL_POINTWISE_TEMPLATE.format(**values)


def test_pointwise_template_parity_without_reasoning():
    values = dict(query="q", title="t", content="c", reasoning="")
    assert safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default, **values
    ) == ORIGINAL_POINTWISE_TEMPLATE.format(**values)


def test_pairwise_template_parity():
    values = dict(
        query="best db",
        title1="Postgres",
        content1="c1",
        title2="SQLite",
        content2="c2",
        reasoning="",
    )
    assert safe_substitute(
        CATALOG["rag_reranker.pairwise_template"].default, **values
    ) == ORIGINAL_PAIRWISE_TEMPLATE.format(**values)


def test_listwise_template_parity():
    values = dict(
        query="q",
        results_list="0. Title: A\n   Content: a...",
        reasoning=', "reasoning": "explanation"',
    )
    assert safe_substitute(
        CATALOG["rag_reranker.listwise_template"].default, **values
    ) == ORIGINAL_LISTWISE_TEMPLATE.format(**values)


def test_system_prompts_match_source():
    # System prompts are verbatim moves; spot-check load-bearing lines.
    assert CATALOG["rag_reranker.pointwise_system"].default.startswith(
        "You are a search result relevance evaluator."
    )
    assert "'choice' (1 or 2)" in CATALOG["rag_reranker.pairwise_system"].default
    assert "'ranking' as an array" in CATALOG["rag_reranker.listwise_system"].default
```

Caveat: `ORIGINAL_POINTWISE_TEMPLATE.format(**values)` requires the sample
`content` to avoid `{...}` sequences that `.format` would parse — the given
sample `{with braces}` IS such a sequence, so use it ONLY in the
`safe_substitute` argument side if `.format` raises: in that case change the
shared `values["content"]` to `"Rust is a systems language"` and add a
separate assertion that `safe_substitute` leaves `{with braces}` intact:

```python
def test_substitute_tolerates_braces_in_values_source_format_cannot():
    out = safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default,
        query="q", title="t", content="has {curly} text", reasoning="",
    )
    assert "has {curly} text" in out
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest Tests/Internal_Prompts/test_reranker_prompt_parity.py -v`
Expected: all pass. A diff means the single-brace conversion or a copy is
wrong — fix the module, never the assertion.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Internal_Prompts/rag_reranker_prompts.py tldw_chatbook/Internal_Prompts/__init__.py Tests/Internal_Prompts/test_reranker_prompt_parity.py
git commit -m "feat(internal-prompts): reranker prompt specs, format-braces converted, parity-tested"
```

---

### Task 6: Migrate reranker to registry + safe rendering + integration test

**Files:**
- Modify: `tldw_chatbook/RAG_Search/reranker.py`
- Test: `Tests/RAG/test_reranker_internal_prompts.py`

**Interfaces:**
- Consumes: `get_internal_prompt`, `safe_substitute` (Task 2); `rag_reranker.*` ids (Task 5).
- Produces: rerankers with registry-backed defaults; zero `.format()` calls on `scoring_prompt_template`; caller-supplied `RerankingConfig` values still win (spec §1 precedence).

- [ ] **Step 1: Add the import**

At the top of `reranker.py` with the other local imports:

```python
from tldw_chatbook.Internal_Prompts import get_internal_prompt, safe_substitute
```

- [ ] **Step 2: Replace the six `__init__` literals**

`PointwiseReranker.__init__` (~213-237) becomes:

```python
    def __init__(self, config: RerankingConfig):
        super().__init__(config)

        # Registry defaults only when the caller supplied none (caller wins).
        if not config.system_prompt:
            self.config.system_prompt = get_internal_prompt(
                "rag_reranker.pointwise_system"
            )
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = get_internal_prompt(
                "rag_reranker.pointwise_template"
            )
```

`PairwiseReranker.__init__` (~439-460): same shape with
`"rag_reranker.pairwise_system"` / `"rag_reranker.pairwise_template"`.

`ListwiseReranker.__init__` (~567-584): same shape with
`"rag_reranker.listwise_system"` / `"rag_reranker.listwise_template"`.

- [ ] **Step 3: Replace the three raw `.format()` render sites**

Pointwise `_score_result` (~325):

```python
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query, title=title, content=content, reasoning=reasoning_part,
        )
```

Pairwise `_compare_pair` (~540):

```python
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query,
            title1=result1.metadata.get("doc_title", "Untitled"),
            content1=result1.document[:300],
            title2=result2.metadata.get("doc_title", "Untitled"),
            content2=result2.document[:300],
            reasoning=reasoning_part,
        )
```

Listwise `rerank` (~616):

```python
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query, results_list=results_list, reasoning=reasoning_part,
        )
```

Note this also fixes a latent pre-existing bug: a CALLER-supplied template
containing stray braces used to crash `.format()`; now it renders.

- [ ] **Step 4: Write the integration test**

First find the result type the reranker consumes:
Run: `grep -n "^from\|^import" tldw_chatbook/RAG_Search/reranker.py | grep -i "searchresult"`
Use that import in the test and construct instances with the fields the
reranker touches (`id`, `document`, `metadata`, `score` — adjust constructor
kwargs to the actual dataclass).

```python
# Tests/RAG/test_reranker_internal_prompts.py
"""Registry overrides must reach the reranker's LLM boundary; caller-supplied
config must still win. Fake lives ONLY at _call_llm."""

import pytest

pytest_plugins = ["Tests.Internal_Prompts.conftest"]

from tldw_chatbook.RAG_Search.reranker import PointwiseReranker, RerankingConfig
# SearchResult import per the grep above.


def _result(i):
    return SearchResult(  # adjust kwargs to the real constructor
        id=str(i), document=f"doc {i}", metadata={"doc_title": f"t{i}"}, score=0.5
    )


@pytest.mark.asyncio
async def test_override_reaches_llm_boundary(scratch_config, monkeypatch):
    scratch_config(
        "[internal_prompts.rag_reranker]\n"
        'pointwise_template = "RATE {query} | {title} | {content} | end{reasoning}"\n'
        'pointwise_system = "CUSTOM SYSTEM"\n'
    )
    reranker = PointwiseReranker(RerankingConfig())
    assert reranker.config.system_prompt == "CUSTOM SYSTEM"

    captured = []

    async def fake_call_llm(prompt, system_prompt=None):
        captured.append(prompt)
        return '{"score": 0.9}'

    monkeypatch.setattr(reranker, "_call_llm", fake_call_llm)
    await reranker.rerank("my query", [_result(0), _result(1)])

    assert captured, "reranker never called the LLM"
    assert captured[0].startswith("RATE my query | t0 | doc 0 | end")


@pytest.mark.asyncio
async def test_caller_supplied_config_beats_registry(scratch_config, monkeypatch):
    scratch_config(
        '[internal_prompts.rag_reranker]\npointwise_system = "REGISTRY SYSTEM"\n'
    )
    config = RerankingConfig(system_prompt="CALLER SYSTEM")
    reranker = PointwiseReranker(config)
    assert reranker.config.system_prompt == "CALLER SYSTEM"


def test_default_when_no_config(scratch_config):
    reranker = PointwiseReranker(RerankingConfig())
    assert reranker.config.system_prompt.startswith(
        "You are a search result relevance evaluator."
    )
    assert 'Return JSON: {"score": 0.0-1.0{reasoning}}' in (
        reranker.config.scoring_prompt_template
    )
```

If the project's asyncio marker differs (check `pyproject.toml` /
`pytest.ini` for `asyncio_mode`), match the existing convention in
`Tests/RAG/` rather than `@pytest.mark.asyncio`.

- [ ] **Step 5: Run the new tests, then the RAG suites**

Run: `python -m pytest Tests/RAG/test_reranker_internal_prompts.py -v`
Expected: 3 passed

Run: `python -m pytest Tests/RAG/ Tests/Internal_Prompts/ -v`
Expected: all pass (note pre-existing unrelated failures in the commit message).

- [ ] **Step 6: Verify no raw format on templates remains**

Run: `grep -n "scoring_prompt_template.format" tldw_chatbook/RAG_Search/reranker.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/RAG_Search/reranker.py Tests/RAG/test_reranker_internal_prompts.py
git commit -m "feat(internal-prompts): rerankers use registry defaults and crash-safe rendering"
```

---

### Task 7: Full verification sweep

**Files:** none created; fixes only if regressions surface.

- [ ] **Step 1: Full registry + touched-subsystem suites**

Run: `python -m pytest Tests/Internal_Prompts/ Tests/Web_Scraping/ Tests/RAG/ -v`
Expected: all pass except failures reproducible on the base commit (verify by `git stash`-free comparison: `git worktree` base is `origin/dev` — run the same command on a clean `origin/dev` checkout if any failure looks pre-existing, and record the comparison in the task notes).

- [ ] **Step 2: Cold-import guard**

The package must not drag config into module import:

Run: `python -c "import sys; import tldw_chatbook.Internal_Prompts; print('config imported:', 'tldw_chatbook.config' in sys.modules)"`
Expected output: `config imported: False`

- [ ] **Step 3: Wider smoke — chat + config suites**

Run: `python -m pytest Tests/Chat/ -x -q` and `python -m pytest Tests/ -k "config" -q`
Expected: same results as base (no new failures).

- [ ] **Step 4: Commit any fixes; otherwise no commit**

If Steps 1-3 forced code changes, commit them:

```bash
git add -A -- tldw_chatbook/Internal_Prompts Tests/Internal_Prompts
git commit -m "fix(internal-prompts): verification-sweep fixes"
```

Then hand off per superpowers:finishing-a-development-branch (PR targets
`dev`; do NOT merge — user gate).

---

## Self-review (performed at plan-writing time)

- **Spec coverage (P1 slice):** §1 catalog/resolver → Tasks 1-2; §2 storage shapes + empty-string + plain-string compat → Task 2 tests; §3 websearch row → Tasks 3-4 (incl. dead-keys fix + legacy stub rule); §3 rag_reranker row → Tasks 5-6 (incl. `.format` removal, caller precedence, `applies="next search"`); golden parity + transport-boundary integration tests → Tasks 3-6. Reset/badge semantics, Settings UI, and the remaining subsystems are P2/P3 by design.
- **Placeholders:** the `<COPY/PASTE ...>` markers are deliberate verbatim-move instructions with exact source anchors, enforced by parity tests — not unspecified work.
- **Type consistency:** `PromptSpec` fields, resolver signatures, and the six/four prompt ids are identical across all tasks.
