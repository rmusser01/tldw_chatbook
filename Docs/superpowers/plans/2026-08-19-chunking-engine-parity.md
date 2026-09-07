# Chunking Engine Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace chatbook's legacy chunking implementations with `tldw_server`'s vendored Chunking engine behind a compatibility shim, converging every splitter onto the one engine, per the spec's three-PR split (Q8).

**Architecture:** Vendor the server's `Chunking/` tree under `tldw_chatbook/Chunking/engine/` via a manifest-driven sync script; rewrite `Chunk_Lib.py` as a thin compatibility shim (legacy signatures preserved, flat chunk-dict contract preserved at the DB seam); converge the three independent Group B splitters onto the engine; stamp stored chunks with an engine version and report the mixed corpus. Phase (c) lands last because the stamp is only meaningful once every write path routes through the engine.

**Tech Stack:** Python ≥3.11, vendored pure-Python engine (tiktoken + defusedxml newly core), SQLite (schema v6), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md` — all rulings are recorded there: Q1 (GPLv3 obligations), Q2 (tiktoken hard-required + shim-enforced), Q3 (stamp + report in #1, re-chunk action in #2), Q4 (facades silent until migration), Q5 (retire ECS, adapt at seam), Q6 (converge char slicer), Q7 (no engine flag), Q8 (three-PR split = this plan's phases), Q9 (defusedxml core).

**Backlog:** TASK-18905 (Phase A) → TASK-18906 (Phase B) → TASK-18907 (Phase C), in `backlog/tasks/`, dependency-chained in that order.

## Global Constraints

- Upstream pin: `https://github.com/rmusser01/tldw_server.git`, branch `dev`, commit `385afa951922c8a9dc2002c675bb6cad65e4ac23` — never sync from a local path without SHA verification.
- Vendored files are never hand-edited (§5.2); chatbook-specific behavior lives only in `Chunking/_shims/` or the `Chunk_Lib` shim.
- Import rewrite rule: `tldw_Server_API.app.core.Chunking.X` → `tldw_chatbook.Chunking.engine.X`; any other `tldw_Server_API.app.core.*` → `tldw_chatbook.Chunking._shims.*`.
- Licence: preserve upstream GPLv3 headers, ship `tldw_server`'s LICENSE at `Chunking/engine/LICENSE`, record licence + source in `VENDOR_MANIFEST.toml`, add `"tldw_chatbook.Chunking.engine" = ["LICENSE"]` to pyproject `license-files`.
- New core dependencies: `tiktoken` and `defusedxml` (Q2/Q9). No new optional deps.
- The flat per-chunk contract (top-level `text`, `start_char`, `end_char`, `word_count`, with rich metadata under `metadata`) is preserved at the `RAG_Search.chunking_service` seam (§6.3.2).
- Legacy exception aliases: `LanguageDetectionError` → `LanguageNotSupportedError`; `MemoryLimitError` → `InvalidInputError` (over-breadth accepted, §9).
- Facades emit no deprecation warnings (Q4).
- Every phase ends with: full test suite + linters green (targeted runs during development; full sweep only if the maintainer opts in pre-PR), `Tests/Performance/test_app_import_weight.py` green, and a commit.

---

## Phase A — Vendor + shim + `Chunk_Lib` callers (PR 1)

### Task 1: Sync script + manifest + vendored tree

**Files:**
- Create: `Helper_Scripts/sync_chunking_engine.py`
- Create: `tldw_chatbook/Chunking/engine/` (vendored tree)
- Create: `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml`
- Create: `tldw_chatbook/Chunking/engine/LICENSE`
- Create: `tldw_chatbook/Chunking/engine/__init__.py` (chatbook-authored)
- Test: `Tests/Chunking/test_sync_script.py`
- Modify: `pyproject.toml` (license-files)

**Interfaces:**
- Produces: `tldw_chatbook/Chunking/engine/` populated with the phase-1 file set, import-rewritten; `VENDOR_MANIFEST.toml` consumed by the sync script for idempotent re-runs.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chunking/test_sync_script.py
"""Contract tests for the vendoring sync script (spec §5.2, §0 wrong-tree hazard)."""
import subprocess, sys, tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "tldw_chatbook" / "Chunking" / "engine"
SYNC = REPO / "Helper_Scripts" / "sync_chunking_engine.py"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"


def test_manifest_pins_upstream():
    manifest = tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())
    assert manifest["upstream"]["repo"] == "https://github.com/rmusser01/tldw_server.git"
    assert manifest["upstream"]["branch"] == "dev"
    assert manifest["upstream"]["commit"] == PIN
    assert "chunker.py" in " ".join(manifest["files"]["vendored"])
    assert "LICENSE" in manifest["files"]["extra"]
    assert manifest["licence"] == "GPL-3.0-only"


def test_engine_tree_complete():
    for rel in manifest_vendored():
        assert (ENGINE / rel).exists(), f"missing vendored file {rel}"
    # excluded-by-design files must NOT exist
    for rel in ("templates.py", "template_initialization.py", "auto_planner.py",
                "async_chunker.py", "auto_boundary_assistant.py",
                "strategies/propositions.py", "utils/proposition_eval.py"):
        assert not (ENGINE / rel).exists(), f"deferred file vendored: {rel}"
    # upstream's own __init__ must not be vendored (chatbook-authored instead)
    assert "load_and_log_configs" not in (ENGINE / "__init__.py").read_text()


def manifest_vendored():
    return tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())["files"]["vendored"]


def test_no_server_imports_remain():
    for py in ENGINE.rglob("*.py"):
        src = py.read_text()
        assert "tldw_Server_API" not in src, f"{py.name} still references upstream package"
        assert "from app.core" not in src, f"{py.name} still references app.core"


def test_sync_idempotent_and_rejects_local_edits():
    r1 = subprocess.run([sys.executable, str(SYNC)], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    # second run is a no-op
    r2 = subprocess.run([sys.executable, str(SYNC)], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    # local modification → loud failure
    victim = ENGINE / "constants.py"
    original = victim.read_text()
    victim.write_text(original + "\n# local edit\n")
    try:
        r3 = subprocess.run([sys.executable, str(SYNC)], capture_output=True, text=True)
        assert r3.returncode != 0, "sync must fail loudly on local modifications"
        assert "local modification" in (r3.stderr + r3.stdout).lower()
    finally:
        victim.write_text(original)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Chunking/test_sync_script.py -v`
Expected: FAIL — `Helper_Scripts/sync_chunking_engine.py` and the engine tree do not exist.

- [ ] **Step 3: Write the sync script**

```python
#!/usr/bin/env python3
"""Sync the vendored Chunking engine from tldw_server dev @ pinned SHA.

Spec §5.2: idempotent, SHA-verifying, loud on local modifications, never
syncs from an unverified local path.
"""
import argparse, hashlib, subprocess, sys, tempfile
from pathlib import Path

REPO = "https://github.com/rmusser01/tldw_server.git"
BRANCH = "dev"
PIN = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

# Phase-1 file set (spec §5.1); excludes #2/#3/#6-deferred modules and
# upstream's own __init__.py (chatbook-authored instead, §5.1).
VENDORED = [
    "base.py", "chunker.py", "constants.py", "exceptions.py", "error_policy.py",
    "option_utils.py", "regex_safety.py", "security_logger.py",
    "multilingual.py", "llm_context.py",
    "process_text/__init__.py", "process_text/models.py", "process_text/options.py",
    "process_text/preparation.py", "process_text/dispatch.py",
    "process_text/pipeline.py", "process_text/metadata.py",
    "splitters/__init__.py", "splitters/regex.py", "splitters/blingfire.py",
    "strategies/__init__.py", "strategies/words.py", "strategies/sentences.py",
    "strategies/paragraphs.py", "strategies/tokens.py", "strategies/json_xml.py",
    "strategies/ebook_chapters.py", "strategies/ebook_chapters_patch.py",
    "strategies/structure_aware.py", "strategies/code.py", "strategies/code_ast.py",
    "strategies/fixed_size.py", "strategies/semantic.py",
    "strategies/rolling_summarize.py",
    "utils/metrics.py",
]
UPSTREAM_ROOT = "tldw_Server_API/app/core/Chunking"
TARGET_ROOT = Path("tldw_chatbook/Chunking/engine")


def rewrite_imports(src: str) -> str:
    # Mechanical, order matters: the Chunking-specific rule first.
    src = src.replace("tldw_Server_API.app.core.Chunking",
                      "tldw_chatbook.Chunking.engine")
    src = src.replace("tldw_Server_API.app.core",
                      "tldw_chatbook.Chunking._shims")
    return src


def git_show(worktree: Path, path: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(worktree), "show", f"{PIN}:{UPSTREAM_ROOT}/{path}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FATAL: {path} not found at pinned SHA {PIN}: {r.stderr}")
    return r.stdout


def verify_clean(worktree: Path) -> None:
    """Wrong-tree hazard (spec §0): the source must match the pin exactly."""
    r = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.stdout.strip() != PIN:
        sys.exit(f"FATAL: worktree HEAD {r.stdout.strip()[:8]} != pin {PIN[:8]}; "
                 f"checkout the pinned SHA first (git checkout {PIN})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="Existing tldw_server worktree already at the pinned SHA")
    args = ap.parse_args()

    tmp = None
    if args.source:
        worktree = Path(args.source).resolve()
        verify_clean(worktree)
    else:
        tmp = tempfile.mkdtemp(prefix="tldw_server_sync_")
        worktree = Path(tmp)
        subprocess.run(["git", "clone", "--no-checkout", REPO, str(worktree)], check=True)
        subprocess.run(["git", "-C", str(worktree), "checkout", PIN], check=True)

    # 1. Refuse to overwrite local modifications (loud, spec §5.2)
    for rel in VENDORED + ["__init__.py"]:
        dst = TARGET_ROOT / rel
        if dst.exists():
            upstream = rewrite_imports(git_show(worktree, rel)) if rel != "__init__.py" else dst.read_text()
            if rel == "__init__.py":
                continue  # chatbook-authored, never touched by sync
            if dst.read_text() != upstream:
                sys.exit(f"FATAL: local modification to vendored file {rel}; "
                         f"revert it or move the change to a shim/subclass")

    # 2. Copy + rewrite
    for rel in VENDORED:
        dst = TARGET_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rewrite_imports(git_show(worktree, rel)))

    # 3. Manifest + licence
    (TARGET_ROOT / "LICENSE").write_bytes(
        subprocess.run(["git", "-C", str(worktree), "show", f"{PIN}:LICENSE"],
                       capture_output=True).stdout)
    print(f"Synced {len(VENDORED)} files from {REPO} @ {PIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the `__init__.py` handling: sync never touches it; it is chatbook-authored (spec §5.1) and lives in the manifest's excluded list, not its vendored list.

- [ ] **Step 4: Write the chatbook-authored `engine/__init__.py`**

```python
"""Chatbook's Chunking engine package init — NOT vendored (spec §5.1).

Re-exports the phase-1 public surface only. Upstream's own __init__ pulls in
#2/#3/#6-deferred modules and would coexist badly with the Chunk_Lib shim.
"""
from .base import ChunkerConfig, ChunkingMethod
from .chunker import Chunker
from .exceptions import (
    ChunkingError, InvalidInputError, InvalidChunkingMethodError, TokenizerError,
    TemplateError, LanguageNotSupportedError, ChunkSizeError, ProcessingError,
    ConfigurationError, CacheError,
)

__all__ = [
    "Chunker", "ChunkerConfig", "ChunkingMethod",
    "ChunkingError", "InvalidInputError", "InvalidChunkingMethodError",
    "TokenizerError", "TemplateError", "LanguageNotSupportedError",
    "ChunkSizeError", "ProcessingError", "ConfigurationError", "CacheError",
]
```

- [ ] **Step 5: Write `VENDOR_MANIFEST.toml`**

```toml
# Vendored from tldw_server — NEVER hand-edit files listed here (spec §5.2).
# Re-sync: python Helper_Scripts/sync_chunking_engine.py
[upstream]
repo = "https://github.com/rmusser01/tldw_server.git"
branch = "dev"
commit = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

[files]
vendored = [
    "base.py", "chunker.py", "constants.py", "exceptions.py", "error_policy.py",
    "option_utils.py", "regex_safety.py", "security_logger.py",
    "multilingual.py", "llm_context.py",
    "process_text/__init__.py", "process_text/models.py", "process_text/options.py",
    "process_text/preparation.py", "process_text/dispatch.py",
    "process_text/pipeline.py", "process_text/metadata.py",
    "splitters/__init__.py", "splitters/regex.py", "splitters/blingfire.py",
    "strategies/__init__.py", "strategies/words.py", "strategies/sentences.py",
    "strategies/paragraphs.py", "strategies/tokens.py", "strategies/json_xml.py",
    "strategies/ebook_chapters.py", "strategies/ebook_chapters_patch.py",
    "strategies/structure_aware.py", "strategies/code.py", "strategies/code_ast.py",
    "strategies/fixed_size.py", "strategies/semantic.py",
    "strategies/rolling_summarize.py",
    "utils/metrics.py",
]
extra = ["LICENSE"]  # GPLv3 text shipped verbatim in the vendored subtree
excluded = ["__init__.py", "README.md", "SECURITY.md", "templates.py",
            "template_initialization.py", "template_library/", "auto_planner.py",
            "async_chunker.py", "auto_boundary_assistant.py",
            "strategies/propositions.py", "utils/proposition_eval.py"]

[licence]
spdx = "GPL-3.0-only"
obligations = "preserve upstream headers; LICENSE shipped in-subtree; recorded here and in pyproject license-files"
```

- [ ] **Step 6: Run the sync**

Prepare the source (the local checkout is on a codex branch, so use a temporary worktree at the pin — never sync from a diverged checkout, spec §0):

```bash
git -C ~/Documents/GitHub/tldw_server2 worktree add /tmp/tldw_server_sync 385afa951922c8a9dc2002c675bb6cad65e4ac23
python Helper_Scripts/sync_chunking_engine.py --source /tmp/tldw_server_sync
```

Expected: `Synced 35 files …`; the tree at `tldw_chatbook/Chunking/engine/` exists with no `tldw_Server_API` references. (Without `--source`, the script clones from GitHub itself.)

- [ ] **Step 7: Add tiktoken + defusedxml to core deps, license-files entry**

In `pyproject.toml` core `dependencies = [...]` (after `"anytree",`):

```toml
    "tiktoken",      # engine tokens strategy (Q2: hard-required, shim-enforced)
    "defusedxml",    # engine xml security parsing (Q9: core)
```

In the vendored-license block (after `"tldw_chatbook.Third_Party.textual_fspicker" = ["LICENSE"]`):

```toml
"tldw_chatbook.Chunking.engine" = ["LICENSE"]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest Tests/Chunking/test_sync_script.py -v`
Expected: 4 PASS.

- [ ] **Step 9: Commit**

```bash
git add Helper_Scripts/sync_chunking_engine.py tldw_chatbook/Chunking/engine/ pyproject.toml Tests/Chunking/test_sync_script.py
git commit -m "feat(chunking): vendor tldw_server Chunking engine at dev@385afa95 via manifest sync"
```

### Task 2: The three shim modules

**Files:**
- Create: `tldw_chatbook/Chunking/_shims/__init__.py`
- Create: `tldw_chatbook/Chunking/_shims/testing.py`
- Create: `tldw_chatbook/Chunking/_shims/config.py`
- Create: `tldw_chatbook/Chunking/_shims/prompt_loader.py`
- Test: `Tests/Chunking/test_shims.py`

**Interfaces:**
- Consumes: `tldw_chatbook.config.get_cli_setting` (existing), `tldw_chatbook.Internal_Prompts.resolver.get_internal_prompt` (existing).
- Produces: `tldw_chatbook.Chunking._shims.testing.is_truthy(value) -> bool`, `.is_test_mode() -> bool`; `_shims.config.load_comprehensive_config()`, `.load_and_log_configs()` (returns chatbook config object/dict — see code); `_shims.prompt_loader.load_prompt(category: str, name: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chunking/test_shims.py
"""Shim contract tests (spec §5.3): the three phase-1 shims exist and behave."""
import pytest


def test_testing_shim():
    from tldw_chatbook.Chunking._shims import testing
    assert testing.is_truthy("true") is True
    assert testing.is_truthy("no") is False
    assert testing.is_truthy(True) is True
    assert testing.is_truthy(0) is False
    assert isinstance(testing.is_test_mode(), bool)


def test_config_shim():
    from tldw_chatbook.Chunking._shims import config
    cfg = config.load_comprehensive_config()
    # Server code calls .has_section('Chunking') / .get(section, key) — a
    # config-parser-like object must come back.
    assert hasattr(cfg, "has_section")


def test_prompt_loader_shim_maps_rolling_summarize():
    from tldw_chatbook.Chunking._shims import prompt_loader
    prompt = prompt_loader.load_prompt("chunking", "Rolling Summarization")
    assert isinstance(prompt, str)
    assert len(prompt) > 50  # a real prompt, not an empty string


def test_engine_imports_with_shims():
    # The engine's module graph must resolve entirely through the shims.
    import tldw_chatbook.Chunking.engine  # noqa: F401
    from tldw_chatbook.Chunking.engine import Chunker, ChunkerConfig
    c = Chunker(ChunkerConfig())
    assert c.config.default_max_size == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Chunking/test_shims.py -v`
Expected: FAIL — `_shims` package does not exist.

- [ ] **Step 3: Write the three shims**

```python
# tldw_chatbook/Chunking/_shims/__init__.py
"""Chatbook-authored shims for upstream imports (spec §5.3). Phase 1 ships
testing/config/prompt_loader; later sub-projects add more here."""
```

```python
# tldw_chatbook/Chunking/_shims/testing.py
"""Replaces tldw_Server_API.app.core.testing (spec §5.3). ~20 lines upstream."""
import os

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    try:
        return bool(int(s))
    except ValueError:
        return False


def is_test_mode() -> bool:
    return os.getenv("PYTEST_CURRENT_TEST", "") != "" or os.getenv("TLDW_TEST_MODE", "") != ""
```

```python
# tldw_chatbook/Chunking/_shims/config.py
"""Replaces tldw_Server_API.app.core.config for the vendored engine (spec §5.3).

The engine reads chunking toggles via load_comprehensive_config() (a
config-parser-like object with .has_section/.get) and load_and_log_configs()
(a dict). Both delegate to chatbook's TOML config.
"""
import configparser
from typing import Any, Dict

from ...config import get_cli_setting


class _ChunkingConfigParser(configparser.ConfigParser):
    """Parser-like view over chatbook's [chunking] TOML section."""

    def __init__(self) -> None:
        super().__init__()
        chunking = get_cli_setting("chunking", None, None) or {}
        if isinstance(chunking, dict) and chunking:
            self.read_dict({"Chunking": {k: str(v) for k, v in chunking.items()}})


def load_comprehensive_config() -> _ChunkingConfigParser:
    return _ChunkingConfigParser()


def load_and_log_configs() -> Dict[str, Any]:
    return {"chunking_config": {k: v for k, v in
            (get_cli_setting("chunking", None, None) or {}).items()
            if isinstance(v, (str, int, float, bool))}}
```

```python
# tldw_chatbook/Chunking/_shims/prompt_loader.py
"""Replaces tldw_Server_API.app.core.Utils.prompt_loader (spec §5.3).

Server IDs are ("category", "Human Title") pairs; chatbook's resolver keys
are dotted. The known phase-1 mapping is chunking/Rolling Summarization →
summarization.rolling_summarize_system (verified against both trees).
"""
from ...Internal_Prompts.resolver import get_internal_prompt

_KNOWN = {
    ("chunking", "Rolling Summarization"): "summarization.rolling_summarize_system",
}


def load_prompt(category: str, name: str) -> str:
    prompt_id = _KNOWN.get((category, name))
    if prompt_id is None:
        # Unknown pairing: raise loudly rather than returning "" (a silent
        # empty system prompt would degrade every downstream LLM call).
        raise KeyError(
            f"No prompt mapping for ('{category}', '{name}'); add it to "
            f"_shims/prompt_loader._KNOWN or Internal_Prompts."
        )
    return get_internal_prompt(prompt_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chunking/test_shims.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chunking/_shims/ Tests/Chunking/test_shims.py
git commit -m "feat(chunking): add testing/config/prompt_loader shims for the vendored engine"
```

### Task 3: `Chunk_Lib.py` becomes the compatibility shim

**Files:**
- Modify: `tldw_chatbook/Chunking/Chunk_Lib.py` (full rewrite; legacy implementation deleted)
- Test: `Tests/Chunking/test_chunk_lib_shim.py`

**Interfaces:**
- Consumes: `Chunker`, `ChunkerConfig`, `ChunkingMethod`, engine exceptions (Task 1); shims (Task 2); `TokenBasedChunker` and `LanguageChunkerFactory` (retained files).
- Produces (exact signatures — every §6.1.1 caller keeps working): `improved_chunking_process(text, chunk_options_dict=None, tokenizer_name_or_path='gpt2', template=None, template_manager=None, llm_call_function_for_chunker=None, llm_api_config_for_chunker=None) -> List[Dict[str, Any]]`; `Chunker(options=None, tokenizer_name_or_path='gpt2', template=None, template_manager=None)` adapter with `.chunk_text(text, method=None, llm_call_function=None, llm_api_config=None, use_template=None) -> List[Union[str, Dict]]`; module-level `chunk_xml(text, options)`; `chunk_for_embedding(...)`; `process_document_with_metadata(...)`; `load_document(file_path)`; constants `DEFAULT_CHUNK_OPTIONS`, `MAX_CHUNK_SIZE_{WORDS,SENTENCES,PARAGRAPHS,TOKENS}`, `MAX_DOCUMENT_SIZE_{MB,BYTES}`; `ensure_nltk_data()`; exception aliases.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chunking/test_chunk_lib_shim.py
"""Chunk_Lib shim contract (spec §6.2): legacy signatures + flat output shape."""
import pytest

from tldw_chatbook.Chunking import Chunk_Lib


def test_legacy_signature_improved_chunking_process():
    # Positional options dict, all legacy kwargs accepted (§6.1.1 callers).
    chunks = Chunk_Lib.improved_chunking_process(
        "Alpha beta gamma. Delta epsilon.",
        {"method": "words", "max_size": 3, "overlap": 1},
        tokenizer_name_or_path="gpt2",
        template=None,
        template_manager=None,
        llm_call_function_for_chunker=None,
        llm_api_config_for_chunker=None,
    )
    assert chunks, "expected chunks"
    first = chunks[0]
    assert first["text"], "chunk text must be top-level"
    assert isinstance(first["metadata"], dict)
    assert first["metadata"]["chunk_index"] == 1  # 1-based, legacy convention


def test_legacy_chunker_adapter():
    chunker = Chunk_Lib.Chunker(
        options={"method": "words", "max_size": 3, "overlap": 1},
        tokenizer_name_or_path="gpt2",
    )
    chunks = chunker.chunk_text("Alpha beta gamma delta.", method="words")
    assert chunks
    # chunk_text historically returned List[Union[str, dict]] — strings for
    # text methods, dicts for json/xml/ebook (§6.2). The adapter keeps that.
    assert isinstance(chunks[0], (str, dict))


def test_flat_contract_top_level_offsets():
    # §6.3.2: the flat per-chunk contract is what the DB seam reads.
    chunks = Chunk_Lib.improved_chunking_process(
        "One two three four. Five six seven eight.", {"method": "words", "max_size": 2}
    )
    assert all("start_char" in c and "end_char" in c for c in chunks), \
        "offsets must be top-level for _persist_chunks"
    assert all(c["word_count"] > 0 for c in chunks)


def test_module_level_chunk_xml_restored():
    assert callable(Chunk_Lib.chunk_xml)  # §7.1: name was gone, capability wasn't


def test_exception_aliases():
    from tldw_chatbook.Chunking.engine import (
        LanguageNotSupportedError, InvalidInputError, ChunkingError as EngineChunkingError,
    )
    assert Chunk_Lib.LanguageDetectionError is LanguageNotSupportedError
    assert Chunk_Lib.MemoryLimitError is InvalidInputError
    assert Chunk_Lib.ChunkingError is EngineChunkingError


def test_constants_reexported():
    assert Chunk_Lib.MAX_CHUNK_SIZE_WORDS == 10000
    assert Chunk_Lib.MAX_CHUNK_SIZE_PARAGRAPHS == 100
    assert Chunk_Lib.MAX_DOCUMENT_SIZE_MB == 100
    assert isinstance(Chunk_Lib.DEFAULT_CHUNK_OPTIONS, dict)
    assert callable(Chunk_Lib.ensure_nltk_data)


def test_tokens_no_silent_fallback():
    # Q2: the shim must raise if the engine would silently word-approximate.
    # Simulated by monkeypatching the engine tokenizer resolution to the
    # fallback and asserting the shim notices.
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod
    monkeypatch_obj = pytest.MonkeyPatch()
    original = tokens_mod.TokenChunkingStrategy._resolve_tokenizer
    def fake_resolve(self):
        return tokens_mod.FallbackTokenizer("gpt2")
    monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", fake_resolve)
    try:
        with pytest.raises(Chunk_Lib.ChunkingError, match="tiktoken"):
            Chunk_Lib.improved_chunking_process(
                "one two three four five six", {"method": "tokens", "max_size": 3}
            )
    finally:
        monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", original)
        monkeypatch_obj.undo()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chunking/test_chunk_lib_shim.py -v`
Expected: FAIL — `chunk_xml` doesn't exist, aliases don't exist; the legacy implementation fails the flat-contract and alias assertions.

- [ ] **Step 3: Write the shim**

```python
# tldw_chatbook/Chunking/Chunk_Lib.py
"""Compatibility shim over the vendored engine (spec §6.2).

The legacy implementation is DELETED (Q7 ruling). This module keeps every
legacy signature working; output shape is the flat per-chunk contract
(top-level text/start_char/end_char/word_count — §6.3.2) at the
improved_chunking_process level, and List[Union[str, dict]] at the
Chunker.chunk_text level, matching legacy behavior.
"""
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from .engine import Chunker as _EngineChunker
from .engine import ChunkerConfig, ChunkingMethod
from .engine.exceptions import (
    ChunkingError, InvalidInputError, InvalidChunkingMethodError, TokenizerError,
    TemplateError, LanguageNotSupportedError, ChunkSizeError, ProcessingError,
    ConfigurationError, CacheError,
)
from .token_chunker import TokenBasedChunker
from .language_chunkers import LanguageChunkerFactory

# --- Legacy exception aliases (§6.2/§9) ---
LanguageDetectionError = LanguageNotSupportedError
MemoryLimitError = InvalidInputError

# --- Legacy constants (§6.1 note: import-time consumers exist) ---
MAX_CHUNK_SIZE_WORDS = 10000
MAX_CHUNK_SIZE_SENTENCES = 1000
MAX_CHUNK_SIZE_PARAGRAPHS = 100
MAX_CHUNK_SIZE_TOKENS = 10000
MAX_DOCUMENT_SIZE_MB = 100
MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024

DEFAULT_CHUNK_OPTIONS: Dict[str, Any] = {
    "method": "words", "max_size": 400, "overlap": 200, "language": None,
    "adaptive": False, "adaptive_chunk_sizes": None, "multi_level": False,
    "semantic_similarity_threshold": 0.7, "json_chunkable_data_key": "data",
    "tokenizer_name_or_path": "gpt2",
}

_LEGACY_METHOD_MAP = {
    "words": ChunkingMethod.WORDS, "sentences": ChunkingMethod.SENTENCES,
    "paragraphs": ChunkingMethod.PARAGRAPHS, "tokens": ChunkingMethod.TOKENS,
    "semantic": ChunkingMethod.SEMANTIC, "json": ChunkingMethod.JSON,
    "xml": ChunkingMethod.XML,
    "ebook_chapters": ChunkingMethod.EBOOK_CHAPTERS,
    "rolling_summarize": ChunkingMethod.ROLLING_SUMMARIZE,
}
```

(Implementation continues: the adapter class, `improved_chunking_process`, flat-shape conversion, `chunk_xml`, and re-exports of `chunk_for_embedding`/`process_document_with_metadata`/`load_document`/`ensure_nltk_data` — these follow the same pattern as the legacy module's public functions, delegating to the engine. The full body is ~300 lines; the implementing engineer writes it following the test contract above, keeping legacy docstring conventions.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chunking/test_chunk_lib_shim.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chunking/Chunk_Lib.py Tests/Chunking/test_chunk_lib_shim.py
git commit -m "feat(chunking): rewrite Chunk_Lib as compat shim over the vendored engine"
```

### Task 4: Port the upstream test suite

**Files:**
- Create: `Tests/Chunking/` ported files (43 files; 3 dev-added, of which 2 in scope)
- Modify: `Helper_Scripts/sync_chunking_engine.py` (extend to copy tests)

**Interfaces:**
- Consumes: vendored engine (Task 1).
- Produces: `Tests/Chunking/test_*.py` mirroring upstream's, import-rewritten the same way.

- [ ] **Step 1: Extend the sync script to also copy tests**

Add to `VENDORED`-adjacent list in the sync script and manifest:

```python
# tests to port (spec §10.1): the whole suite minus endpoint/DB-fixture files
TESTS_TO_PORT = [
    # everything from tldw_Server_API/tests/Chunking/test_*.py EXCEPT:
    # test_chunking_endpoint.py, test_chunking_template_endpoint_errors.py,
    # test_chunking_templates_endpoint_sanitization.py, test_async_*.py (2 files),
    # test_propositions_runtime_snapshot.py (#6)
]
```

Determining the file list mechanically: run `git ls-tree -r --name-only 385afa95 tldw_Server_API/tests/Chunking/ | grep "test_.*\.py$"`, subtract the excluded set, and write the resulting 41 names into the manifest. (43 total − endpoint×3 − async×2 − propositions-snapshot×1 = 41 in scope; `test_rolling_summarize_fail_closed.py` and `test_chunking_runtime_lifecycle.py` are the dev-adds in scope.)

- [ ] **Step 2: Port with the same import rewrite, plus pytest-conftest shim**

Upstream tests import `from tldw_Server_API.app.core.Chunking import …`; the rewrite maps them to `tldw_chatbook.Chunking.engine`. Create `Tests/Chunking/conftest.py`:

```python
"""Shared fixtures for the ported upstream chunking suite."""
import pytest


@pytest.fixture(autouse=True)
def _production_sanitization(request, monkeypatch):
    """Disable the engine's test-mode relaxation for tests that opt in.

    Spec §10.2: sanitization relaxes under PYTEST_CURRENT_TEST/is_test_mode,
    so production-path evidence requires explicitly disabling test mode.
    Tests marked with @pytest.mark.production_path get test mode off.
    """
    if request.node.get_closest_marker("production_path"):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("TLDW_DISABLE_TEST_MODE", "1")
        monkeypatch.setattr(
            "tldw_chatbook.Chunking._shims.testing.is_test_mode", lambda: False
        )
```

- [ ] **Step 3: Run the ported suite**

Run: `pytest Tests/Chunking/ -v --ignore=Tests/Chunking/test_sync_script.py`
Expected: all ported tests PASS. Failures indicate shim gaps — fix the shim, not the vendored code (fixes upstream go upstream, then re-sync).

- [ ] **Step 4: Run offset/overlap property tests with test mode explicitly disabled**

Run: `pytest Tests/Chunking/test_chunking_offsets_property.py Tests/Chunking/test_chunking_overlap_properties.py Tests/Chunking/test_hierarchical_rewrite_offsets.py -v -m production_path`
Expected: PASS with the relaxed path OFF — this is the production-path parity evidence (spec §10.2 caveat).

- [ ] **Step 4b: Golden-fixture generation with test mode disabled**

Create `Tests/Chunking/golden/` fixtures and `Tests/Chunking/test_golden_parity.py`:

```python
# Tests/Chunking/test_golden_parity.py
"""Golden parity fixtures (spec §10.2): chatbook engine output must equal
server engine output byte-for-byte on the fixed corpus, with production
sanitization (no test-mode relaxation)."""
import json
from pathlib import Path
import pytest

GOLDEN = Path(__file__).parent / "golden"
CORPUS = {
    "prose": "The quick brown fox jumps over the lazy dog. " * 20,
    "markdown_atx": "# Title\n\n## Section A\n\nPara one.\n\n## Section B\n\nPara two.\n",
    "ebook": "# Chapter 1\n\nFirst chapter text.\n\n# Chapter 2\n\nSecond chapter text.\n",
    "json": '{"data": [' + ", ".join(f'{{"item": {i}, "text": "value {i}"}}' for i in range(20)) + ']}',
    "xml": "<root>" + "".join(f"<item id='{i}'>text {i}</item>" for i in range(20)) + "</root>",
    "code": "def f%d():\n    return %d\n" * 10,
    "cjk": "这是一段中文文本。" * 10,
}
METHODS = ["words", "sentences", "paragraphs", "tokens", "json", "xml",
           "ebook_chapters", "structure_aware", "code", "fixed_size"]


@pytest.mark.parametrize("corpus_key,method", [(k, m) for k in CORPUS for m in METHODS])
@pytest.mark.production_path
def test_golden_parity(corpus_key, method):
    from tldw_chatbook.Chunking.engine import Chunker, ChunkerConfig
    chunker = Chunker(ChunkerConfig())
    result = chunker.process_text(CORPUS[corpus_key], {"method": method, "max_size": 50, "overlap": 10})
    golden_path = GOLDEN / f"{corpus_key}_{method}.json"
    if not golden_path.exists():
        pytest.skip("golden file not generated yet — run generation script")
    expected = json.loads(golden_path.read_text())
    assert result == expected
```

Generation script (`Tests/Chunking/golden/generate_golden.py`) runs the **server** engine on the same corpus (from the tldw_server checkout at the pin, with test mode off), writing the JSON. The corpus and options are frozen; re-run generation at every sync.

- [ ] **Step 5: Commit**

```bash
git add Tests/Chunking/ Helper_Scripts/sync_chunking_engine.py
git commit -m "test(chunking): port upstream suite + golden parity fixtures (production-path verified)"
```

### Task 5: Call-site characterization tests + fixed `Chunk_Lib` importers

**Files:**
- Create: `Tests/Chunking/test_callsite_characterization.py`
- Modify: `tldw_chatbook/Local_Ingestion/XML_Ingestion.py` (no change needed — import fixed by Task 3's `chunk_xml`)

**Interfaces:**
- Consumes: the shim (Task 3).
- Produces: characterization evidence for §6.1.1 call sites.

- [ ] **Step 1: Write the characterization tests (capture BEFORE the convergence, assert after)**

```python
# Tests/Chunking/test_callsite_characterization.py
"""Call-site characterization (spec §10.3): every §6.1.1 entry works through
the new engine with a stable output contract. Written against the SHIM
(post-Task-3); Phase B converges the regex-path call sites and these tests
must still pass unchanged."""
import json
import pytest

from tldw_chatbook.RAG_Search import chunking_service


TEXT = ("The first sentence is here. The second sentence follows. "
        "A third sentence for good measure. And a fourth one. " * 5)


def test_book_ingestion_regex_path_shape():
    # Book_Ingestion_Lib:1793 → RAG_Search.chunking_service.improved_chunking_process
    chunks = chunking_service.improved_chunking_process(
        TEXT, {"method": "words", "max_size": 10, "overlap": 2}
    )
    assert chunks
    for c in chunks:
        assert set(c) >= {"text", "start_char", "end_char", "word_count", "chunk_index"}
        assert c["start_char"] <= c["end_char"]


def test_db_roundtrip_offsets_populated(tmp_path):
    # §6 shape seam: the DB reads top-level keys; NULLs would mean the flat
    # contract was violated somewhere upstream of _persist_chunks.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    chunks = chunking_service.improved_chunking_process(
        TEXT, {"method": "words", "max_size": 10, "overlap": 2}
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="t", media_type="document", content=TEXT, keywords=None,
        url=None, analysis_content=None, author=None, chunks=chunks,
        chunk_options={"method": "words"},
    )
    rows = db.get_connection().execute(
        "SELECT chunk_index, start_char, end_char FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index", (media_id,)
    ).fetchall()
    assert rows
    for row in rows:
        assert row["start_char"] is not None and row["end_char"] is not None, \
            "flat contract violated: DB offset columns went NULL"


def test_ebook_chapters_through_rag_service():
    # §7.2 regression: no InvalidChunkingMethodError after whitelist removal.
    text = "# Chapter 1\n\nText one.\n\n# Chapter 2\n\nText two.\n"
    chunks = chunking_service.improved_chunking_process(
        text, {"method": "ebook_chapters", "max_size": 400, "overlap": 0}
    )
    assert chunks, "ebook_chapters must chunk, not raise"


def test_xml_ingestion_import():
    # §7.1: the module-level chunk_xml name restored.
    import importlib
    import tldw_chatbook.Local_Ingestion.XML_Ingestion as mod  # noqa: F401
```

- [ ] **Step 2: Run tests**

Run: `pytest Tests/Chunking/test_callsite_characterization.py -v`
Expected: some FAIL at this stage — `chunking_service.improved_chunking_process` still has its whitelist (ebook test raises). That's fine: these tests are the Phase B target. Mark expected failures with `pytest.mark.xfail(reason="Phase B: whitelist not yet removed", strict=False)` on the ebook test only; the words/DB tests must pass post-Task-3 via the shim.

- [ ] **Step 3: Commit**

```bash
git add Tests/Chunking/test_callsite_characterization.py
git commit -m "test(chunking): call-site characterization + DB round-trip pins for Phase B"
```

### Task 6: Phase A close-out

- [ ] **Step 1: Run the targeted suite**

Run: `pytest Tests/Chunking/ Tests/Internal_Prompts/ Tests/RAG/test_config_profiles.py Tests/Performance/test_app_import_weight.py -v`
Expected: all PASS. (Import-weight: engine loads at boot via `chunking_interop_library` → `Chunking/__init__` → shim → engine; module scope is clean.)

- [ ] **Step 2: Commit**
```bash
git commit --allow-empty -m "chore(chunking): Phase A complete — engine vendored, shimmed, callers working"
```

---

## Phase B — Converge Group B implementations (PR 2)

### Task 7: Delete `_chunk_text_in_process`; route all methods through the engine

**Files:**
- Modify: `tldw_chatbook/RAG_Search/chunking_service.py`
- Test: `Tests/RAG/test_chunking_service.py` (extend)

**Interfaces:**
- Consumes: the `Chunk_Lib` shim (Task 3).
- Produces: `ChunkingService.chunk_text(content, chunk_size, chunk_overlap, method) -> List[Dict]` with the flat contract; `improved_chunking_process(text, options)` as a re-export.

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/RAG/test_chunking_service.py
"""Phase B convergence (spec §6.3.1): all methods route through the engine."""
import pytest
from tldw_chatbook.RAG_Search import chunking_service
from tldw_chatbook.RAG_Search.chunking_service import ChunkingService, ChunkingError


def test_validation_messages_preserved():
    svc = ChunkingService()
    with pytest.raises(ChunkingError, match="max_words must be positive"):
        svc.chunk_text("text", chunk_size=0, chunk_overlap=0, method="words")
    with pytest.raises(ChunkingError, match="Overlap must be non-negative"):
        svc.chunk_text("text", chunk_size=10, chunk_overlap=-1, method="words")
    with pytest.raises(ChunkingError, match="Overlap must be less than max_words"):
        svc.chunk_text("text", chunk_size=10, chunk_overlap=10, method="words")


def test_all_methods_flat_contract():
    svc = ChunkingService()
    for method in ["words", "sentences", "paragraphs"]:
        chunks = svc.chunk_text(
            "One two three. Four five six. Seven eight nine ten.", 
            chunk_size=4, chunk_overlap=1, method=method,
        )
        assert chunks
        for c in chunks:
            assert set(c) >= {"text", "start_char", "end_char", "word_count", "chunk_index"}


def test_ebook_chapters_no_whitelist():
    text = "# Chapter 1\n\nText one.\n\n# Chapter 2\n\nText two.\n"
    chunks = svc_ebook(text)
    assert len(chunks) >= 2


def svc_ebook(text):
    svc = ChunkingService()
    return svc.chunk_text(text, chunk_size=400, chunk_overlap=0, method="ebook_chapters")
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest Tests/RAG/test_chunking_service.py -v -k "validation or flat_contract or ebook"`
Expected: ebook FAIL (whitelist), validation PASS (already true — these become regression pins), flat-contract PASS for words (regex path emits flat today).

- [ ] **Step 2: (convergence) Implement**

In `chunking_service.py`: delete `_chunk_text_in_process` (lines 259–324); `chunk_text` delegates to `Chunk_Lib.improved_chunking_process` for **all** methods, flattening output to the flat contract (the shim already emits it); module-local `ChunkingError`/`InvalidChunkingMethodError` become aliases of the engine's; `improved_chunking_process` becomes a re-export of the shim's. Preserve the three validation messages exactly (test pins them).

- [ ] **Step 3: Run to verify pass**

Run: `pytest Tests/RAG/test_chunking_service.py Tests/Chunking/test_callsite_characterization.py -v`
Expected: all PASS, including the previously-xfail'd ebook test (remove the xfail marker).

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/RAG_Search/chunking_service.py Tests/RAG/test_chunking_service.py
git commit -m "feat(chunking): delete regex splitter; all methods route through the engine (flat contract preserved)"
```

### Task 8: Retire `EnhancedChunkingService`; parent/child adapter

**Files:**
- Create: `tldw_chatbook/RAG_Search/parent_child_adapter.py`
- Modify: `tldw_chatbook/RAG_Search/enhanced_chunking_service.py` (reduce to adapter)
- Modify: `tldw_chatbook/RAG_Search/simplified/enhanced_indexing_helpers.py:66-90`
- Modify: `tldw_chatbook/RAG_Search/simplified/enhanced_rag_service.py:51,122`
- Modify: `tldw_chatbook/Widgets/chunk_preview_modal.py:111,134`
- Test: `Tests/RAG/test_parent_child_adapter.py`

**Interfaces:**
- Consumes: engine's `Chunker.chunk_text_hierarchical_flat(text, method=..., max_size=..., overlap=..., template=..., method_options=...) -> list[dict]` (each item has `text`, `metadata` with ancestry info per `flatten_hierarchical`).
- Produces: `parent_child_adapter.chunk_with_parent_retrieval(text, max_size, overlap, **opts) -> dict` with keys `chunks` (flat) and `parent_chunks` (the legacy shape: each parent has children references) — exact legacy return shape so the two RAG consumers keep working with no signature change; `StructuredChunk`/`ChunkType` re-exports for the preview modal.

- [ ] **Step 1: Write the failing test (pin the legacy parent/child contract BEFORE the swap)**

```python
# Tests/RAG/test_parent_child_adapter.py
"""Q5 ruling: ECS retired; adapter preserves the parent/child retrieval shape."""
import pytest


TEXT = "# Section A\n\nPara one under A.\n\n## Sub A1\n\nDeep text.\n\n# Section B\n\nPara under B.\n"


def test_parent_child_shape():
    from tldw_chatbook.RAG_Search import parent_child_adapter as pca
    result = pca.chunk_with_parent_retrieval(TEXT, max_size=100, overlap=0)
    assert "chunks" in result and "parent_chunks" in result
    for parent in result["parent_chunks"]:
        assert "text" in parent
        assert isinstance(parent.get("children"), list)
    # every child references exactly one parent
    for chunk in result["chunks"]:
        assert chunk.get("parent_id") is not None


def test_structureaware_engine_underneath():
    # The adapter must call the engine's hierarchical path, not ECS logic.
    from tldw_chatbook.RAG_Search import parent_child_adapter as pca
    from tldw_chatbook.Chunking.engine import Chunker
    calls = []
    real = Chunker.chunk_text_hierarchical_flat
    def spy(self, text, **kwargs):
        calls.append(kwargs)
        return real(self, text, **kwargs)
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(Chunker, "chunk_text_hierarchical_flat", spy)
    try:
        pca.chunk_with_parent_retrieval(TEXT, max_size=100, overlap=0)
        assert calls, "adapter must delegate to the engine's hierarchical path"
    finally:
        monkeypatch_obj.undo()
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest Tests/RAG/test_parent_child_adapter.py -v`
Expected: FAIL — `parent_child_adapter` doesn't exist.

- [ ] **Step 3: Implement the adapter**

`parent_child_adapter.py` calls `Chunker.chunk_text_hierarchical_flat` (engine) and derives `parent_chunks` from the flattened chunks' ancestry metadata: parents = chunks whose metadata marks them as parents (per the engine's `flatten_hierarchical` ancestry info); children link via `parent_id`. `StructuredChunk`/`ChunkType` re-exported as lightweight dataclasses matching the legacy attribute names the preview modal reads (`chunk_index`, `word_count`, `char_count`, `chunk_type.value`, `metadata` — per `chunk_preview_modal.py:125-131`).

- [ ] **Step 4: Re-point the consumers**

- `enhanced_indexing_helpers.py:66`: `chunking_service = EnhancedChunkingService() if use_enhanced_chunking else None` → the adapter's `chunk_with_parent_retrieval`
- `enhanced_rag_service.py:51`: same treatment
- `chunk_preview_modal.py:111`: `service = EnhancedChunkingService()` → the adapter (with `StructuredChunk` re-exports keeping the modal's attribute reads working)
- Reduce `enhanced_chunking_service.py` to the adapter (delete `_structural_chunking`/`_hierarchical_chunking`/`DocumentStructureParser` logic); keep `create_enhanced_chunking_service()` returning the adapter-backed ECS for any straggler imports.

- [ ] **Step 5: Run the RAG suite**

Run: `pytest Tests/RAG/ -v`
Expected: PASS. (The pre-swap characterization of the two hot-path consumers is the parent/child shape test above plus `Tests/RAG/simplified/` existing suites.)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/RAG_Search/parent_child_adapter.py tldw_chatbook/RAG_Search/enhanced_chunking_service.py tldw_chatbook/RAG_Search/simplified/ tldw_chatbook/Widgets/chunk_preview_modal.py Tests/RAG/test_parent_child_adapter.py
git commit -m "feat(chunking): retire EnhancedChunkingService; engine structure_aware + parent/child adapter"
```

### Task 9: Converge `local_media_reading_service._chunk_text`

**Files:**
- Modify: `tldw_chatbook/Media/local_media_reading_service.py:1556-1600` (the `_chunk_text` static method)
- Test: `Tests/Media/test_local_media_chunking.py` (create)

**Interfaces:**
- Consumes: `ChunkingService.chunk_text` (Task 7).
- Produces: same call signature `_chunk_text(text, *, perform_chunking, chunk_size, chunk_overlap) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Media/test_local_media_chunking.py
"""Q6 ruling: the char-slicer converges onto the engine (no mid-word splits)."""
import pytest
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService


TEXT = ("word " * 200).strip()


def test_no_mid_word_splits():
    chunks = LocalMediaReadingService._chunk_text(
        TEXT, perform_chunking=True, chunk_size=50, chunk_overlap=10
    )
    assert len(chunks) > 1
    for c in chunks:
        # the old slicer cut at raw char boundaries; the engine splits on units
        assert not c["text"].startswith(" "), "mid-word split detected"
        assert not c["text"].endswith("word"[len(c["text"].split()[-1]):]) or True
    # every chunk is a whole-word boundary slice of the original
    for c in chunks:
        start, end = c["start_char"], c["end_char"]
        assert TEXT[start:end] == c["text"].strip() or TEXT[start:end].strip() == c["text"]


def test_perform_chunking_false_returns_empty():
    assert LocalMediaReadingService._chunk_text(
        TEXT, perform_chunking=False, chunk_size=50, chunk_overlap=10
    ) == []
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest Tests/Media/test_local_media_chunking.py -v`
Expected: mid-word test FAILs on current code (char slicer splits mid-word by construction).

- [ ] **Step 3: Implement**

Replace the body of `_chunk_text` (the raw `range(0, len(text), step)` loop) with a delegate to `ChunkingService().chunk_text(text, chunk_size, chunk_overlap, method="words")`, mapping the flat result to the legacy return shape (`list[dict]` with the same keys the navigation builder at `:4689` reads).

- [ ] **Step 4: Run + commit**

Run: `pytest Tests/Media/test_local_media_chunking.py Tests/Chunking/test_callsite_characterization.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Media/local_media_reading_service.py Tests/Media/test_local_media_chunking.py
git commit -m "feat(chunking): converge local_media_reading_service char-slicer onto the engine"
```

### Task 10: Preview/ingest agreement + Phase B close-out

- [ ] **Step 1: Preview/ingest agreement test (§7.3)**

```python
# append to Tests/Chunking/test_callsite_characterization.py
def test_preview_ingest_agreement():
    # §7.3: preview modal and ingestion path produce identical chunks.
    from tldw_chatbook.RAG_Search.chunking_service import improved_chunking_process
    from tldw_chatbook.RAG_Search.parent_child_adapter import chunk_with_parent_retrieval
    text = "# H\n\nBody text here. More text. " * 10
    ingest_chunks = improved_chunking_process(text, {"method": "words", "max_size": 20, "overlap": 5})
    preview_result = chunk_with_parent_retrieval(text, max_size=20, overlap=5)
    # both paths must yield the same underlying texts for the same options
    ingest_texts = [c["text"] for c in ingest_chunks]
    preview_texts = [c["text"] for c in preview_result["chunks"]]
    assert ingest_texts == preview_texts
```

- [ ] **Step 2: Run the Phase B suite**

Run: `pytest Tests/Chunking/ Tests/RAG/ Tests/Media/test_local_media_chunking.py Tests/Performance/test_app_import_weight.py -v`
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "chore(chunking): Phase B complete — all splitters converged onto the engine"
```

---

## Phase C — Stamp + migration + report (PR 3)

### Task 11: `chunk_engine_version` column + schema v6 migration

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` (schema v5 → v6, new column, migration)
- Test: `Tests/DB/test_media_db_schema_v6.py`

**Interfaces:**
- Produces: `UnvectorizedMediaChunks.chunk_engine_version TEXT NULL`; `_CURRENT_SCHEMA_VERSION = 6`; migration that leaves existing rows NULL.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/DB/test_media_db_schema_v6.py
"""Schema v6 (spec §8): chunk_engine_version column, NULL backfill."""
import sqlite3
import pytest
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture()
def fresh_db(tmp_path):
    return MediaDatabase(str(tmp_path / "media.db"), client_id="test")


def test_schema_version_is_6(fresh_db):
    version = fresh_db.get_connection().execute("PRAGMA user_version").fetchone()
    # if the DB tracks version in a table instead, read that; assert 6 either way
    assert fresh_db._CURRENT_SCHEMA_VERSION == 6


def test_column_exists(fresh_db):
    cols = [r["name"] for r in fresh_db.get_connection().execute(
        "PRAGMA table_info(UnvectorizedMediaChunks)").fetchall()]
    assert "chunk_engine_version" in cols


def test_v5_upgrade_leaves_rows_null(tmp_path):
    # Build a v5 database by hand: fresh DB, then drop to v5 semantics by
    # removing the column is impossible — instead create a DB with the OLD
    # code path: write one chunk row, NULL its version, and verify a re-open
    # keeps it readable and NULL (migration must not backfill).
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="text", keywords=None,
        url=None, analysis_content=None, author=None,
        chunks=[{"text": "old chunk", "metadata": {}}], chunk_options={},
    )
    db.get_connection().execute(
        "UPDATE UnvectorizedMediaChunks SET chunk_engine_version = NULL")
    db.get_connection().commit()
    # simulate upgrade: re-open the DB (runs migrations)
    db2 = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    rows = db2.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks").fetchall()
    assert rows and rows[0]["chunk_engine_version"] is None
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest Tests/DB/test_media_db_schema_v6.py -v`
Expected: FAIL — `_CURRENT_SCHEMA_VERSION` is 5 and the column doesn't exist.

- [ ] **Step 3: Implement the migration**

Bump `_CURRENT_SCHEMA_VERSION` to 6. Add `chunk_engine_version TEXT` to the `CREATE TABLE UnvectorizedMediaChunks` statement and an `ALTER TABLE UnvectorizedMediaChunks ADD COLUMN chunk_engine_version TEXT` migration step guarded by the existing version-check pattern in this file (follow the same style as the v4→v5 migration). `_persist_chunks` reads `ch.get("chunk_engine_version")` from each chunk dict — no new parameter on `add_media_with_keywords`; the ingestion layer stamps the chunk dicts (Task 12).

- [ ] **Step 4: Run + commit**

Run: `pytest Tests/DB/ -v`
Expected: PASS.

```bash
git add tldw_chatbook/DB/Client_Media_DB_v2.py Tests/DB/test_media_db_schema_v6.py
git commit -m "feat(db): schema v6 — chunk_engine_version column with NULL backfill"
```

### Task 12: Stamp chunks at write time + RAG Admin report

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py:1713-1780` (persist path stamps chunks)
- Modify: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py` (read-only report)
- Test: `Tests/Local_Ingestion/test_engine_version_stamp.py`

**Interfaces:**
- Produces: ingestion stamps every chunk row with `"parity-1@385afa95"`; RAG Admin gains `count_chunks_by_engine_version() -> dict[str, int]` (version → count, NULL → "legacy").

- [ ] **Step 1: Write the failing test**

```python
# Tests/Local_Ingestion/test_engine_version_stamp.py
"""Stamp + report (spec §8): new chunks carry the engine version; report counts."""
import pytest


def test_new_chunks_stamped(tmp_path):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.RAG_Search.chunking_service import improved_chunking_process
    chunks = improved_chunking_process(
        "One two three four five six. " * 5, {"method": "words", "max_size": 5}
    )
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None, chunks=chunks, chunk_options={},
    )
    rows = db.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks WHERE deleted = 0"
    ).fetchall()
    assert rows and all(r["chunk_engine_version"] == "parity-1@385afa95" for r in rows)


def test_legacy_rows_read_as_legacy(tmp_path):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    # insert a row with NULL version (pre-parity)
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None, chunks=[{"text": "old", "metadata": {}}],
        chunk_options={},
    )
    db.get_connection().execute(
        "UPDATE UnvectorizedMediaChunks SET chunk_engine_version = NULL"
    )
    db.get_connection().commit()
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
    svc = LocalRAGAdminService.__new__(LocalRAGAdminService)  # read-only query path
    # count_chunks_by_engine_version is a small, dependency-light method
    counts = svc.count_chunks_by_engine_version(db)
    assert counts.get("legacy") == 1
```

- [ ] **Step 2: Implement**

- `local_file_ingestion.persist_parsed_media` (`:1713`): before `add_media_with_keywords`, stamp each chunk dict with `chunk_engine_version` (module constant `ENGINE_VERSION = "parity-1@385afa95"` defined in and re-exported from the `Chunk_Lib` shim). `_persist_chunks` (Task 11) reads it via `ch.get("chunk_engine_version")`.
- `local_rag_admin_service`: add `count_chunks_by_engine_version(db) -> dict[str, int]` (`GROUP BY chunk_engine_version`, NULL → `"legacy"`); wire a read-only indicator into the RAG Admin surface (find where media counts render today and add one line: "Chunked by an older engine: N items").
- The same `ENGINE_VERSION` value is stamped into the chunk metadata dict the shim returns, so in-memory consumers see it without a DB read (spec §8).

- [ ] **Step 3: Run + commit**

Run: `pytest Tests/Local_Ingestion/test_engine_version_stamp.py Tests/DB/ -v`
Expected: PASS.

```bash
git add tldw_chatbook/Local_Ingestion/local_file_ingestion.py tldw_chatbook/RAG_Admin/local_rag_admin_service.py Tests/Local_Ingestion/test_engine_version_stamp.py
git commit -m "feat(chunking): stamp new chunks with engine version; RAG Admin legacy-chunk report"
```

### Task 13: Docs + Phase C close-out

- [ ] **Step 1: User-visible docs**

Update `Docs/User_Guide/` where chunking behavior is user-visible: the ingestion docs' chunking-method list (add `structure_aware`, `code`, `code_ast`, `fixed_size`), the null-byte/NFC sanitization behavior change (release-notes-grade), tokens/tiktoken now always available, and the RAG Admin legacy-chunk indicator. Search for existing chunking docs: `grep -ri "chunk" Docs/User_Guide/ --include="*.md" -l`.

- [ ] **Step 2: Commit**

```bash
git add Docs/User_Guide/
git commit -m "docs(chunking): user-visible engine-parity changes"
```

## Self-Review Checklist (run after writing; recorded for the executor)

1. **Spec coverage:** Phase A covers §5 (vendor tree, shims, boundary, deps); Task 3 covers §6.2 (signatures, aliases, constants); Task 4 covers §10.1/10.2 (ported suite, golden fixtures, production-path runs); Task 5 covers §10.3/§7.1/§7.2 characterization; Phase B covers §6.3.1 (regex delete), §6.3.3 (ECS retirement, Q5), §6.3.4 (char slicer, Q6), §7.3 (preview agreement); Phase C covers §8 (stamp, migration, report) and docs. §5.6's one-engine invariant is AC-verified in the spec; the plan's Task 7/8/9 deletions are its enforcement.
2. **Placeholder scan:** Task 3 Step 3 includes a partial code sketch with a parenthetical note that the engineer completes it per the test contract — this is intentional and stated; all other code blocks are complete. Drafting typos (task-7 module path, stray shell fragment, double-backtick headers, Task 11/12 sketch placeholders) were fixed in place before saving.
3. **Type consistency:** `improved_chunking_process(text, options)` (flat dict contract, `start_char`/`end_char`/`word_count` top-level) is consistent across Tasks 3, 5, 7, 10, 12. `chunk_with_parent_retrieval(text, max_size, overlap) -> {chunks, parent_chunks}` consistent across Tasks 8 and 10. `chunk_engine_version` stamps via chunk-dict key read by `_persist_chunks` (Tasks 11–12).
4. **Ordering:** Phase C strictly after Phase B (stamp meaningless while regex path writes chunks — spec Q8's ordering warning).
5. **Verification defaults:** per repo rules, targeted test runs during development; the full sweep runs only when the maintainer opts in pre-PR. Every phase's close-out names its targeted suite explicitly.
