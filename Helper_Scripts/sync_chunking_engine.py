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

# Non-Python files shipped verbatim (manifest `extra`), copied from the repo
# root at the pin. GPLv3 §4 requires the licence text itself to accompany
# distribution, so LICENSES/GPL-3.0-only.txt ships alongside the scope map.
EXTRA_FILES = ["LICENSE", "LICENSES/GPL-3.0-only.txt"]

# tests to port (spec §10.1): the whole suite minus endpoint/DB-fixture files.
# 46 upstream test files − 6 excluded = 40 ported (the brief's "41" counted
# upstream's 43-file snapshot plus the two dev adds minus exclusions; the
# actual dev tree at the pin has 46 test files, so the in-scope count is 40).
# test_rolling_summarize_fail_closed.py and test_chunking_runtime_lifecycle.py
# ARE in scope (dev adds) — runtime_lifecycle is ported but module-skipped as
# endpoint-bound per §10.1.
TESTS_EXCLUDED = {
    # endpoint / DB-fixture files (spec §10.1): land with their own sub-projects
    "test_chunking_endpoint.py",
    "test_chunking_template_endpoint_errors.py",
    "test_chunking_templates_endpoint_sanitization.py",
    "test_async_batch_processor.py",
    "test_async_template_return_type.py",
    # #6-deferred runtime snapshot (spec §10.1: "the propositions one lands with #6")
    "test_propositions_runtime_snapshot.py",
}
UPSTREAM_TESTS_ROOT = "tldw_Server_API/tests/Chunking"
TARGET_TESTS_ROOT = Path("Tests/Chunking")
# upstream filename -> chatbook filename (collision with chatbook's own test)
TEST_RENAMES = {"test_chunking_templates.py": "test_upstream_chunking_templates.py"}

# ---------------------------------------------------------------------------
# Chatbook-side test patches (spec §10.1/§10.2).
#
# The ported tests need skips/guards for surfaces that are NOT vendored in
# Phase A: #2/#3/#6-deferred modules (templates, auto_planner,
# auto_boundary_assistant, propositions, async_chunker), server-only fixtures
# (FastAPI endpoints, AuthNZ, Metrics registry), and the HF-hub tokenizer
# download that chatbook's network guard blocks. Applying these patches here —
# rather than as one-time manual edits — keeps `sync_chunking_engine.py`
# idempotent: a re-sync reproduces the exact ported tree, and an upstream
# drift that breaks an anchor fails loudly instead of silently dropping a
# skip (which would surface as a collection error or a network-guard failure).
# ---------------------------------------------------------------------------

# whole-module skips: file -> reason
TESTS_MODULE_SKIPPED = {
    "test_auto_apply_selection.py": "templates (TemplateClassifier) is deferred to sub-project #2; not in the Phase-A vendored set",
    "test_auto_boundary_assistant.py": "auto_boundary_assistant + server Chat/AuthNZ deps deferred to #6",
    "test_auto_chunking_planner.py": "auto_planner is deferred to sub-project #3; not in the Phase-A vendored set",
    "test_auto_chunking_resolver.py": "Ingestion_Media_Processing.chunking_options is server-side; deferred to #3",
    "test_chunker_process_metrics.py": "server Metrics registry not vendored; engine degrades gracefully to no-op metrics",
    "test_chunking_runtime_lifecycle.py": "exercises FastAPI endpoint + AuthNZ/DB fixtures (spec §10.1 endpoint class); only its rolling_summarize module constants are vendored",
    "test_chunking_templates.py": "templates + template endpoints/DB fixtures are server-side, deferred to #2 (spec §10.1)",
    "test_chunking_templates_validate_schema.py": "FastAPI TestClient endpoint fixture (spec §10.1 endpoint class); lands with #2",
    "test_phase3_3_sanitizers.py": "FastAPI endpoint module tldw_Server_API.app.api.v1.endpoints.chunking is server-side (spec §10.1 endpoint class)",
    "test_propositions_strategy.py": "strategies/propositions.py is deferred to sub-project #6; not in the Phase-A vendored set",
    "test_template_classifier.py": "templates is deferred to sub-project #2; not in the Phase-A vendored set",
    "test_template_hierarchical_options.py": "templates is deferred to sub-project #2; not in the Phase-A vendored set",
    "test_template_learner.py": "templates is deferred to sub-project #2; not in the Phase-A vendored set",
    "test_hierarchical_rewrite_offsets.py": "strategies/propositions.py deferred to #6; not in the Phase-A vendored set",
}

MODULE_SKIP_FMT = (
    "# --- Ported (chunking-engine-parity Task 4) ---------------------------------\n"
    "# Upstream file: tldw_Server_API/tests/Chunking/{upstream}\n"
    "# Skipped: {reason}. Remove this block when the module is vendored in\n"
    "# its own sub-project and re-sync the test from upstream.\n"
    'pytest.importorskip("tldw_chatbook.NoSuchDeferredModule",\n'
    '                    reason="skipped: {reason}")\n'
)

ASYNC_GUARD = (
    "{i}# --- Ported (chunking-engine-parity Task 4) -------------------------\n"
    "{i}# async_chunker depends on server-only http_client/exceptions modules\n"
    "{i}# and is deferred to sub-project #6 (spec §5.1 deferrals).\n"
    "{i}pytest.importorskip(\n"
    "{i}    \"tldw_chatbook.Chunking.engine.async_chunker\",\n"
    "{i}    reason=\"async_chunker deferred to #6 (server http_client/exceptions deps)\",\n"
    "{i})\n"
)

TOK_HELPER_IMPORT = (
    "from Tests.Chunking.conftest import real_hf_cache  # noqa: F401\n"
)

TOK_USEFIXTURES = (
    "    # --- Ported (chunking-engine-parity Task 4) -------------------------\n"
    "    # Loads the real gpt2 tokenizer; the real_hf_cache fixture points the\n"
    "    # HF stack at the real (pre-sandbox) cache with offline mode forced, so\n"
    "    # no network is touched. Skips if gpt2 is genuinely not cached.\n"
    "    @pytest.mark.usefixtures('real_hf_cache')\n"
)


def _fail(anchor: str, name: str) -> None:
    sys.exit(
        f"FATAL: patch anchor not found in {name}: {anchor[:60]!r}…\n"
        f"Upstream content drifted from the pinned expectations; update the\n"
        f"matching patch table (TEST_PATCHES / ENGINE_PATCHES) in\n"
        f"sync_chunking_engine.py before re-syncing."
    )


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        _fail(old, name)
    return text.replace(old, new, 1)


def _insert_module_skip(text: str, upstream_name: str, reason: str) -> str:
    """Insert a module-level importorskip after `import pytest` (adding it if
    the upstream file lacks it), so deferred imports never execute."""
    block = MODULE_SKIP_FMT.format(upstream=upstream_name, reason=reason)
    if "\nimport pytest\n" in text:
        return _replace_once(
            text, "\nimport pytest\n", "\nimport pytest\n\n" + block, upstream_name
        )
    if text.startswith("import pytest\n"):
        return text.replace("import pytest\n", "import pytest\n\n" + block, 1)
    # no pytest import: prepend both
    return "import pytest\n\n" + block + "\n" + text


def _patch_chunker_v2(text: str) -> str:
    name = "test_chunker_v2.py"
    # fixture import for the tokenizer-cache-dependent tests
    text = _replace_once(
        text, "import pytest\n\n", "import pytest\n\n" + TOK_HELPER_IMPORT + "\n", name
    )
    # metrics registry is server-side: skip the whole class
    text = _replace_once(
        text,
        'class TestChunkerMetrics:\n'
        '    """Ensure chunker-specific metrics are registered and populated."""\n',
        'class TestChunkerMetrics:\n'
        '    """Ensure chunker-specific metrics are registered and populated."""\n'
        '\n'
        '    # --- Ported (chunking-engine-parity Task 4) -----------------------\n'
        '    # The server\'s Metrics registry is not vendored; the engine degrades\n'
        '    # gracefully to no-op metrics, so there is no registry to assert\n'
        '    # against in chatbook. Re-enable when a Metrics shim lands.\n'
        '    pytestmark = [\n'
        '        pytest.mark.skip(\n'
        '            reason="server Metrics registry not vendored; engine degrades to no-op metrics"\n'
        '        )\n'
        '    ]\n',
        name,
    )
    # async_chunker test
    text = _replace_once(
        text,
        '    async def test_async_chunker_preserves_language_per_task(self):\n'
        '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        '    async def test_async_chunker_preserves_language_per_task(self):\n'
        + ASYNC_GUARD.format(i="        ")
        + '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        name,
    )
    # tokenizer-cache-dependent tests: real_hf_cache fixture (offline read of
    # the real cache; skip only if gpt2 is genuinely absent on this machine)
    for anchor in (
        '    def test_process_text_tokenizer_override(self):\n'
        '        """tokenizer_name_or_path should use per-call strategy without mutating cached tokens."""\n',
        '    def test_tokens_basic_chunking(self):\n'
        '        """Test basic token-based chunking."""\n',
        '    def test_tokens_preserve_leading_indentation_when_chunking_mid_block(self):\n'
        '        """Token chunks must retain leading whitespace to keep code formatting intact."""\n',
    ):
        text = _replace_once(text, anchor, TOK_USEFIXTURES + anchor, name)
    text = _replace_once(
        text,
        'def test_hierarchical_tokens_offsets_map_to_source():\n'
        '    """Hierarchical tokens path must map local spans to global offsets and preserve exact source slices."""\n',
        '# --- Ported (chunking-engine-parity Task 4) -----------------------------\n'
        '# Loads the real gpt2 tokenizer; the real_hf_cache fixture forces an\n'
        '# offline read of the real cache (no network).\n'
        "@pytest.mark.usefixtures('real_hf_cache')\n"
        'def test_hierarchical_tokens_offsets_map_to_source():\n'
        '    """Hierarchical tokens path must map local spans to global offsets and preserve exact source slices."""\n',
        name,
    )
    # backward-compat functions live in the Chunk_Lib shim, not the engine pkg.
    # Behavioral coverage for the shim equivalents lives in
    # test_chunk_lib_shim.py plus Tests/Chunking/test_shim_backcompat.py
    # (M3): the ported assertions here are skipped, not deleted, so a future
    # engine-package export re-enables them naturally.
    for fn in ("improved_chunking_process", "chunk_for_embedding"):
        text = _replace_once(
            text,
            f'    def test_{fn}(self):\n',
            f'    def test_{fn}(self):\n'
            f'        # --- Ported (chunking-engine-parity Task 4) ---------------------\n'
            f'        # Upstream\'s {fn} is part of the server package init, which chatbook\n'
            f'        # deliberately does not vendor (spec §5.1); the compat equivalent\n'
            f'        # lives in the Chunk_Lib shim (behavioral coverage:\n'
            f'        # Tests/Chunking/test_shim_backcompat.py, M3).\n'
            f'        pytest.skip(\n'
            f'            "{fn} lives in the Chunk_Lib shim, not the engine package "\n'
            f'            "(spec §5.1); behavioral coverage in test_shim_backcompat.py"\n'
            f'        )\n',
            name,
        )
    return text


def _patch_streaming_overlap(text: str) -> str:
    """Guard every function-level async_chunker import (import failure at call
    time, not collection)."""
    anchor = "from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker"
    if anchor not in text:
        _fail(anchor, "test_streaming_overlap.py")
    out = []
    for line in text.split("\n"):
        if line.strip() == anchor:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(ASYNC_GUARD.format(i=indent).rstrip("\n"))
        out.append(line)
    return "\n".join(out)


def _patch_security(text: str) -> str:
    name = "test_security.py"
    return _replace_once(
        text,
        '    def test_concurrent_request_limits(self):\n'
        '        """Test that concurrent requests are limited."""\n'
        '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        '    def test_concurrent_request_limits(self):\n'
        '        """Test that concurrent requests are limited."""\n'
        + ASYNC_GUARD.format(i="        ")
        + '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        name,
    )


def _patch_security_fixed(text: str) -> str:
    name = "test_security_fixed.py"
    return _replace_once(
        text,
        '    async def test_concurrent_request_limits(self):\n'
        '        """Test that concurrent requests are handled properly."""\n'
        '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        '    async def test_concurrent_request_limits(self):\n'
        '        """Test that concurrent requests are handled properly."""\n'
        + ASYNC_GUARD.format(i="        ")
        + '        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker\n',
        name,
    )


def _patch_thread_safety(text: str) -> str:
    name = "test_thread_safety.py"
    text = _replace_once(
        text, "import pytest\n", "import pytest\n\n" + TOK_HELPER_IMPORT, name
    )
    anchor = (
        '    def test_tokenizer_property_concurrent_initialization(self):\n'
        '        """Test concurrent tokenizer property access is thread-safe."""\n'
    )
    return _replace_once(
        text,
        anchor,
        "    # --- Ported (chunking-engine-parity Task 4) -------------------------\n"
        "    # Loads the real gpt2 tokenizer; the real_hf_cache fixture points the\n"
        "    # HF stack at the real (pre-sandbox) cache with offline mode forced, so\n"
        "    # no network is touched. Skips if gpt2 is genuinely not cached.\n"
        "    @pytest.mark.usefixtures('real_hf_cache')\n"
        + anchor,
        name,
    )


def _patch_offsets_property(text: str) -> str:
    name = "test_chunking_offsets_property.py"
    # The 'tokens' method arm needs the real gpt2 tokenizer: request the
    # real_hf_cache fixture (offline read of the pre-sandbox cache) instead of
    # a module-wide skip, so the property test always RUNS when the cache
    # exists on the machine.
    text = _replace_once(
        text,
        "import pytest\n",
        "import pytest\n\n" + TOK_HELPER_IMPORT,
        name,
    )
    return _replace_once(
        text,
        "from tldw_chatbook.Chunking.engine import Chunker\n",
        "# --- Ported (chunking-engine-parity Task 4) -----------------------------------\n"
        "# Spec §10.2: also run under the production sanitization path (test mode\n"
        "# explicitly off) via the production_path marker; see the\n"
        "# _production_sanitization autouse fixture in Tests/Chunking/conftest.py.\n"
        "pytestmark = pytest.mark.production_path\n"
        "\n"
        "# The 'tokens' method arm resolves the real gpt2 tokenizer. The root\n"
        "# Tests/conftest.py sandboxes HOME per test and the repo network guard\n"
        "# blocks HF downloads, so this module pulls in the real_hf_cache fixture\n"
        "# autouse: it points the HF stack at the REAL cache with offline mode\n"
        "# forced (pure local read, no network) and skips with a true reason only\n"
        "# if gpt2 is genuinely absent from this machine's cache.\n"
        "@pytest.fixture(autouse=True)\n"
        "def _tokens_tokenizer_cache(real_hf_cache):\n"
        "    return real_hf_cache\n"
        "\nfrom tldw_chatbook.Chunking.engine import Chunker\n",
        name,
    )


def _patch_overlap_properties(text: str) -> str:
    name = "test_chunking_overlap_properties.py"
    return _replace_once(
        text,
        "pytestmark = pytest.mark.unit\n",
        "# --- Ported (chunking-engine-parity Task 4) ---------------------------------\n"
        "# Spec §10.2: also run under the production sanitization path (test mode\n"
        "# explicitly off) via the production_path marker; Tests/Chunking/conftest.py.\n"
        "pytestmark = [pytest.mark.unit, pytest.mark.production_path]\n",
        name,
    )


TEST_PATCHES = {
    "test_chunker_v2.py": _patch_chunker_v2,
    "test_streaming_overlap.py": _patch_streaming_overlap,
    "test_security.py": _patch_security,
    "test_security_fixed.py": _patch_security_fixed,
    "test_thread_safety.py": _patch_thread_safety,
    "test_chunking_offsets_property.py": _patch_offsets_property,
    "test_chunking_overlap_properties.py": _patch_overlap_properties,
}


def patch_ported_test(name: str, text: str) -> str:
    """Apply chatbook-side patches to one freshly rewritten upstream test."""
    if name in TESTS_MODULE_SKIPPED:
        return _insert_module_skip(text, name, TESTS_MODULE_SKIPPED[name])
    patcher = TEST_PATCHES.get(name)
    if patcher is not None:
        return patcher(text)
    return text


# ---------------------------------------------------------------------------
# Chatbook-side ENGINE patches (ADR-029 diagnostic privacy).
#
# Same idempotency contract as TEST_PATCHES: sync copies upstream at the pin,
# applies these named patches, and the local-modification check in main()
# compares the working tree against the PATCHED text — so the patched files
# are the canonical vendored state, a re-sync reproduces them exactly, and an
# upstream drift under a patch anchor fails loudly (_replace_once) instead of
# silently dropping a privacy repair. Keep each patch scoped to reviewed
# diagnostic call sites; behavior changes belong upstream (spec §5.2's
# shim/subclass rule still applies to anything beyond log-record content).
# ---------------------------------------------------------------------------

def _patch_chunker_stream_diagnostics(text: str) -> str:
    """TASK-19321 (ADR-029): chunk_file_stream diagnostics must not record
    user file paths — directly or through exception text (an OSError
    stringifies with the filename embedded; a UnicodeDecodeError's message
    carries byte context from the file). The repaired records identify the
    file by a stable content-free handle plus safe metadata."""
    name = "chunker.py"
    text = _replace_once(
        text,
        '        logger.info(f"Stream processing file: {file_path} ({file_size} bytes)")\n',
        "        # ADR-029 / TASK-19321: a user file path is private data, so streaming\n"
        "        # diagnostics identify the file by a stable content-free handle instead\n"
        "        # of the path. An operator can recompute the handle for a candidate\n"
        "        # file with:\n"
        "        #   hashlib.sha256(\n"
        '        #       str(Path(candidate).resolve()).encode("utf-8", "surrogatepass")\n'
        "        #   ).hexdigest()[:12]\n"
        "        # to confirm which file a record refers to.\n"
        "        path_ref = hashlib.sha256(\n"
        '            str(file_path.resolve()).encode("utf-8", "surrogatepass")\n'
        "        ).hexdigest()[:12]\n"
        "\n"
        '        logger.info(f"Stream processing file: path_sha256={path_ref} ({file_size} bytes)")\n',
        name,
    )
    text = _replace_once(
        text,
        "        except UnicodeDecodeError as e:\n"
        '            logger.error(f"File stream decoding failed for {file_path}: {e}")\n'
        "            raise InvalidInputError(\n"
        "                f\"Failed to decode file {file_path} using encoding '{encoding_name}'\"\n"
        "            ) from e\n"
        "        except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:\n"
        '            logger.error(f"File stream processing failed: {e}")\n'
        '            raise ChunkingError(f"Failed to process file stream: {str(e)}") from e\n',
        "        except UnicodeDecodeError as e:\n"
        "            # ADR-029 / TASK-19321: no path, and no raw exception text — a\n"
        "            # UnicodeDecodeError's message carries byte context from the file.\n"
        "            # The codec name and byte offset are content-free and keep the\n"
        "            # failure debuggable.\n"
        "            logger.error(\n"
        '                f"File stream decoding failed for path_sha256={path_ref}: "\n'
        "                f\"{type(e).__name__} (encoding '{encoding_name}', byte offset {e.start})\"\n"
        "            )\n"
        "            raise InvalidInputError(\n"
        "                f\"Failed to decode file {file_path} using encoding '{encoding_name}'\"\n"
        "            ) from e\n"
        "        except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:\n"
        "            # ADR-029 / TASK-19321: an OSError here stringifies with the\n"
        "            # filename embedded, so the record keeps the exception TYPE and\n"
        "            # drops the message; the raised ChunkingError still carries the\n"
        "            # full detail to the caller.\n"
        "            logger.error(\n"
        '                f"File stream processing failed for path_sha256={path_ref}: "\n'
        '                f"{type(e).__name__}"\n'
        "            )\n"
        '            raise ChunkingError(f"Failed to process file stream: {str(e)}") from e\n',
        name,
    )
    return text


ENGINE_PATCHES = {
    "chunker.py": _patch_chunker_stream_diagnostics,
}


def patch_vendored_file(rel: str, text: str) -> str:
    """Apply chatbook-side patches to one freshly rewritten vendored file."""
    patcher = ENGINE_PATCHES.get(rel)
    if patcher is not None:
        return patcher(text)
    return text


def rewrite_imports(src: str) -> str:
    # Mechanical, order matters: the Chunking-specific rule first.
    src = src.replace("tldw_Server_API.app.core.Chunking",
                      "tldw_chatbook.Chunking.engine")
    src = src.replace("tldw_Server_API.app.core",
                      "tldw_chatbook.Chunking._shims")
    # Slashed (filesystem-path) form of the same mapping, e.g. upstream
    # chunker.py's docstring pointer at its own README; keeps the vendored
    # tree free of any `tldw_Server_API` text (spec §0/§5.2, test contract).
    src = src.replace("tldw_Server_API/app/core/Chunking",
                      "tldw_chatbook/Chunking/engine")
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

    # 1. Refuse to overwrite local modifications (loud, spec §5.2). The
    # canonical vendored state is upstream-at-pin + rewrite + ENGINE_PATCHES,
    # so anything else in the tree is a local modification.
    for rel in VENDORED + ["__init__.py"]:
        dst = TARGET_ROOT / rel
        if dst.exists():
            if rel == "__init__.py":
                continue  # chatbook-authored, never touched by sync
            expected = patch_vendored_file(rel, rewrite_imports(git_show(worktree, rel)))
            if dst.read_text() != expected:
                sys.exit(f"FATAL: local modification to vendored file {rel}; "
                         f"revert it or move the change to a shim/subclass")

    # 2. Copy + rewrite + chatbook-side engine patches
    for rel in VENDORED:
        dst = TARGET_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(patch_vendored_file(rel, rewrite_imports(git_show(worktree, rel))))

    # 3. Manifest + licence (GPLv3 §4: licence text ships in-subtree)
    for rel in EXTRA_FILES:
        dst = TARGET_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(
            subprocess.run(["git", "-C", str(worktree), "show", f"{PIN}:{rel}"],
                           capture_output=True).stdout)
    print(f"Synced {len(VENDORED)} files from {REPO} @ {PIN}")

    # 4. Tests (spec §10.1): port with the same import rewrite + chatbook-side
    # patches (see TESTS_MODULE_SKIPPED / TEST_PATCHES above) so a re-sync
    # reproduces the ported tree exactly.
    r = subprocess.run(
        ["git", "-C", str(worktree), "ls-tree", "-r", "--name-only", PIN,
         f"{UPSTREAM_TESTS_ROOT}/"],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FATAL: could not list upstream tests at {PIN}: {r.stderr}")
    upstream_tests = sorted(
        line[len(UPSTREAM_TESTS_ROOT) + 1:]
        for line in r.stdout.splitlines()
        if line.startswith(f"{UPSTREAM_TESTS_ROOT}/") and "/test_" in line
        and line.endswith(".py") and line.split("/")[-1].startswith("test_")
    )
    to_port = [t for t in upstream_tests if t.split("/")[-1] not in TESTS_EXCLUDED]
    for rel in to_port:
        rel_path = Path(rel)
        dst_name = TEST_RENAMES.get(rel_path.name, rel_path.name)
        dst = TARGET_TESTS_ROOT / rel_path.parent / dst_name if rel_path.parent != Path(".") \
            else TARGET_TESTS_ROOT / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = subprocess.run(
            ["git", "-C", str(worktree), "show", f"{PIN}:{UPSTREAM_TESTS_ROOT}/{rel}"],
            capture_output=True, text=True)
        if src.returncode != 0:
            sys.exit(f"FATAL: {rel} not found at pinned SHA {PIN}: {src.stderr}")
        dst.write_text(patch_ported_test(rel_path.name, rewrite_imports(src.stdout)))
    print(f"Ported {len(to_port)} test files into {TARGET_TESTS_ROOT} "
          f"({len(upstream_tests) - len(to_port)} excluded per spec §10.1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
