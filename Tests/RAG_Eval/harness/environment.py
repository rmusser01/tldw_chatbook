# Tests/RAG_Eval/harness/environment.py
"""The environment gate for the env-gated retrieval eval harness.

Deliberately light: stdlib plus `importlib.util.find_spec` probes and (only
once the extras are known to be installed) `huggingface_hub`. Nothing here
imports torch, transformers, chromadb, or any `tldw_chatbook` module that
drags the RAG stack in — the whole point is that a machine WITHOUT the
extras can evaluate this gate at collection time and skip cheaply.

Three conditions, each with its own actionable reason:

1. `RAG_EVAL=1` — the harness embeds ~50 documents on a real model and takes
   minutes, so it never runs as part of an ordinary suite.
2. The `embeddings_rag` extras — probed through the app's own cheap
   `embeddings_rag_deps_installed()` (find_spec only, no imports, no
   registry mutation), so the gate and the app agree on what "installed"
   means.
3. The embedding model is already in the local model cache — a harness that
   silently downloads 87 MB on a cold machine is a harness that fails in CI
   for a reason nobody can see. If it is not cached, we say so and skip.
   Checked against the model snapshot's COMPLETE file list, not the minimum
   the loader can scrape by on: downloads are genuinely blocked during a run
   (see the offline latch below), so a half-populated cache has to land on
   the skip-with-reason path rather than raise mid-run.

**The cache-directory invariant.** Condition 3 is only meaningful if the
directory it checks is the directory the run will actually load from. The
app resolves that directory through `tldw_chatbook.config.get_model_cache_dir()`,
which hangs off `get_user_data_dir()` — and the test suite repoints
`HOME`/`XDG_DATA_HOME` at a throwaway sandbox, which would make every run a
cold cache miss and a fresh 87 MB download. So `Tests/RAG_Eval/conftest.py`
points `get_model_cache_dir` back at `model_cache_dir()` below for the
duration of a harness test. Same function, same answer, in the gate and in
the run.

**Why `~` is not good enough here.** `Tests/conftest.py` sandboxes `HOME`
(and `USERPROFILE`) at *module* level — line 57, during collection, before
this module is even imported — so `os.path.expanduser("~")` resolves into a
`tldw_test_config_*` temp directory for the whole session, not just inside
a test. Discovered the hard way: the first RED run skipped with "model is
not in the local model cache (/private/var/folders/.../home/.cache/...)"
on a machine where the model was very much cached. The passwd database is
the sandbox-proof answer, and only the `~` fallback branch below needs it:
`HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME` are not sandboxed, so when
one of those is set it is already the real answer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest
from loguru import logger

#: Env var that opts a run into the (slow, real-model) harness. Defined
#: before the offline latch below, which reads it.
RAG_EVAL_ENV_VAR = "RAG_EVAL"

# ---------------------------------------------------------------------------
# Offline latch — MUST run before `huggingface_hub.constants` is evaluated
# ---------------------------------------------------------------------------
# `huggingface_hub.constants.HF_HUB_OFFLINE` is computed ONCE, at import
# (constants.py L171 in hf_hub 1.12.0), from the environment as it stood at
# that instant; `constants.is_offline_mode()` just returns that global, and
# transformers 5.x imports that same function (`utils/hub.py` L363). So
# setting the env var from a pytest fixture — i.e. at test SETUP, after
# collection has already imported half the world — does nothing at all.
#
# Measured, on a passing gated test with the env var set from the fixture:
#     ENV HF_HUB_OFFLINE='1'   constants.HF_HUB_OFFLINE=False
#                              constants.is_offline_mode()=False
# The enforcement was inert, and a cache miss would have silently downloaded
# into the user's real cache (which this module deliberately points at).
#
# Both halves are needed and neither is sufficient alone:
#   * this env write, which lands before `huggingface_hub.constants` is
#     evaluated in the common case — note that is a weaker condition than
#     "before huggingface_hub is imported", because hf_hub loads its
#     submodules lazily, so `constants` can still be unevaluated while
#     `huggingface_hub` sits in sys.modules; and
#   * `Tests/RAG_Eval/conftest.py`'s `monkeypatch.setattr` on the constant,
#     the only thing that works once `constants` HAS been evaluated with the
#     var unset (any earlier test that touched transformers does this).
#
# Both halves were mutation-tested, forcing the hard case by evaluating
# `huggingface_hub.constants` from `Tests/RAG_Eval/__init__.py` (i.e. before
# this latch):
#     constants pre-evaluated, conftest setattr removed -> is_offline_mode() False
#     constants pre-evaluated, conftest setattr present -> is_offline_mode() True
# and the assertion in test_harness_smoke.py fails ("assert False is True")
# in the first configuration. Setting the constant works because
# `is_offline_mode()` reads it as a module global at call time; transformers
# 5.x imports that same function, so it inherits the change (also measured).
if os.environ.get(RAG_EVAL_ENV_VAR) == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__all__ = [
    "EMBEDDING_MODEL_REPO_ID",
    "PROFILE_EMBEDDING_MODEL",
    "PROFILE_NAME",
    "RAG_EVAL_ENV_VAR",
    "harness_gate",
    "model_cache_dir",
    "skip_reason",
]

#: Profile the harness builds from. Its `default_search_mode` is "hybrid";
#: Task 6 switches `service.config.search.default_search_mode` per pass.
PROFILE_NAME = "hybrid_basic"

#: The embedding model string as the `hybrid_basic` profile spells it. This
#: exact string is the collection-fingerprint input
#: (`collection_fingerprint.py`'s `_index_fields`), so it must never be
#: "helpfully" canonicalized — see `_BARE_HF_MODEL_ID_ALIASES` in
#: `RAG_Search/simplified/embeddings_wrapper.py`.
PROFILE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: The canonical HuggingFace repo id the string above resolves to, and the
#: only id that exists in a HF cache directory.
EMBEDDING_MODEL_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: Escape hatch for a machine that keeps its HF cache somewhere exotic.
MODEL_CACHE_ENV_VAR = "TLDW_RAG_EVAL_MODEL_CACHE"

#: Every file this model's snapshot ships, as any-of groups.
#:
#: Deliberately the COMPLETE snapshot, not the strict minimum. A leave-one-out
#: probe against the real cache (copy the snapshot to a temp dir, drop one
#: file, `AutoTokenizer`+`AutoModel.from_pretrained(..., local_files_only=True)`
#: — the app's actual loader, `Embeddings_Lib._HFEmbedder`, which uses
#: transformers rather than sentence-transformers' own loader) found only two
#: files strictly load-bearing:
#:
#:     omit config.json             -> FAILS (ValueError: Unrecognized model)
#:     omit model.safetensors       -> FAILS (OSError: no file named ...)
#:     omit special_tokens_map.json -> LOADS
#:     omit tokenizer_config.json   -> LOADS
#:     omit tokenizer.json          -> LOADS   (vocab.txt covers it)
#:     omit vocab.txt               -> LOADS   (tokenizer.json covers it)
#:
#: The tokenizer pair is therefore a genuine any-of and is encoded as one.
#: The two optional JSONs are still required here because the gate's job is
#: not "can this scrape by" but "is this cache whole": now that offline mode
#: is genuinely enforced, a half-downloaded cache must land on the
#: skip-with-reason path rather than raise somewhere in the middle of a run.
#: The weights group stays any-of because safetensors-vs-pickle is a real
#: format alternative that changes across snapshots.
_REQUIRED_CACHE_FILES: tuple[tuple[str, ...], ...] = (
    ("config.json",),
    ("model.safetensors", "pytorch_model.bin"),
    ("tokenizer.json", "vocab.txt"),
    ("tokenizer_config.json",),
    ("special_tokens_map.json",),
)

_EXTRAS_HINT = 'pip install "tldw_chatbook[embeddings_rag]"'


def _unsandboxed_home() -> Path:
    """The user's real home directory, immune to `$HOME` sandboxing.

    See the module docstring: the root conftest overwrites `$HOME` at import
    time, so `expanduser` cannot be trusted for this one lookup. The passwd
    database is not reachable on Windows, where `USERPROFILE` (also
    sandboxed) is what `expanduser` reads — nothing better is available
    there, so it degrades to the sandboxed answer and the gate skips with an
    honest "not cached, point %s at it" message rather than downloading.
    """
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path(os.path.expanduser("~"))


def _validate_cache_dir_override(raw: str, *, env_var: str) -> Optional[Path]:
    """Run one env-sourced cache-dir override through `path_validation.py`.

    Qodo PR #1458 finding 3: `_resolve_model_cache_dir` used to build a
    `Path` straight out of an environment variable with no traversal or
    injection screen. Mirrors the treatment `RAGService._keyword_search`
    already gives `config.search.media_db_path` (Qodo PR #1428 finding 1):
    lexical normalization plus `validate_path_simple`'s screen, with
    `probe_existing=False` because filesystem/symlink authority belongs to
    whatever eventually opens the cache, not this gate.

    Args:
        raw: The raw environment variable value.
        env_var: The variable's name, only used to name it in the warning.

    Returns:
        The validated, lexically normalized path, or ``None`` when `raw`
        fails validation — logged as a warning, never raised.
    """
    from tldw_chatbook.Utils.path_validation import validate_path_simple
    from tldw_chatbook.Utils.private_paths import lexical_path

    try:
        return lexical_path(
            validate_path_simple(
                Path(raw).expanduser(),
                require_exists=False,
                probe_existing=False,
            )
        )
    except ValueError as exc:
        logger.warning(
            f"Rejected {env_var}={raw!r} for the model cache directory: "
            f"{exc}; falling back to the default HuggingFace cache location."
        )
        return None


def _resolve_model_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache exactly as huggingface_hub does.

    Mirrors `huggingface_hub.constants`: `HF_HUB_CACHE` wins outright, then
    `$HF_HOME/hub`, then `$XDG_CACHE_HOME/huggingface/hub`, then
    `~/.cache/huggingface/hub` — with the last branch reading the real home
    rather than `$HOME` (module docstring). The highest-priority env var
    that is actually set is run through `path_validation.py`'s
    traversal/injection screen (`_validate_cache_dir_override`); one that
    fails it is logged and resolution degrades straight to the default
    `~/.cache/huggingface/hub` location rather than raising or falling
    through to a lower-priority candidate — an unvalidated env value must
    never reach the filesystem checks `skip_reason()` runs against this
    directory.
    """
    for env_var, suffix in (
        (MODEL_CACHE_ENV_VAR, ()),
        ("HF_HUB_CACHE", ()),
        ("HF_HOME", ("hub",)),
        ("XDG_CACHE_HOME", ("huggingface", "hub")),
    ):
        raw = os.environ.get(env_var)
        if not raw:
            continue
        validated = _validate_cache_dir_override(raw, env_var=env_var)
        if validated is None:
            break
        return validated.joinpath(*suffix)
    return _unsandboxed_home() / ".cache" / "huggingface" / "hub"


#: Captured at import (collection time) — which is AFTER `Tests/conftest.py`
#: has already repointed `$HOME` (its sandboxing happens at module level
#: during collection, before this module is even imported; see the module
#: docstring). Import order buys nothing here; this is correct only because
#: the `~` fallback in `_resolve_model_cache_dir()` goes through
#: `_unsandboxed_home()`'s passwd-database lookup instead of `$HOME`.
_MODEL_CACHE_DIR: Path = _resolve_model_cache_dir()


def model_cache_dir() -> Path:
    """The model cache directory the harness both checks and loads from."""
    return _MODEL_CACHE_DIR


def _cached_file(filename: str) -> Optional[str]:
    """Return the cached path for one repo file, or None when absent.

    Uses `huggingface_hub.try_to_load_from_cache`, which never touches the
    network and never imports torch. A `_CACHED_NO_EXIST` sentinel (the
    "we asked and the file genuinely does not exist upstream" marker) is not
    a string, so it reads as absent here — which is the correct answer for
    "can this file be loaded offline".
    """
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(
            EMBEDDING_MODEL_REPO_ID,
            filename,
            cache_dir=str(_MODEL_CACHE_DIR),
        )
    except Exception:
        return None
    return found if isinstance(found, str) else None


def _embedding_model_is_cached() -> bool:
    """True when every file the loader needs is already on disk."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return all(
        any(_cached_file(name) is not None for name in alternatives)
        for alternatives in _REQUIRED_CACHE_FILES
    )


def skip_reason() -> Optional[str]:
    """Why this environment cannot run the harness, or None when it can.

    Returns:
        An actionable one-line reason naming the exact remedy (env var,
        install command, or model id + cache directory), or None.
    """
    if os.environ.get(RAG_EVAL_ENV_VAR) != "1":
        return "set RAG_EVAL=1 to run the retrieval eval harness"

    from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

    if not embeddings_rag_deps_installed():
        return (
            "the retrieval eval harness needs the embeddings_rag extras: "
            f"{_EXTRAS_HINT}"
        )

    if not _embedding_model_is_cached():
        return (
            f"embedding model {EMBEDDING_MODEL_REPO_ID!r} is not in the local "
            f"model cache ({_MODEL_CACHE_DIR}); the harness never downloads "
            "models. Pre-fetch it, or point "
            f"{MODEL_CACHE_ENV_VAR} at the cache that has it."
        )

    return None


def harness_gate() -> pytest.MarkDecorator:
    """The `pytestmark` an env-gated harness module applies to itself.

    Per-module rather than a directory-wide autouse skip on purpose: the
    always-on metric/gating/fixture-integrity modules in this directory must
    keep running with no env var set, and a directory-wide gate would
    silently take them down with it.
    """
    reason = skip_reason()
    return pytest.mark.skipif(reason is not None, reason=reason or "")
