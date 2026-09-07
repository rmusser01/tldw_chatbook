# TASK-839 Lazy MLX Import Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Parakeet MLX and Lightning Whisper MLX out of unrelated application imports and test collection while preserving installed-provider discovery and first-use behavior.

**Architecture:** `transcription_service.py` will retain cheap `find_spec` discovery but replace module-level native imports with two cached lazy loaders. Existing file, buffer, and streaming paths will call a loader only when they actually need to construct a model, preserving input validation and cached-model behavior.

**Tech Stack:** Python 3.11+, `importlib`, pytest subprocess tests, Ruff

---

## Files

- Create: `Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py`
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`
- Modify: `Tests/ProductionApp/test_chat_root_state_removal.py`
- Modify: `Tests/ProductionApp/test_provider_selection_ownership.py`
- Verify unchanged: `Tests/test_config_stt_provider_probe.py`
- Modify for closeout:
  `backlog/tasks/task-839 - Prevent-optional-MLX-imports-from-aborting-test-collection.md`

ADR required: no

ADR path: N/A

Reason: This defers existing optional imports without changing provider
ownership, dependencies, storage, schema, or service contracts.

### Task 1: Implement the complete lazy import boundary with TDD

**Files:**

- Create: `Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py`
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`

- [ ] **Step 1: Write a failing subprocess import test**

Use the isolated environment helper pattern from
`Tests/test_config_stt_provider_probe.py`. The child process must:

```python
import builtins
import importlib.machinery
import importlib.util
import json
import sys

guarded = {"parakeet_mlx", "lightning_whisper_mlx"}
original_find_spec = importlib.util.find_spec
original_import = builtins.__import__


def find_spec(name, package=None):
    if name in guarded:
        return importlib.machinery.ModuleSpec(name, loader=None)
    return original_find_spec(name, package)


def reject_runtime_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in guarded:
        raise AssertionError(f"optional runtime imported: {name}")
    return original_import(name, *args, **kwargs)


importlib.util.find_spec = find_spec
builtins.__import__ = reject_runtime_import
sys.platform = "darwin"

from tldw_chatbook.Local_Ingestion import transcription_service

print(
    json.dumps(
        {
            "parakeet_available":
                transcription_service.PARAKEET_MLX_AVAILABLE,
            "lightning_available":
                transcription_service.LIGHTNING_WHISPER_AVAILABLE,
            "parakeet_loaded":
                transcription_service.parakeet_from_pretrained is not None,
            "lightning_loaded":
                transcription_service.LightningWhisperMLX is not None,
        }
    )
)
```

Assert a zero return code, both availability flags true, and both symbols
unset. Set `PYTHONPATH` only to the project root—never to a stub directory.

- [ ] **Step 2: Write failing loader lifecycle tests**

Parameterize the two loaders. Reset each flag to true and symbol to `None`,
patch `importlib.import_module`, then assert the loader imports once and returns
the cached symbol on its second call:

```python
assert ensure_import() is expected_symbol
assert ensure_import() is expected_symbol
assert imported_modules == [expected_module]
```

Add an import-failure case using `RuntimeError("unsafe mlx")`. Assert:

- the first call raises `TranscriptionError`;
- the original exception is `__cause__`;
- the availability flag becomes false;
- the backend symbol stays `None`;
- a second call does not retry `import_module`.

- [ ] **Step 3: Write failing execution-path tests**

Construct a `TranscriptionService` with `get_cli_setting` returning supplied
defaults. Patch each ensure function to raise a test-only sentinel and drive
these fresh-model paths:

- Lightning file transcription;
- Parakeet file transcription with `soundfile` reads mocked;
- Parakeet direct-buffer transcription with a valid two-byte sample;
- Parakeet streaming-transcriber creation.

Each call must reach the sentinel immediately before model construction. It
must not require a real MLX import. Do not require the loader to run before
normal input validation or when a matching model is already cached.

- [ ] **Step 4: Run the new file and confirm RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py -q
```

Expected: the subprocess rejects the current eager import and the lazy-loader
tests fail because the ensure functions do not exist.

- [ ] **Step 5: Replace module-level imports with discovery and loaders**

At module scope:

```python
def _optional_module_available(module_name: str) -> bool:
    try:
        return (
            sys.platform == "darwin"
            and importlib.util.find_spec(module_name) is not None
        )
    except (AttributeError, ImportError, ValueError):
        return False


LIGHTNING_WHISPER_AVAILABLE = _optional_module_available(
    "lightning_whisper_mlx"
)
PARAKEET_MLX_AVAILABLE = _optional_module_available("parakeet_mlx")
LightningWhisperMLX = None
parakeet_from_pretrained = None
```

After `TranscriptionError` is defined, add two explicit functions:

```python
def _ensure_lightning_whisper_mlx_import():
    global LIGHTNING_WHISPER_AVAILABLE, LightningWhisperMLX
    if LightningWhisperMLX is not None:
        return LightningWhisperMLX
    if not LIGHTNING_WHISPER_AVAILABLE:
        raise TranscriptionError("lightning-whisper-mlx is not installed")
    try:
        module = importlib.import_module("lightning_whisper_mlx")
        LightningWhisperMLX = module.LightningWhisperMLX
    except Exception as exc:
        LIGHTNING_WHISPER_AVAILABLE = False
        raise TranscriptionError(
            "lightning-whisper-mlx could not be loaded"
        ) from exc
    return LightningWhisperMLX


def _ensure_parakeet_mlx_import():
    global PARAKEET_MLX_AVAILABLE, parakeet_from_pretrained
    if parakeet_from_pretrained is not None:
        return parakeet_from_pretrained
    if not PARAKEET_MLX_AVAILABLE:
        raise TranscriptionError("parakeet-mlx is not installed")
    try:
        module = importlib.import_module("parakeet_mlx")
        parakeet_from_pretrained = module.from_pretrained
    except Exception as exc:
        PARAKEET_MLX_AVAILABLE = False
        raise TranscriptionError("parakeet-mlx could not be loaded") from exc
    return parakeet_from_pretrained
```

Do not catch `BaseException`, retry a failed import, add a loader lock, or add a
dependency registry.

- [ ] **Step 6: Wire only actual model construction**

Inside each existing model-cache miss:

```python
lightning_whisper_cls = _ensure_lightning_whisper_mlx_import()
lightning_model = lightning_whisper_cls(...)
```

and:

```python
parakeet_loader = _ensure_parakeet_mlx_import()
self._parakeet_mlx_model = parakeet_loader(...)
```

Apply the Parakeet form to file, buffer, and streaming model loads. Preserve
all existing input validation, model locks, cached-model fast paths, error copy,
and the streaming method's `None` fallback.

- [ ] **Step 7: Run focused GREEN tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  Tests/Transcription/test_mlx_whisper_transcription.py \
  -k "lazy or model_loading or model_caching or not_available" -q
```

Expected: all selected tests pass without importing an actual MLX backend.

- [ ] **Step 8: Confirm raw symbols are no longer invoked**

```bash
rg -n "LightningWhisperMLX\\(|parakeet_from_pretrained\\(" \
  tldw_chatbook/Local_Ingestion/transcription_service.py
```

Expected: no direct model-construction call remains.

- [ ] **Step 9: Commit the atomic implementation**

```bash
git add \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py
git commit -m "fix(stt): defer optional MLX runtime imports"
```

### Task 2: Remove obsolete ProductionApp stubs

**Files:**

- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`
- Modify: `Tests/ProductionApp/test_chat_root_state_removal.py`
- Modify: `Tests/ProductionApp/test_provider_selection_ownership.py`

- [ ] **Step 1: Remove only import scaffolding**

Delete the `_MISSING_MODULE` sentinels, `sys.modules["parakeet_mlx"] = None`,
and the surrounding restore blocks. Keep the same application/test imports as
ordinary module-level imports. Remove `import sys` and `# ruff: noqa: E402`
only where they become unused. Do not change product assertions or add
replacement stubs.

- [ ] **Step 2: Run the three affected modules**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_provider_selection_ownership.py -q
```

Expected: all tests collect and pass without importing or stubbing MLX.

- [ ] **Step 3: Commit the test cleanup**

```bash
git add \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_provider_selection_ownership.py
git commit -m "test: remove obsolete MLX collection stubs"
```

### Task 3: Scoped verification and closeout

**Files:**

- Modify:
  `backlog/tasks/task-839 - Prevent-optional-MLX-imports-from-aborting-test-collection.md`

- [ ] **Step 1: Run the exact scoped tests**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py \
  Tests/test_config_stt_provider_probe.py \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_provider_selection_ownership.py -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  Tests/Transcription/test_mlx_whisper_transcription.py \
  -k "model_loading or model_caching or not_available" -q
```

Do not run the repository-wide suite.

- [ ] **Step 2: Run touched-file checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_provider_selection_ownership.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py \
  Tests/ProductionApp/test_chat_composition_retirement.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_provider_selection_ownership.py

git diff --check origin/dev...HEAD
```

- [ ] **Step 3: Review scope**

Confirm no citation files, provider defaults, routing, schema, dependencies, or
runtime ownership changed. Confirm only explicit MLX model construction invokes
the loaders and no ProductionApp MLX stubs remain.

- [ ] **Step 4: Complete TASK-839 and commit**

Check all acceptance criteria, add concise implementation notes with RED/GREEN
evidence and the no-ADR decision, set the task Done with Backlog CLI, then:

```bash
git add \
  'backlog/tasks/task-839 - Prevent-optional-MLX-imports-from-aborting-test-collection.md'
git commit -m "docs: close TASK-839"
```
