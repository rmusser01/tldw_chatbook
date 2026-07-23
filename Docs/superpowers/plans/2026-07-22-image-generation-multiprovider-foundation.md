# Image Generation — Multi-Provider Foundation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port tldw_server's `Image_Generation` core into `tldw_chatbook` as a self-contained, 6-backend image-generation engine, re-homed onto TOML + keyring + httpx, proven by a throwaway command-palette demo panel — with no chat UI and no DB schema change.

**Architecture:** The server package is FastAPI/pydantic-free and its adapters are fully synchronous. We port the pure files verbatim, re-home only the config loader (nested TOML + keyring) and a sync `http_client` shim, and drive the blocking adapters from a Textual thread worker. A temporary demo screen renders one generated image to prove the pipeline end-to-end.

**Tech Stack:** Python ≥3.11, Textual, httpx (sync), Pillow, loguru — all already core deps. No new pip dependencies. `stable_diffusion_cpp` requires a user-supplied local `sd` binary at runtime only.

**Design spec:** `Docs/superpowers/specs/2026-07-22-image-generation-multiprovider-foundation-design.md` (read it before starting).

## Global Constraints

- **Server source root (SRV):** `/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Image_Generation` — the files being ported. Read-only reference; copy from here.
- **Destination package (DEST):** `tldw_chatbook/Image_Generation/`.
- **Keep the server's flat field names identical** in `ImageGenerationConfig` (`swarmui_base_url`, `openrouter_image_api_key`, …) so ported adapters/`request_validation`/`listing` need no edits.
- **`loguru` is this app's logger** — ported files keep `from loguru import logger` unchanged (no logger swap).
- **No DB schema migration** in this phase. No writes to ChaChaNotes.
- **Secrets never logged.** The config object holds resolved secrets in memory — never log/serialize it wholesale.
- **Import-light package:** `tldw_chatbook/Image_Generation/__init__.py` must NOT import adapters or Pillow at import time (adapters lazy-load via the registry; `image_format_utils`/Pillow load only when an adapter runs).
- **First-run defaults:** `default_backend = "swarmui"`, `enabled_backends = ["swarmui"]`.
- **Adapters are synchronous & blocking** — never call `.generate()` on the asyncio/UI loop; only from a thread worker.
- **Tests run in the project venv only:** `source .venv/bin/activate` first. Never broad-`pkill pytest` (multi-session environment) — scope test runs to `Tests/Image_Generation/`.
- **Standard import-repoint recipe** (used by every "port" task; macOS `sed`):

  ```bash
  # $F = destination file just copied from SRV
  sed -i '' \
    -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' \
    -e 's#tldw_Server_API\.app\.core\.http_client#tldw_chatbook.Image_Generation.http_client#g' \
    -e 's#tldw_Server_API\.app\.core\.Security\.egress#tldw_chatbook.Image_Generation.http_client#g' \
    "$F"
  ```
- **Every task's commit runs in the worktree** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-foundation` on branch `claude/image-gen-multiprovider-foundation`.

## File structure (created in this plan)

```
tldw_chatbook/Image_Generation/
  __init__.py                 # import-light public surface
  exceptions.py               # port verbatim
  capabilities.py             # port verbatim (defines ResolvedReferenceImage)
  prompt_refinement.py        # port verbatim
  config.py                   # RE-HOME: nested TOML + env + keyring -> flat dataclass
  http_client.py              # NEW: sync fetch_json/create_client + light egress guard
  request_validation.py       # port verbatim
  listing.py                  # port verbatim
  adapter_registry.py         # port; repoint DEFAULT_ADAPTERS strings
  worker.py                   # NEW: request builder + thread-worker entry
  adapters/
    __init__.py
    base.py                   # port verbatim (ImageGenRequest/ImageGenResult/protocol)
    image_format_utils.py     # port; re-home fetch_image_bytes onto local http_client
    stable_diffusion_cpp_adapter.py  # port
    swarmui_adapter.py               # port
    openrouter_image_adapter.py      # port
    novita_image_adapter.py          # port
    together_image_adapter.py        # port
    modelstudio_image_adapter.py     # port
tldw_chatbook/config.py         # MODIFY: add [image_generation] default template block
tldw_chatbook/UI/Screens/image_gen_demo_screen.py   # NEW: throwaway demo panel
tldw_chatbook/UI/image_gen_command_provider.py       # NEW: command-palette entry
tldw_chatbook/app.py            # MODIFY: register the command provider
Tests/Image_Generation/         # NEW test package
```

**Not ported (dropped):** `reference_images.py` (media-DB/object-store coupled; only ModelStudio uses it, and it guards on `reference_image is not None`). `config.py`'s server loader body is replaced, not copied.

---

### Task 1: Package skeleton + exceptions

**Files:**
- Create: `tldw_chatbook/Image_Generation/__init__.py`, `tldw_chatbook/Image_Generation/exceptions.py`, `tldw_chatbook/Image_Generation/adapters/__init__.py`
- Create: `Tests/Image_Generation/__init__.py`, `Tests/Image_Generation/test_package_skeleton.py`

**Interfaces:**
- Produces: package `tldw_chatbook.Image_Generation`; `exceptions.ImageGenerationError(RuntimeError)`, `exceptions.ImageBackendUnavailableError(ImageGenerationError)`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_package_skeleton.py
def test_exceptions_hierarchy():
    from tldw_chatbook.Image_Generation.exceptions import (
        ImageGenerationError, ImageBackendUnavailableError,
    )
    assert issubclass(ImageGenerationError, RuntimeError)
    assert issubclass(ImageBackendUnavailableError, ImageGenerationError)

def test_package_imports_clean():
    import importlib
    mod = importlib.import_module("tldw_chatbook.Image_Generation")
    assert mod is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_package_skeleton.py -v`
Expected: FAIL (`ModuleNotFoundError: tldw_chatbook.Image_Generation`).

- [ ] **Step 3: Copy exceptions.py and create the package files**

```bash
mkdir -p tldw_chatbook/Image_Generation/adapters Tests/Image_Generation
cp "$SRV/exceptions.py" tldw_chatbook/Image_Generation/exceptions.py   # $SRV per Global Constraints
: > tldw_chatbook/Image_Generation/adapters/__init__.py
: > Tests/Image_Generation/__init__.py
```

Write `tldw_chatbook/Image_Generation/__init__.py` (import-light — only re-export the cheap symbols; do NOT import adapters/config/http_client here yet):

```python
"""Multi-provider image generation (ported from tldw_server). Import-light."""
from tldw_chatbook.Image_Generation.exceptions import (
    ImageGenerationError,
    ImageBackendUnavailableError,
)

__all__ = ["ImageGenerationError", "ImageBackendUnavailableError"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_package_skeleton.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/ Tests/Image_Generation/
git commit -m "feat(imagegen): package skeleton + exceptions (ported)"
```

---

### Task 2: Core contracts (capabilities + adapters/base)

**Files:**
- Create: `tldw_chatbook/Image_Generation/capabilities.py`, `tldw_chatbook/Image_Generation/adapters/base.py`
- Test: `Tests/Image_Generation/test_contracts.py`

**Interfaces:**
- Produces: `capabilities.ResolvedReferenceImage`, `capabilities.ReferenceImageCapability`, `capabilities.resolve_reference_image_capability`, `capabilities.resolve_backend_reference_image_capability`; `adapters.base.ImageGenRequest`, `adapters.base.ImageGenResult`, `adapters.base.ImageGenerationAdapter` (Protocol).
- `ImageGenRequest(backend, prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, model, format, extra_params, request_id=None, reference_image=None)` — frozen.
- `ImageGenResult(content: bytes, content_type: str, bytes_len: int)` — frozen.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_contracts.py
def test_request_and_result_dataclasses():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
    req = ImageGenRequest(
        backend="swarmui", prompt="a red dragon", negative_prompt=None,
        width=512, height=512, steps=20, cfg_scale=7.0, seed=-1,
        sampler=None, model=None, format="png", extra_params={},
    )
    assert req.backend == "swarmui"
    assert req.reference_image is None  # default
    res = ImageGenResult(content=b"\x89PNG", content_type="image/png", bytes_len=4)
    assert res.bytes_len == 4

def test_resolved_reference_image_defined_locally():
    # Must be defined in capabilities.py, NOT imported from reference_images (which we dropped)
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
    r = ResolvedReferenceImage(
        file_id=1, filename=None, mime_type="image/png",
        width=None, height=None, bytes_len=3, content=b"abc", temp_path=None,
    )
    assert r.mime_type == "image/png"

def test_adapter_is_structural_protocol():
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenerationAdapter
    from typing import runtime_checkable, Protocol
    # It is a Protocol; a duck-typed object with name/supported_formats/generate satisfies it structurally.
    assert issubclass(ImageGenerationAdapter, Protocol)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_contracts.py -v`
Expected: FAIL (`ModuleNotFoundError` for `capabilities`/`base`).

- [ ] **Step 3: Port the files + repoint imports**

```bash
cp "$SRV/capabilities.py" tldw_chatbook/Image_Generation/capabilities.py
cp "$SRV/adapters/base.py" tldw_chatbook/Image_Generation/adapters/base.py
for F in tldw_chatbook/Image_Generation/capabilities.py tldw_chatbook/Image_Generation/adapters/base.py; do
  sed -i '' \
    -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' \
    -e 's#tldw_Server_API\.app\.core\.http_client#tldw_chatbook.Image_Generation.http_client#g' \
    -e 's#tldw_Server_API\.app\.core\.Security\.egress#tldw_chatbook.Image_Generation.http_client#g' \
    "$F"
done
```

Then open both files and confirm no remaining `tldw_Server_API` imports:
`grep -rn "tldw_Server_API" tldw_chatbook/Image_Generation/capabilities.py tldw_chatbook/Image_Generation/adapters/base.py` → must be empty. (`capabilities.py`'s only cross-module import is a `TYPE_CHECKING`-guarded `config` import, now repointed to `tldw_chatbook.Image_Generation.config` — that module arrives in Task 4; because it's `TYPE_CHECKING`-only, imports work now.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_contracts.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/capabilities.py tldw_chatbook/Image_Generation/adapters/base.py Tests/Image_Generation/test_contracts.py
git commit -m "feat(imagegen): port core contracts (capabilities + adapters/base)"
```

---

### Task 3: Prompt refinement

**Files:**
- Create: `tldw_chatbook/Image_Generation/prompt_refinement.py`
- Test: `Tests/Image_Generation/test_prompt_refinement.py`

**Interfaces:**
- Produces: `prompt_refinement.refine_image_prompt(prompt, *, mode="auto", quality_suffix=..., max_length=None) -> str`; `prompt_refinement.normalize_prompt_refinement_mode(value, *, default="auto") -> str`; constant `DEFAULT_QUALITY_SUFFIX`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_prompt_refinement.py
import pytest

@pytest.fixture
def pr():
    from tldw_chatbook.Image_Generation import prompt_refinement as m
    return m

def test_off_returns_prompt_unchanged(pr):
    assert pr.refine_image_prompt("a cat", mode="off") == "a cat"

def test_basic_always_appends_suffix(pr):
    out = pr.refine_image_prompt("a cat", mode="basic")
    assert pr.DEFAULT_QUALITY_SUFFIX in out

def test_auto_skips_when_prompt_already_detailed(pr):
    detailed = "a cat, highly detailed, cinematic lighting, sharp focus, 8k, masterpiece composition"
    assert pr.refine_image_prompt(detailed, mode="auto") == detailed  # has quality cues -> no append

def test_auto_appends_for_short_sparse_prompt(pr):
    out = pr.refine_image_prompt("a cat", mode="auto")
    assert pr.DEFAULT_QUALITY_SUFFIX in out

def test_normalize_mode_aliases(pr):
    assert pr.normalize_prompt_refinement_mode(True) == "basic"
    assert pr.normalize_prompt_refinement_mode(False) == "off"
    assert pr.normalize_prompt_refinement_mode("none") == "off"
    assert pr.normalize_prompt_refinement_mode("garbage") == "auto"  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_prompt_refinement.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Port the file**

```bash
cp "$SRV/prompt_refinement.py" tldw_chatbook/Image_Generation/prompt_refinement.py
# no cross-package imports; repoint is a no-op but run it for consistency:
sed -i '' -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' tldw_chatbook/Image_Generation/prompt_refinement.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_prompt_refinement.py -v`
Expected: PASS (5 passed). If `test_auto_skips_when_prompt_already_detailed` fails, read the ported `_needs_quality_guidance`/`_QUALITY_CUES` to align the test's cue words — do not weaken the assertion beyond matching the ported logic.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/prompt_refinement.py Tests/Image_Generation/test_prompt_refinement.py
git commit -m "feat(imagegen): port prompt refinement"
```

---

### Task 4: Config re-home (the crux) + config template

**Files:**
- Create: `tldw_chatbook/Image_Generation/config.py`
- Modify: `tldw_chatbook/config.py` (add `[image_generation]` default template block)
- Test: `Tests/Image_Generation/test_config_loader.py`

**Interfaces:**
- Consumes: `tldw_chatbook.config.get_cli_setting`, `tldw_chatbook.config.load_settings`, `tldw_chatbook.config.get_api_key`; keyring (optional).
- Produces: `config.ImageGenerationConfig` (frozen dataclass, server field names); `config.get_image_generation_config(reload=False) -> ImageGenerationConfig`; `config.reset_image_generation_config_cache()`; all server `DEFAULT_*` constants; helper `config._load_image_generation_section() -> dict` (flat mapping).

**Approach:** Copy the server `config.py` to preserve its `DEFAULT_*` constants, `_coerce_*`/`_parse_*` helpers, `ImageGenerationConfig` dataclass, and the `get_image_generation_config()` builder **verbatim**. Replace ONLY its data source: swap `get_config_section("Image-Generation")` for a new `_load_image_generation_section()` that assembles the identical flat mapping from nested TOML + env + keyring, and populate secret fields per spec §4.3.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_config_loader.py
import pytest

@pytest.fixture(autouse=True)
def _reset_cache():
    from tldw_chatbook.Image_Generation import config as c
    c.reset_image_generation_config_cache()
    yield
    c.reset_image_generation_config_cache()

def test_defaults_when_unconfigured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    # No TOML section, no env, no keyring: fall back to documented defaults.
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)  # avoid real keyring
    for var in ("OPENROUTER_API_KEY", "NOVITA_API_KEY", "TOGETHER_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == c.DEFAULT_SWARMUI_BASE_URL
    assert cfg.max_width == c.DEFAULT_MAX_WIDTH
    assert cfg.openrouter_image_api_key in (None, "")  # unconfigured

def test_nested_toml_flattens_to_flat_fields(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {
        "default_backend": "swarmui",
        "enabled_backends": ["swarmui", "openrouter"],
        "swarmui": {"base_url": "http://example:9999"},
        "openrouter": {"default_model": "openai/gpt-image-1", "timeout_seconds": 42},
    }
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == "http://example:9999"
    assert cfg.openrouter_image_default_model == "openai/gpt-image-1"
    assert cfg.openrouter_image_timeout_seconds == 42
    assert cfg.enabled_backends == ["swarmui", "openrouter"]

def test_secret_precedence_env_over_config(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {"openrouter": {"api_key": "from-config"}}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.openrouter_image_api_key == "from-env"

def test_secret_from_keyring_populates_field(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    # keyring-only secret must land on the config field so listing.is_configured sees it (spec §4.2 step 5)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: "kr-secret" if backend == "novita" else None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.novita_image_api_key == "kr-secret"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_config_loader.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Copy server config.py, repoint, and replace the loader source**

```bash
cp "$SRV/config.py" tldw_chatbook/Image_Generation/config.py
sed -i '' -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' tldw_chatbook/Image_Generation/config.py
```

Now edit `tldw_chatbook/Image_Generation/config.py`:

1. **Remove** the server config import: delete the line `from tldw_Server_API.app.core.config import get_config_section` (and any other `tldw_Server_API` import — verify with `grep tldw_Server_API`).
2. **Read `get_image_generation_config()`** and note the local name it binds the section mapping to (e.g. `section = get_config_section("Image-Generation", ...)`). Replace that single assignment's RHS with `section = _load_image_generation_section()`. Keep the rest of the builder (all the `_coerce_*` / `section.get(...)` calls and the `ImageGenerationConfig(...)` construction) verbatim — the flat keys already match the dataclass field names.
3. **Add** these helpers near the top of the module (below the `DEFAULT_*` constants). `_KEYS` maps each nested `[image_generation.<backend>].<toml_key>` to the flat field name the builder reads:

```python
import os
import keyring
from loguru import logger
from tldw_chatbook.config import get_cli_setting, get_api_key

# Secret fields: backend -> (flat_field_name, [env vars in precedence order], keyring_backend_id)
_SECRETS = {
    "swarmui":     ("swarmui_swarm_token",        ["SWARMUI_TOKEN"],                       "swarmui"),
    "openrouter":  ("openrouter_image_api_key",   ["OPENROUTER_API_KEY"],                  "openrouter"),
    "novita":      ("novita_image_api_key",       ["NOVITA_API_KEY"],                      "novita"),
    "together":    ("together_image_api_key",     ["TOGETHER_API_KEY"],                    "together"),
    "modelstudio": ("modelstudio_image_api_key",  ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],   "modelstudio"),
}
# Non-secret nested keys: (backend, toml_key) -> flat_field_name
_NON_SECRET = {
    ("stable_diffusion_cpp", "binary_path"):          "sd_cpp_binary_path",
    ("stable_diffusion_cpp", "diffusion_model_path"): "sd_cpp_diffusion_model_path",
    ("stable_diffusion_cpp", "model_path"):           "sd_cpp_model_path",
    ("stable_diffusion_cpp", "vae_path"):             "sd_cpp_vae_path",
    ("stable_diffusion_cpp", "lora_paths"):           "sd_cpp_lora_paths",
    ("stable_diffusion_cpp", "device"):               "sd_cpp_device",
    ("stable_diffusion_cpp", "default_steps"):        "sd_cpp_default_steps",
    ("stable_diffusion_cpp", "default_cfg_scale"):    "sd_cpp_default_cfg_scale",
    ("stable_diffusion_cpp", "default_sampler"):      "sd_cpp_default_sampler",
    ("stable_diffusion_cpp", "timeout_seconds"):      "sd_cpp_timeout_seconds",
    ("stable_diffusion_cpp", "allowed_extra_params"): "sd_cpp_allowed_extra_params",
    ("swarmui", "base_url"):              "swarmui_base_url",
    ("swarmui", "default_model"):         "swarmui_default_model",
    ("swarmui", "timeout_seconds"):       "swarmui_timeout_seconds",
    ("swarmui", "allowed_extra_params"):  "swarmui_allowed_extra_params",
    ("openrouter", "base_url"):              "openrouter_image_base_url",
    ("openrouter", "default_model"):         "openrouter_image_default_model",
    ("openrouter", "timeout_seconds"):       "openrouter_image_timeout_seconds",
    ("openrouter", "allowed_extra_params"):  "openrouter_image_allowed_extra_params",
    ("novita", "base_url"):              "novita_image_base_url",
    ("novita", "default_model"):         "novita_image_default_model",
    ("novita", "timeout_seconds"):       "novita_image_timeout_seconds",
    ("novita", "poll_interval_seconds"): "novita_image_poll_interval_seconds",
    ("novita", "allowed_extra_params"):  "novita_image_allowed_extra_params",
    ("together", "base_url"):              "together_image_base_url",
    ("together", "default_model"):         "together_image_default_model",
    ("together", "timeout_seconds"):       "together_image_timeout_seconds",
    ("together", "allowed_extra_params"):  "together_image_allowed_extra_params",
    ("modelstudio", "base_url"):              "modelstudio_image_base_url",
    ("modelstudio", "default_model"):         "modelstudio_image_default_model",
    ("modelstudio", "region"):                "modelstudio_image_region",
    ("modelstudio", "mode"):                  "modelstudio_image_mode",
    ("modelstudio", "poll_interval_seconds"): "modelstudio_image_poll_interval_seconds",
    ("modelstudio", "timeout_seconds"):       "modelstudio_image_timeout_seconds",
    ("modelstudio", "allowed_extra_params"):  "modelstudio_image_allowed_extra_params",
}
_GLOBAL_KEYS = [
    "default_backend", "enabled_backends", "max_width", "max_height",
    "max_pixels", "max_steps", "max_prompt_length", "inline_max_bytes",
]

def _read_image_generation_toml() -> dict:
    """Return the raw [image_generation] section dict (nested). Patch point in tests."""
    from tldw_chatbook.config import load_settings
    return load_settings().get("image_generation", {}) or {}

def _keyring_get(backend: str):
    """Namespaced keyring lookup; never raises. Patch point in tests."""
    try:
        return keyring.get_password("tldw_chatbook_imagegen", backend)
    except Exception as e:  # keyring backend may be unavailable
        logger.debug(f"keyring lookup failed for imagegen/{backend}: {e}")
        return None

def _resolve_secret(backend: str, sub: dict):
    field, env_vars, kr_id = _SECRETS[backend]
    for ev in env_vars:                       # 1. env
        v = os.getenv(ev)
        if v:
            return field, v
    cfg_val = (sub or {}).get("api_key")       # 2. config
    if cfg_val and cfg_val != "<API_KEY_HERE>":
        return field, cfg_val
    kr = _keyring_get(kr_id)                    # 3. keyring
    if kr:
        return field, kr
    return field, None                         # 4. optional shared handled by adapter/get_api_key opt-in

def _load_image_generation_section() -> dict:
    """Assemble the FLAT mapping the config builder expects, from nested TOML + env + keyring."""
    raw = _read_image_generation_toml()
    flat: dict = {}
    for k in _GLOBAL_KEYS:
        if k in raw:
            flat[k] = raw[k]
    for (backend, toml_key), field in _NON_SECRET.items():
        sub = raw.get(backend) or {}
        if toml_key in sub:
            flat[field] = sub[toml_key]
    for backend in _SECRETS:
        field, value = _resolve_secret(backend, raw.get(backend) or {})
        if value:
            flat[field] = value
    return flat
```

> Note: `flat` intentionally omits keys that are unset so the builder falls back to its `DEFAULT_*` constants exactly as it did with a sparse INI section. If the builder indexes `section[...]` directly (not `.get`), keep passing through `_coerce_*` as the server does — do not change coercion.

- [ ] **Step 4: Add the `[image_generation]` default block to the app config template**

In `tldw_chatbook/config.py`, find the default config TOML template (search for an existing block like `[chat.images]` or `[embeddings]`). Add the block from spec §4.1 (`[image_generation]` + the 6 `[image_generation.<backend>]` subsections) alongside it, so the section materializes for users. Do NOT put real secrets in the template; leave `api_key = "<API_KEY_HERE>"` where a key would go, matching the `[API]` convention.

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_config_loader.py -v`
Expected: PASS (4 passed). If the builder binds the section under a different local name, wire `_load_image_generation_section()` there; the tests assert behavior, not the internal name.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Image_Generation/config.py tldw_chatbook/config.py Tests/Image_Generation/test_config_loader.py
git commit -m "feat(imagegen): re-home config loader onto TOML+env+keyring; add config template"
```

---

### Task 5: Request validation

**Files:**
- Create: `tldw_chatbook/Image_Generation/request_validation.py`
- Test: `Tests/Image_Generation/test_request_validation.py`

**Interfaces:**
- Consumes: `config.get_image_generation_config`, `config.DEFAULT_*`.
- Produces: `request_validation.validate_image_generation_request(structured: dict, *, config=None) -> list`; `request_validation.effective_inline_max_bytes(config=None) -> int`; `request_validation.allowed_extra_params_for_backend(backend, config) -> set`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_request_validation.py
import pytest

@pytest.fixture
def rv():
    from tldw_chatbook.Image_Generation import request_validation as m
    return m

def _codes(issues):
    return {i.path for i in issues}

def test_valid_request_has_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []

def test_oversize_dimensions_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "width": 9000, "height": 9000, "extra_params": {}}
    issues = rv.validate_image_generation_request(bad)
    assert any("width" in p for p in _codes(issues))

def test_negative_cfg_scale_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "cfg_scale": -1.0, "extra_params": {}}
    assert any("cfg_scale" in p for p in _codes(rv.validate_image_generation_request(bad)))

def test_extra_params_not_in_allowlist_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "extra_params": {"totally_unknown": 1}}
    issues = rv.validate_image_generation_request(bad)
    assert any("extra_params" in p for p in _codes(issues))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_request_validation.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Port + repoint**

```bash
cp "$SRV/request_validation.py" tldw_chatbook/Image_Generation/request_validation.py
sed -i '' -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' tldw_chatbook/Image_Generation/request_validation.py
grep -n "tldw_Server_API" tldw_chatbook/Image_Generation/request_validation.py   # must be empty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_request_validation.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/request_validation.py Tests/Image_Generation/test_request_validation.py
git commit -m "feat(imagegen): port request validation"
```

---

### Task 6: HTTP shim + light egress guard

**Files:**
- Create: `tldw_chatbook/Image_Generation/http_client.py`
- Test: `Tests/Image_Generation/test_http_client.py`

**Interfaces:**
- Produces: `http_client.fetch_json(method, url, *, headers=None, json=None, params=None, cookies=None, timeout=None)`; `http_client.create_client(timeout=None) -> httpx.Client`; `http_client.DEFAULT_MAX_REDIRECTS: int`; `http_client._resolve_redirect_url(base, location) -> str`; `http_client._validate_egress_or_raise(url) -> None`; `http_client.evaluate_url_policy(url, *, allowed_hosts=None) -> object` (has `.allowed: bool`, `.reason: str|None`).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_http_client.py
import pytest

@pytest.fixture
def hc():
    from tldw_chatbook.Image_Generation import http_client as m
    return m

def test_rejects_non_http_scheme(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("file:///etc/passwd")

def test_allows_local_backend_url(hc):
    # user-configured local backends must pass the light guard
    hc._validate_egress_or_raise("http://127.0.0.1:7801/API/GetNewSession")  # no raise

def test_evaluate_url_policy_allowlist(hc):
    r = hc.evaluate_url_policy("https://x.aliyuncs.com/i.png", allowed_hosts={"aliyuncs.com"})
    assert r.allowed is True
    r2 = hc.evaluate_url_policy("https://evil.example/i.png", allowed_hosts={"aliyuncs.com"})
    assert r2.allowed is False

def test_fetch_json_parses(monkeypatch, hc):
    class FakeResp:
        status_code = 200
        def json(self): return {"ok": True}
        def raise_for_status(self): pass
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return FakeResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    assert hc.fetch_json("POST", "http://127.0.0.1:7801/API/x", json={"a": 1}) == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_http_client.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `http_client.py`**

```python
"""Sync HTTP shim + light egress guard for the ported image adapters.

Provides the exact surface the server's http_client exposed to Image_Generation,
backed by httpx.Client. Full SSRF hardening is deferred to task-485; this guard
only rejects non-http(s) schemes and enforces a redirect cap, staying permissive
for user-configured (incl. local) backend base URLs.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import httpx
from loguru import logger
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

DEFAULT_MAX_REDIRECTS = int(os.getenv("HTTP_MAX_REDIRECTS", "5"))
_DEFAULT_TIMEOUT = 120.0


def _validate_egress_or_raise(url: str) -> None:
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ImageGenerationError(f"Refusing non-http(s) URL: {url!r}")
    # task-485: private/link-local/metadata range blocking for API-returned URLs goes here.


def _resolve_redirect_url(base: str, location: str) -> str:
    return urljoin(base, location)


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None


def evaluate_url_policy(url: str, *, allowed_hosts: set[str] | None = None) -> URLPolicyResult:
    _validate_egress_or_raise(url)
    if not allowed_hosts:
        return URLPolicyResult(True, None)
    host = (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in allowed_hosts):
        return URLPolicyResult(True, None)
    return URLPolicyResult(False, f"host {host!r} not in allowlist")


def create_client(timeout: float | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=timeout or _DEFAULT_TIMEOUT,
        follow_redirects=True,
        max_redirects=DEFAULT_MAX_REDIRECTS,
    )


def fetch_json(method, url, *, headers=None, json=None, params=None, cookies=None, timeout=None):
    _validate_egress_or_raise(url)
    with create_client(timeout=timeout) as client:
        resp = client.request(method, url, headers=headers, json=json, params=params, cookies=cookies)
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_http_client.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/http_client.py Tests/Image_Generation/test_http_client.py
git commit -m "feat(imagegen): sync http shim + light egress guard (task-485 placeholder)"
```

---

### Task 7: Image format utils (Pillow bytes layer)

**Files:**
- Create: `tldw_chatbook/Image_Generation/adapters/image_format_utils.py`
- Test: `Tests/Image_Generation/test_image_format_utils.py`

**Interfaces:**
- Consumes: `http_client.{create_client, _validate_egress_or_raise, _resolve_redirect_url, DEFAULT_MAX_REDIRECTS}`.
- Produces: `image_format_utils.{format_from_bytes, content_type_for_format, validate_and_convert_image_output, decode_base64_image, decode_data_url, fetch_image_bytes, reference_image_data_url}`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_image_format_utils.py
import io, pytest
from PIL import Image

def _png_bytes(size=(8, 8)):
    buf = io.BytesIO(); Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG"); return buf.getvalue()

@pytest.fixture
def ifu():
    from tldw_chatbook.Image_Generation.adapters import image_format_utils as m
    return m

def test_format_from_bytes_detects_png(ifu):
    assert ifu.format_from_bytes(_png_bytes()) == "png"

def test_validate_and_convert_output_roundtrip(ifu):
    data, ctype = ifu.validate_and_convert_image_output(_png_bytes(), requested_format="png", max_bytes=10_000_000)
    assert ctype == "image/png" and isinstance(data, (bytes, bytearray))

def test_validate_rejects_when_over_max_bytes(ifu):
    with pytest.raises(Exception):
        ifu.validate_and_convert_image_output(_png_bytes((256, 256)), requested_format="png", max_bytes=10)
```

> Read the ported signature of `validate_and_convert_image_output` first and adjust the keyword names in the test to match it exactly (it may take positional `format`/`max_bytes`). Keep the three assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_image_format_utils.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Port + repoint**

```bash
cp "$SRV/adapters/image_format_utils.py" tldw_chatbook/Image_Generation/adapters/image_format_utils.py
sed -i '' \
  -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' \
  -e 's#tldw_Server_API\.app\.core\.http_client#tldw_chatbook.Image_Generation.http_client#g' \
  tldw_chatbook/Image_Generation/adapters/image_format_utils.py
grep -n "tldw_Server_API" tldw_chatbook/Image_Generation/adapters/image_format_utils.py   # must be empty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_image_format_utils.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/adapters/image_format_utils.py Tests/Image_Generation/test_image_format_utils.py
git commit -m "feat(imagegen): port image format utils (re-homed fetch_image_bytes)"
```

---

### Task 8: Adapter registry

**Files:**
- Create: `tldw_chatbook/Image_Generation/adapter_registry.py`
- Test: `Tests/Image_Generation/test_adapter_registry.py`

**Interfaces:**
- Consumes: `config.get_image_generation_config`.
- Produces: `adapter_registry.ImageAdapterRegistry`, `adapter_registry.get_registry()`, `adapter_registry.reset_registry()`, and `DEFAULT_ADAPTERS` mapping name→"tldw_chatbook.Image_Generation.adapters.<mod>.<Class>".

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_adapter_registry.py
import pytest

@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()

def test_resolve_backend_requires_enabled():
    from tldw_chatbook.Image_Generation.adapter_registry import ImageAdapterRegistry
    reg = ImageAdapterRegistry(config_override={"enabled_backends": ["swarmui"], "default_backend": "swarmui"})
    assert reg.resolve_backend("swarmui") == "swarmui"
    assert reg.resolve_backend("novita") is None      # not enabled
    assert reg.resolve_backend(None) == "swarmui"     # default

def test_nothing_enabled_by_default():
    from tldw_chatbook.Image_Generation.adapter_registry import ImageAdapterRegistry
    reg = ImageAdapterRegistry(config_override={"enabled_backends": [], "default_backend": "swarmui"})
    assert reg.resolve_backend("swarmui") is None

def test_default_adapters_point_at_local_package():
    from tldw_chatbook.Image_Generation.adapter_registry import DEFAULT_ADAPTERS
    assert set(DEFAULT_ADAPTERS) == {
        "stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"
    }
    assert all(v.startswith("tldw_chatbook.Image_Generation.adapters.") for v in DEFAULT_ADAPTERS.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_adapter_registry.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Port + repoint the DEFAULT_ADAPTERS strings**

```bash
cp "$SRV/adapter_registry.py" tldw_chatbook/Image_Generation/adapter_registry.py
sed -i '' -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' tldw_chatbook/Image_Generation/adapter_registry.py
grep -n "tldw_Server_API" tldw_chatbook/Image_Generation/adapter_registry.py   # must be empty
```

The `sed` also rewrites the fully-qualified `DEFAULT_ADAPTERS` values (they contain `...Image_Generation.adapters...`). Confirm they now read `tldw_chatbook.Image_Generation.adapters.<mod>.<Class>`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/test_adapter_registry.py -v`
Expected: PASS (3 passed). (Adapters are lazy-imported only on `get_adapter`, so these tests don't require the adapter modules to exist yet.)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Image_Generation/adapter_registry.py Tests/Image_Generation/test_adapter_registry.py
git commit -m "feat(imagegen): port adapter registry (lazy, local module paths)"
```

---

### Tasks 9–14: Port the six adapters

Each adapter is the same shape: **copy → repoint → focused mocked test → commit.** Do them one at a time (each is its own reviewable task). For every adapter, after copying run the standard repoint recipe and `grep -n "tldw_Server_API" <file>` (must be empty).

#### Task 9: SwarmUI adapter

**Files:** Create `tldw_chatbook/Image_Generation/adapters/swarmui_adapter.py`; Test `Tests/Image_Generation/test_swarmui_adapter.py`.
**Interfaces:** Produces `SwarmUIAdapter` (`name="swarmui"`, `supported_formats={"png","jpg"}`, sync `generate(ImageGenRequest)->ImageGenResult`).

- [ ] **Step 1: Failing test**

```python
# Tests/Image_Generation/test_swarmui_adapter.py
import io, pytest
from PIL import Image

def _png_b64():
    import base64; buf = io.BytesIO(); Image.new("RGB", (8, 8), (10, 10, 200)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def test_swarmui_generate_happy_path(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    def fake_fetch_json(method, url, **kw):
        calls.append(url)
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess-1"}
        return {"images": [{"image": _png_b64()}]}
    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    res = m.SwarmUIAdapter().generate(req)
    assert res.content_type.startswith("image/") and res.bytes_len > 0
    assert any("GetNewSession" in c for c in calls)
```

> The ported adapter imports `fetch_json` at module level (`from tldw_chatbook.Image_Generation.http_client import fetch_json`), so `monkeypatch.setattr(m, "fetch_json", ...)` intercepts it. If it instead calls `http_client.fetch_json`, patch that attribute path instead.

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Copy + repoint**

```bash
F=tldw_chatbook/Image_Generation/adapters/swarmui_adapter.py
cp "$SRV/adapters/swarmui_adapter.py" "$F"
sed -i '' \
  -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' \
  -e 's#tldw_Server_API\.app\.core\.http_client#tldw_chatbook.Image_Generation.http_client#g' \
  -e 's#tldw_Server_API\.app\.core\.Security\.egress#tldw_chatbook.Image_Generation.http_client#g' "$F"
grep -n "tldw_Server_API" "$F"   # empty
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(imagegen): port swarmui adapter`.

#### Task 10: Stable-Diffusion.cpp adapter

**Files:** Create `.../stable_diffusion_cpp_adapter.py`; Test `test_sd_cpp_adapter.py`.
**Interfaces:** Produces `StableDiffusionCppAdapter` (`name="stable_diffusion_cpp"`, subprocess-driven).

- [ ] **Step 1: Failing test** — mock `subprocess.run` and the config paths.

```python
# Tests/Image_Generation/test_sd_cpp_adapter.py
import io, pytest
from PIL import Image

def test_sd_cpp_missing_binary_raises(monkeypatch, tmp_path):
    from tldw_chatbook.Image_Generation.adapters import stable_diffusion_cpp_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError
    # config with no binary path -> unavailable
    req = ImageGenRequest(backend="stable_diffusion_cpp", prompt="cat", negative_prompt=None, width=512,
                          height=512, steps=10, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    with pytest.raises((ImageBackendUnavailableError, Exception)):
        m.StableDiffusionCppAdapter().generate(req)
```

> This asserts the graceful-unavailable path (no local binary in CI). A full happy-path test that mocks `subprocess.run` to drop a PNG into the temp dir is a nice-to-have; add it only if you can inject the output path the adapter expects. Read the ported `_build_command`/`generate` first.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Copy + repoint** (standard recipe, file `stable_diffusion_cpp_adapter.py`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(imagegen): port stable-diffusion.cpp adapter`.

#### Task 11: OpenRouter adapter

**Files:** Create `.../openrouter_image_adapter.py`; Test `test_openrouter_adapter.py`.
**Interfaces:** Produces `OpenRouterImageAdapter` (`name="openrouter"`, chat-completions shape).

- [ ] **Step 1: Failing test** — mock `fetch_json` to return a chat-completions payload embedding a base64 image; set `OPENROUTER_API_KEY`.

```python
# Tests/Image_Generation/test_openrouter_adapter.py
import io, base64, pytest
from PIL import Image

def _b64():
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (0, 180, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_openrouter_extracts_image(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(m, "fetch_json", lambda method, url, **kw: {
        "choices": [{"message": {"images": [{"image_url": {"url": "data:image/png;base64," + _b64()}}]}}]
    })
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="openai/gpt-image-1",
                          format="png", extra_params={})
    res = m.OpenRouterImageAdapter().generate(req)
    assert res.bytes_len > 0
```

> The exact JSON path the walker extracts from may differ; read the ported `_extract_from_node`/`_extract_from_link_value` and shape the fake payload to a form it accepts (a `data:` URL anywhere in the tree is the safest).

- [ ] **Step 2: Run → FAIL.** **Step 3: Copy + repoint.** **Step 4: Run → PASS.** **Step 5: Commit** `feat(imagegen): port openrouter adapter`.

#### Task 12: Together adapter

**Files:** Create `.../together_image_adapter.py`; Test `test_together_adapter.py`.
**Interfaces:** Produces `TogetherImageAdapter` (`name="together"`, `/v1/images/generations`, single call).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_together_adapter.py
import io, base64, pytest
from PIL import Image

def _b64():
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (180, 120, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_together_extracts_image_and_no_v1_doubling(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import together_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    seen = {}
    def fake_fetch_json(method, url, **kw):
        seen["url"] = url
        return {"data": [{"b64_json": _b64()}]}
    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="together", prompt="owl", negative_prompt=None, width=512, height=512,
                          steps=None, cfg_scale=None, seed=None, sampler=None,
                          model="black-forest-labs/FLUX.1-schnell-Free", format="png", extra_params={})
    res = m.TogetherImageAdapter().generate(req)
    assert res.bytes_len > 0
    # default base_url ends in /v1; the adapter must NOT produce /v1/v1/ (spec verification)
    assert "/v1/v1/" not in seen["url"]
```

> If the ported extractor wants a `url`/`data:` field instead of `b64_json`, shape the fake payload to what `_extract_from_node` accepts (read it first). Keep both assertions.

- [ ] **Step 2: Run → FAIL.** **Step 3: Copy + repoint** (standard recipe, `together_image_adapter.py`). **Step 4: Run → PASS.** **Step 5: Commit** `feat(imagegen): port together adapter`.

#### Task 13: Novita adapter

**Files:** Create `.../novita_image_adapter.py`; Test `test_novita_adapter.py`.
**Interfaces:** Produces `NovitaImageAdapter` (`name="novita"`, async submit + poll).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_novita_adapter.py
import io, base64, pytest
from PIL import Image

def _b64():
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (0, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_novita_submit_then_poll(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import novita_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    monkeypatch.setenv("NOVITA_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)  # skip poll delay
    step = {"n": 0}
    def fake_fetch_json(method, url, **kw):
        if "async/txt2img" in url or method.upper() == "POST":
            return {"task_id": "t1"}
        step["n"] += 1
        return {"task": {"status": "TASK_STATUS_SUCCEED"},
                "images": [{"image_url": "data:image/png;base64," + _b64()}]}
    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="novita", prompt="whale", negative_prompt=None, width=512, height=512,
                          steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None, format="png", extra_params={})
    res = m.NovitaImageAdapter().generate(req)
    assert res.bytes_len > 0
```

> Read the ported adapter's exact submit-URL match, success-state strings, and result JSON path; align the fake's branch condition and status value to what the poll loop treats as terminal-success. Keep the final assertion.

- [ ] **Step 2: Run → FAIL.** **Step 3: Copy + repoint** (standard recipe, `novita_image_adapter.py`). **Step 4: Run → PASS.** **Step 5: Commit** `feat(imagegen): port novita adapter`.

#### Task 14: ModelStudio adapter

**Files:** Create `.../modelstudio_image_adapter.py`; Test `test_modelstudio_adapter.py`.
**Interfaces:** Produces `ModelStudioImageAdapter` (`name="modelstudio"`, sync/async/auto modes).

> **Egress note:** the ported adapter calls `evaluate_url_policy(url)` (no `allowed_hosts`), so under the Phase-1 stub it is permissive — ModelStudio's `aliyuncs` host allowlist is intentionally NOT enforced yet; that enforcement is part of **task-485**. Do not add allowlist logic here.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_modelstudio_adapter.py
import io, base64, pytest
from PIL import Image

def _b64():
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (120, 0, 160)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_modelstudio_sync_no_reference_image(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import modelstudio_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    # reference_image=None must never call reference_image_data_url
    monkeypatch.setattr(m, "reference_image_data_url",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(m, "fetch_json", lambda method, url, **kw: {
        "output": {"choices": [{"message": {"content": [{"image": "data:image/png;base64," + _b64()}]}}]}
    })
    req = ImageGenRequest(backend="modelstudio", prompt="lotus", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="qwen-image",
                          format="png", extra_params={"mode": "sync"}, reference_image=None)
    res = m.ModelStudioImageAdapter().generate(req)
    assert res.bytes_len > 0
```

> Shape the fake `fetch_json` payload to whatever the ported sync-mode extractor reads (read `_build_sync_payload`/response handling first). The key assertions: an image comes back, and `reference_image_data_url` is never invoked when `reference_image is None`.

- [ ] **Step 2: Run → FAIL.** **Step 3: Copy + repoint** (standard recipe, `modelstudio_image_adapter.py`). **Step 4: Run → PASS.** **Step 5: Commit** `feat(imagegen): port modelstudio adapter`.

---

### Task 15: Model listing / is_configured

**Files:**
- Create: `tldw_chatbook/Image_Generation/listing.py`
- Test: `Tests/Image_Generation/test_listing.py`

**Interfaces:**
- Consumes: `adapter_registry`, `config`, `capabilities`.
- Produces: `listing.list_image_models_for_catalog() -> list[dict]` (each dict has `id`, `name`, `type="image"`, `is_configured: bool`, `capabilities`, `modalities`).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_listing.py
import pytest

@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import config as c, adapter_registry as r
    c.reset_image_generation_config_cache(); r.reset_registry()
    yield
    c.reset_image_generation_config_cache(); r.reset_registry()

def test_keyring_populated_backend_reports_configured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    # enable openrouter; provide its key only via keyring (spec §4.2 step 5 -> is_configured must be True)
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"enabled_backends": ["openrouter"], "default_backend": "openrouter"}, raising=False)
    for var in ("OPENROUTER_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: "kr" if b == "openrouter" else None, raising=False)
    c.get_image_generation_config(reload=True)
    entries = {e["name"]: e for e in L.list_image_models_for_catalog()}
    assert entries["openrouter"]["is_configured"] is True

def test_disabled_backends_excluded(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c, listing as L
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"enabled_backends": ["swarmui"], "default_backend": "swarmui"}, raising=False)
    c.get_image_generation_config(reload=True)
    names = {e["name"] for e in L.list_image_models_for_catalog()}
    assert "novita" not in names and "swarmui" in names
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Port + repoint**

```bash
cp "$SRV/listing.py" tldw_chatbook/Image_Generation/listing.py
sed -i '' -e 's#tldw_Server_API\.app\.core\.Image_Generation#tldw_chatbook.Image_Generation#g' tldw_chatbook/Image_Generation/listing.py
grep -n "tldw_Server_API" tldw_chatbook/Image_Generation/listing.py   # empty
```

- [ ] **Step 4: Run → PASS** (2 passed).
- [ ] **Step 5: Commit** `feat(imagegen): port model listing / is_configured`.

---

### Task 16: Generation worker (request builder + thread entry)

**Files:**
- Create: `tldw_chatbook/Image_Generation/worker.py`
- Test: `Tests/Image_Generation/test_worker.py`

**Interfaces:**
- Consumes: `adapter_registry.get_registry`, `adapters.base.ImageGenRequest`, `exceptions`.
- Produces:
  - `worker.build_request(*, backend, prompt, negative_prompt=None, width=None, height=None, steps=None, cfg_scale=None, seed=None, sampler=None, model=None, image_format="png", extra_params=None) -> ImageGenRequest`
  - `worker.run_generation(request: ImageGenRequest) -> ImageGenResult` (resolves backend via registry, calls adapter `.generate`; **blocking — call only from a thread**).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_worker.py
import pytest

def test_build_request_defaults_format_png():
    from tldw_chatbook.Image_Generation.worker import build_request
    req = build_request(backend="swarmui", prompt="cat")
    assert req.format == "png"
    assert req.extra_params == {}          # never None
    assert req.negative_prompt is None

def test_run_generation_unknown_backend_raises(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    req = worker.build_request(backend="nope", prompt="cat")
    with pytest.raises(ImageGenerationError):
        worker.run_generation(req)   # registry resolve_backend -> None -> error

def test_run_generation_dispatches_to_adapter(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    class FakeAdapter:
        name = "swarmui"; supported_formats = {"png"}
        def generate(self, req): return ImageGenResult(content=b"x", content_type="image/png", bytes_len=1)
    class FakeReg:
        def resolve_backend(self, name): return "swarmui" if name == "swarmui" else None
        def get_adapter(self, name): return FakeAdapter()
    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    res = worker.run_generation(worker.build_request(backend="swarmui", prompt="cat"))
    assert res.bytes_len == 1
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Write `worker.py`**

```python
"""Request builder + blocking generation entry. The Textual demo screen (and, in
Phase 2, the chat card) call run_generation() from a thread worker — never on the
UI loop, because the adapters are synchronous and blocking.
"""
from __future__ import annotations
from tldw_chatbook.Image_Generation.adapter_registry import get_registry
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError


def build_request(*, backend, prompt, negative_prompt=None, width=None, height=None,
                  steps=None, cfg_scale=None, seed=None, sampler=None, model=None,
                  image_format="png", extra_params=None) -> ImageGenRequest:
    return ImageGenRequest(
        backend=backend, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, cfg_scale=cfg_scale, seed=seed,
        sampler=sampler, model=model, format=image_format,
        extra_params=dict(extra_params or {}),
    )


def run_generation(request: ImageGenRequest) -> ImageGenResult:
    """Blocking. Resolve the backend and invoke its adapter. Raises ImageGenerationError."""
    registry = get_registry()
    resolved = registry.resolve_backend(request.backend)
    if resolved is None:
        raise ImageGenerationError(
            f"Backend {request.backend!r} is not enabled/available. "
            f"Check [image_generation].enabled_backends."
        )
    adapter = registry.get_adapter(resolved)
    if adapter is None:
        raise ImageGenerationError(f"Adapter for backend {resolved!r} failed to load.")
    return adapter.generate(request)
```

- [ ] **Step 4: Run → PASS** (3 passed).
- [ ] **Step 5: Commit** `feat(imagegen): generation worker (request builder + blocking entry)`.

---

### Task 17: Throwaway demo panel + command-palette entry

**Files:**
- Create: `tldw_chatbook/UI/Screens/image_gen_demo_screen.py`
- Create: `tldw_chatbook/UI/image_gen_command_provider.py`
- Modify: `tldw_chatbook/app.py` (add the provider to `COMMANDS`)
- Test: `Tests/Image_Generation/test_demo_screen.py`

**Interfaces:**
- Consumes: `worker.build_request`, `worker.run_generation`, `listing.list_image_models_for_catalog`, `BaseAppScreen`.
- Produces: `ImageGenDemoScreen(BaseAppScreen)`; `ImageGenCommandProvider(Provider)` yielding a "Image Gen (dev)" command that `push_screen`s it.

- [ ] **Step 1: Write the failing test** (Textual async harness)

```python
# Tests/Image_Generation/test_demo_screen.py
import pytest

@pytest.mark.asyncio
async def test_demo_screen_lists_backends_and_generates(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    from tldw_chatbook.UI.Screens.image_gen_demo_screen import ImageGenDemoScreen
    # stub listing + generation so the test needs no backend
    import tldw_chatbook.UI.Screens.image_gen_demo_screen as scr
    monkeypatch.setattr(scr, "list_image_models_for_catalog",
                        lambda: [{"name": "swarmui", "is_configured": True}])
    monkeypatch.setattr(scr, "run_generation",
                        lambda req: ImageGenResult(content=b"x", content_type="image/png", bytes_len=1))
    # A minimal host app that pushes the screen; ImageGenDemoScreen takes the app instance.
    from textual.app import App
    class Host(App):
        def on_mount(self): self.push_screen(ImageGenDemoScreen(self))
    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        # backend select is populated
        assert app.screen.query("#imagegen-backend")
```

> If `BaseAppScreen` requires the real `TldwCli` app rather than a bare `App`, use the project's standard screen-test harness (see an existing `Tests/.../test_*_screen.py` that drives `app.run_test()`), and keep the two assertions: backend select present, and a generate action calls `run_generation`.

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Write the demo screen**

```python
# tldw_chatbook/UI/Screens/image_gen_demo_screen.py
"""THROWAWAY dev panel (Phase 1). Replaced by the real chat card in Phase 2.
Renders one generated image via the low-level rich-pixels/textual-image primitives
(NOT the Console transcript path). Persists nothing.
"""
from __future__ import annotations
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Select, TextArea, Input, Button, Static, Label
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Image_Generation.worker import build_request, run_generation
from tldw_chatbook.Image_Generation.listing import list_image_models_for_catalog


class ImageGenDemoScreen(BaseAppScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="imagegen-demo"):
            yield Label("Image Gen (dev) — throwaway Phase-1 panel")
            opts = [(f'{e["name"]}{"" if e.get("is_configured") else "  (not configured)"}', e["name"])
                    for e in list_image_models_for_catalog()]
            yield Select(opts or [("(no backends enabled)", "")], id="imagegen-backend")
            yield TextArea(id="imagegen-prompt")
            yield TextArea(id="imagegen-negative")
            yield Input(placeholder="seed (blank = -1)", id="imagegen-seed")
            yield Button("Generate", id="imagegen-generate", variant="primary")
            yield Static(id="imagegen-status")
            yield Static(id="imagegen-image")
            yield Static(id="imagegen-meta")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "imagegen-generate":
            self._generate()

    @work(thread=True, exclusive=True, group="imagegen-demo")
    def _generate(self) -> None:
        backend = self.query_one("#imagegen-backend", Select).value
        prompt = self.query_one("#imagegen-prompt", TextArea).text
        negative = self.query_one("#imagegen-negative", TextArea).text or None
        seed_raw = self.query_one("#imagegen-seed", Input).value.strip()
        seed = int(seed_raw) if seed_raw.lstrip("-").isdigit() else -1
        self.app.call_from_thread(self.query_one("#imagegen-status", Static).update, "Generating…")
        try:
            req = build_request(backend=backend, prompt=prompt, negative_prompt=negative,
                                seed=seed, image_format="png")
            res = run_generation(req)               # blocking; we are in a thread
        except Exception as e:  # surface the error clearly (incl. inline_max_bytes cap)
            self.app.call_from_thread(self.query_one("#imagegen-status", Static).update, f"Error: {e}")
            return
        self.app.call_from_thread(self._render_result, res, req)

    def _render_result(self, res, req) -> None:
        # low-level render (rich-pixels), decoupled from the transcript path
        from rich_pixels import Pixels
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(res.content))
        img.thumbnail((80, 40))
        self.query_one("#imagegen-status", Static).update("Done")
        self.query_one("#imagegen-image", Static).update(Pixels.from_image(img))
        self.query_one("#imagegen-meta", Static).update(
            f"backend={req.backend} format={res.content_type} bytes={res.bytes_len} "
            f"seed={req.seed} prompt={req.prompt!r}"
        )
```

- [ ] **Step 4: Write the command provider + register it**

```python
# tldw_chatbook/UI/image_gen_command_provider.py
"""Command-palette entry for the throwaway Image Gen (dev) panel (Phase 1)."""
from textual.command import Provider, Hit, Hits
from tldw_chatbook.UI.Screens.image_gen_demo_screen import ImageGenDemoScreen


class ImageGenCommandProvider(Provider):
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        label = "Image Gen (dev)"
        score = matcher.match(label)
        if score > 0:
            yield Hit(score, matcher.highlight(label),
                      lambda: self.app.push_screen(ImageGenDemoScreen(self.app)),
                      help="Open the throwaway image-generation demo panel")
```

In `tldw_chatbook/app.py`, add the provider to the app's `COMMANDS` set (search for the existing `COMMANDS = {...}` or `get_system_commands`; follow the pattern used by `tldw_chatbook/UI/console_command_provider.py`). Example:

```python
from tldw_chatbook.UI.image_gen_command_provider import ImageGenCommandProvider
# ... in the TldwCli class body:
COMMANDS = App.COMMANDS | {ImageGenCommandProvider}
```

- [ ] **Step 5: Run → PASS** (adjust the harness per the note if `BaseAppScreen` needs the real app).
- [ ] **Step 6: Manual verification (the real proof surface)**

```bash
source .venv/bin/activate && python3 -m tldw_chatbook.app
```
Open the command palette → "Image Gen (dev)" → pick `swarmui` (with a local SwarmUI running) → type a prompt → Generate → an image renders with metadata below it. (With no backend configured, the status shows a clear error — that's the expected unconfigured path.)

- [ ] **Step 7: Commit** `feat(imagegen): throwaway demo panel + command-palette entry`.

---

### Task 18: Cold-start import guard + public surface finalize

**Files:**
- Modify: `tldw_chatbook/Image_Generation/__init__.py` (add lazy accessors, keep import-light)
- Test: `Tests/Image_Generation/test_cold_start.py`

**Interfaces:**
- Produces: `Image_Generation.get_image_generation_config` and `Image_Generation.get_registry` re-exported *lazily* (via `__getattr__`) so importing the package still pulls no adapters/Pillow.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Image_Generation/test_cold_start.py
import sys, importlib

def test_importing_package_pulls_no_adapters_or_pillow():
    # drop anything already imported so the assertion is meaningful
    for name in list(sys.modules):
        if name.startswith("tldw_chatbook.Image_Generation") or name == "PIL":
            del sys.modules[name]
    importlib.import_module("tldw_chatbook.Image_Generation")
    loaded = set(sys.modules)
    assert not any(n.startswith("tldw_chatbook.Image_Generation.adapters.") and n.endswith("_adapter")
                   for n in loaded), "adapters must be lazy"
    assert "tldw_chatbook.Image_Generation.adapters.image_format_utils" not in loaded
    assert "PIL" not in loaded, "Pillow must not import at package import time"

def test_lazy_accessors_available():
    import tldw_chatbook.Image_Generation as ig
    assert callable(ig.get_image_generation_config)
    assert callable(ig.get_registry)
```

- [ ] **Step 2: Run → FAIL** (either Pillow gets imported, or accessors missing).
- [ ] **Step 3: Make `__init__.py` lazy**

```python
"""Multi-provider image generation (ported from tldw_server). Import-light."""
from tldw_chatbook.Image_Generation.exceptions import (
    ImageGenerationError, ImageBackendUnavailableError,
)

__all__ = [
    "ImageGenerationError", "ImageBackendUnavailableError",
    "get_image_generation_config", "get_registry",
]

def __getattr__(name):  # PEP 562 lazy re-export; keeps adapters/Pillow out of import time
    if name == "get_image_generation_config":
        from tldw_chatbook.Image_Generation.config import get_image_generation_config as f
        return f
    if name == "get_registry":
        from tldw_chatbook.Image_Generation.adapter_registry import get_registry as f
        return f
    raise AttributeError(name)
```

> If Task 4's `config.py` imports anything Pillow-touching at module level, move that import inside the function that needs it — `config` must stay Pillow-free so the guard holds.

- [ ] **Step 4: Run → PASS** (2 passed).
- [ ] **Step 5: Run the whole suite**

Run: `source .venv/bin/activate && pytest Tests/Image_Generation/ -v`
Expected: all green.

- [ ] **Step 6: Commit** `feat(imagegen): lazy package surface + cold-start guard`.

---

## Wrap-up (after Task 18)

- [ ] Run the full package suite once more: `source .venv/bin/activate && pytest Tests/Image_Generation/ -q`.
- [ ] `grep -rn "tldw_Server_API" tldw_chatbook/Image_Generation/` → must be empty (no server imports leaked).
- [ ] Confirm the app still boots: `python3 -m tldw_chatbook.app` (splash → app, no import errors).
- [ ] The design spec's later phases (chat card, `/generate-image`, character canvas, variants/TTS), the `Media_Creation` cleanup, and task-485 (egress hardening) are explicitly out of scope here.

**Optional — opt-in live integration tests (spec §8):** the mocked tests above are the automated proof; the *live* proof is Task 17 Step 6 (real backend, real image). If you want an automated live test per backend, add one parametrized test marked `@pytest.mark.optional` that reads creds from env (`OPENROUTER_API_KEY`, a running SwarmUI at `swarmui_base_url`, an `sd` binary path, …) and `pytest.skip(...)` when absent, calls `worker.run_generation(worker.build_request(backend=..., prompt="a red apple on a table"))`, and asserts `res.bytes_len > 0`. Keep these skipped by default so CI stays green without credentials.
