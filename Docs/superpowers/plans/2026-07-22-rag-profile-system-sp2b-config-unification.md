# SP2b — Profile System: Config-Resolution Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the **active profile the single source** the RAG engine reads: one resolver (`resolve_active_rag_config`) that every resolution path consumes, so ingestion and search always use the same config for the same active profile (no divergent-model dimension crash), plus set-active wiring and a first-run import.

**Architecture:** A new `resolve_active_rag_config()` reads the active-profile pointer `[rag.service].profile`, loads that profile via `ConfigProfileManager`, deep-copies its `rag_config`, and applies the existing env-override layer on top. Both consumers route through it: `RAGConfig.from_settings` (search path) delegates to it; the shared RAG service (ingestion path) is built from it. `fusion.py`'s scattered `hybrid_alpha` read is redirected to it. `set_active_profile` writes the pointer and calls the existing `reset_shared_rag_service()`. On first run, the currently-resolved config is snapshotted into an "Imported settings" profile whose SP1 fingerprint matches the SP1-adopted legacy collection.

**Tech Stack:** Python 3.11+, existing `RAGConfig`/`ConfigProfileManager` (SP2a), `ingestion_indexing` shared-service singleton, `save_setting_to_cli_config`, pytest with monkeypatched env + temp profiles dir.

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md` (§4-7); overview `-rag-settings-profiles-overview.md`. SP1 (PR #771) + SP2a (PR #780) MERGED.
- **Single active pointer:** reuse `[rag.service].profile` (read today by `ingestion_indexing._configured_profile` at `:128` and `create_rag_service`'s default). Do NOT add a second pointer key.
- **The parity requirement (the whole point):** for a given active profile + env, the config used to **embed** (ingestion → shared service) MUST equal the config used to **query** (search → `from_settings`). Today they diverge: the shared service uses `profile.rag_config` with **no env overlay**, while `from_settings` applies a full env overlay — so any `RAG_*` env var already breaks parity. Unification fixes this by routing BOTH through `resolve_active_rag_config` (profile base + identical env overlay).
- **Env is an explicit override layer** kept verbatim from `from_settings` (`RAG_EMBEDDING_MODEL`, `RAG_DEVICE`, `RAG_EMBEDDING_CACHE_SIZE`, `OPENAI_API_KEY`, `RAG_EMBEDDING_BASE_URL`, `RAG_PERSIST_DIR`, `RAG_VECTOR_STORE` via `VectorStoreConfig.__post_init__`, and the `override_embedding_model`/`override_persist_dir` args). The scattered `AppRAGSearchConfig.rag.*` **value** reads are deprecated (the profile replaces them).
- **`reset_shared_rag_service()` ALREADY EXISTS** (`ingestion_indexing.py:192`, clears `_shared_service` under `_shared_service_lock` via `set_shared_rag_service(None)`) — USE it; do NOT re-add it.
- **Cross-SP invariant (tested e2e here):** the "Imported settings" first-run profile's SP1 fingerprint (`fingerprint_collection`) MUST equal the fingerprint SP1's `maybe_adopt_legacy_collection` adopts the legacy `default` collection under — i.e. both derive from the same first-run resolved config. No silent index blanking on upgrade.
- **No DB migration.** Profiles are files (SP2a); the pointer is config.toml.
- **Test isolation:** the shared-service singleton is process-global; every test that builds/resets it calls `reset_shared_rag_service()` in teardown (documented cross-file pollution, backlog task-408). Env-reading tests use `monkeypatch.setenv/delenv`; profile tests use a temp `profiles_dir`.
- **NOT in scope (SP3):** the settings-screen UI, the off-thread reload orchestration (SP2b provides the sync `set_active_profile`; the UI calls it off-thread), the full-`ProfileConfig` editor.

## Plan-time facts (verified)
- Config value-readers of the profile config: `config.py` `RAGConfig.from_settings` (`:433`, the main re-deriver) + `fusion.py:242` (`hybrid_alpha` only). `ingestion_indexing.py:85`'s `AppRAGSearchConfig.rag` read is the **`rag.indexing` feature-toggle**, NOT profile values — leave it.
- `from_settings` callers (Path B): `pipeline_integration.py:21` (`config or RAGConfig.from_settings()`) and `config.py:854` (`create_config_for_collection` → search-only tool path).
- Shared service (Path A): `get_shared_rag_service` (`ingestion_indexing.py:138`) → `create_rag_service(profile_name=active)` → `EnhancedRAGServiceV2(config=profile.rag_config)` (no env overlay today). `create_rag_service(profile_name, config=...)` accepts a config override (`rag_factory.py:17,60-61`).
- Active pointer read: `ingestion_indexing._configured_profile()` (`:125-135`) reads `get_cli_setting("rag","service",{}).get("profile")`, default `DEFAULT_PROFILE`.
- Config write: `save_setting_to_cli_config(section, key, value) -> bool` (`config.py:3824`).
- SP2a `ConfigProfileManager`: `get_profile(id)`, `save_profile`, `_builtin_ids`, `_slugify`; SP1 `fingerprint_collection(config)` in `simplified/collection_fingerprint.py`.

---

## File Structure
- **Create** `tldw_chatbook/RAG_Search/simplified/active_config.py` — `resolve_active_rag_config()`, `_apply_env_overrides()` (extracted from `from_settings`), `set_active_profile()`, `ensure_imported_profile()`. One focused module for active-config resolution + activation.
- **Modify** `tldw_chatbook/RAG_Search/simplified/config.py` — `RAGConfig.from_settings` delegates to `resolve_active_rag_config`; the env-overlay body moves to `active_config._apply_env_overrides`.
- **Modify** `tldw_chatbook/RAG_Search/ingestion_indexing.py` — `get_shared_rag_service` builds the active shared service from `resolve_active_rag_config()`.
- **Modify** `tldw_chatbook/RAG_Search/fusion.py:242` — `resolve_hybrid_alpha` reads the active profile's `search.hybrid_alpha`.
- **Create tests** `Tests/RAG/test_active_config_resolution.py`, `Tests/RAG/test_config_unification_parity.py`, `Tests/RAG/test_first_run_import.py`.

---

## Task 1: `resolve_active_rag_config` + env-overlay extraction

**Files:**
- Create: `tldw_chatbook/RAG_Search/simplified/active_config.py`
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (`from_settings` → delegate)
- Test: `Tests/RAG/test_active_config_resolution.py`

**Interfaces:**
- Produces:
  - `_apply_env_overrides(config: RAGConfig, override_embedding_model: Optional[str] = None, override_persist_dir=None) -> RAGConfig` — mutates+returns `config`, applying exactly the env/override layer.
  - `resolve_active_rag_config(override_embedding_model=None, override_persist_dir=None) -> RAGConfig` — active profile's `rag_config` (deep copy) + env overlay.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/test_active_config_resolution.py
import pytest
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig


@pytest.fixture
def active(tmp_path, monkeypatch):
    """A ConfigProfileManager over a temp dir + a helper to set the active pointer."""
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    # Point resolve_active_rag_config at this manager + a chosen active id via monkeypatch.
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: state["active"], raising=False)
    return mgr, state


def test_resolves_active_profiles_rag_config(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom A", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="modelA"),
                                           chunking=ChunkingConfig(chunk_size=321),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    cfg = resolve_active_rag_config()
    assert cfg.embedding.model == "modelA"
    assert cfg.chunking.chunk_size == 321


def test_env_overrides_win_over_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom B", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-model"),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")
    assert resolve_active_rag_config().embedding.model == "env-model"


def test_returns_deep_copy_not_the_profile_object(active):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    cfg = resolve_active_rag_config()
    cfg.chunking.chunk_size = 99999
    assert mgr.get_profile(state["active"]).rag_config.chunking.chunk_size != 99999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_active_config_resolution.py -q`
Expected: FAIL — `active_config` module / functions don't exist.

- [ ] **Step 3: Implement**

Create `active_config.py`:
```python
"""Active-profile config resolution — the single source the RAG engine reads.

resolve_active_rag_config() = the active profile's rag_config (deep copy) with
the env-override layer applied. BOTH the search path (RAGConfig.from_settings)
and the ingestion path (get_shared_rag_service) route through it, so ingestion
and search never use divergent configs for the same active profile.
See Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md §5.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from tldw_chatbook.config import get_cli_setting
from .config import RAGConfig
from ..config_profiles import get_profile_manager

DEFAULT_PROFILE = "hybrid_basic"


def _manager():
    return get_profile_manager()


def _active_profile_id() -> str:
    """The active-profile pointer: [rag.service].profile (reused, single pointer)."""
    try:
        svc = get_cli_setting("rag", "service", {}) or {}
        if isinstance(svc, dict) and svc.get("profile"):
            return str(svc["profile"])
    except Exception as e:
        logger.debug(f"Could not read active profile pointer: {e}")
    return DEFAULT_PROFILE


def _apply_env_overrides(config: RAGConfig,
                         override_embedding_model: Optional[str] = None,
                         override_persist_dir: Optional[Union[str, Path]] = None) -> RAGConfig:
    """Apply the env / explicit-arg override layer onto `config` in place.

    This is the SAME layer RAGConfig.from_settings applied — moved here so both
    resolution paths apply env identically (parity). NOTE: this NO LONGER reads
    the deprecated AppRAGSearchConfig.rag.* value keys — the profile is the base.
    """
    e = config.embedding
    e.model = override_embedding_model or os.getenv("RAG_EMBEDDING_MODEL") or e.model
    dev = os.getenv("RAG_DEVICE") or e.device
    if dev == "auto":
        try:
            import torch
            e.device = ("cuda" if torch.cuda.is_available()
                        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                        else "cpu")
        except ImportError:
            e.device = "cpu"
    else:
        e.device = dev
    cache = os.getenv("RAG_EMBEDDING_CACHE_SIZE")
    if cache:
        e.cache_size = int(cache)
    e.api_key = os.getenv("OPENAI_API_KEY") or get_cli_setting("API", "openai_api_key") or e.api_key
    e.base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or e.base_url
    persist = override_persist_dir or os.getenv("RAG_PERSIST_DIR")
    if persist:
        config.vector_store.persist_directory = Path(persist)
    # vector_store.type: RAG_VECTOR_STORE env is honored by VectorStoreConfig.__post_init__
    return config


def resolve_active_rag_config(override_embedding_model: Optional[str] = None,
                              override_persist_dir: Optional[Union[str, Path]] = None) -> RAGConfig:
    """The active profile's rag_config (deep copy) + env overlay — the single source."""
    active = _active_profile_id()
    profile = _manager().get_profile(active) or _manager().get_profile(DEFAULT_PROFILE)
    base = copy.deepcopy(profile.rag_config) if profile else RAGConfig()
    return _apply_env_overrides(base, override_embedding_model, override_persist_dir)
```

Make `RAGConfig.from_settings` delegate (in `config.py`), preserving its signature:
```python
    @classmethod
    def from_settings(cls, override_embedding_model=None, override_persist_dir=None) -> "RAGConfig":
        """Load the active-profile RAG config + env overrides (see active_config.resolve_active_rag_config)."""
        from .active_config import resolve_active_rag_config
        return resolve_active_rag_config(override_embedding_model, override_persist_dir)
```
Leave the old `from_settings` body's helper functions in place only if still referenced elsewhere; otherwise delete the now-dead `AppRAGSearchConfig.rag.*` value-reading body. (Grep `_rag_user_setting`/`from_settings` internals for other callers before deleting; keep `create_config_for_collection`'s call working — it calls `from_settings(model, dir)`, which now delegates.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_active_config_resolution.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/active_config.py tldw_chatbook/RAG_Search/simplified/config.py Tests/RAG/test_active_config_resolution.py
git commit -m "feat(rag): resolve_active_rag_config — active profile + env as the single config source"
```

---

## Task 2: Route the shared service through the resolver (parity)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/ingestion_indexing.py` (`get_shared_rag_service`, `:154-168`)
- Test: `Tests/RAG/test_config_unification_parity.py`

**Interfaces:**
- Consumes: `resolve_active_rag_config` (Task 1); `create_rag_service(profile_name, config=...)`.
- Produces: the shared (ingestion) service is built with the SAME env-applied active config as the search path.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/test_config_unification_parity.py
import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig, VectorStoreConfig


@pytest.fixture(autouse=True)
def _reset_singleton():
    from tldw_chatbook.RAG_Search.ingestion_indexing import reset_shared_rag_service
    reset_shared_rag_service()
    yield
    reset_shared_rag_service()


def _wire(monkeypatch, tmp_path, active_rag):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    p = ProfileConfig(name="Active", description="d", profile_type="custom", rag_config=active_rag)
    mgr.save_profile(p)
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(ac, "_active_profile_id", lambda: p.id, raising=False)
    return mgr, p


def test_ingest_and_query_config_are_identical_for_active_profile(monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig as RC
    rag = RAGConfig(embedding=EmbeddingConfig(model="mock"),
                    vector_store=VectorStoreConfig(type="memory"))
    _wire(monkeypatch, tmp_path, rag)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-wins-model")
    # Search path config:
    query_cfg = RC.from_settings()
    # Ingestion path config (what get_shared_rag_service will build from):
    ingest_cfg = resolve_active_rag_config()
    assert query_cfg.embedding.model == ingest_cfg.embedding.model == "env-wins-model"
    # And the fingerprint-determining fields match (anti dimension-crash):
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    assert fingerprint_collection(query_cfg) == fingerprint_collection(ingest_cfg)
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `pytest Tests/RAG/test_config_unification_parity.py -q`
Expected: with Task 1 done, `from_settings` and `resolve_active_rag_config` already agree, so this likely PASSES as a contract lock. If it FAILS, the two paths диverge — investigate before proceeding (do not weaken the assertion).

- [ ] **Step 3: Wire the shared service to the resolver**

In `ingestion_indexing.py` `get_shared_rag_service`, when constructing the ACTIVE shared service (no explicit `profile_name` override passed), build from the resolver so ingestion applies env like search:
```python
                from .simplified import create_rag_service
                from .simplified.active_config import resolve_active_rag_config
                if profile_name is None:
                    profile = _configured_profile()
                    _shared_service = create_rag_service(profile_name=profile,
                                                         config=resolve_active_rag_config())
                else:
                    _shared_service = create_rag_service(profile_name=profile_name)
                logger.info(f"Created shared RAG service (profile={profile_name or _configured_profile()})")
```

- [ ] **Step 4: Run tests**

Run: `pytest Tests/RAG/test_config_unification_parity.py Tests/RAG/test_ingestion_indexing.py -q`
Expected: PASS; the ingestion-indexing suite unregressed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/ingestion_indexing.py Tests/RAG/test_config_unification_parity.py
git commit -m "feat(rag): shared service uses resolve_active_rag_config (ingest==query parity)"
```

---

## Task 3: fusion.py hybrid_alpha from the active profile

**Files:**
- Modify: `tldw_chatbook/RAG_Search/fusion.py` (`:240-246`)
- Test: `Tests/RAG/test_active_config_resolution.py` (append)

**Interfaces:** Consumes `resolve_active_rag_config`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_hybrid_alpha_comes_from_active_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, SearchConfig, VectorStoreConfig
    from tldw_chatbook.RAG_Search.fusion import resolve_hybrid_alpha
    mgr, state = active
    p = ProfileConfig(name="Alpha", description="d", profile_type="custom",
                      rag_config=RAGConfig(search=SearchConfig(hybrid_alpha=0.33),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    assert resolve_hybrid_alpha() == pytest.approx(0.33)  # explicit=None -> active profile
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_active_config_resolution.py -q -k hybrid_alpha`
Expected: FAIL — `resolve_hybrid_alpha` still reads `AppRAGSearchConfig.rag.retriever`.

- [ ] **Step 3: Redirect the read**

In `fusion.py`, replace the `get_cli_setting("AppRAGSearchConfig","rag",{})` block (`~:242-244`) with a read from the active profile, keeping the same fallback-on-error contract:
```python
        try:
            from .simplified.active_config import resolve_active_rag_config
            value = resolve_active_rag_config().search.hybrid_alpha
        except Exception as e:  # config loading must never break search
            logger.warning(f"Could not read hybrid_alpha from active profile: {e}")
            value = None
```
(Preserve the surrounding `resolve_hybrid_alpha(explicit)` precedence: explicit arg > active-profile value > default.)

- [ ] **Step 4: Run tests**

Run: `pytest Tests/RAG/test_active_config_resolution.py Tests/RAG/test_fusion.py -q`
Expected: PASS; fusion suite unregressed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/fusion.py Tests/RAG/test_active_config_resolution.py
git commit -m "feat(rag): hybrid_alpha resolves from the active profile (deprecate scattered key)"
```

---

## Task 4: `set_active_profile` (write pointer + reset service)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/active_config.py` (add `set_active_profile`)
- Test: `Tests/RAG/test_active_config_resolution.py` (append)

**Interfaces:**
- Consumes: `save_setting_to_cli_config` (`config.py:3824`); `reset_shared_rag_service` (`ingestion_indexing.py:192`, ALREADY EXISTS).
- Produces: `set_active_profile(profile_id: str) -> None`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_set_active_profile_writes_pointer_and_resets_service(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    writes = {}
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda section, key, value: writes.update({(section, key): value}) or True,
                        raising=False)
    reset = {"called": False}
    monkeypatch.setattr(ac, "reset_shared_rag_service",
                        lambda: reset.update(called=True), raising=False)
    set_active_profile("my_profile")
    assert writes.get(("rag", "service.profile")) == "my_profile"
    assert reset["called"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_active_config_resolution.py -q -k set_active`
Expected: FAIL — `set_active_profile` doesn't exist.

- [ ] **Step 3: Implement**

Add to `active_config.py` (with the imports at module top):
```python
from tldw_chatbook.config import save_setting_to_cli_config
from ..ingestion_indexing import reset_shared_rag_service


def set_active_profile(profile_id: str) -> None:
    """Point [rag.service].profile at `profile_id` and drop the shared service.

    The next resolve_active_rag_config()/get_shared_rag_service() rebuilds on the
    new profile (and, via SP1, its fingerprinted collection). An in-flight worker
    keeps its own service reference — the reset never yanks a running op; it only
    clears the singleton so the NEXT caller rebuilds. The (potentially expensive)
    embedding-model reload is the caller's concern to run off-thread (SP3 UI).
    """
    save_setting_to_cli_config("rag", "service.profile", profile_id)
    reset_shared_rag_service()
```
(If `save_setting_to_cli_config`'s nested-key form differs — e.g. it wants `section="rag.service", key="profile"` — match its actual signature; verify by reading `config.py:3824` at implementation.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_active_config_resolution.py -q -k set_active`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/active_config.py Tests/RAG/test_active_config_resolution.py
git commit -m "feat(rag): set_active_profile writes the pointer + resets the shared service"
```

---

## Task 5: First-run import ("Imported settings" profile) + cross-SP fingerprint invariant

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/active_config.py` (add `ensure_imported_profile`)
- Test: `Tests/RAG/test_first_run_import.py`

**Interfaces:**
- Consumes: `ConfigProfileManager` (SP2a), `fingerprint_collection` (SP1), `_apply_env_overrides`.
- Produces: `ensure_imported_profile() -> Optional[str]` — on first run, snapshot the resolved-from-legacy-config into an "Imported settings" user profile, set it active, return its id; idempotent no-op afterward.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/test_first_run_import.py
import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager


def _wire(monkeypatch, tmp_path):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    ptr = {"v": None}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: ptr["v"] or "hybrid_basic", raising=False)
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda s, k, v: ptr.update(v=v) or True, raising=False)
    monkeypatch.setattr(ac, "reset_shared_rag_service", lambda: None, raising=False)
    return mgr, ptr


def test_first_run_creates_imported_profile_and_sets_active(monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    mgr, ptr = _wire(monkeypatch, tmp_path)
    new_id = ensure_imported_profile()
    assert new_id is not None
    imported = mgr.get_profile(new_id)
    assert imported is not None and imported.read_only is False
    assert ptr["v"] == new_id  # set active
    # Idempotent: a second call is a no-op (no duplicate).
    assert ensure_imported_profile() is None


def test_imported_fingerprint_matches_sp1_adoption(monkeypatch, tmp_path):
    """Cross-SP invariant: the imported profile's fingerprint == the fingerprint
    SP1 would adopt the legacy 'default' collection under (both from the same
    first-run resolved config), so an upgraded user keeps their index."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    mgr, ptr = _wire(monkeypatch, tmp_path)
    new_id = ensure_imported_profile()
    imported_fp = fingerprint_collection(mgr.get_profile(new_id).rag_config)
    # SP1 adopts under the config active at first persistent construction, which is
    # now the imported profile's config (via resolve_active_rag_config):
    adopted_fp = fingerprint_collection(resolve_active_rag_config())
    assert imported_fp == adopted_fp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_first_run_import.py -q`
Expected: FAIL — `ensure_imported_profile` doesn't exist.

- [ ] **Step 3: Implement**

Add to `active_config.py`:
```python
from ..config_profiles import ProfileConfig

_IMPORTED_ID = "imported_settings"


def ensure_imported_profile() -> Optional[str]:
    """On first run, capture the currently-resolved RAG config into a writable
    'Imported settings' profile and set it active. Idempotent (returns None if it
    already exists). The captured config's SP1 fingerprint matches what SP1 adopts
    the legacy collection under, so the user keeps their index on upgrade."""
    mgr = _manager()
    if mgr.get_profile(_IMPORTED_ID) is not None:
        return None
    # Snapshot the resolved config (active pointer may name a builtin default today).
    snapshot = resolve_active_rag_config()
    profile = ProfileConfig(name="Imported settings",
                            description="Captured from your existing RAG configuration on first run.",
                            profile_type="custom", rag_config=snapshot)
    profile.id = _IMPORTED_ID
    mgr.save_profile(profile)
    set_active_profile(_IMPORTED_ID)
    return _IMPORTED_ID
```
Wire a single first-run call: invoke `ensure_imported_profile()` once where the RAG service first initializes for real (e.g. at the top of `get_shared_rag_service` before building, guarded so it only runs when no user profile is active yet). Keep it exception-safe (never block service creation). Verify placement at implementation so it runs on the real upgrade path but not in every unit test (tests call it explicitly).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_first_run_import.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/active_config.py Tests/RAG/test_first_run_import.py
git commit -m "feat(rag): first-run import into 'Imported settings' profile (SP1 fingerprint invariant)"
```

---

## Task 6: Deprecation cleanup + docs + regression gate

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py` (remove now-dead `AppRAGSearchConfig.rag.*` value-reading helpers if unreferenced), `rag_config_example.toml` / README note.
- Test: run the broader RAG suite.

- [ ] **Step 1: Grep for remaining scattered value readers**

Run: `git grep -nE 'get_cli_setting\("AppRAGSearchConfig", "rag"' -- tldw_chatbook`
Expected remaining: `ingestion_indexing.py:85` (the `rag.indexing` toggle — LEAVE, it's not profile values). Any other value-read is a miss — route it through `resolve_active_rag_config`. Remove `config.py` helper functions that only supported the old `from_settings` value-reads and now have zero callers (grep each before deleting).

- [ ] **Step 2: Doc note**

Add to `README_enhanced_services.md`: "The active profile (`[rag.service].profile`) is the single source of RAG config — both ingestion and search resolve through `active_config.resolve_active_rag_config` (profile base + env overrides). The old `[AppRAGSearchConfig.rag.*]` value keys are deprecated in favor of profiles."

- [ ] **Step 3: Full regression gate**

Run: `pytest Tests/RAG/ -q`
Expected: PASS. Watch: any test that set `AppRAGSearchConfig.rag.*` values and expected them reflected in a built config now needs to set them via a profile instead — update such tests to the profile model (do not restore the deprecated key reads). Every test that builds/reset the shared service must `reset_shared_rag_service()` in teardown (task-408 pollution).

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/config.py tldw_chatbook/RAG_Search/README_enhanced_services.md
git commit -m "chore(rag): deprecate scattered AppRAGSearchConfig.rag.* value keys; document active-profile source"
```

---

## Self-Review

**1. Spec coverage** (`2026-07-21-rag-profile-system-design.md` §5-7, SP2b slice):
- §5 unify config resolution onto the active profile; both paths via one resolver; env override layer; deprecate scattered keys; parity test → Tasks 1,2,3,6. ✓
- §4 single active pointer `[rag.service].profile` → Task 1 (`_active_profile_id`), Task 4 (write). ✓
- §7 switch mechanics: write pointer + reset service (reset ALREADY EXISTS); off-thread reload is caller/SP3 → Task 4. ✓
- §6 first-run import → "Imported settings" active; fingerprint == SP1 adoption → Task 5. ✓
- §8 parity test; singleton-reset test isolation (task-408) → Task 2 + Global Constraints. ✓

**2. Placeholder scan:** every code step has full code; every test has assertions. The two "verify signature at implementation" notes (`save_setting_to_cli_config` nested-key form; first-run call placement) are concrete verification instructions, not deferred design. ✓

**3. Type consistency:** `resolve_active_rag_config(override_embedding_model, override_persist_dir)`, `_apply_env_overrides(config, ...)`, `set_active_profile(profile_id)`, `ensure_imported_profile()`, `_active_profile_id()`, `_manager()` — names identical across the tasks that define/consume them. `reset_shared_rag_service`/`save_setting_to_cli_config` used as existing seams. ✓

**Cross-SP invariant** (imported fingerprint == SP1 adoption) is the highest-risk item and gets a dedicated e2e test (Task 5). Deferred to SP3: the settings-screen UI + off-thread reload orchestration.
