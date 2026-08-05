# SP3 — RAG Settings Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the profile system user-facing: repurpose the "Library & RAG" settings category into a "RAG" category whose form edits the **active profile** (not the dead `AppRAGSearchConfig.rag.*` keys), with a profile-manager region (set-active/clone/rename/delete), an extended editor covering every `ProfileConfig` section, index status + backfill, and honest index-change warnings.

**Architecture:** A new pure adapter module `settings_rag_profile_adapter.py` is the single seam between the settings screen and the SP2a/SP2b machinery: load reads `resolve_active_rag_config()`, save writes the active profile via `ConfigProfileManager.save_profile` (refusing builtins), plus profile-list/set-active/clone/rename/delete/index-status/fingerprint-warning helpers — all headless-testable. The screen keeps its existing category/draft/handler machinery (156 wiring points) and is extended pattern-for-pattern: the framework draft holds only the ACTIVE profile's edits (drafts are strictly category-keyed; a dirty draft prompts save/discard before set-active).

**Tech Stack:** Python 3.11+, Textual (`Collapsible`, `@work(thread=True)`), existing settings framework (`SettingsDraft`, per-field `handle_library_rag_*_changed` handlers, `_render_library_rag_detail`), SP1 `collection_indexes.index_status`/`collection_fingerprint.fingerprint_collection`, SP2a `ConfigProfileManager` CRUD, SP2b `active_config.resolve_active_rag_config`/`set_active_profile`, pytest + textual-serve QA captures.

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-21-rag-settings-screen-design.md`; overview `-rag-settings-profiles-overview.md`. SP1 (#771), SP2a (#780), SP2b (#795) merged.
- **Save target:** the RAG category's Save writes the **active profile file** via `get_profile_manager().save_profile(profile)` — NEVER `AppRAGSearchConfig.rag.*` (`build_library_rag_save_sections` and its `SettingsConfigAdapter.save_sections` call are retired for this category). The only config.toml write is the pointer, via SP2b `set_active_profile(profile_id)`.
- **Enum value kept:** `SettingsCategoryId.LIBRARY_RAG = "library-rag"` stays; display title changes `"Library & RAG"` → `"RAG"` (registration at `settings_screen.py:406`); contract-registry blurbs (`:687`) and `SettingsOwnershipRecord.owns_config_sections` (`:1447+`) rewritten to profile ownership (`rag_profiles/<active>.json` + `[rag.service].profile` pointer).
- **Verified fact — rerank toggle semantics:** the service enables reranking iff `profile.reranking_config is not None` (`rag_factory.py`: `enable_reranking = profile.reranking_config is not None`). The UI rerank toggle therefore controls the PRESENCE of `reranking_config` on the profile (creating a default `RerankingConfig()` on enable, setting `None` on disable) AND mirrors `rag_config.search.enable_reranking` for coherence. An integration test must prove the toggle reaches `create_rag_service`'s flag (inert-toggle trap).
- **Verified fact — drafts are strictly category-keyed** (`_settings_drafts: dict[SettingsCategoryId, SettingsDraft]`, deepcopy state save/restore at `:1134/:1172`). NO composite keys. The framework draft holds only the ACTIVE profile's edits; set-active with a dirty draft prompts Save/Discard/Cancel and clears the draft on switch.
- **Builtins are read-only:** when the active profile is a builtin (`profile.read_only` / id in `_builtin_ids`), the editor renders disabled with a "Built-in profile — Clone to edit" banner + Clone button; `save_library_rag_defaults_to_active_profile` refuses builtins as defense-in-depth (SP2a's `save_profile` raises anyway).
- **Index-determining fields** (SP1 fingerprint set: `embedding.model`, `embedding.max_length`, all 10 `ChunkingConfig` fields, `vector_store.distance_metric`) are visually marked ("⚠ re-index") and BOTH triggers — set-active to a differently-fingerprinted profile AND save-of-active with an index-field change — surface the "points at a new empty index — Backfill needed" warning (compare `fingerprint_collection` before/after).
- **Off-thread:** set-active and index-status/backfill run via `@work(thread=True)` / async workers (precedent `settings_screen.py:3368+`); `index_status` touches chroma on disk — never call it on the UI thread during compose.
- **Editor field scope (explicit YAGNI decision, surfaced for the user gate):** every `ProfileConfig` section gets a group — Search (the existing 10 fields), Embedding (`model`, `device`, `batch_size`, `max_length`), Chunking (`chunk_size`, `chunk_overlap`, `chunking_method`), Vector store (`distance_metric`), Reranking (enable toggle, `reranker_model`, `reranker_top_k`) — ~21 fields total. Deferred to follow-up: cache/TTL knobs, parent-doc tuning, remaining chunking flags, `api_key`/`base_url` (secrets don't belong in this form), dead `query_expansion`, internal `pipeline`. File the follow-up task at execution.
- **Value-aware dirty-marking; no giant recompose** (Changed-echo race lesson): groups are `Collapsible` sections inside the existing scroll container; handlers compare against loaded values before staging.
- **QA gate:** textual-serve captures of both regions, builtin read-only state, set-active flow, index-empty warning — user screen approval required before merge.
- Tests: headless pytest for the adapter (temp profiles dir via `ConfigProfileManager(profiles_dir=...)`, monkeypatched `active_config._manager`/`_active_profile_id` — the established SP2b idiom); screen logic via the existing `Tests/UI/test_settings_*` bare-instance/pilot patterns. `reset_profile_manager_cache()` in teardown wherever the default manager could be touched.

## Plan-time facts (verified)
- Screen wiring: imports `:122-128`; registration `:364/:406` (title); contract blurbs `:687`; ownership records `:1447+` (list stale `AppRAGSearchConfig...` keys); state `:1052/:1082`; loaded/current/draft helpers `:2080-2108`; invalid-field selectors `:2749/:2774`; preview rows `:2889`; renderer `_render_library_rag_detail` `:7336`; per-field handlers `:9052+`; save dispatch `_save_library_rag_sections` ~`:9871` region (build at ~`:9797`).
- `SettingsDraft` = `{category, originals, values, set_value(key, original, draft)}` (`settings_config_models.py:63`).
- SP2a CRUD: `get_profile(id)`, `list_profiles()`, `save_profile(p)`, `delete_profile(id)`, `rename_profile(id, new_name)`, `clone_profile(source_id, new_name)` (`config_profiles.py:692-822`); `get_profile_manager(profiles_dir=None)` cached singleton for default dir + `reset_profile_manager_cache()`.
- SP2b: `resolve_active_rag_config()`, `set_active_profile(profile_id)` (validates slug, writes `[rag.service].profile`, resets shared service), `_active_profile_id()`.
- SP1: `index_status(config) -> {"state": "built"|"empty"|"absent", "count", "provenance"}` (`collection_indexes.py:184`); `fingerprint_collection(config) -> str`; `backfill_semantic_index(...)` async (`ingestion_indexing.py:978`).
- `RAGConfig.validate() -> List[str]` (`config.py:447`) — zero callers today; SP3 wires it.
- `Collapsible` precedent: `LLM_Management_Window.py`, `mcp_inspector.py`, etc.

---

## File Structure
- **Create** `tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py` — ALL headless logic: load-from-active-profile, apply-values-to-profile, save-to-active-profile (builtin-refusing), profile listing (builtin/user grouping + active marker), set-active/clone/rename/delete wrappers, rerank-presence semantics, `RAGConfig.validate()` + rerank validation, fingerprint-change detection, index-status fetch.
- **Modify** `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py` — extend the dataclass with the ~11 new editor fields + their coercers/validators; `load_library_rag_defaults` re-pointed at the adapter.
- **Modify** `tldw_chatbook/UI/Screens/settings_screen.py` — retitle; contract/ownership text; profile-manager region + Collapsible groups in `_render_library_rag_detail`; new per-field handlers + selectors (pattern-for-pattern); save dispatch → adapter; set-active worker + dirty prompt; index-status/backfill/warning UI.
- **Create tests** `Tests/UI/test_settings_rag_profile_adapter.py` (headless) + extend the existing settings-screen UI tests.

---

### Task 1: Adapter seam — load/save through the active profile (kills the dead-writes bug)

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py`
- Modify: `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py` (only `load_library_rag_defaults`), `tldw_chatbook/UI/Screens/settings_screen.py` (`_library_rag_loaded_defaults` `:2080`, the RAG branch of the save dispatch ~`:9797/:9871`, ownership records `:1447+`, contract blurb `:687`, title `:406` → `"RAG"`)
- Test: `Tests/UI/test_settings_rag_profile_adapter.py`

**Interfaces:**
- Consumes: `resolve_active_rag_config`, `_active_profile_id` (module-global seams, monkeypatchable), `get_profile_manager`, `ProfileConfig`.
- Produces (exact signatures later tasks use):
  - `load_rag_defaults_from_active_profile() -> SettingsLibraryRagDefaults`
  - `apply_defaults_to_profile(profile: ProfileConfig, values: SettingsLibraryRagDefaults) -> ProfileConfig` (pure; maps each dataclass field onto `profile.rag_config.search.*`)
  - `save_rag_defaults_to_active_profile(values: SettingsLibraryRagDefaults) -> tuple[bool, str]` — `(False, "builtin")` when active is read-only; `(True, "")` on success; `(False, "<error>")` otherwise.
  - `active_profile_info() -> dict` — `{"id", "name", "read_only"}`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_settings_rag_profile_adapter.py
import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, SearchConfig, VectorStoreConfig


@pytest.fixture
def wired(tmp_path, monkeypatch):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: state["active"], raising=False)
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad
    monkeypatch.setattr(ad, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(ad, "_active_profile_id", lambda: state["active"], raising=False)
    return mgr, state


def _user_profile(mgr, state, **search_over):
    p = mgr.clone_profile("hybrid_basic", "My RAG")
    for k, v in search_over.items():
        setattr(p.rag_config.search, k, v)
    mgr.save_profile(p)
    state["active"] = p.id
    return p


def test_load_reads_the_active_profile(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import load_rag_defaults_from_active_profile
    mgr, state = wired
    _user_profile(mgr, state, default_top_k=42, hybrid_alpha=0.25)
    d = load_rag_defaults_from_active_profile()
    assert d.default_top_k == 42
    assert d.hybrid_alpha == 0.25


def test_save_writes_the_active_profile_file_not_config(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile, save_rag_defaults_to_active_profile)
    mgr, state = wired
    p = _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()
    d.default_top_k = 77
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""
    # Reload from disk via a FRESH manager over the same dir:
    mgr2 = ConfigProfileManager(profiles_dir=mgr.profiles_dir)
    assert mgr2.get_profile(p.id).rag_config.search.default_top_k == 77


def test_save_refuses_builtin_active(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile, save_rag_defaults_to_active_profile)
    mgr, state = wired      # active = hybrid_basic (builtin)
    d = load_rag_defaults_from_active_profile()
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert not ok and reason == "builtin"


def test_active_profile_info(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import active_profile_info
    mgr, state = wired
    info = active_profile_info()
    assert info == {"id": "hybrid_basic", "name": "Hybrid Basic", "read_only": True}
```

- [ ] **Step 2: Run to verify RED** — `pytest Tests/UI/test_settings_rag_profile_adapter.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the adapter (core, exact)**

```python
# tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py
"""Adapter between the Settings RAG category and the profile system.

The RAG category edits the ACTIVE PROFILE (SP2a storage, SP2b resolution) —
never the deprecated AppRAGSearchConfig.rag.* keys, which the engine ignores.
All functions here are headless (no Textual imports) and testable.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
from tldw_chatbook.RAG_Search.simplified.active_config import _active_profile_id  # module seam
from .settings_library_rag_defaults import SettingsLibraryRagDefaults


def _manager():
    return get_profile_manager()


def _active_profile() -> Optional[ProfileConfig]:
    return _manager().get_profile(_active_profile_id())


def load_rag_defaults_from_active_profile() -> SettingsLibraryRagDefaults:
    """Current active profile's search settings as the category dataclass."""
    profile = _active_profile()
    if profile is None:
        return SettingsLibraryRagDefaults()
    s = profile.rag_config.search
    return SettingsLibraryRagDefaults(
        default_search_mode=s.default_search_mode,
        default_top_k=int(s.default_top_k),
        fts_top_k=int(s.fts_top_k),
        vector_top_k=int(s.vector_top_k),
        hybrid_alpha=float(s.hybrid_alpha),
        score_threshold=float(s.score_threshold),
        include_citations=bool(s.include_citations),
        citation_style=s.citation_style,
        snippet_max_chars=int(s.snippet_max_chars),
        max_context_size=int(s.max_context_size),
    )


def apply_defaults_to_profile(
    profile: ProfileConfig, values: SettingsLibraryRagDefaults
) -> ProfileConfig:
    """Map the category dataclass onto the profile's SearchConfig (pure)."""
    s = profile.rag_config.search
    s.default_search_mode = values.default_search_mode
    s.default_top_k = int(values.default_top_k)
    s.fts_top_k = int(values.fts_top_k)
    s.vector_top_k = int(values.vector_top_k)
    s.hybrid_alpha = float(values.hybrid_alpha)
    s.score_threshold = float(values.score_threshold)
    s.include_citations = bool(values.include_citations)
    s.citation_style = values.citation_style
    s.snippet_max_chars = int(values.snippet_max_chars)
    s.max_context_size = int(values.max_context_size)
    return profile


def save_rag_defaults_to_active_profile(
    values: SettingsLibraryRagDefaults,
) -> tuple[bool, str]:
    """Persist values into the active profile's file. (False, "builtin") if read-only."""
    profile = _active_profile()
    if profile is None:
        return False, "no-active-profile"
    if profile.read_only:
        return False, "builtin"
    try:
        _manager().save_profile(apply_defaults_to_profile(profile, values))
        return True, ""
    except Exception as e:  # save must never crash the settings screen
        logger.error(f"Saving RAG defaults to active profile failed: {e}")
        return False, str(e)


def active_profile_info() -> dict:
    profile = _active_profile()
    if profile is None:
        return {"id": _active_profile_id(), "name": "(missing)", "read_only": True}
    return {"id": profile.id, "name": profile.name, "read_only": bool(profile.read_only)}
```
(Note `_active_profile_id` is imported as a module attribute of the ADAPTER too so tests monkeypatch `ad._active_profile_id`; import it as `from ...active_config import _active_profile_id` then reference the module-global — the test patches both seams.)

- [ ] **Step 4: Wire the screen (anchored edits)**
  - `_library_rag_loaded_defaults` (`:2080`): `return load_rag_defaults_from_active_profile()` (drop `load_library_rag_defaults(self._app_config_mapping())`).
  - Save dispatch (RAG branch ~`:9797/:9871`): replace `build_library_rag_save_sections(...)` + `save_sections` with `ok, reason = save_rag_defaults_to_active_profile(self._library_rag_current_defaults())`; on `("builtin")` notify "Built-in profile is read-only — Clone to edit"; on success keep the existing post-save draft-clear/refresh flow.
  - Title `:406` `"Library & RAG"` → `"RAG"`; contract blurb `:687` "Affected config" → `"the active RAG profile (rag_profiles/<id>.json) and the [rag.service].profile pointer"`; ownership records `:1447+` → the same two strings (drop every `AppRAGSearchConfig...` entry).
  - Any test asserting the old title/ownership strings: update to the new strings (do not revert).

- [ ] **Step 5: GREEN + regression** — `pytest Tests/UI/test_settings_rag_profile_adapter.py -q` PASS; `pytest Tests/UI/ -q -k "settings"` at/above baseline (update stale-string assertions only). Commit: `feat(settings): RAG category loads/saves the active profile (retire AppRAGSearchConfig writes)`.

---

### Task 2: Profile-manager region (set-active / clone / rename / delete)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py` (add listing + action wrappers), `tldw_chatbook/UI/Screens/settings_screen.py` (`_render_library_rag_detail` `:7336` top section + handlers + worker)
- Test: `Tests/UI/test_settings_rag_profile_adapter.py` (append) + screen test

**Interfaces (adapter additions):**
- `list_profiles_grouped() -> dict` — `{"builtin": [{"id","name"}...], "user": [...], "active_id": str}` (sorted by name within groups).
- `activate_profile(profile_id: str) -> tuple[bool, str]` — wraps SP2b `set_active_profile` (exception → `(False, msg)`); callers run it off-thread.
- `clone_active_or(profile_id: str, new_name: str) -> tuple[bool, str]`; `rename_user_profile(profile_id, new_name)`; `delete_user_profile(profile_id) -> tuple[bool, str]` — thin wrappers converting SP2a's ValueErrors into `(False, reason)`.

- [ ] **Step 1: Failing adapter tests** — grouped listing puts builtins under `"builtin"` with `active_id` correct; `activate_profile` on a valid user id flips the (monkeypatched) pointer and returns `(True,"")`; on `"../evil"` returns `(False, ...)` (SP2b validation) without raising; `delete_user_profile("hybrid_basic")` → `(False, ...)` not an exception. RED → implement wrappers (each is ≤8 lines calling the SP2a/SP2b seam inside try/except ValueError) → GREEN.
- [ ] **Step 2: Screen region.** At the top of `_render_library_rag_detail` (`:7336`) render a "Profiles" block: static line `Active: <name> (built-in)` + a `Select` of all profiles (grouped labels `── Built-in ──` / `── Yours ──` as disabled options, per repo Select conventions) + Buttons `Set active`, `Clone…`, `Rename…`, `Delete`. Rename/Clone prompt for a name via the screen's existing modal-input pattern (reuse whatever `settings_screen` uses for text prompts; if none, a minimal `ModalScreen` with one `Input` — repo precedent in picker modals). Delete confirms first.
- [ ] **Step 3: Set-active worker + dirty prompt.**
```python
    @work(exclusive=True, thread=True, group="settings-rag-set-active")
    def _rag_set_active_worker(self, profile_id: str) -> None:
        ok, reason = activate_profile(profile_id)   # SP2b resets the shared service
        self.call_from_thread(self._rag_after_set_active, ok, reason)
```
Before dispatching: if `self._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG)` → confirm Save/Discard/Cancel (save via Task 1's save path first when chosen; discard clears the draft via the existing `_settings_drafts.pop(category, ...)` flow). `_rag_after_set_active` refreshes the category (existing refresh helper), shows the fingerprint warning if Task 4's check says the index changed (until Task 4 lands, just refresh + notify).
- [ ] **Step 4: Builtin read-only state.** When `active_profile_info()["read_only"]`: render the banner `Built-in profile — read-only. Clone to edit.` and set `disabled=True` on the editor inputs (the renderer already has all field ids via `_library_rag_field_selector` `:2774` — iterate those selectors).
- [ ] **Step 5: GREEN + commit** `feat(settings): RAG profile manager region (set-active/clone/rename/delete, off-thread)`.

---

### Task 3: Extended editor groups + validation (rerank presence semantics)

**Files:**
- Modify: `settings_library_rag_defaults.py` (dataclass + coercers + `validate_library_rag_defaults`), `settings_rag_profile_adapter.py` (extend load/apply + `validate_full_config`), `settings_screen.py` (Collapsible groups, handlers, selectors, preview rows)
- Test: adapter tests (append) + one integration test proving the rerank toggle reaches the service flag

**New dataclass fields (exact — extend `SettingsLibraryRagDefaults`, all with the current-profile defaults):**
`embedding_model: str`, `embedding_device: str`, `embedding_batch_size: int`, `embedding_max_length: int`, `chunk_size: int`, `chunk_overlap: int`, `chunking_method: str` (∈ words/sentences/paragraphs), `distance_metric: str` (∈ cosine/l2/ip), `enable_reranking: bool`, `reranker_model: str` (empty = default), `reranker_top_k: int`.

- [ ] **Step 1: Failing adapter tests.** (a) load round-trips the new fields from a profile with distinctive values; (b) **rerank presence**: saving with `enable_reranking=True` on a profile whose `reranking_config is None` creates one (`profile.reranking_config is not None` after reload) and sets `rag_config.search.enable_reranking=True`; saving with `False` sets it to `None`; (c) **inert-toggle integration**: after (b), `create_rag_service(profile_name=<id>)` (with mocked embeddings/memory store, monkeypatched manager) yields a service whose `enable_reranking is True` — the flag genuinely threads.
- [ ] **Step 2: Extend `apply_defaults_to_profile` + `load_rag_defaults_from_active_profile`** with the new field mappings (`profile.rag_config.embedding.model = values.embedding_model` etc.), and the rerank block:
```python
    if values.enable_reranking:
        if profile.reranking_config is None:
            from tldw_chatbook.RAG_Search.reranker import RerankingConfig
            profile.reranking_config = RerankingConfig()
        if values.reranker_model:
            profile.reranking_config.model_name = values.reranker_model
        profile.reranking_config.top_k_to_rerank = int(values.reranker_top_k)
        profile.rag_config.search.enable_reranking = True
    else:
        profile.reranking_config = None
        profile.rag_config.search.enable_reranking = False
```
(Verify `RerankingConfig`'s actual field names — `model_name`/`top_k_to_rerank` per `reranker.py` — at implementation; adjust mapping if they differ.)
- [ ] **Step 3: Validation.** `validate_full_config(values) -> list[str]` in the adapter: build a scratch profile copy, `apply_defaults_to_profile`, return `profile.rag_config.validate()` (**first caller ever**) plus rerank checks (`reranker_top_k >= 1`, `reranker_top_k <= default_top_k` warning). Extend `validate_library_rag_defaults` to include these messages; the screen's existing invalid-field machinery (`:2749/:2774`) gets selector entries for each new field.
- [ ] **Step 4: Screen groups.** In `_render_library_rag_detail`, wrap the existing 10 search fields in `Collapsible(title="Search", collapsed=False)` and add `Collapsible` groups **Embedding**, **Chunking**, **Vector store**, **Reranking** with the new fields — each field one handler + one selector entry, copied pattern-for-pattern from `handle_library_rag_default_top_k_changed` (`:9062`) with value-aware dirty-marking (compare to loaded before `set_value`). Index-determining fields' labels get the suffix `" ⚠ re-index"` (`embedding_model`, `embedding_max_length`, `chunk_size`, `chunk_overlap`, `chunking_method`, `distance_metric`).
- [ ] **Step 5: GREEN + regression + commit** `feat(settings): full RAG profile editor groups + RAGConfig.validate wiring + rerank presence semantics`.

---

### Task 4: Index status, backfill, and index-change warnings

**Files:**
- Modify: `settings_rag_profile_adapter.py` (+2 helpers), `settings_screen.py` (status row + Backfill + warning surfacing)
- Test: adapter tests (append)

**Adapter additions:**
- `index_change_pending(values: SettingsLibraryRagDefaults) -> bool` — pure: scratch-apply values to a copy of the active profile, compare `fingerprint_collection(before.rag_config) != fingerprint_collection(after.rag_config)`.
- `fetch_index_status() -> dict` — `index_status(resolve_active_rag_config())` wrapped in try/except → `{"state": "unknown"}` on error. Callers run it off-thread.

- [ ] **Step 1: Failing tests.** `index_change_pending` False for a `default_top_k` change, True for `chunk_size`/`embedding_model`/`distance_metric` changes; `fetch_index_status` returns `{"state": "absent", ...}` for a memory-store profile (SP1 behavior) and never raises. RED → implement → GREEN.
- [ ] **Step 2: Status row.** Below the Profiles block: `Index: <state> · <count> vectors · built with <provenance model / chunk>` populated by a `@work(thread=True)` fetch on category show + after set-active/save (never during compose). `Backfill` button → async worker calling `backfill_semantic_index()` (`ingestion_indexing.py:978`; follow its existing call sites for args) with start/finish notifications.
- [ ] **Step 3: Warnings at both triggers.** (a) Save path: when `index_change_pending(current_values)` before saving → include "This change re-points to a new (empty) index — run Backfill" in the save notification; (b) set-active: in `_rag_after_set_active`, if the new active fingerprint ≠ previous → same warning. Both reuse one message constant.
- [ ] **Step 4: GREEN + commit** `feat(settings): RAG index status + backfill + honest re-index warnings`.

---

### Task 5: Retire dead code, full regression gate, QA captures

**Files:**
- Modify: `settings_library_rag_defaults.py` (delete `build_library_rag_save_sections`, `_rag_section`, and any now-unreferenced helper — grep each for zero callers first), `settings_screen.py` (drop the dead imports `:124`)
- Test: full suites + textual-serve captures

- [ ] **Step 1: Dead-code sweep.** `git grep -n "build_library_rag_save_sections\|_rag_section"` → remove zero-caller helpers + their imports; keep `SettingsLibraryRagDefaults` and the coercers (still the form model). Any test of the removed builders: delete the test WITH the builder (it tested writing deprecated keys) — note each in the report.
- [ ] **Step 2: Regression gate.** `pytest Tests/UI/ -q` (settings-related at/above baseline; note pre-existing failures per the shell/snapshot baseline) and `pytest Tests/RAG/ -q` (unaffected, sanity).
- [ ] **Step 3: QA captures** via the textual-serve + playwright recipe (memory: `tldw-chatbook-dev-environment`): (1) RAG category with Profiles block + Search group open; (2) builtin active → read-only banner + disabled fields; (3) user profile active → editable, index status row; (4) set-active flow warning; (5) collapsed/expanded groups. Save under `Docs/superpowers/qa/rag-settings-sp3-2026-07/`. **STOP for user screen approval** — the captures gate the PR.
- [ ] **Step 4: Commit** `chore(settings): retire AppRAGSearchConfig save builders + SP3 QA captures`.

---

## Self-Review

**1. Spec coverage** (`2026-07-21-rag-settings-screen-design.md`): §1 category repurpose/retitle + contract text → T1. §2 save target (profile file + pointer only) → T1/T2. §3 two regions: profile manager (set-active/clone/rename/delete, builtin read-only + Clone-to-edit, off-thread set-active) → T2; editor groups incl. rerank-presence semantics → T3. §4 drafts: category-keyed framework draft = active profile only + Save/Discard prompt on switch (composite key verified unsafe → spec's fallback) → T2. §5 validation (`RAGConfig.validate` first caller + rerank) → T3. §6 layout (Collapsible in scroll container, value-aware dirty, no giant recompose) → T3. §7 index status + Backfill + both warning triggers → T4. §8/§9 tests + QA gate + all four plan-time verifications → resolved in "Plan-time facts" + T3 inert-toggle integration test. Editor-field YAGNI cut is explicit in Global Constraints + surfaced at the user gate; follow-up filed at execution.

**2. Placeholder scan:** none — every step has code or exact anchors + named repo patterns; the two "verify at implementation" notes (RerankingConfig field names, backfill args) name the exact file to check.

**3. Type consistency:** `load_rag_defaults_from_active_profile`/`apply_defaults_to_profile`/`save_rag_defaults_to_active_profile`/`active_profile_info` (T1) consumed by T2-T4 as written; `list_profiles_grouped`/`activate_profile` (T2) used by T2's worker; `index_change_pending`/`fetch_index_status` (T4) referenced in T2's after-set-active hook exactly once, guarded "until Task 4 lands". ✓
