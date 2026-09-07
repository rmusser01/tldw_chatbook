# SP2a — Profile System: Storage Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing `ConfigProfileManager` into a *correct*, file-backed, user-facing profile store: fix the broken serialization round-trip, fix the builtins that silently ignore their own settings, mark builtins read-only, give profiles a rename-safe stable id, store one file per user profile, and add real CRUD.

**Architecture:** All changes live in `tldw_chatbook/RAG_Search/config_profiles.py`. `ProfileConfig` gains a stable `id` and a `read_only` flag. `ProfileConfig.from_dict` routes `rag_config` through `RAGConfig.from_dict` (proper nested-dataclass reconstruction). The ~13 builtins are corrected to the real dataclass field names + valid enum values and marked `read_only`. The single `custom_profiles.json` blob is replaced by one JSON file per user profile under the existing `rag_profiles/` dir (legacy blob auto-migrated). CRUD (`save_profile`/`delete_profile`/`rename_profile`/`clone_profile`) guards against mutating builtins.

**Tech Stack:** Python 3.11+, existing `ProfileConfig`/`RAGConfig` dataclasses, `json`, pytest. Tests pass an explicit temp `profiles_dir` to `ConfigProfileManager(...)` (note: `get_profile_manager()` re-instantiates per call — it is NOT a singleton — so tests construct the manager directly with a temp dir).

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md`; overview `Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md`. This plan is **SP2a only** — the storage foundation.
- **OUT OF SCOPE (this is SP2b, a later PR):** the config-resolution unification (`resolve_active_rag_config`), routing `config.py`'s loader through the active profile, deprecating `AppRAGSearchConfig.rag.*` value keys, the active-profile pointer wiring, `reset_shared_rag_service()`, the first-run "Imported settings" import, and the embed==query parity test. **Do NOT touch `config.py`'s loader, `ingestion_indexing.py`, `rag_factory.py`'s resolution, or `[rag.service].profile` in SP2a.**
- **Owner decision:** builtins are fixed to use the REAL fields (not left as-is), so fast/balanced/accuracy actually differ.
- **Real dataclass fields (verbatim):** `ChunkingConfig.chunk_size`, `.chunk_overlap`, `.chunking_method` (NOT `.size`/`.overlap`). `SearchConfig.default_top_k`, `.default_search_mode` (values `"plain"|"semantic"|"hybrid"`), `.score_threshold` (NOT `.top_k`/`.default_type`). `VectorStoreConfig.type` ∈ `{"chroma","memory","auto"}` (NOT `"in_memory"`).
- **Value mapping for `default_type` → `default_search_mode`:** `"keyword"→"plain"`, `"semantic"→"semantic"`, `"hybrid"→"hybrid"`. This is a value map, not a blind rename.
- **No DB migration** (profiles are files). **Builtins are never written to disk.**
- **Persistence lives together:** all edits in `config_profiles.py`; new tests in `Tests/RAG/test_config_profiles.py`.
- Google-style docstrings, type hints on public methods, snake_case.

## Plan-time facts (verified)
- `ProfileConfig.from_dict` is at `config_profiles.py:114-151`; the bug is lines 117-121 (`RAGConfig(**data["rag_config"])`).
- `RAGConfig.from_dict` (`config.py:395-430`) rebuilds nested dataclasses correctly — route through it.
- `asdict()` only emits real dataclass fields, so `to_dict()` already drops the builtins' dead attributes; the builtin fix is independent of the round-trip fix.
- Builtins set dead attrs at (approx) lines 176/178-181, 198-201, 218-221, 298-300, 319-322, 349-351, 367-371.
- `validate_profile` reads dead attrs at `config_profiles.py:731` (`.chunking.overlap >= .chunking.size`) and `:738` (`.search.top_k`).
- `_save_custom_profiles` (blob writer) at `config_profiles.py:527-545`; `_load_custom_profiles` at `:449-467`; blob path `profiles_dir/"custom_profiles.json"`.
- `create_custom_profile` at `:477-525` keys `_profiles` by `name.lower().replace(" ","_")` (the de-facto slug).
- **Note (flag for SP2b, do NOT fix here):** `rag_factory.py:95-99`'s auto-detect reads `config.search.default_type` — a phantom field. After the builtin fix that attr no longer exists, so `hasattr(config.search,"default_type")` is `False` and that branch simply never fires (no crash). Leave it; SP2b's resolution work owns it.

---

## File Structure
- **Modify** `tldw_chatbook/RAG_Search/config_profiles.py` — `ProfileConfig` (add `id`, `read_only`; fix `from_dict`); builtins (field names + values + `read_only=True` + explicit `id`); `validate_profile` (real fields); storage (`_load_custom_profiles`/`_save_custom_profiles` → per-file + legacy migration); CRUD (`save_profile`/`delete_profile`/`rename_profile`/`clone_profile`).
- **Create** `Tests/RAG/test_config_profiles.py` — all SP2a tests.

---

## Task 1: Fix the ProfileConfig serialization round-trip

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py:114-151` (`ProfileConfig.from_dict`)
- Test: `Tests/RAG/test_config_profiles.py`

**Interfaces:**
- Produces: `ProfileConfig.from_dict(data)` returns a profile whose `.rag_config` is a real `RAGConfig` with real sub-dataclass instances (not raw dicts).

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/test_config_profiles.py
import json
from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig,
)


def _profile(**over):
    rag = RAGConfig(
        embedding=EmbeddingConfig(model="round-trip-model"),
        chunking=ChunkingConfig(chunk_size=333, chunk_overlap=77),
    )
    return ProfileConfig(name="RT", description="d", profile_type="custom", rag_config=rag)


def test_round_trip_reconstructs_nested_dataclasses():
    p = _profile()
    restored = ProfileConfig.from_dict(json.loads(json.dumps(p.to_dict())))
    # These attribute accesses raise AttributeError today (sub-configs are dicts).
    assert isinstance(restored.rag_config, RAGConfig)
    assert isinstance(restored.rag_config.embedding, EmbeddingConfig)
    assert isinstance(restored.rag_config.chunking, ChunkingConfig)
    assert restored.rag_config.embedding.model == "round-trip-model"
    assert restored.rag_config.chunking.chunk_size == 333
    assert restored.rag_config.chunking.chunk_overlap == 77
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_config_profiles.py::test_round_trip_reconstructs_nested_dataclasses -q`
Expected: FAIL — `AttributeError: 'dict' object has no attribute 'model'` (sub-config is a raw dict).

- [ ] **Step 3: Fix `from_dict`**

In `config_profiles.py`, replace the `rag_config` reconstruction (lines 117-121):
```python
        rag_config = (
            RAGConfig.from_dict(data["rag_config"])
            if isinstance(data["rag_config"], dict)
            else data["rag_config"]
        )
```
(Leave the `reranking_config`/`processing_config` branches as-is — those are flat dataclasses; verify at implementation that `RerankingConfig`/`ProcessingConfig` have no nested dataclass fields. If either DOES nest, give it the same `.from_dict` treatment.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_config_profiles.py::test_round_trip_reconstructs_nested_dataclasses -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py Tests/RAG/test_config_profiles.py
git commit -m "fix(rag): ProfileConfig round-trip reconstructs nested dataclasses"
```

---

## Task 2: Fix the builtin profiles (real fields + valid enum values) + validate_profile

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` — `_load_builtin_profiles` (all builtins) and `validate_profile` (~:731,:738)
- Test: `Tests/RAG/test_config_profiles.py` (append)

**Interfaces:**
- Produces: builtins whose declared chunk/overlap/top_k/mode actually take effect on the real dataclass fields.

- [ ] **Step 1: Write the failing test (append)**

```python
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager


def _mgr(tmp_path):
    return ConfigProfileManager(profiles_dir=tmp_path / "profiles")


def test_builtins_apply_declared_settings(tmp_path):
    m = _mgr(tmp_path)
    fast = m.get_profile("fast_search")
    assert fast.rag_config.chunking.chunk_size == 256      # was silently 400 (dead attr)
    assert fast.rag_config.chunking.chunk_overlap == 32
    assert fast.rag_config.search.default_top_k == 5

    # Builtins are meaningfully differentiated, not all default:
    sizes = {name: m.get_profile(name).rag_config.chunking.chunk_size
             for name in ("fast_search", "high_accuracy", "long_context")}
    assert len(set(sizes.values())) == 3, sizes


def test_builtins_use_valid_search_mode_and_store_type(tmp_path):
    m = _mgr(tmp_path)
    valid_modes = {"plain", "semantic", "hybrid"}
    valid_types = {"chroma", "memory", "auto"}
    for name in m.list_profiles():
        p = m.get_profile(name)
        assert p.rag_config.search.default_search_mode in valid_modes, name
        assert p.rag_config.vector_store.type in valid_types, name
    # bm25_only was keyword-only -> plain (value-mapped, not "keyword")
    assert m.get_profile("bm25_only").rag_config.search.default_search_mode == "plain"


def test_validate_profile_reads_real_fields(tmp_path):
    m = _mgr(tmp_path)
    bad = m.get_profile("hybrid_basic")
    # Force an overlap >= size on the REAL fields; validate must catch it.
    bad.rag_config.chunking.chunk_overlap = bad.rag_config.chunking.chunk_size + 1
    warnings = m.validate_profile(bad)
    assert any("overlap" in w.lower() for w in warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_config_profiles.py -q -k builtins or validate`
Expected: FAIL — `fast_search` chunk_size is 400 (default), modes/store-type invalid, validate reads dead attrs.

- [ ] **Step 3: Fix every builtin + validate_profile**

In `_load_builtin_profiles`, apply these substitutions to EVERY builtin (there are ~13; do not miss any — search the method for each dead pattern):
- `.chunking.size = N`   → `.chunking.chunk_size = N`
- `.chunking.overlap = N` → `.chunking.chunk_overlap = N`
- `.search.top_k = N`     → `.search.default_top_k = N`
- `.search.default_type = "keyword"` → `.search.default_search_mode = "plain"`
- `.search.default_type = "semantic"` → `.search.default_search_mode = "semantic"`
- `.search.default_type = "hybrid"`  → `.search.default_search_mode = "hybrid"`
- `.vector_store.type = "in_memory"` → `.vector_store.type = "memory"`

Leave builtins that already use the correct fields (`hybrid_enhanced`, `hybrid_full` use `chunk_size`/`default_search_mode`/`default_top_k`) unchanged.

In `validate_profile` (~:731, :738):
```python
        if profile.rag_config.chunking.chunk_overlap >= profile.rag_config.chunking.chunk_size:
            warnings.append("Chunk overlap should be less than chunk size")
```
```python
                > profile.rag_config.search.default_top_k
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_config_profiles.py -q`
Expected: PASS. Then grep to confirm no dead attrs remain:
`grep -nE "\.chunking\.(size|overlap)\b|\.search\.(top_k|default_type)\b|\"in_memory\"" tldw_chatbook/RAG_Search/config_profiles.py` → no matches.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py Tests/RAG/test_config_profiles.py
git commit -m "fix(rag): builtin profiles use real dataclass fields + valid enum values"
```

---

## Task 3: Stable `id` + read-only builtin marker on ProfileConfig

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` — `ProfileConfig` dataclass, its `to_dict`/`from_dict`, `_load_builtin_profiles` (stamp `id`+`read_only`), add a `_slugify` helper.
- Test: `Tests/RAG/test_config_profiles.py` (append)

**Interfaces:**
- Produces:
  - `ProfileConfig.id: str` (stable, filename-safe; backfilled from name-slug when absent).
  - `ProfileConfig.read_only: bool` (True for builtins).
  - `_slugify(name: str) -> str` (module-level helper).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_builtins_are_read_only_with_ids(tmp_path):
    m = _mgr(tmp_path)
    hb = m.get_profile("hybrid_basic")
    assert hb.read_only is True
    assert hb.id == "hybrid_basic"


def test_profileconfig_id_backfilled_and_round_trips():
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, _slugify
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    p = ProfileConfig(name="My Cool Profile", description="d",
                      profile_type="custom", rag_config=RAGConfig())
    assert p.id == _slugify("My Cool Profile")  # auto-derived when not given
    import json
    restored = ProfileConfig.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.id == p.id
    assert restored.read_only is False
    # Old files with no id/read_only keys still load (backfill):
    legacy = {k: v for k, v in p.to_dict().items() if k not in ("id", "read_only")}
    legacy_restored = ProfileConfig.from_dict(legacy)
    assert legacy_restored.id == _slugify("My Cool Profile")
    assert legacy_restored.read_only is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_config_profiles.py -q -k "read_only or id_backfilled"`
Expected: FAIL — `ProfileConfig` has no `id`/`read_only`, no `_slugify`.

- [ ] **Step 3: Implement**

Add a module-level helper near the top of `config_profiles.py`:
```python
import re


def _slugify(name: str) -> str:
    """Filename-safe stable slug for a profile display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "profile"
```

Add fields to the `ProfileConfig` dataclass (after `profile_type` / before component configs is fine; keep dataclass-default ordering valid — all new fields have defaults so put them after the last existing field or with defaults):
```python
    id: Optional[str] = None
    read_only: bool = False
```
In `ProfileConfig.__post_init__` (add one if absent):
```python
    def __post_init__(self):
        if not self.id:
            self.id = _slugify(self.name)
```
In `to_dict`, add `"id": self.id, "read_only": self.read_only,`.
In `from_dict`'s final `cls(...)` call, add `id=data.get("id"), read_only=data.get("read_only", False),` (the `__post_init__` backfills id when the key is missing/None).

In `_load_builtin_profiles`, stamp every builtin with `read_only=True` and an explicit `id` equal to its `_profiles[...]` key (e.g. `id="hybrid_basic", read_only=True`). Key `self._profiles` by that same id.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_config_profiles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py Tests/RAG/test_config_profiles.py
git commit -m "feat(rag): ProfileConfig stable id + read_only builtin marker"
```

---

## Task 4: File-per-profile storage + legacy-blob migration

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` — `_load_custom_profiles` (`:449-467`), `_save_custom_profiles` (`:527-545`), add `_profile_path(id)` + `_migrate_legacy_blob()`.
- Test: `Tests/RAG/test_config_profiles.py` (append)

**Interfaces:**
- Consumes: `ProfileConfig.id`, `read_only` (Task 3); `_slugify` (Task 3).
- Produces:
  - `_profile_path(profile_id: str) -> Path` → `profiles_dir/f"{profile_id}.json"`.
  - `_save_one(profile: ProfileConfig) -> None` → writes that profile's file.
  - `_load_custom_profiles()` reads every `*.json` user-profile file (skips reserved names) and one-time migrates a legacy `custom_profiles.json` blob into per-file, then renames the blob to `custom_profiles.json.migrated`.

- [ ] **Step 1: Write the failing test (append)**

```python
import json as _json


def test_user_profile_saved_as_own_file(tmp_path):
    m = _mgr(tmp_path)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    p = ProfileConfig(name="Sales RAG", description="d", profile_type="custom",
                      rag_config=RAGConfig())
    m.save_profile(p)  # (Task 5 adds save_profile; if running Task 4 first, call m._save_one(p))
    assert (tmp_path / "profiles" / f"{p.id}.json").exists()
    # A fresh manager over the same dir loads it back, correctly:
    m2 = _mgr(tmp_path)
    loaded = m2.get_profile(p.id)
    assert loaded is not None and loaded.read_only is False
    assert isinstance(loaded.rag_config, RAGConfig)


def test_legacy_blob_migrated_to_per_file(tmp_path):
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    legacy = ProfileConfig(name="Legacy One", description="d",
                           profile_type="custom", rag_config=RAGConfig())
    (pdir / "custom_profiles.json").write_text(
        _json.dumps({"profiles": [legacy.to_dict()]}))
    m = _mgr(tmp_path)  # construction triggers load+migrate
    assert m.get_profile(legacy.id) is not None
    assert (pdir / f"{legacy.id}.json").exists()
    assert (pdir / "custom_profiles.json.migrated").exists()
    assert not (pdir / "custom_profiles.json").exists()
    # Idempotent: a second manager doesn't choke on the already-migrated blob.
    _mgr(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_config_profiles.py -q -k "own_file or legacy_blob"`
Expected: FAIL — no per-file storage / migration yet.

- [ ] **Step 3: Implement**

Add helpers and rewrite the storage methods:
```python
_RESERVED_PROFILE_FILES = {"custom_profiles.json", "custom_profiles.json.migrated"}


    def _profile_path(self, profile_id: str) -> Path:
        return self.profiles_dir / f"{profile_id}.json"

    def _save_one(self, profile: "ProfileConfig") -> None:
        """Write a single user profile to its own file (never builtins)."""
        if profile.read_only:
            return
        with open(self._profile_path(profile.id), "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def _load_custom_profiles(self):
        """Load user profiles from per-file JSON; migrate a legacy blob once."""
        self._migrate_legacy_blob()
        for path in self.profiles_dir.glob("*.json"):
            if path.name in _RESERVED_PROFILE_FILES:
                continue
            try:
                with open(path, "r") as f:
                    profile = ProfileConfig.from_dict(json.load(f))
                profile.read_only = False
                self._profiles[profile.id] = profile
            except Exception as e:
                logger.error(f"Failed to load profile {path.name}: {e}")

    def _migrate_legacy_blob(self):
        """One-time split of the old custom_profiles.json blob into per-file."""
        blob = self.profiles_dir / "custom_profiles.json"
        if not blob.exists():
            return
        try:
            with open(blob, "r") as f:
                data = json.load(f)
            for pdata in data.get("profiles", []):
                profile = ProfileConfig.from_dict(pdata)
                profile.read_only = False
                target = self._profile_path(profile.id)
                if not target.exists():  # never clobber an existing per-file profile
                    with open(target, "w") as out:
                        json.dump(profile.to_dict(), out, indent=2)
            blob.rename(self.profiles_dir / "custom_profiles.json.migrated")
            logger.info("Migrated legacy custom_profiles.json to per-file profiles")
        except Exception as e:
            logger.error(f"Legacy profile blob migration failed: {e}")
```
Delete the old `_save_custom_profiles` blob writer, and replace its call sites (in `create_custom_profile` at `:521`) with `self._save_one(custom_config)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_config_profiles.py -q`
Expected: PASS (the `own_file` test's `save_profile` lands in Task 5; if Task 5 not yet done, temporarily call `_save_one` — but implement Task 5 next so the final suite is green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py Tests/RAG/test_config_profiles.py
git commit -m "feat(rag): file-per-profile storage + legacy blob migration"
```

---

## Task 5: CRUD with read-only guards

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` — add `save_profile`, `delete_profile`, `rename_profile`, `clone_profile`.
- Test: `Tests/RAG/test_config_profiles.py` (append)

**Interfaces:**
- Consumes: `_save_one`, `_profile_path` (Task 4); `_slugify`, `id`, `read_only` (Task 3).
- Produces:
  - `save_profile(profile) -> ProfileConfig` (refuses read-only; writes file; registers).
  - `delete_profile(profile_id) -> bool` (refuses builtins; removes file + registry entry).
  - `rename_profile(profile_id, new_name) -> ProfileConfig` (changes display name, KEEPS id/file).
  - `clone_profile(source_id, new_name) -> ProfileConfig` (deep-copies any profile — incl. a builtin — into a new writable user profile with a fresh unique id).

- [ ] **Step 1: Write the failing test (append)**

```python
import pytest


def test_cannot_mutate_builtins(tmp_path):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.delete_profile("hybrid_basic")
    with pytest.raises(ValueError):
        m.rename_profile("hybrid_basic", "Nope")
    with pytest.raises(ValueError):
        m.save_profile(m.get_profile("hybrid_basic"))  # read_only


def test_clone_builtin_creates_writable_copy(tmp_path):
    m = _mgr(tmp_path)
    clone = m.clone_profile("high_accuracy", "My Accuracy")
    assert clone.read_only is False
    assert clone.id != "high_accuracy"
    assert clone.rag_config.chunking.chunk_size == m.get_profile("high_accuracy").rag_config.chunking.chunk_size
    assert (tmp_path / "profiles" / f"{clone.id}.json").exists()
    # Editing the clone does not touch the builtin:
    clone.rag_config.chunking.chunk_size = 111
    m.save_profile(clone)
    assert m.get_profile("high_accuracy").rag_config.chunking.chunk_size != 111


def test_rename_keeps_id_and_file(tmp_path):
    m = _mgr(tmp_path)
    c = m.clone_profile("hybrid_basic", "Before")
    old_id = c.id
    renamed = m.rename_profile(c.id, "After")
    assert renamed.id == old_id                 # id stable across rename
    assert renamed.name == "After"
    assert (tmp_path / "profiles" / f"{old_id}.json").exists()


def test_delete_removes_file_and_entry(tmp_path):
    m = _mgr(tmp_path)
    c = m.clone_profile("hybrid_basic", "Temp")
    assert m.delete_profile(c.id) is True
    assert m.get_profile(c.id) is None
    assert not (tmp_path / "profiles" / f"{c.id}.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/test_config_profiles.py -q -k "mutate_builtins or clone_builtin or rename_keeps or delete_removes"`
Expected: FAIL — CRUD methods don't exist.

- [ ] **Step 3: Implement**

```python
    def _unique_id(self, base_slug: str) -> str:
        candidate, n = base_slug, 2
        while candidate in self._profiles:
            candidate = f"{base_slug}_{n}"
            n += 1
        return candidate

    def save_profile(self, profile: "ProfileConfig") -> "ProfileConfig":
        """Persist a user profile (refuses read-only builtins)."""
        if profile.read_only:
            raise ValueError(f"Profile '{profile.id}' is read-only")
        self._profiles[profile.id] = profile
        self._save_one(profile)
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        prof = self._profiles.get(profile_id)
        if prof is None:
            return False
        if prof.read_only:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be deleted")
        self._profiles.pop(profile_id, None)
        path = self._profile_path(profile_id)
        if path.exists():
            path.unlink()
        return True

    def rename_profile(self, profile_id: str, new_name: str) -> "ProfileConfig":
        prof = self._profiles.get(profile_id)
        if prof is None:
            raise ValueError(f"Profile '{profile_id}' not found")
        if prof.read_only:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be renamed")
        prof.name = new_name  # id + filename stay the same (rename-safe)
        self._save_one(prof)
        return prof

    def clone_profile(self, source_id: str, new_name: str) -> "ProfileConfig":
        src = self._profiles.get(source_id)
        if src is None:
            raise ValueError(f"Source profile '{source_id}' not found")
        new_id = self._unique_id(_slugify(new_name))
        clone = ProfileConfig.from_dict({
            **src.to_dict(),
            "id": new_id,
            "name": new_name,
            "read_only": False,
            "profile_type": "custom",
        })
        return self.save_profile(clone)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/test_config_profiles.py -q`
Expected: PASS (whole file).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py Tests/RAG/test_config_profiles.py
git commit -m "feat(rag): profile CRUD (save/delete/rename/clone) with read-only guards"
```

---

## Task 6: Regression gate + docstring

**Files:**
- Modify: `tldw_chatbook/RAG_Search/config_profiles.py` (module/class docstring note only)
- Test: run the broader RAG suite

- [ ] **Step 1: Run the RAG suites that touch profiles**

Run: `pytest Tests/RAG/ -q`
Expected: PASS at/above baseline. Because `create_rag_service` resolves builtins via this manager, confirm `Tests/RAG/simplified/` (service construction) still passes — the builtin field-name fix changes builtin chunk sizes, which changes their SP1 fingerprints; that is expected and must not fail any test that asserts a *specific* collection name for a builtin (update such an assertion to the new value if any surface, do not revert the builtin fix).

- [ ] **Step 2: Add a class docstring note to `ConfigProfileManager`**

```python
    """Manages RAG configuration profiles.

    Builtins are read-only clone-seeds; user profiles are stored one JSON file
    per profile under ``rag_profiles/`` keyed by a stable ``id`` (rename-safe).
    NOTE: the active-profile pointer and config-resolution wiring live in SP2b,
    not here — this class is storage + CRUD only.
    """
```

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/RAG_Search/config_profiles.py
git commit -m "docs(rag): document profile storage model (SP2a)"
```

---

## Self-Review

**1. Spec coverage** (`2026-07-21-rag-profile-system-design.md`, SP2a slice):
- §1 profile unit is `ProfileConfig` → used throughout. ✓
- §2 file-per-profile + legacy migration + stable id vs rename → Tasks 3,4,5. ✓
- §3 read-only builtins (~13), clone-to-edit → Tasks 3,5. ✓
- Round-trip correctness (implied by "file-per-profile" working) → Task 1. ✓
- Builtin correctness (owner decision to fix) → Task 2. ✓
- §4 single active pointer, §5 config unification, §6 first-run import, §7 switch/reset, parity → **SP2b (out of scope, explicitly excluded in Global Constraints).** ✓
- §8 testing: round-trip, builtin immutability, clone→editable, stable-id survives rename, slug disambiguation (`_unique_id`), legacy-blob idempotent → Tasks 1-5. ✓

**2. Placeholder scan:** every code step has full code; every test has assertions. The Task-4 `own_file` test references `save_profile` (Task 5) — flagged inline with the `_save_one` fallback; final suite green after Task 5. ✓

**3. Type consistency:** `ProfileConfig.id`/`read_only`, `_slugify`, `_profile_path`/`_save_one`, `save_profile`/`delete_profile`/`rename_profile`/`clone_profile`, `_unique_id` — names identical across the tasks that define and consume them. `RAGConfig.from_dict` used consistently. ✓

**Deferred/flagged for SP2b (not gaps):** `rag_factory.py:95` phantom `default_type` auto-detect; the active pointer + `resolve_active_rag_config` + `reset_shared_rag_service` + first-run import + parity test.
