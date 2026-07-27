# TASK-847 / TASK-848 / TASK-849 — implementation report

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-hardening` (branch `feat/agent-path-hardening`)
**Interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, run from this worktree.

All three defects were reproduced first (real, unmocked repro scripts under the scratchpad), confirmed to fail against the pre-fix code, then fixed, then re-confirmed to pass. Full pytest run below.

---

## TASK-847 — `is_sensitive_path` fail-closed on unresolvable paths

**Defect.** `_resolved()` caught only `(OSError, RuntimeError)`. A path containing an embedded NUL byte raises `ValueError` from `Path.resolve()`, which escaped `is_sensitive_path` as an uncaught exception instead of the promised `True`.

**Reproduction (before fix):**
```
$ .venv/bin/python3 - <<'EOF'
from pathlib import Path
from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path
is_sensitive_path(Path("bad\x00path"))
EOF
```
Result: `ValueError: lstat: embedded null character in path` raised out of `is_sensitive_path`, not `True` returned.

**After fix:**
Same script → `is_sensitive_path(...)` returns `True`.

**What changed.** `_resolved()`'s except clause broadened from `(OSError, RuntimeError)` to bare `Exception` (with a debug log, `# noqa: BLE001`, matching the module's existing lazy-accessor pattern). `_resolved()` got a new docstring explaining why the catch must be unconditional: the fail-closed guarantee has to hold for *any* resolution failure, not just the two most common ones. `is_sensitive_path`'s own docstring already stated the guarantee correctly (it was aspirational, not yet true) — no textual change was needed there.

**Derived from:** nothing new to derive; this was a pure implementation-narrower-than-promise gap, fixed at the one function that owns path resolution.

**Test:** `Tests/Utils/test_sensitive_paths.py::test_nul_byte_path_fails_closed_for_real` — real (unmocked) NUL-byte path, alongside the pre-existing `test_unresolvable_path_fails_closed` (which monkeypatches `_resolved` itself and stays useful as a "no matter how it fails" pin).

**Files:** `tldw_chatbook/Utils/sensitive_paths.py`, `Tests/Utils/test_sensitive_paths.py`.

---

## TASK-848 — extend the denylist beyond the active user data folder

Three sub-gaps, all explicitly in scope per the task description and the TASK-846 audit's CRITICAL-3/CRITICAL-4.

### AC#1 — vector-store and sibling-profile paths under a widened sandbox root

**Defect.** `chromadb/chroma.sqlite3` (plaintext chunks of the same conversations/notes `ChaChaNotes.db` protects) and files directly inside `rag_profiles/` (plaintext RAG/embedding-provider config) were reachable once a sandbox root was widened to contain them. `chromadb`/`rag_profiles` are themselves in the existing-directory exemption list (legitimate containers), but nothing refused a *file* sitting directly inside either.

**Reproduction (before fix):** wrote `chroma.sqlite3` under `default_chroma_persist_directory()` and a profile JSON under `default_rag_profiles_dir()`; `is_sensitive_path()` returned `False` for both.

**After fix:** both return `True`; the container directories themselves, and an existing per-collection subdirectory nested inside `chromadb/`, still return `False` (stay reachable).

**Derived from:** `RAG_Search.simplified.config.default_chroma_persist_directory()` (already existed, honors `RAG_PERSIST_DIR`/config overrides — used as-is). `RAG_Search.config_profiles.default_rag_profiles_dir()` — **new**, added because no accessor existed before (only an inline `get_user_data_dir() / "rag_profiles"` literal in `ConfigProfileManager.__init__`); `ConfigProfileManager.__init__` now calls it too, so there is exactly one spelling of the name.

**Mechanism:** generalized the existing "any file directly in `get_user_data_dir()` is refused, existing directories stay reachable" rule (previously hardcoded to one directory) to a new `SensitivePathContext.direct_child_denied_dirs` tuple, populated by `_direct_child_rule_container_dirs()`: `user_data_dir`, the effective config directory, the chroma persist directory, the rag_profiles directory.

**Tests:**
- `Tests/Utils/test_sensitive_paths.py::test_chroma_vector_store_file_is_refused`, `::test_rag_profile_file_is_refused` (unit-level, both directions: file refused, container + nested existing subdir reachable).
- `Tests/Tools/test_file_tool_sandbox.py::test_read_file_refuses_chroma_vector_store_file_even_when_sandbox_root_contains_it` (tool-level, widened root).
- `Tests/Tools/test_file_tool_sandbox.py::test_default_sandbox_configuration_still_works_end_to_end` (pre-existing, unmodified) covers the "default still works" direction — the default sandbox root (`tool_sandbox`) is a sibling of `chromadb`/`rag_profiles`, never touching them.

### AC#2 — skill trust/grant store (audit CRITICAL-3)

**Defect.** `get_user_data_dir() / "skills" / "trust"` (manifest, script grants, generation marker, snapshots) was structurally unreachable by any existing rule: `skills` is an existing-directory exemption by design, so everything nested under it — including `trust/` — inherited that exemption. `skill_script_grants.json` in particular is the plain, unauthenticated JSON file that authorizes script *execution*.

**Reproduction (before fix):** wrote all four (manifest, grants, marker, a snapshot file) under the real trust store dir; all four returned `is_sensitive_path() == False`.

**After fix:** all four return `True`; a sibling file directly in `skills/` (outside `trust/`) still returns `False` (the container exemption still holds everywhere except this one carve-out).

**Derived from:** two **new** small pure accessor functions, added because none existed before (app.py built both directory names as inline literals):
- `Skills_Interop.local_skills_service.default_local_skills_store_dir(user_data_dir)` → `user_data_dir / "skills"`.
- `Skills_Interop.skill_trust_store.default_trust_store_dir(local_skills_store_dir)` → `local_skills_store_dir / "trust"`.

`app.py`'s service-wiring block (previously `local_skills_store_dir = get_user_data_dir() / "skills"` / `trust_store_dir = local_skills_store_dir / "trust"`) now calls both functions, so `sensitive_paths.py` derives the identical path `app.py` uses to build the live `SkillTrustStore` — never a re-spelled `"skills"`/`"trust"` literal. Also promoted the generation-marker filename from an inline `"generation_marker.json"` literal in `app.py` to a public `MARKER_FILENAME` constant in `skill_trust_store.py`, used by both `app.py` and the test.

**Mechanism:** the whole `trust/` subtree is refused by **ancestry** (added to `SensitivePathContext.dirs`, the same list `~/.ssh` etc. use), not just direct children — a file several levels inside `snapshots/` is refused exactly like the manifest itself. This is a deliberate, documented exception carved back out of the `skills/` container exemption (see the module docstring and `_sensitive_skill_trust_dir()`'s own docstring).

**Test:** `Tests/Utils/test_sensitive_paths.py::test_skill_trust_store_paths_are_refused_via_the_actually_used_accessors` (constructs a real `SkillTrustStore` + `FileSkillTrustGenerationMarkerStore`, derives `manifest_path`/`snapshots_dir` from the instance and `_SCRIPT_GRANTS_FILENAME`/`MARKER_FILENAME` from the owning modules — no literal path spelled in the test) and `::test_skills_directory_itself_stays_reachable_outside_the_trust_carve_out`.

### AC#3 — config.toml's `.bak`/`.tmp` sidecars (audit CRITICAL-4)

**Defect.** `UI/Screens/settings_screen.py`'s Advanced config save writes a full plaintext `.bak` backup before overwriting, plus a `.tmp` during the atomic swap — both byte-identical to the live config, API keys included. The denylist named only `_get_effective_config_path()` itself, nothing else in that directory.

**Reproduction (before fix):** wrote `config.toml.bak`, `config.toml.tmp`, and an arbitrarily-named `config.toml.pre-lab-cleanup` (matching what the audit found live on the audit machine) beside the real effective config path; all three returned `False`.

**After fix:** all three return `True`.

**Decision — generalized, not sidecar-suffix-mirrored.** The task asked to consider mirroring `_DB_SIDECAR_SUFFIXES` (`-wal`/`-shm`/`-journal`) *or* applying the user-data-dir's direct-child-file rule to the effective config directory. Chose the **general direct-child-file rule**, not a `.bak`/`.tmp`-suffix enumeration, because the audit's own live evidence showed the problem is broader than two suffixes: `runtime_policy.json`, `ui_state.toml`, and an arbitrarily-named hand-made backup (`config.toml.pre-lab-cleanup`) were all unprotected too, and a suffix enumeration would still miss the next one. The effective config directory was added to the same `direct_child_denied_dirs` set used for AC#1 above — one mechanism, reused, rather than a second parallel one. Existing directories placed directly in the config dir (the real `feed_cache`/`themes`/`tokenizers` this app already writes there) stay reachable via the same "is it a directory" gate.

**Derived from:** `config._get_effective_config_path().parent` — the same accessor the app itself uses (honors `TLDW_CONFIG_PATH`).

**Tests:** `test_config_toml_bak_and_tmp_sidecars_are_refused`, `test_any_other_file_directly_in_the_effective_config_dir_is_refused` (the `.pre-lab-cleanup`-style arbitrary name), `test_existing_directory_directly_in_the_effective_config_dir_stays_reachable`.

### The MCP-filename residual fragility (assessed, not fixed)

The audit noted `sensitive_paths.py` re-spells the three MCP filenames (`mcp_permissions.json`, `local_mcp_store.json`, `mcp_execution_log.jsonl`) as its own literals joined to `get_user_data_dir()`. Investigated: **no single source of truth exists anywhere in the codebase for these three names** — `app.py` (lines ~4028/5241/5248) and `MCP/unified_control_plane_service.py` (`Path(store.path).with_name(...)` at lines ~2073/2430) each independently spell them as inline literals too. Fixing this properly means introducing shared constants across `MCP/local_store.py`, `MCP/unified_context_store.py`, `MCP/server_target_store.py`, `MCP/unified_control_plane_service.py`, and `app.py` — none of which is named in TASK-848's four ACs, and none of which `sensitive_paths.py` alone can fix without touching those other modules' own construction sites. **Decision: left as-is**, documented as a known, deliberately out-of-scope residual fragility (matches the audit's own framing — "one refactor away," not a live bug). Recommend a follow-up task scoped explicitly to the MCP store-path family if this is to be closed.

**Files:** `tldw_chatbook/Utils/sensitive_paths.py`, `tldw_chatbook/Skills_Interop/local_skills_service.py`, `tldw_chatbook/Skills_Interop/skill_trust_store.py`, `tldw_chatbook/Skills_Interop/__init__.py`, `tldw_chatbook/RAG_Search/config_profiles.py`, `tldw_chatbook/app.py`, `Tests/Utils/test_sensitive_paths.py`, `Tests/Tools/test_file_tool_sandbox.py`.

---

## TASK-849 — agent-created directory shadowing a not-yet-created state file

**Defect.** The direct-child-file rule is gated on "is this an existing directory" so a *pre-existing* container stays reachable. But `WriteFileTool`'s `create_directories=True` path only ever validated the **final file** being written (e.g. `search_history.db/note.txt`), never the new directory levels `Path.mkdir(parents=True)` creates on the way there (`search_history.db` itself). Nothing stopped an agent from planting a directory at that exact name before the app ever created `search_history.db` as a SQLite file; the app's later `sqlite3.connect(...)` on that path then fails outright — a denial of service, no disclosure.

**Reproduction (before fix), full chain:**
```python
# widened sandbox root == get_user_data_dir()
await WriteFileTool().execute(
    file_path="search_history.db/note.txt", content="hello", create_directories=True,
)
# -> {'action': 'created', ...}; search_history.db/ now exists as a DIRECTORY.
import sqlite3; sqlite3.connect(str(shadow_dir))
# -> sqlite3.OperationalError: unable to open database file
```
Also confirmed directly: `is_sensitive_path(user_data_dir / "search_history.db")` returned `False` once that directory existed.

**After fix:** the same `WriteFileTool.execute` call returns `{'error': "Refused: creating '.../search_history.db' would collide with a protected path"}`; the directory is never created.

**What changed.** Added `refuses_new_directory_chain(target_dir, context=None)` to `sensitive_paths.py`: walks upward from `target_dir` while each level does not yet exist, calling `is_sensitive_path` on each (reusing the *existing* direct-child-file rule verbatim, never a separate check or a loosened gate), stopping at the first already-existing ancestor (mirroring exactly what `Path.mkdir(parents=True)` would actually create). `WriteFileTool.execute` now calls it immediately before `path.parent.mkdir(parents=True, exist_ok=True)` and refuses if it returns `True`.

Critically, `is_sensitive_path`'s own "is an existing directory" gate was **not** changed — an already-existing `search_history.db/` directory (predating this fix, or created by any other means) still reads as an ordinary container, by design: there is no name-independent way to tell "legitimate pre-existing container" from "illegitimate pre-existing collision," and the whole point of the existing-directory gate is to avoid a name enumeration. The fix closes the hole at **creation time** instead, which is also the only place the agent tools can actually cause the collision.

**Reconciliation with containers.** Verified both directions end to end through the real `WriteFileTool`:
- legitimate: writing into a brand-new subdirectory nested inside an *already-existing* container (`tool_sandbox/brand_new_subdir/note.txt`) still succeeds, under both the default sandbox root and a widened one.
- legitimate: the real, unmocked default sandbox configuration (`get_user_data_dir()/tool_sandbox`, a sibling of `search_history.db`) still works end to end for write/read/list/glob/grep (pre-existing test, unmodified, still passes).
- illegitimate: the exact `search_history.db/note.txt` collision is refused before any directory is created.

**Tests:**
- `Tests/Utils/test_sensitive_paths.py::test_refuses_new_directory_chain_blocks_the_collision` (multi-level: collision is one level above the leaf target, not at the leaf itself), `::test_refuses_new_directory_chain_allows_existing_containers`, `::test_an_already_existing_directory_still_reads_as_an_ordinary_container` (documents the deliberate is-dir-gate boundary above).
- `Tests/Tools/test_file_tool_sandbox.py::test_write_file_refuses_to_plant_a_directory_shadowing_a_state_file` (tool-level, end to end), `::test_write_file_still_creates_legitimate_new_nested_directories` (companion "still works" test, same widened-root config).

**Files:** `tldw_chatbook/Utils/sensitive_paths.py`, `tldw_chatbook/Tools/file_operation_tools.py`, `Tests/Utils/test_sensitive_paths.py`, `Tests/Tools/test_file_tool_sandbox.py`.

---

## Test run

```
cd /Users/macbook-dev/Documents/GitHub/wt-path-hardening
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q
```

First run: 1 failed (my own `test_directory_shadowing_a_not_yet_created_state_file_is_refused`, which incorrectly asserted `is_sensitive_path` should flag an *already-existing* directory — inconsistent with the deliberate "existing directory stays reachable" design; rewritten to `test_an_already_existing_directory_still_reads_as_an_ordinary_container` asserting the correct, unchanged behavior), 892 passed, in 232.84s.

Second run (after the test fix, no production-code changes): **893 passed, 0 failed, 12 warnings, in 230.56s.**

No pre-existing baseline failures encountered beyond the ones already known (`pytest-mock`/`numpy` absent; six `test_chat_api_key_*` failures in `Tests/UI/test_tools_settings_window.py` are out of the run scope for this task — not in `Tests/Utils/`, `Tests/Tools/`, or `Tests/Agents/`).
