# TASK-853: Fix the two skills-trust path containment checks that can't actually reject anything

## Summary

Two independent containment checks in the skills-trust code were true by construction and rejected
nothing:

1. `Skills_Interop/skill_trust_store.py`'s `FileSkillTrustGenerationMarkerStore` validated the
   marker path against `base_dir=self.marker_path.parent` -- the candidate's own parent -- so
   `get_safe_relative_path(candidate, base)` could never be `None`.
2. `Skills_Interop/local_skills_service.py`'s `_is_unsafe_scratch_root` only checked "root nested
   inside container", never "root encloses container", and `self.store_dir` (which holds the skill
   index directly) wasn't even in the container list.

Both are now fixed. `tldw_chatbook/Skills_Interop/skill_trust_store.py`,
`tldw_chatbook/Skills_Interop/local_skills_service.py`, and `tldw_chatbook/app.py` were changed,
plus 13 test files (call sites of the now-required `store_dir` constructor argument, and new
containment tests).

## Check #1 -- skill_trust_store.py marker-path validation

### What was passed as the containment base, and why

Added an explicit `store_dir: Path` field to `FileSkillTrustGenerationMarkerStore` (no default --
every caller must supply the trust store's own real directory). `load_marker`/`save_marker` now
call `_validated_trust_file_path(self.marker_path, base_dir=self.store_dir)` instead of
`base_dir=self.marker_path.parent`. `build_skill_trust_marker_store_with_fallback` gained a
`store_dir` parameter it threads into the fallback `FileSkillTrustGenerationMarkerStore`; `app.py`'s
live construction site passes `store_dir=trust_store_dir` -- the exact same directory already used
to build `SkillTrustStore(store_dir=trust_store_dir, ...)`, so the marker's containment base is now
an independent fact about the real trust store, not something re-derived from the candidate path
itself.

### Reproduction -- before

```
$ .venv/bin/python -c "
from tldw_chatbook.Skills_Interop.skill_trust_store import FileSkillTrustGenerationMarkerStore
... build a marker at <tmp>/totally-unrelated-elsewhere/marker.json ...
marker = FileSkillTrustGenerationMarkerStore(marker_path)   # old 1-arg constructor
marker.save_marker(generation=1, manifest_digest='deadbeef')
"
PRE-FIX: save_marker OUTSIDE the trust store SUCCEEDED (vulnerable) -> .../totally-unrelated-elsewhere/marker.json exists: True
```

(Obtained by `git stash push` on the three touched production files, running the repro, then
`git stash pop` to restore the fix.)

### Reproduction -- after

```
marker = FileSkillTrustGenerationMarkerStore(marker_path, store_dir=store_dir)  # store_dir = <tmp>/trust
marker.save_marker(generation=1, manifest_digest='deadbeef')
```
```
POST-FIX: rejected as expected -> unsafe skill trust path
```

A legitimate, co-located marker (`store_dir / "marker.json"`) still round-trips:
```
POST-FIX legitimate co-located marker round-trips: {'generation': 1, 'manifest_digest': 'cafebabe'}
```

New regression test: `Tests/Skills/test_skill_trust_store.py::test_file_marker_store_rejects_marker_outside_store_dir`.

## Check #2 -- local_skills_service.py `_is_unsafe_scratch_root`

### What changed and why

`_unsafe_scratch_root_containers` now returns `[self.store_dir, self.skills_dir, <trust store dir
if wired>]` (previously omitted `self.store_dir`). `_is_unsafe_scratch_root` now checks BOTH
directions per container:

```python
return any(
    get_safe_relative_path(root, container) is not None
    or get_safe_relative_path(container, root) is not None
    for container in containers
)
```

`self.store_dir` was added to the container list per the audit's explicit recommendation (AC #2),
even though `skills_dir`/`trust_store_dir` are nested under it and would already be caught
transitively by the new "root encloses container" direction -- `store_dir` also holds the skill
index file directly, so it's listed for clarity and defense in depth.

### Reproduction -- before

```
$ .venv/bin/python -c "
... build LocalSkillsService(store_dir=<tmp>/user_data, trust_service=...) ...
service._is_unsafe_scratch_root(<tmp>/user_data/skills/nested-scratch)   # nested inside skills_dir
service._is_unsafe_scratch_root(<tmp>/user_data)                        # encloses skills_dir + trust_dir
service._is_unsafe_scratch_root(<tmp>)                                   # encloses store_dir
"
nested-inside root flagged unsafe (expect True): True
enclosing (containing) root flagged unsafe (expect True, but PRE-FIX is False): False
grandparent (further enclosing) root flagged unsafe (expect True, but PRE-FIX is False): False
```

### Reproduction -- after

```
nested-inside root flagged unsafe (expect True): True
enclosing (containing) root flagged unsafe (expect True): True
grandparent (further enclosing) root flagged unsafe (expect True): True
legitimate unrelated root flagged unsafe (expect False): False
```

New regression test: `Tests/Skills/test_skill_script_service.py::test_is_unsafe_scratch_root_rejects_both_containment_directions`
(exercises both directions from the real `script_service` fixture's `store_dir`/`skills_dir`/
`trust_service.trust_store.store_dir`, plus confirms a genuinely unrelated root is still accepted).

## Legitimate-operation regression found and fixed

`Tests/Skills/test_skill_script_service.py::test_scratch_root_config_knob_is_reachable` broke after
the check #2 fix: its `custom_root = tmp_path / "custom-scratch"` happened to be nested directly
under `LocalSkillsService.store_dir`, because that fixture's `store_dir` resolves to the same
`tmp_path` the test also uses (`script_service` -> `make_trust_service` -> `LocalSkillsService(
store_dir=trust.skills_dir.parent)` == `tmp_path`). Once `store_dir` became a protected container,
that sibling directory was (correctly) rejected -- a fixture artifact, not a real production
scenario a user would hit. Fixed by moving `custom_root` onto a genuinely unrelated
`tmp_path_factory.mktemp("custom-scratch")` directory, matching the realistic shape of a real
`[skills] script_scratch_root` value. Test passes again post-fix, confirming a legitimate configured
scratch root is still honored.

## Test call-site migration

`FileSkillTrustGenerationMarkerStore.store_dir` has no default, so every direct construction site
needed updating (~28 across 13 files: `Tests/conftest.py`, `Tests/Utils/test_sensitive_paths.py`,
and 11 files under `Tests/Skills/`). Per the audit's "derive, don't re-spell" theme, each was updated
to re-derive `store_dir` from the marker path already in scope (e.g. `marker_path.parent`, or the
already-named `trust_store_dir`/`tmp_path` variable), rather than hardcoding a fresh literal. The
one deliberate exception is the existing symlink-escape test
(`test_file_marker_store_rejects_marker_parent_symlink_escape`), which passes `store_dir=marker_parent`
(the symlinked directory itself) to preserve its original intent of testing a symlinked containment
base.

Verified with a script that parses every `FileSkillTrustGenerationMarkerStore(...)` call site and
confirms `store_dir` appears in the argument list -- zero misses.

## Tests run

```
.venv/bin/python -m pytest Tests/Skills Tests/Utils/test_sensitive_paths.py Tests/Library/test_skill_script_grant_panel.py -q
```
Result: **411 passed**, 1 pre-existing `RequestsDependencyWarning`, 0 failures, in ~151s.

(Interim runs during the fix: `Tests/Skills` alone 377 passed; `Tests/Skills/test_skill_trust_store.py
+ test_local_skills_service.py + Tests/Utils/test_sensitive_paths.py` 76 passed;
`Tests/Skills/test_skill_script_service.py` 37 passed; `Tests/Library/test_skill_script_grant_panel.py`
3 passed.)

No pre-existing failures in scope were touched (`test_tools_settings_window.py`'s six
`test_chat_api_key_*` failures and the `pytest-mock`/`numpy`-dependent tests were not run, per the
stated baselines).

## Follow-up task filed

`backlog/tasks/task-900 - Tools_Settings_Window-raw-TOML-save-writes-the-wrong-config-path-non-atomically.md`
(TASK-900) -- `UI/Tools_Settings_Window.py::_save_raw_toml_config` has the same wrong-config-path
(`DEFAULT_CONFIG_PATH` literal instead of `config._get_effective_config_path()`) and non-atomic-write
(`open(path, "w")` + `toml.dump` instead of `atomic_write_text`) shape that TASK-851 fixed in
`config.py`'s three encryption entry points, but wasn't one of 851's named entry points so was left
unfixed. Filed with AC requiring a regression test that derives the expected path via
`config._get_effective_config_path()` rather than a re-spelled literal, plus an atomicity test and a
no-profile-override round-trip check.

ID collision check: local scan found no duplicates and next-free id 869 pre-filing; `origin/dev`'s
`backlog/tasks` tree (fetched fresh) had ids up to 899 not yet present in this worktree, so the task
was filed as **TASK-900** (safely above both) rather than the locally-suggested 869, and a second
scan post-filing confirmed no local duplicates and no collision against `origin/dev`.

## Files changed

- `tldw_chatbook/Skills_Interop/skill_trust_store.py`
- `tldw_chatbook/Skills_Interop/local_skills_service.py`
- `tldw_chatbook/app.py`
- `Tests/conftest.py`
- `Tests/Utils/test_sensitive_paths.py`
- `Tests/Skills/test_verify_content_binary.py`
- `Tests/Skills/test_e2e_run_skill_script.py`
- `Tests/Skills/test_skills_library_flow.py`
- `Tests/Skills/test_skill_trust_service.py`
- `Tests/Skills/test_skill_remote_fetch.py`
- `Tests/Skills/test_skill_trust_service_reset_posture.py`
- `Tests/Skills/test_local_skills_service.py`
- `Tests/Skills/test_skill_trust_keyring_autounlock.py`
- `Tests/Skills/test_skill_trust_store_reset.py`
- `Tests/Skills/test_trust_tolerates_unsupported.py`
- `Tests/Skills/test_skill_trust_store.py`
- `Tests/Skills/test_skill_script_service.py`
- `backlog/tasks/task-853 - ...md` (status, plan, notes, AC checkboxes)
- `backlog/tasks/task-900 - ...md` (new)
