---
id: TASK-853
title: >-
  Fix the two skills-trust path containment checks that can't actually reject
  anything
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:34'
updated_date: '2026-07-27 14:38'
labels:
  - security
  - skills
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two independent containment checks in the skills-trust code are broken in ways that make them pass by construction.

Skills_Interop/skill_trust_store.py:93-96 and :110 call _validated_trust_file_path(self.marker_path, base_dir=self.marker_path.parent) -- i.e. the base directory is the candidate path's own parent. _validated_trust_file_path (:544-554) rejects when get_safe_relative_path(candidate, base) is None, but with base == candidate.parent that condition can never be true, so the check accepts any path. The correct pattern already exists two methods away: _validated_manifest_path (:490-491) passes base_dir=self.store_dir, and store_dir was available at the marker's construction site (app.py:5186). A reproduction confirmed base_dir=path.parent accepts a marker path in a totally unrelated directory, while base_dir=the real store correctly rejects it.

Skills_Interop/local_skills_service.py:1765-1804 _unsafe_scratch_root derives its container list correctly (self.skills_dir, trust_store.store_dir) but tests containment backwards: get_safe_relative_path(root, container) is non-None only when root is INSIDE container, so a scratch root nested inside the stores is flagged unsafe while a scratch root that CONTAINS both stores is not -- the opposite of what the docstring (:1782-1791) promises. A reproduction with real container paths showed: a root inside skills_dir is (correctly) flagged unsafe, but skills/ itself and the user's whole default_user directory -- both of which contain the stores -- are not flagged. self.store_dir (which holds the skill index tldw_chatbook_skills.json) isn't even in the container list. [skills] script_scratch_root = ~/.local/share/tldw_cli/<user>/skills would pass today and place a script's cwd one level above every trusted bundle and the trust manifest/snapshots/grants.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 skill_trust_store.py's marker-path validation passes base_dir=self.store_dir (or an equivalent real containing directory), not the candidate's own parent
- [x] #2 _unsafe_scratch_root rejects a candidate root that is inside a container AND a candidate root that contains a container (both directions), and includes self.store_dir in its container list
- [x] #3 A test exercises both containment directions for each check using the real store/scratch-root attributes (not hardcoded literal paths) and confirms both a nested-inside and a containing candidate are rejected
- [x] #4 Existing trust-store and scratch-root tests that only asserted the working direction are extended to cover the previously-vacuous direction
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce check #1 (skill_trust_store.py marker-path validation): construct FileSkillTrustGenerationMarkerStore with a marker path in an unrelated directory, show base_dir=marker_path.parent accepts it.
2. Reproduce check #2 (local_skills_service.py _is_unsafe_scratch_root): show a scratch root that CONTAINS skills_dir/trust store (e.g. store_dir or its parent) is NOT flagged unsafe, while a root nested inside skills_dir already is.
3. Fix check #1: add an explicit store_dir field to FileSkillTrustGenerationMarkerStore (and thread it through build_skill_trust_marker_store_with_fallback + app.py's construction site), replacing base_dir=self.marker_path.parent with base_dir=self.store_dir.
4. Fix check #2: add self.store_dir to _unsafe_scratch_root_containers, and check containment in both directions (root inside container OR root contains container) in _is_unsafe_scratch_root.
5. Update every test call site constructing FileSkillTrustGenerationMarkerStore directly (now requires store_dir) to derive it from the marker path already in scope rather than a fresh literal.
6. Re-run both reproductions post-fix to confirm rejection; add new tests covering both containment directions for each check using real store/service attributes; run the Tests/Skills + Tests/Utils/test_sensitive_paths.py + Tests/Library/test_skill_script_grant_panel.py suites to confirm no regressions to install/trust/script-execution flows.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both tautological containment checks.

Check #1 (skill_trust_store.py): FileSkillTrustGenerationMarkerStore gained an explicit `store_dir: Path` field (no default), and `_validated_trust_file_path` is now called with `base_dir=self.store_dir` in both load_marker and save_marker, instead of `base_dir=self.marker_path.parent` (which made the check true by construction). `build_skill_trust_marker_store_with_fallback` now takes `store_dir` and threads it into the fallback marker store; app.py's construction site passes `store_dir=trust_store_dir` (the same real trust-store directory already used for SkillTrustStore itself). Reproduced pre-fix: a marker file in a totally unrelated directory was accepted and written to. Post-fix: rejected with "unsafe skill trust path"; a legitimate marker co-located with store_dir still round-trips.

Check #2 (local_skills_service.py): `_unsafe_scratch_root_containers` now includes `self.store_dir` alongside `self.skills_dir` and the trust store's directory. `_is_unsafe_scratch_root` now checks containment in BOTH directions (`get_safe_relative_path(root, container)` OR `get_safe_relative_path(container, root)`), so a root that ENCLOSES a store is rejected in addition to a root nested inside one. Reproduced pre-fix: `store_dir` itself and its parent (both of which enclose skills_dir/trust_dir) were silently accepted as scratch roots, while a root nested inside skills_dir was already (correctly) rejected. Post-fix: both directions are rejected; a genuinely unrelated sibling directory is still accepted.

All ~28 test call sites constructing FileSkillTrustGenerationMarkerStore directly were updated to pass store_dir, re-derived from the marker path already in scope (e.g. `store_dir=marker_path.parent`) rather than a fresh literal, per the audit's "derive, don't re-spell" theme. Added a dedicated containment test to each file (test_skill_trust_store.py::test_file_marker_store_rejects_marker_outside_store_dir, test_skill_script_service.py::test_is_unsafe_scratch_root_rejects_both_containment_directions) covering both directions from the real store/service attributes, plus extended the existing marker-parent-symlink test to pass its own store_dir. One existing test (test_scratch_root_config_knob_is_reachable) needed its "custom" scratch root moved off of tmp_path (which happened to equal the service's own store_dir in that fixture) onto a genuinely unrelated tmp_path_factory directory, since a plain tmp_path-nested root is now correctly rejected -- this was a fixture artifact, not a real legitimate-use regression.

Verification: Tests/Skills (377 passed), Tests/Utils/test_sensitive_paths.py + Tests/Library/test_skill_script_grant_panel.py (all green). Baselines noted (pytest-mock/numpy absent, pre-existing test_tools_settings_window.py failures) were not touched by this change.

Filed TASK-900 for the sibling wrong-path/non-atomic-write defect found in UI/Tools_Settings_Window.py::_save_raw_toml_config (same shape as TASK-851's config.py fixes, a fourth entry point TASK-851 didn't cover), left unfixed per scope.

Files changed: tldw_chatbook/Skills_Interop/skill_trust_store.py, tldw_chatbook/Skills_Interop/local_skills_service.py, tldw_chatbook/app.py; 13 test files under Tests/Skills, Tests/Utils, Tests/conftest.py.
<!-- SECTION:NOTES:END -->
