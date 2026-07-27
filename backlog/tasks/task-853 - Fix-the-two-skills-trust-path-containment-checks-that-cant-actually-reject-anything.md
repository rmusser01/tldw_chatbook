---
id: TASK-853
title: >-
  Fix the two skills-trust path containment checks that can't actually reject
  anything
status: To Do
assignee: []
created_date: '2026-07-27 04:34'
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
- [ ] #1 skill_trust_store.py's marker-path validation passes base_dir=self.store_dir (or an equivalent real containing directory), not the candidate's own parent
- [ ] #2 _unsafe_scratch_root rejects a candidate root that is inside a container AND a candidate root that contains a container (both directions), and includes self.store_dir in its container list
- [ ] #3 A test exercises both containment directions for each check using the real store/scratch-root attributes (not hardcoded literal paths) and confirms both a nested-inside and a containing candidate are rejected
- [ ] #4 Existing trust-store and scratch-root tests that only asserted the working direction are extended to cover the previously-vacuous direction
<!-- AC:END -->
