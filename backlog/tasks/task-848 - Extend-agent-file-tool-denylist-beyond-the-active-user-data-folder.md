---
id: TASK-848
title: Extend agent file-tool denylist beyond the active user data folder
status: Done
assignee: []
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 05:07'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The denylist refuses files sitting directly in get_user_data_dir(), which covers the app's own state. Under a deliberately widened sandbox root, chromadb/chroma.sqlite3 (plaintext chunks of the same conversations and notes ChaChaNotes.db protects) and sibling profile folders remain readable. Reviewed as a disclosure asymmetry rather than a permission-gate bypass -- skill trust manifests are HMAC+keyring authenticated and script grants are digest-pinned, so tampering fails closed. Filed from the PR #953 review.

Also explicitly in scope for this task (folded in from the TASK-846 path-hardening audit, recommendations 3 and 4):

Skill trust/grant store. get_user_data_dir()/skills/trust/ (skill_trust_manifest.json, skill_script_grants.json, generation_marker.json, snapshots/, built at app.py:5181-5200) is not named by Utils/sensitive_paths.py at all, and the module's existing resolved.parent == user_data_dir rule cannot reach it: skills is an existing directory, so it is exempted by design (sensitive_paths.py:48-51 names skills among the intentionally-reachable containers), and everything nested under it inherits that exemption. skill_script_grants.json (Skills_Interop/skill_trust_service.py:49, joined at :661) is deliberately kept outside the MAC'd manifest -- it is the plain, unauthenticated JSON file that has_script_grant (:1452-1465) consults to authorize script execution, so it is not covered by the manifest's own HMAC+keyring integrity check either. A live check confirmed is_sensitive_path() returns False for skill_trust_manifest.json, skill_script_grants.json, generation_marker.json, the snapshots directory, and the skill index tldw_chatbook_skills.json. Reachability still requires a widened sandbox root or a bound workspace folder (the reachability path this task already covers), so this is uncovered surface rather than a live bypass today -- but it is the exact "one-step gate bypass" class the mcp_permissions.json denylist entry exists to prevent, and it sits squarely alongside the vector-store/sibling-profile gap this task was filed for.

config.toml sidecars. UI/Screens/settings_screen.py:5596-5605's Advanced config save writes a full plaintext backup (config.toml.bak) before overwriting, plus a config.toml.tmp during the atomic swap; both carry every plaintext API key the live config holds. The denylist only names _get_effective_config_path() itself -- nothing else in that directory -- so neither sidecar is covered, and the equivalent DB-sidecar treatment the module already has (_DB_SIDECAR_SUFFIXES for -wal/-shm/-journal, and the MCP permission store's own .bak via the user_data_dir file rule) does not extend to ~/.config/tldw_cli, where config.toml actually lives. Two hand-made copies already sit unprotected on disk on the audit machine (config.toml.bak-1785079350, config.toml.pre-lab-cleanup, both 45 KB).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Vector-store and sibling-profile paths are refused under a widened sandbox root,The default sandbox configuration still works end to end,A test pins both directions
- [x] #2 The skill trust/grant store (skill_trust_manifest.json, skill_script_grants.json, generation_marker.json, snapshots/) under get_user_data_dir()/skills/trust/ is refused by the denylist even though skills/ itself remains reachable
- [x] #3 config.toml's .bak and .tmp sidecars (and any other file directly in the effective config's directory) are refused by the denylist
- [x] #4 A test derives the trust-store and config-sidecar paths from the same accessors the app uses (SkillTrustStore/SkillTrustService attributes; _get_effective_config_path()) rather than asserting literals
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce all three explicitly-scoped gaps first: (a) the skill trust/grant store is not sensitive despite being nested under the exempted `skills` container; (b) config.toml's .bak/.tmp sidecars are not sensitive; (c) chromadb/chroma.sqlite3 and a rag_profiles file are not sensitive under a widened sandbox root.
2. Add single-source-of-truth path accessors so sensitive_paths.py can derive these locations instead of re-spelling literals: default_local_skills_store_dir()/default_trust_store_dir() (Skills_Interop) and default_rag_profiles_dir() (RAG_Search.config_profiles); wire app.py/ConfigProfileManager through them too so there is exactly one spelling of each name.
3. Refuse the whole skills/trust subtree by ancestry (a deliberate carve-out from the skills/ container exemption), documented in both the module docstring and the new resolver's own docstring.
4. Generalize the existing "any file directly in get_user_data_dir()" rule to three more container directories via a new SensitivePathContext.direct_child_denied_dirs field: the effective config directory, the ChromaDB persist directory, and the rag_profiles directory.
5. Assess the audit's residual-fragility note about the three MCP filenames being re-spelled literals in sensitive_paths.py; decide whether to fix within this task's scope.
6. Add regression tests deriving every new path from the same accessors the app uses (never literals); prove existing containers (skills/ itself, chromadb/, rag_profiles/, tool_sandbox/, an existing config-dir subdirectory) stay reachable, and that the default sandbox configuration still works end to end.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced all three explicitly-scoped gaps first (real, unmocked repro scripts), confirmed each returned is_sensitive_path()==False before the fix, then confirmed True after.

AC#1 (vector-store/sibling-profile paths, widened root): generalized the existing "any file directly in get_user_data_dir(), existing directories stay reachable" rule into a new SensitivePathContext.direct_child_denied_dirs tuple covering four container directories instead of one: get_user_data_dir(), the effective config directory, the ChromaDB persist directory (RAG_Search.simplified.config.default_chroma_persist_directory(), already existed), and the RAG-profile store (RAG_Search.config_profiles.default_rag_profiles_dir(), newly added -- no accessor existed before; ConfigProfileManager.__init__ now calls it too). chroma.sqlite3 and a rag_profiles file are now refused; the container directories and an existing per-collection subdirectory nested in chromadb/ stay reachable.

AC#2 (skill trust/grant store, audit CRITICAL-3): get_user_data_dir()/skills/trust/ was structurally unreachable because `skills` is an existing-directory container exemption and everything nested under it inherited that exemption. Added two new small pure accessors (no prior single source of truth existed -- app.py built both names as inline literals): Skills_Interop.local_skills_service.default_local_skills_store_dir() and Skills_Interop.skill_trust_store.default_trust_store_dir(); app.py's service wiring now calls both instead of re-spelling "skills"/"trust", and the generation-marker filename was promoted from an app.py literal to a public MARKER_FILENAME constant in skill_trust_store.py. The whole trust/ subtree is refused by ancestry (like ~/.ssh), not just direct children, so manifest/grants/marker/every nested snapshot file are all covered; skills/ itself (outside trust/) stays reachable.

AC#3 (config.toml .bak/.tmp sidecars, audit CRITICAL-4): DECISION -- generalized the direct-child-file rule to the effective config directory (config._get_effective_config_path().parent) rather than enumerating .bak/.tmp suffixes, because the audit's own live evidence showed unprotected files beyond those two suffixes (runtime_policy.json, ui_state.toml, an arbitrarily-named hand-made backup). One mechanism, reused from AC#1, catches all of them plus any future one; existing directories placed directly in the config dir (the real feed_cache/themes/tokenizers) stay reachable via the same is-a-directory gate.

Assessed the audit's note that sensitive_paths.py re-spells the three MCP filenames (mcp_permissions.json, local_mcp_store.json, mcp_execution_log.jsonl) as its own literals. DECISION -- left as-is: no single source of truth exists anywhere in the codebase for these names (app.py and MCP/unified_control_plane_service.py each independently spell them too); fixing it properly requires touching MCP/local_store.py, MCP/unified_context_store.py, MCP/server_target_store.py, MCP/unified_control_plane_service.py, and app.py, none of which is named in this task's ACs. Documented as a known, deliberately out-of-scope residual fragility; recommend a follow-up task scoped to the MCP store-path family.

AC#4: every new test derives its path from the actual accessor/constructor the app uses (default_local_skills_store_dir/default_trust_store_dir, a real constructed SkillTrustStore's own manifest_path/snapshots_dir, _SCRIPT_GRANTS_FILENAME/MARKER_FILENAME, config._get_effective_config_path(), default_chroma_persist_directory(), default_rag_profiles_dir()) -- never a re-spelled literal.

Verified: `pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q` -> 893 passed, 0 failed.

Files: tldw_chatbook/Utils/sensitive_paths.py, tldw_chatbook/Skills_Interop/local_skills_service.py, tldw_chatbook/Skills_Interop/skill_trust_store.py, tldw_chatbook/Skills_Interop/__init__.py, tldw_chatbook/RAG_Search/config_profiles.py, tldw_chatbook/app.py, Tests/Utils/test_sensitive_paths.py, Tests/Tools/test_file_tool_sandbox.py.
<!-- SECTION:NOTES:END -->
