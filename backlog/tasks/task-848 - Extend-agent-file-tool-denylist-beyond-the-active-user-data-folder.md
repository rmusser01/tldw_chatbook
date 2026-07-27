---
id: TASK-848
title: Extend agent file-tool denylist beyond the active user data folder
status: To Do
assignee: []
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 04:36'
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
- [ ] #1 Vector-store and sibling-profile paths are refused under a widened sandbox root,The default sandbox configuration still works end to end,A test pins both directions
- [ ] #2 The skill trust/grant store (skill_trust_manifest.json, skill_script_grants.json, generation_marker.json, snapshots/) under get_user_data_dir()/skills/trust/ is refused by the denylist even though skills/ itself remains reachable
- [ ] #3 config.toml's .bak and .tmp sidecars (and any other file directly in the effective config's directory) are refused by the denylist
- [ ] #4 A test derives the trust-store and config-sidecar paths from the same accessors the app uses (SkillTrustStore/SkillTrustService attributes; _get_effective_config_path()) rather than asserting literals
<!-- AC:END -->
