# TASK-846 audit — backlog filing report

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-hardening` (branch `feat/agent-path-hardening`)
**Source:** `.superpowers/sdd/task-846-audit.md`, "Recommended backlog tasks" section (14 items).
**ID range used:** task-851 through task-862 (12 new tasks). Range was free at filing time (pre-scan showed
`dups: NONE`, `next free: 851`); post-filing scan below also shows `NONE` — but a concurrent session could
still file into 851-862 before merge, so re-run the scan immediately before merging.

## Tasks filed

| ID | Title | Priority | Labels | Source recommendation(s) |
|---|---|---|---|---|
| task-851 | Route config encryption through the effective config path, not the default one | **High** | security, config | Rec 1 (CRITICAL-1) |
| task-852 | Make should_encrypt_config() detect real provider API keys, and unify the four sensitive-key lists | **High** | security, config | Rec 2 (CRITICAL-2 + IMPORTANT-4 + INV-4) |
| task-853 | Fix the two skills-trust path containment checks that can't actually reject anything | **High** | security, skills | Rec 5 (CRITICAL-5 + CRITICAL-6) |
| task-854 | Fix MCP server's media DB lookup to use the real config key/accessor | medium | security, mcp, config | Rec 6 (IMPORTANT-1) |
| task-855 | Make MCP store module defaults derive from get_user_data_dir(), not ~/.config/tldw_cli | medium | security, mcp | Rec 7 (IMPORTANT-2) |
| task-856 | Decide the fate of Utils/log_sanitizer.py: wire it in fixed, or delete it | medium | security, config | Rec 8 (IMPORTANT-3) |
| task-857 | Make workspace folder-binding consult the sensitive-path denylist, not just home/root | medium | security, tools | Rec 9 (IMPORTANT-6) |
| task-858 | Reconcile the evals/prompts/media/rag/subscriptions DB path maps with get_*_db_path() | medium | security, db, config | Rec 10 (IMPORTANT-7) |
| task-859 | Delete or implement the unread [subscriptions.security] switches and stale metadata denylist | medium | security, config | Rec 11 (IMPORTANT-8 + IMPORTANT-9) |
| task-860 | Fix sql_validation.VALID_TABLES to match the real schema (keyword_collections is live-broken) | medium | security, db | Rec 12 (INV-1 + INV-2) |
| task-861 | Sweep hardcoded ~/.config/tldw_cli and ~/.local/share/tldw_cli call sites onto the real accessors | medium | security, config | Rec 13 (IMPORTANT-10) |
| task-862 | Make sensitive-path and skills-fixture tests re-derive paths instead of re-spelling them | medium | security, tools | Rec 14's two named test sites (`Tests/Utils/test_sensitive_paths.py:73-75`, `Tests/conftest.py:566-575` + `Tests/Skills/test_skills_library_flow.py:84-106`) |

Three Criticals (CRITICAL-1, CRITICAL-2, CRITICAL-5+6) map to three new tasks (851, 852, 853) — all set
`priority: high`. The other three Criticals (CRITICAL-3 skill trust/grant store, CRITICAL-4 config.toml
sidecars) were **not** filed as new tasks per the exception below; they're folded into TASK-848.

Recommendation 14 itself (the cross-cutting "tests must re-derive, not re-assert" rule) was **not** filed
as a new task — it is already TASK-846's own acceptance criterion #3 ("Tests for those checks derive their
paths the way the app does rather than asserting literals"). Only the two concrete test sites it named were
filed, as task-862.

## TASK-848 amendment (not a new task)

Recommendations 3 (skill trust/grant store coverage, CRITICAL-3) and 4 (`config.toml` `.bak`/`.tmp` sidecars,
CRITICAL-4) were appended to **TASK-848** ("Extend agent file-tool denylist beyond the active user data
folder") rather than filed separately, per instruction — they're extensions of that task's existing scope
(the agent file-tool denylist).

- **Description**: two new paragraphs appended after the original text — "Skill trust/grant store." (naming
  `get_user_data_dir()/skills/trust/*`, the `skills` directory-exemption rule that structurally hides it,
  and the plain-unauthenticated-JSON status of `skill_script_grants.json`) and "config.toml sidecars." (naming
  `.bak`/`.tmp` written by `UI/Screens/settings_screen.py:5596-5605`, the existing DB-sidecar pattern to
  mirror, and the two real unprotected copies already on disk).
- **Acceptance criteria**: appended 3 new AC items (#2, #3, #4) alongside the original #1 — trust/grant store
  coverage, config.toml sidecar coverage, and a test requiring both to be derived from the real accessors
  (`SkillTrustStore`/`SkillTrustService` attributes, `_get_effective_config_path()`) rather than literals.
- TASK-848's priority was left as-is (unset/default) — the instruction to set `priority: high` applied to
  newly-filed tasks derived from Critical findings, and 848 was not newly filed.

## Post-filing duplicate scan

```
dups: NONE
next free: 863
```

Files present for the new range: `task-851` through `task-862`, twelve files, no collisions with any
pre-existing task id.

**Re-check before merge**: this scan is a snapshot from this filing session. Per the collision protocol,
re-run the scanner script against `backlog/tasks/` immediately before merging this branch, since a
concurrent session could file into 851-862 in the meantime.
