# Chatbook Buddy task closeout — 2026-10-11

TASK-21236 now serializes controller injection/clear with the existing construction lock. A mounted Roleplay journey opens Manage Buddy from saved disabled preferences and preserves disabled state. Its negative control fails when explicit management stops ensuring a controller.

The unchanged startup import guards exposed eager artwork-attribution imports. Those imports now occur inside the existing validation/archive-reading functions; feature, digest, lineage and hostile-input validation remain intact. Startup fixtures retain their admitted bootstrap profile.

The affected round trips exposed an inherited activation refusal: full review validation requested the profile directory while a transaction held only database authority. Directory/capacity preflight stays before the SQL operation, including after character file preparation. Source, staged bytes, UUID, actor revision and visual authority still revalidate inside Character and Persona commits. Buddy conversion keeps its source guard both before insertion and after writes. Removing the inner guard causes the new pre-insert assertion to fail; source bytes were restored afterwards.

Local Persona views now always contain policy_rules under ADR-079. Actor Pack projection previously rejected that known field, even when empty. The repair carries rules through the existing Persona format, validates every rule with the existing strict schema, and preserves them through reviewed copy/import and re-export. Malformed fields and coercible permission values are rejected; local portrait/identity fields remain excluded. The existing review displays the carried actor fields.

## Verification

On the repaired working tree based on a3c47c6c72d885db2af0971515ddcdcb6cee78fa, the three activation/conversion/policy modules passed 104 cases. The final setter/startup/import-closure/management/attribution/lineage gate passed 26 cases. Across both runs, 129 unique test identities passed, with no failures, errors or skips. The setter RED, disabled-management mutation, policy RED and conversion source-guard mutation all failed at their intended assertions. Independent review found no actionable issues.

The [retained receipt](artifacts/buddy-closeout-20261011/verification.json) includes commands, Python/runtime identity, exact case identities, source and raw-result hashes, failure excerpts and static results. Ruff introduced zero diagnostics compared with exact HEAD; ten touched files pass formatting. Existing app_service_wiring formatting debt remains unchanged. Compilation and whitespace checks pass. Production Bandit has the same two baseline findings and no errors or introduced findings. Black is absent; existing Ruff formatting was used.

## Accepted historical UAT

TASK-31585 closes against the requester-approved revised criterion: verified merged repairs and accepted native/local human voice evidence retain their original source attribution. The [acceptance closeout](artifacts/buddy-uat-20261003/acceptance-closeout-20261011.json) preserves the former criterion, requester decision and nine verified original hashes. Exact historical native revision/exit and full human microphone/playback OpenAI realtime remain coverage limits. No accepted UAT or provider/audio run was repeated.

ADR required: no. Existing ADR-074, ADR-079, ADR-126 and ADR-139 govern these routine repairs. No storage, provider, runtime authority or dependency contract changed. Full suites were not run and normal user profiles were preserved.
