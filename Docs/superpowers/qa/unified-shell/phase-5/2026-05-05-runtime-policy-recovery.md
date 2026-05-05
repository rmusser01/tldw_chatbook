# Phase 5.3 Runtime-Policy Recovery

Date: 2026-05-05
Task: `TASK-6.3`
Parent Task: `TASK-6`
Branch: `codex/unified-shell-phase5-runtime-policy-recovery`
Base: `origin/dev` at `264490d7`

## Goal

Apply the Phase 5 recovery taxonomy to visible runtime-policy denial states in service-backed destination shells. The UX goal is to stop treating policy denials as generic service failures and instead show what is unavailable, why policy blocked it, what the user can do next, the recovery target, and the authority owner.

## Applied Blockers

| Destination | Representative reason code | Blocked workflow | Recovery state |
| --- | --- | --- | --- |
| Skills | `authority_denied` | Attach local Skills to Console | Policy denied; review workspace policy or ask the authority owner to allow the action. |
| Library | `wrong_source` | Use Library sources in Console | Wrong source; switch to the required source, then retry. |
| Personas | `server_auth_required` | Attach Personas context to Console | Server sign-in required; reconnect or configure server credentials in Settings. |
| W+C | `server_session_invalid` | Stage W+C context in Console | Server session expired; re-authenticate the active server profile. |

## UX Verification

Focused Textual widget walkthroughs mounted each affected destination through `DestinationHarness`, injected policy-denied service responses, waited for the async snapshot terminal state, and verified:

- Visible status label.
- `Unavailable:` field.
- `Why:` field from the policy message.
- `Next:` recovery step.
- `Recovery:` target.
- `Owner:` authority owner.
- Disabled Console action tooltip repeats the reason and next step.

## Automated Evidence

- Red regression: `python -m pytest Tests/UI/test_destination_shells.py -q -k "policy_denial or policy_denied"`
- Red result before implementation: `3 failed`, showing raw policy messages without recovery taxonomy fields.
- Focused runtime-policy verification: `python -m pytest Tests/UI/test_destination_shells.py -q -k "policy_denial or policy_denied"`
- Focused runtime-policy result after implementation: `4 passed`
- Affected destination and tracking verification: `python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q`
- Affected destination and tracking result: `78 passed`

Representative regression tests:

- `test_watchlists_collections_policy_denial_uses_runtime_recovery_taxonomy`
- `test_personas_policy_denial_uses_runtime_recovery_taxonomy`
- `test_library_policy_denial_uses_runtime_recovery_taxonomy`
- `test_skills_destination_policy_denied_surfaces_policy_message`

## Residual Risk

- This slice verifies representative policy denial paths in service-backed destination shells, not every runtime-policy action in the registry.
- Live server failure classification still depends on upstream service adapters producing the correct `PolicyDeniedError.reason_code`.
- Optional dependency states remain outside this slice and still need Phase 5 taxonomy application.
- Parent `TASK-6` remains In Progress until optional-dependency blocker families and final running-app closeout QA are complete.
