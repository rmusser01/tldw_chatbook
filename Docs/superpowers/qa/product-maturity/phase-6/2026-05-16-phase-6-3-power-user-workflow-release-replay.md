# Phase 6.3 Power-User Workflow Release Replay

<!-- PHASE_6_3_POWER_USER_RELEASE_METADATA:BEGIN -->
```json
{
  "task": "TASK-13.3",
  "parent_task": "TASK-13",
  "persona": "power-user",
  "decision": "power_user_release_replay_recorded",
  "verified_workflows": [
    "grounded-answer-loop",
    "source-to-artifact-loop",
    "agent-run-loop",
    "monitoring-loop",
    "study-loop",
    "recovery-loop"
  ],
  "p0_p1_findings": [],
  "screenshot_gate": "not_required_no_visible_ui_changes",
  "final_focused_replay_result": {
    "command": "python -m pytest -q Tests/UI/test_product_maturity_phase6_power_user_replay.py --tb=short",
    "passed": 2,
    "failed": 0
  }
}
```
<!-- PHASE_6_3_POWER_USER_RELEASE_METADATA:END -->

## Environment

- Date: 2026-05-16
- Branch: `codex/phase6-power-user-replay`
- Scope: Product Maturity Phase 6.3 release-hardening replay
- App under test: running Textual app through the mounted `TldwCli` harness
- Terminal size: `180x50`
- Persona: experienced user repeating core workflows quickly from Home, Console, and Library

## Workflow Matrix

| Workflow | Status | Running-app path verified | Severity | Friction / decision |
| --- | --- | --- | --- | --- |
| grounded-answer-loop | verified | Home next action opens Console; Console exposes live agent control, RAG/source state, provider readiness, and composer controls. | none | Provider credentials may still be blocked, but the blocked state is visible and recoverable. |
| source-to-artifact-loop | verified | Console exposes Artifacts readiness and Save Chatbook action; Library and Artifacts remain top-level destinations. | P2 | Full live artifact generation is outside this replay; Phase 2 and Console artifact contracts cover persistence seams. |
| agent-run-loop | verified | Console Run Inspector, live-work source summary, approvals, and tool-call review controls remain visible for repeat agent control. | none | ACP runtime remains an explicit future risk, not a hidden release blocker. |
| monitoring-loop | verified | Watchlists active work can open Console live-work context and follow through to the Watchlists run details tab. | none | Mounted replay verifies the routing seam rather than live external polling. |
| study-loop | verified | Library exposes Study Dashboard, Flashcards, Quizzes, Collections, Search/RAG, and Import/Export from the same source workbench. | none | Deeper study-generation services remain previously tracked downstream risks. |
| recovery-loop | verified | Failed Watchlists live work carries visible recovery text and a primary action to the source run. | none | Recovery copy is actionable for the staged failure path. |

## Running-App Replay Notes

1. Started on Home as a returning/power user with model and Library content readiness enabled.
2. Used the Home primary action to jump directly into Console.
3. Verified Console shows live agent control, live-work sources, RAG/source readiness, Artifacts readiness, and command-palette affordance.
4. Used top navigation to enter Library and verified Search/RAG, Import/Export Sources, Collections, Study Dashboard, Flashcards, and Quizzes remain visible from the Library workbench.
5. Entered Library Search/RAG mode without leaving Library.
6. Entered Import/Export from Library and verified the ingest destination opens.
7. Staged a failed Watchlists live-work item into Console with recovery copy.
8. Activated the Console live-work primary action and verified it opens Watchlists run details.

## Power-User Speed And Recovery Findings

- Home provides a direct `Start in Console` path for fast repeat use when model setup is ready.
- Top navigation stays available across Home, Console, Library, Search/RAG mode, and Import/Export.
- `Ctrl+P` remains discoverable in the shell and bound in the app command bindings.
- Library keeps source acquisition, retrieval, collection, and study paths grouped in one source workbench.
- Console live-work handoff preserves source, title, status, recovery, and follow-through target.

## P0/P1 Decision

No P0 or P1 release blockers were found in this replay.

P2/P3 accepted residuals:

- This replay verifies mounted routing and state seams, not live provider generation or full external service execution.
- ACP runtime launch and full write sync remain explicitly deferred outside Phase 6.3.
- Full visual screenshot approval is reserved for Phase 6.4 unless this slice changes visible UI.

## Residual Risk

Release readiness still depends on:

- `TASK-13.4`: keyboard/focus/accessibility and visual sweep with rendered screenshots for any visual changes.
- `TASK-13.5`: recovery/setup/documentation alignment.
- `TASK-13.6`: packaging/configuration/data-safety validation.
- `TASK-13.7`: public roadmap release closeout.

## Verification

- `python -m pytest -q Tests/UI/test_product_maturity_phase6_power_user_replay.py --tb=short`
- Regression file: `Tests/UI/test_product_maturity_phase6_power_user_replay.py`
