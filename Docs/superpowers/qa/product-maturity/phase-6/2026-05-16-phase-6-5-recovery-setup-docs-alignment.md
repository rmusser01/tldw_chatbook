# Phase 6.5 Recovery, Setup, And Documentation Alignment

<!-- PHASE_6_5_RECOVERY_DOCS_METADATA:BEGIN -->
```json
{
  "task": "TASK-13.5",
  "parent_task": "TASK-13",
  "decision": "recovery_setup_docs_alignment_recorded",
  "recovery_blockers_checked": [
    "provider-model",
    "server",
    "acp-runtime",
    "mcp-management",
    "optional-dependency",
    "missing-source"
  ],
  "p0_p1_findings": [],
  "screenshot_gate": "not_required_no_visible_ui_changes",
  "final_focused_replay_result": {
    "command": "python -m pytest -q Tests/UI/test_product_maturity_phase6_recovery_docs.py Tests/UI/test_product_maturity_phase6_release_hardening_plan.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short",
    "passed": 8,
    "failed": 0
  }
}
```
<!-- PHASE_6_5_RECOVERY_DOCS_METADATA:END -->

## Environment

- Date: 2026-05-16
- Branch: `codex/phase6-recovery-docs`
- Scope: Product Maturity Phase 6.5 release recovery/setup/documentation alignment
- App under test: running Textual app through the mounted `TldwCli` harness
- Recovery guide: `Docs/Development/release-recovery-setup.md`
- Visual approval rule: no visible UI code changed in this slice, so actual rendered screenshot approval is not required.

## Recovery Matrix

| Blocker | Status | Running-App Evidence | Recovery/Docs Alignment | Severity | Documentation |
| --- | --- | --- | --- | --- | --- |
| provider-model | verified | Home exposes `Model: Blocked`; Console exposes provider setup copy and OpenAI API-key recovery. | Recovery guide points users to Settings plus `OPENAI_API_KEY` / provider API settings. | none | `Docs/Development/release-recovery-setup.md` |
| server | verified | Home exposes server sync status and local-mode recovery: server sync is optional in local mode. | Recovery guide distinguishes local mode from server-backed work and routes reconnect/auth to active-server setup. | none | `Docs/Development/release-recovery-setup.md` |
| acp-runtime | verified | ACP exposes `Runtime not configured`, why/next/owner copy, and disabled launch/follow states. | Recovery guide keeps ACP runtime setup owned by ACP, not global Settings. | none | `Docs/Development/release-recovery-setup.md` |
| mcp-management | verified | Console points MCP recovery to MCP; MCP screen exposes server/scope management and inventory prompt. | Recovery guide keeps MCP server/tool management owned by MCP and documents `pip install -e ".[mcp]"`. | none | `Docs/Development/release-recovery-setup.md` |
| optional-dependency | verified | Existing disabled-action recovery tests cover missing optional dependency groups and owners. | Recovery guide documents optional extras including `pip install -e ".[embeddings_rag]"`. | none | `Docs/Development/release-recovery-setup.md` |
| missing-source | verified | Home, Console, and Library expose missing-source or no-source-selected recovery copy. | Recovery guide routes users through Library sources, Search/RAG, Import/Export Sources, and staging into Console. | none | `Docs/Development/release-recovery-setup.md` |

## Running-App Replay Notes

- Home showed model-blocked, missing-source, server/local-mode, and next-action recovery copy.
- Console showed provider/model setup guidance, RAG/source not-staged state, MCP recovery routing, and ACP session-payload status.
- ACP showed runtime-not-configured recovery with why/next/owner fields.
- MCP showed server/scope management language and inventory inspection recovery.
- Library showed source-service unavailable and no-source-selected recovery language.
- Optional dependency recovery remains covered by existing disabled-action recovery seams and is now linked from the release recovery guide.

## Documentation Alignment

- Added `Docs/Development/release-recovery-setup.md` as the release-candidate recovery guide.
- The guide mirrors current running-app labels rather than introducing unsupported actions.
- README remains the source for install extras, environment variables, MCP optional dependency, and web-server setup.
- Verification commands use portable `python -m pytest ...` forms instead of machine-specific absolute paths.

## P0/P1 Decision

No P0 or P1 release blockers were found.

Accepted residuals:

- The Console provider copy still includes legacy provider-detail phrasing in some internal state rows; the actionable recovery path remains visible and is not blocking.
- ACP runtime implementation remains future work; the release requirement here is honest blocked-state recovery, not runtime launch completion.
- Optional dependency recovery is verified through existing disabled-action seams and docs alignment rather than by uninstalling local packages in this environment.

## Residual Risk

Release readiness still depends on:

- `TASK-13.6`: packaging/configuration/data-safety validation.
- `TASK-13.7`: public roadmap release closeout.

## Verification

- `python -m pytest -q Tests/UI/test_product_maturity_phase6_recovery_docs.py Tests/UI/test_product_maturity_phase6_release_hardening_plan.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short`
- Regression file: `Tests/UI/test_product_maturity_phase6_recovery_docs.py`
