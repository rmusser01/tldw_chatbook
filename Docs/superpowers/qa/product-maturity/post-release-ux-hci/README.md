# Post-Release UX/HCI Functional Validation

Status: planned; `TASK-60` active

This index exists because prior mounted layout checks and phase closeout evidence are not enough to prove the app is usable. Each top-level destination and cross-screen workflow must be driven in the actual app, evaluated from a senior UX/HCI perspective, and backed by real rendered screenshots before acceptance.

## Evidence Rules

- Actual screenshots are required for screen approval. Do not use generated SVGs, code-layout renderings, or ASCII-only mockups as approval evidence.
- Prefer textual-web with CDP/browser automation when it exposes the real Textual UI. Use an actual terminal screenshot when browser rendering is unavailable, misleading, or insufficient.
- Screen acceptance requires both visual approval and actual-use functionality evidence.
- A screen that renders but has dead controls, invisible input, broken focus, unclear recovery, or unusable cross-screen handoffs is not accepted.
- P0/P1 findings require Backlog follow-up tasks before the related audit task can close.

## Required Screens

| Screen | Evidence Status | Screenshot Approval | Functionality Status | Follow-Up |
| --- | --- | --- | --- | --- |
| Home | pending | pending | pending | `TASK-60.2` |
| Console | pending | pending | pending | `TASK-60.2` |
| Library | pending | pending | pending | `TASK-60.2` |
| Artifacts | pending | pending | pending | `TASK-60.2` |
| Personas | pending | pending | pending | `TASK-60.2` |
| Watchlists | pending | pending | pending | `TASK-60.2` |
| Schedules | pending | pending | pending | `TASK-60.2` |
| Workflows | pending | pending | pending | `TASK-60.2` |
| MCP | pending | pending | pending | `TASK-60.2` |
| ACP | pending | pending | pending | `TASK-60.2` |
| Skills | pending | pending | pending | `TASK-60.2` |
| Settings | pending | pending | pending | `TASK-60.2` |

## Required Cross-Screen Workflows

| Workflow | Evidence Status | Acceptance Gate |
| --- | --- | --- |
| Home active work to details and Console | pending | actual-use handoff or clear blocked recovery |
| Library Search/RAG to Console context | pending | evidence handoff or clear blocked recovery |
| Console composer, send/block, Chatbook save/resume | pending | visible input and recoverable send/save path |
| Artifacts/Chatbooks to Console resume | pending | no dead controls |
| Personas/Skills/MCP/ACP context to Console | pending | verified handoff or honest blocked future work |
| Watchlists/Schedules/Workflows runs to Console | pending | verified handoff or honest blocked future work |

## Severity Rules

| Severity | Meaning | Close Rule |
| --- | --- | --- |
| P0 | Blocks basic use or makes a primary screen unusable. | Must be fixed before the screen/workflow can be accepted. |
| P1 | Seriously degrades a core workflow or creates misleading state. | Must be fixed or linked to an accepted follow-up task before closeout. |
| P2 | Confusing, inefficient, or weak recovery, but usable. | May remain with explicit residual risk and follow-up. |
| P3 | Polish that does not hide status, recovery, or user control. | May remain as backlog polish. |

## Required Template

Use `walkthrough-template.md` for every screen and workflow evidence file.
