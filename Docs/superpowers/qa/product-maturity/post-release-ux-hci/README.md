# Post-Release UX/HCI Functional Validation

Status: in progress; `TASK-60` active

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
| Home | recorded | pending | next-best action opens Library; active-work Console payload verified | `TASK-60.3` verified |
| Console | recorded | pending | visible composer input, blocked-send recovery, and source-readiness summary verified | `TASK-60.3` verified |
| Library | recorded | pending | Search/RAG mode, blocked empty recovery, and source-to-Console handoff verified | `TASK-60.3` verified |
| Artifacts | recorded | pending | empty state, Open Console recovery, and Chatbook resume payload verified | `TASK-60.3` verified |
| Personas | recorded | approved | loading blocker fixed; CCP route approved as destination-native workbench | `TASK-60.5` |
| Watchlists | recorded | pending | deterministic loading recovery and active-run Console follow verified | `TASK-60.6`; `TASK-60.3` verified |
| Schedules | recorded | pending | empty/recovery baseline and active-run Console follow verified | `TASK-60.3` verified |
| Workflows | recorded | pending | empty/recovery baseline and active-run Console follow verified | `TASK-60.3` verified |
| MCP | recorded | pending | local overview and no-server recovery verified; Console source readiness classifies MCP as future service-depth | `TASK-60.3` verified; `TASK-60.4` planning |
| ACP | recorded | pending | runtime-not-configured recovery and fixture-backed session handoff verified | `TASK-60.3` verified; `TASK-60.4` planning |
| Skills | recorded | pending | empty state, import-not-wired recovery, and valid-skill Console attach verified | `TASK-60.3` verified |
| Settings | recorded | pending | preference and scope baseline recorded | `TASK-60.3` |

## Required Cross-Screen Workflows

| Workflow | Evidence Status | Acceptance Gate |
| --- | --- | --- |
| Home active work to details and Console | verified | actual-use handoff or clear blocked recovery |
| Library Search/RAG to Console context | verified/recoverable | evidence handoff or clear blocked recovery |
| Console composer, send/block, Chatbook save/resume | verified/recoverable | visible input and recoverable send/save path |
| Artifacts/Chatbooks to Console resume | verified | no dead controls |
| Personas/Skills/MCP/ACP context to Console | verified/recoverable | verified handoff or honest blocked future work |
| Watchlists/Schedules/Workflows runs to Console | verified | verified handoff or honest blocked future work |

Current workflow evidence: `2026-05-22-cross-screen-workflow-validation.md`.

## Severity Rules

| Severity | Meaning | Close Rule |
| --- | --- | --- |
| P0 | Blocks basic use or makes a primary screen unusable. | Must be fixed before the screen/workflow can be accepted. |
| P1 | Seriously degrades a core workflow or creates misleading state. | Must be fixed or linked to an accepted follow-up task before closeout. |
| P2 | Confusing, inefficient, or weak recovery, but usable. | May remain with explicit residual risk and follow-up. |
| P3 | Polish that does not hide status, recovery, or user control. | May remain as backlog polish. |

## Required Template

Use `walkthrough-template.md` for every screen and workflow evidence file.
