# Structured Document Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver local YAML, JSON, and JSONL diagnostics and reversible formatting in both Notes applications without changing accepted source content or losing language metadata.

**Architecture:** Preserve the existing Notes and File Notes save authorities. Add portable language metadata, then qualify separate Python/browser engines and connect them through revision-checked editor adapters. Split delivery into the three independently reviewable plans below.

**Tech Stack:** Python 3.12+, Textual 8.2.8, SQLite; server FastAPI/Pydantic and SQLite/PostgreSQL; shared React/TypeScript UI, browser workers, pytest, Vitest, Playwright. Parser dependencies are qualification candidates, not installations approved by this planning artifact.

**Spec:** [Approved design](../specs/2026-09-27-structured-document-editing-design.md)

## Global Constraints

- "Check after 400 ms of typing inactivity and on initial open or language change."
- "Initial common limits are 1 MiB of UTF-8 source, nesting depth 100, 100 displayed diagnostics, and a 2-second processing deadline per request."
- "Syntax errors never add a save veto."
- "The empty file is zero records." This applies to JSONL, not JSON.
- "Do not implement formatting as a generic object parse followed by serialization."
- "Formatting is deterministic and idempotent within each engine; identical whitespace between the two engines is not required."
- "Run targeted checks only unless the user explicitly authorizes a full suite."
- All remaining spec requirements apply. No LSP, external lint command, validation API, format-on-save, application-schema validation, or Markdown block linting.
- Respect stricter existing file/editor limits: File Notes currently limits interactive bodies to 200,000 characters. The engine limit does not make an excerpt editable.
- Do not put new settings in deprecated Settings surfaces. Use design tokens, source CSS modules, and the existing bundle builder.

## Ownership, status, and execution prerequisites

Design owner: [TASK-33096](../../../backlog/tasks/task-33096%20-%20Design-structured-document-validation-and-formatting-across-Chatbook-and-server-Notes.md).
Status: planned; no implementation steps executed.

ADR required: yes
ADR path: [ADR-194](../../../backlog/decisions/194-structured-note-language-and-local-editor-validation.md)
Reason: portable storage/sync metadata and bounded local editor processing.
Server governing ADR: `Docs/ADR/031-notes-capability-sync-domains.md` in the server repository. Before server implementation, record an extension under that repository's ADR assessment workflow for the new payload version; do not rewrite accepted ADR-031 rationale.

Paths in the subplans use these roots, with no implication of a shared Git index:

- **C:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook`
- **S:** `/Users/macbook-dev/Documents/GitHub/tldw_server`
- **U:** `S/apps/packages/ui`
- **W:** `S/apps/tldw-frontend`

Resolve fresh managed worktrees at execution time and substitute their actual roots
for C/S. Both current workspaces contain unrelated work; never clean, stash, or
stage it. Each implementation unit gets a repo-local Backlog task before edits,
using the CLI and a fresh collision scan. F1/E1/I1 below are plan-unit labels, not
uncreated Backlog IDs. Only link actual allocated IDs, with backward dependencies.
Mark tasks In Progress and record the unit's plan before code changes.

Read each repository's AGENTS guidance and relevant nested guidance, the spec,
the accepted ADR, and area lessons before starting a unit. Server PostgreSQL
migrations and shared web/extension module loading must be considered even when
developing on SQLite and the web client.

## Ordered delivery plans

| Plan | Deliverable | Entry / exit gate |
| --- | --- | --- |
| [Foundations](2026-09-27-structured-document-foundations.md) | Exact source transport, portable language, recovery and export boundaries | F1–F6 deliver useful fixes/contracts before UI adoption. |
| [Engines](2026-09-27-structured-document-engines.md) | Qualified format/check engines and terminable worker owners | E1 fixtures precede E2/E3; E4 requires qualified engines. |
| [Editors](2026-09-27-structured-document-editors.md) | Undo-qualified editors, diagnostics UI, source downloads, cross-app evidence | I1 qualifies history early; I2/I3 integrate only after their foundation and engine dependencies. |

E1 and I1 qualification may run before all foundation work is merged. This is a
dependency allowance, not authorization to spawn agents or edit the same files
concurrently. User chooses the execution method after plan delivery.

## Shared interfaces

Implement matching wire fields in `C/tldw_chatbook/Document_Editing/models.py`
and `U/src/services/document-editing/types.ts`. Use snake_case in worker messages.
Known modes are `text`, `markdown`, `yaml`, `json`, `jsonl`; storage also preserves
unknown future strings. Unsupported modes never fall through to JSON/YAML parsing.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class RequestKey:
    authority_id: str
    document_id: str
    session_generation: int
    draft_revision: int
    language_revision: int
    request_id: int

@dataclass(frozen=True)
class DocumentRequest:
    key: RequestKey
    text: str
    language: str | None
    newline: Literal["\n", "\r\n"] = "\n"
    has_bom: bool = False
    is_excerpt: bool = False

@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning"]
    message: str
    start: int
    end: int

@dataclass(frozen=True)
class CheckResult:
    key: RequestKey
    state: Literal["checked", "unsupported", "resource_limited", "failed"]
    complete: bool
    diagnostics: tuple[Diagnostic, ...]
    truncated: bool
    format_allowed: bool

@dataclass(frozen=True)
class FormatResult:
    key: RequestKey
    state: Literal["formatted", "unchanged", "blocked", "resource_limited", "failed"]
    replacement: str | None
    check: CheckResult
```

Python exports `check_document(request: DocumentRequest) -> CheckResult` and
`format_document(request: DocumentRequest) -> FormatResult` from `engine.py`.
TypeScript exports `checkDocument(request: DocumentRequest): CheckResult` and
`formatDocument(request: DocumentRequest): FormatResult` from `engine.ts`.
Format runs an authoritative check itself and refuses nonqualified input.
`replacement` is present only for `formatted`; `unchanged` creates no edit.

Editor adapters provide `apply_format(result, current_key, editable) -> bool`
(TypeScript `applyFormat`) through their editor-specific bridge. They check exact
key equality, current authority/conflict/editability, and replacement difference,
then perform one history transaction. Worker results never save directly.

## Verification commands and evidence

Run Python selections from C or S with the repository's existing environment:
`python -m pytest <explicit test files> -q`. Do not install dependencies globally.
Run shared UI tests from U using `bun run test <explicit test files>`; its script
uses `vitest run` and its local config. Run browser history tests from W using
`bunx playwright test e2e/notes-structured-editing.spec.ts --project=chromium`.
Use available browsers to cover Firefox/WebKit before claiming support there;
missing browsers are unverified, not skipped-as-passed.

Each unit follows red → minimal change → targeted green → diff review → scoped
commit. Use exact touched Python paths for Ruff/formatter checks and server Bandit,
and exact TypeScript paths for ESLint/Prettier checks from W; do not mass-format
large existing modules. Run the existing frontend typecheck after a shared type
change. Classify inherited failures against the untouched base rather than
silently raising thresholds. Commit only the unit's explicit file list, never `-A`.

## Spec coverage map

| Spec criterion | Owning plan units |
| --- | --- |
| AC1 editing and diagnostics | E1–E4, I2, I3 |
| AC2 invalid/empty draft preservation | F5, I2, I3, I5 |
| AC3 exact source round trips | F1, F2, F3, F4, F6, I5 |
| AC4 full YAML and existing file authority | F1, I2 |
| AC5 preservation and idempotence | E1, E2, E3 |
| AC6 normal undo and stale results | I1, I2, I3 |
| AC7 durable language and conflicts | F2–F5, I2, I3 |
| AC8 mixed-version compatibility | F3, F4, F6, I5 |
| AC9 common language fixtures | E1, E2, E3 |
| AC10 offsets and truthful states | E1–E4, I2, I3 |
| AC11 bounded work and responsive UI | E4, I5 |
| AC12 source-safe secondary editors | I3, I4 |
| AC13 downloads and file representation | F1, F6, I4 |
| AC14 privacy and parser safety | F4, F5, E2–E4, I5 |
| AC15 targeted and live evidence | Every unit; final I5 release report |

Self-review: checked against all 15 spec criteria, verified existing integration
paths, separated new files from existing files, and retained explicit gates for
unqualified libraries/history. No test result or migration version reservation is
claimed by this document. Recheck schema/version allocation at implementation
time because both repositories change concurrently.
