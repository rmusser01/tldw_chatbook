# Post-Release UX/HCI Walkthrough Evidence

## Metadata

- Task: `TASK-60.4.4`
- Screen or workflow: Library/Search-RAG evidence to Console answer, saved Chatbook artifact, Home/Artifacts resume metadata, and exported/imported Chatbook citation payloads
- Date: 2026-05-24
- Branch: `codex/task6044-citation-qa-closeout`
- App command: `python -m tldw_chatbook.Web_Server.serve --host 127.0.0.1 --port 8871 --title "Citation QA Closeout"`
- Evidence method: textual-web/CDP screenshot for rendered Console baseline, plus focused mounted/functional tests for source-to-answer-to-artifact carry-through
- Actual screenshot path: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-24-citation-snippet-closeout-textual-web.png`
- Screenshot approval: not requested; no visual UI change in this tranche
- Reviewer: Codex

## User Goal

A user should be able to retrieve evidence from Library/Search-RAG, stage it into Console, get an answer with visible citation markers, save or resume the answer as a Chatbook artifact, and export/import the Chatbook without losing source identity, snippet text, source authority, or machine-checkable citation metadata.

## Steps Attempted

1. Started the actual app through textual-web and captured the rendered Console surface with CDP/browser automation.
2. Verified Library/Search-RAG handoff payloads preserve snippets, citation labels, provenance, local/server authority, and workspace eligibility.
3. Verified Console staged context exposes evidence summary and citation instructions without leaking raw `evidence_bundle` metadata.
4. Verified streaming and non-streaming Console answers validate `[S#]` citation labels against the staged evidence bundle.
5. Verified saved Chatbook artifact metadata surfaces compact citation/evidence summaries for Home and Artifacts resume payloads.
6. Verified exported Chatbook ZIPs preserve citation validation, evidence bundle, snippets, readable citation reports, manifest metadata, and import-side citation sidecar hydration.

## What Worked

- Library/Search-RAG evidence bundles keep source identity, snippet text, citation labels, authority labels, and active-context eligibility through the handoff boundary.
- Console model context includes explicit citation instructions and staged evidence, and the answer validator marks validated, unknown, uncited, blocked, stale, missing, or insufficient-evidence states.
- Saved Chatbook artifacts expose citation status, cited IDs, evidence bundle ID, source count, and snippet count for Home/Artifacts resume.
- Chatbook exports include both machine-checkable JSON metadata and user-readable Markdown citation reports.
- Importing a Chatbook with exported citation payloads persists the citation/evidence payloads back into the conversation RAG context sidecar.

## What Broke Or Slowed The Workflow

- The rendered textual-web screenshot only proves the Console surface loads; the complete evidence chain is verified through mounted and service-level functional tests rather than a live LLM/RAG backend session.
- No live external server or remote sync target was used in this closeout. Server-authority payloads are covered by focused tests, but full server-backed sync remains outside `TASK-60.4.4`.

## Nielsen Norman Heuristic Findings

- Visibility of system status: Console and artifact metadata now distinguish validated, unverified, insufficient-evidence, and blocked citation states instead of silently dropping provenance.
- Match between system and real world: Source authority, snippets, and `[S#]` citation labels use terms users can map back to retrieved evidence.
- User control and freedom: Users can save/resume/export cited answers without losing the underlying evidence payload.
- Consistency and standards: The same evidence bundle and citation validation payload shape carries through Library, Console, Home, Artifacts, export, and import.
- Error prevention: Blocked/cross-workspace evidence is marked unavailable for grounding and cannot be cited as trusted evidence.
- Recognition rather than recall: Saved artifact summaries expose citation/evidence counts and IDs so users do not need to remember which sources were used.
- Flexibility and efficiency of use: Export/import keeps full machine-checkable payloads for downstream automation while readable reports support manual review.
- Aesthetic and minimalist design: No new visual chrome was added in this tranche; the UI continues to surface compact summaries rather than full raw payloads.
- Error recognition, diagnosis, and recovery: Unknown, uncited, blocked, and insufficient-evidence states include explicit validation status and recovery copy.
- Help and documentation: This evidence file and task closeout document the supported path and the remaining server/live-backend boundary.

## Keyboard And Focus Findings

- No new keyboard or focus behavior changed in this tranche.
- Existing Console composer/focus behavior remains governed by the previously approved Console UI work and screenshot evidence.

## Empty Error Setup State Findings

- Missing or ineligible evidence stays recoverable: the user sees an insufficient/blocked evidence state rather than an apparently grounded answer.
- Optional dependency and package recovery are covered by `TASK-60.4.5`, which is already complete and keeps advanced RAG/server features source-honest.

## Cross-Screen Handoff Findings

- Library/Search-RAG to Console: verified by handoff payload tests and Console staging tests.
- Console answer to saved Chatbook artifact: verified by citation summary and Home/Artifacts active-work adapter tests.
- Saved/exported Chatbook to downstream reuse: verified by export/import tests preserving citation payloads and readable citation reports.

## Power-User Repetition Findings

- Shortcuts: no new shortcut behavior added.
- Batch actions: export preserves all conversation-level citation payloads instead of requiring per-message manual copy.
- State persistence: evidence bundles and citation validation survive save, resume, export, and import.
- Recovery paths: invalid or unavailable evidence remains visible as validation metadata instead of being silently trusted.
- Repeated-use friction: citation reports are bounded for readability while full evidence remains in JSON for automation.

## Severity Decisions

| Finding | Severity | Follow-Up Task | Decision |
| --- | --- | --- | --- |
| Full live backend RAG/LLM session was not used for closeout | P2 | `TASK-60.4.1` / future server-sync runtime work | Accept for `TASK-60.4.4`; source-to-answer-to-artifact contract is covered by focused app/service tests and rendered-app baseline evidence. |
| Server-backed handoff/sync is not part of this tranche | P2 | `TASK-60.4.1` / future sync work | Accept; this tranche closes local carry-through and export/import preservation only. |

## Acceptance Decision

- Accepted: yes
- Reason: The citation/snippet payload contract now carries source identity, snippets, authority, validation status, saved artifact summaries, readable export reports, and import-side machine-checkable metadata through the full local workflow.
- Required follow-up before acceptance: none for `TASK-60.4.4`.
