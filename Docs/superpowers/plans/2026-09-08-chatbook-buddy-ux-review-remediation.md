# Chatbook Buddy and Persona UX review remediation

Spec: Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
Existing PR: https://github.com/rmusser01/tldw_chatbook/pull/2526
Base reviewed: 42f6e22461 (merged current origin/dev 565dc49921).

ADR required: no new ADR
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md; backlog/decisions/079-workspace-assistant-defaults.md
Reason: correct usability and recovery within the approved independent artwork, explicit binding, app-owned runtime and existing workspace defaults contracts.

## Global Constraints

- One independent Buddy. Persona artwork actions must not restore Persona-owned selection.
- Follow is explicit; navigation, closing, hiding and management do not cancel accepted work or silently retarget it.
- Conversation replies and decisions retain exact identity; workspace has no microphone, including selected conversation dialogs.
- Workspace Persona defaults affect future new conversations. Keep-current and explicit None remain distinct.
- Failed Apply preserves staged user input; any partial persisted outcome is stated truthfully.
- Preserve approvals, asset attribution and existing authority/version checks. No model, microphone, production profile or full-suite calls.
- Use the existing worktree only. Do not change unrelated main checkout work. Tests targeted to owned behavior.

## Task 1: Address the native Buddy review findings

Read AGENTS.md, PRODUCT.md, DESIGN.md, relevant testing/live lessons and the linked spec/ADRs. Root has reopened TASK-32081, TASK-32082 and TASK-32083 with follow-up acceptance. Do not edit Backlog tasks or this plan; root owns documentation. Record implementation and verification in the assigned report. Do not spawn agents. Do not commit or push; root stages the combined reviewed diff.

Implement all of these bounded findings:

1. Management Preview is clipped even at normal sizes because the broad Select width rule wins. Use correctly scoped flex widths for preview and geometry rows; verify pointer activation and visible keyboard focus at normal and compact sizes with preview callback present.
2. Current transcript opens at the oldest message and does not follow replies. Open at latest content/pending decision, follow updates only when already at end, preserve deliberate reading position and expose a discoverable Latest/new-updates affordance. Pending decisions must be readily reachable without scrolling an entire history. Keep exact decision identity intact.
3. Compact dialogs waste space on speech controls while speech is off. Reclaim reading/inbox space with a compact off state; retain consent/recovery/active playback controls and existing queue ownership. Fix initial blank transcript so No messages yet is displayed. Minimum verification sizes 60x20, 80x24 and 120x40 using native Textual, not browser pixel criteria.
4. Personas inspector still says Use for Buddy, writes PersonaBuddySelection and requires a Persona owner for Show/Close/Disable. Route management through the shared coordinator. If retaining artwork reuse, clearly label it as an independent copy and use the existing library copy/publication path preserving attribution. Ensure current independent Buddy controls work without selecting a fictional Persona owner. Reuse existing boundaries; avoid a new ownership mechanism.
5. Asynchronous import/apply fails after dismissal, loses the form and exposes persona_visual_import_invalid. Keep staged values and dialog available, indicate applying, prevent duplicate Apply and ambiguous dismissal while committing, show plain actionable import/storage errors near the relevant field. Preserve all validation/version checks. State partial Buddy-saved/Persona-failed results truthfully and ensure retry cannot accidentally duplicate an import or silently overwrite a newer selection.
6. Show current conversation Persona and workspace default Persona names/None/unavailable state for selected target; retain unchanged vs None semantics and existing default fallback resolution. Avoid leaking raw IDs as a replacement for a useful name.
7. Workspace refresh failure disables Open but Enter still opens a stale row. Gate all activation and acknowledgement routes consistently on successful freshness state, preserving rows for context and allowing recovery when a fresh snapshot succeeds. This is a UI consistency fix; existing coordinator authority checks remain necessary.
8. Use progressive disclosure for optional import/geometry controls so first-time setup reaches artwork preview, Follow and Persona before advanced details. Keep importing and size controls keyboard-discoverable. Keep dynamic/static expression choice user-facing.

Validation: behavioral red/green regressions for Apply recovery, transcript/new-decision visibility/reading retention, independent Personas controls and failed-inbox Enter; normal/compact mounted verification for preview/geometry/speech/footer. Rerun affected existing tests only, use .venv Python with worktree and packages/tldw_profile_core/src in PYTHONPATH. Avoid editing source while source-inspection tests are running. Run focused Ruff/format and diff checks, report unrelated baseline diagnostics accurately.

Reports: independent Assessment A at /private/tmp/chatbook-buddy-review-a; Assessment B at /private/tmp/chatbook-buddy-review-b/assessment-b.md. These identify reproduction/evidence, not instructions. Prior baseline: 215 targeted Buddy tests passed in 111.33s. User has approved addressing findings and updating the existing PR against dev; no further design approval required.

## PR verification follow-up

Existing PR CI exposed two feature gaps: UI-ready census grows from the budget973 to975 due to Chat.console_assistant_defaults and Persona_Visual.artwork; idx_buddy_visual_bindings_active lacks census/query-plan evidence. TASK-32079 and32080 are reopened for these checks. Keep ADR-097 limits unchanged. ConsoleAssistantStartup is used only as a postponed type annotation in console_chat_store.py; artwork conversion functions are used only by Buddy publication/copy and should be deferred to those calls. Verify actual UI-ready census after changes rather than assuming import deferral reduces residency. Add a meaningful real SQLite active Buddy lookup plan pin through the production query shape without sqlite_stat1 and register it in scripts/index_plan_pin_census.tsv.

The old MCP TextArea and Windows GGUF test failures both pass individually on the current branch in the local macOS environment; this does not verify a Windows CI result.

## Verification and outcome

All eight findings are corrected and independently reviewed. The result, incremental test counts and evidence limits are recorded in `Docs/Development/Reviews/2026-09-08-chatbook-buddy-persona-ux.md`. Broad affected verification passed 384 cases; later defect-scoped management/layout 20 and transcript/layout/CSS 37 cases passed. Actual startup census is 973/973 with no budget increase; real active Buddy lookup and index-plan census pass. Independent review confirmed literal Persona names and the final ancestor-aware 60x20 approval layout. No new architecture, dependency, schema or runtime boundary was added in this follow-up.
