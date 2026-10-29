# Character Chat Selection Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline.

**Goal:** Preserve the current provider/model and character chat through launch and modal resume.

**Architecture:** Resolve character-launch defaults from a worked source session's
pair, preserving the existing pristine/no-source behavior. Both character launch
routes use the existing workspace-alignment seam.

**Tech Stack:** Python 3.12, Textual 8, SQLite, pytest.

**Spec:** ../specs/2026-10-29-character-chat-selection-design.md

ADR required: no

ADR path: backlog/decisions/006-provider-aware-generation-settings.md;
backlog/decisions/095-conversation-owned-console-generation-settings.md.
Workspace alignment follows the existing TASK-15120 ruling.
Reason: restore existing source-owned generation and activation contracts.

## Global constraints

- Ordinary model Apply affects its exact chat only.
- Preserve canonical default convergence for untouched sessions.
- Character cards own prompts and greetings; source conversations remain intact.
- Use targeted verification; do not run a full suite without user opt-in.

## Review focus

- A worked chat must outrank stale controls even when saved defaults differ too.
- A pristine chat must still adopt changed canonical defaults.
- Both Personas handoff and Console picker must align the actual target workspace.
- Duplicate handoffs and modal cancellation must preserve the selected character chat.
- A reply must reach the selected provider/model with character context.

## Task 1: Repair character launch

Files: UI/Console_Modules/session.py, character.py, wiring.py;
Tests/UI/test_console_session_controller.py; Docs/Architecture/character-chat.md.

Interfaces: reuse ConsoleChatStore.session_settings/is_pristine_session,
default_console_session_settings, and the existing workspace alignment callback.

- [x] Add failing parameterized regressions for both launch routes with stale
      controls, saved/default mismatches, and a named source workspace.
- [x] Run the new cases and confirm provider and registry assertion failures.
- [x] Add one shared character-default resolver; retain the existing default
      resolver for pristine or missing source settings.
- [x] Wire both routes to the resolver and align the registry at activation.
- [x] Run the affected character/session/workspace tests and scoped static checks.
- [x] Run actual-app UAT with an isolated profile and local HTTP provider,
      exercising launch, Apply, Cancel, and reply; retain screenshots and receipts.
- [x] Obtain an independent review and resolve actionable findings.
- [ ] Update task, architecture notes and verification evidence; commit and open
      a PR against dev.
