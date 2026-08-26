---
id: TASK-22507
title: Resume prior character chats from Roleplay
status: In Progress
assignee: []
created_date: '2026-08-26 15:14'
updated_date: '2026-08-26 15:31'
labels:
  - roleplay
  - console
  - ux
  - conversations
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-26-roleplay-resume-prior-character-chat-design.md
  - >-
    backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users who discover a saved local character conversation in Roleplay need to continue that authoritative chat in Console, rather than only copying a bounded transcript into draft context. Resuming must preserve the saved conversation and historical character behavior while keeping Roleplay as a read-only discovery surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting a recent saved local character conversation opens its read-only preview, hides card-level actions, and Back to card restores them.
- [ ] #2 The preview exposes Resume chat as the sole primary action in the approved three-row vertical hierarchy, with contained controls and usable transcript space at 80x24 and standard widths.
- [ ] #3 Preview loading, empty, and failure states are distinct; Resume remains available from a valid row even when preview text cannot load.
- [ ] #4 Resume passes only the validated local conversation ID to Console and never aliases the bounded transcript handoff or RAG scope.
- [ ] #5 An already-live matching Console session is activated without duplication and preserves its live draft and settings; the active matching duplicate wins.
- [ ] #6 A closed conversation is restored through the canonical Console path with its saved tree, active leaf, prompt, roleplay provenance, policies, speech preferences, and pinned prefill within existing safety limits and without a provider call.
- [ ] #7 Earlier pending Console intents reach their existing terminal or transient-release outcome first, after which Resume becomes the final active-session target and the Console composer receives focus.
- [ ] #8 Missing, failed, or cancelled resume preserves the prior active Console session and durable conversation and removes only a partial runtime session created by that attempt.
- [ ] #9 Historical character behavior remains authoritative: version-2 metadata stores the character-name snapshot, version-1 conversations are not guessed or backfilled from the current card, and future versions remain fail-closed.
- [ ] #10 Send transcript to Console draft remains a separate bounded 6000-character context action, and Open in Library remains unchanged.
- [ ] #11 Targeted automated tests cover navigation, ordering, live-session reuse, hydration, failure atomicity, metadata compatibility, focus, contrast, and compact layout behavior.
- [ ] #12 ADR-046 is amended before implementation to record the metadata-version and historical character-name authority decision.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-046 before code to define roleplay metadata v2, historical character-name authority, v1 no-guess behavior, and no schema migration.
2. Add failing metadata and persistence tests, then write v2 roleplay context and persist the saved character-name snapshot through existing merge-safe seams.
3. Add failing hydration tests, then restore the saved character identity through canonical hydration and remove current-card name lookup from the resume path.
4. Add failing rollback and opener tests, then share exact runtime cleanup and make the canonical Console opener active-match-first, tri-state, and cancellation-safe.
5. Add failing navigation-order tests, then capture an ID-only pre-mount context and settle earlier pending Console intents before Resume becomes the final active target.
6. Add failing Roleplay UI tests, then implement distinct preview states, hidden card actions, the three-row Resume hierarchy, single-flight navigation, focus behavior, compact layout, and generated CSS.
7. Run the documented targeted feature gate, regenerate consolidated CSS, self-review against all acceptance criteria, add implementation notes/evidence, and mark Done only when the repository Definition of Done is satisfied.

Detailed executable plan: Docs/superpowers/plans/2026-08-26-roleplay-resume-prior-character-chat-implementation.md
<!-- SECTION:PLAN:END -->
