# Chatbook Buddy burndown implementation plan

Goal: import and explicitly publish Migu for a local Persona, choose Use for Buddy, and operate its floating companion across Chatbook screens.
Architecture: retain ADR-074's Persona Visual repository/resolver and local Persona authority. One app-owned controller holds immutable selection, source leases and preferences; disposable native Textual views render prepared frames without owning authority.
Tech stack: Python 3.12, Textual 8, Pillow/Rich, existing SQLite and profile-local configuration.
Spec: Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md (Persona Visual/Buddy sections).
ADR required: no.
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md.
Reason: implement the existing approved runtime/UI boundary; no schema or dependency change.

User authorized implementation after the UAT report. Preserve all unrelated dirty changes. Work in the affected current checkout to repair its observed regression; no blanket staging or commits. Execute inline, with the independent publication repair delegated under the parallel-work skill.

## 1. Startup and publication repairs
- [x] Re-run the actual mounted-authoring import failure, correct the misspelled disclosure import, and run the mounted tests.
- [x] Add failing production-shaped import/replace/edit→Save tests using real SQLite, fix source-root/publication compatibility without weakening containment, rerun targeted authoring/publication tests.
Files: Widgets/Console/console_workspace_context.py; Persona_Visual/publication.py and authoring source handling; Tests/UI/test_personas_persona_visual_authoring.py. Publication regression tracked in TASK-19054.

## 2. Controller and preferences (TASK-19055)
- [x] Add Persona_Visual/buddy.py with explicit local selection, default-off preferences, bounded geometry, source-scoped leases and pinned precedence. Test two simultaneous sources, expiry, disabled/missing personas and no implicit retargeting before implementation.
- [x] Resolve/revalidate local record and exact binding identity off-thread. Shield and drain blocking preparation under one controller lock. Fence selection, generation and geometry after awaits. Test cancellation/replacement using barriers.
- [x] Persist only local selection/window preferences; preserve enabled state while unavailable. Clear transient leases on restart.

## 3. Native view and application wiring
- [x] Add Widgets/Persona_Widgets/persona_buddy.py with absolute screen overlay, raster frame preparation, no layout budget, labelled collapse/close/move/resize/reset controls, focus preservation and viewport clamp.
- [x] Add app lifecycle host wiring to mount only on primary screens, hide for modal/auth/splash/recovery screens, replace views on navigation and drain on shutdown.
- [x] Add explicit Use for Buddy in local Workbench and use trusted existing Console activity for state leases; server-source and highlighted persona changes must not retarget selection.
- [x] Add tests for painted normal/wide/80x24 layouts, resize, hidden animation, reduced motion, modal hit testing and stale view work.

## 4. Verification and handoff
- [x] Run import-provenance plus affected controller/runtime/authoring/widget/app tests and scoped Ruff/format/compile/diff checks.
- [x] Mutation-check authority, lease, cancellation, geometry and modal guards.
- [x] Launch isolated real app with HOME/XDG/config/data set before imports; import/publish Migu through production flow and exercise selection, navigation and window controls. Report any physical terminal mouse verification limits explicitly.
- [x] Perform Impeccable review on final visible UI, update UAT evidence and task notes; check only demonstrated acceptance criteria. No full test sweep without user request.

## Verification outcome

Implementation and the listed verification runs are complete; complete acceptance sign-off remains open. The real isolated app imported and saved Migu, explicitly selected its Buddy, painted frames, and passed keyboard geometry/collapse checks. Live terminal checks verified restart persistence, mouse collapse/close, modal ownership, primary-screen navigation, and viewport clamp. Physical drag/resize and live provider/audio behavior remain unverified. Main Persona Visual/view/editor run: 594 passed. Final host/controller integration: 47 passed (22 overlap). CSS/source sync and scoped static checks pass. Diagnostic inventory/ledger and design-governance failures remain; AC8 is not checked and tasks19054/19055 remain In Progress. See qa/buddy-uat-2026-09-04/repair-report.md for evidence and limitations.

Bounded additions: restore the missing Console rail-height accessor and repair the existing appearance-button width/padding crash exposed by the targeted rail tests. These restore the Console surface needed for app-wide Buddy UAT; no new architectural decision.

## Authorized follow-up closeout — 2026-09-05

- [x] Native parser and release-position regressions; real Terminal move/resize.
- [x] Mounted real DeepSeek send, setup refusal and draft-recovery repair, idle return.
- [x] Real local speech recognizer through mounted dictation with a synthetic recorder fixture.
- [x] Diagnostic environment pruning, reviewed metadata repairs, current inventory reconciliation.
- [x] Existing Console controller extraction and lower screen ratchet; approved token catalog and CSS sync.
- [x] Actual microphone and local transcription; real local Kokoro synthesis/output sink, readback lifecycle repair, Buddy idle return.
- [ ] Configured OpenAI realtime UAT (credential absent).

ADR required: no new ADR.
ADR paths: existing ADR-074 (Buddy), ADR-069 (project-instruction refusal), ADR-029
(diagnostics), ADR-042 (tokens), and the approved DESIGN.md section 7 decomposition
contract. These repairs preserve their existing authority and ownership boundaries.
Evidence, task closeout, targeted verification and precise limits:
`qa/buddy-uat-2026-09-04/followup-report.md`.

Voice continuation: existing ADR-037 trusted message speech now also governs readback; two born-red lifecycle tests and mounted real Kokoro output validate its stopped/failed cleanup. See the follow-up report for exact hardware evidence and remaining credential gate.
