---
target: tldw_chatbook/Widgets/Persona_Widgets/buddy_management_modal.py
score: 24
max_score: 40
p0: 0
p1: 4
method: dual-agent
timestamp: 2026-09-09T00-37-01Z
slug: widgets-persona-widgets-buddy-management-modal-py
---
Method: dual-agent (A: chatbook_design_review · B: chatbook_evidence_review)

# Native Chatbook Buddy and Persona review

Reviewed commit `42f6e22461` on PR [2526](https://github.com/rmusser01/tldw_chatbook/pull/2526), including current dev `565dc49921`. Scope: native Textual Console Menu → Buddy, conversation interaction, workspace inbox/default Persona, and Personas inspector. Flashcards is excluded. The server/shared WebUI work is tracked separately in PR [2933](https://github.com/rmusser01/tldw_server/pull/2933).

## Design assessment

The flow fits Chatbook's terminal workbench: named explicit targets, preserved drafts, meaningful activity groups and independent artwork support controlled work without leaving the current destination. Its main weakness is access to the current state: preview controls can be invisible, old transcript content hides current replies and decisions, and the current Persona is unnamed. Legacy Persona-owned Buddy actions contradict the new model.

These are pre-remediation scores, not a claim about the corrected implementation:

| Heuristic | Score / 4 | Evidence |
| --- | --- | --- |
| Visibility of system status | 2 | New replies/decisions and current assignment hidden |
| Match with real-world expectations | 3 | Explicit conversation/workspace terms |
| User control and freedom | 3 | Cancel, drafts and navigation continuity |
| Consistency and standards | 2 | Personas still advertises old ownership |
| Error prevention | 3 | Explicit identities and validation |
| Recognition rather than recall | 2 | Current assignment requires reconstruction |
| Flexibility and efficiency | 2 | Preview clipped; repeated transcript scrolling |
| Aesthetic and minimalist design | 2 | Secondary controls consume compact reading space |
| Error recovery | 2 | Import failure loses staged corrections |
| Help and documentation | 3 | Useful scope copy; impossible legacy recovery instruction |
| Total | 24/40 | Acceptable, with major gaps to correct |

## Findings and corrections

| ID | Priority | Reproduction and user impact | Approved correction | Verification target |
| --- | --- | --- | --- | --- |
| CB-1 | P1 | Preview is outside the body clip at normal and compact sizes. Focus reaches an invisible button; click fails. Broad Select width overrides flex. | Scope flex widths to preview and geometry rows; reserve button and labels. | Actual pointer activation, visible keyboard focus and compact geometry. |
| CB-2 | P1 | Twenty-message conversation opens at message 00. New reply increases scroll extent without following; pending approval is below all history. | Open at current content; follow when already at end, preserve deliberate reading, expose Latest/new updates and direct pending-decision access. | Long history, new reply, reader scrolled away, actual pending decision. |
| CB-3 | P1 | Personas inspector enables Use for Buddy but disables current independent Buddy actions with Select the Persona currently used by Buddy. Legacy action writes Persona-owned selection. | Shared management entry and independent artwork copy; manage current Buddy without requiring a Persona owner. | Mounted independent selection and artwork-copy persistence. |
| CB-4 | P1 | Missing ZIP plus changed width: Apply dismisses, exposes persona_visual_import_invalid, and reopening loses both changes. Saved preferences remain intact. | Keep staged dialog during validation/apply; actionable recovery, duplicate-submit guard, accurate partial-save outcome and safe retry. | Real library failed import, preserved values, successful correction, partial Persona failure. |
| CB-5 | P2 | Keep current assignment gives no name/None/unavailable state, including workspace defaults. | Display effective current Persona/default for explicit target while keeping unchanged and explicit None distinct. | Conversation and workspace resolution including missing/inactive default. |
| CB-6 | P2 | Blank transcript equals initial cached empty string, suppressing No messages yet. At 60×20 speech-off controls leave two transcript rows. | Initialize empty state correctly and compact speech-off presentation; keep active queue and consent controls. | Empty and compact dialogs with speech off/on/consent. |
| CB-7 | P2 | After failed inbox refresh, Open is disabled but Enter invokes retained row opener. This is inconsistent gating, not demonstrated authority bypass. | Apply freshness gating to every pointer/keyboard open and acknowledgement path; recover on fresh snapshot. | Failure → Enter/click → recovery without acknowledgement or focus theft. |
| CB-8 | P2 | Import and size details precede Follow/Persona; first-time setup presents too many unrelated controls together. | Progressively disclose optional import/geometry; keep installed preview and Dynamic/Static readily accessible. | First-time ordering and keyboard discovery of advanced fields. |

CB-1, CB-4 and CB-6 were independently reproduced by both assessments. CB-2, CB-3 and CB-5 came from A's native workflow review; B independently found CB-7. CB-8 is A's cognitive-load recommendation applied within the approved design.

## Workflow walkthrough

A first-time user can open Console Menu → Buddy, select built-in Pixel Migu, preview, Apply and interact without creating a Persona. In the reviewed baseline, the missing Preview button breaks the safe trial and the failed-import path discards effort. Optional authoring details should follow the common selection path.

An experienced user benefits from exact-target replies while another Console conversation is selected and from separate durable drafts. Reopening at old history and hiding pending decisions adds repeated manual scanning. Keeping current content visible must not pull a reader away from older context they deliberately opened.

A workspace user can group Needs you, Running and Results without marking them read. The default-Persona explanation correctly limits changes to future conversations, but does not identify the current effective default. Voice input remains unavailable for workspace interaction; named queued read-aloud is separate.

A keyboard or compact-terminal user can reach Apply/Cancel and recover focus after Escape, but invisible Preview focus and limited transcript height prevent efficient use. The correction uses terminal cells and native Textual behavior rather than applying browser typography metrics.

## Strengths preserved

- Fresh independent artwork selection leaves Persona records unchanged; Follow remains fixed across navigation.
- Actual off-screen reply tests preserve both Buddy and Console drafts, route to the exact live/saved target and keep work alive after close.
- Inbox projection is scoped and read-only on open. Explicit result acknowledgement and supported approvals retain their own authority.

## Evidence and limits

A used isolated full-app profiles and supplementary real-widget/controller probes, with PNG/SVG captures at 120×40, 80×24, 60×20 and long-history probes at 100×36. B mounted three dialogs using production stylesheet sources at 80×24 and 140×45, inspected 17 SVG/compositor captures, traversed focus, and exercised validation, failed refresh and draft recovery. A's direct approval probe used the real approval boundary outside a provider run; its idle label does not establish a normal running approval's status.

The detector ran once and returned exit 0 / `[]`. Python/TCSS are unsupported, so this result provides no native accessibility assurance. Browser overlay injection is inapplicable to a Textual application; mounted native rendering supplied the evidence. B's PNG conversion failed because Cairo/native SVG conversion was unavailable; A's PNGs were rendered with local Menlo. No real-terminal mouse, screen reader, measured native contrast, provider turn, microphone or audio claim is made.

Before fixes, root's targeted combined baseline passed 215 tests in 111.33 seconds. A also passed three real-flow tests and B passed 46 targeted tests; these overlap the baseline and must not be added as unique coverage. No full repository suite was run.

Questions skipped: the user already requested correcting all findings and updating the PR against dev. The existing Buddy design remains the implementation authority.
