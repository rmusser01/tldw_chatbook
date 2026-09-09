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
| CB-3 | P1 | Personas inspector enables Use for Buddy but disables current independent Buddy actions with Select the Persona currently used by Buddy. Legacy action writes Persona-owned selection. | Shared management entry and controls for the current independent Buddy; remove the obsolete Persona-owner selection transaction. | Mounted Manage/Show/Close/Disable actions at 80x24 and 120x40; lower-layer migration/attribution coverage retained. |
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

## Remediation result

All eight findings are addressed within the existing ADR-139/ADR-079 contracts. Preview and geometry use scoped flex rules; optional authoring is disclosed after Follow/Persona. Current assignments are named literally, including names containing brackets. Apply retains staged values while saving and recovering, prevents duplicate submission/dismissal, and guards import publication and preference revisions. The Personas inspector manages independent Buddy state without requiring a Persona selection. Workspace activation and acknowledgement share the same freshness gate.

Conversation interaction opens at current content, follows replies only while the reader is at the end, preserves deliberate reading, and exposes Latest/new updates plus a direct pending-decision jump. Empty conversations render their status, and speech-off is compact. Full production CSS initially exposed a second compact defect: the nested approval body clipped buttons despite correct outer coordinates. Buddy-scoped auto-height and flexible decision controls now paint Approve once and Deny at 60x20 and 80x24; the regression checks ancestor clipping as well as position.

Final verification is incremental over frozen production changes:

- Broad affected Buddy library/controller/UI/journey, Personas inspector, CSS integrity and assignment/default group: **384 passed**.
- After literal-name and smallest-screen scroll fixes, affected management and layout modules: **20 passed**.
- After the final scoped approval CSS fix, transcript/layout and CSS integrity group: **37 passed**. Counts overlap and must not be added as unique coverage.
- Coordinator recovery suite: **17 passed**, including a saved-Buddy/failed-Persona outcome and safe retry that keeps the existing Persona assignment.
- Startup census and creation-default group: **30 passed**, actual UI-ready imports **973/973** with no budget increase.
- Active Buddy lookup has a real SQLite plan assertion without sqlite_stat1; the index census passes with **282 entries / 67 pins**.
- Eleven focused Python files pass Ruff and formatting. Baseline-aware lint across all changed Python files introduces no diagnostics (361 before, 358 after); large existing files retain formatting debt. Generated CSS and whitespace checks pass.

Independent code review raised the literal-name parser issue and confirmed its correction. A subsequent scoped review inspected the final 60x20 render and ancestor-aware regression and found no remaining Important or Critical issues in the fixes. Reviewers did not rerun implementer tests; root separately inspected the final compact approval and management previews. The pre-fix heuristic score remains 24/40; no new score is inferred from passing tests.

The previous CI head's unrelated MCP and GGUF cases both pass exact local reruns on merged dev, but macOS execution does not verify the Windows CI job. No full suite, real provider/microphone/audio or packaged-terminal user session was run. Current CI is reported on the PR.

See [implementation plan](../../superpowers/plans/2026-09-08-chatbook-buddy-ux-review-remediation.md) for scope and architectural references.
