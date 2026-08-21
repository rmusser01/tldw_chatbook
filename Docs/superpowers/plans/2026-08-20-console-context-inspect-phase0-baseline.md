# Console Context and Inspect Phase 0 Baseline Implementation Plan

> **For agentic workers:** Execute this plan with the repository's test-first and verification-before-completion practices. Keep each production UI change in its own follow-up task.

**Goal:** Establish a trustworthy latest-`dev` baseline for improving the Console Context and Inspector columns without reversing accepted UX decisions or overlapping mobile work.

**Architecture:** Phase 0 changes only Backlog and user documentation. It records the current product contracts, separates confirmed defects from research hypotheses, and creates two atomic implementation tasks. Any future structural reorganization must pass a decision gate and, if selected, amend or supersede the applicable ADR before code changes.

**Tech Stack:** Python 3.11+, Textual 8.x, Backlog.md, Markdown

---

## Baseline

- Source branch: `origin/dev`
- Reviewed commit: `a1d6df3f89244e918a1fb12facbd4ed0d927c24c`
- Review perspectives: first-time technical, first-time non-technical, regular technical power user, regular non-technical power user
- Evaluation basis: Nielsen Norman Group heuristics, rendered Textual geometry, current source/tests, accepted ADRs, and completed Backlog decisions
- Baseline regression evidence: `Tests/UI/test_console_right_rail.py` and `Tests/UI/test_console_narrow_layout.py` pass on the reviewed commit (15 tests)

## Protected current contracts

These are constraints, not remediation candidates in this programme unless new behavioral evidence justifies a superseding decision.

1. Staged Sources remain in Inspector, above source readiness (TASK-400).
2. Left-rail section bodies are capped and scroll internally so all section headers remain visible (TASK-15110 owner ruling).
3. Horizontal collapsed handles are exactly `Context->` and `<-Inspect`; open header controls are exactly `<---------|Context` and `Inspect|--------->` (TASK-16001).
4. Explicit rail preferences survive compact rendering. They are honored only when a rail plus the 40-column usable-transcript floor fits: Context at 70+ columns and Inspector at 74+ columns (ADR-043 amendment).
5. Responsive resize transfers focus from a hidden rail to its visible reveal handle; manual collapse retains its established focus behavior (ADR-043).
6. Console does not own a native add-source picker. Library-to-Console handoff remains the staging route (ADR-017).
7. Phone, touch-target, hover, soft-keyboard, and served-browser work belongs to TASK-18911. This programme may test terminal widths but must not absorb that mobile scope.

## Finding ledger

Disposition vocabulary: **implement now** means a reproduced, decision-compatible defect; **research first** means a plausible usability issue that needs task-based evidence or a design decision; **remove** means the finding conflicts with an accepted contract or is not presently a defect; **coordinate** means another active task owns the relevant surface.

| # | Review finding | Evidence / current decision | Four-persona impact | Disposition |
|---|---|---|---|---|
| 1 | Exact 100-column Context-only layout can expand far beyond the viewport | Reproduced at 100x30: Context measured about 255 columns; transcript began around x=257 and Inspector handle around x=1362. Conflicts with ADR-043's grid-resolution intent. | Blocks all four personas; first-time users are least able to recover. | **Implement now** in a dedicated geometry task. |
| 2 | “Context” can imply sources even though staged Sources live in Inspector | TASK-400 deliberately moved sources; ADR-017 retains “Console context” as the left rail identity. | Strongest comprehension risk for both first-time personas. | **Research first** with terminology and card-sorting evidence; do not move Sources back. |
| 3 | Action-required Inspector content may be buried by long informational content | Source ordering and deep-overflow probe support the risk; no failure ordering study yet. | Highest risk for regular technical users managing approvals and first-time users responding to blocks. | **Research first**; test task-oriented ordering before changing it. |
| 4 | Seven peer left-rail sections create high scan cost | Current rail contains Sessions, Workspaces, Conversations, Model, Agent, Details, and conditional Character. TASK-14810 intentionally split the first three. | Affects all personas, especially first-time non-technical users. | **Research first**; validate grouping and progressive disclosure. |
| 5 | Nested section scrolling can feel cumbersome | TASK-15110 explicitly chose “cap sections, scroll inside” to keep every header on-screen and test-pinned it. | Power users may incur extra scrolling; first-time users benefit from visible headers. | **Remove** from remediation scope absent contrary behavioral evidence. |
| 6 | Inspector information hierarchy is weak across Sources, run state, tools, approvals, artifacts, and settings | Static hierarchy review; no observed task failure yet. TASK-400 pins Sources at the top. | All four personas; technical power users face the greatest density. | **Research first**, preserving Sources placement and action semantics. |
| 7 | Manual rail collapse can leave no focused widget | Reproduced, but ADR-043 explicitly preserves established manual-collapse focus behavior while requiring focus handoff only for responsive hiding. Existing tests encode this contract. | Mainly keyboard power users. | **Research first** as a proposed behavior change, not a bug fix. |
| 8 | Deep Inspector overflow lacks the product-standard `▼ more — scroll` cue | Reproduced: Inspector scrolls, but no fold hint is rendered. | First-time users may assume lower content does not exist; power users lose position feedback. | **Implement now** in a dedicated fold-hint task. |
| 9 | Single-pane rail discovery depends on status chips because edge handles hide | Current docs and ADR-043 define status chips as the narrow route. TASK-18911 owns served-phone reachability. | First-time users may miss the route; non-technical users are most affected. | **Coordinate** with TASK-18911 for mobile; research terminal discoverability separately. |
| 10 | Provider/model/session facts repeat across status chips, Context, and Inspector | Static content inventory; no evidence yet that redundancy is harmful rather than useful recognition. | Power users may perceive clutter; first-time users may benefit from confirmation. | **Research first** with task-frequency and glanceability evidence. |
| 11 | Selection or active-context state may not clearly explain what Inspector is inspecting | Heuristic inference; not yet reproduced as a wrong-context action. | All personas when switching tabs or conversations. | **Research first** with cross-tab and cross-conversation task tests. |
| 12 | “Attach context” sounds like a picker but opens the Context rail; staging happens in Library | Current behavior and docs are intentional; ADR-043 shares the rail-switch path. | Strong terminology risk for first-time users, technical and non-technical. | **Research first** for copy; preserve behavior until evidence supports a rename. |
| 13 | Empty Sources state lacks a direct recovery action | ADR-017 rejected a Console-native add-source picker; Library handoff owns staging. | First-time users need a path forward. | **Research first** for a Library deep link or clearer instruction; no embedded picker without a superseding ADR. |
| 14 | “Scope” depends on technical retrieval vocabulary | Current scope row and picker are documented but not validated with non-technical users. | Highest risk for non-technical personas. | **Research first** for inline explanatory copy. |
| 15 | Collapsed Inspector badges are terse and may be hard to interpret | Abbreviations are width-budgeted (`1 appr`, `art`); tooltips preserve full meaning. | First-time and non-technical users may not decode them. | **Research first** within fixed geometry. |
| 16 | Source-related vocabulary varies among Sources, context, evidence, readiness, and RAG | Confirmed in UI/docs; TASK-400 reflects distinct underlying concepts, so simple global renaming may be wrong. | All personas, strongest for first-time non-technical users. | **Research first** with a concept model and terminology matrix. |
| 17 | User guide describes the pre-TASK-14810 rail and pre-TASK-16001 controls | Confirmed documentation drift. | Misleads all personas and support workflows. | **Implement now** in Phase 0 as factual documentation correction. |
| 18 | Status-chip density can overflow or obscure rail-entry actions | Existing responsive/status work spans earlier tasks; TASK-18911 owns phone tapability. No new terminal regression reproduced in this review. | Narrow-terminal and phone users. | **Coordinate** with TASK-18911; file only if a distinct terminal defect is reproduced. |
| 19 | Technical terms such as RAG, MCP, ACP, and prefill lack consistent local explanation | Documentation offers detail, but the dense rails rely on recognition. A prior terminology pass exists. | Non-technical personas, especially first-time. | **Research first**; prefer concise contextual help over more persistent chrome. |
| 20 | Visual hierarchy relies on many similar bordered sections | ADR-017 mandates a text-only bordered-section language. | Scan speed issue for all personas. | **Research first** within the accepted visual language. |
| 21 | Rail presentation has limited visual differentiation or brand character | ADR-017 intentionally rejected icons/glyph expansion for this pass. | Primarily aesthetic; lower task impact. | **Remove** from current remediation scope. |
| 22 | ASCII directional rail controls look unconventional | TASK-16001 explicitly approved and test-pinned the exact labels and full-width hit targets. | Possible first-impression concern, but the current behavior is intentional. | **Remove** from implementation scope. |
| 23 | Some sections may be irrelevant outside active agent or character modes | Current Agent and conditional Character surfaces are mode-sensitive; no task failure study yet. | Non-agent and non-character users may see avoidable chrome. | **Research first** before changing visibility or persistence. |
| 24 | Collapsed Inspector's persistent strip competes with transcript width | The handle is the designed discoverability affordance and carries action badges; narrow single-pane mode already hides it. | Power users may value the width; first-time users need discovery. | **Research first**; preserve the current route meanwhile. |

## Execution tasks

### Task 1: Reconcile the current documentation and Backlog state

**Files:**
- Modify: `Docs/User_Guide/console.md`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `backlog/tasks/task-14810 - Separate-Console-rail-Sessions-Workspaces-and-Conversations.md`
- Modify: `backlog/tasks/task-19638 - Reconcile-Console-Context-and-Inspect-UX-baseline-before-remediation.md`

1. Replace the obsolete single Session block description with the current Sessions, Workspaces, and Conversations sections.
2. Replace obsolete glyph-only rail controls with the exact current full-width and collapsed labels.
3. State the 70/74-column explicit-open floors and retained preference behavior without promising that rails open at every width.
4. Add a dated TASK-14810 reconciliation note based on fresh verification; do not mark Done unless its recorded repository gates are supportable.

### Task 2: File the two confirmed implementation slices

**Files:**
- Create: `backlog/tasks/task-19639 - Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns.md`
- Create: `backlog/tasks/task-19640 - Add-an-Inspector-overflow-fold-hint.md`

1. Re-sweep all remote refs, worktrees, and local task files immediately before filing.
2. Make each task depend on TASK-19638.
3. Keep the geometry task terminal-specific and aligned with ADR-043.
4. Keep the hint task limited to overflow discovery; do not reorder Inspector content.

### Task 3: Verify Phase 0

1. Run Backlog duplicate-ID and task-format checks.
2. Run documentation link or formatting checks available in the repository.
3. Run `git diff --check`.
4. Re-run the focused Console rail tests to prove documentation/task edits did not alter the baseline.
5. Record verification in TASK-19638 and close only if its completion gates are honest.

## Decision gate before structural IA work

Do not proceed from the two confirmed fixes into hierarchy or terminology changes automatically. First collect task-based evidence across all four personas for: finding the right rail, attaching/staging a source, narrowing scope, responding to an approval, switching workspace/conversation, and returning to the transcript. If evidence supports changing source ownership, rail structure, persistence, or the rejected Console-native source action, amend or supersede ADR-017/TASK-400/TASK-15110 before implementation.

ADR required: no

ADR path: N/A

Reason: Phase 0 records and respects existing decisions; it makes no production behavior or long-lived application-structure change.
