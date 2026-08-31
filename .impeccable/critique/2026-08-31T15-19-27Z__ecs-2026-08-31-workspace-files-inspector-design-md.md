---
target: Workspace Files Inspector written design
total_score: 28
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 4
timestamp: 2026-08-31T15-19-27Z
slug: ecs-2026-08-31-workspace-files-inspector-design-md
---
Method: dual-agent (A: /root/ux_assessment_a · B: /root/evidence_assessment_b)

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3 | Identity and terminal save outcomes are explicit, but loading, filter progress, pagination, and stable status placement are not. |
| 2 | Match System / Real World | 2 | Recovery copy still exposes terms such as binding, root, publication, and durability at stressful moments. |
| 3 | User Control and Freedom | 3 | Undo, Revert, dirty guards, Back, Close, and Escape are strong; ending a clean edit lease and cancelling pre-publication Save are undefined. |
| 4 | Consistency and Standards | 3 | Safe dismissal matches the product, but entry placement and Back/Close variants are not pinned to incumbent Console grammar. |
| 5 | Error Prevention | 4 | Exact baselines, fail-closed editability, per-operation revalidation, and no automatic retry are excellent. |
| 6 | Recognition Rather Than Recall | 2 | No exact tree-mark legend, row anatomy, action grouping, or disabled-reason carrier is specified. |
| 7 | Flexibility and Efficiency | 3 | Filtering and keyboard intent are present, but no executable focus/key matrix or direct accelerator is defined. |
| 8 | Aesthetic and Minimalist Design | 2 | The generic two-pane file-manager composition and five peer editor actions lack a Chatbook-specific hierarchy. |
| 9 | Error Recovery | 4 | Draft preservation, exact comparison, stale-scope refusal, and uncertain-publication recovery are unusually complete. |
| 10 | Help and Documentation | 2 | Reasons exist, but technical outcomes and compact decorations lack contextual plain-language help. |
| **Total** |  | **28/40** | **Good foundation; interaction specification needs hardening before planning.** |

## Design Specificity Verdict

**Behaviorally bespoke; visually under-specified.** The non-activating inspector, direct-user authority, canonical-root edit lease, Agent Change Review separation, and honest publication outcomes are unmistakably Chatbook. The proposed visual composition—binding selector, filter, file tree, and editor—could belong to any IDE or file manager. The spec needs a concrete Neon Workbench grammar for authority, lease ownership, save state, action priority, truncation, tree marks, and narrow transitions.

The deterministic scan returned exit code 0 with `[]`: zero findings, rules, severities, locations, or false positives. This is not a clean bill of health for the future interface; the target is Markdown and the six named implementation files do not exist, so there was no component markup or TCSS to inspect.

No reliable browser overlay exists. The target is an unimplemented Textual design rather than a viewable route, so mutable injection and a live server were correctly skipped. The two supplied screenshots and incumbent Console source served as fallback evidence.

## Overall Impression

The hard part—the trust model—is good. The biggest opportunity is to turn that rigor into an equally precise interaction contract. Without that pass, implementation will be forced to invent entry placement, lease release, responsive geometry, focus transitions, and status hierarchy in code, exactly where this Console has already accumulated clipping and modal-state regressions.

## What’s Working

1. **Authority is visible product behavior.** The pinned inspected-versus-active identity directly supports local-first control and prevents accidental activation.
2. **Recovery preserves user work.** Dirty guards, exact Base/Draft/Disk identities, no silent retargeting, Copy draft, and no automatic retry form a strong safety net.
3. **Scope is disciplined.** View-before-edit, one buffer, bounded traversal, no autosave, and three independent delivery slices keep v1 reviewable.

## Priority Issues

### [P1] The two entry points are not placed precisely enough

**Why it matters:** “All-workspaces list” is ambiguous between the grouped conversation browser and the separate workspace switcher. The active rail already overflowed when three compact actions shared a 24–30-column row, and the grouped workspace header currently has only a flexible label plus a three-cell collapse toggle. A hover-only action copied from the screenshot would also fail keyboard discoverability.

**Fix:** Name the exact surfaces. Put active-context **Show Files** on its own action row beside or below **RAG Scope**. Put a permanently focusable, text-labeled compact **Files** action in each grouped-browser workspace header. Define its cell budget, focus order, truncation priority, unavailable-reason behavior, and keep the separate workspace switcher unchanged.

**Suggested command:** `$impeccable shape`

### [P1] A clean editor can hold a root-wide lease with no explicit release path

**Why it matters:** After Save, an `EditingClean` modal may still block a new overlapping agent run. Nothing tells the user that the root remains reserved or how to release it without closing the inspector.

**Fix:** Add a pinned contract row such as `Editing <folder> · overlapping agent writes paused`, plus **Done editing** to return to Viewing and release the lease. Define release after Save, Revert, same-root navigation, cross-root navigation, binding change, conflict, failed publication, uncertain publication, and dismissal. An agent admission failure must name Workspace Files as the owner and point back to the inspector.

**Suggested command:** `$impeccable harden`

### [P1] Responsive and keyboard behavior is conceptual rather than executable

**Why it matters:** No width/height thresholds, pane minimums, action overflow, low-height behavior, initial focus, tree key model, or conflict layout are defined. Existing Console history shows compact controls can become clipped yet remain clickable. Two implementations could satisfy the prose while behaving materially differently.

**Fix:** Add a measured responsive/input table for at least 80×24, 100×30, 120×40, and 160×50. Define wide/narrow thresholds, optional full-screen takeover at the smallest size, pane and scroll ownership, pinned regions, header compression, fold indicators, path ellipsis, initial/remapped focus, tree Arrow/Enter behavior, focus after every state transition, and a narrow Base/Draft/Disk selector with one comparison viewport.

**Suggested command:** `$impeccable adapt`

### [P1] Dirty-guard and Save cancellation do not yet form an operable flow

**Why it matters:** The spec says Save is cancellable before publication but defines no cancel action. It also does not define dirty-guard focus, Escape/backdrop behavior, or whether the guard is inline or nested. Five peer editor actions make disk-changing, draft-only, and lease-ending actions visually equivalent.

**Fix:** Use an inline guard that keeps the modal owner mounted. Focus **Keep editing** by default; Escape/backdrop also means Keep editing; show the pending destination in the prompt. Provide visible **Cancel save** only during `SavingPrePublication`, replacing it with non-interactive `Finishing save…` after linearization. Use one primary action per state: `[Save] [Undo] [Redo] [More…]`, with Revert and Copy draft under More; recovery states replace the ordinary bar.

**Suggested command:** `$impeccable harden`

### [P2] The left pane lacks a precise multi-binding and status grammar

**Why it matters:** Filter bounds currently span the workspace while paths are root-relative, so identical paths in different bindings are ambiguous. A row can simultaneously be selected, Git-conflicted, Unsaved, and Edited this visit; without precedence and a cell budget, narrow rows become symbol soup. Search progress, partial results, zero results, and truncation are also invisible in the contract.

**Fix:** For v1, scope filtering to the selected binding and make changing bindings a guarded typed transition. Define first-open selection, stable sorting, duplicate-path handling, unavailable roots, and filter states (`idle`, `searching`, `partial`, `complete`, `truncated`, `cancelled`, `failed`). Add a row anatomy and precedence contract: indentation/caret, type glyph, elided name, one primary text state, optional secondary mark, with Conflict > Unsaved > Git > Edited-this-visit and a visible legend/help path.

**Suggested command:** `$impeccable distill`

## Persona Red Flags

**Alex — power user:** There is no specified accelerator for Show Files, focus-filter, or return-to-tree. Entry behavior is not stable enough for muscle memory. Five peer edit actions slow scanning, and a clean editor may unexpectedly block an agent run.

**Sam — accessibility-dependent user:** Tree expansion, loading, paging, and filter truncation announcements are absent. Disabled reasons risk becoming tooltip-only. “Text/icon semantics” does not define accessible glyph-plus-label behavior or narrow conflict reading order.

**Riley — stress tester:** Initial loading, unreadable directories, zero/only-excluded results, file disappearance, directory mutation between pages, and stale Git arrival have service defenses but no complete UI states. Pre-publication cancellation is promised but not operable.

**Morgan — solo builder/operator:** Non-active inspection perfectly fits the workflow, but the modal must continuously distinguish viewing from root reservation. The uncertain-publication state must lead with whether the draft is safe and one recommended next action, not filesystem terminology.

## Minor Observations

- Replace ambiguous `Back / Close inspector` wording with exact per-layout labels.
- Use **Filter paths…**, not the screenshot’s **Search files…**, because matching is path/name based and bounded.
- Specify directory-first versus lexical sorting, case sensitivity, long-name ellipsis, and full-path reveal.
- Allow true full-screen presentation at minimum width if the backdrop costs necessary editor cells.
- Put the read-only reason adjacent to disabled Edit and expose it without hover.
- Keep Saved visible long enough to perceive and announce it without moving focus.
- Define whether Undo history preservation across resize is mandatory; remove “where the editor permits” if it is.
- Define root-level Git-unavailable separately from file-level Git state.

## Questions to Consider

- What prevents this from feeling like a miniature IDE rather than a Chatbook authority inspector?
- Should a clean editor reserve an entire root, or should the lease end immediately after Save/Revert unless the user keeps editing?
- Can one compact pinned contract row communicate inspected workspace, active Console workspace, access mode, draft/save state, and lease ownership without becoming telemetry soup?
- If users must understand filesystem durability to choose recovery, has the interface translated the system state far enough?
