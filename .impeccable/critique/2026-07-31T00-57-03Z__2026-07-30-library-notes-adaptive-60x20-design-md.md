---
target: Library Notes Adaptive 60x20 design pre-implementation review
total_score: 26
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 3
timestamp: 2026-07-31T00-57-03Z
slug: 2026-07-30-library-notes-adaptive-60x20-design-md
---
Method: dual-agent (A: notes_ux_assessment_a · B: notes_evidence_assessment_b)

# Pre-implementation Critique — Library Notes Adaptive 60×20 Design

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3/4 | Persistent draft/save/conflict text is strong; import/export progress and retry guidance remain underspecified. |
| 2 | Match System / Real World | 3/4 | Navigator, Editor, and Preview are plain; Context, versions, Overwrite/Reload, and cycling controls remain system-centric. |
| 3 | User Control and Freedom | 3/4 | Back hierarchy, conditional Reload, and Cancel-first deletion are strong; there is no local Esc contract, undo, or history. |
| 4 | Consistency and Standards | 2/4 | The wide workflow loses direct access to incumbent actions despite the compatibility promise. |
| 5 | Error Prevention | 3/4 | Revisioned saves and conflict gating are excellent; the supported Textual runtime is not yet guaranteed. |
| 6 | Recognition Rather Than Recall | 3/4 | Core actions remain visible, but Context and cyclic choices require users to remember hidden content. |
| 7 | Flexibility and Efficiency | 2/4 | Keyboard traversal survives, but Adapt adds no quick switcher, note accelerators, or pane workflow. |
| 8 | Aesthetic and Minimalist Design | 3/4 | Single-stage tasks reduce noise; compact Navigator still presents six equal actions. |
| 9 | Error Recovery | 3/4 | Errors retain drafts and conflicts are actionable; there is no diff, merge, save-both, or crash recovery. |
| 10 | Help and Documentation | 1/4 | Labels carry the experience; contextual help, shortcut guidance, and recovery instructions are absent. |
| **Total** |  | **26/40** | **Acceptable — revise before implementation** |

## Design Specificity Verdict

**Authored safety model, category-generic knowledge model.**

The canonical draft, serialized revision queue, conflict tokens, Database Notes
boundary, and Console handoff are recognizably Chatbook. The
Navigator → Editor → Context drill-in is a credible terminal adaptation but
could belong to almost any notes application. That is acceptable for Adapt
only if it remains a migration-friendly foundation rather than hardening the
flat record model into the permanent Notes architecture.

The deterministic scan returned zero generic findings, but that is low-signal:
the target is a Markdown design contract, not implemented markup. Source
inspection found several concrete contract gaps that the detector cannot see.

No visual overlay is available. The proposal is an unimplemented Textual TUI
with no browser DOM; browser injection would not produce reliable evidence.

## Overall Impression

The design is unusually strong where note applications most often fail:
preserving user work through saves, conflicts, navigation, and resize. Its
weakest point is not the state model but the transition from specification to a
real 60×20 screen. The document currently promises more vertical space,
runtime behavior, and wide-layout compatibility than the repository guarantees.

## What’s Working

1. **Loss prevention is designed as a system, not a status label.** Monotonic
   revisions, a serialized save driver, guarded Reload, duplicate conflict
   gating, and navigation vetoes form a credible no-silent-loss contract.

2. **Responsive behavior preserves task state.** The Back hierarchy, canonical
   draft, stable `note_id` identity, focus restoration, and one scroll owner per
   region treat resize as interaction design rather than CSS shrinkage.

3. **The scope stays honest.** Adapt does not fabricate backlinks or declare
   Obsidian parity before the knowledge-relationship model exists.

## Cognitive Load

Four of eight checks fail, which is high cognitive load:

- **Chunking fails:** Navigator exposes six actions together.
- **Visual hierarchy fails:** the design does not identify primary versus
  secondary Navigator actions.
- **Minimal choices fails:** Navigator, Sync, Context, and the extensible
  template chooser each exceed four available decisions.
- **Working memory fails:** cyclic Sort/Direction/Conflict controls hide their
  alternatives, and wide users must remember which utilities moved to Context.

Single focus, grouping, one-thing-at-a-time sequencing, and progressive
disclosure pass.

## Emotional Journey

- **Entry:** relief when rail/canvas clipping disappears and `‹ Library`
  establishes location.
- **Open/edit:** confidence from stable row identity and immediate
  `Unsaved changes`.
- **Resize:** potential peak; preserving draft, caret, selection, scroll, and
  Preview across 170 → 60 → 170 would be a memorable local-first proof.
- **Context:** friction on wide screens because existing utilities become a
  mandatory extra hop.
- **Conflict:** an appropriately serious valley handled with good loss
  prevention, though still without diff or save-both.
- **Save failure:** reassurance can become entrapment unless the vetoed Back
  state says exactly how to retry or exit safely.

## Priority Issues

### [P0] The required Textual 8 runtime is not yet a supported installation contract

**Why it matters:** The design relies on Textual 8 focus and widget behavior,
but `pyproject.toml` still permits `textual>=3.3.0`, `requirements.txt` is
unpinned, ADR-022’s implementation task remains in progress, and TASK-1333 has
no dependency on it. The design can pass locally on Textual 8.2.7 while failing
for a supported installation.

**Fix:** Make TASK-1333 depend on TASK-400, or explicitly absorb ADR-022’s
metadata and CI work. Dependency is the cleaner choice because TASK-400 already
owns the runtime-floor change.

**Suggested command:** `$impeccable harden`

### [P1] Adapt regresses the incumbent wide workflow

**Why it matters:** The spec promises wide-layout compatibility but moves
keywords, Console handoff, Copy, exports, and Delete behind Context at every
width. Wide users gain an unnecessary navigation step and lose simultaneous
access to controls they have today.

**Fix:** Preserve the incumbent wide editor fields and utility access at
`>=120`. Keep Context as the compact drill-in, or make it an optional wide
view without removing direct wide actions.

**Suggested command:** `$impeccable adapt`

### [P1] The 60×20 guarantee has no complete row budget

**Why it matters:** Global navigation consumes three rows, the footer one,
Library adds a header and bordered layers, and Editor still needs a header,
three-row title, status, actions, and body. Conflict content tightens this
further. “Useful,” “readable,” and “fully usable” cannot be verified without
content-box minima. Long titles can also wrap and consume the budget.

The spec additionally promises Space activation for Buttons, while the pinned
Textual 8.2.7 Button binding exposes Enter only.

**Fix:** Add explicit normal/conflict row budgets including all shell chrome,
define minimum body/Preview content-box heights, make compact titles one-line
and markup-disabled with cell-aware ellipsis, and either guarantee Enter only
or implement/test a Notes-specific Space binding.

**Suggested command:** `$impeccable adapt`

### [P1] Active drafts do not reliably win compact-stage and recompose races

**Why it matters:** On a breakpoint crossing, rail focus currently outranks an
active dirty/error/conflict workflow. A user can shrink into a rail-only stage
while a draft remains active, and Browse Notes re-entry is not explicitly
defined as resume. Separately, `LibraryScreen` has many whole-screen recompose
origins; a promise to capture before “an unrelated recompose” is not a central
enforcement mechanism.

**Fix:** Make dirty/error/conflict sessions outrank rail focus, define Browse
Notes as resuming the active region until explicit Back completes it, and
intercept every `recompose=True` through one capture/rehydration seam rather
than auditing individual call sites.

**Suggested command:** `$impeccable harden`

### [P2] The migration seam still leaves permanent orchestration in a 14k-line screen

**Why it matters:** `LibraryScreen` remains responsible for responsive state,
drafts, serialized saves, conflicts, focus, services, and navigation. Pure
snapshots help testing but do not make the orchestration portable to the later
dedicated workbench.

**Fix:** Define a cohesive Database Note session owner before implementation,
or explicitly record this concentration as deferred migration debt. If a new
cross-module coordinator is introduced, update the ADR check because it becomes
a long-lived interface decision.

**Suggested command:** `$impeccable shape`

## Persona Red Flags

### Alex — keyboard-first power user

- Wide workflows become slower when utilities move behind Context.
- Six Navigator actions still require long traversal.
- Cyclic Sort/Direction/Conflict controls hide direct choices.
- No note accelerators, quick switcher, or pane workflow arrive in Adapt.

### Jordan — first-time user

- “Context” does not predict keywords, metadata, Console, export, copy, and
  deletion.
- The known no-notes versus no-filter-match copy defect remains.
- `Save failed — edits kept` does not state the next safe action.
- Immediate creation still permits accidental Untitled records.

### Sam — keyboard/accessibility-dependent user

- Field labels and status announcement behavior are not specified.
- No local Esc/Back contract can force long reverse traversal.
- Hidden mounted stages must be removed from focus order, not merely hidden
  visually.
- Space activation is promised but unsupported by the current Button runtime.

### Rina — Chatbook research user

- Draft safety and Console handoff are distinctive strengths.
- Flat rows still lack provenance, linked media, sources, and relationships.
- Concentrating more behavior in `LibraryScreen` makes the later knowledge
  workbench and Database/File Notes boundary harder to extract.

## Minor Observations

- The spec says conflict policy is unchanged, but guarded Reload is a deliberate
  editor conflict-interaction change. Narrow the non-goal to storage/sync
  conflict policy and retain the current no-new-ADR conclusion explicitly.
- Normalize shortened file references such as
  `css/components/_agentic_terminal.tcss` to their real
  `tldw_chatbook/...` paths.
- TASK-1333 terms such as “every capability,” “focus intent,” and “relevant
  widget identity” should link to explicit transition/control lists.
- Add ADR-011 heartbeat, worker-backlog, timer-registry, route-switch, and soak
  evidence or explain why this is not a major screen replacement.
- Give title, body, and keywords persistent labels; test deterministic focus
  order after conflict.
- Use a more actionable recovery string, such as
  `Draft changed — Reload not applied. Choose again.`
- Consider `Saved 14:32` rather than bare `Saved`.
- Stable mounted surfaces must be excluded from layout, focus, and action
  queries when hidden.

## Questions to Consider

- Is wide compatibility a real constraint, or is Adapt authorized to add a
  Context hop for architectural uniformity?
- When a dirty draft exists, should anything except explicit Back ever reveal
  the Library rail as the active compact stage?
- Is a small session coordinator now cheaper than extracting save/conflict
  orchestration from `LibraryScreen` during Shape?
- Which exact rows disappear or scroll when conflict, validation, or a long
  title competes with the 60×20 editor body?
