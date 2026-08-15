---
target: TASK-16350 Library modal dismissal design
total_score: 32
max_score: 40
na_heuristics: ""
p0_count: 1
p1_count: 3
timestamp: 2026-08-15T05-13-48Z
slug: 08-14-task-16350-library-modal-dismissal-design-md
---
# TASK-16350 Library modal dismissal critique

## Design Health Score

| # | Heuristic | Score | Key issue |
| --- | --- | ---: | --- |
| 1 | Visibility of System Status | 3 | Mutation progress is visible, but a rejected close gesture has no explicit acknowledgement. |
| 2 | Match System / Real World | 4 | Escape, Cancel, Close, and backdrop behavior match familiar modal conventions. |
| 3 | User Control and Freedom | 3 | Multiple exits converge safely; an active mutation intentionally blocks exit. |
| 4 | Consistency and Standards | 2 | Shared grammar is strong, but base-picker adoption conflicts with the already-safe enhanced picker inheritance. |
| 5 | Error Prevention | 4 | Unknown clicks fail closed and generic gestures cannot confirm high-risk actions. |
| 6 | Recognition Rather Than Recall | 3 | Visible controls help, but overlapping picker-transient precedence is not defined. |
| 7 | Flexibility and Efficiency | 4 | Keyboard, visible-control, and pointer paths are supported. |
| 8 | Aesthetic and Minimalist Design | 3 | The shared primitive is small; the proposed exact launch inventory could become excessive if generalized. |
| 9 | Error Recovery | 3 | Stable-ID recovery is valuable, but the exact-object focus branch lacks eligibility checks. |
| 10 | Help and Documentation | 3 | The spec and ADR are detailed; mutation-time dismissal feedback remains underspecified. |
| **Total** |  | **32/40** | **Good foundation; correction required before implementation.** |

## Design Specificity Verdict

The design is strongly authored for Chatbook rather than category-interchangeable. It models Library-specific deletion fingerprints, skill trust, model installation consent, Git trust and authorization, Prompt collection mutations, File Notes nesting, typed negative results, and Textual overlay/MRO behavior.

The deterministic detector returned no findings. That result is not evidence that the design is safe: the detector does not parse Python, Textual MRO, launch ownership, worker races, typed results, or focus restoration. Source inspection found the issues below.

Browser overlays were not used. The target is a local Markdown design governing a Python Textual TUI, with no reliable browser-rendered product surface; source inspection, runtime MRO inspection, and existing mounted-test contracts were the appropriate fallback.

## Overall Impression

The cancellation grammar, typed result ownership, destructive-action safety, and mutation-race analysis are strong. The largest opportunity is to make shared-picker adoption explicitly compatible with the existing enhanced picker before implementation, then tighten the contract so every concrete modal and combined transient state is actually exercised.

## What's Working

1. Exact negative values remain modal-owned: `False`, `None`, and `PromptDeleteDecision(False, fingerprint)` are not flattened into a generic cancellation result.
2. The event boundary stays small: the mixin owns classification, one-shot/top-screen safety, and focus recovery without learning how to delete, install, trust, authorize, or mutate.
3. The Prompt collection analysis identifies a real Textual message-pump race and proposes the narrow worker-based correction needed to keep close rejection responsive.

## Priority Issues

### [P0] Shared picker adoption creates an inconsistent enhanced-picker MRO

Why it matters: the spec places `SafeModalDismissMixin` in `FileSystemPickerScreen`, while `EnhancedFileDialog` already inherits `SafeModalDismissMixin, BaseFileDialog`. Once the base contains the mixin, that declaration cannot form a consistent MRO and enhanced picker imports can fail before the app mounts.

Fix: explicitly change `EnhancedFileDialog` to inherit only `BaseFileDialog` after base adoption. Preserve its `action_smart_dismiss`, `_SUPPRESSED_BASE_HANDLERS`, persistence behavior, content selector, and exact results. Add `EnhancedFileOpen`, `EnhancedFileSave`, and an import smoke test to the compatibility matrix.

Suggested command: `$impeccable harden`.

### [P1] Result-family tests are weaker than the all-modal acceptance criteria

Why it matters: mounting every concrete class but testing gestures only “for each result family” could let one `None` modal stand in for a sibling with a stale binding, override, or visible Cancel handler.

Fix: define one contract row per concrete reachable modal with factory, selector, exact negative predicate, visible control, guard, and success-result type. Exercise visible Cancel, terminal Escape, and backdrop for every row.

Suggested command: `$impeccable audit`.

### [P1] Plain picker Escape precedence is undefined for overlapping transient state

Why it matters: path editing, search, and recent locations can be active independently. Single-state tests can pass while repeated Escape behaves unpredictably when states overlap.

Fix: state a deterministic order and pin at least one combined-state real-key test. Reuse the enhanced picker order—path, search, recent—where those states exist.

Suggested command: `$impeccable clarify`.

### [P1] Focus eligibility must be identical for exact-object and stable-ID recovery

Why it matters: the same opener object can remain mounted while becoming hidden, disabled, or unfocusable. Restoring it merely because it is attached can strand keyboard focus on an unusable control.

Fix: apply one eligibility predicate to both candidates: attached/mounted, displayed/visible, enabled, and focusable. Cover an ineligible still-mounted original as well as replacement, missing, duplicate, and ineligible-ID cases.

Suggested command: `$impeccable harden`.

### [P2] The fixed-point inventory needs a narrow, auditable edge oracle

Why it matters: constructor scanning can count objects never presented and miss factory-produced runtime types. A generalized Python call-graph analyzer would add complexity without proving runtime reachability.

Fix: enumerate production-default owner files/functions and declared injected presenter seams, inspect presenter arguments/local assignments where practical, and join them with direct route tests for controllers/factories. Do not build a general call-graph engine.

Suggested command: `$impeccable distill`.

## Persona Red Flags

**Alex, power user:** literal implementation would make `EnhancedFileOpen` and `EnhancedFileSave` unimportable. Overlapping picker states also have no declared repeated-Escape order.

**Sam, keyboard/accessibility-dependent:** focus recovery can target a mounted but hidden or disabled original control, leaving the post-dismiss state technically focused but unusable.

**Riley, stress tester:** result-family representatives can miss one broken concrete modal, and constructor-only inventory can look exact while missing injected factories. Combined transient states and repeated input expose both gaps.

## Minor Observations

- Existing File Notes callbacks may duplicate mixin focus restoration; implementation should remove the redundant owner or prove the two paths are idempotent.
- Add an inventory column for the allowed non-dismissible-gate exception even if every current Library row is dismissible.
- Record the exact mutations used during RED/GREEN verification in Implementation Notes.
- Rejected Escape/backdrop input during Prompt mutation may feel inert. A single truthful status acknowledgement could help, but a toast or new state machine is unnecessary unless user testing shows confusion.

## Questions to Consider

- Should shared-base adoption treat `EnhancedFileOpen` and `EnhancedFileSave` as mandatory compatibility clients rather than incidental non-Library regressions?
- Does “exact fixed point” mean statically constructed types or modals demonstrably presented by supported production routes?
- Should the existing progress line remain the only feedback when a close request is rejected during mutation, or should the modal update it once with “Finish the current collection change before closing”?
