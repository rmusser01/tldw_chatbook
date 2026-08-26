---
target: TASK-19639 Console exact-100-column containment design
total_score: 29
max_score: 40
na_heuristics: ""
p0_count: 0
p1_count: 2
timestamp: 2026-08-20T07-38-51Z
slug: 913-console-exact-100-column-containment-design-md
---
Method: dual-agent (A: task_18913_ux_second_pass_a · B: task_18913_ux_second_pass_b)

## Design Health Score

| # | Nielsen heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 3 | The fix restores the work surface, but the unhandled 101→100 path can still make it disappear silently. |
| 2 | Match with the real world | 3 | Established Console language is preserved, though Context/Inspect remains less obvious to non-technical newcomers. |
| 3 | User control and freedom | 3 | Preferences remain reversible; direction-dependent resize behavior would undermine predictability. |
| 4 | Consistency and standards | 4 | Reusing ADR-043’s minimum-width waiver is the right product-standard mechanism. |
| 5 | Error prevention | 2 | The current spec misses the resize deduplication boundary and defines width ambiguously. |
| 6 | Recognition rather than recall | 3 | Both rails remain text-labeled and discoverable. |
| 7 | Flexibility and efficiency | 3 | Keyboard and stored-state behavior are protected, but reading-state continuity is not yet proven. |
| 8 | Aesthetic and minimalist design | 3 | The three-region composition is purposeful, but the readable transcript is denser than “55 columns” suggests. |
| 9 | Error recovery | 2 | A stale live layout has no diagnosis; users must guess that resizing or collapsing a rail will recover it. |
| 10 | Help and documentation | 3 | Existing help remains; the broadened override semantics and stale width comments need correction. |
| **Total** |  | **29/40** | **Good foundation; two P1 corrections required before implementation.** |

## Design-specificity verdict

The design is strongly specific to Chatbook’s terminal-native Console and its actual Context/Inspector priority, persistence, keyboard, and Textual geometry contracts. The central decision remains correct: keep Context open at 100 and reuse the established transcript-minimum waiver.

The deterministic detector returned `[]` with exit code 0: zero rules, findings, locations, or false positives. No browser overlay was attempted because this is a native Textual TUI without a DOM. Fallback evidence used production CSS, the real Console hierarchy, Textual compositor geometry, live terminal resizing, focus probes, and the four stored-preference states.

## Overall impression

The proposed UX outcome is right, but the written design is not implementation-ready yet. A one-line state change would fix cold start while leaving a reproducible live-resize failure, and the current width criterion can certify the outer pane while overstating what users can actually read.

## What’s working

- The design preserves user authority: rail visibility policy, Inspector priority, stored preferences, labels, and manual controls remain unchanged.
- The 100-column diagnosis is exact: the grid has 96 content columns, while 30 Context + 56 main minimum + 11 Inspector handle requires 97.
- Runtime simulation proves the chosen mechanism works: waiving the main minimum yields contained widths of 30 + 55 + 11.

## Priority issues

- **[P1] Exact-100 live resize bypasses the proposed state change.** Widths 100 and 101 share `compact-before-auto-open`, so the resize handler returns early on 101→100. Give exact 100 its own semantic width band—or include waiver eligibility in the deduplication key—and test 101→100, 100→101, 99→100, and 100→99 explicitly. Suggested command: `$impeccable adapt`.
- **[P1] “Transcript ≥55 columns” measures the wrong thing unless selectors are named.** After the waiver, `#console-main-column.region.width` is 55, but the transcript region’s content is 51 and the native transcript content is 49. Require main allocation >=55 for the default and native transcript content >=40 in every state. Suggested command: `$impeccable layout`.
- **[P2] Resize verification does not yet prove experience continuity.** Assert focus per transition, selected-message and scroll-anchor continuity, zero preference-save calls, and unobscured paint evidence. Suggested command: `$impeccable harden`.
- **[P3] Comments and flag semantics will mislead maintainers.** Describe compact override as layout-minimum-waiver authority and correct stale 3-cell-handle and 24-column-rail comments. Suggested command: `$impeccable clarify`.

## Persona red flags

- **First-time technical:** cold start could look fixed while 101→100 still fails; “55 columns” overstates the 49-column readable viewport.
- **First-time non-technical:** overflow resembles lost data or a crash, with no diagnosis or recovery; dense wrapping makes guidance harder to scan.
- **Regular technical power user:** direction-dependent geometry violates cold/live convergence; selection and anchor changes could interrupt inspection.
- **Regular non-technical power user:** an unreliable learned layout encourages rail toggling as recovery, unintentionally changing preferences.

## Minor observations

- The four-state matrix is valuable; only Context-open/Inspector-closed fails at 100 in current production geometry.
- Manual-collapse focus loss is a known, separately protected contract and should remain outside this geometry task.
- Rail-section overflow guidance remains correctly separated into TASK-19428.
- No new ADR is required if this remains within ADR-043.

## Questions to consider

- Is the promise 55 columns allocated to the main pane, or 55 readable transcript columns?
- Should resize deduplication be keyed to semantic layout state rather than nominal bands?
- What test proves the user stayed on the same message, not merely that a widget remained visible?
