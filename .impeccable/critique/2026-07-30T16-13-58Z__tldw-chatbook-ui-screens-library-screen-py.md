---
target: File Notes full-app UAT
total_score: 23
max_score: 40
na_heuristics:
p0_count: 2
p1_count: 0
timestamp: 2026-07-30T16-13-58Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: /root/uat_design_review_a · B: /root/uat_detector_review_b)

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 2 | Rich state exists, but the Files source, compact status, and primary actions disappear. |
| 2 | Match System / Real World | 3 | Database, Files, staging, review, and commit fit local-first Git users. |
| 3 | User Control and Freedom | 2 | Back, unstage, and cancel exist, but compact users cannot advance. |
| 4 | Consistency and Standards | 2 | The source strip is incomplete and Tab does not produce a usable compact action. |
| 5 | Error Prevention | 4 | Exact staged-state matching and unrelated-change protection are excellent. |
| 6 | Recognition Rather Than Recall | 2 | Hidden source and actions require guessing rather than recognition. |
| 7 | Flexibility and Efficiency | 2 | Pane switching and bulk controls are sound, but the compact keyboard path fails. |
| 8 | Aesthetic and Minimalist Design | 2 | A diagnostic absolute path displaces task-critical state and actions. |
| 9 | Error Recovery | 2 | The staged-state mismatch is precise but does not name the safe recovery step. |
| 10 | Help and Documentation | 2 | Inline keyboard guidance exists, but its compact-mode promise is not fulfilled. |
| **Total** |  | **23/40** | **Acceptable foundation with blocking reachability failures.** |

## Design Specificity Verdict

**LLM assessment:** The intended information architecture is strongly authored
for Chatbook: Database/Files authority switching, a disk-backed notes
workspace, session-scoped staging, and an exact reviewed commit are coherent
with its local-first control model. The count-and-promise language is unusually
specific and effective. The rendered layout does not yet uphold that model at
supported terminal sizes.

**Deterministic scan:** The required detector returned exit code 0 with `[]`
(zero findings, no rules or locations) for
`tldw_chatbook/UI/Screens/library_screen.py`. This is a detector coverage limit,
not evidence against the PTY failures: runtime Textual allocation and compact
terminal geometry are outside its checks. Source inspection independently
located both layout causes.

**Visual evidence:** Browser overlays do not apply to a native Textual TUI, so
no web server or DOM injection was attempted. The fallback evidence is a real
Chatbook process in a PTY at `160x45`, `120x40`, and `40x20`, preserved under
`Docs/superpowers/qa/file-notes-full-app-uat-2026-07-30/`.

## Overall Impression

The guarded commit interaction inspires confidence once reached: it names
scope, blocks an unsafe complete index, and finishes with a truthful result.
The largest opportunity is simpler than a redesign—make that already-good
workflow visibly enterable and keyboard-reachable at every supported size.

## What's Working

- The review and success states pair an exact note count with “unrelated
  changes untouched,” which gives high-stakes reassurance without vague claims.
- Exact-match validation rejects unrelated staged content without moving HEAD
  or changing the index.
- Navigator/Editor switching is an appropriate compact workspace model, and
  wide editing through commit is coherent.

## Cognitive Load

- `Database |` asks users to infer a missing source instead of recognizing it.
- At `40x20`, the complete linked-root path consumes roughly four rows while
  status and the next action disappear—a priority inversion.
- Compact users must remember session state while searching for controls that
  keyboard traversal does not visibly reveal.
- The safety detail itself is not excessive; layout allocation, not copy
  volume, is the failure.

## Emotional Journey

The wide review and commit create a strong confidence peak. Discovery begins
with a trust-breaking valley when Files is absent, and compact use ends after
the user has already edited a note but cannot prepare it for commit. That
abandonment dominates the otherwise reassuring safety model. The unrelated
staging refusal feels safe, though slightly incomplete without a direct
recovery instruction.

## Priority Issues

### [P0] The File Notes entry disappears at normal widths

**Why it matters:** At both `120x40` and `160x45`, users cannot discover or
enter a core workspace through the visible interface.

**Evidence:** `library_screen.py` constrains the two source buttons to automatic
width, but the intervening `Static("|")` remains flexible and consumes the
remaining horizontal space, pushing Files outside the strip.

**Fix:** Give the separator a stable class/ID with `width: 1; min-width: 1`.
Keep both labeled buttons focusable and visibly indicate the selected source.

**Suggested command:** `$impeccable layout`

### [P0] Prepare session becomes a compact-terminal dead end

**Why it matters:** A user can edit and save at `40x20` but cannot reach Stage
or Commit, so the primary journey fails after work has already been invested.
Keyboard-only users have no compensating route.

**Evidence:** The complete linked-root path is rendered in auto-height rows
without elision. The Git surface is a non-scrollable `Vertical`, with status
and actions after the list, so constrained height clips the actionable tail.

**Fix:** Clamp and middle-elide the persistent root summary to one line while
retaining the full path on demand, and make the Prepare workflow vertically
keyboard-scrollable so focusing an action brings it into view. Root clamping
alone is geometry-dependent and is not sufficient.

**Suggested command:** `$impeccable layout`

### [P2] The safe staged-state refusal omits the recovery step

**Why it matters:** Users know why review was blocked but may not know which
state must change before retrying.

**Fix:** Preserve the exact safety sentence and add concise recovery copy:
“Commit or unstage the unrelated staged changes, Refresh, then review this
session again.”

**Suggested command:** `$impeccable clarify`

## Persona Red Flags

**Alex (power user):** The advertised keyboard path cannot visibly reach Stage
at `40x20`, turning a fast terminal workflow into a dead end.

**Jordan (first-time File Notes user):** `Database |` gives no recognizable
evidence that File Notes exists; the staged-state refusal explains policy but
not the next safe action.

**Sam (keyboard-dependent user):** Focus may move to an off-viewport Files
button, and compact traversal cannot make a staging action usable, so the
workflow is not keyboard-operable despite labeled controls.

## Minor Observations

- A disabled selected-source button can read as unavailable; retain a visible
  selected marker in addition to disabled-state behavior.
- Keep leading success copy definitive—“Committed 2 session notes”—before the
  reassurance.
- The root basename or trailing segments are sufficient for persistent
  orientation; the complete absolute path can remain available on demand.
- Existing two-line fixed-region fitting is not the cause; the unbounded outer
  linked-root status and non-scrollable action surface are.

## Questions to Consider

- What one next action must remain visible or immediately scrollable after
  editing at every supported terminal size?
- Is the complete absolute root important enough to displace current status
  and commit controls?
- Should every safety refusal name both the protected condition and the exact
  safe recovery path?
