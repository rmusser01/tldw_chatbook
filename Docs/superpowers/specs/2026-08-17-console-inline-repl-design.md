# Console inline REPL prompt — Phase C design spec (task-17655)

Status: DRAFT — awaiting owner approval. No implementation tasks may be
filed until this spec is approved (task-17655 AC#2).

## Context and the honest question

The bottom-stack de-clutter programme (tasks 17650-17659) reached its
target silhouette: the composer is a one-row dense-form bar growing to
four rows with the draft, floating one blank row clear of the status row
above and the footer below; the workbench frame closes at the grid; the
transcript gained seven content rows at 150×44. The reference
architecture (batrachianai/toad: MainScreen = Conversation + one Footer,
prompt inline in the conversation flow) motivated the programme.

Phase C asks whether to go the rest of the way: move the prompt INTO the
transcript flow, REPL-style — the conversation scrolls, and the place you
type is the tail of the history rather than a fixed bar.

**The honest framing this spec owes the owner:** the original motivation
for Phase C was chrome reduction, and that is already delivered — the
fixed bar now costs 3 rows total (1 content + 2 deliberate air). What an
inline prompt still buys is a *feel*, not rows:

- The prompt reads as the next message — typing continues the transcript
  instead of happening in a separate control.
- Empty conversations start at the top, REPL-style, instead of the input
  sitting at the bottom of an empty pane.

What it costs:

- A fixed, muscle-memory location for the input is lost; the prompt's
  screen position varies with history length.
- Every composer affordance needs a rehomed answer (table below), and the
  owner's standing ruling — keep ALL send affordances (banner, Send,
  Dictate) — makes the toad-pure form (no buttons, bell on not-ready)
  unavailable. The inline row would carry the same controls the bar does.
- The transcript is a windowed, virtualized scroller (TASK-15455/15777):
  an always-mounted interactive tail widget inside it intersects the
  windowing, follow-mode, selection, and approval-card machinery — the
  riskiest seams in the app (see the programme memory's trap list).

## Recommendation

**Defer.** Ship nothing for Phase C now. The programme captured the
measurable benefit; the remaining benefit is aesthetic preference with
structural risk attached. Revisit if, after living with the current
layout, the owner still wants the conversational feel — this spec then
serves as the starting design. The alternative (approve now) is written
out in full below so the decision is informed, not deferred by vagueness.

## Design (if approved): the tail-mounted composer

### Architecture

- `ConsoleComposerBar` stops being a sibling of the transcript region and
  mounts as the transcript's permanent TAIL ROW — the last child of the
  scrollable message flow, after the newest message, before nothing.
- It keeps its id, children, dense-form styling, and 1-4 row growth; all
  existing send affordances stay per the owner ruling. The left edge
  remains the focus carrier.
- The status row, staged-evidence strip, prompt-queue shelf, and footer
  stay where they are (outside the scroller): only the composer moves.
  Staged evidence and the queue annotate the NEXT send, so they remain
  glued to wherever the prompt is — they move into the tail cluster with
  it, directly above the composer row.

### Interaction model

- **Follow mode:** when the transcript is tail-anchored (the default),
  the prompt is always visible at the bottom of the viewport — identical
  to today. Scrolling up scrolls the prompt off-screen with the history.
- **Typing recall:** any printable key (the existing screen-level
  redirect) snaps the viewport back to the tail and focuses the prompt —
  typing is never lost into a scrolled-away input.
- **During a run:** the collapsed run-status variant (status + Stop +
  Expand) renders in the same tail slot; streaming output appears ABOVE
  it, so the Stop control rides the tail of the stream.
- **Empty conversation:** the prompt sits at the TOP of the empty
  transcript under the get-started/ready guidance — the REPL feel that
  motivates this design.
- **Approval cards** (which mount inside the transcript) appear above the
  tail cluster, never below the prompt.

### Affordance rehoming table

| Affordance | Home in the inline design |
|---|---|
| Draft area, ghost text, paste-collapse, history recall | Unchanged, inside the tail row |
| Send / Dictate / Stop / disabled-reason banner | Unchanged, on the tail row (owner ruling) |
| Composer ▾ collapse | Unchanged: same-height content swap in the tail slot |
| Menu (Save as Chatbook, overflow) | Unchanged, on the tail row |
| Attachments indicator / recovery / voice status | Unchanged, tail-row statics |
| Staged-evidence strip, prompt-queue shelf | Move with the composer into the tail cluster |
| Status row, footer | Unchanged (outside the scroller) |
| F6 pane cycling | The transcript and composer merge into ONE pane stop; F6 order shrinks by one |

### Migration and testing strategy

1. Spike first (throwaway): mount the bar as the transcript tail behind a
   dev-only flag; live-probe the three risk seams (windowing prune/
   hydrate around an interactive tail, follow-mode after send, selection
   walk skipping the tail row). Abandon cheaply if any seam fights.
2. If the spike holds: one implementation task for the move + follow
   mechanics, one for the staged/queue cluster relocation, one for the
   contract-test migration (the bottom-stack geometry pins largely
   invert: composer INSIDE `#console-native-transcript`, shelf pins move
   with it; the F6/tab-region tables lose a pair).
3. Bundle-loading painted probes at every step — this programme's lessons
   (outline activation, style-vs-paint) all recurred at exactly this kind
   of seam.
4. A `[console] inline_prompt` kill switch for at least one release,
   defaulting per the owner's call at approval time.

## Open decisions for the owner

1. Approve the deferral (recommended), approve the spike, or approve the
   full design?
2. If approved: does the empty-conversation top-anchored prompt match
   your intent, or should an empty session keep the prompt at the bottom?
3. If approved: default `inline_prompt` on or off at first ship?
