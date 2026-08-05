# Speech destination — Console-grammar redesign

**Status:** draft for review
**Date:** 2026-07-27
**Supersedes:** nothing. Follows the Lab frame programme (#940, #966, #991, #998, #1010, #1023).

## Purpose

The owner's words, and the thing every decision below answers to:

> This screen's goal is to allow a user to test/configure the various STT/TTS
> options available, and identify which would work best for them.

Speech is an **evaluation tool**, not a synthesizer with settings attached.
The core loop is: pick a provider and voice → synthesize a known piece of
text → listen → change one variable → synthesize again → decide. Everything
that serves that loop earns its place on screen; everything else earns a
collapsed group.

This reframes the redesign. It is not "make the form shorter" — it is "make
the comparison possible".

## Why now

Measured on the shipped screen at 120×40 (`Tests`-free probes against the
running app):

- The Playground form is **93 rows in a 34-row viewport**.
- `🔊 Generate Speech` renders at **y=60** — 21 rows below the fold. So do
  Play, Pause, Stop and Export (y=65). Reachable only by scrolling ~2.5
  screens.
- Every label/input pair costs **4 rows**; two explanatory notices cost **5**
  each.
- Speech Recognition's privacy panel is **23 cells wide**: its title
  truncates to "Privacy Settin", and a 45-character privacy notice is laid
  out in a **14-cell** box, rendering as a green block with the text clipped.

The rows are not gratuitous. The Playground genuinely has 57 controls. The
defect is that all 57 are treated as equally important, all the time.

## Scope: six views, not four

All six rail entries become body views. Today only four are:

| Rail entry | Today | After |
| --- | --- | --- |
| TTS Playground | view | view |
| TTS Settings | view | view |
| AudioBook/Podcast | view | view |
| Voice Cloning | **pushes a separate screen** | **view** |
| Speech Recognition | view | view |
| Audio Effects | **disabled, no implementation** | **placeholder view** |

**Voice Cloning** currently calls `push_screen(VoiceCloningWindow())`, which
leaves the Lab frame entirely — the rail, mode strip and inspector all
disappear, then come back. Cloning a voice and then *trying* it is one loop;
breaking the frame in the middle of it is the wrong seam. It becomes the
fifth view.

**Audio Effects** stays as an explicit placeholder. Per the owner: it will
become real sound-effect generation for a future "studio" view. A
placeholder that says so is honest; a permanently disabled row that says
nothing is not.

## Control inventory (the evidence)

Attributed to the owning widget class, not guessed from id prefixes:

| View | Controls | Composition |
| --- | --- | --- |
| TTS Playground | **57** | `tts-` 34, audio player 6, higgs 5, generation 3, kokoro 2, reference 2, chatterbox 1, clear 1 |
| TTS Settings | **79** | chatterbox 17, higgs 16, audio.cpp 15, kokoro 8, elevenlabs 7, defaults 5, alltalk 4, openai 3 |
| Voice Cloning | **29** | backend select/status, profile CRUD, export, confirm dialog |
| Speech Recognition | **15** | provider, language, punctuation, commands, transcript, history, 4 exports |
| AudioBook/Podcast | **15** | import, preview, chapter editor, voice assignment, generation settings, log |
| Audio Effects | 0 | unbuilt |

`STTS_Window.py` holds **151 distinct ids** (Settings 79 + Playground 57 +
AudioBook 15). With Dictation's 15 and Voice Cloning's 29 the destination
totals **195**. Any plan that treats this as one screen rebuild is wrong
about the size.

Note where the provider parameters actually live, because it is not where
the prefixes suggest: **Chatterbox has 17 controls in Settings and 1 in the
Playground; Higgs has 16 in Settings and 5 in the Playground.** The
Playground's own provider knobs are the `tts-`-prefixed ones
(`tts-exaggeration-input`, `tts-stability-input`, `tts-higgs-top-p-input`
and so on). Settings is the larger surface, not the Playground.

## The design grammar

Read off the running Console screen, not invented:

1. **One row per thing.** Never a box per control.
2. **Visible commands as a text action strip** (`CommandStrip` +
   `WorkbenchAction`), packed left — not chunky buttons stacked down the page.
   Textual's `Button` carries a default `min-width`; the strip must set
   `min-width: 0` or six actions spread across the pane and the last two fall
   off the edge.
3. **State as `Label: value` chips** on one line, the way Console states
   provider/model/character/RAG above its composer.
4. **A single bordered input**, like Console's composer, with a placeholder.
5. **Recovery copy as one line**, not a five-row block.
6. **Reverse-video section heads** (`Text`, `Result`), matching Console's
   inspector bars rather than inventing a third heading style.

Icons and emoji are retained — they are the rail's only per-item visual
anchor, and dropping them was one of the things that made an earlier attempt
unreadable.

## The central rule: comparison axes vs tuning knobs

Every Playground control is one of two kinds, and the purpose statement
decides which:

**Comparison axes — always visible.** The variables you change to compare:
`provider`, `model`, `voice`, `language`, `format`, `speed`. Six controls,
one chip row.

**Tuning knobs — collapsed, provider-scoped.** Set once per provider, rarely
touched. In the Playground these are the `tts-`-prefixed provider
parameters: stability/similarity/style/speaker-boost,
exaggeration/cfg-weight/candidates/seed/temperature/validate-whisper,
higgs temperature/top-p/repetition-penalty/delimiter/multi-speaker/
voice-cloning, kokoro use-ONNX, and audio post-processing (normalize,
target dB, preprocess text). That is roughly 24 of the Playground's 57.

Only the selected provider's group renders. ElevenLabs' parameters never
appear while Chatterbox is selected. That is the difference between 57
always-on controls and ~7 visible with the rest one keystroke away.

## Result history, not a single result

To "identify which would work best" you must compare takes. A Result pane
showing only the latest generation asks the user to remember what the
previous one sounded like.

The Result region is a short **history list**: newest first, one row per
generation carrying voice, format, duration and timestamp, with Play and
Export per row. It fills with real content as the user works, which is also
what stops the pane reading as empty.

Retention is session-scoped. Persisting takes across restarts is out of
scope here.

## Responsive contract

- **≥ 64 cells of pane width:** Text and Result side by side.
- **< 64 cells:** the split stacks, **and every one-row row stacks with it** —
  the action strip becomes one action per row, the chip row one chip per row.
  Nothing is dropped and nothing is truncated.
- The pane scrolls when stacked. `1fr` children must become `auto`/fixed when
  stacking, or they compress to fit and the overflow is clipped rather than
  scrollable (measured: `virtual == container` while children needed 17 rows
  in a 3-row box).
- Verified by measurement, not inspection: zero controls cut off at 80×24.

## Per-view requirements

**TTS Playground.** Core chip row; text input; action strip (Generate,
Play, Pause, Stop, Export, Clear, Random text, Refresh catalog); result
history; provider status and any provider restrictions as one line each;
collapsed provider-scoped tuning group.

**TTS Settings.** Owns **persisted defaults** per provider — the values the
rest of the app uses, including Console dictation and AudioBook generation.
One collapsible block per provider, only the configured ones expanded.

**Speech Recognition.** Live transcript as the primary region; provider and
language as chips; punctuation and command switches; the four exports
(copy, text, markdown, timestamps) as an action strip; history list. The
privacy notice becomes one line with detail in the inspector — never a
14-cell green block.

**Voice Cloning.** Profile list, backend select and status, profile CRUD
and export, all inside the frame.

**AudioBook/Podcast.** Keep the collapsible structure — it is already the
closest view to this grammar. Import, content preview, chapter editor, voice
assignment, generation settings, generation log.

**Audio Effects.** One line stating what it will be and that it is not built,
naming the future studio view.

## Resolved: Settings vs Playground ownership

**Settings owns persisted defaults. The Playground owns session-scoped
overrides that do not write back.**

Settings edits what the app uses everywhere -- including Console's dictation
and AudioBook generation. The Playground edits the current experiment: you
change a voice or a temperature to hear the difference, and leaving the
screen does not silently rewrite your configuration.

Consequences that phases must honour:

- A Playground override is visibly an override. The chip shows the effective
  value; the fact that it differs from the saved default is stated, not
  implied.
- The Playground offers an explicit "save as default" path, because the
  purpose is to *identify what works best* and then keep it. Comparison
  without a way to commit the winner is half a tool.
- Settings never reads Playground state. One direction only.

## Acceptance criteria

Per phase, and measured rather than eyeballed:

- Every control the view had before is reachable after. Enumerated by id
  against the pre-change widget, not judged by looking.
- The view's primary action is above the fold at 120×40 **and** 80×24.
- Zero controls truncated or clipped at either size, proven with
  `widget.render_line(0).text`, not `content_region.width` -- the latter
  reported 16 for a 15-character label that did not render.
- No region overflows a non-scrolling container: `virtual_size.height` equals
  `container_size.height`, or an ancestor genuinely scrolls.
- The CSS bundle reproduces from its source modules.
- Full-suite comparison against a worktree pinned to the branch's parent, so
  pre-existing failures are classified rather than inherited or blamed.

## Verification approach

Unit tests do not catch what this redesign is about. The Lab frame programme
shipped a blank body past 78 green tests, and a dead collapse handle past
tests that called the method instead of pressing the button. So:

- Drive the real app under `App.run_test()`, navigate to the view, and
  measure widget geometry.
- Assert rendered text, not intent.
- Mutation-check every new guard: break the thing it guards and confirm it
  fails, or it is decoration.
- Capture screenshots at 120×40 and 80×24 and look at them. Three separate
  defects in this workstream were invisible to a green suite and obvious in
  a render.

## Retirement of the legacy window

`STTS_Window.py` is 5,900 lines holding 151 of the 195 controls. It is not
deleted in one step and not left forever.

Each phase moves one view's controls into a new pane and removes that view's
branch from `watch_current_view`. The window shrinks phase by phase. After
phase 5 it holds only what AudioBook still needs; after phase 6 it should be
deletable, and the final phase's definition of done includes deleting it or
recording exactly what still depends on it.

The dead `.stts-sidebar` rule retired in #1023 is the pattern: when the last
consumer goes, the code goes with it in the same change.

## Risks

- **Console depends on STT settings.** Console has a dictation entry point;
  changing which layer owns transcription defaults can affect it. Any change
  to Settings ownership must be checked against Console, not just Speech.
- **Provider parameters are validated somewhere.** Moving a control must not
  drop its validation. Each phase checks where the old widget read and
  validated the value before re-siting it.
- **`lab_speech_status.py` already owns the capability line.** The rail
  summary and the inspector's recovery detail exist and are tested; the
  redesign consumes that seam rather than re-deriving dependency state.
- **The prototype is a shell.** `SpeechPlaygroundPane` covers 4 of 57
  controls with hardcoded values and no wiring. It demonstrates grammar and
  is not a head start on functionality.

## Phasing

Each phase ships independently and is reviewable on its own.

1. **Playground** — the grammar, the axis/knob split, result history,
   override-vs-default handling. It is the view the purpose statement is
   about, and it establishes the pattern the other five follow.
2. **TTS Settings** — the largest surface at 79 controls and where most
   provider parameters actually live. Moved ahead of the others: "configure"
   is half the brief, the axis/knob split buys the most here, and phase 1's
   override model is meaningless until the defaults it overrides have a
   proper home.
3. **Speech Recognition** — worst remaining visual defects (23-cell panel,
   bottom pinning, clipped privacy copy).
4. **Voice Cloning** — promote to a view inside the frame.
5. **AudioBook/Podcast** — least broken; mostly grammar alignment.
6. **Audio Effects** — placeholder view, and the point at which
   `STTS_Window.py` should be deletable.

## Non-goals

- Changing the **engine**: provider APIs, synthesis/transcription pipelines,
  audio processing. Interaction may change -- the owner's brief is "recreate
  all functionality, not necessarily behavior", so a control may be reached
  differently than before as long as the capability survives. What must not
  change is what the engine does with it.
- Persisting generation history across restarts.
- Building sound-effect generation (the future studio view).
