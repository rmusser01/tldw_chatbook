# Dynamic and Static character expression playback

Status: Implemented on codex/buddy-import-design; integration pending
Date: 2026-09-07
Creator: tldw-project
Programme: [Buddy imports and characters](2026-09-07-buddy-character-programme-design.md)

## User experience

Add a two-value selector, Character expressions, to F9 Settings → Appearance:

| Choice | Help text |
| --- | --- |
| Dynamic | Animate expressions when available; use a still pose otherwise. |
| Static | Change expressions without playing animations. |

Persist `appearance.character_expression_mode` as `dynamic` or `static`, default
`dynamic`. Use the existing staged Settings save/cancel mechanism. Saving applies
to visible characters immediately, without restarting or reimporting content.
Cancelling does not persist or apply an unsaved change. Invalid stored values
resolve to the documented Dynamic default through the normal config validator.

Effective playback requires Dynamic, `appearance.animations_enabled`, and no
`appearance.reduce_motion`. When global preferences suppress motion, show the
reason beside the selector without rewriting the saved Dynamic preference.
`console.react_character_expressions` remains a separate automatic-selection
control: when off and no manual reaction is selected, render the neutral portrait
without animation. Preserve the existing explicit manual-reaction override, which
may still animate under Dynamic. Hiding character avatars stops playback altogether.
This setting does not enable the floating Buddy.

## Rendering design

Keep expression choice in `UI/Console_Modules/character.py` and the existing
Visual Identity resolver. It chooses operational or manually selected reaction
using the current authority, actor, session and pack-version rules. Animation is
presentation of that resolved asset, not another sentiment or emote resolver.

Prepare a bounded sequence off the UI thread from validated asset bytes. Decode
composited RGBA frames, respecting image disposal, transparency and encoded frame
duration/loop metadata. Support the animated formats the native validator accepts;
animated WebP is the preferred new-conversion output. Existing static PNG/WebP
and legacy expression sets continue to work.

Reuse the Buddy frame-preparation/playback logic where its behavior is suitable;
extract a small shared helper rather than hosting the floating-window widget in
the character rail. No generic animation framework or new graphical dependency.

One visible character owns one timer. Prepared frames are keyed by immutable
asset identity, renderer mode and target geometry. Do not run database queries,
decode images or reload whole rail/transcript widgets on each frame. Repeated
resolution of the same identity must not restart playback. A new expression
starts its own sequence. Finite loops stop on the final composited frame; infinite
loops continue only while visible and motion is enabled.

Invalidate old work on character, authority, conversation, pack version, effective
motion preference or render generation change. Include renderer mode and geometry
in the generation; a Settings save must repaint even when the selected asset is
unchanged. Recheck the generation after worker completion and before paint.
Pause on hidden screens/rail; resume from the current frame with a fresh clock,
without catching up hidden time. Stop and release frames on unmount/actor removal.
Resizing re-prepares at the current geometry without changing expression selection.

## Static pose and fallback

Use the first fully composited frame of the encoded expression. Display that exact
frame in the conversion's Static preview. The independently selectable character
portrait can use another source pose. A separate per-expression poster picker is
out of scope: it would require a new portable playback-metadata contract simply to
preserve a nonzero frame choice. Static mode must not erase stored animation data.

Fallback order is selected expression's first composited frame, neutral portrait, existing
text/initials fallback. A single-frame asset is a normal case, not an error.
Unsupported animation, invalid timing or playback-preparation failure may show a
valid bounded static preview; malformed or over-budget images are rejected without
decoding them through a less strict path. Show a concise capability/fallback reason
in the character preview, not a recurring toast on every frame.

Use the existing image and decoded-pixel limits. Independently preflight animation
preparation before allocating all frames. Retain at most 64 MiB of accounted RGBA
buffers for the active character across active and in-flight preparation, including
both old and new geometry during a resize. Prepare only one job at a time; stale
jobs discard results and release buffers. Decode composited frames sequentially,
downscale to the bounded display size immediately, and do not retain full-size
copies. If even one required decode canvas exceeds the preparation budget, use an
already valid static fallback or neutral portrait without starting that decode.
Do not store every frame in the transcript image cache or as a Rich renderable;
retain at most the current and next rendered frame. The buffer budget is not a
claim about total process RSS; native codec allocations need a measured regression
check. Exceeding a limit selects static rendering for that asset.
Use a playback ceiling of 30 paints/second; coalesce overdue paints using elapsed
time rather than extending the encoded animation duration. A loop's individual
frame delays must be positive. Do not invent successful motion for corrupt timing.

## Storage and compatibility

Use existing immutable Visual Identity assets with animation metadata; no database
migration is planned. Playback preference is local appearance config, never part of
the character's personality or imported visual asset. Dynamic/Static survives restart.
Setting Static must leave re-exported visual assets byte-identical.

## Acceptance and evidence

- A real two-frame expression paints visibly distinct frames in a mounted Console
  rail; switching to Static freezes a representative pose while an operational
  change still selects another expression.
- Save/cancel, restart, missing/invalid config, global animation off, Reduce motion,
  reaction off and avatar hidden each produce the specified behavior.
- Finite/infinite loops, unequal delays and disposal/transparency have independent
  fixtures and decoded-pixel assertions. Static preview matches encoded frame zero.
- Saving Dynamic/Static or Reduce motion while the same asset remains selected
  immediately changes the rendered result; late preparation cannot restore motion.
- Manual reaction precedence remains unchanged when automatic reactions are disabled.
- Fake-clock tests prove pause/resume, timer disposal and that unchanged refreshes
  do not restart the sequence; measured callback/query counts catch redundant work.
- Slow decoding followed by a session/account/character/version switch cannot paint
  stale frames. Resize and unmount during decoding have the same protection.
- Static assets, unsupported animation, empty/corrupt data and memory-limit fallback
  are tested without weakening image validation.
- Import/export preserves animation bytes regardless of display mode. Capture a
  rendered frame or short recording for the visible product path before completion.

ADR required: yes.
ADR path: [ADR-144](../../../backlog/decisions/144-character-expression-playback.md).
Reason: motion preference and rendering lifecycle; existing ADR-067 and ADR-074
remain the expression and Buddy ownership boundaries.
