# Dynamic and Static character expression playback

Status: Design review
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
`console.react_character_expressions` remains a separate expression-selection
control: when off, render the neutral portrait without animation. Hiding character
avatars stops playback altogether. This setting does not enable the floating Buddy.

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

Invalidate old work on character, authority, conversation, pack version or render
generation change. Recheck the generation after worker completion and before paint.
Pause on hidden screens/rail; resume from the current frame with a fresh clock,
without catching up hidden time. Stop and release frames on unmount/actor removal.
Resizing re-prepares at the current geometry without changing expression selection.

## Static pose and fallback

For converted packs, read the validated conversion metadata's preview frame index
for the selected expression. For other assets, use the first fully composited frame.
The portrait is also stored independently on the character. Static mode must not
erase frames or convert stored data to PNG.

Fallback order is selected expression's preview frame, neutral portrait, existing
text/initials fallback. A single-frame asset is a normal case, not an error.
Unsupported animation, invalid timing or playback-preparation failure may show a
valid bounded static preview; malformed or over-budget images are rejected without
decoding them through a less strict path. Show a concise capability/fallback reason
in the character preview, not a recurring toast on every frame.

Use the existing image and decoded-pixel limits. Cap retained prepared data to
64 MiB for the active character; exceeding it selects static mode for that asset.
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
- Finite/infinite loops, unequal delays, disposal/transparency and a nonzero static
  preview index have independent fixtures and decoded-pixel assertions.
- Fake-clock tests prove pause/resume, timer disposal and that unchanged refreshes
  do not restart the sequence; measured callback/query counts catch redundant work.
- Slow decoding followed by a session/account/character/version switch cannot paint
  stale frames. Resize and unmount during decoding have the same protection.
- Static assets, unsupported animation, empty/corrupt data and memory-limit fallback
  are tested without weakening image validation.
- Import/export preserves animation bytes regardless of display mode. Capture a
  rendered frame or short recording for the visible product path before completion.

ADR required: yes, as part of the programme's motion/lifecycle decision; existing
ADR-067 and ADR-074 remain the expression and Buddy ownership boundaries.
