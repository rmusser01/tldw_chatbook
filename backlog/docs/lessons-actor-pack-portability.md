# Lessons: Actor Pack portability

## Verify imported characters through their next edit

TASK-32024's real archive → activation → reopen → edit → export test failed even
though initial import succeeded: the imported manual visual pack lacked the private
`profile_pack_id` required by the editor. Activation now assigns that identifier
without exporting it. Round-trip evidence should include an ordinary edit, not only
an immediate re-export of untouched imported data.

## Public notices and private authority digests need different string limits

The same test imported an 8,800-character notice, then failed on re-reviewing the
same UUID. The authority digest serialized the entire stored graph through the
portable actor payload validator, whose 4096-character string limit rejected the
notice JSON. Hashing exact stored JSON fields before that canonical graph hash
retains stale-review detection without applying a display/payload limit to notices.
A regression changes only the stored notice and verifies that review becomes stale.

## Verify visible animation pixels, and wait for hydrated character handoff

**TASK-32025, 2026-09-07.** The first lossless-WebP timeline check downgraded four
real Buddy packs to static because the encoder rewrote RGB values under alpha zero.
All visible pixels and alpha values were identical. A failing transparent-pixel
regression and the seven-pack probe established the cause; checking exact alpha and
RGB only where coverage is nonzero preserved every supported animation. The test
still rejects a one-unit alpha change.

The mounted conversion-to-Console flow also showed that awaiting `_select_character`
did not mean its card was loaded: the handler schedules `ccp-load-character`.
Waiting for the existing worker, then checking destination and selected identity,
made the established handoff consume the newly created character. Verify the consumer's
hydrated state, not merely the completion of its scheduling wrapper.
