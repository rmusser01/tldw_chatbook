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
