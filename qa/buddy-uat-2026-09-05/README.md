# Migu Buddy UAT fixes on current dev

Latest review rebase: `dev` at `d9e2e3d507`.
Earlier follow-up repairs were rebased onto `b52080fee0`.
Original PR verification base: `68f9d865fad623db6ec02e19632090c1140b3c89`.
Work branch: `codex/migu-buddy-uat-fixes`. Task: TASK-31585.

## Scope

Current dev already contains the Buddy implementation and realtime/fleet
extractions (PRs #1927, #1939, #1970 and #2094). This patch carries only the
remaining UAT defects forward; it preserves the newer Buddy widget, controller
boundaries, durable turn admission and diagnostic inventory format.

- Apply final native mouse-release coordinates for move and resize. Existing
  coordinate-based hit testing already handles XTerm events with no widget.
- Permit profile-owned Persona Visual imports, drafts and active-version edits.
  Pin directory device/inode while retaining file metadata, digest, descriptor,
  containment and symlink checks. Publication may change ancestor metadata.
- Route readback through trusted Manual Speak snapshots and playback ownership,
  so terminal playback clears the speaking presentation. Actual playback also
  owns an exact Buddy voice lease; stale terminal callbacks release only their
  own lease and preserve concurrent voice owners.
- Terminalize cancelled/unavailable project-instruction recovery. Refused
  transient echoes are excluded from provider history; already durably accepted
  turns retain recovery ownership under current dev's existing contract.
- Restore refused drafts after the acceptance callback consumed them, only when
  the composer still owns the same visible session. Preserve newer edits.
- Prune nested Python environments from diagnostic scanning, preserving
  serialized POSIX ordering and application modules merely named `venv`.
- Remove exception capture from lazy LLM mount and Library lifecycle failures.
  Reviewed inventory changes are exactly two diagnostic digests: 14 LLM-window
  calls and 104 Library-screen calls remain. No owners or sinks were added.

ADR required: no new ADR. Existing ADR-074 (Persona Visual/Buddy), ADR-037
(trusted speech), ADR-069 (project instructions), and ADR-029 (private diagnostic
boundary) govern these repairs. No new schema, dependency or provider transport.

- Preserve the stable Send width while bounding the busy transcription chip,
  keeping the cancel-capable microphone reachable at 80 columns. Narrow copy
  reads “Local transcription busy — queued.” or “Queued”; the full explanation
  stays in its tooltip. Ordinary listening layout is unchanged.

## Verification

Tests import the actual PR tree using `PYTHONPATH` and `python -m pytest`.
The existing development Python environment supplies dependencies only.
No full repository suite was run.

- Initial rebased targeted run: **414 passed, 3 failed**. All three remaining
  voice failures were investigated and repaired; see the follow-up below.
- Rebased diagnostic suite: **69 passed, 1 skipped**. The skip requires unavailable
  historical Git objects; current-source and metadata assertions remain enabled.
- Speech ownership, autoplay and Buddy adapter gate: **102 passed**.
- Final complete voice-chip and mounted dictation selection: **103 passed**.
- Focused narrow-chip/mounted-microphone regression selection: **13 passed**.
- Independent final chip review: **15 passed**; no actionable review findings.
- Diagnostic inventory rebuilt without drift: 574 owners, 1,334 TASK-492 calls,
  7,599 TASK-494 calls and 10 sink files. The changed counts come from rebased dev.
- All six derived-artifact preflight checks passed: CSS, path inventory,
  diagnostic inventory, task IDs, schema allowlist and index pins.
- Changed Python files pass fatal-rule checks; the missing Library `Iterable`
  annotation import is repaired. New readback tests pass Ruff, changed ranges
  were formatted, and `git diff --check` passes. No full repository sweep.

### Verification repairs

The original eight failures reproduced after the rebase. Four composer fixtures
needed the supported capture-off setting because their in-memory harness has no
durable trace repository. The token-omission fixture now permits the base request
to fit while the optional instruction row exceeds its budget. Setup retry covers
both transient refusal and durable acceptance: a durable pending response blocks
new sends until explicit discard, after which a fresh send succeeds.

The metadata registry now follows two extracted Library diagnostics and removes
one retired by `5dd1077df6`. The exception-type forwarding guard admits only the
existing safe conditional type-name/literal fallback. No privacy check was disabled.

The broader voice gate exposed a real 80-column clipping regression after Send
width stabilization, plus stale fixed-width/reason-visibility expectations. Bare
composer harnesses also omitted the split Console stylesheet, causing false
visual failures. They now load the production Console sheet; production CSS did
not need changing.

### Rebased live evidence

`rebased-live-evidence.json` records actual runs from this PR worktree:

- **Kokoro:** 128,000 PCM bytes drained; Migu idle → speaking → idle; speech
  presentation cleared, no app exception, normal configuration unchanged.
- **DeepSeek:** expected synthetic reply received; run completed; Migu
  idle → thinking → speaking → idle; normal configuration unchanged.
- The first rebased Kokoro replay exposed the missing Buddy playback lease;
  regression tests failed before the repair and the real replay passed afterward.
- Physical macOS dragging is still awaiting a foreground Terminal window. The
  background gesture did not change geometry, so this replay is not a pass.
- OpenAI realtime is blocked because no credential is configured.

## Earlier live evidence and limits

`prior-live-evidence.json` contains minimized synthetic-fixture results from the
older working checkout. It is historical live evidence, not a claim that those
hardware runs were repeated on this PR tree. Real Terminal move/resize, a real
DeepSeek reply, a five-second microphone/local-transcription check, and local
Kokoro playback were exercised there. Kokoro drained 128,000 PCM bytes with
Migu idle → speaking → idle; microphone capture stopped after 5.02 seconds,
recognized the known phrase, released the device, and returned to idle.
Normal configuration remained unchanged. Microphone content was neither saved
nor sent to a provider. Temporary speech dependencies/model were not added to
the project or enabled in the user's normal profile.

Before final sign-off: finish physical dragging on the final branch, review
updated CI, then run live OpenAI realtime UAT with a credential configured in
the application's settings. That provider was not configured during this UAT. Local speech success
does not validate the OpenAI realtime transport.


## Qodo review follow-up

All three initial Qodo findings were addressed: refusal restores captured
undo/redo only when the same visible draft has not been edited; successful
sends remain history barriers. Publication and cleanup now consume the shared
`validate_canonical_directory` result, preserving the existing strict spelling,
symlink, descriptor, identity and containment policy. Lazy mount errors retain
an allowlisted view identifier (unknown otherwise) without exception capture.

The latest-dev rebase passed 74 Console regressions and 150 Buddy/publication/
setup regressions. Review fixes passed 174 undo/draft/LLM/publication/path tests
and two diagnostic privacy guards. The diagnostic statement change is exactly
one safe-view argument on the existing 14-call LLM owner; no sink changed.
Live provider/playback records above predate these review-only follow-ups and
retain their original source digests. TASK-31585 remains In Progress for the
explicit native-dragging and OpenAI realtime UAT gaps; the user authorized
review and merge of the verified fixes with those follow-ups recorded.
