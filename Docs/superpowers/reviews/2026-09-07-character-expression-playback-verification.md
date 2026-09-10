# Character expression playback verification

Historical evidence: the results below describe the original source branch, not
the current dev integration. Combined-code results are recorded in
[the integration verification](2026-09-10-buddy-feature-integration-verification.md).

Date: 2026-09-07
Task: TASK-32023
Creator: tldw-project
Decision: [ADR-144](../../../backlog/decisions/144-character-expression-playback.md)
Plan: [Playback implementation](../plans/2026-09-07-character-expression-playback.md)

## Delivered behavior

F9 Appearance exposes Dynamic/Static character expressions through the existing
staged save/revert path and search index. Global animation-off and Reduce motion
suppress playback; explicit manual reactions retain their existing precedence.
Static selects encoded frame zero while continuing to change expressions.

The Console uses existing immutable resolution bytes and a disposable avatar
widget. Off-thread preparation enforces native image limits, serializes decoding,
accounts 64 MiB across active/in-flight RGBA buffers and normalizes GIF/WebP/APNG
loop semantics. Presentation uses elapsed visible time at at most 30 paints per
second, without a database query or avatar remount per frame. Static portrait,
neutral fallback and text fallback preserve source bytes. The graphics widget
updates its image in place with fixed layout; missing graphics capabilities fall
back to the terminal mosaic renderer.

## Executed evidence

- Pure decoder/model tests cover real GIF/WebP/APNG pixels, first-frame Static,
  unequal durations, infinite/finite loops, APNG default-image separation,
  transparency/background disposal, invalid data/timing, and buffer reservations.
- A mounted real Console with a real disposable SQLite database paints two
  different expression frames, retains its widget on unchanged/forced refresh,
  switches the same asset to Static, and produces an identical native Actor Pack
  archive before/after the mode change.
- Mounted avatar tests exercise actual timer updates, overlay/hidden-time pause,
  stale preparation, removal during preparation, graphics image replacement,
  neutral fallback, renderer failure and resource release.
- The focused F9 Appearance run passed **11 tests**, including save, revert,
  loaded defaults and live refresh signaling.
- The broad avatar/controller/geometry/copy-budget run initially passed 141 tests
  but exposed a background mount-failure teardown error. The geometry worker now
  catches that failure. Subsequent testing exposed a child-removal race and an
  unnecessary visibility invalidation; both received focused fixes and regression
  coverage. The final focused avatar/geometry/Console run passed **19 tests**,
  including the graphics and child-removal regressions.
- New modules/tests pass Ruff. Existing changed files have no new Ruff diagnostics
  relative to the branch's pre-implementation HEAD; unrelated baseline lint debt
  was not reformatted or suppressed. Changed ranges were formatted and diff-check
  passed.

Use the repository interpreter, with `PYTHONPATH=packages/tldw_profile_core/src`
when the bundled package has not been installed into that interpreter. Color-pixel
tests require a color-capable harness (`env -u NO_COLOR FORCE_COLOR=1`). Root test
fixtures isolate configuration and profile storage.

## Product capture and memory probe

The Console integration test optionally writes three SVG screenshots when
`TLDW_PLAYBACK_EVIDENCE_DIR` is set. The verified captures contain 72 red avatar
pixel fills and no blue fills for Dynamic's first frame, 72 blue/no red for the
second frame, and 72 red/no blue for Static. The capture enters a real conversation
before rendering; an earlier capture correctly revealed that setup guidance was
covering the rail. No provider request or user profile was used.

A disposable process decoded and released a two-frame 1024×1024 WebP eight times.
Accounted retained buffers returned to zero after every cycle. Measured peak RSS
was 125,960,192 bytes, with 4,767,744 bytes of peak growth after fixture setup.
This is a measured regression fixture, not a guarantee that the whole application
or every native codec consumes at most 64 MiB.

## Baseline limitations and scope

The existing Settings focused-input test expects width greater than ten cells but
gets three cells on this checkout. It fails unchanged on the pre-implementation
runtime and is excluded from the focused green Appearance selection. The existing
color-only geometry assertion also fails under the inherited NO_COLOR environment;
its color-capable run passes. These are distinct from the playback regressions
fixed above. No full-suite pass is claimed.

Terminal graphics image replacement was verified in the mounted test harness;
real Kitty/Sixel terminal-protocol output was not exercised. Buddy-to-character
conversion, Petdex remote import, portable attribution changes, server endpoints
and new avatar art remain separate work.
