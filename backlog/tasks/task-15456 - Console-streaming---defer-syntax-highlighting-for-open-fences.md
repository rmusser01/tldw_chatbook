---
id: TASK-15456
title: Console streaming: defer syntax highlighting for open fences
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: Textual's `Markdown.append` re-parses only from the last completed top-level block, so a reply that is one long code fence re-parses the whole fence and re-runs Pygments over the entire fence-so-far on every 0.2 s tick, synchronously on the event loop (`textual/widgets/_markdown.py:1445-1509`, `MarkdownFence` highlight at `:895-901`); a growing paragraph is a remove+remount per tick. Multi-block prose is genuinely O(delta) — the worst case is exactly the long-code-block replies this audience produces.

Fix direction: throttle fence-interior appends (e.g. plain-text tail while the fence is open, highlight at fence close, or a slower cadence for fence-interior deltas). Keep final rendered output byte-identical at stream end. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Streaming a long single code fence no longer re-highlights the full fence every tick (evidence)
- [x] #2 Final rendered message identical to today's output at stream end (test)
- [x] #3 No behavior change for multi-block prose streaming
<!-- AC:END -->

## Implementation Plan

1. Locate the append seam: `ConsoleMarkdownMessage.sync_message` in
   `Widgets/Console/console_transcript.py`, which prefix-diffs the assistant
   body and calls `Markdown.append(delta)` on pure growth.
2. Add a conservative, cheap fence-parity detector
   (`_console_markdown_body_ends_in_open_fence`): scans lines for CommonMark
   fence delimiters (<=3 leading spaces, backtick/tilde run >=3, matching
   char + run length to close), returning True only when confident the body
   ends inside an open fence. False negatives/positives both degrade safely
   (never drop or reorder content -- only affect timing of a flush).
3. In `sync_message`'s growth branch, route the delta through a new
   `_append_or_defer_body_delta` helper: while `message.status == "streaming"`
   AND the body currently ends in an open fence AND a wall-clock deadline
   (`_FENCE_APPEND_DEFER_SECONDS`, monotonic-based) hasn't elapsed, buffer the
   delta instead of calling `markdown.append()`. Otherwise flush the full
   buffered+new delta in one `markdown.append()` call and reset the deadline.
   Non-append edits (variant switch/`update()`) and any non-streaming status
   always flush immediately, guaranteeing correct final content.
4. Tests (new file, existing suites untouched):
   - Unit tests for the fence detector against CommonMark edge cases
     (open/closed, tilde vs backtick, mismatched run length, indented,
     inline code, ambiguous info-string-with-backtick).
   - Evidence test: simulate a long streamed fence with a fake monotonic
     clock: assert `MarkdownFence` reconstructions are bounded, not 1:1 with
     ticks (contrasted against the disabled-throttle shape, which the probe
     shows is 1:1).
   - Byte-identical final render test: streamed-with-throttle vs direct
     unthrottled render of the same final content produce the same
     `Markdown.source` and the same `MarkdownFence` highlighted output.
   - Multi-block prose regression test: no fence anywhere -> still exactly
     one `Markdown.append()` per delta (AC#3, unthrottled).
5. Run the new tests plus the existing console_transcript/streaming/
   tail-follow suites unmodified; measure per-tick append cost before/after
   with an isolated probe; update the task file and commit.

## Implementation Notes

**Approach.** The append seam is `ConsoleMarkdownMessage.sync_message`
(`Widgets/Console/console_transcript.py`), which prefix-diffs the assistant
body against `self._body_text` and calls `Markdown.append(delta)` on pure
growth. Added a conservative fence-parity scanner,
`_console_markdown_body_ends_in_open_fence`, that tracks CommonMark
fence-delimiter parity line-by-line (<=3-space indent, backtick/tilde run
>=3, closing run length >= opening, no info string on the closing line;
backtick info strings containing a backtick are treated as ambiguous and
never open a fence). It intentionally does not track blockquote/list
indentation contexts -- misjudging those only costs either a missed
optimization or a delayed flush, never dropped/reordered content, because
every code path that consumes it still ends with a full, correct
`markdown.append()`/`markdown.update()` of the true accumulated text.

New helper `_append_or_defer_body_delta` on `ConsoleMarkdownMessage`: while
`message.status == "streaming"` AND the body ends inside an open fence AND a
monotonic deadline (`_FENCE_APPEND_DEFER_SECONDS = 1.0`) hasn't elapsed, the
delta is buffered (`self._pending_fence_delta`) instead of calling
`markdown.append()`. The deadline is wall-clock (not a tick counter) so a
slow/stalled model doesn't stretch staleness indefinitely just because fewer
sync ticks landed while it paused. Any of: fence closing, `status` leaving
`"streaming"`, or the deadline elapsing forces an immediate flush of the
*complete* buffered text in one `markdown.append()` call -- append order and
final content are unaffected, only the number of
`Markdown.append`/`MarkdownFence`-reconstruction (Pygments) passes changes.
A non-append edit (variant switch, retry, DB-resume rebind) discards any
pending buffer and falls back to the existing full `markdown.update()`.

**Evidence (AC#1).** `MarkdownFence.__init__` runs `highlight()` (Pygments)
over the whole fence-so-far every time it's reconstructed, which happens once
per relevant `Markdown.append()`/`.update()` call -- this is the exact
mechanism the audit names. Counting `MarkdownFence.__init__` invocations
(not `.highlight()` directly -- see the note in
`test_long_open_fence_stream_defers_highlight_work`, where an earlier version
counting `.highlight()` was contaminated by Textual's unrelated
`Stylesheet.apply` -> `notify_style_update()` path, which also calls
`highlight()` on every node in a CSS-reapplied subtree) over a simulated
30-tick, 0.2s-cadence open-fence stream: **5 reconstructions with the
throttle active vs 30 with it disabled** (verified directly, not just via the
test's looser bound). A test asserting the bound would fail against the
disabled-throttle shape (30 > 15), satisfying "born red."

**Final-render evidence (AC#2).** A new test streams a fence-containing
message in in small chunks through the throttled path (with several
deadline-forced mid-fence flushes) and compares the final mounted
`MarkdownFence` against one built by a single unthrottled render of the
identical final text: same `.source`, same `.code`/`.lexer`, and
identical `._highlighted_code` (plain text and spans) -- i.e. Pygments ran
over the exact same final content either way.

**AC#3 (untouched prose).** The throttle only ever activates when the
detector confirms an open fence; a `no-fence` regression test asserts
exactly one `Markdown.append()` per delta across 10 streamed word-by-word
prose growths (matching the pre-existing, unmodified
`test_streaming_appends_without_reparse`/`test_streaming_append_activates_flavor_when_marker_closes`
in `Tests/UI/test_console_transcript_markdown_widget.py`, both still green).

**Performance probe (isolated, not committed).** A standalone script
(`t15456_probe.py`, scratchpad) streamed a 120-tick / ~360-line synthetic
Python fence at simulated 0.2s cadence with real wall-clock timing:
- Isolated `MarkdownFence.__init__` (highlight) cost: BEFORE 121 calls /
  5057.9ms total vs AFTER 21 calls / 998.1ms total -- **5.07x speedup**,
  call count 121 -> 21 (matches the ~5x predicted by the O(ticks x size)
  vs O(ticks/N x size) math for a 1.0s/0.2s = 5-tick defer window).
- Full `refresh_messages()+pilot.pause()` tick cost only improved **1.23x**
  (273.9ms -> 223.1ms mean/tick) -- reported honestly alongside the 5.07x
  number because it's dominated by per-frame Textual costs (DOM
  reconciliation, compositing an already-large mounted widget) this task
  does not touch, not by the highlighting work itself.

**Files modified:**
- `tldw_chatbook/Widgets/Console/console_transcript.py` -- fence detector,
  throttle constant, `ConsoleMarkdownMessage` state fields, and
  `_append_or_defer_body_delta`.
- `Tests/UI/test_console_transcript_fence_throttle.py` (new) -- detector unit
  tests, bounded-highlight evidence test, byte-identical final-render test,
  no-fence regression test.

**Backlog CLI note.** `backlog task edit 15456 ...` hit the documented
5-digit-id bug (`lessons-backlog-hygiene.md`): it printed `Updated task
TASK-` and silently wrote a stray, empty `backlog/tasks/task-task- - .md`
instead of touching this file. Deleted the stray file; this file was hand-edited
directly per that lesson's guidance.
