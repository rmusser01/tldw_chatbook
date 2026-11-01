# Console radio portraits: implementation-plan review

Date: 2026-10-23
Scope: [Implementation plan](../plans/2026-10-23-console-radio-portraits-implementation.md), [design](../specs/2026-10-23-console-radio-portraits-design.md), [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md) and nine To Do implementation tasks. Review assumes the existing local Console's one-audible-owner model.

The user requested corrections to all three implementation-plan findings and a
second review before implementation. Only planning documents and task criteria
are changed. No product implementation or task completion is claimed.

## Corrections

1. **P1: software-queue exhaustion is not audible completion.**
   `Audio/streaming_sink.py::_callback` currently marks drain when queued bytes
   have been submitted; `_notify_loop` invokes `_teardown_stream`, which aborts.
   The plan now keeps natural drain nonterminal until native output-finished
   acknowledgement, retaining the stream, registry and speech owner. The existing
   sounddevice `CallbackStop`/`finished_callback` boundary preserves pending audio;
   interruption still uses immediate abort. Delayed final-block tests cover both
   observed and unobserved callers, invalid DAC timing, cancellation, missing or
   duplicate acknowledgements and bounded failure. TASK-36544 owns the sink change;
   TASK-36545 prevents early hands-free completion/listening and stale-owner cleanup;
   TASK-36550 qualifies the final audible syllable and sentence handoff.

2. **P1: helper tests did not prove continuous publication.**
   The existing notification thread blocks on its work queue; ordinary audio
   blocks do not each emit a public event. The plan now defines a timed wait capped
   at 20 ms on that existing thread, bounded/coalesced callback state and a single
   latest snapshot. A production-loop loud/pause/loud test must update without
   calling `_publish_presentation`, including between callbacks and during native
   drain. TASK-36544 has an explicit acceptance criterion; TASK-36550 carries the
   installed-path qualification. There is no additional publisher thread or UI
   clock dependency.

3. **P2: display removal contradicted ordinary-layout row reservation.**
   Textual's `DOMNode.display=False` removes layout space; `visible=False` reserves
   it. The ordinary-layout test now asserts displayed-but-invisible meter contents,
   while a separate compact-layout test asserts row removal. Tests measure the
   actual portrait, meter and controls as well as the outer rail. TASK-36549 criteria
   require stable geometry through timing loss/recovery and speech transitions;
   compactness depends on measured layout rather than voice state. No extra
   container is required.

## Revised-plan review

Re-read the revised contracts against the sink, pump, TTS owner cleanup and
hands-free completion path, existing Audio fake factories, installed sounddevice
callback wrapper/documentation and installed Textual visibility properties.

- Native completion is separate from qualified mouth timing: invalid timing cannot
  truncate audio or fabricate a successful finish. The backend-independent callback
  retains its exception guard; only the sounddevice wrapper emits `CallbackStop`.
- Extending the internal factory completion keyword includes its fake callers.
  Existing public event types and `feed` behavior remain compatible; the changed
  timing of natural drain is explicit, including for callers with no observer.
- Stop/replacement can still abort a pending tail without joining the notification
  thread. Late/duplicate callbacks cannot turn interruption into successful drain
  or clear the next owner. The existing buffered-duration plus five-second pump
  watchdog remains the missing-acknowledgement cleanup bound.
- Repeated publication cannot make old audio valid indefinitely. The plan tests
  expiry of the last mapped interval. It also covers a blocked legacy listener:
  >100 ms stale snapshots close the mouth, queued work stays bounded and audio/Stop
  remain independent. A 20 ms wait is a nominal cadence, not a realtime guarantee.
- The nine-unit dependency graph is unchanged. The corrections fit the existing
  sink, owner, radio-frame and qualification tasks. Storage, artwork generation,
  local authority, revisions and existing motion-policy requirements remain intact.
- ADR-242 is amended rather than duplicated because it already owns the PCM,
  presentation and stable-rail contracts. To Do tasks retain empty implementation
  plans/notes and unchecked criteria; the shared plan holds their prepared steps.

Verdict: All three findings are addressed in the revised plan and affected task
criteria. No additional blocking planning issue was found in this pass. The plan
is ready for implementation; this review does not establish runtime correctness.

## Planning verification

- A `.venv/bin/python` structural check inspected 13 plan/spec/ADR/review/task
  documents: 30 local Markdown links resolve, all 15 Python examples parse with
  `ast.parse`, and the nine units retain lower-numbered task dependencies.
- All nine implementation tasks remain To Do with 56 unchecked criteria total.
  Their implementation plans and notes remain empty. Backlog MCP read-back confirms
  the four edited tasks and their review-document links.
- The repository's Backlog file, ID and Windows filename guard functions pass on
  the nine feature task files. All nine feature IDs occur exactly once across the
  live, completed and archived buckets; no unrelated tasks were renumbered.
- Scoped `git diff --check` passes. These are planning/document checks, not product
  tests; parsing the examples does not execute them.

Not checked: Product tests, real sound output/timing, image-provider quality and
live radio-frame layout. Those remain implementation/qualification obligations.
