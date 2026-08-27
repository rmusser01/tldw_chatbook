# TASK-3070.12 Console Realtime Controller Design

**Status:** approved by the owner 2026-08-24; review amendments and mandatory
evidence-only `dev` drift amendments applied 2026-08-25 and 2026-08-26. The
ownership design, final ratchets, and 56/0/1 classification are unchanged.

**Task:** `TASK-3070.12 - Extract Console realtime orchestration ownership`

**Depends on:** TASK-3070.11 and the approved
`2026-08-23-console-decomposition-wave6-closeout-amendment.md`

## Context

`ChatScreen` still owns the Console realtime coordination policy even though the
transport, session, finite-state machine, audio, transcript, and diagnostic primitives
already have narrower owners. The Wave 6 closeout inventory freezes this coherent
family at 57 screen methods and 1,997 physical source lines:

- 56 policy methods move out of `ChatScreen`;
- `_repaint_console_realtime_chip` is the exact screen-owned presentation stay;
- no framework-bound realtime delegate is justified;
- the conservative maximum screen residue is 19 lines, for a minimum net removal of
  1,978 lines and 56 direct methods.

The exact method membership remains executable source-of-truth in
`Tests/Architecture/test_console_wave6_closeout_inventory.py`. This design does not
reclassify that reviewed inventory.

The branch was rebased and revalidated on `dev` at `8b0180118` before implementation.
The required current-base drift gate found that four unrelated commits had added 37
net lines while leaving the exact 57-method realtime family, its 1,997/19 source
spans, and the 56/0/1 classification unchanged. At this amended base, `ChatScreen`
has 20,054 lines and 633 direct methods. All seven closeout inventory
tests pass, and 147 focused realtime tests pass. Two mounted Buddy assertions already
fail because their test app leaves the lazily initialized `persona_buddy_controller`
as `None`; both failures reproduce before this extraction. One older TASK-3070.9
intermediate line ceiling also fails because current `dev` has 20,054 lines against
its 19,922-line delivery ceiling; the realtime extraction's conservative result is
18,076 lines, so this task must earn that test back without weakening it.

The combined TASK-3070.12/.13 conservative projection remains valid at this base:
17,420 lines and 562 direct methods after the frozen removals, below the immutable
17,727 / 593 ceilings by 307 lines and 31 methods. The implementation must not hide
the baseline defects and must keep all currently passing behavioral coverage green.

The delivery rebase onto `dev` at `794ae11521` found 56 further unrelated net
`ChatScreen` lines from boot-import deferral, off-loop avatar prerendering, and the
next-send price feature. The exact realtime family, 1,997/19 spans, 56/0/1
classification, and 633-method pre-extraction count are unchanged. Conservative
pre-extraction arithmetic is therefore 18,132 / 577, but the actual rebased
extraction is 17,676 / 577: 400 lines below the unchanged 18,076 / 577 final
ratchet because the complete move removes 2,434 physical lines. The current-base
gate accepts only this reviewed 56-line pre-extraction drift and independently
requires the implemented branch to satisfy the unchanged final ratchet. The
combined TASK-3070.12/.13 projection becomes 17,476 / 562, still 251 lines and 31
methods below the immutable 17,727 / 593 ceilings.

The delivery bases through `65cf855371` changed only the repository diagnostic
inventory (`b53169e1f1`) and citation provenance in the Console chat controller/store
(`3daa56bf4f`), followed by an unrelated backlog task record (`65cf855371`), in
this scope; those bases did not change the screen counts or ownership classification.

The later delivery base `f9a06ff625` simplified per-turn change review and removed
73 further `ChatScreen` lines and 8 direct methods without changing the exact
realtime family, 1,997/19 spans, or 56/0/1 classification. The rebased extraction
is therefore 17,603 / 569, still 473 lines and 8 methods below the unchanged
18,076 / 577 ratchet. The combined TASK-3070.12/.13 projection becomes 17,403 /
554, 324 lines and 39 methods below the immutable 17,727 / 593 ceilings. The
generated diagnostic inventory records the same 49 realtime statements under the
controller, with 90 remaining on `ChatScreen`; aggregate delivery counts are 538
owners, 1,241 TASK-492 calls, 7,360 TASK-494 calls, and 8 sink files.

The final delivery base `c23113e2e0` changed only durable postcommit close handling
in the Console chat controller/store. It leaves `ChatScreen`, the realtime family,
and all ownership/count projections unchanged; its three additional TASK-492
diagnostics bring the aggregate delivery inventory to 538 owners, 1,244 TASK-492
calls, 7,360 TASK-494 calls, and 8 sink files.

The subsequent delivery base `0bde972ca8` changes only Watchlists behavior and the
generated diagnostic inventory in this scope. It leaves the Console source and all
realtime ownership/count projections unchanged; three additional Watchlists
TASK-494 diagnostics bring the aggregate inventory to 538 owners, 1,244 TASK-492
calls, 7,363 TASK-494 calls, and 8 sink files.

The latest delivery base `6bed8d6f59` changes only Library rail behavior,
configuration, styling, documentation, and tests in this scope. It leaves the
Console source, realtime ownership/count projections, and diagnostic inventory
unchanged.

The subsequent delivery base `6c535bcd16` changes only the Library Prompt reader,
its evidence, documentation, tests, styling, and generated diagnostic inventory in
this scope. It leaves the Console source and realtime ownership/count projections
unchanged; two additional Library TASK-494 diagnostics bring the aggregate
inventory to 538 owners, 1,244 TASK-492 calls, 7,365 TASK-494 calls, and 8 sink
files.

The delivery base `4cee590b83` adds the first-run setup wizard and updates
the shared Console chat controller, but does not change `ChatScreen` or the
realtime controller family. Four additional wizard TASK-494 diagnostics bring the
aggregate inventory to 538 owners, 1,244 TASK-492 calls, 7,369 TASK-494 calls, and
8 sink files; the ownership and size projections remain unchanged.

The final delivery base `5732d276dd` changes only the change-review screen and its
focused tests. It leaves the Console realtime source, ownership/count projections,
and diagnostic inventory unchanged.

## Goals

1. Give the 56 reviewed realtime policy methods one explicit, non-DOM owner.
2. Preserve session/tap/sink identity, first-words buffering, transcript publication,
   usage, fallback, reconnect, barge-in, remount, teardown, diagnostic, and privacy
   behavior.
3. Leave Textual composition, DOM access, focus, framework callbacks, and chip repaint
   presentation on `ChatScreen`.
4. Reduce `ChatScreen` by at least the frozen conservative projection without raising
   either size ratchet.
5. Make the new owner directly testable with plain fakes and named dependencies.

## Non-goals

- Redesigning realtime UX, transport, audio, transcript, retry, timeout, or fallback
  behavior.
- Moving or replacing the existing transport/session/FSM/audio owners.
- Fixing adjacent realtime behavior or adding provider/microphone integration.
- Raising or lowering the Wave 6 ratchet in this task. TASK-3070.14 owns the final
  measurement and earned ceiling reduction.
- Adding a compatibility mixin, dynamic method facade, screen callable delegates, or
  mirrored realtime state.
- Running the local full test suite. Required GitHub Actions remain the broad
  integration gate.

## Considered Approaches

### 1. Dedicated `ConsoleRealtimeController` (chosen)

Create `tldw_chatbook/UI/Console_Modules/realtime.py` with the session model,
controller-only constants, and the 56 policy methods. Construct it in
`UI/Console_Modules/wiring.py` and install it as `screen._realtime`.

This follows `DESIGN.md` section 7, provides one inspectable owner, removes the full
reviewed method set, and supports isolated tests without mounting Textual.

### 2. Realtime mixin on `ChatScreen`

A mixin would make the source file smaller but would leave ownership on the screen,
retain implicit access to all screen state and DOM, and defeat the architectural gate.
It is rejected.

### 3. Thin screen delegates to a service

Keeping 56 screen methods as forwarders would preserve old call sites but retain the
direct method inventory, create dual navigation paths, and consume unjustified
residue. No realtime method is framework-bound, so this approach is rejected.

## Ownership and Module Boundary

`ConsoleRealtimeController` owns Console-specific realtime orchestration:

- session construction and lifecycle sequencing;
- connect, reconnect, tick, tap, pump, sink, and close coordination;
- FSM intent handling and barge-in policy;
- transcript row lifecycle, output accumulation, usage, and persistence decisions;
- failure classification, sanitization, notification, and pipeline fallback;
- remount-safe Buddy generation acquisition/release;
- retained close-worker tracking and final teardown.

The controller must not query the DOM, focus widgets, open modals, reference sibling
controllers, or depend on a `ChatScreen` object. It receives named dependencies for
every interaction outside its boundary.

`ChatScreen` retains:

- Textual lifecycle and input callback boundaries;
- `_repaint_console_realtime_chip`, including its DOM/presentation work;
- the compatibility attribute surface described below;
- a small `on_key` call into the controller entrypoint;
- `on_unmount` awaiting controller teardown.

The existing realtime transport, protocol, audio, transcript, FSM, persistence, and
diagnostic modules keep their current responsibilities.

## Construction and Dependencies

The existing `build_console_controllers()` entrypoint in
`UI/Console_Modules/wiring.py` constructs the controller and assigns
`screen._realtime`; this task must not add a second wiring API.
Dependencies are explicit, named, and late-bound when the target can change across
mounts or test substitution. They cover:

- current session settings and API credentials;
- current app runtime/store/Buddy services;
- dictation state and transcript adoption;
- pipeline fallback dispatch;
- transcript row append/update and metadata synchronization;
- screen-owned chip repaint and voice-chip restoration;
- notifications and callback marshaling.

Framework services are read through late-bound callables so controller behavior
follows the currently mounted screen/app services. A stable `app_instance` snapshot
may be captured only where identity is intentionally fixed for the controller's
lifetime; that exception must be named and justified in wiring. The controller must
not keep an ambient screen reference or use generic `getattr` reachbacks as a hidden
service locator.

Dictation and hands-free wiring call the controller directly at call time for the
session accessor, transcript adoption, and loop entry. Auto-speak reads the current
controller state directly. No wiring callback points back through a removed screen
method.

## State and Compatibility

The new module owns `ConsoleRealtimeSession` and constants used only by the realtime
controller. `chat_screen.py` imports only the chip-message mapping needed by
`_repaint_console_realtime_chip`; it must not re-export controller implementation
symbols merely to preserve test patch handles.

The existing private attributes `_console_realtime` and
`_console_realtime_close_worker` remain temporarily available as fail-loud forwarding
descriptors on `ChatScreen`:

- reads and writes forward to the corresponding controller state;
- the descriptors never store shadow copies;
- access before wiring raises `RuntimeError` with an actionable message;
- the descriptors are the only compatibility bridge; callable delegates are
  forbidden.

Tests that monkeypatch provider, model, voice, VAD, timeout, or helper symbols through
`chat_screen_module` must patch the new owning `realtime` module instead. Preserving
stale re-exports would make tests pass while production resolves a different symbol.

## Runtime Flows

### Enter and connect

1. A Textual action or dictation/hands-free callback asks the controller to enter.
2. The controller creates and installs the session before entering the FSM. This
   ordering is required because a synchronous FSM intent can repaint immediately.
3. The controller acquires the Buddy generation, creates callbacks/sink/tap, and
   starts the retained connection worker using the existing ordering and identities.
4. Ready, frame, transcript, audio, usage, and reply events marshal through the same
   UI-thread boundary as today.

### Transcript and reply lifecycle

The controller preserves seed items/text, input transcript adoption, pending row
metadata, first-output buffering, audio availability, playback timing, transcript
empty-state handling, final row publication, and persistence. Screen callbacks passed
through wiring perform only the presentation operation they name.

### Reconnect and fallback

Connection failures preserve the current tokenization, sanitization, reconnect state,
tap buffering, timeout values, and retry/fallback decisions. Exhaustion invokes the
existing pipeline fallback dependency. Errors continue to avoid exposing credentials,
raw provider payloads, or private audio/transcript data through notifications or
diagnostics.

### Barge-in and Escape

`ChatScreen.on_key` remains the framework boundary. It calls one small controller
entrypoint that returns whether Escape was consumed. The controller owns the realtime
barge trigger and FSM policy; the screen owns event stopping/prevention and unrelated
key handling. No removed realtime method remains as an `on_key` helper delegate.

### Exit, remount, and teardown

Exact loop exit continues to cancel active work, close tap/sink/session resources,
release only the acquired Buddy generation, restore presentation, and retain any
close worker that outlives the screen transition. A replacement screen may acquire a
new generation without a stale screen releasing it.

`ChatScreen.on_unmount` awaits `self._realtime.teardown()`. Controller teardown
preserves both paths already characterized by tests:

- release/close of an active session;
- waiting for a retained close worker after the visible session has already exited.

Repeated teardown remains safe and does not rearm or leak workers.

## Error Handling and Privacy

The extraction preserves existing error categories and user-visible messages.
Controller callbacks must continue to be safe after cancellation or remount, and
late-arriving events must not mutate a successor session. Cleanup is best-effort only
where the existing implementation is best-effort; failures that currently surface
must continue to surface.

Diagnostics keep the existing event names and redaction boundaries. API keys, raw
provider failures, audio frames, and private transcript content must not enter logs,
diagnostic fields, or notification strings.

## Verification Strategy

Implementation follows red-green-refactor and uses only focused local gates.

### RED-first architecture tests

Add or tighten source-inspected tests that fail before extraction and prove:

- all 56 exact move methods are absent from `ChatScreen` and owned by
  `ConsoleRealtimeController`;
- `_repaint_console_realtime_chip` is the only realtime family stay;
- there are no realtime screen callable delegates, dynamic facades, or mixins;
- the controller contains no Textual DOM queries or sibling-controller references;
- the two compatibility descriptors have no shadow storage and fail loudly before
  wiring;
- wiring supplies the reviewed named dependency boundary;
- `ChatScreen` meets the conservative line and direct-method reduction without a
  ratchet increase.

### Isolated controller tests

Use plain fakes, not mounted Textual or broad mocks, to cover:

- construction and session-before-FSM ordering;
- callback marshaling and stale-session rejection;
- connect/reconnect/tap buffering and fallback;
- transcript/reply/audio/usage lifecycle;
- barge-in consumption;
- active and retained-worker teardown;
- generation-safe Buddy release;
- error sanitization and privacy.

### Mounted and lower-layer regression tests

Keep the applicable assertions in `Tests/UI/test_console_realtime_wiring.py`, changing
only the receiver and owning-module patch handles required by the extraction. Correct
the two pre-existing lazy-Buddy test setup defects only if necessary to exercise the
same product contract; do not change production behavior to satisfy stale fixture
assumptions.

Run the focused realtime UI suite and related Chat, Audio, LLM protocol, architecture,
privacy, and diagnostic inventory suites. Do not run the local full suite.

Run targeted mutation checks over the extracted orchestration branches that carry
behavioral policy (session-before-FSM ordering, stale-session rejection, reconnect
exhaustion/fallback, generation-safe release, and retained-worker teardown). The
repository has no mutation-testing framework, so these use its established manual
discipline after a checkpoint commit: apply one temporary semantic edit, run the exact
focused pytest node(s) and observe RED, restore the edit with `apply_patch`, then rerun
GREEN and confirm a clean diff. The implementation plan must name each temporary edit
and its focused pytest command. Do not add a mutation dependency or run an unbounded
repository mutation sweep.

### Static and repository gates

Run targeted Ruff and formatter checks on modified Python files, isolated compile,
`git diff --check`, backlog ID validation, diagnostic inventory validation, and the
Wave 6 size/inventory gates. Review the final diff for accidental public API, logging,
privacy, or unrelated changes.

No real microphone or provider call is required because this is a behavior-preserving
ownership extraction and the transport/audio contracts remain unchanged.

## Delivery Sequence

1. Freeze architecture expectations with failing tests.
2. Add the controller state model and plain-fake tests.
3. Move policy in coherent lifecycle slices while keeping focused tests green.
4. Wire the controller and compatibility descriptors; retarget owning-module patches.
5. Remove all 56 screen methods and verify the exact stay/residue contract.
6. Run the focused behavioral, architecture, static, privacy, diagnostic, and
   repository gates, including the bounded mutation checks.
7. Record task implementation notes and evidence. Rebase on latest `origin/dev` again
   before PR completion, then rerun the exact family/projection inventory. If any
   later rebase changes the reviewed 57-method family, invalidates the 56/0/1
   classification, or breaks the conservative projection, stop and amend the
   design/task before further implementation or delivery. Never rewrite the frozen
   evidence or raise the ratchet to accommodate drift.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this task directly applies the accepted controller/region ownership rules in
`DESIGN.md` section 7 and the approved Wave 6 closeout amendment. It changes no
storage, schema, service contract, security/privacy policy, provider boundary,
dependency, or long-lived application structure. A new ADR would duplicate existing
decisions.
