# Lessons: what counts as evidence a change works

Working knowledge about testing in this repo. Not decisions (see `backlog/decisions/`)
and not point-in-time audits — these are traps that have actually cost time here, kept
so the next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore is ignored. If you add one, bring the incident.

---

## Textual's geometric center is not the painted row for an even-height one-line control

**TASK-16001, 2026-08-13.** A compositor regression helper sampled
`region.center.y` to verify that collapsed rail copy was vertically centered.
The focused RED failures initially hid the helper defect. Once the rail copy was
correct, all eight visual cases raised `AttributeError`: in the installed Textual,
`Region.center` is a `(float, float)` tuple, not an object with `.y`. Replacing it
with `region.y + region.height // 2` removed that error but still sampled an empty
row for even-height controls. Measurement showed a 28-row button at `y=7` painted
its sole centered row at `y=20`, the upper middle:
`7 + (28 - 1) // 2`.

**What to do.** For a one-line Textual control, sample the integer painted middle
row with `region.y + (region.height - 1) // 2`; do not assume `Region.center` has
named coordinates or that lower-middle sampling matches Textual's alignment.
Keep a separate one-painted-row assertion so the midpoint check cannot pass on a
multi-line control.

---

## A geometry harness must mount the production hierarchy and stylesheet

**TASK-16221, 2026-08-14.** The first Watchlists Read geometry harness mounted
`ArticleListPane` directly where production mounts a detail wrapper, title, and
nested pane. With the inner table capped at 42 rows, the simplified harness
reported a contained, painted pager. A production-shaped probe showed the real
detail pane growing to 51 rows inside a 50-row ITEMS region, placing the pager
exactly outside the clipped box; the rendered frame contained none of Previous,
Page 1, or Next. Rebuilding the harness with the real wrapper/title hierarchy
made the regression fail, and the correct 40-row table cap kept the pager inside.
During final live QA, a second simplified host loaded consolidated widget/screen
CSS but omitted the app bundle; its regions measured 8/5 rows until the harness
used the exact `TldwCli.CSS_PATH` stack.

**What to do.** For layout limits, reproduce every production ancestor that
contributes rows and load the same stylesheet sources in the same order as the
application. Assert containment and compositor text, not only a child widget's
declared height. A shortened DOM or partial stylesheet can make both overflow
and clipping tests pass for a product path that still hides its controls.

**Recurred, TASK-16478, 2026-08-15.** A picker-comparison investigation
rendered `EnhancedFileOpen` in a bare `App` (widget DEFAULT_CSS only) and
concluded the dialog was fine; the user's live app showed no Select/Cancel
buttons at all. Under the app bundle, the bare `Select { width: 100% }` rule
beat the dialog's DEFAULT_CSS, crushed the filename input to 6 columns, and
laid the buttons out at x=161/178 inside a 152-wide dialog -- clipped. The
bare-host screenshot even contained the buttons, and a truncated text
extraction of the bundled render hid their absence. The fix's regression test
(`Tests/UI/test_enhanced_file_dialog_bundle_css.py`) registers the exact
`TldwCli.CSS_PATH` stack and asserts button containment -- it failed red
against the unfixed bundle without touching app code.

---

## An exact live-test gate must be the first gate that can skip the test

**TASK-15676, 2026-08-13.** The opt-in Moonshot/Z.ai paid harness required an
exact provider flag plus a nonblank provider key, and its user documentation
showed that two-part command. The first default run never reached that contract:
the repository's `slow` marker skipped the test with `Need --run-slow`. Removing
that marker exposed the same problem from `optional` and `--run-optional`. Even
with the documented environment flag and key present, the documented command
could not execute the live case because unrelated collection-time gates ran
before the test body's explicit safety check.

**What to do.** When the public contract is an exact environment/key gate, do
not also apply repository markers whose plugins skip before the test body unless
every required CLI flag is part of the documented contract. Keep descriptive
markers such as `integration`/`allow_network`, put the paid-call guard at the
top of the test, and cover its truth table plus the default skip reason. That
makes default collection safe while ensuring the opt-in command can actually
reach the code it claims to verify.

---

## A targeted async completion must not rebuild a surface that is transitioning away

**TASK-15706, 2026-08-13.** Database Notes began loading its folder tree in a
background worker. If the user switched to File Notes before that worker
finished, the completion path tried to synchronize `#library-notes-canvas`.
That canvas was legitimately absent during the source transition, so the shared
sync helper used its generic full-screen recompose fallback. The fallback
invalidated the just-pressed Files source control and intermittently left the
transition without its retained File Notes surface. Folder-tree tests all
passed; only the production-shell source-switch tests reproduced the race.

**What to do.** An async completion that owns one optional child surface should
first confirm that exact surface is still mounted. If it is absent because the
user navigated away, treat the result as cached state and skip the paint; do not
invoke a generic whole-screen fallback. Verify the fix through the real route
transition, and compare the same test against the untouched baseline before
attributing nearby focus failures to the branch.

---

## A schema-version label does not make a synthetic database historical

**TASK-15705/TASK-15707, 2026-08-12.** Raising ChaChaNotes from v35 to v36
first broke migration tests that had either stamped a tiny hand-written database
with the then-current version or pinned ``_CURRENT_SCHEMA_VERSION`` while using
the evolving bootstrap SQL. The former skipped migration with required tables
missing; the latter silently included columns that did not exist at the claimed
historical version. Focused tests for the new v36 migration passed, but the full
DB suite exposed both fixture classes: current-version fixtures failed during
startup maintenance, while an incomplete v24 fixture failed only after reaching
a much later migration.

**What to do.** A migration fixture must prove the historical preconditions that
matter to the migration under test: schema version, required tables/columns, and
the absence of fields being introduced. When later code needs a complete current
database with one malformed record, create a real current database and alter only
that record or table; do not label a partial schema as current. After every schema
bump, run the complete DB migration suite, not only the new migration module.

---

## A "slow-accept listener" does not delay TCP connect() — it delays accept()

**TASK-15473, 2026-08-11.** Writing an evidence test that the event loop stays
responsive during a non-blocking socket probe, the task's own brief suggested "a
slow-accept listener" as the portable way to simulate an unresponsive server. Timed
directly before writing the test: a real `socket.listen()`ing server that never calls
`accept()` still let a client's `socket.create_connection()` complete in ~7ms. TCP's
three-way handshake completes at the OS kernel level as soon as a connection is
queued in the listen backlog — independent of whether the application ever calls
`accept()`. A "slow-accept" listener therefore cannot be used to create connect-side
delay; it only delays whatever happens *after* the client tries to read/write, which
this probe (connect-then-immediately-close, no data exchange) never does.

What actually produced a real, mutation-verified delay: connecting to a private,
non-routed address (`10.255.255.1`) that neither answers the SYN nor sends back an
ICMP unreachable — a genuine kernel-level "black hole" — measured to hang for the
full requested timeout in this sandbox (no immediate "network unreachable"). The
resulting test caught a real regression: reverting the probe to a blocking
`socket.create_connection` call inside the coroutine dropped a 5ms-period heartbeat
task from ~44 ticks to 0 during the same ~0.25s window.

**What to do.** Before trusting "slow accept" (or similar accept-side framing) to
simulate a connect-side timeout in a test, time it directly — a bound-and-listening
socket with a deliberately delayed `accept()` will not slow down a bare `connect()`
on any common OS. For a genuine connect-timeout test, either use a real black-hole
address (accepting the environment-dependence, verify empirically first) or
mutation-test whatever mechanism you do use against the blocking equivalent it's
supposed to replace — the ~44-vs-0 heartbeat contrast is what proved this test was
not vacuous.

---

## Style probes are not render evidence — capture the frame

**TASK-15421 AC3, 2026-08-11.** The Studio exact-ID input's typed text
vanished while focused in the live TUI. The hunt fixated on border rules for
hours because every probe asked `styles.border` — which was empty, correctly,
in both the live-matching harness and run_test — so the harness appeared to
CONTRADICT the live app and the divergence got recorded as an unexplained
live-vs-run_test cascade anomaly. There was no divergence: the reset-tier
accessibility rule `*:focus { outline: solid }` paints the outline OVER the
widget's outermost rendered lines (its own comment warns of this), and on a
height-1 widget that line IS the only content line. The obscuring reproduced
in run_test all along; no probe ever looked at a rendered frame. One
`export_screenshot()` assertion (`assert "studio-model" in frame`) found in
minutes what specificity analysis could not, and now pins the fix in
`Tests/UI/test_speech_live_render_defects.py` — a file whose own docstring
already teaches a version of this lesson ("the tests asserted the things a
test naturally reaches for ... none of which is what was wrong").

**What to do.** When the defect is "the user cannot SEE something," the
oracle must be the rendered frame, not computed styles: in run_test that
means `app.export_screenshot()` (the SVG carries every glyph as text, so a
plain `in` assertion works) or the compositor strips the existing UI tests
use — NOT `App.export_text()`, which does not exist in this repo's Textual
(8.2.7; the probe that first tried it died on AttributeError) — and live it
means the tmux `capture-pane` text. `styles.border`, `styles.height`, and
`region` all report the widget's own properties and are blind to anything
painted over it — outlines, overlays, tooltips, sibling z-order.
Before declaring a live-vs-harness divergence, confirm both sides were asked
the SAME question at the same oracle level; here the "divergence" was one
side being read at the style level and the other at the pixel level.

---

## Preserve the visible set across a reorder, not its former first row

**TASK-15455, 2026-08-11.** Console transcript windowing initially preserved a
lazy window across refreshes by finding the first previously visible message id
that still existed in the new ordered list. That was correct for append-only
streaming and session-local deletes, but wrong for branch/path reorder: moving a
later visible message ahead of that chosen id put it into the newly computed
hidden prefix. The focused windowing tests were all green. The pre-existing
signature-cache reorder contract caught the missing mounted row.

**What to do.** When a windowed projection accepts a reordered full list,
preserve the minimum new index of every surviving previously visible item (plus
any explicit selection handoff), not the new index of one former boundary item.
Include an order-sensitive DOM assertion in the reachable regression set; cache
counts alone prove reuse, not that every reused row stayed visible.

---

## A fix proven at one layer can be unreachable through the product path

**TASK-15420, 2026-08-11.** TASK-2260 (2026-08-04) shipped custom-endpoint
model/voice passthrough in `OpenAITTSBackend`, pinned by mutation-verified
backend tests and a real-socket keyless server test, and its user guide was
live-verified — by varying the *voice*. The Console speak path, however, had
been rerouted through the request-admission layer (2026-07-26), whose
`resolve_legacy_route` allowlist rejected every non-official OpenAI *model id*
before the backend was even constructed. The documented flow (exact custom
model name) failed on every Console speak for weeks while all ~2,900 TTS-area
tests stayed green: the tests proved the backend layer, the live check varied
the one axis the upstream layer did not constrain, and nothing exercised the
full admitted path with a custom model. Found only by end-to-end UAT driving
the real TUI against a request-recording mock server.

Sub-trap from the same session: `TTSAudioResponse.byte_stream` is lazy — a
probe that calls `synthesize_default` and prints the response "succeeding"
proves nothing, because the HTTP request only fires when the stream is
consumed. The first counterfactual probe "passed" with zero requests at the
server; only draining the stream produced the real request.

**What to do.** A regression test for a passthrough/compatibility contract
must enter at the outermost admitted path (here: `synthesize_default` down to
the adapter), not the layer that was fixed — any layer added above the fix
inherits the chance to re-impose the constraint. When live-verifying, vary the
axis the bug is about (the model, not just the voice). And never trust a
lazy-stream API's return value as evidence of I/O — consume it and assert at
the far end (the recording server).

**TASK-13204, 2026-08-10.** A clone-shutdown regression initially asserted
`TTSAdapterRegistry._total_leases()` while the provider was past its bounded
shutdown deadline. The registry had already moved the active adapter record
into its retained closing-record collection, so `_total_leases()` returned zero
even though the exact record still held a lease. That false measurement first
made the race look fixed. Inspecting the retained record proved the real bug:
generic late-operation cleanup could release an executing clone lease before
its protected materialization boundary finished. The corrected regression
asserts the closing record's lease count and mutation-fails when the protected
execution waiter is removed.

**What to do.** When testing ownership after a bounded shutdown transitions to
retained cleanup, measure the owner in its terminal collection or assert the
actual release/close barrier. A convenience counter scoped to active/retired
records may truthfully return zero after records are transferred, while
definitive cleanup is still outstanding.

---

## A registry entry needs both inventory and behavioral-ratchet coverage

**TASK-13203, 2026-08-10.** The new `tts.profile_migration_backup` SQLite owner
was present in the central policy registry, the curated owner inventory, and focused
migration lifecycle tests. Those suites all passed. The complete private-SQLite gate
still failed because the owner was absent from the generic centralized-backup
behavior matrix, so its declared `centralized_backup_allowed` capability had no
owner-parameterized seam test.

**What to do.** When adding a policy-registry owner, search for both inventory
ratchets and capability-derived behavioral matrices. Passing the feature's own test
does not prove that every capability bit has generic seam coverage; run the registry
module's complete contract suite before closeout.

---

## Deleting a duplicate guard requires tests for every bypass mode

**TASK-859, 2026-08-02.** During specification review, deleting
`SecurityValidator.ALLOWED_SCHEMES` looked like clean policy consolidation because
`Utils.egress` also has scheme policy. Direct characterization disproved that:
`Utils.egress` returns allowed immediately when `[web_security].enabled = false`,
before evaluating its scheme policy. Without the subscription-local HTTP/HTTPS
allowlist, disabled egress would therefore admit `ftp://` subscriptions. The
regression test that disables egress and submits an FTP URL remains red without
the local boundary and green with it.

**What to do.** Before deleting an apparently duplicate guard, characterize every
disabled or bypass mode of the surviving owner and test that owner's boundary in
each mode. Consolidate only after evidence shows the remaining guard preserves
the contract there.

---

## A fake written to match your call site validates the mistake

**The trap.** You write a test double to match how you are calling the real thing. If
the call is wrong, the double is wrong in the same way, and the test passes forever.

**What happened.** Three times on one branch (task-684 series):

- `cancel_media_ingest_jobs_batch` is keyword-only; it was called positionally. The
  fake declared a positional parameter, because it was written to match the call.
- The remote-ingest poller asked for an `offset` the real client did not accept. The
  fake declared `offset`. **Pagination was dead in production** and 900+ tests were green.
- `MediaIngestJobStatus.result` was typed as a different domain's model. Every fixture
  matched the wrong model, so **every completed job was unparseable** and the queue
  would have shown jobs stuck at "queued" forever.

**What to do.** For anything crossing a seam you do not own, assert against the **real
signature** and a **verbatim captured payload**, not a hand-written double:

```python
assert _accepts_keyword(RealService.method, "offset")     # the real signature
LIVE_RESPONSE = { ... }                                    # pasted from the wire
```

A fake can agree with a wrong assumption; `inspect.signature` cannot.

**Sharpest variant (task-16847, 2026-08-16).** A double can stand in for an attribute
that does not exist at all. `a8082fe85`'s launch test set
`instance.call_from_thread = ...` and `instance.push_screen = ...` on a
`ChatScreen.__new__` instance — but `Screen` defines *neither* (both are App-only in
Textual 8), so pressing `y` in the real app raised `AttributeError` inside the thread
worker while the test stayed green, and the repo-wide guard
(`Tests/test_call_from_thread_guard.py`) sat red on dev for two days. When a unit test
must fake threading/navigation seams, patch the **collaborator** (`app`) — never spell
a new attribute onto the class under test; an instance monkeypatch is also an
existence claim, and nothing checks it.

**Widest blast radius (TASK-17065, 2026-08-17).** One module diverged from the house
pattern in *two* ways at once, and the single fake at its seam mirrored both.
`RAG_Search/reranker.py` grew its own credential path (`self._settings =
load_settings()`, then a hand-rolled `if/elif` reading
`settings["API"]["<provider>_api_key"]` — a key `load_settings()` never builds) *and*
its own dispatch convention (a positional argument list handed to `chat_api_call`
through `run_in_executor`, which forwards positionals only). The seam fake,
`Tests/RAG_Search/test_reranker_degraded_paths.py`'s `def
fake_chat_api_call(api_key, messages_payload, provider, model, temp, maxp)`, declared
the caller's own wrong order *and* planted a `_settings` table so the call got past
the credential gate. Agreeing with both defects, it left ~2,500 green tests unable to
see that reranking completed a scoring call for **zero of the 29 providers**
`chat_api_call` dispatches — for the entire life of the feature. Binding the same call
through `inspect.signature(chat_api_call).bind(...)` printed the truth in one line:
`api_endpoint='THE-API-KEY'`, `api_key='openai'`, `temp='gpt-4o-mini'`,
`system_message=0.25`, `streaming=128` — the mis-binding had also silently switched
STREAMING on, a third defect nobody had filed.

Two rules out of it:

- **A fake at a seam you share with the rest of the app must bind against the real
  signature, never re-type the call site's argument list.** Note how far this reaches:
  even the guard written specifically to catch this
  (`test_reranker_dispatch_binding_against_the_real_chat_api_call_signature`) first
  asserted a literal tuple *it typed itself*, so it guarded a copy of the caller and
  caught nothing. It only became evidence once it drove the real `_call_llm_impl` and
  observed what landed. **And know exactly what `bind` buys you**, because the fixed
  fake's first docstring over-claimed it and the final review caught that:
  `inspect.signature(...).bind()` checks arity and keyword *names* only — it is BLIND
  to order. Re-measured on this very call:
  `bind("THE-KEY", [...], "openai", "gpt-4o-mini", 0.25, 128)` is ACCEPTED (it simply
  lands the key in `api_endpoint`), while `bind(provider="x", ...)` raises
  `unexpected keyword argument`. What actually catches a mis-ordering is the landing
  ASSERTIONS on a guard that drives the real caller — plus, cheaply, refusing
  positional arguments at the fake (`assert not args`) when the seam is keyword-only
  by contract. Mutation-checked: reverting the call site to positional now fails with
  *"positional arguments landed here: ('openai',)"*, not with a bind error.
- **A feature that resolves credentials itself is a divergence to justify, not a
  default.** The fix here was a DELETION: all 29 handlers already resolve their own key
  or need none, and every other `chat_api_call` caller in the repo
  (`UI/Tools_Settings_Window.py`, `UI/Screens/evals_screen.py`,
  `Chat/console_provider_gateway.py`) already passes keywords and omits `api_key`. The
  reranker was the sole outlier, and being the sole outlier is exactly what broke it.
  Before writing a lookup at a shared seam, count the callers who do not have one.

---

## Mutation-test every guard you add

**The trap.** A test that cannot fail is worse than no test: it reports safety that
does not exist.

**What happened.** Three fixes in one session were **vacuous** and only mutation
testing caught them:

- A precedence trap patched `get_cli_setting` to raise — but the code under test wraps
  that call in `except Exception:`, which **swallowed the AssertionError**. Deleting the
  entire precedence branch still passed.
- A wait helper's timeout was checked in the loop *header*, so a pause overshooting the
  deadline exited without re-testing the condition — reporting "never mounted" for a
  widget that had just appeared. The guard for "can the helper still time out?" passed,
  because it waited on a condition that is never satisfied.
- A first-run classifier keyed on an in-memory counter that resets each run, so a real
  failure in the first batch after restart was downgraded. Three tests passed with the
  bug present; none modelled a restart.

**What to do.** After writing a guard, break the thing it guards and confirm the test
fails. If it still passes, the guard is decorative. Prefer **recording and asserting
after the fact** over raising inside code that catches broadly.

---

## Mount dispatch can be attached before it is mounted

**TASK-15459, 2026-08-13.** A deterministic `asyncio.Event` barrier released
the Library source worker while `LibraryScreen.on_mount()` was still awaiting.
The fresh snapshot reached the screen and advanced its state generation, but
the rendered generation stayed behind and the targeted-sync recorder remained
empty. Instrumentation at the snapshot boundary showed the exact lifecycle
state: `is_attached` was true while `is_mounted` was false. The reconciliation
scheduler used `is_mounted`, so it silently discarded completion during the
Mount dispatch window. Changing only that scheduler boundary to attachment
authority made the RED race pass, while the detached-completion regression
continued to return `SUPERSEDED` with zero DOM calls.

**What to do.** When work may complete during a Textual Mount handler, do not
infer that attachment and mounting flags change together. Gate message-pump
scheduling on the lifecycle property the operation actually needs (attachment
for queuing to the screen), then keep a second current/attached guard at DOM
execution time. Prove both halves with Event barriers: completion during Mount
must render, and completion after detach must do nothing.

---

## Trigger cancellation from the state the test claims to cancel

**TASK-3771, 2026-08-11.** A QwenCloud native-tool regression claimed to prove
that cancelling after an incomplete streamed function call never executes the
tool. The test actually set its cancellation flag in the mock server's
``on_request`` callback, before the response body or partial call reached the
stream parser. Replacing both partial-call fixtures with ordinary final text
still passed: the test proved only pre-response cancellation and closure.

The corrected test waits until the real ``ConsoleChatStore`` receives a visible
checkpoint that follows an incomplete tool-call delta, then requests
cancellation while the chunked response remains non-terminal. A text-only
mutation now fails at the fixture guard, and the test separately proves the
live response closes exactly once without executing or pairing the partial
call.

**What to do.** If a cancellation test names a lifecycle state (after headers,
after one chunk, after partial tool state), trigger cancellation from an
observation downstream of that exact state. Do not trigger it at request
receipt and infer that later layers ran. Mutation-replace the special prefix
with a normal successful response; the test must fail for the reason it claims
to cover.

---

## A surviving mutant usually means a SECOND writer satisfies your assertion

**The trap.** You delete the code under test, the test stays green, and the reflex is
to strengthen the assertion. The real question is *who else produces the asserted
outcome* — if a second mechanism writes the same state, the test is measuring that
mechanism, not your feature.

**What happened.** Task-3313 (2026-08-09), twice in one session:

- Deleting the "Retry this batch" **options restore** stayed green because ingest
  options deliberately persist across submits — the form still held the values the
  restore was supposed to bring back. Fix: the test now *corrupts* the options and
  metadata between submit and retry, so only the restore can produce the asserted
  values. The mutant then failed on the exact line.
- Deleting the **fresh pre-flight trigger** stayed green because the test's
  programmatic `path_input.value = …` had armed the 0.8s typing debounce, which fired
  *after* the re-stage at test speed and re-ran the analysis the mutant no longer
  requested (found by wrapping the trigger and printing call stacks: the second call
  came from `_run_debounced_library_ingest_preflight`). In production that timer fires
  as a no-op long before a human reaches the button — a pure test-speed artifact. Fix:
  the harness stops the pending debounce before the action under test.

**What to do.** When a mutant survives, instrument for the *second writer* (wrap the
seam, record call stacks) before touching the assertion. Then either perturb the state
so only the code under test can restore it, or silence the background mechanism in the
harness — and re-run the mutant to see it actually die.

---

## A widget reference captured before a structural recompose is a silent key sink

**The trap.** A Pilot test grabs `path_input = screen.query_one(...)`, performs an
action that lands a state change, then `set_focus(path_input)` and `pilot.press(...)`.
If the state change took the STRUCTURAL recompose path, every form widget was
replaced; the captured reference is a detached widget. Focusing it "works"
(`screen.focused` even reports it), but keys go nowhere — no error, no typing, no
`Submitted`.

**What happened.** Task-3314 (2026-08-09): the inline-consent pilot tests captured the
ingest path Input, then let a pre-flight result land — which changes the type-group
set and forces the canvas's context-preserving full recompose. Enter then never fired
`Input.Submitted`; three probes (handler spies, app-level recorders, a source-level
log) all showed the handler simply never ran, and typing `x` changed nothing. The fix
was one line: re-query the input *after* the forecast settles. (Related but distinct
from the "pin object identity across the in-place path" lesson — here the identity
break is *expected*, and the test must follow it.)

**What to do.** In pilot tests, re-query any widget you focus or press *after* the
last action that can recompose its region; treat "keys typed but value unchanged" as
a detached-focus symptom, not a key-routing mystery.

---

## A new widget in a shared row needs geometry assertions, not just display/text

**The trap.** You add a small widget to an existing `Horizontal` row and assert it
displays with the right text. Textual's default `Widget`/`Static` width is `1fr`, so
the new child quietly claims the entire row and pushes every sibling off the screen
edge. Display and text assertions stay green — they never look at positions.

**What happened.** TASK-2154.1 added `#console-compact-status-marker` as the first
child of the Console control-bar action row. Six new Pilot tests passed (marker
visible, correct label, rails behave), but the 80x24 UAT screenshot showed **every
control button gone**: the bare `Static` took the row's full 78 cells and the buttons
laid out at x=79+, off-screen. One line — `marker.styles.width = "auto"` — fixed it;
a regression test asserting each button's `region` stays inside the screen locks it,
and was itself mutation-checked by deleting the width line (it fails: `x=90 + 16 > 90`).

**What to do.** When you mount anything into a laid-out row/column you do not own,
assert the **neighbours' geometry** (`region.x + region.width <= screen width`), not
only your widget's `display`. If your widget is a `Static`, set `width = "auto"`
explicitly unless you actually want it to eat the row.

---

## A dynamic Button label can repaint without reflowing its width

**TASK-3795, 2026-08-08.** Speech Lab composed its primary audio.cpp action as
`Test`, then passive runtime state changed the same mounted Button to
`Start & Test Connection`. Every test asserted the new label string and passed.
The live browser UAT nevertheless rendered only `Star`: Textual repainted the
reactive label but retained the original 16-cell layout width. The first honest
geometry regression measured 16 cells against the 25 required for the new label
and failed until the update called `refresh(layout=True)`.

**What to do.** When a mounted auto-width widget changes content, verify the
rendered region after refresh, not just its value. For dynamic Button labels,
request a layout refresh and assert that `region.width` can contain the visible
label; a correct reactive value does not prove that layout was recomputed.

---

## Test embedded panes at their allocated width, not the terminal width

**TASK-13205, 2026-08-11.** The Speech Lab clone-result geometry regression
mounted the pane in a 134-column Pilot viewport and proved every action was
inside the split. Live UAT still clipped **Save as Voice Profile**: the real
screen reserves a catalog rail, leaving the pane about 100 cells. At that width
the managed provenance wraps onto an extra row and the last action began one
row below the split. Re-running the same containment assertion at the pane's
actual allocated width reproduced the failure and justified a one-row minimum
height correction.

**What to do.** For a pane embedded beside a fixed or responsive rail, test the
pane at the width its parent actually allocates, including the wrapping-heavy
state. A terminal-size test can be truthful for a standalone harness and still
miss clipping caused by the production parent layout.

---

## Passing the suites a change touches is not passing the suites it can reach

**The trap.** You run the tests near your edit. The breakage is somewhere that merely
*depends* on it.

**What happened.**

- Deleting a module left a doc-enforced inventory (`Tests/RuntimePolicy`) listing it.
  Full-tree `--collect-only` passed: that only catches import errors, not stale
  assertions *about* the codebase.
- Adding a screen attribute in the wrong method broke the media viewer on **restored
  sessions**. `Tests/Library` (859 green), the state-level tests, and a live server run
  all passed and were blind to it.

**What to do.** Ask what your change can *reach*, not what it touches. For deletions,
that includes **inventories, audits and architecture-contract tests**, which assert
about the codebase rather than importing it. `--collect-only` over the full tree is
necessary and not sufficient.

---

## A re-export hides a dependency from a module-name grep

**What happened.** `test_plaintext_ingest_events.py` imported a deleted function *via*
`ingest_events`, so grepping the deleted module's name never matched it. Worse, a
per-symbol reachability scan excluded it because the filename
`test_plaintext_ingest_events.py` *contains* `ingest_events.py` as a substring — the one
file that disproved the conclusion was skipped as "internal".

**What to do.** Match paths **exactly**, never by substring. Run
`pytest --collect-only` over the whole tree before deleting anything; it is the only
check that sees through a re-export.

---

## Compare failure *sets* from identical commands, never counts

**What happened.** A 3-vs-4 failure count between two *different* invocations
(one file alone vs. that file plus another suite) read as a regression. It was not.
Later, the same file gave 6, then 4, then 0, then 1, then 0 failures on unchanged code.

**What to do.** Run the **identical command** on your branch and on a clean `origin/dev`
worktree, and diff the failure **sets**. Counts across differing commands are
meaningless. Machine load changes which tests lose a race — this repo regularly has
10+ concurrent pytest processes from parallel agents.

---

## A low-rate intermittent needs a loop, not a rerun

**What happened.** Five single-run attempts to capture a flaky test's traceback all
passed. Looping the file until one failed produced it on the first loop — and the
assertion identified the cause immediately (a test waiting on *state* then asserting on
*DOM*, before the recompose that renders it).

**What to do.** For anything below ~30% failure rate, loop with `-rf` and capture. A
single run is not a diagnostic. And prefer waiting for the **widget** over waiting for
the state that implies it.

---

## Reloading an IPC module can split exact type identity across spawn

**TASK-601, 2026-08-08.** A POSIX import-boundary test called
`importlib.reload()` on the module defining `WorkerContainmentIdentity`. The
already-imported executor retained the old dataclass object, while spawned workers
imported the reloaded class. Pickling still succeeded, but the parent's deliberate
exact-type bootstrap check rejected every worker identity; one test-side reload
therefore surfaced as 19 executor startup failures.

**What to do.** Test fresh-import boundaries in a subprocess. Do not reload a module
that owns IPC or serialized contract classes inside the shared pytest process; stale
importers keep the previous class identity even though the module name is unchanged.

---

## One thread must own reaping each spawned process

**TASK-601, 2026-08-08.** The local STT reader handled pipe EOF by calling
`Process.join()` before checking whether the controller had already detached that
generation. During graceful recycle, the controller also joined the same
`multiprocessing.Process`. On POSIX, the competing `waitpid()` calls occasionally left
the controller observing an unknown exit code and reporting a live worker even though
the child had exited. Removing the reader join entirely then exposed the other half of
the contract on macOS: an unreaped crashed group leader made `killpg()` return
`EPERM`.

**What to do.** Decide process ownership under the lifecycle lock before joining. A
reader may reap an unexpectedly exited generation only while it is still current; once
the controller detaches that generation, only the controller may reap it. Cover both
branches: a deterministic stale-reader test and a repeated real-spawn recycle test.

---

## A text scan for "is this method called?" passes vacuously

**What happened.** TASK-895 needed a guard proving every `WatchlistBundleService`
method has a production caller. The first version grepped the tree for `.create(`,
`.rename(`, `.delete(` and friends. It passed — against code where the methods were
still unwired.

`.create(` matches `completions.create(` in the OCR backends. `.rename(` matches
`os.rename(`. The guard was measuring the existence of unrelated method names on
unrelated objects. It was caught only by mutation: unwiring a call and watching the
guard stay green.

Rewritten as an AST walk that resolves the receiver before counting a call.

**What to do.** A guard that asserts "X is used" must resolve *what* X is, not match its
name. Bare-name greps are fine for finding candidates and useless as evidence, because
method names are not unique across a codebase — the more generic the verb (`create`,
`delete`, `run`, `send`), the more certainly the grep is counting something else.

And whatever the guard is: **mutate the thing it claims to protect and watch it fail.**
This one looked authoritative, ran green, and proved nothing.

---

## Catalog registration does not prove a TTS package is text-ready

**TASK-13202, 2026-08-10.** The pinned audio.cpp `release-0.5.1` server
successfully registered a standalone PocketTTS GGUF, and its catalog advertised
a TTS task. That looked sufficient to classify the package as ready for the
first no-reference sample. Real synthesis disproved the assumption: PocketTTS
requested a separate voice embedding (`alba.safetensors`) that the standalone
GGUF does not contain. Registration and task metadata were both true, but the
promised user journey was still impossible.

**What to do.** Before classifying an exact local-model recipe as text-ready,
run a model-specific complete-WAV request against the pinned real server and
confirm any required voice/reference material is present. Catalog registration
proves that the server accepted the model; it does not prove that text alone is
a complete synthesis input.

---

## Full component coverage, zero feature coverage

**TASK-1210, 2026-07-27.** Watchlists never checked on a schedule — not at the
wrong interval, never at all. Every component involved was tested and green:

- `test_watchlist_projection.py` — rows become `ScheduledTask`s with a `next_run_at`
- `test_watchlist_check_handler.py` — the handler checks feeds and records results
- `test_scheduler_loop.py` — the loop dispatches due tasks to their handler
- `test_config_flags.py` — asserted the flags' defaults, and **asserted the broken
  values**, pinning the bug in place

Each component was correct. Nothing tested them *joined*, and the join was where
the feature lived: `app.py` only registered the `watchlist_job` handler when a
flag was true, and that flag shipped false, so the loop logged "no handler
registered for task type" and moved on. Silently, forever.

The config test is the sharpest part. It was not absent — it was present,
passing, and encoding the defect as the expected value. A test that asserts
current behaviour without asking whether that behaviour is *right* converts a
bug into a requirement.

**What to do.** For any feature that spans components, own one test that drives
the real path end to end — here, a real `Subscriptions_DB` row through the real
projection, the real queue and a real `SchedulerLoop`, asserting the result lands
back in the database. Component tests tell you the pieces work; only that test
tells you the feature does.

And when a test asserts a configuration default, make it state *why* that value
is right, not just what it is. `assert enabled is False` is unfalsifiable
documentation of whatever was there when it was written.

## A green suite says nothing about installs that are not yours

**The trap.** The suite runs where every optional extra is already installed. It
therefore cannot see a dependency that is *declared* optional but has become
*mandatory to boot* — the one environment it never tests is the plain install.

**What happened.** 2026-07-27: the app died on start with
`RuntimeError: Unable to resolve default chat screen`. `aiohttp` is optional —
at the time declared only in the `[websearch]`/`[all-tools]` extras (task-1262
has since given image generation its own `[image_generation]` extra), and
registered `"aiohttp": False` in `Utils/optional_deps.py` — but the
`/generate-image` console feature had quietly wired it onto the **default**
screen's import chain:

```
UI/Screens/chat_screen.py
  -> Chat/console_generate_image.py        (ImageGenerationService)
    -> Media_Creation/image_generation_service.py
      -> Media_Creation/swarmui_client.py  -> import aiohttp   (module scope)
```

Nothing was red. No test asserted that the default route resolves *without* the
extras, so the suite was structurally blind to a total boot failure.

Two multipliers made it worse:

- **The masking cost more time than the bug.** `ScreenRoute.load_screen_class()`
  catches `ImportError` and returns `None`, by design, so one broken optional
  screen cannot break navigation. For the *default* screen that turned a precise
  `ModuleNotFoundError: No module named 'aiohttp'` into a message naming neither
  the module nor the file that imported it.
- **The obvious suspect was innocent.** The only dirty file in the tree was a
  `.tcss` whose diff was a regenerated timestamp comment. Reproducing first and
  reading the traceback cost one command; guessing from `git status` would have
  cost the session.

**What to do.** When a feature adds an import to a screen module, check whether
the new chain reaches an optional dependency — the import that breaks boot is
rarely the one you wrote, it is three hops down. Guard boot-critical routes with
a test that simulates absence, and run it in a **subprocess**: `sys.modules` is
process-global, so an unrelated earlier test that imported the package gives a
false pass.

```python
class _BlockAiohttp:                       # meta-path finder, installed first
    def find_spec(self, name, path=None, target=None):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ImportError("simulated missing aiohttp")
        return None
```

See `Tests/Utils/test_optional_import_deferral.py` (the aiohttp section) and
`Tests/UI/test_screen_navigation.py` (`screen_load_error`). And when a resolver
degrades a failure to `None` on purpose, give callers for whom it is *fatal* a
way to ask why — a graceful contract should not also be a silent one.

---

## Measure a dead-code graph from both ends

**TASK-1211, 2026-07-28.** The audit that scoped this retirement measured the island
by walking *outward* from `BriefingGenerator` — who imports it, who imports them —
and arrived at ~5,100 LOC across 11 files.

Walking the other direction, *down* from the scheduler that was about to be
deleted, found the chain kept going:

```
textual_scheduler_worker  →  sole importer of Event_Handlers/subscription_events.py
                          →  sole importer of subscription_ingest_worker.py
                          →  sole caller of Subscriptions/content_processor.py
```

The real island was 8,148 lines across 13 files, plus a fourth module left
deliberately in place. Deleting only what the outward walk found would have
orphaned two files silently — the exact state that made this island expensive to
diagnose in the first place: dead, but with importers a grep can point at.

**Why one direction is not enough.** The outward walk answers "what does this dead
thing depend on?" The downward walk answers "what depended on it and is about to
become dead?" A retirement needs both: the first bounds what you may delete, the
second bounds what your deletion *creates*.

**What to do.** Before deleting a module, list its importers *and* list what it
uniquely imports. Anything it is the sole importer of joins the removal set, and
you recurse. Then re-run the runtime import trace afterwards — if a module you
kept is still in `sys.modules` with no caller, you have made a new orphan and
should either wire it, delete it, or file it. Filing is acceptable; silence is
not.

Corroboration is worth seeking: TASK-813's notes had already reached the same
conclusion about `subscription_events` from the other direction months earlier
(`handle_add_subscription` has zero dispatchers). A prior investigation's notes
are cheaper than re-deriving the graph.

## A missing extra fakes a code regression — check the env before blaming the code

**The trap.** The mirror image of the entry above. There, everything was installed
and the suite went blind. Here, an optional extra is *absent* and a test fails with a
message describing a defect that does not exist. The failure text names production
behaviour, so it reads as a regression, and you go fix code that was never broken.

**What happened.** 2026-07-28, task-1261. `test_nltk_download_false_is_not_logged_as_success`
was failing on dev with "no WARNING/ERROR mentioning punkt was logged". That is
precisely what a deleted `logger.warning` looks like. It was filed as one — *"the
warning was lost in a refactor"* — with `git log -L` even producing a plausible
culprit commit that had genuinely rewritten that function, and an orphaned
over-indented comment left behind as the apparent fingerprint.

All of it was wrong. `nltk` is an optional extra, and it was not installed. The test
sets `NLTK_AVAILABLE = True` to simulate presence, but `_ensure_nltk()` still runs a
real `import nltk`, so it returned early and never reached the warning. Installing
`nltk` turned the test green with no code change at all. The confirming probe written
to "verify" the diagnosis had been run in the same interpreter, so it hit the same
early return and agreed — a second wrong answer from the same cause reads as
corroboration.

**What to do.** When a test asserts that a log/branch/side effect is missing, check
whether the code path can even be *reached* in your environment before concluding the
behaviour was removed. One command settles it:

```bash
python -c "import importlib.util as u; print(u.find_spec('nltk') is not None)"
```

And a test that forces an availability flag must also stub the import that flag
stands for — otherwise it silently depends on which extras you installed:

```python
monkeypatch.setattr(Chunk_Lib, "NLTK_AVAILABLE", True)   # not sufficient alone
monkeypatch.setitem(sys.modules, "nltk", fake_nltk)      # _ensure_nltk() still imports
```

Corollary: a probe re-run in the same broken environment is not independent evidence.
Vary the thing you suspect — here, install the package — and see if the symptom moves.

---

## A property test with no deadline override is load-sensitive

**TASK-1260, 2026-07-28.** `test_safe_paths_always_validate` failed once inside a
three-directory run, passed alone, passed on re-run, and passed on a clean
baseline with the identical command. It is a Hypothesis `@given` property that
creates a `TemporaryDirectory` and up to four directories per example (the
strategy yields 1-5 components; the loop walks `components[:-1]`, the last being
the file), with no `settings(...)` override and no Hypothesis profile in
`Tests/conftest.py` — so it runs under the default **200 ms per-example
deadline**. On a machine with 10+ concurrent pytest processes, filesystem work
crosses that.

**The cost is in attribution, not in the failure.** Establishing that it was not
a regression took five runs across two worktrees: the identical command on a
clean pre-change baseline, `Tests/Utils/` with and without a newly added file in
that same directory, the test alone, and a re-run to show intermittency. The
failure was indistinguishable from a real regression at the moment it appeared,
and it appeared while unrelated work was in flight.

**What to do.** When a failure appears in a run that mixes suites: before
anything else, check whether the test is a Hypothesis property with no deadline
override. If it is, the load hypothesis is cheap to confirm — run it alone, then
re-run the mixed command. Do not skip the clean-baseline run, though: "it's
probably a flake" is the same shape of reasoning as "it's probably unrelated",
and this repo has punished both.

The durable fix is a Hypothesis profile registered once in `Tests/conftest.py`,
not a per-file patch — other property files carry the same exposure.

---

## A filter with no admitted callers is an off switch

**TASK-1240, 2026-07-28.** The persistent app log wrote zero bytes. The handler
was attached, the path resolved, the directory existed, the level was INFO.

`PersistentDiagnosticFilter` admits a record only if it carries a marker set
exclusively by `log_persistent_metadata()`. That function has **zero production
call sites**. Every ordinary `logger.info(...)` is rejected, so the sink is
correctly enforcing a boundary that nothing was ever migrated to cross.

The privacy work that introduced it is sound and has its own ADR. The gap is
between decision and implementation: ADR-029 requires logs be metadata-only
*with respect to user and model content*, and the design's stated goal was to
"keep persistent diagnostics **useful** without retaining private payload
values". Admitting nothing satisfies the letter of the exclusion list and defeats
the goal.

**What to do.** When a sink produces nothing, check the **admission predicate
before the plumbing**. Handler attached, path correct and level correct are all
consistent with a filter that rejects everything. Then ask the question that
distinguishes the two failures: *how many callers does the admitted path have?*
Zero is the tell.

And when the answer implicates a deliberate security boundary, record the gap
and hand the decision to that work's owner. Loosening a privacy filter to make
your own diagnostics visible is not a fix you get to make alone.

This is the fourth instance of one shape in a single session: a closed import
cycle, a flag gating the only executor, a prompt surface with no consumer, and
now a log sink with no admitted caller. Each was built, wired, and given nothing
to carry — and each read as live to a grep.

---

## Run source-inspection tests on a supported interpreter before changing them

**TASK-15706, 2026-08-13.** A repository-wide collection under Python 3.14
failed in `test_profile_store_lock.py` because the test compared integer source
lines with `None`. The production code and test were unchanged from `dev`;
inspection showed Python 3.14 emitted 12 `dis.findlinestarts()` entries whose
line is `None`. The same repository collected all 42,613 tests under the
installed, project-supported Python 3.12 interpreter.

**What to do.** When a test derives source locations from bytecode or
introspection APIs, check the interpreter version and inspect the raw API output
before patching application code. Re-run collection under a supported project
interpreter to distinguish an interpreter-assumption failure from a product
regression, and record both results.

---

## A suite that no gate runs can rot invisibly for days

**TASK-1310, 2026-07-28.** `Tests/UI/test_settings_configuration_hub.py` carried 22
failures at dev tip — byte-identical at base and branch, so none were caused by the
branch that finally surfaced them (TASK-1234's review, the first time this hub ran in
that review cycle). The suite was last known green before #1050; nothing narrower
caught the drift for days across two deliberate, well-reasoned refactors
(`d15882398` "own provider selection by lifetime" and `1df0c4cb4` "reconcile privacy
lifecycle eval and packaging hardening") that each correctly updated every production
call site and left this one 253-test file behind.

Both refactors were textbook-good: each shipped its own new, correct test coverage
(`Tests/Provider/test_provider_model_resolution.py`; the batched-save-adapter test in
this same file) and left zero live bugs — `grep` across all of `tldw_chatbook/` for the
removed symbols (`chat_api_provider_value`, `save_setting_to_cli_config` imported into
`settings_screen`) came back empty on both counts. The damage was entirely to *this*
suite's ability to say so: 22 tests calling a removed signature/attribute is worse than
0 tests, because a red suite nobody gates reports exactly as much confidence as a suite
that does not exist, while still costing the CI minutes of everyone who happens to run
it directly.

**What to do.** A suite this large (253 tests, a whole product surface) needs a home in
routine verification, not opportunistic discovery via someone else's PR review. The
Settings/Console-area verification gate must include
`Tests/UI/test_settings_configuration_hub.py` going forward — not because it is
special, but because "carries the hub's tests" is exactly the kind of suite that rots
silently when nothing runs it: too large to eyeball, too domain-specific for a generic
CI matrix to catch by accident, and it will not fail loudly for anyone except the next
person who happens to touch that screen.

---

## A hung test under timeout_method="thread" kills the whole run — and a hang can be an optional-dep condition

**TASK-1466, 2026-07-30.** The full-suite baseline run for the test-suite audit died
at ~3% progress, twice. `test_pyaudio_recording_flow` stops its recording loop from
inside the chunk callback, but the callback is gated behind webrtcvad speech
detection and the test's synthetic buffer is silence: with `webrtcvad` installed
(it is, locally and in CI), the callback never fires and the loop never exits.
After the 300s timeout, pytest-timeout's `thread` method — the repo's configured
method, and the only one that works for threaded/async tests — cannot cancel the
test, so it dumps stacks and **terminates the entire pytest process**. One hung
test cost the whole run, every run, on every webrtcvad machine; on machines
without the extra the callback fires per chunk and the test is green, which is why
it survived review. Its sibling `test_sounddevice_recording_flow` failed on clean
dev for the same root cause (a 4-sample chunk is smaller than one 20ms VAD frame,
so the VAD loop body never executes) — masked because serial runs died at the hang
before reaching it.

**What to do.** A test that stops its own loop from inside a callback must also
bound the loop at the *source* (here: the mocked `stream.read` flips the stop flag
after N reads) so no gating change can make it unbounded. When a test's behavior
depends on an optional dependency, run it in both installed and absent
configurations before trusting green. And treat "the run died at N%" as possibly
ONE test: the timeout stack dump names it — **except when the hang is an awaited
asyncio primitive; see the next entry.**

---

## The timeout stack dump does NOT name the hung test when the hang is an awaited Event

**TASK-3316 / TASK-14912, 2026-08-10.**
`test_file_notes_collections_source_transition_blocks_mutation_through_recompose`
drove `_select_library_rail_row` as a fire-and-forget `asyncio.create_task` and
then `await`ed an `asyncio.Event` only that coroutine could set. Its stub returned
`None` — correct when written (`eb036a6a1`) — until PR #1439 retyped
`_flush_library_note_save` to return `NoteFlushOutcome`; the caller then died one
line in on `AttributeError: 'NoneType' object has no attribute 'kind'`. **Nobody
retrieves a `create_task` result, so the exception was swallowed**, the signal
became unreachable, and `await event.wait()` blocked forever.

Two things this cost, beyond the one test:

1. **The stack dump was useless.** TASK-1466's advice above does not hold here: a
   coroutine suspended at an `await` has no frames on any thread stack, so
   pytest-timeout printed only `MainThread` idle in `selectors.select` and never
   named the test. Diagnosis required inspecting the *task object*
   (`task.done()` / `task.exception()`), not the dump. Reproduced deliberately
   while writing the bound: 25s timeout, stacks dumped, process terminated,
   **zero** tests reported in the summary line.
2. **Every test after it in the file silently never ran**, so the file's pass
   count was a lie for as long as the hang existed. Repairing that one test
   revealed three further failures the hang had been hiding.

**What to do.** Never `await <event>.wait()` on a signal only background work can
set. Route it through `Tests/UI/background_signals.py`:
`wait_for_background_signal(event, task, what=...)` when the test owns the task
(it re-raises what the task swallowed), or `wait_for_signal(event, what=...)` when
the product owns the work (timeout-only, but a named failure in seconds instead of
a dead process). `Tests/UI/test_background_signal_bounds.py` enforces this by AST
over the whole directory — grep cannot, because it cannot tell an unbounded
`await ev.wait()` from one already inside `asyncio.wait_for`, nor an
`asyncio.Event` from a Textual `Worker`/retained-operation handle whose `.wait()`
re-raises and therefore cannot strand.

Two corollaries worth carrying:

* **A file that has ever contained a hang has an UNKNOWN pass count** until it is
  re-run whole. A previously recorded count for such a file is not evidence.
* **In practice the re-raise branch fires less often than you would expect.** Of
  four sites broken deliberately with the stale-stub shape, only one propagated;
  the other three product paths caught the `AttributeError` internally, logged it,
  and returned early — so the bound reported "finished without signalling" rather
  than the exception. That is still a 1-3s named failure instead of a dead run,
  but it means the helper's early-return branch is not a corner case: do not drop
  it in favour of "just re-raise".

---

## `--deselect` with a wrong nodeid is silently ignored

**2026-07-30, same audit.** A full baseline attempt was relaunched with
`--deselect "Tests/Audio/test_recording_service.py::test_pyaudio_recording_flow"` —
missing the `TestAudioRecordingIntegration::` class qualifier. pytest does not
error, does not warn, and does not report `1 deselected`; the run simply hung on
the very test the flag was meant to exclude, and the mistake was only visible ~15
minutes later when progress stalled at the same file.

**What to do.** After launching a run with `--deselect`, confirm the header line
says `N deselected` before walking away. Copy nodeids from pytest's own output
(`--collect-only -q` or a failure line), never reconstruct them by hand — class
nesting is invisible in the source-file mental model.

---

## Removing a per-test gc.collect() can unmask cross-test coupling — with a rotating victim

**TASK-1468, 2026-07-30.** task-1454 replaced the double `gc.collect()` after every
test with a periodic collect (every 25). A 10-test batch then started failing ONE
UI test — a *different* one on consecutive identical runs (first the Skills trust
panel test, then a Library git-notes test). Alone, each test passed. With
`TLDW_TEST_GC_EVERY=1` the batch passed 10/10; on pre-change dev it passed 10/10.

The mechanism: a Textual `App` is a reference cycle that refcounting never frees —
only the cycle collector does. Per-test collection had been silently guaranteeing
each app-mounting test a garbage-free predecessor; without it, the previous app's
remains (timers, context vars, screen state) linger into the next app's lifetime,
and which test breaks depends on heap state at the time. A rotating victim is the
tell that you are looking at ambient-state interference, not a broken test.

**What to do.** When narrowing global per-test cleanup, ask what CLASS of object it
was silently reclaiming, and scope the cleanup to the tests that produce that class
(here: per-test collection in app-mounting dirs, periodic elsewhere) rather than
tuning frequency — no interval above 1 protects adjacent producers. And triage any
"deterministic" batch failure by rerunning the identical batch twice before
believing determinism: the rotating victim only shows on the second run.

---

## Measure the identical-run noise floor before reading an A/B outcome diff

**TASK-1459, 2026-07-30.** The CSS parse-cache canary (full Tests/UI, cache on
vs off) showed 12 pass->fail flips and zero recoveries — directional, exactly
what cross-instance cache corruption would look like, and the spike's gate said
any diff means fall back or no-go. Before ruling, a control pair of two
IDENTICAL cache-off runs was diffed: **28 regressed + 4 recovered against each
other** — the machine's own flip rate was ~3x the "cache effect", the flagged
tests A/B'd identically in isolation with the cache on and off, and the
cache-on run sat between the two cache-off runs on failures and wall time
(which grew from 13:37 to 23:12 across the evening as concurrent sessions
loaded the machine).

**What to do.** An "outcome diff must be empty" gate is unfalsifiable on a
machine whose identical-run diff is nonzero — and on shared dev machines it
usually is. Before attributing an A/B diff to the change, run the A/A control
and use *its* magnitude as the acceptance bound; attribute individual flips
only via isolation reruns under both configurations. Directionality alone
(12-vs-0) is not attribution: later runs on a loading machine flip
asymmetrically toward failure.

---

## A crash mid-transfer proves nothing durable if the checkpoint is never written mid-transfer

**TASK-595 Task 10, 2026-07-31.** Asked to write a real-subprocess, SIGKILL-based test
proving "a valid sidecar survives a mid-fetch crash and a fresh provision resumes it via
a Range request," the obvious design was: pause a fixture connection mid-download, kill
the child while the socket is blocked, then assert the durable checkpoint resumed. That
design cannot pass — not because of a bug, but because of how the code under test is
*supposed* to work: `_fetch_one_file` only calls `atomic_write_json` on the sidecar AFTER
`stream_fetch` returns successfully for that call; a SIGKILL during the first-ever
transfer of a file leaves no sidecar at all (confirmed directly by an existing test,
`test_provision_cancel_mid_fetch_releases_lease_and_preserves_prior_active`, which
asserts `not sidecar_path.exists()` after an asyncio-cancelled mid-fetch). A kill timed
mid-socket-read can only ever produce an *orphan* (Task 2's GC correctly deletes it), not
a resumable checkpoint — the two are mutually exclusive outcomes of the same kill point.

A durable-but-partial checkpoint (`bytes_done < size, complete: false`) only exists when
one `stream_fetch` call already returned successfully with fewer bytes than the
descriptor declares — which requires either a pre-seeded sidecar (what the existing
in-process tests do, legitimately, to isolate the resume *logic*) or, for a test that
must produce this state via a genuine, un-seeded crash: declare a file's size larger than
what the fixture route currently serves, let that first GET complete normally (a valid,
un-truncated HTTP response, so `stream_fetch` returns without error), and freeze the
*next* phase (pre-verify, via a local `threading.Event` a progress-callback hook blocks
on) before its hash comparison can clear the sidecar entry on mismatch. The kill then
lands after a real checkpoint is durable, not during the socket read that would prevent
one existing at all.

**What to do.** Before designing a crash-recovery test for any resumable-transfer
system, read the exact write-ordering of its checkpoint (grep for where the persisted
state is actually written, not just where progress callbacks fire) and check for an
existing test that already documents the "no sidecar" case — it will save you from
building a scenario the implementation cannot produce. Then time the kill off a
deterministic signal (a callback blocking on a never-set local event) rather than a
sleep or a byte-count race, so the exact crash point is provable, not probabilistic.
Mutation-test the resulting guard afterward: a one-line break in the resume path
(`resume_from = 0` unconditionally) and in the orphan classifier (`return False`
unconditionally) each turned the corresponding assertion red, confirming neither guard
was decorative.

---

## Related

- `lessons-live-verification.md` — why the suite could not see seven of these defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps

---

## The shared UI harness never loads the app stylesheet — geometry conclusions under it are void (2026-07-30)

**Incident.** The V2 live gate failed its composer-overflow item AFTER the defect had
been "fixed" twice, each fix RED-first, mutation-checked, 500k-trial fuzzed, and
approved through two full review rounds. The real cause was one CSS rule —
`#console-composer-expanded { height: 1 }` — cropping the grown 4-row draft to a single
painted line. No test could see it: `Tests/UI` harnesses build `ConsoleHarness`, a bare
`App[None]` that pushes `ChatScreen(app_instance)` directly. The `TldwCli` instance is
only a service container there; the App that runs owns the stylesheet, and it has none.
Every rule in `tldw_cli_modular.tcss` silently does not apply under these harnesses.
Both instruments used to verify the fixes — `widget.render_line(...)` (the widget's own
paint, blind to a parent's crop) and `widget.region` (layout placement, not clipped
paint) — were also individually unable to see cropping. A 30-second tmux run of the
real app reproduced the user's report on the first try.

**What to do.** Any assertion about on-screen geometry — heights, clipping, whether a
row is visible — must run under a harness whose `CSS_PATH` is the real bundle (see
`_CssTrueConsoleHarness` in `Tests/UI/test_console_composer_overflow.py`), or against
the real app in tmux. `render_line`/`region` alone prove what a widget WOULD paint,
never what the screen shows; the composited screen is the only authority (third
recorded instance of this lesson class). When a live report contradicts a green suite,
suspect the harness before the reporter.

**Fourth instance (2026-08-07, task-2859 item 10, padding not clipping this time).** A
`.library-rag-result-snippet { padding: 0 1; }` bundle rule (fixing a snippet sitting
flush against its card border) tested green with `snippet.region.x ==
title_row.region.x` under `DestinationHarness` (`Tests/UI/test_library_content_hub.py`)
— because `region` never reflects padding at all (only layout position/size;
`content_region = region.shrink(styles.gutter)` is the one that does), AND because
`DestinationHarness` is a bare `App` with no `CSS_PATH`, so `title_row.styles.padding`
itself came back `Spacing(0,0,0,0)` regardless of what the .tcss said. Direct proof:
`screen.app.css_path == []` and `type(screen.app).CSS_PATH is None` under this harness,
vs. the real string when `TldwCli` is imported and inspected directly outside any test.
Moving the exact same assertion to `LibraryHarness` (`Tests/UI/test_library_shell.py`,
which sets `CSS_PATH` to the real bundle — "Mount a single LibraryScreen with the real
app stylesheet" is literally its docstring) reproduced the missing-padding RED
correctly and went GREEN once the CSS rule existed. Two independent traps stacked here,
either one alone would have hidden the bug: use `content_region`, not `region`, for
padding; and know which harness in a file actually loads CSS before trusting geometry
from it — `Tests/UI/test_library_content_hub.py` uses `DestinationHarness` (no CSS) for
most of its tests, `Tests/UI/test_library_shell.py` uses `LibraryHarness` (real CSS) —
same directory, same screen under test, opposite answer to "does this rule apply".

**Fifth instance (2026-08-08, task-3200 review round 1, cascade PRIORITY not just
missing rules this time).** `MainNavigationBar.DEFAULT_CSS` ghosts a straddling nav
button by setting `color`/`background`/`opacity`/`text-opacity` all to `$background`
`!important`, intending a pixel-exact invisible button once it's also `disabled`. The
bare-`App`-no-`CSS_PATH` test (`test_nav_strip_never_renders_a_partial_destination_
label`) confirmed exact-match compositor colors and stayed green — but live tmux
showed the ghosted fragment as a faintly-but-genuinely readable `rgb(43-62,43-62,
43-62)` against `rgb(16,16,16)`, not a match. Root cause was NOT a missing bundle
rule (the earlier four instances) but a PRIORITY one: `tldw_chatbook/css/components/
_buttons.tcss`'s app-wide `Button:disabled { opacity: 50%; }`, loaded via `App.
CSS_PATH`, outranks ANY widget `DEFAULT_CSS` rule as a TIER, regardless of
`!important` on the `DEFAULT_CSS` side — confirmed by direct introspection
(`button.styles.opacity` read `0.5` under the real `TldwCli` app + `HomeHarness`
despite the widget's own `opacity: 100% !important`, but read `1.0` under the bare
test harness where no `CSS_PATH` rule existed to compete). This codebase had already
hit and fixed the identical defect once before (`Tests/UI/test_mcp_inspector.py`'s
`test_disabled_action_buttons_stay_legible_with_bundled_css`, for the MCP inspector's
action buttons) — that precedent was not consulted before initially trying
`!important` in `DEFAULT_CSS`, which was the wrong tier to fight from. The fix that
actually works: add the override to a `CSS_PATH`-bundled source file (`_navigation.
tcss` here), in the SAME tier as the rule being overridden, where ordinary
specificity resolves it without needing `!important` at all.

**What to do (all five instances).** Never trust a bare-`App`-no-`CSS_PATH` test's
color/opacity or geometry as proof of live behavior — it can miss a rule entirely
(instances 1-4) or miss a PRIORITY inversion where `CSS_PATH` beats `DEFAULT_CSS`
regardless of `!important` (instance 5). Before shipping any "hide via CSS" trick
(`color == background`, opacity-to-zero, etc.), grep for prior art
(`Button:disabled`, `:disabled` opacity overrides already exist for MCP inspector)
and verify with `button.styles.opacity`/`get_visual_style()` under a REAL-bundle
harness or live tmux, not just a bare widget construction.

---

## A zero-latency fake makes loop-starvation bugs invisible (2026-07-30)

**Incident.** Live dictation never emitted a final during capture — voice commands
(classified on finals) were completely dead in the field — while 300+ dictation tests
stayed green. `_processing_loop` transcribed each 0.5 s audio window synchronously on
the same thread that runs the silence-finalize check; real transcription took 4-5 s per
window (proven with `sys._current_frames()` stack dumps against a live microphone), so
the check starved indefinitely. Every test fake transcribed in ~0 ms, which makes the
serial design behave identically to a concurrent one. The probe ladder that isolated it,
in increasing depth: call the transcriber+classifier directly (chain worked) → measure
`captured_bytes` during silence (VAD worked) → shim the finalize method (never called
despite an 8.6 s silence age at a 2.0 s threshold) → per-tick thread stack dumps (loop
permanently inside the transcriber). Fix: segment-at-silence architecture; the RED test
that pins it gives the fake a CONTROLLABLE latency (2× the threshold) — with a fast fake
it cannot fail.

**What to do.** Any fake standing in for an operation whose real latency exceeds the
loop/timer cadence it shares a thread with must be able to sleep. Test both fast and
slow. And when a threaded pipeline works in tests but not live, dump the worker's stack
(`sys._current_frames()[thread.ident]`) once a second before theorizing — it answered in
one run what three cheaper probes could only narrow. Bonus rig: macOS `say` through the
speakers + the real microphone is a full live STT test harness needing no human.

---

## In a render-from-state UI, the in-place updater must own EVERY conditional (2026-08-04)

**What happened.** Four separate times in the Library ingest arc (tasks 2100, 2130,
2140, 2230), a canvas element was rendered by a `compose()`-time conditional while the
hot paths deliberately skip recompose (job ticks and text-input edits must preserve
focus, cursor position, and scroll). Each time the element was correct on first render
and wrong forever after:

- "Recent ingests" expanded into an empty unlabeled shell after a clear.
- The commit-summary line rendered for PDF selections and **never** for plain text —
  a PDF adds an options panel, which forces the structural recompose that happens to
  mount the line; a text-only pre-flight applies through the non-structural path,
  which mounts nothing. It also went stale after Clear ("0 will import · 1 will match"
  above an empty field).
- The invalid-option field marker was applied at compose time only, so it never
  toggled on the edit path it existed to serve — the field stayed marked after
  becoming valid and never got marked after becoming invalid, while the gate line
  instructed the user to "fix the highlighted options".

**Why tests kept missing it.** The harness passes when it *re-queries* after the
update, because a fresh query returns whatever was composed most recently. The failure
only appears when you assert that the widget you held is the widget still mounted.

**What to do.** Two rules, both cheap:

1. Anything the in-place updater does not explicitly own must be **always mounted and
   `display`-managed**, never conditionally composed. If a canvas-level element can
   appear and disappear, the updater sets its content *and* its visibility.
2. Pin it with **object identity**, not a re-query:

```python
before = screen.query_one("#library-ingest-start", Button)
...trigger the hot path...
assert screen.query_one("#library-ingest-start", Button) is before   # no recompose
```

A re-query test agrees with the bug; `is` does not.

---

## A new distinction must be learned by every surface that aggregates the old one (2026-08-04)

**What happened.** TASK-2231 introduced "matched" as an outcome distinct from "done"
(a dedup match is not a fresh import). The row got its own glyph and word, and the
change looked complete. Review found two surfaces still folding matched into done:
the per-batch group header, and the top-level queue tally — which produced two
contradictory summaries on the same screen ("2 done" directly above "1 done ·
1 matched"). The completion toast was a third surface, caught only because it was
grepped for by hand. TASK-2220 had the same shape one PR earlier: adding a `SKIPPED`
job state updated the row and the tally but missed `queue_show_clear_finished`, which
still tested `state in (DONE, FAILED)` — so a queue holding only skipped rows could
not be cleared at all.

**What to do.** When you add a state, an outcome, or any new bucket, grep for **every
predicate that enumerates the old set** and every surface that counts it, then list
them before writing code. In this feature that list was: the row builder, the group
header, the queue tally, the completion toast, the "show clear" gate, the "finished"
count, the ledger snapshot, and the durable-history filter — eight places, and the
first attempt updated three.

Related: a fixture that omits the interesting axis hides the bug. My own attempt-marker
test used jobs with no `detected_type`, so it passed while every *typed* row rendered
the marker in the wrong position.

---

## An error fallback that returns a valid-looking value becomes a confident lie (2026-08-04)

**What happened.** `_safe_size(path)` returned `0` on `OSError` — a reasonable "I don't
know" for summing sizes. TASK-2160 then added an empty-file classifier that read
`size == 0` as "this file is empty and will fail", and surfaced it as user-facing
copy: *"1 empty file will fail — notes.txt is 0 B."* For an unreadable or unstatable
file, that sentence is a measurement nobody took, and the file was pulled out of its
type group on the strength of it.

**What to do.** A sentinel is safe for aggregation and unsafe for classification. When
a new consumer needs to *distinguish* failure from a real value, give it a probe that
says so — `_statted_size()` returning `None` on `OSError` — and leave the summing
caller on the old fallback. Before reusing any helper whose docstring says "or `0` on
error", ask what your caller will conclude from that zero.

---

## "the field is set" is not "the resource is ready" — check what the real object does before it's ready (2026-08-04/05)

**What happened.** TASK-2360's bug report said reconnect audio was dropped because
`session.session is None` during the reconnect window. Reading the wiring showed
`session.session` is actually reassigned to the NEW provider session quite early
(`_connect_console_realtime`, before `await provider_session.connect()` even runs) —
so a fix gated only on "is `session.session` set" would have looked correct and
still leaked frames into a session with no live transport yet. The REAL drop
mechanism was one layer deeper: `OpenAIRealtimeSession._enqueue` silently discards
anything sent before `connect()` populates its outbound queue. A test double
(`FakeRealtimeSession.append_audio`) that just appends to a list, with no
before-connect gate, would have made a wiring test pass for the wrong reason —
proving frames "reached the session" when a real session would have swallowed them.

**What to do.** When a bug report names a field as the cause ("X is None during the
bad window"), verify what that field actually holds moment-to-moment, not just
whether it is set — a reassigned-but-not-yet-live reference passes an `is not None`
check while still being unsafe to use. And before trusting a wiring test built on a
fake, ask whether the fake reproduces the real object's OWN pre-ready guard, or is
simply more permissive than production in exactly the window under test.

---

## A test failure dismissed as "pre-existing noise" can be a shipped crash

**TASK-2610, 2026-08-06.** `test_production_settings_actions_cross_the_pushed_screen_boundary`
failed with `DuplicateIds: lab-speech-row-playground` for weeks. Across many tasks — in
multiple programs, by multiple sessions — it was checked once ("fails identically on the
base commit"), labeled "pre-existing, unrelated," and waved through. Every one of those
dismissals was locally correct and collectively wrong: the failure was a 100%-reproducible
user-facing crash — navigating to Lab ▸ Speech took the whole app down, making the Speech
Lab (playground, voice profiles, audiobooks, voice cloning) unreachable. It was found only
when a live-verification pass tried to drive that navigation for an unrelated feature.

**Mechanism worth knowing on its own:** Textual's `MessagePump._get_dispatch_methods`
walks the MRO and invokes EVERY class's `on_mount` for a single Mount event. A subclass
handler that calls `super().on_mount()` therefore runs the parent handler TWICE.
`STTSScreen.on_mount` did exactly that over `LabFrameScreen.on_mount` — which mounts the
rail rows — so the second run collided on the row ids. Sibling screens without their own
`on_mount` never crashed, which is why the bug looked screen-specific. If a parent
`on_mount` does real work, `super().on_mount()` in a child is a crash, not a courtesy.

**What to do.** "Fails identically on base" proves you didn't cause it — it does not prove
it's noise. Before re-dismissing a persistently failing test, spend the five minutes to
classify WHAT the failure would mean for a user if the tested path were driven live; a
`DuplicateIds`/exception-shaped failure in a screen-mount test is an app crash until shown
otherwise. Budget one live drive of the affected surface the FIRST time a failure gets the
"pre-existing" label — that is when it is cheapest, and every later dismissal inherits the
first one's diligence or its negligence.

---

## A green result is not evidence until you have confirmed it could have gone red (2026-08-06)

**What happened, twice, on the same feature (PR-T3 fix rounds A and B).**

**Instance 1.** Fix round A added three tests for the Advanced runner's confirm-arm
behavior, each simulating "press Run, then press it again." Two of the three showed a
false PASS on first write — not because the code was right, but because Textual's
`Button._on_click()` ignores a click while the widget still carries the 0.2s
`-active` press-animation class (`textual/widgets/_button.py`). A bare second
`pilot.click()` inside one pump window landed on a still-cooling-down button and was
silently dropped, so the second half of the test never ran at all — the assertion
checked state that the first press had already produced, and the bug under test (does
the SECOND press do the right thing) was invisible. Caught only by comparing against
`_press_run_again` (`Tests/UI/test_mcp_inspector.py`), a helper an earlier test in the
same file already built for exactly this trap (`await pilot.pause(0.3)` before the
second click, with a comment naming the cooldown).

**Instance 2.** Reviewing fix round A, a reviewer who had JUST been told about
Instance 1 built a mutation harness specifically to check whether those three tests
actually pin their fix — reverting the fix and confirming the tests go red. The first
run reported all-green: reverting the fix changed nothing, which would itself have
been a serious finding (the tests pin nothing). The cause was a second, unrelated
mechanism: Python served stale `__pycache__` bytecode instead of the mutated source,
so the mutation never took effect and the "tests" exercised the OLD, already-fixed
code both times. Fixed by overriding `get_code` (forcing recompilation) rather than
trusting the file on disk had been re-read.

**The shared shape.** Two unrelated mechanisms — a UI framework's click-debounce, and
Python's bytecode cache — each silently prevented the code under test from being
exercised at all, while the harness reported success. Neither is specific to this
feature; both recur anywhere a test fires two rapid interactions through real Textual
widgets, or anywhere a mutation/characterization check edits a `.py` file and reruns
pytest without clearing `__pycache__`. And Instance 2 happened to someone who was
*specifically hunting* for exactly this class of false pass, one incident report old —
knowing the trap exists did not protect against a different instance of it. The
mitigation has to be mechanical, not a mental note.

**What to do.**
1. When a test simulates two rapid interactions through a real widget (double-click,
   press-then-confirm, retry), use a helper that waits out any framework-level
   debounce/cooldown before the second interaction, and name the cooldown in the
   helper's docstring so the next person does not have to rediscover it.
2. Before trusting ANY mutation/characterization test result — your own or a
   reviewer's — clear `__pycache__` for the touched modules (or run with `python -B`,
   or override `get_code`) and confirm the specific new/changed assertions go RED
   against the reverted code, not just that "some tests failed somewhere." A run count
   or an exit code is not enough; read which tests failed and why.
3. Treat "the mutation test passed on the first try" as itself slightly suspicious —
   it is the same shape as a guard that cannot fail (see "Mutation-test every guard
   you add," above), and here the false-positive mechanism was in the test harness's
   plumbing, not the guard's logic.

---

## 900+ green tests never exercised the first edit of a seeded row

**TASK-2451, 2026-08-06.** Enriching the seeded 'Default Assistant' character card
(`character_cards` id=1) meant writing a conditional `UPDATE` to that row. A quick
manual prototype — construct a real `CharactersRAGDB` against a temp file, then run
that `UPDATE` — crashed immediately with `sqlite3.DatabaseError: database disk image
is malformed` (`SQLITE_CORRUPT_VTAB`). Nothing about the prototype's content mattered:
even `UPDATE character_cards SET description = 'x' WHERE id = 1` crashed the same way,
on a completely fresh database, through the real constructor. Root cause: row 1's
`INSERT` in `_FULL_SCHEMA_SQL_V4` ran *before* `character_cards_fts` and its
`character_cards_ai` trigger were created later in the same script, so row 1 was never
indexed into the FTS5 shadow tables — on every database this schema had ever produced.
The first `UPDATE` to that row makes `character_cards_au` ask FTS5 to remove index
entries that were never inserted, and FTS5 reports that as disk corruption. This means
`update_character_card(1, ...)` — an ordinary user editing the built-in Default
Assistant via the normal Roleplay editor — already crashed the app, on every existing
install, before this task touched anything. The full `Tests/ChaChaNotesDB/` +
`Tests/DB/` suites (900+ tests, routinely green) never caught it, because no existing
test performs a write against character id=1 as the first write after database
creation — every test that touches character cards either inserts a fresh row first or
edits a different id.

**What to do.**
1. Before trusting a migration's `UPDATE`/`INSERT` against a long-lived seeded row,
   prototype it against a database built the SAME way production builds one (the real
   constructor, not a hand-rolled minimal fixture) and actually run the write — do not
   reason about FTS5/trigger ordering from reading the schema SQL alone.
2. "900+ passing tests" is not evidence a specific write path has ever been exercised.
   Ask what the very FIRST write to a specific row would look like, and whether any
   test performs exactly that — a seeded/default row (id=1, "the default X") is
   disproportionately likely to be read constantly and written to never, in the whole
   existing suite.
3. A `content=`/`content_rowid=` FTS5 table's row must be created strictly after the
   external-content table's own `INSERT` trigger exists, or the row is invisible to the
   index forever while still being readable via plain `SELECT` (FTS5 can satisfy an
   unfiltered `SELECT` straight from the content table, so `SELECT rowid FROM fts_tbl`
   looks fine and gives no warning). `PRAGMA integrity_check` also reports "ok" in this
   state — it does not catch this class of defect. The tell is `SQLITE_CORRUPT_VTAB`
   (not generic `SQLITE_CORRUPT`) the first time a row in that state is deleted or
   updated. Fix forward with `INSERT INTO fts_tbl(fts_tbl) VALUES ('rebuild')`, which is
   safe and idempotent for exactly this "shadow tables drifted from content" situation.

---

## A single red run is not causation — run both arms

**The incident (2026-08-07, Console decomposition wave 4).** I deleted 35 lines of
provably dead code from `on_button_pressed` — a branch whose button id had been
removed by an earlier commit, confirmed dead by a whole-repo sweep and by an
independent reviewer. Immediately afterwards
`test_console_workspace_context_rail.py::test_conversation_status_row_label_and_value_are_separate_visual_runs`
failed. It failed again in isolation. It passed on the commit before the deletion.
Three signals all pointing the same way, and all of them wrong.

The controlled version: three runs with the change and three without.

- **Without** the deletion: 1 passed, 2 failed.
- **With** the deletion: 2 passed, 1 failed.

Same distribution. The test is nondeterministic on its own — it asserts on
`_composited_rows(...)[0]` and the rail's composited rows do not always arrive in
the same order (filed as task-3025, as a possible product nondeterminism rather
than only a flaky test).

Had I stopped at "it passes on the parent commit and fails on mine", I would have
reverted a correct deletion and gone looking for a mechanism that does not exist.
A subagent on the same task had earlier called the same test "a cross-file flake,
investigated and cleared" after re-running it a couple of times — which was the
right conclusion reached by a method that would equally have produced the wrong
one.

**What to do.**
1. Before attributing a failure to your change, run it **N times on both arms**
   (three is usually enough to expose a coin-flip; one is never enough). Restore
   your change with `cp` from a scratch copy or an `Edit`, **never**
   `git checkout --`, which silently discards uncommitted work — that has cost a
   whole test rewrite in this repo before.
2. "Passes on the parent commit" is one sample, not a control. So is "fails in
   isolation" — isolation removes cross-file order dependence, but says nothing
   about nondeterminism inside the test itself.
3. A test that indexes into a rendered/composited collection (`rows[0]`,
   `children[2]`) is a prime candidate: it will fail the moment ordering varies,
   and ordering varies for reasons that have nothing to do with your diff. When
   you find one, ask whether the *product's* ordering is guaranteed before you
   "fix" the test to match whatever it did today.
## A fixed `pilot.pause()` before querying a worker-mounted widget is an ordering landmine (2026-08-05)

**Incident.** TASK-2154.3 added one event-loop turn to the Console left rail's
settling path (a mid-recompose fit-pass defer in
`ConsoleWorkspaceContextTray._fit_height_to_content`). Every targeted suite stayed
green, but at FILE level
`test_console_workspace_many_conversations_keep_lower_status_reachable` failed with
`NoMatches: '#console-new-workspace-conversation'` — and passed standalone, and the
whole file passed on HEAD. Bisecting showed ANY preceding pilot test (not just the
ones the diff touched) tripped it: on a warm event loop the legacy-alias mount chain
(`call_after_refresh` → `run_worker` → `await mount()`) lands one turn later than the
test's single fixed `await pilot.pause()`. A scratch test with a polling wait proved
the button still mounts promptly — pure test-timing fragility, no production
regression — so the fix was one `_wait_for_selector(...)` line in the test, not a
production change.

**What to do.** Never query a control that mounts through an async worker
(`run_worker`/`call_after_refresh` chains, e.g. ChatScreen's out-of-band legacy
aliases) after a fixed pause; poll for it like every other async-mounted widget.
When a test fails only at file level, bisect pairs (predecessor + victim) on your
tree AND on HEAD before touching production code — the pair run on HEAD (green) vs
your tree (red) separated "my change added a turn" from "the test never mounted" in
two 5-second runs, and a generous-timeout scratch replica answered the only question
that mattered: does the widget eventually appear at all?

---

## A keyboard funnel through `Button.press()` dies silently when the button gains a real disabled state

**The trap.** A key handler that "clicks" a button via `Button.press()` inherits
Textual's guard: `press()` returns early when `self.disabled or not self.display`
(Textual 8.x), posting no `Pressed` message and raising nothing. The moment that
button gains a genuine `disabled=True` state, the keyboard path stops reaching the
handler — no error, no test failure unless a test drives the *key*, and any
side-effect the key path performed first (stash, arming flags) is left stranded.

**What happened.** TASK-2154.6 gave the Console Send button a real disabled state
(FR-04). The Enter hotkey in `ChatScreen.on_key` captured the draft into a pending
stash and then routed through `query_one("#console-send-message").press()`. With
Send disabled (blocked/empty draft) the press no-opped: the blocked-attempt
feedback (toast + transcript system row) never fired, the stash stayed pending,
and the *next* Enter was swallowed as a duplicate of the stranded one. Only a
from-source read of `Button.press()` surfaced it; every existing test pressed the
button directly, so nothing else would have caught it.

**What to do.** Before adding `disabled=True` to any button, grep for
`.press()` and `pilot.click` on its id across both production and test code —
those callers silently change behavior. A keyboard funnel that must keep working
while the button is disabled needs an explicit branch that dispatches the same
handler directly (the Console voice-send path's synthesized
`handle_...(Button.Pressed(button))` pattern), plus a test that drives the *key*
in the disabled state.

---

## A keyword `-k` suite deselects behavior-affected tests whose names lack the keywords (2026-08-05)

**Incident.** TASK-2154.7 changed the Console provider-recovery resolution
(which blocker wins: provider vs model). The task's prescribed verification was
`pytest Tests/ -k "onboarding or setup_card or setup_modal or readiness"` —
it reported 3 failures. The full run of every file that calls the changed
helpers reported **6**: `test_console_empty_transcript_choose_model_opens_settings`,
`test_console_blocked_inspector_explains_impact_and_next_action`, and
`test_console_empty_transcript_exposes_beginner_activation_actions` assert the
same card action/inspector copy but share no substring with any filter keyword,
so `-k` silently deselected them. One more
(`test_console_add_api_key_recovery_tolerates_missing_session_settings`) only
surfaced by grepping Tests/ for callers of the changed functions — it
monkeypatches the display helper to return `settings=None`, a defensive
contract the rewrite had to keep.

**What to do.** A `-k` filter matches test *names*, not behavior. Before
trusting it as a completion gate, `Grep` Tests/ for every function you
changed (`_console_provider_recovery_action`, `_build_console_setup_card_state`,
...) and run the full files that reference them — renamed or
indirectly-exercised callers are exactly where stale expectations hide.

---

## Classifying user copy by loose substring invents the blocker you name first

**Incident.** TASK-2154.12. `build_console_disabled_reason` mapped the
setup-blocker sentence onto a short "Send blocked — …" reason with ordered
substring checks, `"model"` first. The real missing-API-key copy is "Add API
key in Settings > **Providers & Models** before sending." — which contains
"model" as a substring of the settings screen's name, so the Console spent
weeks telling users to "choose a model" when the actual blocker was a missing
key (and the missing-endpoint copy hit the same trap). The parametrized
mapping tests never caught it because they fed clean synthetic strings
("Provider setup needed: OpenAI missing API key") that share no wording with
the strings production actually emits; the mis-mapping only surfaced in a
live UAT walkthrough of the reason strip.

**What to do.** When tests parametrize a classifier over free text, include
the **verbatim production strings** as cases (grep the producers, paste them
in) — synthetic inputs exercise the branches you designed, not the text you
ship. And when substring-matching user-facing copy, match the most specific
phrase first and treat UI names ("Providers & Models") as false-positive
carriers for every keyword they happen to contain.

---

## Textual BINDINGS on a child are preempted by an ancestor's `on_key` that stops the event

**Incident.** TASK-2154.11 made the Console transcript's jump-to-latest pill
(`ConsoleTranscriptJumpPill`, a child of `ConsoleTranscript`) keyboard
activatable by adding `BINDINGS = [Binding("enter", ...), Binding("space", ...)]`.
The pilot test pressing `enter` on the focused pill kept failing: the action
never fired. Key events bubble from the focused widget up the DOM *before*
App-level binding dispatch (`App._on_key` -> `_check_bindings` over
`focused.ancestors_with_self`) ever runs, and `ConsoleTranscript.on_key`
stops `enter` mid-bubble — so the pill's binding table was consulted nowhere.
The widget-level `key_<name>`/`on_key` path is the only dispatch guaranteed
to reach a focused child first.

**What to do.** When making a child widget key-activatable inside a parent
that has its own `on_key` handler (transcripts, lists, message rows),
intercept the key in the child's own `on_key` (stop + prevent_default), the
idiom `ConsoleTranscriptActionButton.on_key` already uses — do not rely on
the child's `BINDINGS`, and write the pilot key-press test first: it is the
only thing that reliably exposes the preemption.


---

## A long no-match input does not prove a matched scanner is linear

**TASK-856, 2026-08-08.** The sanitizer's long-input regression used only a
string with no credential labels. It therefore exercised the scanner's cheap
no-match path while completely missing repeated suffix scans after successful
quoted-label matches. The final whole-branch review added a dense matched-input
probe and measured the old quoted path performing **94,996,790 characters** of
CR/LF search work on only **46,888 input characters**.

**What to do.** A complexity claim about a scanner needs adversarial input that
repeatedly takes the expensive matched branch. Count deterministic work—such as
characters searched or cursor visits—and assert a structural bound alongside
the exact output. Do not use wall-clock thresholds: they are noisy and can pass
a superlinear implementation on a fast machine.

---

## A button's region width proves nothing about whether its label renders

**Incident.** TASK-2154.14 (DS-01) relabeled the Console composer's `☰`
button to `Menu`, widening it 4 -> 6 cells. `button.region.width` and
`content_region.width` (6 and 4) both said the 4-cell label fit, and every
geometry assertion was green — but the painted UAT capture read `Me`.
Textual 8's `Button` reserves `line-pad: 1` (one column each side of every
rendered line) *inside* the content region, on top of padding, so the real
label budget is `region - padding - 2`. The trap compounds: `line-pad: 0`
is rejected by the TCSS parser (`_process_integer` errors on a literal `0`,
and the stylesheet loses every rule after the bad one — the generated
bundle documents an earlier collision), so the pad can only be cleared
inline (`button.styles.line_pad = 0`, which parses fine). The existing
`region.width == 14` pin on the neighboring `Composer ▾` toggle had encoded
the same +2 chrome without naming it; tightening that button to 12 only
worked *because* the pad was cleared.

**What to do.** When budgeting a Textual button label, verify with
`button.render_line(0).text` or a painted SVG/text capture — never with
region arithmetic alone. If a label needs its button's full content width,
set `styles.line_pad = 0` in Python (the CSS form does not parse) and
record the budget math in a comment, the way `_bounded_button` call sites
in `console_composer_bar.py` now do.

---

## A conflict-free rebase can still replay a test for an obsolete base contract

**Incident.** PR #1435 added `Tests/MCP/test_library_tools.py` on a branch
whose original base allowed raw `tools/call` dispatch. Current `dev` had since
added a typed security refusal requiring execution through the permission-gated
action. The rebase was entirely conflict-free because the feature commit
created the test file, so Git had no overlapping lines to flag; the focused
suite was what exposed the stale expectation. A prior task note even claimed
the test had been updated, but the committed tree still expected raw dispatch.

**What to do.** Treat a clean rebase as transport evidence, not compatibility
evidence. Re-run the feature's complete focused suite after rebasing, and verify
claimed conflict adaptations in the committed files themselves. Newly created
tests are especially likely to preserve assumptions that the new base has
intentionally invalidated without producing a textual conflict.

---

## A clear/cleanup assertion cannot pin the thing it cleans up — observe transient state DURING the window

**TASK-3170 Task 8, 2026-08-07 (Console auto-retrieve send-path injection).** The
in-flight "Retrieving…" placeholder staging call
(`self._stage_console_library_rag_launch(placeholder)` inside
`_maybe_auto_retrieve_for_send`, `chat_screen.py`) was pinned by **no test**.
Replacing the stage call with a bare `pass` left all 22 existing tests in
`Tests/UI/test_console_auto_rag_on_send.py` green. The reason: every existing
assertion about the placeholder was a **clear** assertion —
`assert screen._pending_console_launch_context is None` after a timeout, a
failure, or a zero-result outcome — and all three stay true whether the
placeholder was staged and then cleared, or never staged at all. The
assertions were only *transitively* meaningful; delete the stage call and they
go vacuous while continuing to pass. A future refactor could therefore remove
the only in-flight signal the user gets during the retrieval window and the
suite would say nothing.

**Fix:** a new test whose fake retrieval service observes
`screen._pending_console_launch_context` and `screen._has_staged_console_evidence()`
**from inside** the `search()` call — the only moment the claim ("a placeholder
is staged while retrieval runs") is actually true — then asserts the settled
launch afterwards is a *different* object with `status == "staged"`. Written
RED-first: the stage call was stubbed to `pass` before the test existed,
confirmed exactly 1 failure (the new test) against 22 unaffected; reverted via
Edit, production source byte-identical to the pre-fix commit.

**What to do.** Transient state — an in-flight marker, a spinner, a lock held
across an `await`, a placeholder later replaced or cleared — must be observed
**during** the window it exists, from inside the awaited call or an equivalent
hook, or it is not tested at all. `assert x is None` taken after the fact
passes identically whether `x` was set-then-cleared or never set: it is a
clear/cleanup assertion, not a presence assertion, and clear assertions cannot
pin the thing they clean up.

---

## A config default that also ships in the config template cannot be mutation-tested through config

**TASK-3170 Task 8, 2026-08-07, same task as above.** Mutating the read
site's fallback for `rag_auto_retrieve_on_send` — `get_cli_setting("chat_defaults",
"rag_auto_retrieve_on_send", False)` → `..., True)` — failed **zero** of the
20 tests then covering the feature. The cause: Task 7 had already added
`[chat_defaults] rag_auto_retrieve_on_send = false` to `config.py`'s DEFAULT
CONFIG TEMPLATE, so every freshly-bootstrapped test config carries the key
explicitly. The lookup therefore always resolves the template's stored value
and never falls through to the Python-level default argument — the literal
`False` in the `get_cli_setting(...)` call is dead code for every test, and
for every real user with a current config. The mutation is only reachable for
a user whose `config.toml` **predates** the key, a state no test that boots a
fresh config can ever produce.

**Fix:** `test_toggle_default_is_off_at_the_read_site` monkeypatches
`get_cli_setting` with a recording stub and asserts the literal default
argument handed to it is `False`, independent of whatever the template
supplies.

**What to do.** Before trusting a "defaults to off" (or any) test for a
config-backed value, check whether that same default also ships in the app's
default config template or any fixture/bootstrap path the test uses. If it
does, every test config already carries the key, and the code-level fallback
can drift to the wrong value with **zero** test failures — a mutation of the
fallback argument is invisible through the normal read-and-assert path
because that path is testing the template, not the code. The read site's own
literal default needs a separate, direct assertion (stub the accessor, assert
the literal argument passed to it), not an inference from observed runtime
behavior.

## A guard test must be PROVEN to discriminate — twice in one day it wasn't (2026-08-08, tasks 1359/2832)

Two review-verified tests, written by the same controller, both passed while
guarding nothing:

1. **`capsys` does not observe loguru.** The task-2832 log-privacy test
   asserted a secret query never appears in `capsys.readouterr()`. The
   reviewer emitted `logger.warning("… query=<the secret>")` DURING that
   exact test via a plugin — **1 passed**. loguru's default handler binds
   pytest's *global* stderr capture at import, so the per-test fixture sees
   nothing (and `capfd` misses it too). The house pattern is a list-appending
   sink: `sink_id = logger.add(lambda m: records.append(str(m)))` /
   `logger.remove(sink_id)` in `finally` — ~15 files already use it.
2. **A single-chunk MockTransport body makes any early-abort test vacuous.**
   The task-1359 crawl regression test proved a body was "read in full past
   the sniff window" — but `httpx.Response(200, content=bytes_blob)` delivers
   ONE `iter_bytes()` chunk, which the read loop appends before any abort
   check runs, so the whole body is captured even under the buggy predicate.
   Only multi-chunk (generator) delivery lets an abort actually cut a body.

**What to do.** For any test whose value is "this would catch the
regression": run the regression. Mutate the guarded code back to the buggy
shape (Edit-based, unique marker strings) and READ the red result before
trusting the green one. Both of these were caught only because the review
step re-ran the pre-fix code against the new test; neither red-check had been
done by the author, and both tests were the SOLE pin for their spec clause.

## A regenerated gate artifact is stale the moment something merges ahead of it

**Incident (task-3750, 2026-08-08).** `Docs/security/production-diagnostic-inventory.json`
is a checked-in artifact that a test regenerates and byte-compares. Its most recent
regeneration was commit `f990464ed` — and `f990464ed` was `origin/dev`'s tip, where the
test **failed**. The commit that regenerated the file left it stale on arrival: the
author ran `--write` on a branch, and the PRs that merged ahead of theirs moved line
numbers and added diagnostics. Green on the branch, red on dev, nobody at fault.

Two things follow.

1. **"The gate passed on my branch" is not evidence the gate passes on dev** for any
   test that compares against a regenerated whole-tree artifact. The only honest check
   is after the final rebase/merge — `test_screen_size_ratchet.py` already says exactly
   this in a comment ("a budget derived from a stale base fails the moment it merges"),
   which is how you know it is a repo-wide pattern and not one script's quirk.
2. **Design these artifacts so unrelated churn cannot invalidate them.** The inventory
   hashed each logger call's *line number* alongside its text, so any refactor that
   shifted lines failed a security gate with the call count unchanged and the sink
   topology byte-identical. Measured on dev: of 47 drifted entries, 28 were pure line
   movement. A gate that fires on no-ops trains reviewers to regenerate without reading,
   which destroys exactly the review it exists to force. Key such artifacts on content,
   and keep multiplicity (a sorted list, never a set) so deletions still register.

**What to do.** Before blessing a regenerated artifact, classify the drift instead of
running `--write` and staring at a 1,000-line diff: walk each file's history for the
revision whose digest reproduces the checked-in value, and diff content-vs-position.
That is what separated the 28 no-ops from the 19 real changes here, and it is what made
it cheap to actually read the diagnostics being newly blessed.
---

## A mounted widget with healthy data can still paint nothing — assert the painted region

**Incident.** TASK-3793, 2026-08-08. The Console rail's character avatar was
invisible (even the no-character placeholder was gone) and Roleplay thumbnails
painted black stripes — while every existing avatar test passed, because they
asserted the widget mounted, the DB bytes decoded, PIL produced an image, and
the mosaic content was non-empty. Two layout root causes, both invisible to
composition assertions: (1) the default-width avatar `Static` inside the
auto/auto `ClickableAvatarBox` (task-1661) resolved to 0x0 under Textual
8.2.8 — mounted, composed, painted nothing; a headless repro against the
owner's real DB image showed `region 0x0`. (2) The three thumb containers
reserved `max-width 24` *plus* `padding: 0 1`, so every 24-cell mosaic line
folded at 22 content columns; the continuation rows painted black (stripes)
and the folded 17-row stack exceeded `max-height 10` (bottom clipped) —
`region 22x17` where 24x10 was expected.

**What to do.** For image/avatar/rendering surfaces, pin the *painted region*
(`widget.region`, `render_line(n).text`, or an SVG/text capture) in addition
to mount state and data health — a green mount test says nothing about paint.
Two layout traps to budget for: a default-width child of an auto/auto
container collapses to 0x0 under Textual 8 (size it explicitly from the
renderable grid, as `explicit_cell_size()` now does for mosaics), and padding
inside a max-width container folds full-width lines into black continuation
rows on dark themes (content width = max-width − padding; drop the padding or
shrink the build width). The regression shape that caught both: mount the
real holder and assert region non-zero with height == mosaic rows.
## A wired kwarg is not a working option — assert the OUTPUT varies with the INPUT (task-3301, 2026-08-07)

**The incident.** Task-3301 wired the ingest form's "Chunk size" through to the
chunking service and wrote a test that a plaintext file chunked with
`{"method": "sentences", "max_size": 120}` produces more than one chunk. It
produced exactly one — 2,389 characters of it. The chunking stack's methods
size in their OWN units (`sentences` = sentence COUNT, `words` = word count),
so a form labeled "characters · 100–5000" feeding a hardcoded
`method: "sentences"` meant "120 sentences per chunk": the option was dead at
a SECOND layer even after the kwarg plumbing was fixed, and the PDF path had
shipped this exact combination for months (`max_size: 500` sentences ≈ one
chunk per document) without any test noticing — because every existing test
asserted the kwarg ARRIVED, none asserted the output CHANGED.

**What to do.** For any "wire option X through" task, the end-to-end test must
vary the option and assert the observable output varies with it (governance),
not merely that the value lands in a call's kwargs. A kwarg can land perfectly
and still be a no-op because of unit or key-name mismatches downstream
(`size` vs `max_size` was ALSO live here — `improved_chunking_process` reads
only the latter). The kwargs-arrival test and the governance test catch
disjoint bug classes; you need both.

---

## An app-importing pytest probe outside `Tests/` bypasses the suite's own config isolation

**TASK-3894 (P1 eval harness) Task 4, 2026-08-09.** Capturing real chunk-count numbers
for a new fixture corpus, a throwaway probe was written under the scratchpad directory
and run with plain `pytest`. It imported `tldw_chatbook` to call the real chunking path
— and because it lived **outside `Tests/`**, `Tests/conftest.py`'s config-isolation
fixtures (which sandbox `HOME`/`XDG_DATA_HOME` before `load_settings()` ever runs) never
applied, because pytest only collects and applies a directory's `conftest.py` for tests
under that directory. The probe's `load_settings()` therefore ran against the user's real
`~/.config/tldw_cli/config.toml`. No damage this time — the file's mtime was unchanged
afterward, confirming it was read-only — but that was luck, not design: nothing in the
probe prevented a write path from firing, and the probe read as an ordinary pytest
invocation the whole time it ran.

**What to do.** The rule is narrower than "use pytest for anything that imports the
app" — it is **a probe that imports the app must live under `Tests/`**, where the
isolation fixtures are actually collected and applied. A pytest-shaped file outside that
tree runs with none of the suite's safety and is functionally the same as a bare
`python -c` invocation against the app. If you need a throwaway probe, put it in
`Tests/` (even a temp file there) and delete it afterward — or better, promote whatever
it measured into a real, permanently-checked-in test, as this incident did
(`test_the_bare_word_will_appears_nowhere_in_the_corpus`).

---

## A hand-rolled normalizer used as a safety guard must be proven canonical, not just plausible

**TASK-3894 (P1 eval harness) Task 4, 2026-08-09.** A fixture corpus needed a guard
proving no keyword-category query's unique token accidentally overlapped a
vocabulary-mismatch/paraphrase pair's vocabulary. A hand-rolled `_stem()` (strip one
suffix, return) stood in for FTS5's real porter stemmer and looked like a deliberate,
safe over-approximation — the guard's own comment called it "stricter than a real
tokenizer." Review found the opposite was true for a whole class of words: because
`_stem` stripped exactly one suffix and stopped, suffix order decided the result, so two
spellings of the *same* word produced two different stems (`readings`→`reading` but
`reading`→`read`; `classes`→`class` but `class`→`clas`). FTS5's porter tokenizer
collapses every one of these pairs to one stem. The guard was therefore **weaker than
the mechanism it stood in for, in exactly the direction that matters**: it would score a
keyword-reachable pair as "no overlap" and let it ship silently. Two real fixtures
already carried exactly this escape — `vm-blood-pressure`'s "reading" against
`note-hypertension-followup`'s "readings"; a `pr-workout-time` "classes"/"class" pair
that had only been caught by hand, not by the guard.

**What to do.** When a hand-rolled normalizer stands in for a real one as a safety
check, "it looks stricter" is not evidence — an ordering artifact made the opposite true
here, unnoticed by the author. Fix by making the reduction a **fixed point** (re-apply
until the word stops changing) so the result is a function of the word family rather
than of which suffix rule fires first, then test it against the *real* mechanism's known
collisions (the actual inflection families a porter/whatever-you're-approximating
stemmer folds together), not only against your own corpus's current wording. This was
caught only because review independently re-derived what a real stemmer would do on
these words and diffed it against the guard's actual output — not by reading the
guard's code, which reads as reasonable on its own.

---

## HF offline enforcement must be set before `huggingface_hub.constants` EVALUATES, not merely "before import"

**TASK-3894 (P1 eval harness) Task 5, 2026-08-09.** A harness that embeds real documents
through a real model needed a hard guarantee that a run never downloads anything, even
on a cache miss. The first version set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` from
a pytest autouse fixture at test setup and the code claimed downloads were blocked. They
were not: an instrumented check showed `ENV HF_HUB_OFFLINE='1'` alongside
`constants.HF_HUB_OFFLINE=False` and `constants.is_offline_mode()=False` in the same
process. `huggingface_hub.constants.HF_HUB_OFFLINE` is computed **once, at import**, from
the environment as it stood at that instant; `is_offline_mode()` (which `transformers`
also imports directly) just returns that frozen global. An env var written from a
fixture at test *setup* — after collection has already imported half the world — arrives
too late to matter, and a cache miss would have silently downloaded ~87 MB into the
user's real `~/.cache/huggingface/hub`, the very directory the harness was pointed at.
The first fix attempt also used the wrong condition: "before `huggingface_hub` is
imported" is not sufficient, because hf_hub loads its submodules lazily —
`huggingface_hub` can already sit in `sys.modules` while `huggingface_hub.constants` is
still unevaluated, confirmed directly by forcing the hard case (evaluating `constants`
from a module that runs before the latch): the latch still worked, proving "before
import" was never the load-bearing condition. The fix that actually closes the hole has
two parts, and both were needed: set the env vars at **module top** of the gate module,
guarded on the harness's own opt-in env var, so they land before `constants` is
evaluated in the common case; and, for the case where something earlier in the same
session already evaluated `constants` with the var unset,
`monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)` directly on the frozen global —
the only thing that still works at that point. Mutation-tested independently: removing
either half alone reintroduces `is_offline_mode() == False` through a different path.

**What to do.** For any library that freezes an "offline"/"safe mode" flag into a
module-level constant at import time (huggingface_hub is one instance; do not assume any
other library isn't), "set the env var before you need it" is insufficient — the real
requirement is "before that constant is evaluated," which can be earlier than the import
of the top-level package if the package lazy-loads its submodules. Assert the *resolved*
state (`is_offline_mode()`), never the env var's string value: a `"1"` that arrived too
late reads as success on an env-var check while the flag it was meant to control stayed
`False`. And when closing a hole like this with a two-part fix, mutation-test each half
independently — here, either half alone was silently insufficient.

**`HF_HUB_OFFLINE` is not the only frozen constant in that module (TASK-16965,
2026-08-17).** `huggingface_hub.constants.HF_HUB_CACHE` is likewise computed
once at import, from `expanduser("~")` — so any fixture that sandboxes `HOME`
(this repo's `Tests/conftest.py` does) points every later model load at an empty
cache and makes a genuinely cached model unloadable under pytest. Same
mechanism, opposite blast radius: this lesson's is a download you did not want,
that one's is a load you did want and silently did not get. See "A metric can be
graded on fallback content" at the end of this file for what that cost.

---

## "Order-dependent" in the backlog is a hypothesis, not a diagnosis — a state flip is not proof the DOM caught up

**TASK-3022, 2026-08-07.** The backlog described two `Tests/UI/test_library_shell.py`
tests as "order-dependent notes-tail failures" (plus a third found during this task's
own sweep, `test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums`).
All three, when actually run alone, repeatedly (3/3, 3/3, and 2/3 samples respectively)
failed with `NoMatches` on a widget query, not intermittently the way real cross-test
pollution would present. Each had the identical shape: poll a plain/reactive attribute
(`_library_note_detail`, `_library_notes_view` + `_library_note_autosave_state`,
`_library_notes_sync_running`) in a `for _ in range(N): await pilot.pause(...)` loop,
then immediately do a **one-shot** `screen.query_one(...)` on a widget the same state
transition is supposed to (re)mount. Task-699 (2026-07-26) had already diagnosed and
fixed the first known instance of exactly this shape in the same file; these three were
new instances introduced by later test additions that never saw that diagnosis.

**Why it happens.** The Python attribute write and the Textual recompose that renders it
are not atomic. A handler sets `self._library_note_detail = new_value` and only later
`await`s back into the event loop for the recompose to actually mount the widget that
implies. A poll loop watching the ATTRIBUTE exits the instant it flips — one event-loop
tick before the widget it implies is guaranteed to exist. Whether a given run's timing
window is wide enough to hide this varies with machine load, which is exactly why it had
been filed as "order-dependent" rather than diagnosed: it LOOKS like flakiness (some runs
pass) without actually depending on any other test.

**What to do.**
1. Do not accept a backlog description of "order-dependent"/"flaky" at face value — a
   test that fails when run completely alone, even once, is not proof of cross-test
   pollution. Run it alone, several times, before hunting for a preceding-test trigger
   that may not exist.
2. Once a poll loop has established the STATE you care about, wait for the WIDGET too
   via `_wait_for_selector` (this file's helper — polls `screen.query`, a list, so zero
   matches is just "not yet") before reading it — never a bare `query_one` right after a
   state-only poll, since it raises the moment the DOM lags the state by even one tick.
   Cheap enough to apply proactively, not just after a failure is observed.

---

## A non-breaking space does NOT stop Rich/Textual from wrapping there (2026-08-07)

**Incident.** Task-2859 item 5's mid-unit wrap fix ("Prompts 144.0 / KB" splitting a
size number from its unit in the Library rail's narrow Details column) was first
"fixed" by replacing the space between number and unit with U+00A0 (non-breaking
space) — the textbook answer, and it read correctly in two quick manual checks (widths
20 and 29). A live tmux capture at the batch's own required 170x50 caught it still
broken: `"144.0"` on one line, `"KB"` alone starting the next, NBSP already in place.
Direct proof against `rich._wrap` (the module every plain `Static` wraps through):
`rich._wrap.words()` tokenizes with `re_word = re.compile(r"\s*\S+\s*")`, and Python's
`re` module's Unicode-aware `\s` **matches U+00A0 identically to an ordinary space** —
confirmed with `re.match(r"\s", "\xa0")` returning a match. So `"144.0\xa0KB"` is parsed
as TWO separate "words" for wrap purposes exactly like `"144.0 KB"` was; NBSP only
prevented the SPECIFIC widths tried by accident (enough room remained either way, or
"Prompts" itself already pushed the whole tail to the next line together). At the
rail's real width, exactly enough room remained for "144.0" alone but not "144.0" plus
"KB", so the split happened right where NBSP was supposed to prevent it.

**What to do.** Never assume a non-breaking space stops Rich/Textual `Static` word-wrap
— it does not, because Rich's own wrap tokenizer uses a plain Unicode-aware `\s` regex
that does not special-case U+00A0. If a number/unit (or any two-token) pair must never
split, either remove the space between them entirely (verified stable at every width
20-29 for this exact case — `_unbreakable_size_text` in `library_screen.py`), or use a
genuinely non-whitespace-classified character (e.g. U+2060 WORD JOINER, category `Cf`,
zero-width) if a visible gap must be preserved. Test the actual wrap behavior against
`rich._wrap.divide_line`/`words` (or a live capture at the real target width) — a
narrower or wider width than the one actually shipped can hide this exact bug either
way, which is why two quick manual checks at the wrong widths both looked fine.

---

## `Button.press()` called from an ancestor's own click handler silently breaks message bubbling one hop early (2026-08-07)

**Incident.** Task-2859 item 5: making a rail section header's LABEL (not just its
`▸`/`▾` toggle chip) clickable, by adding `DestinationRailSectionHeader._on_click`
that resolves the toggle `Button` and calls `.press()` on it. A live capture showed the
toggle's own CSS class flip to `-active` (proof `.press()` ran) but the section never
opened — no `Button.Pressed` handler anywhere fired, in the widget, the screen, or the
app. Reproduced deterministically in isolation with a minimal `Horizontal` header
wrapping a `Static` + `Button`: calling `child_button.press()` FROM the container's own
`_on_click` (itself invoked because the Static's Click bubbled there) breaks
propagation; calling the exact same `.press()` from the Static's own `_on_click`, or
from a plain test coroutine, works fine. Root cause, found by monkeypatching
`Message._bubble_to` to log every hop: `Message.__post_init__` stamps `self._sender =
active_message_pump.get(None)` — a CONTEXTVAR tracking whichever widget's message
dispatch is CURRENTLY executing, not the widget whose code literally calls
`post_message()`. Since `Button.Pressed(self)` is constructed inside `Button.press()`
while `active_message_pump` still reads as the HEADER (we are executing inside the
header's own dispatch of the bubbled Click), the new message's `_sender` becomes the
header. `MessagePump._on_message`'s bubble step has a special case: `if
message._sender is not None and message._sender == self._parent: message.stop()` —
"parent is sender, so we stop propagation after parent" (an optimization to avoid a
widget's own self-directed message re-bubbling past the ancestor that sent it) — and
this exact shape matches by coincidence, so the Pressed message reaches the header (one
hop) and then dies, never reaching the screen-level handler every real consumer
(Console/Home/Library rails) expects it at.

**What to do.** Calling `widget.press()` (or constructing/posting any `Message`) from
inside ANOTHER widget's own event-handler execution is not equivalent to the target
widget doing it itself — the message's `_sender` provenance silently changes, and
Textual's own "parent is sender" bubble-stop can misfire as a result. Fix: reset the
`active_message_pump` contextvar (`from textual.message_pump import
active_message_pump`) to the actual sending widget around the call —
`token = active_message_pump.set(target_widget); try: target_widget.press() finally:
active_message_pump.reset(token)` (see `DestinationRailSectionHeader._on_click` in
`tldw_chatbook/Widgets/destination_rail.py`). Proving this class of bug needs watching
the FULL bubble chain, not just checking that `.press()` "ran" — the visual `-active`
class flip is a false-positive signal; instrument `Message._bubble_to` (or count how
far a message travels) when a `Button.Pressed` handler mysteriously never fires despite
the button visibly reacting to the press.

## Constructing a widget directly in a test is not the same as driving it through real navigation -- and a "real navigation" pytest attempt can itself be a non-deterministic regression gate (2026-08-08)

**TASK-3200.** Fixing the shared `MainNavigationBar`'s mid-word tab-label clip (a
straddling destination button now gets a CSS "ghost" treatment — colors matched to the
bar's background — instead of `display: none`, since hiding a button changes the
strip's virtual size and can cascade). `Tests/UI/test_master_shell_navigation.py`
already had — and my own new tests added — coverage that constructed
`MainNavigationBar(active="settings")` directly inside a bare `TestApp`, at 80/100
columns, both an early and a late active destination, and every one of those tests
passed cleanly. Live tmux verification at 80 columns then reproduced a DIFFERENT bug
in the exact same scenario: navigating from Home to Settings via the command palette
left "Schedules" straddling and fully readable (un-ghosted) while an unrelated,
already-off-screen "Watchlists" stayed ghosted for no reason. Root cause: `on_resize`
ghost-checked directly, without first re-scrolling for the CURRENT viewport — a real
screen-to-screen navigation fires several resize events while content is still
settling, and whichever resize is the LAST one to land can ghost-check against a
scroll position computed for an EARLIER, narrower or offset layout. A widget
constructed directly in a `TestApp` sees exactly one clean mount → one clean resize;
it structurally cannot reproduce a sequence of several interleaved resizes racing a
still-settling scroll target.

**Second half of the incident, easy to miss.** Having root-caused it live, I wrote a
pytest test driving the REAL app (`Tests.UI.app_factory._build_test_app()`) through an
actual `NavigateToScreen` message, polling for the nav bar to reach a correct state.
It reliably failed against the buggy code and passed against the fix — until re-run a
few more times each way: the SAME buggy code sometimes passed within an 8s poll, and
disabling `MainNavigationBar`'s periodic 0.5s interval (which calls the same
scroll-then-ghost pair on its own schedule, independent of `on_resize`) via monkeypatch
made even the FIXED code fail a 20s poll. The honest conclusion: overlapping
`call_after_refresh` chains from several resize events can interleave such that the
last ghost-check to physically execute is not guaranteed to be from the freshest
chain — a genuine, narrow residual race that the fix reduces but does not eliminate,
and that the ALWAYS-PRESENT interval (this codebase's existing, intentional
"settle every tick" mechanism) papers over within a variable amount of real time. No
pytest timeout value reliably distinguished buggy from fixed once the interval was
back in play, so a test whose pass/fail depended on it was providing FALSE confidence,
not real regression coverage — it was deleted (along with its exclusive helpers)
rather than shipped in that state.

**What to do.**
1. A test that constructs a widget directly and pumps `pilot.pause()` is evidence the
   widget's OWN logic is internally consistent — it is NOT evidence the widget behaves
   correctly when driven by the app's real navigation/layout churn, which can fire the
   same hooks (e.g. `on_resize`) multiple times with different timing than a synthetic
   single-shot test ever produces. For a defect involving scroll/layout state that must
   "settle", drive the real navigation path at least once (post the actual message,
   wait for the actual screen class to change) BEFORE declaring victory on
   direct-construction tests alone — that is what surfaces this class of bug.
2. Before trusting a NEW test as a regression gate, re-run it several times against
   BOTH the buggy and the fixed code, not once each. A single RED + a single GREEN can
   still be a coin flip if the code involves unmocked real-time async settling
   (`call_after_refresh` chains, `set_interval` timers) — the fact that it says
   "5 failed" once and "3 passed" against the identical buggy commit later in the same
   session is itself the tell, not a fluke to shrug off.
3. If a bug is fundamentally a timing race between two async mechanisms (here: a
   settle-chain fix and an always-on periodic interval), a fast deterministic pytest
   assertion may not exist for it at all. Neither "leave the interval running,
   generous timeout" nor "disable the interval, even more generous timeout" reliably
   discriminated buggy from fixed here. Don't force a flaky or falsely-reassuring test
   into existence to satisfy a coverage checklist — ship the deterministic tests that
   DO reliably discriminate (the direct-construction geometry/rendered-text ones, in
   this case) and rely on documented, reproducible LIVE verification (tmux, before vs.
   after) for the part that genuinely can't be pinned by a fast unit test.

## A mutation test can stay green because a *second* self-healing mechanism rescued the mutated code (2026-08-09)

**Incident (task-3200 round 4 / task-3225).** `MainNavigationBar.on_resize` was
wired to a focus-aware recenter, with `test_resize_does_not_strand_the_focused_
button` as its regression guard. Reverting the wiring did not turn the test red.
The first diagnosis (round 3) was "the scenario never strands" -- true, but only
half of it. A hand-built scenario that DOES strand *still* passed against the
reverted code, because two independent backstops healed it faster than any
wall-clock assertion could look: the widget's own 0.5s settle interval, and --
the one nobody had accounted for -- a "best-effort nudge"
(`scroll_to_widget(focused)`) buried inside the ghost pass, which fired off a
*stale* region that still measured as straddling. Traced: with the fix reverted,
`scroll_x` went 86 -> 75 (wrong) -> 96 (rescued) inside 40ms.

**The rule.** When a mutation test refuses to go red, "the scenario is wrong" is
only the first hypothesis. The second is "something else fixed it for me."
Before trusting any timing-sensitive guard, enumerate every mechanism in the
system that could reach the same end state -- periodic intervals, deferred
re-checks, best-effort nudges -- and either suppress them for the duration of
the assertion (isolating the unit actually under test) or pick a scenario they
provably cannot reach. Here: suppress the interval via a test-local subclass
(patching the instance attribute does nothing -- `set_interval` captured a bound
method at mount), and choose a case that drags the button *fully off-screen*
rather than into a straddle, since the nudge only rescues straddlers. Result:
3/3 red on revert, 3/3 green on restore.

**Corollary on assertions.** "not straddling" was also too weak an invariant to
distinguish the good state from the worst one: a button dragged entirely
off-screen is not straddling either, and it is strictly worse (invisible, yet
still focused and Enter-navigable). Assert the property a user would name
("still fully visible"), not the negation of the specific bug you last fixed.

## An "invisible" CSS class that touches the box model is a layout change -- and a different CSS tier can hide that from your tests (2026-08-09)

**Incident (task-3200 round 4 / task-3225).** The nav bar makes a clipped tab
invisible with a CSS class instead of `display: none`, specifically so that
geometry never changes (hiding a tab reflows the strip, breaks `max_scroll_x`,
and cascades into new clipped tabs). The rule declared
`border: solid $background !important` -- and Textual's `Button.-style-default`
default is `border: none` plus `border-top`/`border-bottom: tall`, i.e. **zero
horizontal border cells**. So "make it invisible" silently made every ghosted
button **2 cells wider** (measured: 14 -> 16), reflowing every later button and
pushing an already-corrected, focused tab back into a clipped position one
layout pass after the correction landed. That was the whole "mysterious ~0.3s
drift-back": a settle pass's own trailing invisibility pass undoing the settle.

**Two generalisable traps.**

1. Invisibility rules must declare colors only. `border`, `padding`, `width` and
   `visibility` all move the box. If you want the Textual primitive for
   "invisible but still occupies space", note that `visibility: hidden` makes
   `Widget.region` return an EMPTY region (`outer_size` keeps its real value,
   `region.width` drops to 0) -- so any code that reads `.region` to decide
   whether to *un*-hide it can never see the widget again. Measured, and the
   reason that approach was rejected here.
2. **A widget-level `DEFAULT_CSS` bug can be invisible in the real app and live
   only in your tests.** This one never bit production: the bundle's
   `Button { border: none; }` sits in the `CSS_PATH` tier, which outranks widget
   `DEFAULT_CSS` regardless of `!important`, so the bad declaration was silently
   discarded in the running app -- and *only* applied in the bare `App()` test
   harness, which is where the entire deterministic suite for this feature runs.
   The harness was modelling a different layout regime than production. When a
   geometry finding comes out of a bare-widget test, re-measure it under a
   bundled-CSS harness (`CSS_PATH = tldw_cli_modular.tcss`, as
   `test_mcp_inspector.py`'s `InspectorAppWithBundledCSS` does) before deciding
   what it means -- in both directions: a bug the harness shows may not exist
   live, and a bug live may not show in the harness.

## A static symlink fixture does not prove a scanner is no-follow (TASK-13200, 2026-08-09)

**What happened.** The guided audio.cpp package scanner correctly skipped a
nested symlink that existed before scanning, so its original path-escape test
passed. A mutation fixture then replaced an already-queued directory with a
symlink. The next `scandir(path)` followed the new target and produced an exact
candidate outside the selected tree. A pre-open `lstat` alone still left a
smaller replacement window while the iterator was being opened.

**What to do.** Test no-follow traversal at three boundaries: a link present at
discovery, a queued directory replaced before traversal, and replacement while
the directory iterator opens. Fence the queued identity both immediately before
and immediately after opening the iterator; close without iterating if either
observation differs or becomes a symlink/reparse point. For files, combine a
no-follow open with `fstat` identity/type comparison before reading metadata.
Static fixtures prove policy for stable trees, not race safety.

PR #1463 review exposed a second portability trap: treating a missing
`O_NOFOLLOW` as flag value zero silently removed the primary file-open fence
while leaving the post-open identity check looking reassuring. Add an explicit
missing-capability mutation test. If the platform cannot guarantee no-follow,
fail closed before `open()` instead of opening first and rejecting afterward.
## A targeted subtree swap must account for route-owned siblings outside that subtree (2026-08-09)

**Incident (task-13213).** The Library optimized Notes/Media navigation by
replacing only `#library-canvas`. The Notes `Database | Files` source strip is
not a canvas child; it is a route-owned sibling composed above the shell grid.
The optimization therefore made Notes look selected while omitting the only
entry into file-backed notes, and it could leave the same strip stale after
leaving Notes. A shell-identity regression stayed green because it asserted
only the widgets the optimization deliberately preserved, never the contextual
sibling it had skipped. Once the route boundary was tested, the original code
failed at both 120x40 and 160x45.

**The rule.** Before introducing a targeted recompose, inventory every
route-owned surface, including siblings and wrappers outside the replacement
host. Encode that inventory as a structural signature and use the targeted path
only when the mounted signature matches the destination; otherwise await the
canonical full composition seam. Tests must assert both halves of the boundary:
contextual chrome appears on entry and disappears on exit. If stable child IDs
are reused, hide and detach the outgoing subtree before mounting its replacement
or Textual will reject the duplicate even when the route signature matches.

---

## Adding a resource of a GUARDED KIND obliges you to run that kind's inventory suite, not just your feature's tests

**Hybrid-fusion cluster (TASK-3996) Task 5, 2026-08-09.** The new notes/conversations
keyword sub-legs opened SQLite directly:
`sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)`. The choice was deliberate and
well-argued in the commit (read-only, never `CharactersRAGDB`, whose constructor does
schema and client-registration work on the user's main DB), it was covered by six new
tests, and it survived a full task review. It was also, the whole time, **already
failing a committed repo-wide guard**: `Tests/DB/test_private_sqlite_inventory.py`
asserts that the only production `sqlite3.connect` call sites are the private-sqlite
seam's own, with every owner enumerated in an inventory document and pinned by a ratchet
count. Nothing in the task's own test selection — the feature's tests, the RAG_Search
sweep, the eval battery — includes `Tests/DB/`, so the violation was invisible to every
run that was made. It only surfaced in a review round, from reading, not from a red
test.

This is the failure mode of targeted-test discipline (which is otherwise right: this
repo's rule is branch-relevant files plus a `--collect-only` sweep, not routine full
suites). Targeted selection is chosen from *the files you changed*. An inventory guard
lives in a directory you did not touch and asserts a property of the whole repo, so
"relevant to my change" and "relevant to the guard" are different sets, and the guard's
whole purpose is to notice the case where the author did not think it applied.

**What to do.** Before finishing a task, ask what KIND of thing it added, and whether
that kind is under a census: a raw DB connection (`Tests/DB/test_private_sqlite_inventory.py`),
a CSS class or token, a tool gate, a screen route, a config key. If yes, run that kind's
inventory suite *and add the new resource's row to its inventory*, in the same commit —
a guard you satisfy by exempting yourself is not satisfied. Name those suites in the
dispatch when the task is known up front to add such a resource, because the agent doing
the work is exactly the one who will not think to look for them. The fix here was to
route the sub-legs through `connect_private_sqlite` with a registered owner
(`rag.chachanotes_keyword_leg`, read-only URI), add the inventory row, and bump the
ratchet — which is what the guard existed to make happen, roughly a day later than it
should have.

---

## `Widget.focus()` is deferred — a same-handler capture of `app.focused` sees the old widget

**task-3311, 2026-08-09.** The Ingest Clear handler called `path_input.focus()` and
then a structural recompose helper that captures `app.focused` to restore focus
afterwards. In Textual 8, `Widget.focus()` does NOT set focus synchronously — it
queues `screen.set_focus` through `app.call_later` — so the capture still saw the
just-clicked Clear button, and the post-recompose restore targeted the NEW Clear
button, hidden for an empty path. `Screen.set_focus` silently no-ops on a
non-focusable widget, so focus stayed wherever the recompose prune dropped it: the
rail search box (typed path tail became a Library search) or nowhere (a leading
"/" ran the global focus-search binding). Live it presented as a 2-of-4
intermittent; headless, with a preflight staged, it failed deterministically on
iteration 0 of an 8-pass loop.

**What to do.** When a handler must hand focus somewhere before code later in the
SAME handler reads `app.focused`/`screen.focused` (capture-and-restore helpers,
recompose context savers), use the synchronous `Screen.set_focus(widget)`, not
`widget.focus()`. And remember `set_focus` on a non-focusable (hidden/disabled)
widget does nothing and reports nothing — a focus-restore path that can name a
display-managed widget needs the target to be focusable, or a fallback.

---

## A responsive focus handoff must cover both directions of widget replacement

**TASK-16220, 2026-08-14.** The first Console rail fix moved focus from a rail
that disappeared at a resize breakpoint to its reveal handle. That passed both
single-transition regressions. Independent review then exercised consecutive
boundaries: 117→118 focused the Context handle correctly, but 118→129 reopened
Context and hid that focused handle, leaving focus as `None`. The handoff had
modeled rail→handle replacement but not handle→rail replacement.

**What to do.** When responsive layout replaces one focusable representation
with another, test both directions and at least one consecutive transition.
Capture the logical owner before applying visibility, then synchronously focus
the visible counterpart after the update; two isolated one-way tests do not
prove keyboard continuity across adjacent bands.

---

## Bisecting dev-baseline test rot without a checkout: `git archive` trees run against the same venv

**task-3315, 2026-08-09.** `Tests/UI/test_library_shell.py` carried 56 failures on
the dev base, and the question that decided every repair was WHERE each family
broke: the ingest arc, dev's own churn, or the very PR that authored the pins.
With mutating git commands off-limits (shared checkout, other agents active),
`git archive <sha> | tar -x -C scratch/tree_<sha>` + `cd tree_<sha> && <worktree>/
.venv/bin/python -m pytest ...` reproduced the suite at any historical commit —
cwd wins over the editable install on sys.path (verify once: print
`tldw_chatbook.__file__`). Running the 60x20 geometry family at `6b4ccf475` (the
notes-adaptive PR #1439 merge that INTRODUCED those tests to dev) and at the dev
base proved the identical 14-test failure set at both: the family was born broken
at its own merge, and the media-ingest arc was exonerated in one run. The same
technique caught that the pins' authoring-branch snapshot (`42c994486`) was itself
too broken to run — "the tests passed when written" is not a safe assumption for
a PR whose battery only ever ran `-k` slices.

**What to do.** When a full-file suite is red on a base you didn't build, don't
reason from blame alone: extract the tree at the suspect merge commits with
read-only `git archive` and run the failing family there. A failure set identical
at the introducing merge and at base names the culprit (pins merged unvalidated)
and scopes the fix to re-pinning; a set that appears only later points at product
churn to bisect further. Corollary of the incident: line numbers in an earlier
failure report drift as you edit the file — re-derive the failing STATEMENT
before diagnosing (a "status query" failure here was actually the post-completion
query racing the finish-of-run recompose, three asserts later than first read).

---

## A heuristic candidate list is not a complete remediation inventory (2026-08-09)

**Incident (TASK-2118 final review).** A spelling-filtered logger sweep was
correctly documented as heuristic AC evidence, but its content-bearing subset
was later copied into a follow-up task as though it were the complete
summarization privacy inventory. Reviewing every logger call in the two owned
modules found many more prompt, response/output, credential-fragment, private
endpoint/path, and exception/error-detail diagnostics that the filter was never
designed to find.

**The rule.** Preserve the stated proof boundary when evidence crosses into a
follow-up. Build remediation inventories from the complete owning population,
grouped by stable module/function/diagnostic identity; use heuristic matches
only as candidates or cross-checks, never as the denominator.

---

## A line-independent diagnostic digest can still be indentation-sensitive (TASK-14651, 2026-08-09)

**Incident.** The persistent-diagnostic inventory described its call digests as
position-independent. Moving an unchanged multiline Library diagnostic into a
more deeply nested block still changed its digest because
`ast.get_source_segment()` retains continuation-line indentation. Later in the
same reconciliation, range-formatting three already-reviewed logger calls made
the architecture gate red again even though their AST behavior was unchanged.

**The rule.** Treat this inventory as line-number-independent, not
whitespace-independent. When reviewing a delta, compare the actual logger-call
AST/source as well as the digest so a pure indentation change is not mistaken
for a policy change. Run formatter gates before the final generated-artifact
refresh, then rerun the inventory checker after formatting. Do not refresh the
manifest first and assume later formatting is harmless.

---

## A generated-inventory rebase conflict is a new review boundary (TASK-14651, 2026-08-10)

**Incident.** Rebasing the diagnostic-privacy PR 79 commits onto current dev
conflicted in the generated manifest. Regenerating it made the architecture
gate green, but also imported 17 upstream diagnostic additions since the prior
reviewed base. Sixteen carried implicit tracebacks, exception messages, bound
session/message IDs, a media ID, or user-entered trim values. Treating the
generator as a mechanical conflict resolver would have silently blessed every
one.

**The rule.** A governed generated artifact must be re-reviewed when its source
population changes during rebase. Compare the diagnostic call population from
the last reviewed base to the new base, classify each added or changed call
under the governing ADR, extend the guard for newly observed syntax such as
dynamic `exception=` values, stdlib exception/stack capture, chained
`bind(...)` fields, and direct keyword-format values, then regenerate. Passing
the generator proves consistency, not policy compliance.

---

## An aggregate scanner label must not erase an explicitly accepted candidate identity (TASK-13201, 2026-08-09)

**What happened.** Guided audio.cpp launch tests placed Supertonic and PocketTTS
in separate temporary roots, so each rescan returned discovery state `exact`.
The real user package directory held both reviewed GGUF files. The scanner
correctly returned one `ambiguous` discovery containing two exact candidates,
and both candidates had been explicitly accepted with their recipe, root,
configuration, and weight identities. Launch revalidation nevertheless required
the aggregate discovery state to be `exact`, so it rejected the otherwise
unchanged two-model setup before creating the generated configuration.

**What to do.** Keep unresolved discovery ambiguity and accepted-candidate
identity as separate concepts. Never silently choose from an ambiguous result,
but once the user has explicitly accepted a candidate, revalidate that exact
candidate and require one matching identity; do not reapply the aggregate label
as if no choice had been made. Multi-model integration fixtures must include the
common real layout where several supported packages share one selected root,
not only the tidier one-directory-per-model arrangement.

---

## A repository scanner must decode source explicitly or a platform can silently disappear files

**Console model-picker verification (TASK-3600/TASK-14812, 2026-08-10).** The
blocking-I/O architecture suite reported eight stale baseline entries for two
Chatbooks modules even though the referenced `.glob()` and `ZipFile` calls were
still present. On Windows, `_scan_package` used `Path.read_text()` without an
encoding. The locale codec could not decode bytes in those UTF-8 source files;
the scanner caught `UnicodeDecodeError` and skipped each entire module. The
stale-baseline assertion was therefore reporting missing scan input, not clean
code. Reading the same files explicitly as UTF-8 restored the intended findings
and made all six guard tests pass.

**What to do.** Repository-wide source scanners must use an explicit source
encoding, normally `encoding="utf-8"`, and must treat decode failures as visible
evidence rather than silently interpreting them as a clean file. When an
inventory entry becomes "stale" while the named code is visibly still present,
inspect the scanner's input and exception path before deleting the baseline.

---

## Windows device names still capture filenames with extensions

**task-14811.1, 2026-08-10.** A new auxiliary-attempt migration test named its
SQLite fixture `aux.db`. The preceding 16 focused cases passed, but Windows
resolved that basename as the reserved `AUX` device (`\\.\aux`) even with the
`.db` suffix. The private-SQLite directory verifier then correctly rejected the
device path as a missing/non-directory parent, producing a long security-stack
trace that initially looked like a database privacy regression. Renaming the
fixture to `attempts.db` made the unchanged production path pass; 18 focused
tests then completed green.

**What to do.** Keep temporary and fixture basenames away from Windows reserved
devices (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`), including
when an extension is present. When a Windows path unexpectedly canonicalizes to
`\\.\<name>`, inspect the basename before weakening private-path validation.

---

## Credential-presence probes must never print regex captures, and live config reads still need an isolated home

**task-14811.5, 2026-08-10.** A PowerShell probe intended to print only the
names of configured providers reused the automatic `$Matches` variable across
regex operations. The later operation replaced the expected provider-name
capture, so the script printed three credential values instead. In the same
verification pass, a helper that did not instantiate the application database
still imported configuration code that ensured a chat-dictionaries directory
under the real user profile. Neither behavior was required to prove the
feature, and the exposed credentials had to be treated as compromised and
rotated.

**What to do.** A credential-presence probe should emit a fixed provider label
only after testing whether a value is non-empty; never print, interpolate, or
retain the matched credential and never depend on PowerShell's process-global
`$Matches` state for the label. For a real-provider smoke, point
`TLDW_CONFIG_PATH` at the existing config only for read access, set `HOME` to a
validated scratch directory before Python imports the application, use an
explicit in-memory or scratch database, hash the real config before and after,
and remove the verified scratch path on exit. A helper that can make billable
requests must also require an explicit confirmation flag.

---

## Captured async exceptions retain non-serializable transport locals

**TASK-14811.6, 2026-08-11.** The full parallel CI suite intermittently exceeded
a fake WebSocket server's five-second receive allowance. Its handler stored the
original exception for a later test-side re-raise. That exception retained the
handler traceback, including the live `websockets.asyncio.server.ServerConnection`
local. pytest-xdist then failed while serializing the report with an execnet
`DumpError`, hiding the ordinary timeout behind an internal runner error on both
macOS and Ubuntu. The same signature reproduced on the latest `dev` baseline.
The first regression opened its own client connection and repeated the failure
when its test frame was serialized, so the durable regression had to exercise
the content-only exception copy without creating any transport objects.

**What to do.** When an async test helper captures an exception across a task,
thread, process, or worker boundary, do not retain its traceback-bearing object.
Store a content-only diagnostic exception (type name plus message), and test that
its traceback is absent without putting a transport in the regression's own frame.
Keep positive wire waits long enough for full-suite scheduling contention while
leaving negative "nothing arrived" grace windows deliberately short. Verify the
helper with the same xdist distribution flags used by CI; an isolated serial pass
does not exercise report transport.
## A forecast-equals-receipt governance test proves nothing about the backend it does not drive (TASK-14827, 2026-08-10)

**Incident.** The 14820-14826 arc rebuilt the Library ingest forecast so the
commit line, consent line, tooling fold and Start gate all derive from one
`IngestForecast`, and pinned it with
`test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder`: real
pre-flight, real submit, real DB, forecast counts asserted equal to the actual
job outcomes file by file. Strong evidence — for the LOCAL backend, the only one
it drives. In the same review round, TWO server-path divergences were found by
reading, not by the suite. (1) Local tooling gaps were subtracted from a
server-bound forecast, so five .mp3 on a machine without the audio extra read
"0 will import · 5 will fail (need tooling)" for a batch the server would have
transcribed in full. (2) An unsupported file was forecast "will skip" while
`build_server_ingest_kwargs` raised and the job landed as `✗ failed`. Both sat
inside the arc's own governed area, with 1,900 tests green.

**The second trap, which the first fix walked into.** The two backends refuse
DIFFERENT sets, so "ask the backend" is not a formality. Locally, unsupported
means `get_type_group(...) == UNSUPPORTED_GROUP`. The server additionally
refuses everything it has no media type for — raster images, deliberately left
server-unmapped — while NOT refusing a web page, because the submit path routes
pages to the clipper before the ingest-jobs mapping is ever consulted. A
predicate derived from either backend alone is wrong in both directions: reuse
the local verdict and images are promised as imports; ask
`server_media_type_for` alone and every server-mode URL import is condemned. The
fix asks the same functions the submit path asks, in the same order.

**The rule.** When a screen makes a promise about an outcome and more than one
backend can deliver that outcome, the governance test is per BACKEND, not per
screen. A second such test is cheap: keep everything real down to the narrowest
seam that cannot run in a test — here `TLDWAPIClient`, i.e. the network — so the
real request builder, the real response schemas, the real registry and the real
reconciler all participate. Bind every call to the stand-in against
`inspect.signature` of the real client method so the double cannot absorb a
drifting call site, and state in the test docstring exactly what the stub decides
(this one accepts everything handed to it, so it proves what the app SENDS and
what it refuses to send — never what a real server does with a file it received).
Anything the stub would have to invent is a fixture you must leave out and name:
this fixture holds no 0-byte file, because the app sends one and only the server
decides.

**The follow-through (TASK-14910, 2026-08-11).** That last sentence turned out to
be the finding, not the caveat. A fixture you cannot write because the outcome is
unknowable is usually pointing at a CLAIM the product should not be making: the
same forecast counted that 0-byte file as a certain failure while admitting, one
segment later, that "server tooling isn't checked from here". The fix was not a
cleverer stub — it was to make the outcome knowable, by refusing to send a 0-byte
file at all (the client already knows why, the local backend already refuses one,
and the round trip buys nothing). The fixture then grew to hold `empty.txt`, whose
fate is now decided entirely by code the test runs for real; the stub never sees
it. So: when a governance test has to leave a case out, name it AND file it — the
gap is evidence about the product, and the honest close is usually to remove the
unknowability rather than to keep the fixture short forever.

---

## An `await event.wait()` on a fire-and-forget task hangs on the task's OWN exception — and the timeout dump names nothing (2026-08-10, TASK-3316/TASK-15104)

**Incident.** `Tests/UI/test_screen_navigation.py::test_file_notes_collections_
source_transition_blocks_mutation_through_recompose` hung forever on dev, so
pytest-timeout's `thread` method killed the whole pytest process and **every
test after it in the file never ran** (the task-1466 class). The test drives the
screen coroutine as a background task and then waits for a signal only that
coroutine can set:

```python
source_switch = asyncio.create_task(screen._select_library_rail_row(...))
await sync_returned.wait()          # unbounded
```

The coroutine never got there. Its stub `_flush_library_note_save` returned
`None`, matching the seam's `-> None` signature at the time the test was written
(`eb036a6a1`, 2026-07-27). PR #1439 (`6b4ccf475`, on dev 2026-08-08) retyped the
seam to `NoteFlushOutcome` and made the caller read `note_flush.kind` — so the
awaited path died on `AttributeError: 'NoneType' object has no attribute 'kind'`
one line in. A `create_task` result nobody retrieves swallows that exception
whole, and the signal became unreachable. **This predates the media-ingest arcs:**
running the test from `git archive` copies of both sides of that merge is
decisive — `86e511781` (its first parent) *1 passed in 3.64s*, `6b4ccf475`
*hang → process killed*.

TASK-2512 rediscovered the same harness drift when its repository run reached
about 83% and stopped advancing. The exact node timed out at 300 seconds on
both the feature branch and clean `origin/dev` `8d764c03`; TASK-15104 then
changed only the stub to `NoteFlushOutcome(PERMITTED)`. The exact node passed
in 1.08 seconds and its eight-node adjacent group passed in 2.18 seconds. That
branch/clean-dev comparison plus the typed-stub mutation proved a shared test
harness defect, not an MCP runtime regression.

**The trap that cost the most time.** task-1466's advice, "the timeout stack dump
names it", does NOT hold here. The dump showed only `MainThread` idle in
`selectors.select` under `run_forever` — the test coroutine is *suspended at an
await*, so it has no frames on any thread stack. The dump is silent about which
`await` and silent about the exception. The only thing that talks is bounding the
wait and asking the task: `await asyncio.wait({waiter, task}, timeout=...)`, then
`task.result()`. That turned a 300-second process-killing silence into
`AttributeError` in 0.9s.

**The rule.** A test that awaits a condition a *background task* must produce has
two failure modes fused into one hang: the task can return early, and the task
can raise. Bound the wait at its source and settle both — if the task is done and
the signal is not set, re-raise its exception (or report the silent early return)
instead of waiting. The bound is not belt-and-braces; it is the only thing that
converts "the run died and you do not know why" into a named failure. Mutation
proof for this one: restoring the stale `return None` with the bound in place
fails in 2.2s naming `AttributeError`, where before it hung.

**Two corollaries, both paid for here.**
1. *A monkeypatched stub is a copy of a contract with no type checker behind it.*
   Nothing warns when production retypes the seam; the stub keeps the old shape
   and fails at the call site, which may be somewhere nothing is watching. When
   you change a seam's return type, grep the tests for stubs of that name — the
   same PR left `test_screen_navigation.py` with three of them.
2. *You cannot know a file's pass count while it contains a hang.* Bounding this
   one test took the file from "died at test 12" to `126 passed` — and revealed
   two more hard failures (`_library_note_dirty` became a read-only property; the
   prompt editor's guarded exit now needs a running App) that had been invisible
   for days behind the hang, plus one load-sensitive flake. "That file is green"
   was never true; it was never finishing.

---

## A test double mirroring the BASE class is blind to the SUBCLASS the app actually runs (TASK-14751, 2026-08-10)

**Incident.** TASK-14751 added a keyword-only `keyword_source_types` kwarg to
`RAGService.search` and had the Library's hybrid arm pass it down. Fourteen new
tests over real media + ChaChaNotes databases were green, the Library suites
were green, 413 targeted tests were green. Then the informational gated run
(`RAG_EVAL=1 pytest Tests/RAG_Eval/`) crashed three tests with
`TypeError: EnhancedRAGServiceV2.search() got an unexpected keyword argument
'keyword_source_types'`. `EnhancedRAGServiceV2` is the class the Library
actually resolves at runtime, and it overrides `search()` with an explicit
signature, so it does not inherit new base-class kwargs. Every double in the
unit suites (`_ProfileRagService`, `FakeRagService`, the new spy) was written
to mirror `RAGService.search` — the base — so nothing in ~2,500 unit tests
could see it. The same class's docstring already warned about this for
`metadata_allowlist`, and the warning was not enough to prevent the repeat.

**The rule.** When you add a parameter to a method, `grep` for overrides of that
method (`def <name>(` in the package) before you add the caller, and add at
least one test that drives the class the PRODUCTION resolver returns, not the
base class your doubles copy. A doubles-only suite pins the contract you wrote
down, never the object graph that runs. Corollary: an "informational, expect
no metric movement" gated run is not ceremony — this one earned its slot by
being the only thing in the repo that built the real runtime class, and it
caught the defect as a hard crash, not as a metric delta. (After the fix its
deltas were +0.000 in all three modes exactly as predicted, which is why the
crash, not the numbers, was the whole value of running it.)

---

## Retuning a numeric constant obliges you to grep its LITERAL VALUES, not just its symbol (TASK-4110, 2026-08-09/10)

**Incident.** Shipping the RAG hybrid-fusion `rrf_k` default `60 -> 5` took
three separate rounds to find every place the old value had leaked into prose,
because each round only swept one surface:

- **Task 5 round 3** grepped the SYMBOL (`rrf_k`, `DEFAULT_RRF_K`,
  `resolve_rrf_k`) and found four docstrings/comments still asserting `k=60`
  verbatim.
- **Task 6** grepped a downstream literal VALUE, `0.016` — the fused-score
  ceiling (`1/(60+1)`) that `k=60` produces arithmetically — and found a fifth
  location, a module docstring that named no `rrf_k`-family symbol at all,
  only the number the old constant happened to produce.
- The **final whole-branch review** found the seventh and eighth — two
  docstrings (`Event_Handlers/Chat_Events/chat_rag_events.py`,
  `RAG_Search/pipeline_builder_simple.py`) whose precedence-chain prose read
  "... -> active profile -> 60" — that neither earlier grep could have caught,
  because a bare literal `60` sitting inside an English arrow chain matches
  neither a symbol grep nor a `0.016`-shaped value grep.

Eight stale locations, three different grep strategies, three review rounds,
on a value everyone involved already knew had been retuned.

**What to do.** When a numeric constant is retuned, a symbol-only grep is not
a complete sweep. Enumerate every SHAPE the old value can still appear in and
grep each one separately: the symbol itself; any literal downstream
arithmetic consequence the docs may quote (a derived ratio, a ceiling, a
percentage — here, `0.016`); and the bare literal in comparison/precedence
prose ("-> 60", "an order of magnitude below X", "defaults to 60"). A
docstring can assert a stale number while never naming the constant that used
to produce it — that is exactly what lets it survive a symbol-only grep, and
exactly why an inline-literal arrow chain needs a human reading the prose, not
a tool, to catch on the first pass.

**The same class again, one level up: a stale VOCABULARY, and a sweep that
stopped one file short — twice in one review round (TASK-15700,
2026-08-13).** The keyword leg's default MATCH construction moved
`and_stopword_trim -> and_then_prefix`. The implementer swept and corrected
the affected prose; the review then found **two Importants that were both
twins of corrections already made elsewhere** — `_is_fts5_stopword`'s
docstring still said the list runs on every default search (the module
comment 3,380 lines above had already been rewritten to say the opposite),
and a test's property (b) still called the all-primary case "EVERY sub-leg
under the shipped `and_stopword_trim`" (the production docstring one
directory away had already been fixed). Re-sweeping the full RAG scope
mechanically then found **two more sites the review had not listed**. And a
third shape survived all of that into the closing task: the phrase "this
SPARSE **49-document** corpus", copied from a pre-P2ab README paragraph into
**three** newly written files (`config.py`, `rag_service.py`, a test
docstring) — a corpus that has held **172** documents since 2026-08-11, so
the number qualifying the arc's own headline cost figure was wrong in every
place the arc itself had written it.

Two additions to the rule above, both cheap:

- **Sweep for the VALUE *and* the VOCABULARY.** A retuned enum-like value
  (`and_stopword_trim`) leaves the same debris a retuned number does, plus a
  second kind: prose that describes what the old value DID ("drops function
  words", "never runs a second query") without naming it. Grep the old
  identifier, then grep the old behaviour's distinctive phrases.
- **Fix by re-sweeping the whole scope, never by patching the flagged
  lines.** Every finding in this incident was a *class* with more members
  than the reviewer listed; patching the reported line and stopping is what
  produced the twins. After the last edit, re-run the grep over the full
  scope and read every surviving hit aloud as a claim about today — the four
  that survived here were all correct historical statements, and knowing
  that is the difference between a clean sweep and an unfinished one.
## A gate with several conditions can close for the WRONG one — open the others or the test pins nothing (TASK-14911, 2026-08-11)

**Incident.** `start_enabled` on the Library ingest canvas is a conjunction:
registry present AND media DB present AND a non-blank path AND nothing-importable
false AND no option errors AND no path error. The new test staged a folder of
images in server mode and asserted `state.start_enabled is False` — the defect
being that it was `True`. The first run of that test, written BEFORE the fix,
reported the gate already closed. Not because the gate worked: the shared screen
harness (`Tests/UI/app_factory.py`) leaves `media_db = None`, so a *different*
conjunct was False the whole time. Had the test asserted only `start_enabled`, it
would have passed against the unfixed code, pinned nothing, and stayed green
forever after someone later broke the backend-aware gate.

It was caught only because the same test also asserted the specific flag the fix
introduces (`selection_has_nothing_importable`) and the gate line's own wording —
those two went red while the boolean did not.

**What to do.** For any multi-condition gate:

1. In the fixture, explicitly OPEN every condition except the one under test
   (`app.media_db = SimpleNamespace()` here), and say in a comment why — a shared
   harness's defaults are not neutral.
2. Assert the REASON, not just the closed state: the flag the fix sets, and the
   user-visible sentence naming it. A boolean shared by six causes cannot
   discriminate between them.
3. Run the test before the fix and READ which assertion fails. "It failed" is not
   enough when the boolean can fail for free.

---

## Owner state is not evidence that retained rows are mounted (TASK-14904, 2026-08-10)

**Incident.** Session Git workspace tests waited until the owner-published row tuple
had two entries, then programmatically pressed Stage or Unstage. The retained
`ListView` was still clearing and mounting that generation. A second status render
could cancel the row worker mid-clear, leaving `Pilot.pause()` waiting 30 seconds on
pruned child message pumps even though the action service and owner state were
correct. Adding one disclosure control changed timing enough to make the latent race
repeatable.

**What to do.** Treat the immutable model projection and the mounted row generation
as separate readiness boundaries. Disable row-derived mutations while rows are being
replaced, and make tests wait for model count, mounted count, and list visibility to
agree before pressing a row action. While a `ListView.clear()`/extend cycle is in
flight, poll service state with `asyncio.sleep`; `Pilot.pause()` deliberately waits on
every message pump and can turn transient child teardown into a harness deadlock.

---

## A compaction threshold is not a send-admission ceiling (TASK-14913, 2026-08-11)

**Incident.** The first Console memory release routed both an unknown automatic
budget and a threshold crossing with no replaceable units to a single
"compaction cannot run safely" blocker. The exact prepared request had not
exceeded a known provider input ceiling, but the default `Ask` policy made every
send on an unrecognized model fail before dispatch. A user-supplied bounded
budget could still reach the same false blocker when no older complete unit was
eligible for replacement. Policy, lifecycle, serialization, and modal tests all
passed because none asserted the ordinary send outcome for an unavailable
compaction decision.

**What to do.** Keep the compaction high-water threshold and provider send
admission as separate decisions. `UNKNOWN_WINDOW` and `NON_COMPACTABLE` mean
"do not compact now"; they block the message only when the immutable prepared
request also proves `known_overflow`. Test the complete decision cross-product:
unknown model with inherited automatic budget, unknown model with a bounded
custom budget, and known mandatory-material overflow. A settings-only assertion
does not prove that the next send consumes the saved policy correctly.

---

## A reviewed-safe label needs adversarial provenance evidence (TASK-3796, 2026-08-10)

**Incident.** TASK-3796's exhaustive ledger initially classified 199 diagnostics as
private and froze 324 as reviewed-safe. Final review found that
`general-2efc909241862caf` rendered `event.get("type")` from a Cohere streaming
response. The value looked like bounded status metadata in the source review, but an
unknown provider event can choose that string. A sentinel passed through the real
`summarize_with_cohere()` generator and fully consumed the response; it reproduced the
provider-controlled value in captured diagnostics. Restoring the historical
interpolation made that sentinel fail, and the corrected ledger became 200 private /
323 reviewed-safe.

**What to do.** Do not freeze a dynamic diagnostic merely because its field name
sounds operational. Prove where the value originates. For response events, config
values, and adapter metadata, drive an adversarial distinctive value through the real
production function and capture the actual logger path. A reviewed-safe classification
is evidence only after the sentinel proves the producer—not the reviewer—bounds the
value.

---

## A boundary projection must reject fields it does not understand (TASK-3796, 2026-08-10)

**Incident.** TASK-3796's first permanent manifest-boundary test rebuilt an allowlist
projection from known top-level fields and excluded the derived `task_492_calls`
summary. That made two checks look stronger than they were: a newly introduced
top-level section disappeared before hashing, and a self-consistent generator could
change both the owner counts and their derived summary while the summary-delta
assertion still agreed with itself. Review mutants adding an `unreviewed_section` and
forging `task_492_calls` exposed both gaps. The repair normalizes a deep copy of the
complete manifest, masks only the two explicitly owned count/digest fields, and
recomputes the TASK-492 summary independently from every owner row.

**What to do.** For a governed artifact, project by copying the whole schema and
masking the narrowly authorized fields; do not reconstruct an allowlist of fields to
retain. Validate derived totals from their primary rows before normalization. Then
mutate an unknown field and mutate the derived value independently: both must make the
boundary test red. Equality between two values produced by the same regeneration path
does not independently validate either one.

---
## A test written in the same PR as the change it pins can be born red — and then that PR's own omission ships (TASK-14920, 2026-08-11)

**Incident.** `7dbbc401b` (TASK-2154, FB-07) moved every Console "Save as..."
confirmation from `severity="information"` to `severity="success"` — and shipped
`test_console_save_as_savers_confirm_at_success_severity`, which asserts all four
destinations do so. It missed the Chatbook destination. Nobody noticed for four days,
because the test it shipped alongside called `console._save_console_message_as_media(...)`
— a seam decomposition wave 3 (`391b7bf69`, merged ~9 hours earlier the same day) had
already moved onto `ConsoleMessageController`. The test raised `AttributeError` on line
one of its four calls and **never once ran green**; the same PR's
`test_console_settings_save_fires_success_toast` was born red the same way against
wave 2's `_ensure_active_console_session_settings`. Repointing them at their controllers
made the first fail on exactly the assertion the PR had forgotten to satisfy — proving
the never-green test had been masking a real, shipped defect the whole time.

Verified with `git archive 7dbbc401b | tar -x` into a scratch tree: both tests fail at
the very commit that introduced them.

**What to do.** A test added in the same change as the behaviour it pins deserves the
same "confirm it could have gone red" treatment as a guard: run it, read the pass count,
and read WHICH assertion moved. And when a decomposition wave moves a seam, the delegator
shim it leaves behind for the "direct-call convention" is only as good as its coverage —
wave 3 kept `_save_console_message_as_note` and `_save_console_message_image` on
`ChatScreen` and silently dropped the other three savers. `Tests/UI/test_console_moved_seam_guard.py`
now checks that shape mechanically (AST + the live classes), because "the AttributeError
is loud" is only true for tests that anything actually runs.

---

## Production's broad `except` turns a stale test double into an INVERTED contract (TASK-14920, 2026-08-11)

**Incident.** `a6cc05d8b` ("seed dynamic character chat templates") moved the character
handoff's greeting seam from `store.append_message(...)` to
`store.seed_character_roleplay(...)`. Six tests across two suites
(`test_console_native_chat_flow.py`, `test_personas_workbench.py`) drove that handoff
through hand-rolled store doubles implementing only `create_session` and
`append_message`. The handoff wraps its seed call in `except Exception: logger.warning(...)`,
so the double's missing method surfaced as a swallowed `AttributeError` — not as an error.
The tests did not blow up; they quietly started observing "no greeting was ever appended"
and their assertions (`identity_at_append is None`, `store.messages == []`) began pinning
the ABSENCE of the behaviour they were written to prove.

That is worse than the familiar "a fake written to match your call site" trap two entries
up: there the double agrees with a wrong assumption, here the double's *silence* is
laundered by production's own error handling into a false negative that reads like data.

**What to do.** When production calls a collaborator behind a broad `except`, a stub double
for that collaborator cannot report drift. Subclass the real collaborator instead and
override only what you need to observe:

```python
class _CharacterHandoffStore(ConsoleChatStore):      # real store, persistence=None
    def append_message(self, session_id, *, role, content, persist=False, **kwargs):
        self.identity_at_append = {...}              # observe
        return super().append_message(...)           # production behaviour intact
```

The greeting text then comes from production's own template expansion, so the assertion
`== ["Hello User, I am Elara."]` is a live end-to-end claim (mutation-checked: passing
`global_default="Zed"` turns it red) instead of a re-implementation of the thing under test.

## Two tests failing on the SAME missing class can have opposite causes — and the feature commit's own test diff is the authority (TASK-15121, 2026-08-11)

**Incident.** Two tests in `Tests/UI/test_console_native_chat_flow.py` went red on dev with
what looked like one symptom: the Console send button no longer carried
`console-send-blocked`. The obvious reading was a CSS-vocabulary rename — follow it in both
tests and move on. Both readings were wrong in different directions:

- `test_console_composer_stop_is_subdued_when_idle` mid-stream with an EMPTY draft: the
  button was still genuinely `disabled`, just for a different reason
  (`console-send-inactive`, the empty-draft gate) than the one pinned.
- `test_console_duplicate_send_during_stream_does_not_break_stop_control` mid-stream with a
  draft loaded: the button was `disabled is False` and `console-send-ready`. The single
  `composer.load_draft("second send")` between the two tests is the whole difference.

Had both been "fixed" as a rename, the second would have kept asserting a control was
unavailable when it is now deliberately available — the exact class of silent claim
task-14920 lost a real bug behind for four days.

Neither the class name nor the production code said which reading was right. What settled
it was the **test diff of the commit that caused it**: `git log -S 'send_blocked = not
queue_presentation.send_enabled'` named `14cc326e4` ("feat(console): add visible prompt
queue"), and that commit's own diff to a SIBLING test file
(`Tests/UI/test_console_send_disabled_state.py`) rewrote the same assertions to the new
contract and renamed a test from `..._while_run_blocked_still_shows_feedback` to
`..._queues_draft_behind_accepted_run`. The author changed the contract deliberately
(ADR-046: "once accepted, the normal `Send` action becomes `Queue`") and updated one test
file, not two.

**What to do.** On a post-merge test failure that looks cosmetic, `git log -S` the assertion's
symbol, then read that commit's *test* diff before its production diff — an author who meant
the change usually left the new contract written out somewhere. And classify each failure
separately even when the symptom string is identical: same missing class, different inputs,
different truth. Where a pinned behaviour really was removed, say so in the test and pin what
replaced it (here: the duplicate send must still not start a second run, must land in the
bounded queue, and must not break Stop) rather than deleting the assertion.

---

## What is LISTENING on your machine can change what the test suite does (2026-08-11)

**The trap.** A suite can be environment-dependent in a direction nobody checks: not a
missing dependency, but an *extra* process. If production code probes a hardcoded
localhost port, then whether a developer happens to be running a local server decides
which branch the tests take — and the difference is invisible, because the escape's
failure mode is *success*.

**What happened.** task-15111. `Tests/UI`'s Console suites were opening real TCP
connections to `127.0.0.1:8080` and `127.0.0.1:11434` on every test that mounted
`ChatScreen` with an unconfigured provider. Mechanism: a blocking setup card starts
`_maybe_start_console_local_discovery` → `discover_local_servers`, whose candidate list
*always* leads with those two well-known defaults regardless of config, and
`probe_models_endpoint` builds a real `httpx.AsyncClient` when none is injected. A
record-only socket shim logged **386 connect attempts across 20 test files in the first
12% of `Tests/UI` alone** — on a machine that happened to have an `audiocpp` server bound
to 8080. Exactly one test in the suite had ever stubbed the `console_local_server_discovery`
seam; every other Console test fell through to the network.

Two things made it worse than "a stray GET":

- **The escape was self-concealing.** `_get_models_payload` ends in
  `except Exception: return None, "No models endpoint..."`. Blocking the socket, raising,
  timing out and answering all look identical from outside. A guard that only *raises* is
  therefore not enough — the guard has to **record** the attempt and something has to
  assert on the record, or the code under test simply eats it.
- **It could have POSTed.** `_configure_native_ready_console` points the Console at
  `http://127.0.0.1:9099` and several tests then drive a REAL send through
  `ConsoleProviderGateway`. On CI nothing listens, `_is_reachable`'s `GET /health` fails,
  and the send stops — so the suite looked read-only. Standing up a stand-in server on
  9099 and re-running one such test showed it going on to send **two POSTs to
  `/v1/chat/completions`** (streaming, then the non-streaming fallback) carrying the
  test's prompt. With a real llama.cpp on that port, `pytest` would have driven inference
  on the developer's server.

**What to do.** Default-deny sockets in the test configuration (`Tests/network_guard.py`,
installed at conftest *import* time so collection and post-test worker threads are covered
too), with an explicit `@pytest.mark.allow_network` opt-in, and fix the seams that build a
real client so the guard is a backstop rather than the mechanism. And when you want to know
what a suite would do against a live endpoint, do not reason about it: **bind a stand-in
server on the port and read what it receives.** Recording connects tells you a socket
opened; recording requests tells you the verb, the path and the body — which is the
difference between "reads something" and "writes to your server".

**Windows follow-up (TASK-15100).** The first task after the guard landed exposed a
platform boundary the original evidence missed: on Windows, Python 3.12's Proactor event
loop creates its self-pipe with the TCP fallback for `socket.socketpair()`, connecting to
an ephemeral `127.0.0.1` port. Because the guard is installed at conftest import time and
defaults to denied, `pytest-asyncio` could not even construct the event-loop fixture; every
async test failed in setup before the autouse fixture (and therefore before an
`allow_network` marker) could change the guard state. TASK-15100's focused local suites
produced twelve setup/teardown errors without executing one test. The task did not weaken
the shared guard as an unrelated drive-by change; its local-only verification process
temporarily emptied the guard's family set inside that pytest process, with every selected
test using SQLite or injected fakes.

**What to do on Windows.** A process-wide egress guard must distinguish the event loop's
loopback self-pipe from application egress *before* async fixtures are created (or use a
guarding layer that does not intercept the runtime's wakeup channel). Do not paper over
the issue by marking broad UI suites `allow_network`: that restores the exact application
escape the guard exists to detect. TASK-15458 replaced the temporary family-set workaround
with ADR-058's thread-local, dynamic `socketpair()` exemption: only the calling thread is
permitted while the captured real socketpair call is active, and `finally` restores nested
depth on success or error. Literal Windows commands
`python -m pytest Tests/test_network_guard.py -q`,
`python -m pytest Tests/Library/test_library_media_content.py -q`, and their combined form
passed without changing `_INET_FAMILIES`; focused tests also proved same-thread direct
egress stays blocked and recorded after an exception, while concurrent-thread egress stays
blocked and recorded during socketpair. Keep live/external clients stubbed.

---

## A capability decision is only as pinned as the final adapter kwargs (2026-08-11)

**The trap.** Checking a resolved provider/model/endpoint and then attaching a
provider-specific request feature does not prove that the checked endpoint is the one the
adapter will call. A lower layer may reload configuration or fall back to its own endpoint
after the capability decision has already been made.

**What happened.** task-15263 added strict JSON Schema enforcement for the visual
compaction evaluator's documented OpenAI GPT-4o routes. The initial implementation checked
`ConsoleProviderResolution.base_url`, but the prepared-request dispatcher did not forward
OpenAI's resolved base URL; `chat_with_openai` could therefore reload a configured endpoint
later. The report could have claimed `provider_json_schema` based on the official endpoint
while the final call went to a custom OpenAI-compatible proxy. Self-review caught the gap
before the PR. The evaluator-only prepared request now pins the checked endpoint into the
final adapter kwargs, and a test asserts both the immutable response format and exact
`api_base_url`. A mutation that removed the endpoint guard made the custom-proxy case fail.

**What to do.** For provider capability gates, test the final dispatched kwargs, not only
the resolver result or an intermediate request object. If a lower adapter can reload config,
make the capability-bearing request pin the checked endpoint (without changing unrelated
callers), and mutation-test the unsupported route so fallback labeling cannot silently
become an unsupported capability claim.

## A race a live replay cannot trigger is often a STATE you can construct deterministically (TASK-14903, 2026-08-10)

**Incident.** A live click killed the whole app once — `AttributeError:
'NoneType' object has no attribute 'region'` inside Textual's
`Screen._forward_event` text-selection begin, ~1s after a terminal resize.
THREE live replay attempts (same screen, same resize, same click) never
triggered it again, and the originating task shipped with the crash merely
noted. Task-14903 reproduced it 100% deterministically on the first attempt —
not by replaying the timing, but by reading the framework source to name the
intermediate state the race passes through (widget pruned from the DOM, parent
already `None`, compositor's cached map not yet reflowed) and then
constructing that state directly at the seam: `await widget.remove()` with no
subsequent pause (prune complete, reflow pending), then the MouseDown driven
through `App.on_event`, the exact call the live crash traversed. The
"irreproducible" race was a two-line setup once expressed as a state instead
of a schedule.

**What to do.** When a race defies replay, stop replaying. Read the code that
crashed until you can name the exact intermediate state the timing window
produces (here: three facts — `parent is None`, stale compositor map, event
dispatched between them), then build THAT state through the narrowest public
seams available and drive the same entry point the production path uses. A
reproduction that constructs the state is strictly better than one that races
the clock: it is deterministic, it documents the mechanism in its
preconditions (each setup line asserts one fact of the attribution), and it
pins the upstream behavior — if a dependency bump fixes the bug, the
state-construction test fails loudly and tells you the workaround can be
retired, which no timing-based replay could ever do.
## Moving a config read onto the app-config snapshot silently DEFAULTS it in every `_build_test_app` test — including passing ones (TASK-15210, 2026-08-11)

`ChatScreen._maybe_auto_retrieve_for_send` used to read the auto-RAG toggle live via
`get_cli_setting("chat_defaults", "rag_auto_retrieve_on_send")`. Task-14803 (commit
`5be9e6a04`) moved that read onto the frozen per-turn `ConsoleTurnExecutionContext`, whose
`rag_defaults` are built from `app.app_config`. Both sources agree in the shipping app.
They do not agree in `Tests/UI`.

`Tests/UI/app_factory._build_test_app` patches `tldw_chatbook.app.load_settings` to return
a synthetic `{"tldw_api": ..., "first_run": ...}` — **no `[chat_defaults]`, no `[console]`**.
Production then behaves *correctly*: `_provider_readiness_app_config` only re-sources from
`load_settings()` when the snapshot carries the `general`/`logging` markers only a real
disk load emits, and this one does not, so it hands back the synthetic dict verbatim. Net
effect: `save_setting_to_cli_config(...)` still writes the toggle, `get_cli_setting` still
reads it True, and the code under test sees False.

Measured, not inferred: instrumenting one mounted test printed
`get_cli_setting=True` / `app_config chat_defaults.rag_auto_retrieve_on_send='MISSING'` /
`resolved ctx rag_defaults={'auto_retrieve_on_send': False, ...}` in the same run. The live
app assigns `self.app_config = load_settings()` (app.py), whose result carries both the
toggle and both markers — so the shipping path was fine and only the harness was blind.

**The part that cost the most.** One test went red and was triaged. Its sibling,
`test_send_proceeds_when_auto_retrieve_fails`, stayed GREEN — because with retrieval never
firing, the exploding backend it installs is never called and the test degenerates into
"an ordinary send works". A moved read does not announce itself by failing; it can just as
easily hollow out a passing test, and nothing in a green run points at it.

**What to do.** When you move a read from a live settings accessor onto a snapshot,
grep the tests that *enable* that setting and check they enable it through the new source —
a `save_setting_to_cli_config` + `_build_test_app` pair no longer reaches the code. Give
the mounted test the app's real shape (`app.app_config = load_settings()`, exactly what
`app.py` does) rather than teaching the product to fall back. And any test whose subject is
"X still works when Y fails" should assert **that Y was actually attempted** — here,
`exploding_search.await_count == 1` — or it cannot tell "handled" from "never happened".

## Textual component CSS must be proven on the concrete widget class (TASK-1990.1, 2026-08-11)

**Incident.** TASK-1990.1 extended Textual's Markdown block classes through a
Python mixin and declared the new inline component names on that mixin. Pure
parser tests passed because the expected component spans were present, and the
TCSS bundle compiled, but a real compositor test painted narration, speech,
action, and emphasis identically. Textual's class construction had populated
the concrete block's internal component registry before the ordinary mixin
attribute could affect it; type-oriented TCSS also needed a stable class hook
because the rendered blocks were concrete subclasses.

**What to do.** When adding Textual component styles through subclasses, declare
`COMPONENT_CLASSES` on every concrete widget class and give the widget a stable
CSS class selector. Then verify both `get_component_rich_style()` and final
compositor segments. Span-level or stylesheet-compilation tests alone do not
prove that Textual registered or painted a component.

## A report's "already handled / out of scope" is an UNTESTED CLAIM (supervisor-fleet PR 3a-1, 2026-08-11)

**Incident.** PR 3a-1 Task 5 gave a background sub-agent its own wall-clock ceiling and
reported the containment story in two halves. It **mutation-tested the TIME half** and was
right. It stated the **COUNT half** — "aggregate live children are still bounded by
`[agents] max_live_subagents`" — from *reading the code next to it*, and wrote that into the
report and a docstring as settled. The review then proved by **execution** that it was false:
two consecutive `run_turn` calls each spawning two blocking children ran **4 simultaneously
against a configured cap of 2**, because `run_turn` built a brand-new `FleetCoordinator` every
call, and Console built a brand-new `AgentService` per `run_reply` and injected no coordinator
at all. Before this PR the bug was structurally impossible (children could not outlive their
turn), so the claim had been true right up until the change that broke it — which is exactly
the shape that survives a careful read.

That claim had already propagated: the plan's own seam map said `FleetCoordinator` was
"already reusable, no reset, never pruned", Task 5 relied on it, and the fix (a
per-conversation coordinator owned by the bridge) had to be a whole extra task. The retraction
is now pinned by `test_live_children_are_not_capped_across_turns` so it cannot be silently
re-assumed.

**This was the third instance in one programme**, all the same shape — a confident reading
stated as a finding:

1. A "vanishing row" window a fix was ordered for; measurement showed it was **sub-millisecond
   against a 200ms poll**, i.e. unobservable.
2. A run-log "closed writer" diagnosis, **wrong twice**: the records were being *misfiled into
   the next turn's tree* (not dropped), and `close()` was never a barrier at all — it fsyncs
   and returns without clearing `_active`, and `append()` opens its own handle per record.
3. This one.

**What to do.** When a report says a risk is *already handled*, *unreachable*, *out of scope*,
or *unchanged by this task*, treat it as a hypothesis until a test executes it. Write the test
that would go red if it were false — and prefer probing with a recording double over reading
the call path, because the two failures above were both found by a probe and missed by a read.
A dismissal is a claim about behavior, and behavior is the one thing reading cannot establish.
---

## A mechanism sentence is an ORACLE — read your prose against your own tables (TASK-15020, 2026-08-11)

**Incident, twice in one arc, and the second time the prose had already
shipped.**

1. **Paper arithmetic the code refutes by 1 ULP.** Task 6 measured the RAG
   eval's scoped category flipping 0.000 -> 1.000 and wrote the mechanism
   into `golden.toml`, `README.md` and a test comment: the FTS-only row
   "exactly ties" the semantic leg's rank 9, so the tie-break convention
   decides placement. False as the shipped code evaluates it.
   `reciprocal_rank_fusion` computes `(1.0 - alpha) * fts_rrf`, and
   `1.0 - 0.7` is `0.30000000000000004`, so the FTS-only row scores exactly
   `0.05` against the semantic row's `0.7/14 = 0.049999999999999996` — a
   **strict win by 6.94e-18**. The tie-break never runs. The paper form
   `0.3 * (1/6)` IS bit-identical to the semantic value, which is where the
   phantom tie came from. The *same arc's predecessor* (the weighting arc)
   had already learned a 1-ULP lesson; it recurred one arc later, in prose.
2. **A claim contradicted by a table in the same document.** The same
   section said the class is "FTS-only" and would read 0.000 at the old
   `rrf_k=60`. Re-running the counterfactual gave **0.286**: two of the
   seven targets sit at vector rank 12 and 20, inside the over-fetched
   pool, and reach rank 1 at `rrf_k=60`. The author's own rank vector
   `(3,4,9,9,9,9,9)` could never have come from FTS-only arithmetic — the
   contradiction was already printed above the sentence.
3. **A filed task pointing its own fix at the wrong lever.** Task 7 filed
   TASK-15400 blaming the keyword leg's silence on function words in an
   implicit-AND MATCH. Measured across all 60 golden queries: a
   stopword-trimmed AND rescues **1 of 40**; OR-of-tokens rescues **34**.
   The dominant cause is AND-strictness over CONTENT words — visible in the
   author's own token table, in the same report, unread against the
   author's own prose. A filed task is an oracle for whoever implements it,
   and as filed it would have sent them at a 1-in-40 lever.

**What to do.** Treat any sentence asserting a MECHANISM — in a docstring,
a fixture comment, a README, a test comment, or a filed task's description
— as an assertion that must be checked against the running system, at the
same standard as an `assert`. Specifically: never state a numeric mechanism
in paper arithmetic; **read the provenance the engine already records**
(here, `metadata["hybrid_fusion"]` carried `fts_rank`, `vector_rank` and
the fused scores all along). And before shipping an explanation, read it
back against the tables in your own document — in all three incidents the
refuting data was already on the page. Distinct from the stale-prose trap
(see "Retuning a numeric constant obliges you to grep its LITERAL VALUES"):
that prose went stale, this prose was wrong when written.

---

## A declared divergence no test can distinguish from its own removal is a comment, not a decision (TASK-15020, 2026-08-11)

**Incident.** Task 8 made the Library RAG window's depth follow the active
profile, and deliberately kept one difference from the Console seam that
now shares its resolution: the window clamps a >50 profile down to
`LIBRARY_RAG_TOP_K_MAX`, while Console stays uncapped. It was stated in the
report, in the code comment, and covered by a test — of the *clamped* arm
only. The reviewer mutated the SHARED seam to clamp unconditionally
(`min(value, LIBRARY_RAG_TOP_K_MAX)` in `library_rag_profile_top_k`),
erasing the divergence outright. **199 tests stayed green.** A difference
the author had chosen on purpose could be deleted by anyone, at any time,
with the whole suite agreeing. The fix was one test asserting BOTH arms
together — profile 100 gives Console 100 and the window 50 — after which
the reviewer's exact mutation reds precisely that test (1 failed / 153
passed).

**What to do.** When you deliberately make two call sites behave
differently, the pin is not "test the interesting arm" — it is **one test
that asserts the pair**, so the assertion states the DIFFERENCE rather than
one of its sides. Then mutate toward *sameness* (make both arms agree) and
confirm red; the usual mutation habit of breaking the guarded behaviour
misses this class entirely, because unifying two arms breaks neither arm's
own test. Applies to any intentional asymmetry: a clamp on one path, a
stricter timeout for one caller, a feature gated in one surface and not
another. Sibling of "Mutation-test every guard you add" and "A guard test
must be PROVEN to discriminate", with the twist that here the thing left
unpinned was a DESIGN DECISION, not a behaviour.
## Holding ONE database instance turns an intermittent schema-cache race into a permanent one (TASK-15463, 2026-08-11)

Caching `SubscriptionsDB` instead of rebuilding it per service call (a ~52-statement
`executescript` per call, ~85x the cost of a held instance) made two Watchlists UI tests
fail deterministically with `sqlite3.OperationalError: no such table: subscription_items`
— on a table that `sqlite_master`, queried microseconds later on the SAME connection,
listed. An immediate retry of the identical UPDATE succeeded.

A timestamped probe over `SubscriptionsDB.__init__` / `_get_connection` explained it:

```
0.0614  INIT app instance      (main thread)
0.4502  INIT second instance   (FTS-backfill worker thread) -- _initialize_schema
0.4511  CONN opened on the app instance, by an asyncio.to_thread worker   <-- inside that window
0.6884  second instance's _initialize_schema finishes (238 ms)
3.0311  that worker's UPDATE: "no such table: subscription_items"
```

A connection opened while another connection is rewriting the schema caches a view without
the tables being rewritten. With a database rebuilt per call, that view lived for one call
and the next call built a fresh connection — so the defect surfaced only as an
*intermittent* flake, already documented in `Tests/UI/test_watchlists_inspector.py` as
"self-healed on an immediate retry". Hold the instance and the poisoned connection lives as
long as the thread does: every write that lands on it fails.

The fix was to remove the second `_initialize_schema` (the FTS-backfill worker now shares
the app's one instance — thread-local connections are exactly what makes sharing the
*instance* safe), not to add a retry.

**What to do.** Before caching any long-lived DB handle, find every OTHER construction of
that DB class against the same file — each one re-runs schema setup, and any connection
opened during it can be born stale. And treat a documented "it self-heals on retry" flake
as a live bug with a shortened fuse: it is one held connection away from being permanent.
Probing the mechanism cost ~20 minutes (init/connection timeline + one retry inside the
failing call); guessing at "sqlite locking" would have cost far more and fixed nothing.

## A "we tried this and it broke X" comment is dated evidence, not a standing constraint (TASK-15454, 2026-08-11)

`ConsoleWorkspaceContextTray.sync_state` carried a long, careful comment (TASK-251,
July) explaining that the obvious `if state == self.state: return` guard had been
implemented, had broken click targeting on grouped browser rows, and had been
withdrawn — naming the two tests that failed. A separate test pinned the
unconditional recompose so nobody could quietly reintroduce it. Every downstream
comment in the file, in `chat_screen.py`, and in two test modules repeated the
conclusion: "an equality guard here is unsafe".

Re-guarding it started by reproducing that: apply the naive guard, run the two named
tests. **Both passed.** Widening to the whole 309-test `test_console_native_chat_flow.py`
plus `test_console_rail_sections.py` produced only the two tick-gating pins (which pin
the unconditional recompose itself) and one failure that also fails at HEAD. The
regression had been dissolved by later, unrelated work — most plausibly TASK-1900's
non-echoing search input and TASK-1191's collapse of the fit-pass from three deferred
hops to one.

Two things follow, and the second matters more than the first:

1. **Re-run the witness before designing around it.** Fifteen minutes of `git log -S`
   plus two test invocations turned "this is forbidden" into "this was forbidden in
   July, for a reason that no longer exists". Without that, the natural move is to
   design elaborately around a constraint that is not there — or, worse, to accept the
   comment and skip the work entirely.
2. **A dissolved regression is not a licence to do the naive thing.** The comment's
   *diagnosis* outlived its symptom, and it was the valuable part: state equality
   answers "does this widget REMEMBER this state", which is a different question from
   "is this widget SHOWING it". Those two came apart once and can come apart again.
   The guard that shipped therefore checks the second question directly — `compose()`
   records the row ids/keys it built; the guard compares that against the rows read
   back out of the live DOM — and both directions are mutation-tested (`return state
   == self.state` reds the safety tests; `return False` reds the skip tests).

**What to do.** Treat every "deliberately reverted / do not reintroduce" comment as an
experiment with a date on it. Re-run its named witnesses first; record the result in
the task either way. Then keep the diagnosis even when the symptom is gone.
## A synthetic test config lets tests pin states no user can reach (TASK-15270, 2026-08-11)

**The trap.** A test-app factory that hands the app a small hand-written config is not a
neutral simplification. Every default the real config file carries is *absent*, so the
code under test takes fallback branches, and assertions written against those branches
look like product contracts while pinning states the shipped template never produces.

**What happened.** `Tests/UI/app_factory._build_test_app` patched `load_settings` to a
three-key dict. `ChatScreen._provider_readiness_app_config` re-sources from
`load_settings()` only when the snapshot it was handed carries the sections a real load
always emits (`_CONSOLE_LIVE_CONFIG_MARKER_SECTIONS`: `general`, `logging`) — a
deliberate guard so an injected test config is never overwritten by the developer's real
one. The synthetic dict carried neither marker and no `[chat_defaults]`/`[console]`
section, so every mounted Console test read a `ConsoleTurnExecutionContext` frozen at
defaults. `test_send_proceeds_when_auto_retrieve_fails` was green for two months without
once calling the exploding backend it existed to exercise (task-15210).

Sourcing the factory's config from the real (per-test sandboxed) `load_settings()` turned
**31 green tests red across 6,016**, and the interesting part is *why* — almost none were
product regressions:

- **Arranged through a seam production does not read.** `console_image_view.
  _chat_images_config` prefers the raw TOML nested under `COMPREHENSIVE_CONFIG_RAW`
  whenever the snapshot has it. Four avatar tests set `app_config["chat"]` instead, which
  the shipping app would have ignored — they were pinning the fallback shape.
- **Passing on a fallback the template removes.** Three llama.cpp URL tests reached that
  branch only via `provider_config_key(...) or "llama_cpp"`; the template ships
  `[chat_defaults] provider = "OpenAI"`, so the fallback never fires for a real user.
- **Absence as arrangement.** Two "cli config fallback" tests asserted `"library" not in
  app_config` rather than arranging the absence they needed.
- **Copy for an unreachable state.** Several first-run/UAT replays assert "Choose
  provider" — the branch for *no provider selected*. A genuinely fresh install has
  `provider = "OpenAI"` and no key, so the product says "Set up provider". The tests
  described a clean run no user has.

**What to do.** Give the test app the same config source the app uses, sandboxed per
test (the root conftest already re-points `TLDW_CONFIG_PATH`/`HOME`/`XDG_*`), so what a
test persists is what the app reads. And when a test needs a state — no provider, no
`[library]` section, a feature off — **arrange it explicitly**; a state you inherited
from an empty fixture is a state you never chose, and the day the fixture gets honest you
cannot tell which of your assertions were ever real.

---

## A cancellation flag does not make check-and-commit atomic

**TASK-3401.20, 2026-08-10.** The first generated-video teardown fix checked a
screen-owned cancellation flag before publishing a managed file. Review found a
real gap between that check and the filesystem commit: unmount could win in the
middle, close the staged stream, and still leave a committed file without durable
message metadata. A second version shielded `asyncio.to_thread()`, but cancellation
of the awaiting coroutine did not stop the executor thread; releasing ownership in
the async `finally` could still close bytes the thread was using. Timing-only tests
missed both defects because they never proved which side had reached lock
acquisition or the final commit boundary.

**What to do.** Make the state transition linearizable: share one lock across the
final active check and commit, and cancel under that same lock. When blocking work
runs through `asyncio.to_thread()`, retain an explicit executor task and keep resource
ownership until that task actually finishes; coroutine cancellation alone is not
completion. Put durable commit-winning metadata finalization inside the shielded
unit, but leave stale-screen UI refresh outside it and normally cancellable. Tests
must use events or instrumented locks to force both cancel-wins and commit-wins
orders, including cancellation after commit but before metadata append; a sleep and
an assertion that “nothing happened yet” are not evidence of ordering.

## Returning from a Textual handler does not make its detached child cancellation-safe

**TASK-3402, 2026-08-11.** An H3 image edit originally awaited its whole operation
inside the real `Button.Pressed` handler. That kept the screen's MessagePump occupied,
so the visible Stop press could not run. Moving the operation into an app-owned task
fixed Stop responsiveness, but the old “outer cancellation drains success” test kept
passing without ever cancelling anything: it awaited the now-immediate handler return,
then released and awaited the detached operation normally. Directly cancelling the
actual operation task exposed the gap—its `asyncio.to_thread()` runner continued while
the owning coroutine removed the registry entry before durable settlement. App
shutdown had the same problem because Textual did not drain arbitrary tasks created
with `asyncio.create_task()`.

**What to do.** For a detached, app-owned operation, test and own cancellation at the
detached task—not at a caller that has already returned. The owned task must shield the
real runner, translate cancellation into the exact shared event, await the runner to
settlement, and only then re-raise. The application shutdown path must explicitly
cancel and drain those registered tasks before tearing down screens or persistence.
Use barriers to prove both success-wins (durable append exactly once) and
cancellation-wins (no card), and mutation-check that the test fails if shielding,
event propagation, or shutdown draining is removed.

---

## Keyboard focus does not prove a nested compact control is visible (TASK-15506, 2026-08-11)

**Incident.** TASK-15506 moved File Notes push provenance into a collapsed
`Collapsible` inside the push workflow's `VerticalScroll`. At 40x20, expanding
the disclosure and pressing Tab moved focus to the nested Endpoint details
button, so a focus-only regression passed. The button was still absent from
`Screen._compositor.visible_widgets`: Textual had scrolled only far enough to
show the disclosure's earlier content, leaving the focused action below the
fixed footer. A normal non-animated `scroll_visible()` call still stopped
short. Scrolling the exact focused descendant with `force=True` and
`immediate=True` brought it into the compositor deterministically.

**What to do.** For controls nested inside disclosures within a compact scroll
owner, assert both `has_focus` and compositor visibility. If framework focus
navigation leaves the control outside the viewport, handle descendant focus
at the narrow owning component and call `scroll_visible(animate=False,
force=True, immediate=True)` on the exact control. Do not infer reachability
from focus state or a nonzero layout region alone.

## An "indexed" query can still scan the table the index exists to avoid — and the plan assertion can miss it (TASK-15469, 2026-08-11)

TASK-15469 replaced a `metadata LIKE '%active_dictionaries%'` full scan of
`conversations` with a lookup over a trigger-maintained index table. The new query
joined the index table to `conversations`, and the test asserted
`"SCAN conversations" not in plan`. It passed. It proved nothing:

* The query aliases the table (`conversations AS conversation`), and
  `EXPLAIN QUERY PLAN` prints the **alias**, so the plan said `SCAN conversation` —
  which the assertion's literal `"SCAN conversations"` never matches. The test would
  have passed with a plan made entirely of full scans.
* And there really was one. SQLite's planner chose `conversations` as the outer loop
  of the second branch (`SCAN conversation` + a covering-index probe per row): a full
  scan of the very table the index was built to stop reading. Only the FIRST branch
  used the new index; nothing in the assertion covered the second.

It surfaced from a **timing** arm, not from the plan test: on a 10,000-conversation
DB, "used-by for a dictionary attached to nothing" measured 2.07 ms when it should
have been unmeasurable. `CROSS JOIN` (which pins the left table as the outer loop and
disables that particular join reordering) took the same arm to 0.00 ms and the whole
click's DB work from 7.8 ms to 0.54 ms. The plan test now asserts on the alias prefix,
asserts `conversations` is reached only by `SEARCH ... USING INDEX
sqlite_autoindex_conversations_1`, and asserts the plan is non-empty.

One more planner subtlety worth knowing: this project never runs `ANALYZE`, so the
planner works from default row-count estimates and reliably prefers the index. Running
`ANALYZE` on a small dev database flips it back to `SCAN` on the tiny index table —
so a plan captured on a hand-seeded 50-row fixture with `ANALYZE` is not the plan
production runs.

**What to do.** When the claim is "no full-table scan", (1) grep the plan for the
identifier the query actually uses — the alias, not the table name — and assert
positively on what SHOULD happen (`SEARCH ... USING INDEX <name>`), not only
negatively on what should not; (2) assert the plan is non-empty, or an empty result
satisfies every "not in" assertion; (3) check EVERY branch of a compound query; and
(4) keep one timing arm whose expected value is ~zero (a lookup with no hits) — a
scan cannot hide from that, and it is what caught this one.

---

## An absent catalog surface still needs its synthetic identity reserved

**TASK-13216, 2026-08-12.** The replacement Console task tools were correctly absent
from the external MCP and Hub inventories, and every literal-name absence test passed.
Review still found that an external MCP profile could use the reserved `__local__`
profile ID. Its projected Hub key then collided with the synthetic workspace provider's
`local:__local__` permission identity. The same review found a current guide describing
"session-todo tools" without any literal `todo_write` or replacement name, so the stale
name scan also reported clean while the documented inventory was wrong.

**What to do.** For a synthetic catalog namespace, reserve its normalized identity at
every ingress and projection seam: save/import, load, runtime composition, and raw
catalog conversion. Prove the derived permission key cannot be forged, while pinning
nearby valid and case-distinct IDs. For negative documentation contracts, pair literal
stale-name scans with an exact positive sentence describing the current boundary;
synonyms can preserve a stale claim without preserving any searched token.

---

## A property that holds "by construction" holds for the COMPONENT — measure it at the MERGE the requirement actually names (TASK-15400, 2026-08-12)

**Incident.** The MATCH-construction arc pre-registered a hard constraint:
whatever the keyword leg's expression becomes, the golden set's one
vector-blind fixture (`kw-plant-maintenance-record`, which only the keyword
path can find) must keep its hybrid rescue. The spec then argued that the
favourite candidate — `and_then_or`, "AND first, OR only when the AND
returns nothing" — satisfied it **by construction**: *a nonempty AND never
falls back*, so the fixture's own row can never change.

That premise is TRUE, and the sweep verified it directly: the notes
sub-leg's row for that query was still there, still stamped `and`, still
its sub-leg's rank 1. **The conclusion was false anyway.** Measured, the
rescue was GONE — the fixture dropped out of the fused top-10 entirely.

The guarantee was about a **sub-leg**; the constraint was about the **leg**.
`RAGService._keyword_search` merges its four source sub-legs with
`interleave_rankings` — a round-robin over sub-leg position. The media and
conversations sub-legs returned zero AND rows for that query, fell back to
OR, and injected ten rows each; media is first in the round-robin, so the
untouched notes row moved from leg rank 1 to leg rank **2**. Fusion consumes
*leg* rank: `0.3/6 = 0.0500` became `0.3/7 = 0.0429`, which loses to the
vector rank-11 row's `0.04375`. Nothing about the fixture's own row changed;
everything about its position did. The same displacement then decomposed a
whole category exactly — scoped recall 1.000 → 0.429 is the four
note-targeted scoped queries falling behind a media fallback row while the
three media-targeted ones keep rank 1 (3/7, the measured cell to the digit).

**Why it was caught.** Only because the constraint's probe was written at
the **output** — "is this document in the FUSED top-10" — rather than at the
component the guarantee described. A probe asserting "the notes sub-leg
still returns its AND row at rank 1" would have passed, and the arc would
have shipped a construction that silently deleted the one rescue the whole
fixture exists to detect.

**What to do.** When a design argues a property holds "by construction",
write down two scopes before believing it: **what object the guarantee is
about**, and **what object the requirement is about**. If they differ by
even one level of composition — sub-leg vs leg, row vs list, component vs
merged output, one writer vs the aggregate — the argument is about a
different thing than the requirement and proves nothing about it. Put the
acceptance probe at the level the requirement names.

Two corollaries worth carrying:

- **Any positional merge (round-robin, concatenation, fixed source order)
  makes every component's rank a function of every OTHER component's row
  COUNT.** A change that only ADDS rows in one place still re-ranks
  everything downstream. Treat "this change is additive" as a claim about
  the component, never about the merged list.
- **Necessary is not sufficient, and the margin is measurable.** Re-fusing
  the same run with the fixture restored to leg rank 1 and nothing else
  changed put it back at **slot 10 of 10** — so even fixing the merge
  rescues it with zero headroom. When you find the blocking mechanism,
  measure what fixing it actually buys before scoping the follow-up around
  it (this one became TASK-15700 with that number in its description).

---

## An infrastructure "agent stopped" report is a claim, not evidence

**PR 3a-2 Task 5, 2026-08-13.** The harness twice reported the Task 5
implementation agent stopped ("stopped by the user"; after a Claude Code
process restart it also refused to resume the agent — "won't be resumed").
The worktree was verified clean at the briefed HEAD, twice, and a fresh
agent was dispatched into `.worktrees/fleet-pr3a2` with the same brief.
Both reports described an agent that was never stopped: the pre-restart
Claude Code process had survived as an orphan — `ps` showed TWO
`claude --resume <same-session-id>` processes — and its subagent kept
editing, committing, and pushing. The fresh agent adopted a commit that
sat one beyond the briefed HEAD, then collided mid-edit: "string not
found" on a file the supposedly-stopped agent had rewritten seconds
earlier (mtime observed under a minute old). It halted itself and wrote
an incident file instead of the report
(`.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/task-5-incident-shared-worktree.md`).

The same orphan was still working at Task 6 close-out, HOURS later: three
fully-formed backlog task files appeared untracked in the worktree between
one of the Task 6 agent's commands and the next — filed under the exact
ids that agent's own sweep had just derived — followed four minutes later
by a commit made on top of the Task 6 agent's fresh commits. Two agents
were doing the same close-out in one worktree, neither told about the
other, because a "stopped" report had been believed twice.

**Why the clean-tree check wasn't enough.** "Verified clean at HEAD X" is
a statement about one instant. An agent alternates minutes-long quiet
stretches (gate batteries, provider calls) with bursts of writes, so any
point-in-time check taken during a quiet stretch passes.

**What to do.** Treat "the agent was stopped" as a claim to verify, never
a premise. Before dispatching into a worktree a reportedly-stopped agent
occupied, verify quiescence by OBSERVATION over a real interval — stable
`git log`, stable `git --no-optional-locks status --porcelain`, no fresh
file mtimes, held for minutes, not sampled once — and check `ps` for a
second `claude --resume <session-id>` process, the smoking gun in both
sightings. An OS process outlives the harness's account of it; only the
OS can tell you it is gone.

---

## A `0.00s` pytest summary is a usage error wearing a pass's clothes

**PR 3a-2 Task 5 gate verification, 2026-08-13.** A gate run passed
pytest a nonexistent path — `Tests/Chat/test_console_mcp_approval.py`;
the file lives in `Tests/UI/` — so pytest exited 4 after collecting
nothing. The habitual `| tail` read showed only "1 warning in 0.00s",
which was nearly recorded as an empty-but-fine run. The one line that
mattered — `ERROR: file or directory not found` — was at the HEAD of the
output, above everything tail kept. The same shape recurred within hours
in the same PR: a background gate run launched with a relative
`.venv/bin/python` that does not exist in that worktree "completed with
exit code 0" — the trailing `| tail -3` laundered the interpreter's
failure into the pipeline's success — and only READING the output file
revealed `no such file or directory: .venv/bin/python`. No tests had run
in either case, and both runs wore a green-looking coat.

**What to do.** A `0.00s` (or near-instant) pytest summary means nothing
ran: treat it as a FAILED gate, never a fast pass. Read the HEAD of the
output — usage errors print before the summary line, and exit codes
piped through `tail` are the pipe's, not pytest's. A gate passes only on
a READ, nonzero passed-count that matches the expected number; "no tests
ran", a count you didn't read, and a summary too fast to be real are all
the same verdict.
## A truncated pytest diff is not the diff (task-15512)

**Incident.** A failing `assert service.calls == [...]` printed its summary line
as `assert [{'include_ci...tions'), ...}] == [{'include_ci...tions'), ...}]`,
followed by one `At index 0 diff:` line that pytest itself had cut mid-value. I
read the visible fragment as a *scope* change and wrote that into the task file
as the diagnosis. It was wrong: the actual delta was `top_k` (5 vs 15), which
sat past the truncation point. The wrong diagnosis then travelled -- into a task
another person would have picked up, pointing them at "a search silently
widening its scope", which is a much more alarming and entirely fictional bug.

**Rule.** When a collection assertion fails, do not diagnose from the summary
line. Re-run that single test and read the full comparison, or print the two
values. The `...` in pytest's output is not an ellipsis for your benefit -- it
is hiding the part you need.

## Fixing a crash is how you find out what it was hiding (task-15512)

**Incident.** Three Settings tests failed with a timeout waiting for a toast. The
cause was a stdlib-logging call written in loguru's `{}` style, which raises
`TypeError` when the record is formatted; `_pytest.logging.LogCaptureHandler.
handleError` re-raises deliberately, so the Textual save worker died mid-save.
Fixing the log call made ONE of the three pass -- and the other two then failed
on their real assertion, which was a genuine product bug (pressing Save marks
untouched fields dirty-and-empty, and one of them aborts the save).

**Rule.** A crash in a code path masks every assertion downstream of it. After
fixing one, re-run and expect NEW failures rather than green; treat "same count
of failures, different reasons" as progress. This is the third time in this
programme that repairing a run-killing defect exposed defects nobody had counted
(see the hang-class sweep and the harness-config work).

**Corollary on severity.** The same log bug behaves differently in the two
environments: production stdlib logging *swallows* the formatting error and
carries on, so nothing was broken for users -- only the warning was lost. It was
tempting, and I did briefly claim, that a failing save in tests meant a failing
save in the product. Check which layer makes a failure fatal before assigning it
user impact.
## A DOM swap moved into a worker is invisible to `pilot.pause()` (task-15461, 2026-08-11)

**The trap.** `Pilot.pause(delay)` is `await self._wait_for_screen()` then
`await asyncio.sleep(delay)`. `_wait_for_screen` drains the **message pump** — it posts a
callback to every widget on the screen and waits for them all to come back. It knows
nothing about Textual **workers**. So a UI update scheduled with `call_next` is covered
by every `pilot.pause()` in the suite; the identical update scheduled with `run_worker`
is covered only by whatever wall-clock `delay` the test happened to pass.

**What happened.** Replacing Watchlists' whole-screen `refresh(recompose=True)` on a tab
click with a region-scoped swap also moved the swap from a `call_next` callback (which is
what `refresh(recompose=True)` is, internally: `_recompose_required = True;
call_next(self._check_recompose)`) onto the screen's existing surface-refresh drain, which
ran as `run_worker(..., group="wc_surface_refresh")`. Nothing about the swap's *duration*
changed — instrumented at ~250 ms for the Artifacts pane before and after — but the
suite's shared helper opens a section with `pilot.pause(0.2)`, and 250 > 200. Eight tests
in `test_watchlists_artifacts_pane.py` began failing with `NoMatches:
#watchlists-artifacts-pane`, **passing in isolation and failing in a full-file run**,
because the margin was machine load. Two more flipped between runs. It looked exactly
like flakiness and was not: it was a deterministic ordering change, mis-read as noise
because the symptom was load-dependent.

Scheduling the drain with `call_next` fixed all ten and cost nothing else — it is also
strictly safer than the worker it replaced, whose own comment explains that it needed a
private worker group so the screen's several `run_worker(..., exclusive=True)` call sites
could not cancel it mid-swap. A `call_next` callback cannot be cancelled by a worker at
all.

**What to do.** Before moving any DOM mutation onto a worker, ask what the tests (and the
app's own idle handling) actually wait for. `run_worker` is for *work*; the mount/remove
pair that lands its result belongs on the pump. And when a batch of tests starts failing
together in a full run while passing alone, do not reach for "flaky" — check whether the
change under review moved something out of what the harness waits on.

## A region factory that reads state before its `await` loses whatever lands in the gap (task-15461, 2026-08-11)

**The trap.** Textual's own `Widget.recompose` removes its children **first** and calls
`compose()` afterwards, so it always reads widget state on the late side of the yield.
Hand-rolled in-place swaps usually do the opposite — build the replacement first, so a
factory that raises leaves the old content standing rather than an empty box — and that
inversion opens a window: state read, `await remove()`, state changes, `await mount()`.

**What happened.** `watch_active_section` dispatches the new section's loader and the
region swap in the same breath. `WatchlistsWorkbench.refresh_region_content` calls the
region factory (which reads `self._loaded_rules`) *before* its remove/mount awaits. The
loader — an `AsyncMock` in the test, a fast local query in production — completed during
the removal, wrote its rows to the screen, then looked for its pane and could not find it:
the replacement existed but was not yet mounted. Result: an Alert-rules table that stayed
empty over a `_loaded_rules` holding the row, with nothing left to correct it. The
whole-screen recompose being replaced had never had the gap, purely because of Textual's
ordering.

**What to do.** When you replace a recompose with a hand-rolled swap, re-apply the state
*after* the mount (`_reseed_active_section_pane`) rather than trusting the read that
happened before it. Reactive assignments make the re-apply free when nothing moved, so
the cost is a few lines and the failure mode it closes is silent.

---

## A pathname stat and an open-handle stat need not expose the same native identity field (TASK-2062.1, 2026-08-13)

TASK-2062.1's local-GGUF admission passed on Linux and macOS but rejected an
unchanged file on Windows. CPython 3.12's Windows pathname `stat` compatibility
surface reports creation time through `st_ctime`, while `fstat` on the already
opened descriptor retains the file's ChangeTime. Comparing the complete tuples
made an unchanged pathname and its own open handle look different. The first
two native Windows runs also exposed test-only POSIX assumptions before the
real identity mismatch became visible.

The correction compares only fields with shared pathname/descriptor semantics
when proving the name still refers to the opened file on Windows, while keeping
the descriptor-to-descriptor recheck strict, including ChangeTime. Tests mutate
device, inode, mode, size, and mtime independently, and the exact three-OS lane
runs the Windows reparse and replacement cases instead of accepting skips.

**What to do.** For TOCTOU defenses, distinguish the two questions: whether a
pathname still names the opened object, and whether the opened object changed
after inspection. Do not assume every portable `stat_result` field has identical
meaning across pathname and handle APIs. Preserve strict handle rechecks, test
each stable identity field, and require native-platform evidence for filesystem
security claims.

**TASK-16230 follow-up (2026-08-14).** Host-independent Windows doubles initially
made the one-time Notes import reader look race-safe while its real `CreateFileW`
call still included `FILE_SHARE_WRITE`, and its pathname-to-handle check accepted
`st_ino == 0`. Review showed that a same-size rewrite with restored mtime could be
admitted, while a zero inode provides no promised file identity. The correction
denies write/delete sharing on source-file handles, keeps the directory-pin share
mode separate, and fails closed unless pathname and handle expose the same nonzero
inode for the same device. Test the native share-mode arguments and unavailable-ID
case explicitly; tuple equality alone does not prove the object stayed immutable.

## A whole-screen recompose is doing four things you did not ask it for

**The trap.** Converting `refresh(recompose=True)` to a region-scoped rebuild looks
like a pure narrowing: same content, fewer widgets. It is not. The recompose was also
providing services the new path silently drops, and none of them fail loudly.

**What happened.** Task-15475 (2026-08-11/13) converted four surfaces. Every one of
these was caught by an EXISTING test, not by reading the diff:

* **Mouse-capture release.** `BaseAppScreen.refresh`/`recompose` release
  `App.mouse_captured` before and after the teardown (task-627): an `Input` has no
  `_on_hide`, so a widget torn down while capturing leaves a dangling capture and
  every mouse click app-wide is silently swallowed from then on. A region swap tears
  widgets down too and got none of that. Now extracted to
  `release_mouse_capture_for_teardown` / `sweep_stale_mouse_capture` and called by
  both converted screens.
* **Callback ordering.** Textual runs a screen's recompose BEFORE its
  `call_after_refresh` callbacks, so "select the category, then focus a field in it"
  worked by construction. A region rebuild driven from a worker (or from the region's
  own `_check_recompose`) is a DIFFERENT pump with no ordering against the screen's
  callback list: the Speech deep link ran against a pane that did not exist yet and
  dropped its focus on the floor, leaving the user on `nav-home`. Follow-ups must hang
  off the swap itself.
* **Post-layout geometry.** Anything reading `virtual_size`/`container_size` (here an
  inspector overflow indicator) must still run after a REFRESH; read inline at the end
  of the swap it sees pre-layout zeros and renders the wrong state.
* **The repaint short-circuit.** `Widget.refresh(recompose=True)` returns before
  `_set_dirty`. Drop `recompose=True` and a plain reactive assignment now resolves
  `self.app` — which raises `NoActiveAppError` in every bare-screen unit test that
  sets that reactive. `repaint=False` restores the property honestly (the screen
  renders nothing from the value; its children do).

Also worth knowing: scoped is not automatically faster. Two separately-awaited region
swaps each drove their own layout pass and measured 105 ms against the 69 ms the
whole-screen recompose appeared to cost. Both numbers were wrong to compare — the Lab
frame defers its body mount OUT of the recompose, so that 69 ms excluded the expensive
half. Wrapping the swap in `self.batch()` (what `Widget.recompose` itself uses) took
it to 88 ms, and the honest end-to-end measure — trigger to content actually on
screen — was 325 ms before, 146 ms after.

**What to do.** Before converting a recompose, list what it did besides re-render:
grep the screen's `refresh`/`recompose` overrides, its `call_after_refresh` call
sites, and any geometry reads. Port each explicitly. Measure trigger-to-content, never
"time the two coroutines", and batch the swap.

**Review round 1 added three more, all measured, none visible in the diff:**

* **A "container" you empty may not be yours.** `remove_children()` on a frame region
  is only safe if the region holds nothing but mode content. `#lab-rail` and
  `#lab-inspector` each carry a frame-composed collapse header as their FIRST child —
  which is precisely why `LabScreen._populate_regions` APPENDS with `mount_all` and
  says so. The blanket removal destroyed both collapse buttons on the first click,
  permanently (no keyboard binding, no recompose left to restore them). If the
  existing code mounts with `mount_all` rather than replacing, that is a signal:
  something else already lives there.
* **Focus does not "stay put" when the widget under it is destroyed — it MOVES, to a
  neighbour you did not choose.** Both conversions landed the user on a collapse
  affordance one Space away from destroying their own context
  (`settings-category-group-domain-defaults`, `lab-rail-collapse`). Capture the focus
  token before a teardown and restore it by id, but defer the restore and yield to
  the rebuilt subtree — a freshly mounted widget may have focused itself ON PURPOSE
  (`ResultsGrid` does, so its advertised shortcuts work), and an eager restore wins
  the FIFO race and silently kills that.
* **`exclusive=True` is the wrong supersede primitive for a teardown.** It cancels the
  in-flight worker, and the cancellation can land inside `remove_children` — leaving a
  region emptied and never refilled when the superseding swap does not rebuild that
  same region, and skipping the post-teardown capture sweep. A lock plus a revision
  check supersedes just as firmly and lets the loser return before touching a widget.
  Accumulate any per-call flags (`rail_dirty`) across superseded calls, or the
  survivor silently drops the loser's work.

And one about the evidence itself: **a test that asserts on the nearest visible text
can be satisfied by a different code path.** Neutering the sync-rows region rebuild
left all six evidence tests green, because the assertion read a summary `Static` that
another path keeps current. Assert on the widgets ONLY the mechanism under test
writes, then mutation-check by neutering that mechanism.

## An absolute event-count pin records which side of a race the author's machine won (TASK-15458, 2026-08-13)

**Incident.** Task-15458's perf pin asserted `markdown_updates ==
[id(markdown_before)]` — "opening the media item parses the document exactly
once". It was written and verified on Windows, where it passed. On macOS it
failed 3/3 with `markdown_update_count=2`, and it had been red on `dev` from
the moment it merged. The count was not flaky, and it was not a platform quirk
of the test: it was reporting a real defect that the authoring machine happened
to hide. Opening a media item issues two `refresh(recompose=True)` calls — the
"Loading media…" one at click time, and one when the detail worker resolves.
Textual's `recompose()` awaits child teardown BEFORE it calls `compose()`, so a
worker landing inside that await gets picked up by the in-flight compose, and
the worker's own recompose then parses the whole 49 KB / 2,000-line document a
SECOND time. Windows lost that race the other way (both refreshes coalesced
into one recompose), so the same production code produced 1 there and 2 here.
A/B on the open click: 922/914/935 ms and 2 parses with the arrival recompose
unconditional, 710/730/841 ms and 1 parse with an identity guard on the
already-composed detail.

**What to do.** An absolute count over a window that spans a scheduling race
pins your machine's timing, not the contract. Two habits fix it. First, scope
the count to the interaction the claim is about — the sibling test in the same
file already did this (`parse_count_before_navigation = len(markdown_updates)`,
then assert no growth across the click), and it was green on both platforms
because the delta cannot absorb an unrelated race. Second, when you do want an
absolute count, first make it deterministic in PRODUCTION (here: the guard),
then pin it — a total that is only stable on one OS is evidence about the OS.
And treat a count that differs from the notes' recorded value as a defect
report until proven otherwise: the number was right, the code was wrong.
## When a screen really is widget-bound, COUNT widgets — a wall-clock A/B can't resolve the change (task-15462, 2026-08-13)

**What happened.** Profiling the Watchlists push turned up a genuine piece of waste: the
screen's `region_layout` reactive defaults to "nothing collapsed" while the shipped
first-run default collapses the RIGHT_RAIL, so every visit composes the expanded
Inspector rail and `on_mount` immediately swaps it for the one-line collapsed header. A
prototype removing the swap was measured against dev the obvious way — run the probe
process on dev, then run it with the fix, compare medians. It reported **35% faster**.

That number was an artifact. Re-run with the two arms interleaved *inside a single app
run* and ABBA-ordered (so monotonic machine drift cancels instead of favouring whichever
arm is measured second), the same change came out at **median delta −1 ms, faster in 6 of
12 pairs**. Repeated identical configurations on this machine ranged **360–925 ms within
one run** — the noise floor swallows anything under roughly 30%.

The noise-free measurement had been available all along and agreed with the paired
result: instrumenting the swap showed it discards **13 widgets** and mounts 1. A
dose-response sweep (feed page 0/24/60/100 items → 86/170/260/344 widgets →
200/218/244/342 ms) put the screen's cost at **~0.55 ms per widget**, so 13 widgets is
5–10 ms of a ~450 ms push — 1–2%, exactly what the paired A/B failed to detect.

**What to do.** Establish whether the screen is widget-bound *first* (survey + a
dose-response sweep over something that varies the widget count). If it is, size every
candidate lever by the widgets it removes and use wall clock only to confirm a prediction
big enough to clear the noise floor. If you must A/B by wall clock, interleave the arms
within one process and alternate their order; a fixed dev-then-fix ordering across
processes measures drift as effect.

This is the mirror of the defer-past-first-paint lesson, not a contradiction of it.
There, widget count *over*-predicted, because Schedules and Console were sync/DB-bound and
their hidden mass cost nothing to skip. Watchlists is genuinely widget-bound — 13 sqlite
statements and ~10 ms of application code for a whole push, everything else Textual's
per-widget CSS apply and mount. The rule is the same in both cases: find out what the
screen is bound by before choosing what to count.

---

## A test's stimulus can rely on the exact inefficiency your fix removes (task-15459, 2026-08-13)

**Incident.** task-15459 made `LibraryScreen._apply_local_source_snapshot` skip its
`refresh(recompose=True)` when the incoming snapshot is byte-for-byte identical to
what is already rendered — the point of the task, since a warm revisit's reconcile
fetch confirming the app-scoped cache verbatim no longer needs to repaint. Two full
background suite runs afterward reported 14 failures. `test_library_note_recompose_
and_fifty_route_cycles_return_to_baseline` was one: its stress loop called
`_apply_local_source_snapshot` five times with a `dict()`-copied but otherwise
UNCHANGED snapshot, purely to force a recompose and verify a dirty note-editor
session survives being torn down and rebuilt repeatedly. That loop's own assertion
("Generic source-snapshot completion never recomposed the Notes workbench") is
exactly the behavior the fix intentionally removed — the test's PASS depended on
the inefficiency, not on anything the task changed being wrong.

Reflexively "fixing" this by loosening the guard, or by deleting/skipping the test,
would both have been mistakes: the guard is correct (measured 2 composes → 1 for a
real warm revisit), and the test's underlying intent (repeated recomposes must not
corrupt a dirty session) is still a real requirement worth pinning — its STIMULUS
was just now inert. The fix was to vary a harmless field (the notes count) each
loop iteration, restoring a genuine data change that still forces the recompose
under the new contract, matching what a real background refresh would look like.

Of the other 13 reported failures, mutation-bisection (temporarily reverting BOTH
halves of the production diff to their pre-task behavior with `Edit`, confirming
the SAME failure still reproduces, then restoring — never `git checkout --`, which
discards uncommitted work) showed 9 were pre-existing (reproduced identically with
the diff neutralized, mostly drift from an unrelated recent merge) and 4 were
load/order flakiness that passed reliably in isolation. Zero were real regressions.

**What to do.** When an optimization correctly removes redundant work and a test
goes red, do not assume either "the test is now wrong, ignore it" or "my change
broke something" — read what the test's assertion is actually FOR. If it names the
mechanism you just changed ("never recomposed", "recompose count", "refresh was
called"), check whether that mechanism was the test's STIMULUS (how it drove the
scenario) or its OUTCOME (what it was actually verifying). A stimulus that no
longer fires needs a new stimulus that still exercises the real requirement; an
outcome assertion that no longer holds needs the assertion updated to the new
contract. Across a batch of full-suite failures, mutation-bisect each one against
your own diff before writing any of them off as "pre-existing" or accepting any as
"caused by my change" — a batch this size will usually contain both, plus plain
flakiness, and a single red run distinguishes none of them.

---

## An unchanged-skip guard is only as reliable as its least reliable compared field (task-15459, 2026-08-13)

**Incident.** task-15459's `_apply_local_source_snapshot` compared an incoming
snapshot against the currently-rendered one and skipped a recompose when they were
equal — the flagship AC test asserted this held across a reconcile fetch that
should have confirmed the cache verbatim. Review reproduced the test failing
intermittently at exactly that assertion. Root cause: the flat comparison included
`study_counts` (`study_decks`/`flashcards_due`/`quizzes`) and two rail badge
counts (Prompts, Skills) — every one fetched by a `..._or_none` helper whose own
docstring says it swallows ANY exception and degrades to `None`. Under thread-pool
contention, two fetches of the SAME unchanged data could legitimately disagree on
one of these fields (one call transiently raised, the other did not), making the
guard fire a full recompose for a coin-flip on a decorative badge — "fails safe"
(a spurious recompose, not a missed one) but non-deterministic, which is exactly
as unacceptable for an "exactly once" acceptance criterion as failing unsafe.

The first attempt at writing THIS test only asserted the guard's happy path — it
never modeled a field that changes independently of the state a user would call
"the data." A single flat `==` over a snapshot dict is only as trustworthy as its
least reliable member field.

**What to do.** Before folding several fields into one equality check that gates
an expensive operation, audit each field's OWN fetch contract, not just its type.
A field fetched by a helper that swallows exceptions and degrades to a sentinel
(`None`, `""`, an empty collection) is not equivalent in reliability to a field
whose fetch either succeeds or aborts the whole call — the former can flap between
two fetches of otherwise-identical state, the latter cannot (barring the state
genuinely changing). Split the comparison into domains — STRUCTURAL fields that
must gate the expensive operation, and DECORATIVE/best-effort fields that should
be patched through a cheaper path (an in-place widget update, a `None`-tolerant
merge) instead of ever gating it. To prove the split actually closes the gap, do
not just re-run the flaky test and hope: inject the exact transient exception
deterministically (a fake service that raises on its Nth call, not the Mth) so the
flap is reproducible on demand, and mutation-test the fix by temporarily re-
merging the domains to confirm the ORIGINAL failure message comes back verbatim.

---

## A parent `on_mount()` cannot assume nested descendants are mounted (TASK-2702, 2026-08-13)

**Incident.** Three Library Prompt-history tests repeatedly crashed while a
`PromptBlockEditor` was being replaced during rapid recomposition. Its `on_mount()`
queried `#prompt-editor-validation`, a grandchild inside the editor's status container,
and raised `NoMatches`. Instrumentation at the exception showed the editor was attached
and all three direct containers already existed, but their nested children did not. An
unconditional `call_after_refresh` removed that race, then exposed the opposite defect:
two ordinary-mount tests observed an empty footer because they legitimately inspected it
before the deferred callback ran.

**What to do.** A Textual parent's Mount event guarantees neither that every descendant
message pump has finished mounting nor that consumers will wait through an extra refresh.
Initialize synchronously when the required descendants are present; if `NoMatches` proves
the nested-mount window is still open, defer that same initialization once. The deferred
callback must no-op when its original widget has detached. Verify both paths: a rapid real
recompose must kill the synchronous-only implementation, while an immediate normal-mount
assertion must kill unconditional deferral. TASK-2702's final full Prompt-canvas run passed
279 tests only after both boundaries were pinned together.

## A full-suite sweep is a checkpointed pipeline, not a command (task-15211)

**Incident.** Three attempts to run all of `Tests/UI` in one pytest invocation
died at 25-32%, each time losing everything. The fourth attempt split the 503
modules into 16 chunks, appended each chunk's summary and failures to a results
file as it completed, and skipped already-recorded chunks on relaunch. It
survived a hung chunk, an environment process-kill, and a TCC lockout, and
finished: 10,811 passed, 117 attributed failures.

**What the monoliths actually died of.** Not slowness: a product defect. The
Lab/LLM screen's Ollama probe held two event-loop threads open, so pytest
PRINTED ITS FINAL SUMMARY and then never exited -- zero CPU, main thread
joining a non-daemon thread. A wrapper waiting on the child sees an eternal
hang after a successful-looking run. Diagnosis that worked without root:
compare `ps -o time` across an interval (zero accrual = hung, not slow), then
`sample <pid>` for native thread stacks -- two threads parked in kevent were
the loops that should have died with their screen.

**Rules.** (1) Never run a >20-minute suite as one process; checkpoint per
chunk and make relaunch skip recorded work. (2) "The log stopped growing" has
two different causes -- a hung TEST (mid-run) and a hung EXIT (summary already
printed); check for the summary line before assuming the former. (3) Keep the
sweep's worktree frozen and ship fixes from another one; the sweep's chunk
results stay comparable, and later chunks re-finding an already-fixed class is
CONFIRMATION, not new work.
## A permanent gate must read its immutable baseline from a PINNED revision, not the live file it exists to police (TASK-15103, 2026-08-11)

**Incident.** TASK-15103's complete-history denominator — the thrice-reviewed
proof that every diagnostic transition since the stored baseline was
consumed exactly once — read that stored baseline from the live
`production-diagnostic-inventory.json`. The gate's entire lifecycle ends
with regenerating that exact file, so the first LEGITIMATE regeneration
broke it: the stored-revision scan went hunting through all of dev history
for post-repair populations that exist in no dev-reachable revision, and 10
gate nodes fell over — first on a merge-conflict-markered historical blob's
SyntaxError, which had nothing to do with the actual defect. The evidence
was always available immutably: `incident.recorded_base` pins the dev
revision whose committed manifest IS the stale baseline, byte-identical on
all 19 owner rows. One read-from-`recorded_base:`-tree change fixed all 10.

**Companion lesson from the same day.** The freeze-first plan this gate
belonged to never converged against live dev: three boundary re-freezes in
one day (17→18→19 owners), each invalidated by dev advancing while the
evidence was being rebuilt, zero production repairs shipped. Inverting the
order — repair to the frozen contracts first, regenerate and prove ONCE at
the end — landed all 43 repairs plus the gate in one session, and the next
dev advance (11 rows + a sink-topology change) was correctly surfaced as a
NEW incident (task-15600) instead of another re-freeze of this one.

**Postscript (TASK-15700, 2026-08-13): both halves of that forecast held,
and the row still did not ship.** The merge fix restored `and_then_or`'s
rescue **at exactly slot 10**, and scoped recall went 0.429 -> 1.000 — so
the mechanism was correctly identified and its counterfactual correctly
measured. The row was then disqualified on a *different* constraint (no
gated cell down > 0.02), by a mechanism one level further out again: tier 2
confines fallback rows inside the keyword LEG, but tier 2 still enters
FUSION, where a fallback row carrying a vector rank becomes a MERGED row
that outscores any fts-only row. The corollary to carry: **fixing the
composition level the last defect lived at buys you exactly that level.**
Before claiming a fix unblocks a candidate, ask what the NEXT composition
step does with the rows you just re-ordered.

---

## A pre-registered rule the owner then overrides must be recorded as TWO facts, never one (TASK-15700, 2026-08-13)

**Incident.** The keyword-leg arc re-ran TASK-15400's construction sweep
under a decision rule registered **before** the run (max leg census subject
to three hard constraints; ties broken by fewest extra FTS statements, then
smallest code delta). The rule ran to completion and produced a winner:
`prefix`, on a tie-break measured at 240 vs 460 SQLite statements over the
60 golden queries. But by then the arc's own reviews had **measured** a
failure shape the tie-break predates — a construction that widens as the
PRIMARY self-displaces inside one sub-leg's bm25-ordered, LIMITED result
set, where the new tiered merge can protect nothing — and the tied
runner-up (`and_then_prefix`) is immune to it by construction while being
measurement-identical on every captured axis. The owner's standing
stability-over-quick-wins ruling cut against the tie-break, and
`and_then_prefix` shipped.

**The two tempting ways to write that down are both dishonest.** Editing
the rule after seeing its output ("fewest statements — *unless* the row is
structurally unsafe") retroactively makes the sweep unfalsifiable: a rule
amended to fit its own result never rejected anything. Recording only the
shipped value is worse in a quieter way — a later reader assumes the
measurement chose it. That second failure nearly shipped here: the backlog
record read "WINNER under the rule = `prefix`" and said nothing about what
actually shipped, so the one file neither the implementation nor its review
touched was the file that told the wrong story.

**What to do.** When a standing judgement overrides a pre-registered rule,
keep both facts and keep them adjacent, at **every** site that records the
outcome — config comment, the function's own docstring, the test names, the
task record, the PR body: (1) the rule was applied verbatim and produced X,
with the deciding number; (2) the owner ruled Y ships instead, on this
named dimension, at this measured price. Rename any pin whose name asserts
the wrong provenance — `test_the_shipped_default_is_the_sweeps_winner`
became `..._is_the_owner_ruled_construction` precisely because the old name
was a false claim that would have stayed green forever. And police the
*evidence*, not just the value: the census pin added in the same task states
in its own docstring that the census is the number **both** qualifiers score
and is therefore **not** evidence for the ruling — the ruling's evidence is
structural and its price is statements, neither of which a census can see.
A number that cannot see the decision must never be cited as its
justification.

## A harness convenience call that bypasses the production entry path verifies nothing about that path (tasks 15862/15970, 2026-08-13)

**Two incidents in one live pass, same shape.** (1) The wake-UI freshness
suite injected `FleetDrained` events by calling `on_fleet_drained` from the
test coroutine — whose context carries Textual's `active_app` under
`run_test`. Production delivers the drain from the CHILD's daemon thread,
whose `call_soon_threadsafe`-copied context has no `active_app`; a
transcript-poll timer created in that bare context dies on its first tick
(`Timer._tick` reads the ContextVar; an asyncio task inherits its CREATION
context). The suite went green against a fix that did not work — live
frames showed "arm-poll" logged and zero beats, the exact frozen-UI bug the
tests claimed to kill. (2) The user-wins-ties wiring test staged the draft
with `composer.load_draft(...)`, which writes the canonical segments
directly; a live draft typed with real keys was invisible to
`draft_text()` at probe time (pane showed the text, probe read `''`), and a
wake fired straight through the user's held draft — the deferral the test
"proved" (task-15970).

**The rule.** Before trusting a test that drives an event or input, ask
which THREAD, CONTEXT, and ENTRY POINT production uses, and drive that. A
drain must come from a plain thread; typed input must be typed
(`pilot.press`), not loaded. If the harness path and the production path
diverge at any of those three, the test is verifying the harness. The fix
pattern for (1): route the drain through
`threading.Thread(target=...)` in the test, and hop UI arming through the
message pump (`call_later`) in production — after which reverting the hop
fails three tests instead of zero.
---

## A control that holds a second variable fixed measures the PAIR, not the thing you named (TASK-15965, 2026-08-13)

**Incident.** The PRF probe needed to know how many of its 22 target cells a
rescue could have been *seen* in at all, so it ran a control: feed the
retrieval the target document itself — the best expansion any feedback set
could ever produce — and count how many targets that lifts into the top-10.
It returned **8 of 22**, and I wrote that number down as a property of the
retrieval path: "the four-seam path caps ANY query-widening technique at
8/22; 14 of 22 cells are never observable." The probe printed a matching
sentence on every run: *"a target an oracle feed cannot lift into the top-10
could not have been rescued by any real feedback set."*

The control had **two** moving parts, not one. It fixed the path — which is
what I named — and it also fixed the **term selector** used to build the
oracle expression (the pre-registered TF `tf/|D|` top-8). The review re-ran
the control changing **only the ranking key**: same path, same oracle feed,
same composition, same k. It returned **15 of 22** (rarest-8-by-corpus-DF),
and at N=1-rarest with the query side dropped, **22 of 22**. Meanwhile 22 of
22 oracle expressions matched their target at k=200 in *every* row — so
nothing was ever unreachable; the misses were displacement whose severity
scales with **expansion breadth**, which is the selector's property, not the
path's. The defensible statement was narrower than the one I shipped (the
path has no cross-seam ranking and a per-seam `top_k`, so a pass matching K+
notes buries non-note targets *however hard that bites depends on breadth*),
and the correction made the arc's null **stronger**: ≥15 observable cells,
still 0 rescued.

**Two tells were on the page before the review.** (1) The number was a bound
that *flattered the conclusion* — a smaller observable population makes a null
easier to explain away. A bound you would be glad to have is one to measure
twice. (2) The printed claim quantified over something the control never
varied: "could not have been rescued by **any** real feedback set", from a
single selector.

**What to do.** Before reading a control's output as a property of X, write
down every variable the control holds fixed and ask which of them could have
produced the number. If a fixed variable is plausibly load-bearing, **vary it
and re-run** — here that was one parameter and one re-run, and it was the
difference between a measured bound and a bound of my own making. Then make
the instrument carry the scope: the probe now prints the selector-comparison
table instead of the universal sentence, and an assertion fails the run if a
non-pre-registered selector ever reaches the verdict. Sibling of "A mechanism
sentence is an ORACLE" — but distinct in cause: that prose was refuted by data
already on the page, this prose was wrong because the refuting data had not
been collected. Also sibling of "A property that holds 'by construction' holds
for the COMPONENT": both are scope errors, one about composition, this one
about which variable the measurement was actually of.

## Complete invalidation coverage is not evidence a cache is race-free (task-15471, 2026-08-14)

The starred-conversations cache added in task-15471 had provably complete invalidation
coverage — every writer went through `set_mark`/`clear_mark` on the one app-owned service
instance, and the suites were green. The review's interleaving probe still found it serving
stale data in **103 of 300 naturally-scheduled rounds**: a cache-missing reader held its rows
across the transaction COMMIT (a GIL-releasing sqlite call) and stored them AFTER a concurrent
writer had invalidated — so the "invalidated" cache got repopulated with the pre-write
snapshot and stayed wrong until the next write. Two lessons with teeth:

- **Auditing who invalidates answers the wrong question.** The bug was populate-after-
  invalidate *ordering*, which no amount of invalidation-coverage evidence touches. The fix
  shape is a generation counter captured under the lock before the read and compared before
  the store (store only if unchanged); a lost race then costs one skipped store, never a
  stale entry.
- **Only a dedicated interleaving probe surfaced it.** Unit suites exercise reader and writer
  on one thread; the deterministic repro needed a reader paused at commit-exit while a writer
  ran, and the natural repro needed a tight cross-thread hammer. When a change introduces a
  read-cache whose writers live on other threads, a probe of this shape is part of the
  evidence bar — green functional tests alone said this cache was fine.
---

## Consent must bind every independently mutable authority input (TASK-208, 2026-08-13)

**Incident.** TASK-208's first reviewed duplicate override fingerprinted the
folder path, form options, warning text, and previewed active job IDs. The app
correctly re-expanded the folder at submission time, but the Boolean override
then bypassed every match in that new expansion. A newly added member, a newly
active job absent from preflight, or an unchanged warning sentence with a larger
affected-file count could therefore ride stale consent.

**What to do.** For two-step consent, enumerate every input that can change
between arming and use, including derived cardinalities that do not alter copy.
Carry a privacy-safe identity of the exact consented set across the authority
boundary and compare it only after authoritative recomputation. An override must
cover every current match; bounded identity material must record truncation and
fail closed. Test mutations between the two user actions through at least one
real UI-to-authority path, not only each boundary in isolation.
---

## An expected value computed THROUGH the code under test cannot fail — the reference has to come from upstream of it (TASK-16071, 2026-08-14)

**Incident.** The rank-fair four-seam merge arc pinned that the merge preserves
each seam's own ordering: whatever order a seam returned its rows in, the
merged list must contain that seam's rows in exactly that relative order. The
pin needed a reference — "the order the notes seam returned" — and it got one
the obvious way: a **single-source `search()` call** per seam, notes only, then
media only, and so on. That reads like the seam's own ranking, and for a
one-seam query it *is* the same rows in the same order.

It is also the code under test. `search()` runs the merge on its way out, so
the reference travelled through the very function the pin was written to
police. The mutation exposed it: **reverse every seam's ranking before the
interleave**, and the merged list and the reference both came back reversed —
identical to each other, as always. The suite reported **5 passed**. A test
suite reported green against an implementation that had inverted the ordering
property it existed to pin.

The fix is one line of plumbing and no cleverness: a `_seam_ranking` helper
that calls the seam methods directly (`_search_notes` / `_search_media` /
`_search_conversations` / `_search_prompts`), upstream of the merge. With the
reference sourced there, the same mutation reds immediately:

```
E  AssertionError: the notes seam's rows were reordered by the merge:
   seam order [('note','7820e1a3…'), ('note','59470e66…')],
   merged order [('note','59470e66…'), ('note','7820e1a3…')]
2 failed, 3 passed
```

**Why it was invisible.** Every version of the trap looks like reuse, which is
usually a virtue: the reference is fetched by the same public API, on the same
data, in the same test — and a single-source search really is "the same"
ranking, right up until the merge is the thing you are measuring. Nothing about
the call site says "this expression is a function of the code under test"; you
have to ask.

**What to do.** For any test whose assertion compares an output against an
expected value the test itself computes, write down where that expected value
came from and check it does not route through the function under test — a
public API that *wraps* the unit is the common way it does. Source references
from the layer below (the seam method, the raw query, a pasted fixture), and
prove it by mutation: if the mutation moves the output and the expected value
identically, the test is measuring nothing. Sibling of "A surviving mutant
usually means a SECOND writer satisfies your assertion" but distinct in cause —
there a second mechanism produced the asserted state; here there is only one
mechanism and the test asked it to grade its own work.

---

## A fix recorded only in a gitignored file is not a fix — the diff is the deliverable, the scratch is the diary (TASK-16071, 2026-08-14)

**Incident.** A review round on the same arc raised four minors. The
implementer's close-out reported all four addressed, and the working ledger and
task report — both under the worktree's gitignored `.superpowers/sdd/`
directory — described the corrections in detail and accurately. The re-review
checked the **diff** rather than the write-up and found two of the four existed
nowhere else: the collateral-swap identities with their direction (the
rank-fair rotation's cost landing on the NOTE seam) and the rigorous
`r ≥ (p+1)/3` rank-fair bound had been *written down*, not *shipped*. Both were
supposed to land in the tracked `Tests/RAG_Eval/README.md`, which a later
reader would consult; instead they lived in a file that is deleted with the
worktree and invisible to anyone who does not have it.

Nothing was wrong with the corrections themselves, which is what makes the
shape durable: writing the fix and shipping the fix feel identical while you
are doing it, and the bookkeeping that says "addressed" is written by the same
person in the same session, from the same paragraph.

**What to do.** When a review item's remedy is *prose* — a README correction, a
docstring, a task's Notes — close it by naming the tracked file and the text,
then verify with `git diff`/`git status` that the change is in the diff before
recording it as addressed. Treat any scratch or SDD directory as a diary: it is
where you think, never where a deliverable lives. Sibling of the hygiene entry
"Gitignored working files die with their worktree", which is about the same
directory but a different failure — that one loses a record you correctly wrote
there; this one never wrote it anywhere else in the first place.

---

## A repro helper that ASSERTS the bug turns your suite into the bug's guard (TASK-16300, 2026-08-14)

**Incident.** The wake-integrity arc (15970/15971) needed a Console screen that
was mounted but not displayed. It built one through the real navigation API —
push a modal over Chat, navigate to Library — and its helper `_leak_resident_chat`
closed with a *precondition* assertion:

```python
assert chat in app.screen_stack, (
    "harness precondition: the nav-under-a-pushed-screen path must "
    "leave the Chat screen resident in the stack ..."
)
```

That state was a bug: `App.switch_screen` pops only the top of the screen stack,
so navigating under a modal replaced the MODAL and left the outgoing screen
running. When the leak was fixed one day later, four of that file's six tests
went red **on that assertion line** — not on a single behavioural assertion.
The failure output read exactly like "the residency fix regressed the wake
layer". It had not: mutating the 15970 probe fix and both 15971 gates back out
still turned the same tests red once their setups were rebuilt, so the tests
were sound and only their *construction* had been harvested from the defect.

The trap is that the helper was written the RIGHT way by every other rule —
real production APIs, no hand-built screens, no `load_draft` shortcut — and
fidelity to production is precisely what welded it to production's defect.

**What to do.** Before asserting a state as a harness precondition, ask whether
that state is a *contract* or an *observation*. A contract ("the composer holds
the typed text") is worth pinning. An observation of current behaviour,
especially one you reached for because it was convenient, must not be phrased as
a requirement — build the state from the smallest API that produces it legitimately
(here: push a modal over Console; push a second Console screen), and if it is only
reachable through a defect, say so in the docstring and file the defect. Note also
what the wording cost: "the nav path MUST leave the screen resident" is how a
known bug acquires a guard, and the next reader has to decide whether the test or
the fix is wrong.
## A screen refresh is not evidence that a restored child tree is settled (TASK-13207, 2026-08-14)

TASK-13207's real Settings → Model Library → Settings run returned the reviewed
package while an unrelated Speech/TTS draft was detached. The result worker
merged correctly, but publishing the draft before terminal acknowledgement
overlapped the restored panel's queued recompose: Textual `Select` mount events
occasionally observed their overlay children between removal and mount, and the
short cleanup fence could remain attached. Immediate fake leases hid the race;
a mounted test with a deliberately slow lease exit reproduced it.

**What to do.** Treat result acknowledgement, lease exit, and restored-child
composition as separate observation boundaries. Do not publish draft state
until the exact result claim is acknowledged and cleanup authority is released.
Exercise mounted handoffs with a delayed lease exit; a screen-level idle or
refresh observation alone does not prove that a recomposing child tree settled.

---

## Authority tests must vary representation and interleave the guarded write (TASK-16309, 2026-08-14)

**Incident.** The one-time Notes import executor passed its focused execution,
receipt, retry, privacy, and crash-recovery suites, but final adversarial review
still reproduced two authority escapes. First, the approval digest NFC-normalized
title, content, keywords, and template name even though execution stored their
exact Python text. Reusing an approval with composed versus decomposed Unicode
therefore reached the target instead of conflicting. Second, membership-only
execution checked a note version before an unversioned membership write. A
deterministic update inserted between those operations let the stale membership
complete. The new RED tests respectively observed the substituted target call and
the stale attached membership despite the earlier suite being green.

**What to do.** An authority digest must encode the exact representation consumed
by the effect unless the effect itself canonicalizes to the same representation;
include canonical-equivalence substitutions for every execution-effective text
field in approval tests. An optimistic check is evidence only when the expected
version participates in the atomic mutation that grants the effect. Reproduce the
read/write interleaving deterministically, assert the stale write changes nothing,
and cover every idempotent write shape (new row, revive, and already-active row).
Green sequential and crash-recovery suites do not substitute for either probe.

## An AC's enumeration of hot call sites is not the cost profile (task-15764, 2026-08-15)

Task-15764's AC enumerated the difflib work to move off the event loop by name --
`_segment_for_diff` x2, `build_change_diff`, `added_and_removed_text`,
`classify_change_type` -- and an implementation scoped to that list would have been
green on every thread-identity test while leaving most of the stall in place. The
dominant cost was `ContentExtractor.calculate_change_percentage`, a
`difflib.SequenceMatcher.ratio` over the two full raw texts that sits three lines
above the enumerated block and is not in the enumeration. Mechanism, corrected by
the independent review (the implementer's 16.2 s / "99.8%" figure on a 160 KB Latin
page pair did NOT reproduce -- Latin text at that size hits `autojunk`'s fast path,
20-40 ms across four content shapes, and autojunk incidentally returns a
meaningless `pct` for it, a separate pre-existing oddity): character-level
`ratio()` goes quadratic only when the character repertoire is large enough that
autojunk junks nothing (CJK / unicode-heavy pages) -- measured clean 4x per
doubling, extrapolating to ~1 s at 160 K chars and **~7 minutes at the 10 MB
fetch cap**. The off-loop move is thus MORE justified than the original numbers
suggested, and the review's own stall probe corroborated the shape independently
(164.7 ms -> 18.4 ms max stall on the same seam). Keep both halves of this
incident: measure the whole operation, and expect your headline number to be
re-run by a skeptic. The lesson: before implementing a perf task scoped by a list of
call sites, run one measurement that would catch an omission -- a wall/stall probe
around the whole operation, not around the listed calls. If the numbers do not drop
when the listed sites move, the list was wrong, and the AC's own wording ("the
difflib work") almost always licenses fixing the omission in the same change --
record the addition explicitly rather than silently widening scope.

## A version-stamp rollback fixture is a promise every future migration must keep — centralize it or it breaks serially (task-15765/task-16197, 2026-08-15)

Three "historical" ChaChaNotes fixtures were built top-down: bootstrap a fresh
DB (which lands at `_CURRENT_SCHEMA_VERSION`), hand-drop the newer artifacts,
stamp `db_schema_version` back, reopen, and let the migration chain replay.
Each fixture carried its own private drop list — and every migration that
shipped a non-idempotent artifact broke them serially: `88f5f535a` (V33→V34
unguarded `ADD COLUMN compaction_representation`) broke them and task-15730
repaired them one by one; two days later `9174975b0` (V35→V36 bare
`CREATE TABLE note_folders`, task-15705) broke them AGAIN — and, decisively,
its author fixed the ONE fixture they knew about
(`test_dictionary_attachment_index.py`) and missed the other two, producing
task-15765 and task-16197 with the identical "table note_folders already
exists" error, each then repaired in a separate task (16201, 16207). Four
repair tasks for two migrations is the signature of state duplicated where no
gate forces it to stay in sync. The fix is structural, not another patch: one
shared per-version removal registry (`Tests/ChaChaNotesDB/schema_rollback.py`)
consumed by every rollback fixture, a completeness ratchet that fails BY NAME
with instructions the moment `_CURRENT_SCHEMA_VERSION` outruns the registry,
and a rollback-replay sweep over every historical target that compares the
replayed schema's object inventory against a fresh bootstrap. The sweep paid
for itself on its first run: a defensively-copied trigger drop in the V28
entry left DBs rolled back to V20..V27 silently missing ALL conversations
sync triggers after replay — a corruption no per-test fixture would ever
notice, caught only because the sweep asserts parity with a fresh DB rather
than "the test I care about passes".

**Final shape (task-16840, 2026-08-16): the registry was itself the debt, and
the durable end state is no second copy at all.** Within a week of shipping,
the registry had grown hand-written v38/v39 entries — the ratchet was
enforcing exactly the toil the guard existed to remove. The replacement is
the knowledge-free primitive that already lived in the repo: patch
`_CURRENT_SCHEMA_VERSION` to N and bootstrap, and the production chain itself
builds a genuinely vN-shaped DB (`Tests/ChaChaNotesDB/historical_bootstrap.py`)
— real sync triggers, zero future artifacts, so the "already exists"
collision class is impossible by construction and a schema bump costs
nothing anywhere. Three generalisable findings from the replacement:
(1) **a parity oracle derived from the system under test is the identity on
that system's deterministic defects** — the old sweep caught its mutations
only where the registry happened to be a DIVERGENT second copy (the review
verified: true for the DROP COLUMN shape — entry 30's bare DROP would have
raised — but FALSE for the emptied-step shape, whose entry 36 was DROP IF
EXISTS and would have stayed green too; the old design deserves less credit
than this entry first gave it);
re-run against the single-source architecture, the review's own MUT shapes
(emptied V35→V36 step; a `DROP COLUMN messages.usage_json` seeded into
V37→V38) leave the bootstrap-replay-parity sweep 35/35 green while the
migrations' CONSUMER tests red by name (9 note-folder tests; 7 usage_json
tests) — so artifact correctness must be pinned by consumers, and the sweep's
honest job is the genuine historical upgrade matrix (resume from every vN,
stamp/dispatch wiring, stop-resume vs straight-through parity; an unwired
`migration_steps` entry reds all 35 cases with "Migration path undefined").
(2) **check the claimed-pristine baseline**: the "v4" base schema has drifted
to bake in `conversation_local_marks` (a V17 artifact), so a bootstrap at ANY
version carries it — a fixture whose migration-under-test must CREATE an
artifact the base also ships has to drop that one artifact itself
(single-migration knowledge no future bump can invalidate), or the test
silently pins the base's copy instead of the migration's. (3) the
genuine-shape fixtures came out STRONGER and cheaper: the v17 fixture now
proves V17→V18 redefines LIVE sync triggers (the registry version had to
assert them absent), and bootstrap-at-vN measured FASTER than
bootstrap-current-then-rollback (~80-130ms vs ~220-255ms + replay) — the
registry was never even a perf win.

## A silently-shadowed upstream sentinel is a defect class, not a file-local bug (task-16502, 2026-08-15)

Textual 8.x removed `Select.BLANK` (the blank-selection sentinel, renamed
`Select.NULL`) — but referencing it does NOT raise: the lookup falls through the
MRO to `Widget.BLANK: ClassVar[bool] = False`, an unrelated render flag added in
the same major version. Every use of the old sentinel silently became the boolean
`False`: comparisons went permanently dead, and passing it as a Select's initial
`value=` crashed at mount with `InvalidSelectValueError: Illegal select value
False.` Task-565 (2026-07-25) established exactly this mechanism and swept it —
**scoped to settings_screen.py only**, because that was the file under review.
Three weeks later the identical construct in `console_model_popover.py` crashed
the Alt+M popover at mount for any session without a configured model, and was
reported by a user. A grep at that point found **66 remaining `Select.BLANK`
usages across 23 files**, including several sites that had independently
discovered the trap and worked around it locally with comments, and several that
deliberately exploit the `False` value as a synthetic placeholder option — so the
eventual sweep (task-16503) needs per-site classification, not find-and-replace.

**What to do.** When a fix reveals that an upstream rename/removal fails
*silently* (shadowed attribute, `getattr` default, `__getattr__` fallback) rather
than loudly, the first grep result count is the real scope of the defect. Sweep
repo-wide in the same arc, or file the sweep task immediately with the grep count
and the classification burden recorded — a Done task documenting the mechanism
does not stop the next file from shipping the same crash. Evidence here: the
mechanism was fully documented on the board for three weeks while the
user-reachable crash sat live in another file.
## A dodged flake can be the only visible symptom of a deterministic bug (task-15773, 2026-08-15)

Task-15478 hit a once-in-a-full-file-run flake in `ChapterEditorWidget`/`Select`'s
mount sequence when the chapter table populated ~999 rows in one reactive update,
and (honestly, documented as a dodge) reduced the test's chapter density until it
went 0/4. Task-15773 owned the flake and started, per the reproduce-first brief, by
stress-running the interleave -- 34 un-gated iterations across three shapes, zero
trips. What found it was a five-minute CHARACTERIZATION probe of what the code
deterministically does: `chapters = reactive([], recompose=True)` on a widget whose
`compose()` is static meant `watch_chapters` populated the current DataTable and the
scheduled recompose then threw that subtree away -- the settled table had **0 rows
after every single update**, in the minimal host and in the real STTS host alike
(`detected=13 table_rows=0` at HEAD). The "rare flake" was just the narrow-window
crash variant of a 100%-reproducible data-loss defect: the remount re-ran the
Select's Compose->Mount on every data arrival, and any teardown landing between the
fresh Select's registration and its Compose dispatch made its child-mount a silent
no-op (`_pruning`) while `Mount` still fired -- `NoMatches: No nodes match
'SelectOverlay'`. Once the mechanism was named, a gated `_on_compose` interleave
reproduced the exact exception on the first run, every run, and the fix (drop the
recompose; populate the persistent children in place) closed both the flake and the
always-empty table. Two halves to keep: (1) before stress-running a flake, spend one
probe characterizing what the code does deterministically under the flake's stimulus
-- the flake may be the tail of a bug whose body is fully reproducible; (2) a
repetition budget that finds nothing (34/34 clean here) is not evidence the race is
gone -- the gated one-run interleave was both stronger and cheaper.

## Re-verify a residual's CAUSAL hypothesis before building the fix around it (task-15778, 2026-08-15)

Task-15461's Implementation Notes recorded a residual with a cause attached:
the cold Read tab's wall-clock regressed "because the scoped path does the
CONTENT remount as its own discrete remove/mount pair rather than inside one
batched recompose -- Textual's `batch()` is the obvious next move." Task-15778
was filed around that hypothesis. A neutered-batch A/B on the same HEAD
refuted it: **zero** in-swap layout passes and zero compositor refreshes with
AND without `App.batch_update`, because the entire swap already runs inside
`_drain_surface_refresh`'s single `call_next` callback -- a paint-atomicity
that 15461's own `run_worker` -> `call_next` move had bought silently, one
task before it filed the residual blaming its absence. The batch shipped
anyway, but as an explicit contract (survives a future awaiting factory or a
drain restructure), documented as such -- not as the measured win the task
title promised. Two probe traps that nearly hid this: (1) counting layout
passes over the whole settle window attributed 3 post-swap passes (loader,
reseed) to the swap -- bracket the exact call under test, not the settle;
(2) the first probe "confirmed" the premise with numbers that were real but
belonged to a different mechanism. The residual's fix-shaped hypothesis is a
hypothesis; A/B the mechanism (here: neuter the proposed fix on the same
HEAD) before writing the Implementation Notes around it.

## "Nothing happened" cannot name WHICH guard stopped it — count the dispatch (task-15860, 2026-08-16)

Second occurrence in one arc, so it is a class rather than an accident. A
headless-wake test asserted the shipped behaviour "a wake into a busy session
never streams" by giving the loop a window and checking the provider double
recorded no payload. Mutating the guard it was written for -- bypassing
`send_refusal_copy` inside `ConsoleFleetWakeCoordinator._attempt` -- left it
**green**, because `submit_draft` refuses a busy session on its own. The read
site is double-guarded, so an absence-of-effect assertion is satisfied by
EITHER guard and can never say which one it is testing; the test claimed
coverage of the coordinator's gate while actually pinning the controller's.
(The viewless landing hit the identical shape earlier in the same arc: an
unguarded `_apply_world_info` survived because the applier was unreachable in
that rig AND wrapped in a broad `except`.) The repair is cheap and general:
count the DISPATCH, not the effect -- wrap the next seam (`controller.
submit_draft`) with a recorder and assert the list is empty, which fails the
moment the outer guard stops firing. Under the same mutation the repaired test
died with its sibling (2 failed); restored, 13 passed. Corollary for the other
direction: a mutation that leaves everything green is a finding about your
tests, not a nuisance -- both survivors in this arc were real gaps.

## A registry that self-heals on the next attempt is invisible to every test that takes another attempt (task-15860, 2026-08-16)

Mutating `_deliver` so delivered run ids never left the in-memory pending
registry killed exactly ONE test out of fourteen -- and not the exactly-once
test, which is the one whose subject it is. The reason: `_rows_for` drops any
run the durable ledger already shows delivered, so the leak is repaired by the
very next `_attempt`, and any assertion taken after a retry sees a healthy
registry. Only an observation taken at a moment when no further attempt is
coming can see it; here that moment was app exit (`ConsoleRuntime.dispose()`
mid-delivery). When a component has a self-healing path, the state it heals is
untestable through the normal flow -- so a test for it has to pin a TERMINAL
moment (quit, crash, teardown) on purpose. That is also the argument for
keeping such a test when it looks redundant next to the happy-path one.
---

---

## An unbounded wait default turns leaked test rounds into post-suite interpreter hangs

**TASK-16789, 2026-08-15.** After flipping the human-prompt timeouts to a
no-deadline default (ADR-067), `Tests/Chat/test_console_skill_script_confirm.py`
printed "1 failed, 28 passed in 7.49s" — and then the pytest process sat at 0%
CPU for 20+ minutes producing no output (the `| tail` wrapper hid everything
until exit). `sample <pid>` showed the main thread in
`wait_for_thread_shutdown`: the run was over, and the interpreter was waiting
for a non-daemon worker thread. The failing test's assert had skipped its
`resolve_pending_skill_script(...)` cleanup, leaving the confirm round armed;
with the old 120s default that leaked worker self-resolved at process exit in
≤120s (invisible), with no deadline its 1s poll loop never exits at all.

**What to do.** When a wait loop's default becomes unbounded, every fixture
that can arm a round must fail it closed on teardown — the file's
`make_controller` now sets `_shutdown_requested` after each test, which
resolves any still-armed round at its next poll. Diagnosis signature to
recognize next time: pytest's own timing says the suite finished but the
process idles at 0% CPU; macOS `sample` shows `wait_for_thread_shutdown`;
`kill -ABRT` (with `PYTHONFAULTHANDLER=1`) dumps the stuck thread stacks into
stderr.

---

## A cross-suite ordering failure can be an app KILLING ITSELF, not an object crossing the boundary (task-15860, 2026-08-16)

`Tests/UI/test_console_headless_wake_fires.py` +
`Tests/UI/test_console_store_continuity.py` run together gave **1 failed, 4
passed**; each file alone was green. Every hypothesis on the obvious list was
about something *surviving* the test boundary — an undisposed app-owned
`ConsoleRuntime`, a pending delivery, a leaked DB handle, a module singleton,
a daemon thread. All of them were wrong. Four *identical* wake rounds in one
process were green, and the two poisoners followed by a plain no-wake nav probe
were green: nothing accumulated. What actually happened was that the THIRD
app killed itself — a `console-sync` worker whose screen had been closed raised
`NoMatches`, Textual's default `exit_on_error=True` handed it to
`App._handle_exception`, and from then on every `post_message` was silently
dropped, so the next `NavigateToScreen` produced 15 seconds of total silence
and "stuck on LibraryScreen". The prior tests contributed timing pressure, not
state.

**What to do.** Before hunting for the leaked object, ask whether the victim
app is still ALIVE: dump `app.is_running` / `app._closing` / `app._closed` /
`app._exception` at the point of the symptom. A dead Textual app is
indistinguishable from a hung one from the outside — the message queue is
empty, the workers list is empty, the loop is running, and nothing logs. Two
corollaries that generalise: (1) `is_mounted` stays **True** for a screen
Textual has already closed (the removed surface reported `is_mounted=True`
with `is_running=False` and no children), so a mount check is not a liveness
check — `_closing`/`_closed` are; (2) a per-file green gate structurally
cannot see this class, because the damage needs several app lifetimes in one
process. Running the whole directory in one invocation is what surfaces it.

## A coroutine that re-arms itself from `finally` escapes the framework's teardown sweep (task-15860, 2026-08-16)

Textual cancels a node's workers in `Widget._on_unmount`
(`workers.cancel_node(self)`). `ChatScreen._sync_native_console_chat_ui`
re-armed itself with `self.run_worker(...)` inside its own `finally` — which
runs *after* that sweep — so the worker it created was never in the cancelled
set, ran a full DOM sync against a screen with no children, and killed the app.
The instrumentation that named it in one pass: wrap `DOMNode.run_worker`
filtered to the suspect group and log `traceback.format_stack()` at creation;
the creating frame was the `finally` itself. Generalises to any self-scheduling
loop (timers re-arming timers, callbacks re-posting themselves): the framework's
"cancel everything this node owns" happens once, and anything scheduled after
it is invisible to it. Guard the re-arm on the owner still being alive, not
just the body.

## A `MagicMock(spec=Cls)` answers every METHOD truthily — a new guard predicate must not be one (task-15860, 2026-08-16)

A teardown guard was added as `ChatScreen._console_screen_torn_down()`, reading
`_closing`/`_closed`. Three `Tests/UI/test_ui_responsiveness.py` tests that
drive `ChatScreen._sync_native_console_chat_ui(mock)` against a
`MagicMock(spec=ChatScreen)` went red: the spec'd mock auto-provides every
method in `dir(Cls)`, and the auto-returned `MagicMock` is TRUTHY, so the new
guard reported "this screen is torn down" for every mocked screen and the code
under test returned before doing anything. Measured three ways: 15 passed at
the pre-fix baseline, 3 failed with the method form, 15 passed with the
identical logic moved to a module-level `_console_screen_is_torn_down(screen)`.
The reason the module form is immune is the same mechanism read the other way —
`_closing`/`_closed` are set in `__init__`, so they are NOT in `dir(Cls)`, a
spec'd mock raises `AttributeError` for them, and `getattr(screen, "_closing",
False)` correctly reads a mocked (or never-mounted) screen as LIVE.

**What to do.** When adding a *predicate* that new early-returns depend on, ask
what a spec'd mock of the host class will return for it before choosing where
it lives. A module function reading raw attributes is the mock-safe shape; a
method is not. The failure is nasty because it is silent — the guard fires, the
body is skipped, and the assertion that fails is about something else entirely.
Trap-detection note: neutralising the method's BODY does not restore the tests
(the mock never calls it), so the usual "mutate the fix off and compare" check
reports "identical failure sets, not mine" — the only honest discriminator is a
real pre-fix baseline worktree.
## A parity test that passes against the pre-fix tree proves nothing (TASK-16811, 2026-08-16)

The first version of `test_focus_token_parity.py` asserted a selected
NavigationButton's resolved background equals the transcript's selected-row
colour — and passed both post-fix AND against the unfixed tree. Two masks
stacked: `run_test()` auto-focuses the first focusable widget, and the app
bundle's generic `Button:focus { background: $ds-focus-bg }` rule (app tier
beats any DEFAULT_CSS rule) painted the canonical colour over the shadowed
`.active` rule the test meant to probe. The divergence only exists on the
UNFOCUSED active state. The test became meaningful only after blurring
(`app.set_focus(None)`, plus asserting `focus` absent from the pseudo-class
set) — verified by running the corrected test in a throwaway worktree at the
pre-fix commit, where it finally failed. Rules: (1) a regression test for a
visual fix is only evidence once it has been RUN against the pre-fix tree
and observed red there; (2) any style probe on a widget mounted first in a
test App is probing the focused state whether you meant it or not.
## A bare scroll_to(max) walk is not a user gesture — it self-terminates the moment the boundary stops moving

**TASK-16851, 2026-08-16.** The head-pinned-selection fix (refuse tailward
hydration while over the high mark with a blocked prune) passed its stall pin
but "failed" its Esc-recovery pin: after Esc unblocked the prune, an 80-round
`scroll_to(y=max_scroll_y)` walk never advanced a single chunk. Probe: reader
parked at exactly `scroll_y == max_scroll_y`, so every subsequent `scroll_to`
produced NO scroll_y change — `watch_scroll_y` never fired, nothing scheduled
hydration, and the loop measured the harness gesture, not the product. Every
REAL input path (wheel-down, PageDown, End) has its own boundary hook and
recovered immediately. The pre-existing two-sided walks had only ever worked
because hydration kept GROWING max_scroll_y under them, re-arming the watcher
each round — a walk test that relies on that is green only while the feature
under test keeps moving the goalposts for it.

**What to do.** Drive boundary-walk tests with the product's real gestures
(`action_page_down()`, wheel events, `scroll_end`) — or at minimum pair the
positioning `scroll_to` with one. Before concluding a recovery path is broken,
check whether the loop's gesture can still produce a state change at all.

Same task, implementation twin worth remembering: a decision that walks
`self.children` (the hydration refusal reusing `_compute_prunable_prefix`)
must run under the widget's reconcile lock — read mid-reconcile, the transient
child order faked a "blocked prune" and stalled a selection-free End drain
(218 messages stranded in the born-red End-race pin).

## A guard added by a later ADR can hollow out an older test without turning it red (task-15860, 2026-08-16)

`Tests/Chat/test_console_runtime_lifetime.py`'s two AC#2 approval pins —
"leaving Console denies a parked approval round" and "a round from the
previous visit is not resurrected" — build the controller with
`app is None`. That was fine when they were written. Then ADR-067 added a
no-`app` guard to `request_mcp_approvals` that denies every name on the spot,
and from that moment the rounds never reached the poll loop at all: both tests
passed on the guard's verdict, not on the cancellation signal they claim to
pin. Measured while mutation-testing a change to that exact signal: with
`_is_session_cancelled`'s visit check deleted outright — fail-open for every
session-scoped round — the whole file was still **14/14 green in 0.98s**.

**The tell was the clock.** A file whose tests are supposed to poll on a 1.0s
granularity cannot finish in less than one poll interval. After wiring a
`call_from_thread` app the same file takes 2.81s and the same deletion fails
both pins.

**What to do.** When you change a signal, mutation-test the OTHER files that
claim to pin it, not only your own — a green neighbour is not evidence.
And when a suite that exercises timed waits runs impossibly fast, that is a
finding, not good luck.

## pytest silently drops a directory argument when a file inside it is also listed (task-15860, 2026-08-16)

A gate invocation passed `Tests/Agents/` *and*
`Tests/Agents/test_agent_runs_wake_ledger.py` (the second arrived from a
separate "wake suites" list). pytest collected **283** tests instead of
**1,733** — the directory arg was collapsed against the more specific
file — and reported a perfectly healthy `282 passed, 1 skipped`, exit 0.
Nothing in the output says a thing was skipped; the only evidence is the
count, and 282 looks like a normal number.

**What to do.** Never pass a directory and a path inside it in the same
invocation. And "READ every count" means read it against what you MEANT to
run: `--collect-only -q | tail -2` on the exact argument list first, then
compare. A count you have not predicted cannot be checked.

## Textual's `run_test` disables notifications, so a toast assertion can never see a toast (task-15860, 2026-08-16)

`App.run_test()` defaults `notifications=False`, which sets
`_disable_notifications` and makes `Screen._extend_compose` skip the
`ToastRack` entirely. A test asserting on rendered toast widgets fails
forever under the default; a test asserting on `app._notifications` passes
without proving anything reached a screen. Pass `notifications=True` and
assert on the widget.

Second trap in the same assertion: `Toast` is a `Static` that never calls
`update()`, so its `renderable` is empty — a helper reading `renderable`
reports "no toast" for a toast that is on screen. Read `Toast.render()`.

## Do not commit to a file the running suite imports — `inspect`/`linecache` read source off disk lazily (task-15860, 2026-08-16)

A 59-minute single-process Console population (3,404 tests) came back with
**4 failures unique to the branch**, all in
`test_console_prompts_controller.py::test_screen_keeps_a_real_delegation_for_
every_outside_caller[...]` — a test that does
`inspect.getsource(getattr(ChatScreen, name))`. Its message showed it had read
a *different method's* body entirely.

The cause was a comment-only commit to `chat_screen.py` (net +7 lines at line
14903) made **while the run was in flight**. Each method's `co_firstlineno` is
fixed at import; `inspect.findsource` calls `linecache.checkcache`, re-reads
the now-changed file, and every method defined below the edit reports source
shifted by the delta. The tell was not the assertion text but the SPLIT: the
two parametrisations that passed are defined at lines 5330 and 6264, the four
that failed at 16684, 16790, 16794 and 17198 — a clean line-number boundary at
the edit point. Re-run on a stable tree: 37 passed.

**What to do.** While a long run is in flight, stage edits somewhere the run
does not import, or wait. This bites any assertion built on `inspect.getsource`
/ `inspect.getsourcelines` / traceback rendering — and those are exactly the
architecture-contract tests that a big single-process gate is there to run.
## A born-red run that dies on ImportError is not born-red evidence (TASK-16838, 2026-08-16)

The in-flight-guard test file imported the new `_IN_FLIGHT_URL_CHECKS`
registry at module top. Run against the pre-fix tree (worktree at
`1af8c0414`) it "failed" — but on collection, with `ImportError: cannot
import name '_IN_FLIGHT_URL_CHECKS'`. That red proves only that the test
mentions a symbol the fix adds — the same red a typo would produce — and it
says nothing about whether the bug (the 15764 double-check interleave) is
reproduced or the assertions could catch it. Rewritten with a lazy
`getattr(svc, "_IN_FLIGHT_URL_CHECKS", set())` lookup so the file COLLECTS
on both trees, the pre-fix run reddened on the behaviour itself: the manual
entrant's gated fetch fired while the scheduled fetch was still in flight
("went to the network too"), the exact double-report the review had
demonstrated. Rule: when new-code symbols would make a born-red file
unimportable at base, reference them lazily (or split the white-box asserts
out) so the base-tree run fails on the assertion that carries the evidence,
not on `import`.

---

## A per-tick view value needs its CACHE KEY and its SCOPE mutation-tested; display assertions see neither (turn-activity line, 2026-08-16)

The Console's in-flight assistant row gained a live activity line (`⚙
read_file · 4s`) refreshed on the 0.2s poll. Every display assertion was
green, and mutation testing then found two defects that no rendered-text
assertion could have seen:

1. **The cache key.** `ConsoleTranscript` has TWO renderers — markdown (the
   default for assistant rows, which carries the line in its *header*) and
   plain. `_message_row_signature` is built from the PLAIN renderer only, so
   the markdown row's elapsed advanced solely as a side effect of the plain
   renderer embedding the same string. Disabling the plain branch left the
   first paint correct and froze every later tick — the tell was not "the
   line is missing" but "the FIRST tick passed and the SECOND did not".
2. **The scope.** Stamping the value on every message instead of only the
   in-flight row changes nothing a reader can see (a row with content never
   renders it; only assistant rows can) — but it lands in every row's
   signature, so the whole transcript re-derives and re-syncs once per
   second for the entire turn. The mutant SURVIVED a suite of rendered-row
   assertions.

**What to do.** For any value the poll re-supplies each tick: (a) mutate the
signature/cache key and require a test that paints two ticks differing in
*nothing but that value*; (b) mutate the scope and assert **blast radius**,
not pixels — `row_render_signatures()` and
`message_signature_compute_counts()` make "exactly one row moved" a direct
assertion. Also worth knowing for this widget: a signature that renders one
of two renderers silently couples them, so name the field in the signature
outright rather than relying on it riding along inside rendered text.
## A guard sitting behind an earlier early-return is unreachable, so no fixture can own it (task-15860, 2026-08-16)

Third mutation-survivor in this arc, and a different shape from the two
above (which were two guards in SERIES at one read site). The launch-wake
loop skips a marked conversation that owes nothing —
`if not wake.has_pending(cid): continue` — and mutating that line to
`if False:` left the whole suite **green**. Investigating rather than
patching around it: every test that exercised an unowed mark used exactly
ONE mark, and with one unowed mark the function returns earlier, at
`if not wake.seed_from_marks(): return 0`, before the loop runs at all.
The guard was not weakly tested, it was *unreachable* for every fixture in
the file, so no assertion anywhere could have distinguished "we checked
each conversation" from "we never got that far". The repair is a fixture
change, not an assertion change: a test with TWO marks, one owed and one
not, gets past the earlier return and then asserts the unowed conversation
was never hydrated. Under the same mutation it now dies alone (1 failed,
8 passed). Rule: when a mutation survives, before touching assertions ask
whether the mutated line *executes* under any fixture you have — an
earlier `return` upstream of it is the commonest reason it does not, and
it is invisible in the diff you are mutating.

## A "constructs nothing" pin needs an observer the production code cannot lie to (task-15860, 2026-08-16)

The owner's ruling on wake-at-launch required that an install with no
background work pay one indexed read and build NOTHING, so startup stays
byte-identical. "Nothing was constructed" is exactly the claim a weak test
states and never checks, so the pin took four independent observations:
the marks service's call list, the four `ConsoleRuntime` slots being
`None`, no `deferred_launch_wake` task ever created — and **the absence of
the `agent_runs.db` FILE on disk**, because constructing the agent bridge
opens (and creates) it. The filesystem one is the observation that cannot
be satisfied by a mock, a stub or a lazily-`None` attribute. It earned its
place under mutation: removing both empty-marks guards was caught by the
call-count and by a sibling test's runtime assertion, and removing only
the outer guard was caught *solely* by the task-name observation — the
`None` slots stayed `None` because the inner function had its own guard.
Two guards in depth meant no single observation covered both mutations;
the four together did. Corollary: a no-work pin also needs a control that
runs the same probes WITH work present and watches every one flip,
otherwise a hook that never runs at all satisfies it perfectly.

## Your test's own harness can make the guard you are testing unreachable (task-15860, 2026-08-17)

The close-out gate had to prove one invariant no per-landing test owned:
deliveries are serialized **app-wide**, enforced by one line in
`ConsoleFleetWakeCoordinator._attempt` — `if self._delivering is not
None: return`. Two successive drafts of that test passed, and **survived
neutering that exact line**. Both were worthless, for two different
reasons, and both reasons are general:

1. **The first draft used one conversation.** A second completion in the
   same conversation is refused by the *per-session busy* gate several
   lines earlier, so `_delivering` was never the thing under test. A test
   of gate N must construct the state where gates 1..N-1 all pass —
   otherwise it is a test of gate 1 wearing gate N's name.
2. **The second draft used two conversations and still survived**, because
   the observation was "no second payload reached the provider". The
   provider double stalls in its readiness probe, and the stall belongs to
   the GATEWAY, not to a turn: with the guard removed, the second wake
   turn genuinely started and then parked at the same stall, streaming
   nothing. The two outcomes — "refused" and "started, then blocked
   identically" — are indistinguishable at the observation point the test
   was reading.

The fix was to count *entries into the readiness probe*, which separates
"a turn started" from "a turn produced output". The mutation then killed
the test immediately.

**The rule:** when a mutation survives, do not first suspect the
assertion's strength — ask **what the harness itself does to the code path
after the mutated line.** A shared blocking double, a fixture that stops
upstream, a stall that is global rather than per-attempt: each converts
"the guard fired" and "the guard did not fire" into the same measurement.
Pick an observation that is downstream of the mutated line but *upstream*
of whatever the harness blocks on.

## Measure the invariant, then write the assertion — the honest answer may not be the one the plan states (task-15860, 2026-08-17)

The same gate had to pin "exactly-once across a restart mid-commit". The
plan and the shipped User Guide both asserted the strong form: a restart
between a wake being accepted and the app exiting re-announces nothing.
Rather than encode that, the test was written to *measure* first — die
inside the window (the ledger stamp raises, leaving rows committed and the
ledger unstamped, which is byte-identical to a process kill there), then
relaunch and read what the conversation holds.

It holds **six** rows, not four: the same child result announced to the
supervisor twice, and paid for twice. `_deliver`'s own comment predicts
it ("a lost stamp risks one re-announce at a later claim, never a lost
result"); the user-facing doc had quietly promised more than the code
does. The live pass then reproduced it by accident — an app quit while a
wake turn sat blocked produced exactly one duplicate notice at the next
launch.

Two things followed, and both are the point. The doc was corrected to the
measured behaviour. And the test asserts the **bound** (at most one
re-announce, the row shape, no USER row on any of it, and that a third
launch adds nothing) rather than the measured number — so closing the
window later is an improvement, not a test failure. **Encoding a plan's
claim as an assertion turns an unverified sentence into a fixture that
future work must preserve.** Measure, then decide which part is the
invariant and which part is merely today's value.
---

## A fixture keyed to the code's invented config section hides a total production failure (task-17382, 2026-08-17)

`summarize_with_llama` indexed `loaded_config_data["llama_api"]` in ten places.
No such section has ever existed — the loader builds `llama_cpp_api` — so every
llama.cpp summarization raised `KeyError` before contacting a server, and the
`except` at the bottom returned an error STRING rather than raising. The
deep-search caller tested `summary.startswith("Error:")`, which no
provider-prefixed message matches, so `"Llama: Error occurred while processing
summary with Llama: 'llama_api'"` was stored AS the result's evidence content
and the synthesis was built from it. Citation verification kept passing because
it matches quotes against `original_content` first, so the reports were graded
sound while the model had never been shown its sources.

The reason this survived a security review of that very file:
`test_summarization_diagnostic_privacy.py`'s fixture stubbed the settings dict
with a `"llama_api"` key — the name the summarizer had invented. The tests fed
the code its own mistake and passed. The same fixture stubs `api_keys` and
`local_api_ip`, which is exactly why the Kobold and TabbyAPI summarizers'
identical defect (task-17383) also stayed invisible. Fixing the code then broke
those tests, which is the only reason anyone looked.

**What to do.** A fixture standing in for configuration must be keyed to what
the LOADER produces, not to what the code under test reads — those are the same
string only when the code is right, and a stub that mirrors the code's
assumption can never fail. When you fake a provider response, fake what the
SERVER sends: my own first fake returned llama.cpp's native `{"content": ...}`
shape, which is what the buggy parser read, so it passed while the live
endpoint (`/v1/chat/completions`, `choices[0].message.content`) returned "No
choices in response data" on every call. Cheapest check available: print the
real `load_settings()` keys once and compare, or assert the section exists.

---

## A metric can be graded on fallback content, and nothing in it says so (task-17370, 2026-08-17)

Every live research baseline recorded in this repo reports
`citation_accuracy 1.00` and healthy `claim_support_rate`. All of them were
measured with per-result summarization failing: first instantly (wrong config
section), then a 404, then an unparseable payload, and once those were fixed, a
timeout at exactly the shipped 30s per call on a local 27B. Each failure fell
back to raw source text, which is the CORRECT degradation — and completely
invisible in the metrics, because a report built from source text still
resolves its markers and still verifies its quotes.

The tell was uniformity: six summarizations completing in exactly `30.0s` is a
timeout, not a latency distribution.

**What to do.** When a pipeline has a degradation path, a metric that only
grades the OUTPUT cannot tell you which path produced it — so record the path
alongside the number (which stage ran, which fell back), and treat suspiciously
round, uniform timings as a budget being hit rather than work being done. Also:
absence of an error log is not evidence of success when the code logs successes
at INFO through stdlib `logging`, whose default level hides them; the runs above
showed zero "Summarization successful" lines whether they worked or not.

**Second instance, and the sharper rule when the number is a DELTA (TASK-16965,
2026-08-17).** Same shape, opposite tell — and no tell at all. TASK-16965 had to
answer "does cross-encoder reranking help retrieval here?" by running the gated
eval set twice, once reranked and once not, and reading the difference.
`CrossEncoderReranker` honours the TASK-3502 contract: a model that fails to
load DEGRADES (returns the caller's ordering untouched) rather than raising. And
`Tests/conftest.py` sandboxes `HOME`, while
`huggingface_hub.constants.HF_HUB_CACHE` is computed from `expanduser("~")` **at
import** — so under pytest `CrossEncoder(...)` raises `OSError` ("couldn't
connect ... and couldn't find them in the cached files") on a machine where the
model IS cached. Measured directly, before the probe was written. Compose those
two facts: every window comes back in its original order, every metric is graded
on un-reranked output, and the before/after table reads a flawless **0.000 delta
on all 105 cells** — a NULL result, publishable-looking, pre-registered as an
acceptable outcome, and entirely fabricated. Unlike task-17370's uniform `30.0s`
timings there is no tell whatsoever: a real null and a never-ran null are the
same table. The run therefore repoints the constant
(`monkeypatch.setattr(constants, "HF_HUB_CACHE", real_cache)` — hf_hub 1.x reads
it at call time off the module attribute) and **asserts the work happened**:
`rows_scored > 0` and `rows_failed == 0`, per pass. It scored 3,621 rows, 0
failed, and moved 1,950 — which is what makes the verdict it did produce
(HARMED, bimodal) mean anything at all.

**What to do.** Recording the path is enough when a bad path makes the number
look good; it is NOT enough when the measurement is an A/B and the subject
degrades to the identity, because then the failure mode is the null hypothesis
itself and no reader can tell the two apart. So: **a measurement whose subject
degrades silently must assert, inside the run, that it did work** — a positive
count of units processed and a zero count of failures — or its null is
unfalsifiable and must not be published. Write those assertions BEFORE you look
at the numbers; a 0.000 delta is the one result that never prompts anyone to go
looking for a bug. Corollary worth its own grep: the frozen-at-import
huggingface_hub constants bite in more than one place — `HF_HUB_OFFLINE` (see
"HF offline enforcement must be set before `huggingface_hub.constants`
EVALUATES" above, where the blast radius is an unwanted download) and
`HF_HUB_CACHE` (here, where the blast radius is a load you wanted and silently
did not get, under any fixture that moves `HOME`).


## When you find one inert declared surface, enumerate its whole namespace

**TASK-16174 / TASK-17600, 2026-08-16..18.** Three separate arcs each found
one config surface that was declared, switchable, sometimes documented — and
implemented by nothing:

1. `include_parent_docs` / `parent_size_threshold` /
   `parent_inclusion_strategy`: shipped, set to `true` by three profiles,
   **read by nothing** (TASK-16174 retired them).
2. `result_reranking`: a middleware declared with `enabled = true`, listed by
   the `high_accuracy` pipeline, handled by a bare `pass`.
3. `reranking_strategy`: a config key with **zero readers**, which
   TASK-16965's own design doc simultaneously told users was the lever for
   selecting a reranking strategy.

Each was found by accident, while doing something else. Nobody looked for the
CLASS until the third one — and when TASK-17600 finally enumerated the
namespace instead of the single filed name, `result_reranking` turned out to
be **one of eight**: eleven middleware names were declared by pipelines and
four implemented, with seven falling off an `if/elif` and no-opping silently.
Two entire pipelines (`technical_docs`, `research_papers`) consisted of
nothing but unimplemented middleware, and three names referenced no
definition block at all.

**What to do.** The first inert surface you find is a sample, not the
population. Before closing, enumerate its whole namespace **in both
directions** — declared-but-unimplemented AND implemented-but-undeclared —
and write the enumeration as a test rather than a one-off grep, because the
grep answers today and the test answers forever. Give that guard a
self-check (`test_the_guard_can_see_the_names_it_is_guarding`): a namespace
guard whose parser silently stops matching becomes a green test that
guarantees nothing, which is the same failure it was written to prevent.

**A corollary this cost us directly:** a doc can *create* the surface. The
`reranking_strategy` claim was written by the arc that measured the feature,
in the same commit series that carefully documented everything else
truthfully — so include documentation in the sweep, and check that the lever
a doc names is one the code actually reads.
