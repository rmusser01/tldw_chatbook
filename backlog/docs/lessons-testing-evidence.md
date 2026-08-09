# Lessons: what counts as evidence a change works

Working knowledge about testing in this repo. Not decisions (see `backlog/decisions/`)
and not point-in-time audits — these are traps that have actually cost time here, kept
so the next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore is ignored. If you add one, bring the incident.

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
ONE test: the timeout stack dump names it.

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
modules found many more prompt, response/output, credential-fragment, and
private endpoint/path diagnostics that the filter was never designed to find.

**The rule.** Preserve the stated proof boundary when evidence crosses into a
follow-up. Build remediation inventories from the complete owning population,
grouped by stable module/function/diagnostic identity; use heuristic matches
only as candidates or cross-checks, never as the denominator.
