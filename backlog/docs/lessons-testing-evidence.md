# Lessons: what counts as evidence a change works

Working knowledge about testing in this repo. Not decisions (see `backlog/decisions/`)
and not point-in-time audits — these are traps that have actually cost time here, kept
so the next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore is ignored. If you add one, bring the incident.

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
declared only in the `[websearch]`/`[all-tools]` extras, and registered
`"aiohttp": False` in `Utils/optional_deps.py` — but the `/generate-image`
console feature had quietly wired it onto the **default** screen's import chain:

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

## Related

- `lessons-live-verification.md` — why the suite could not see seven of these defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps
