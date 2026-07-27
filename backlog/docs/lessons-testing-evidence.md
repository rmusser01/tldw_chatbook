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

## Related

- `lessons-live-verification.md` — why the suite could not see seven of these defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps
