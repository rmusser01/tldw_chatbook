# Lessons: Textual framework traps

Working knowledge about the Textual version this repo pins (8.x, currently 8.2.8) —
API gotchas that cost a debug cycle when first met. Not decisions (see
`backlog/decisions/`) — these are traps that have actually cost time here, kept so the
next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore gets ignored. If you add one, bring the incident.

---

## `run_worker` does not forward positional arguments to the callable

**TASK-16314 review round, 2026-08-14 (commit `14373a7ac`).** The trajectory screen's
live poller needed to run `self._live_rebuild_worker(revision)` in an exclusive worker
carrying the revision the rebuild was built for, so a slow older rebuild could be
identified and dropped when a newer revision had since been observed. The natural call
— `run_worker(self._live_rebuild_worker, revision, thread=True, exclusive=True)` —
cannot work: Textual 8.2.8's signature is
`run_worker(work, name='', group='default', description='', exit_on_error=True,
start=True, exclusive=False, thread=False)` with **no `*args` passthrough**. The extra
positional binds to `name` (verified by introspection against the installed 8.2.8),
and the worker invokes the callable with zero arguments — for a method with a required
parameter that surfaces as a confusing `TypeError` inside the worker; for a callable
with optional parameters it would run silently on defaults, which is worse. The pilot
caught it; the fix is a closure:

```python
self.run_worker(
    lambda: self._live_rebuild_worker(revision),
    thread=True,
    group="trajectory-live",
    exclusive=True,
)
```

**What to do.** When a worker callable needs arguments, close over them (a `lambda`
or `functools.partial`) — never pass them positionally after the callable to
`run_worker`; they are constructor parameters of the worker, not arguments of your
function, and nothing warns you. Same discipline for `timer`/`set_interval` callbacks
that need captured state. If the arguments are cheap to recompute inside the worker,
prefer passing a zero-argument method and re-reading state there — it also avoids
holding stale references across `exclusive=True` cancellations.

---

## Related

- `lessons-testing-evidence.md` — includes the Pilot-harness traps (detached widget
  references after recompose, bare-`App` harnesses that never load the app stylesheet)
- `lessons-live-verification.md` — why a green suite can still miss live-only defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps
