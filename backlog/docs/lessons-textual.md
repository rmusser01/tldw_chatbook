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

## A mutable `reactive([])` default is one shared object — and reassigning an empty-equal value does not un-share it

**TASK-15771, 2026-08-15.** `reactive([])` / `reactive({})` with a literal default
installs the *same* list/dict object on every instance of the widget class:
`Reactive._initialize_reactive` does `default_or_callable() if callable(...) else
default_or_callable`, and a literal is not callable. Any in-place mutation
(`.append`, `[k] =`, `.insert`, `.clear`, ...) then leaks across every instance and
every future instance, including screen remounts. The package-wide AST sweep found
**41** such declarations — in two rounds: the first sweep reported 27, and the task
review proved it structurally blind to the subscripted-generic spelling
`reactive[list[dict[str, Any]]]([])`, which parses as `Call(func=Subscript(...))`
and slipped past a `Name`/`Attribute`-only function-name match (14 more sites, all
in `UI/Watchlists_Modules/`; runtime-identical bug). An AST detector for a call
must unwrap `ast.Subscript`, or the generically-annotated spelling of the exact
same call is invisible to it. Born-red two-instance tests demonstrated the leak on
`CharacterVoiceWidget.characters`/`voice_assignments`,
`ChapterEditorWidget.chapters`, and `CollectionsTagWindow.selected_keywords`
(`Tests/Widgets/test_reactive_default_aliasing.py`).

The half of the trap that reverses classifications: **"reassigns before use" is not
a defense when the reassigned value is empty-equal.** `Reactive._set` in 8.2.8 only
runs `setattr(obj, self.internal_name, value)` inside
`if always or self._always_update or current_value != value:` — so
`self.chapters = chapters if chapters else []` in `ChapterEditorWidget.__init__`
compared `[] != []`, stored nothing, and the instance kept aliasing the class-shared
default it then `.insert()`ed into. The site read as safe in review; the born-red
test proved it was not. Only a mutation-sites trace plus this mechanism check
classifies correctly.

**What to do.** Always declare mutable reactive defaults as callables —
`reactive(list)` / `reactive(dict)` / `reactive(set)`, or
`reactive(lambda: [seed])` for non-empty defaults.
`Tests/Architecture/test_reactive_mutable_default_inventory.py` pins the package
at zero for the forms it detects — mutable literals, comprehensions, module-level
shared mutables, and `list()/dict()/set()` call results, in both the bare and the
subscripted-generic call spellings. It does NOT see shared mutable *instance*
defaults (`reactive(SomeClass())`; 5 known occurrences at review time) — a green
run is not clearance for that form. If the guard fails, fix the default — never a
"reassign before use" workaround.

---

## Square brackets in widget text are console markup — an unescaped `[Word]` is deleted from the UI without a warning

**TASK-19734, 2026-08-21.** The import wizard's mislabelled tag checkbox was
relabelled to name what it actually does — prefix imported item names with
`[Imported]` — written as the plain string
`'Prefix imported item names with "[Imported]"'`. Textual renders label and
`Static` text as console markup, so `[Imported]` parsed as an (unknown) style tag
and was dropped: the shipped control read `Prefix imported item names with ""`.
The behaviour is silent — no exception, no log line, no styling change — and it
only surfaced because a test asserted the substring was in the rendered label:

```python
>>> str(Checkbox('… with "[Imported]"').label)
'… with ""'
>>> str(Checkbox(r'… with "\[Imported]"').label)   # escaped
'… with "[Imported]"'
```

Escape with a raw string and a leading backslash (`r'\[Imported]'`) anywhere a
literal bracket must reach the user, and **assert on the rendered label**
(`str(widget.label)` / `str(static.renderable)`), never on the source constant —
a test that checks the constant passes while the UI shows nothing. This bites
hardest on exactly the copy that most needs to be literal: names, prefixes,
placeholders, file globs, `[Imported]`-style markers.

---

## `Static.update()` lays out the view by default — a one-row repaint is not free

**TASK-21117, 2026-08-23.** The Console Inspector's outer scroll hint is a
pinned one-row `Static` whose copy blanks at the bottom of the rail. Splitting
the pure-scroll path off the geometry reconcile removed every whole-rail
`refresh(layout=True)` from a wheel gesture (8 → 0 over 8 frames), but a probe
counting `Screen._refresh_layout` calls showed the gesture still cost 11 screen
layout passes where it should have cost 9. The residue was the copy repaint
itself: `Static.update(content)` takes `layout: bool = True` and calls
`self.refresh(layout=layout)`, so painting two characters into a slot whose
height is pinned by inline styles still scheduled a view layout — twice per
gesture, once entering and once leaving the bottom.

```python
def update(self, content: VisualType = "", *, layout: bool = True) -> None:
    ...
    self.refresh(layout=layout)          # layout=True unless you say otherwise
```

Pass `layout=False` when the slot's size cannot change (height pinned at
compose time, width container-driven), and keep a test that holds the pinned
geometry to account (`assert hint.region == hint_region` across the gesture) so
the assumption cannot rot into stale geometry. The same applies to the
`Static.content` setter, which is `refresh(layout=True)` with no opt-out — and
note that `content` is also the cheapest way to read back what was last painted,
which is how that repaint skips a no-op write without a shadow copy that every
other writer would have to remember to invalidate.

**Recurred twice more, so it is now a guarded census — TASK-21595, 2026-08-25.**
After TASK-21692 (the composer blink: 396 `Widget.arrange` per 6 ticks on an
*idle* composer) and TASK-21134 item 7 (media-viewer match-nav), a package-wide
AST census of every repeating-clock root found two more: `SplashScreen.
_update_animation`, repainting a full-viewport `Static` at the shipped cards'
0.01–0.1 s `animation_speed` — **10–100 whole-screen reflows per second during
startup** — and `PersonaBuddyWidget._paint_frame` at the pet's own frame rate.
Both measured 20 → 0 `_refresh_layout` / `reflow` / `arrange` per 20 ticks.
`Tests/Architecture/test_timer_path_static_update_inventory.py` now rebuilds
that census on every run and fails any clock-reachable `.update(` that neither
passes `layout=` nor carries a stated exemption. Two things that census taught:

1. **Grepping `set_interval` is not a census.** One of the two clocks is a
   `set_timer` callback that re-arms *itself* — an interval spelled as a chain
   of one-shots. Enumerate roots structurally (including the self-rearming
   one-shot), then walk the call graph; the repaint is usually four to six hops
   from the timer, in a different module.
2. **`layout=False` is only sound where the box cannot be content-sized, and the
   pin may not be in the stylesheet.** The Persona Buddy frame's real pin is
   `frame.styles.width/height = "100%"` assigned *inline* by `_apply_geometry`,
   which beats every sheet — so mutating its CSS to `auto` (even rebuilding the
   generated bundle) left the geometry test green, a surviving mutant that was a
   finding about the test. Prove the claim with an A/B against the layout engine
   instead of reading CSS: paint the content with `layout=True` and record the
   geometry, *scrub* with a deliberately different shape (without the scrub the
   second arm inherits the first arm's geometry and passes vacuously), then
   paint the same content with `layout=False` and compare — carrying sibling
   regions and painted per-row cell widths, not just `outer_size`.

---

## `set_timer(0.0)` never fires — silently

**TASK-21110, 2026-08-23.** The splash/initial-screen overlap is armed with
`self.set_timer(SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS, ...)`. A/B-ing that
delay, the "0.0 s" arm looked like the best of both worlds: the boot-time win with
none of the splash-animation stutter the 0.2 s arm showed. It was measuring nothing.
Textual 8's `Timer._run` computes `count = int((now - start) / _interval + 1)`, so a
zero interval raises `ZeroDivisionError` **inside the timer's own asyncio task**. Nobody
retrieves that task's exception, so there is no traceback in normal operation, no log
line, and no callback — the arm's `import_on_loop` was still 430 ms, i.e. the pre-import
had never happened at all. Reproduced standalone in nine lines:

```python
class A(App):
    def on_mount(self):
        self.set_timer(0.0, lambda: fired.append("zero"))   # never runs
        self.set_timer(0.2, lambda: fired.append("two"))    # runs
# -> fired == ["two"], plus an un-retrieved ZeroDivisionError task
```

**What to do.** Treat 0 as an invalid `set_timer` delay. If "as soon as possible" is
what you mean, use `call_after_refresh` (or `call_later`); if a constant feeds the
delay, branch on `> 0` rather than trusting whoever edits it next, and keep a test that
the zero case still schedules. And when a perf arm comes back looking free, check that
the thing you were measuring actually ran before you believe it.

---

## Monkeypatching an `@on`-decorated handler on the class does not patch it

**TASK-21110, 2026-08-23.** An instrumentation probe wrapped
`TldwCli.on_splash_screen_closed` to timestamp splash close, and recorded the splash
closing at 6.09 s on a boot where it demonstrably closed at 3.53 s. Textual's
`_MessagePumpMeta` snapshots `@on`-decorated handlers as **raw function objects** into
`cls._decorated_handlers` at class-creation time, so a later class-attribute assignment
is invisible to that dispatch — the original still ran. Worse, the naming-convention
fallback in `_get_dispatch_methods` skips a method only when it carries `_textual_on`,
which the replacement did not, so the wrapper was *also* dispatched, a second time,
much later. One handler, two invocations, and a timestamp from the wrong one.

**What to do.** Do not patch a method that is both `@on`-decorated and named
`on_<message>`; instrument the thing it calls, or the message's own sender (here,
`SplashScreen.close`). Calling such a handler directly from a test is fine — it is only
class-level replacement plus framework dispatch that splits in two.

---

## Related

- `lessons-testing-evidence.md` — includes the Pilot-harness traps (detached widget
  references after recompose, bare-`App` harnesses that never load the app stylesheet)
- `lessons-live-verification.md` — why a green suite can still miss live-only defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps
