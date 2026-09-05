# Library screen decomposition — the recipe

Source documents (read these first; this doc assembles their operative
rules into one checklist-shaped reference for whoever runs the next
subsystem series):

- Plan: `Docs/superpowers/plans/2026-09-01-library-decomposition-foundation.md`
- Spec: `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md`
- Parent doctrine: `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
  (approved rev 3) and `DESIGN.md` §7 — the One Rule, the One Home Rule
  (`UI/Library_Modules/`), the Six Migration Rules, and the dependency-naming
  canon this doc cites by example below.

Everything here is Library-specific application of that doctrine, not a
restatement of it.

## 1. The per-subsystem PR series

Each of the eleven subsystems below (see §8) ships as a small series of
pure-move PRs, always in this order:

1. **State PR** — a `Library<Subsystem>State` dataclass in
   `UI/Library_Modules/library_<subsystem>_state.py` holding every field the
   subsystem exclusively owns, moved verbatim (identical defaults; computed
   defaults become constructor arguments so `__init__` evaluation order is
   preserved). The screen keeps every original attribute name alive as a
   generated getter/setter `@property` shim, between sentinel comments,
   pointing at `self._<subsystem>_state.<field>`.
2. **Controller PR(s)** — one controller per PR (Six Migration Rules, rule
   1; never batch two subsystems, never batch two controllers of the same
   subsystem into one PR). Each controller owns a cluster of the subsystem's
   moved methods under their original names; the screen keeps one-line
   delegators for every externally-referenced name (`@on` handlers,
   `action_*`, anything a test reaches directly).
3. **Cleanup PR** — the one PR type allowed to edit tests. Deletes the
   sentinel-wrapped shim block, retargets remaining screen-side references
   to the state object directly, deletes delegators nothing external still
   reaches (prove deadness with `grep -rn "<delegator>" Tests/ tldw_chatbook/`
   per delegator before deleting it), retargets test attribute paths and
   patch targets with **assertions kept byte-for-byte**, and lowers the
   `_BUDGETS` ratchet row (§6) to the post-cleanup measurement.

### The byte-for-byte canon

Moved method bodies are **never edited** — not even to retarget a call or an
attribute. Every name a moved body references that is not the controller's
own state is rebound in the constructor, under the *same name* the body
already used, so the body reads identically before and after the move. Two
binding kinds only (a third kind — "reach through `screen` because the
target has no controller of its own yet" — is retired, not available to
this plan):

1. **Framework services** (`run_worker`, `post_message`, `set_timer`,
   `set_interval`, `is_mounted`, `call_after_refresh`, …) are live-read from
   the screen via `@property` on every access — never snapshotted. A value
   captured once at construction goes stale the instant a test replaces the
   attribute on the screen instance afterward.
2. **Everything else** the body depends on that is not its own state is a
   **named constructor dependency** — a controller's dependencies are its
   signature, discoverable by reading the constructor, not by reading every
   `@property` on the class. Each is a callable the *caller* (the screen)
   constructs to close over the screen's own attribute lookup at *call*
   time, not at construction time — this is exactly why a monkeypatched
   name keeps working after the move: the lambda re-reads `self.<name>`
   (or the successor controller) on every invocation.

The canonical worked example is `ConsoleDictationController.__init__`
(`tldw_chatbook/UI/Console_Modules/dictation.py:659`), whose own docstring
states this rule in the words above. Read it before writing the first
Library controller constructor — every Library controller's constructor
should be recognizable as the same shape.

## 2. Field ownership — the authoritative script, and the ≥2-subsystems rule

**Do not hand-list which fields belong to a subsystem.** Compute exclusive
ownership mechanically, per subsystem, with this script (verbatim from the
plan's Task 6 Step 1 — swap the `conv_fields`/`OTHER_SUBSYSTEM` prefixes for
the subsystem in hand):

```python
.venv/bin/python - <<'PY'
import ast
from collections import defaultdict
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
def attrs(m, store_only=False):
    for node in ast.walk(m):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if not store_only or isinstance(node.ctx, ast.Store): yield node.attr
init = next(m for m in methods if m.name == "__init__")
fields = set(attrs(init, store_only=True))
conv_fields = {f for f in fields if f.startswith(("_library_conversation", "_conversation"))}
OTHER_SUBSYSTEM = ("_library_notes","_library_media","_library_prompt","_library_skill","_library_ingest","_library_export","_library_collections","_library_rag")
for f in sorted(conv_fields):
    users = [m.name for m in methods if m.name != "__init__" and f in set(attrs(m))]
    non_conv = [u for u in users if "conversation" not in u]
    tagged = [f"{u} ({next((p.lstrip('_') for p in OTHER_SUBSYSTEM if p in u), 'shell/plumbing')})" for u in non_conv]
    print(f"{f}: non-conversation users={tagged or 'NONE'}")
PY
```

Classification of the output:

- `NONE` → moves into the subsystem's state object.
- Non-subsystem users that are **shell/plumbing methods** (rail switch,
  snapshot apply, shell-state build) → still moves; shims keep them working
  and that subsystem's cleanup PR retargets them.
- Non-subsystem users belonging to **another subsystem** (name matches
  another subsystem's prefix) → **stays on the screen as shared shell
  state**, accessed via a named dependency callable, never forced into a
  subsystem state object. Record each such decision in this doc's
  per-subsystem table entry when that subsystem's series lands.

**The ≥2-subsystems rule:** a field referenced by two or more subsystems is
shared shell state by definition and is never moved by an extraction PR.
Known examples from the spec: `_library_selected_row_id` (226 refs),
`_library_lifecycle` (83), `_library_snapshot_state_generation` (35),
`_pending_library_source_open` (29).

## 3. Monkeypatch-name routing — do not move a patched name until cleanup

**Any name a test monkeypatches on `LibraryScreen` keeps its whole call
graph routed through the screen** until that subsystem's cleanup PR
retargets the tests. This prevents the "monkeypatch bypass breaks tests
inside a pure move" failure: a moved internal call silently starts
resolving free names through its *new* module's globals instead of the
screen's, so a test's `monkeypatch.setattr(LibraryScreen, "<name>", ...)`
stops reaching the call it used to intercept.

**The known list is four names, not three** (PR 0a's execution corrected
the plan's draft "trio" framing once the tests were actually read):
`_list_local_source_snapshot`, `_refresh_local_source_snapshot`,
`_apply_local_source_snapshot`, `_refresh_library_note_detail`. These stay
screen-routed for the reason the spec gives — the trio (plus the note-detail
refresh) is shared shell infrastructure feeding notes+media+conversations
counts, has dozens of internal call sites, and is monkeypatched directly
across `Tests/UI/test_library_entry_compose_once.py`,
`Tests/UI/test_library_prompts_canvas.py`, and `Tests/UI/test_library_shell.py`.

**A second, distinct failure mode surfaced by PR 0a: module-globals
coupling.** `_INGEST_OPTIONS_CACHE_ATTR`, `_read_library_ingest_options_from_config`,
and `_library_ingest_options_for` are module-level `FunctionDef`s that read
each other via Python's ordinary free-variable resolution — which binds to
the *defining* module's `__globals__`, fixed at definition time, not to
wherever the function is later re-exported. `_library_ingest_options_for`
internally calls `_read_library_ingest_options_from_config`; several tests
(`Tests/UI/test_library_ingest_options_cache.py`,
`Tests/UI/test_library_screen.py::test_load_ingest_options_from_config`, the
three `test_task_33*_options_round_trip_persisted_config` tests) patch
`get_cli_setting` / `_read_library_ingest_options_from_config` **on the
`library_screen` module object**. While both functions live in
`library_screen.py` the patch reaches the internal call; move only
`_library_ingest_options_for` (or move both to a *different* module and
leave a bare re-export) and the internal call keeps resolving the free name
through its own module's globals — the patch is silently bypassed and 5
tests fail deterministically. The fix is not a shim: both functions had to
stay together, in `library_screen.py`, permanently (see
`tldw_chatbook/UI/Screens/library_screen.py:755` and
`tldw_chatbook/UI/Library_Modules/screen_helpers.py`'s module docstring for
the full trace).

**The general rule this generalizes to:** before moving any function (not
just a method — this bit module-level `FunctionDef`s, which the per-method
"@on stays, everything else can move" mental model does not cover), `grep
-rn "<name>" Tests/` for direct monkeypatching of it or of any name it
reads from module globals. A hit on either means: keep the whole
call-graph on its current module until that surface's cleanup PR proves the
patches have been retargeted.

**Three more bypass shapes, found by the conversations exemplar (§11 has
the full incident write-ups): unbound fake-`self` calls, an instance-
attribute monkeypatch on a real object, and — the one that isn't a test at
all — a shared helper resolving a subsystem's attribute name at RUNTIME
via `getattr(screen, f"...")`/a dict-of-name-strings. None of these are
"monkeypatching a name" in the literal sense this section's `grep`
recommends, so that grep alone will not find them. §11's per-shape fix
recipe (leave the method real, retarget the fixture to match, or route the
dynamic lookup through `operator.attrgetter`) is the accommodation for
each.**

**One more shape, and its exception to "defer to cleanup": a hardcoded-
file-path census, not a monkeypatch bypass, ships red.** A test that
AST-walks a specific source file by path (e.g. `library_screen.py`) for a
call/name it expects to still live there is not a monkeypatch-routing
hazard in the sense above — it does not silently pass while checking
nothing, it goes loudly RED the moment the move lands, because the census
over the old file now finds zero matches. That distinction matters for
timing: every other bypass shape in this section is deliberately left for
the subsystem's cleanup PR to retarget, because until then the test still
*passes* (just without exercising what it thinks it does). A path-census
guard has no such grace period — it is red at the very commit boundary
that moves the code, so the no-red-ships precedence wins: retarget the
census in the SAME PR-stage that moved the code (to the new file, or to
whichever set of files best expresses the invariant), not deferred to
cleanup. Cleanup still handles the rest of that subsystem's routing work
as usual.

**A sixth bypass shape, found by the skills series (wave-4 task 2): bare
`self` used as an IDENTITY-COMPARED or SCREEN-IDENTITY argument, not
merely as an attribute-lookup receiver — and its close cousin, an
unbound-attribute escape that silently takes a `getattr` default.** Every
prior shape in this section either fails loudly (a real exception) or is
a test-only bypass safely deferred to cleanup. This shape is neither: it
is a genuine, silent, PRODUCTION behavior change a pure move introduces,
found only by running the full battery (a same-name-forwarding regex and
a byte-for-byte body diff cannot see a semantic identity bug or a
literal-string `getattr` that never resolves).

- *Identity-compared/screen-identity `self`, three confirmed instances,
  two exclusion-worthy forms:*
  - **Form A — a framework API's own identity filter.** Textual's
    `WorkerManager.cancel_group(node, group)` filters `worker.node ==
    node` by identity. A moved body calling `self.workers.cancel_group(
    self, ...)` compares the CONTROLLER against workers actually
    registered with the SCREEN as their node (since `run_worker` always
    forwards to `self._screen.run_worker(...)`) — permanently `False`,
    silently making the cancellation a no-op. Found by reading the
    framework's own source before moving, not by a test failure.
  - **Form B — a shared shell helper's own screen-identity check.**
    `_library_screen_is_current(screen)` (`screen_helpers.py`) does
    `current_screen = getattr(screen.app, "screen", screen); return
    current_screen is screen`. A moved body forwarding bare `self` makes
    this permanently `False` (`real_screen is controller` can never be
    true), silently no-opping every guarded branch. Found the hard way —
    a first draft moved four such names, the wiring/ratchet battery
    stayed green, and two real Pilot-driven UI tests (each pressing a
    real button and asserting a real DOM mount) failed, confirmed via a
    paired baseline (both pass on the pre-move tree, fail on the
    four-moved draft).
  - **Form C — the identical Form-B shape, inlined instead of routed
    through the shared helper.** `self.app.screen is self`, found the
    same way — a SECOND draft moved a method carrying this exact
    comparison inline; 8 `Tests/Skills/` tests (the wave's own
    fourth-root trap) failed, confirmed genuine via the same
    paired-baseline method.

  All three exclude the carrying method(s) entirely (keep them
  screen-resident, full-bodied, untouched — no accommodation exists,
  unlike duck-typed attribute access, because identity can never be
  satisfied by a proxy object). Named late-binding dependencies (the
  usual binding-kind-2 shape) cover any remaining mover that calls the
  excluded method.

- *Unbound-attribute escape via `getattr(self, "<literal>", default)` —
  found by a THIRD draft's own post-landing review, not the move's own
  battery.* `getattr` with a literal string name and a default is
  invisible to the recipe's own `self.<attr>` `ast.Attribute` census (the
  name never appears as a literal `self.<name>` expression) AND produces
  no exception when unbound — it just silently returns the default,
  forever. A controller missing the corresponding framework-service
  property (here, `focused`) degrades a real behavior — permanently,
  quietly — with no red test anywhere in the standard battery, because
  `has_focus`-shaped DOM assertions can be satisfied by an UNRELATED
  Textual default-focus fallback landing on the same widget by
  coincidence (confirmed empirically: a `.has_focus`-only assertion
  passed identically whether the property existed or not). **Any future
  subsystem's controller-PR sweep should grep the WHOLE moved-body
  source for `getattr(self, "<literal-string>"` (not just `self.<attr>`
  accesses) and confirm each literal name actually resolves on the
  controller** — a repeat scan after adding a name is cheap and closes
  the whole class for that controller, per the skills series' own
  precedent (found exactly one instance, `focused`, confirmed by re-scan
  after the fix). The covering test for a finding in this class needs a
  signal PROVABLY tied to the code path (e.g. a spy on the exact
  `query_one`/method call the fix's own logic makes), not a bare DOM
  end-state assertion — the same false-negative risk (an unrelated
  framework fallback satisfying the assertion) applies to writing the
  test as it did to finding the bug.

- *A battery-found hazard shrinking the mover set legitimately amends the
  RED tuple — expected, not a smell.* When a controller-PR's own
  verification battery (not the static census) surfaces one of these
  shapes after the RED wiring commit already pinned a cluster-name tuple,
  correcting that tuple (and the accompanying "N of 127" counts) in the
  same commit that lands the fix is the expected shape of the work, not
  evidence the RED commit was wrong when it was written — the census that
  produces the RED tuple is necessarily static, and these shapes are, by
  definition, only found by running code. Re-deriving the exact final
  counts (movers, exclusions, per-category tallies) after every such
  correction — not just patching the number that changed — is what keeps
  the tuple and its own docstring narrative from drifting apart.

**A seventh bypass shape, found by the ingest series' own state PR
(wave-5 task 1) — and unlike every shape above, one a STATE PR can trigger,
not only a controller PR.** A test builds a screen via
`object.__new__(<Screen>)` or `<Screen>.__new__(<Screen>)`, skipping
`__init__` entirely, then hand-sets a flat `_library_<subsystem>_<field>`
name as a PLAIN instance attribute. This is safe before that subsystem's
state PR lands (no property exists yet, so the assignment just creates an
ordinary instance attribute) — but the instant the state PR's generated
`@property` shim exists, the SAME assignment invokes the property setter,
which reaches for `self._<subsystem>_state` to route the value into. A
bypassed-`__init__` instance never had that attribute constructed, so every
such test fails immediately: `AttributeError: '<Screen>' object has no
attribute "_<subsystem>_state"`.

This is categorically different from every prior shape in this section.
Shapes 1–6 above are all either a real exception (caught immediately, at
the point the flawed assumption fires) or a TEST-ONLY bypass that stays
silently green until a CONTROLLER-PR's own method move reaches it later —
which is exactly why the recipe defers fixing them to that subsystem's
cleanup task. This shape fails at the STATE PR itself, for every affected
test, the moment the shim installs — deferring it is not an option; the
no-red-ships rule requires fixing it in the SAME commit that installs the
shim. The fix is mechanical and minimal: one line inserted immediately
after the bypassing construction (`screen._<subsystem>_state =
<StateClass>()`), zero assertions or other lines touched. The ingest
series found 27 such sites across 6 files (`Tests/App/test_submit_library_
ingest_job.py` [5], `Tests/integration/test_library_ingest_flow.py` [2, the
`<Screen>.__new__(<Screen>)` spelling], `Tests/UI/test_library_ingest_
canvas.py` [16], `Tests/UI/test_library_ingest_inline_consent.py` [1],
`Tests/UI/test_library_ingest_retry_last.py` [1], and — the one this
series' own first sweep MISSED and a later review caught —
`Tests/UI/test_parakeet_v2_install_ui.py` [2]); a 7th file,
`Tests/UI/test_library_screen.py`, uses the identical `object.__new__`
STRING in its own docstring prose (describing a PRIOR, already-fixed
bypass, task-3022) but its actual fixture constructs a real
`LibraryScreen(MagicMock())` — a false positive, confirmed by reading the
fixture body, not the grep hit alone.

**The filter-blindness lesson this shape's own miss teaches: the census for
this shape must be a CONTENT grep across the WHOLE `Tests/` tree, never a
`-k <subsystem-name-substring>` name-based sweep.** `Tests/UI/
test_parakeet_v2_install_ui.py` touches `_library_ingest_form` directly
(two Parakeet-v2-install-result tests stage ingest form state to assert it
survives an install completion) but its own filename, its own test names,
and its own `-k` keywords contain neither "ingest" nor "library" — every
`-k "ingest and library"`/`-k "ingest"`-shaped sweep this task ran
collected ZERO tests from it, so the file was never exercised post-move
until a dedicated reviewer pass grepped by CONTENT (`object.__new__(
LibraryScreen)` OR `LibraryScreen.__new__` co-occurring with
`_library_ingest_` — or the equivalent for whichever subsystem is moving —
across every file in `Tests/`, not a keyword-filtered subset) and caught
it. **Any future subsystem's state-PR fix pass for this shape must run
that content-grep repo-wide as its OWN closing verification step**, exactly
the way this entry's own worked example does above — a name-based test
filter is a discovery aid, never a completeness proof, for a shape whose
defining characteristic is that the affected file need not mention the
subsystem's name anywhere a keyword filter would see it.

**A related, separately-recorded process lesson from the same task: an
in-place `git checkout <base-commit> -- tldw_chatbook Tests` baseline
overlay of the CURRENT worktree is interruption-unsafe for any
long-running background sweep.** This task's own first pristine-baseline
attempt used exactly that overlay; a session-usage-limit interruption's
recovery step restored the shared worktree back to `HEAD` while the
overlay's sweep was still reading files in the background, silently
invalidating every result it would have produced with no error of its own.
Redone with an isolated `git worktree add <scratch-path> <base-commit>`
plus its own `uv venv`/`pip install -e ".[dev]"` — a directory no other
recovery action in the SAME worktree can touch. See §7's own sweep-evidence
entry for this task's full before/after numbers; the standing rule this
generalizes to: **an isolated worktree, not a same-tree checkout overlay,
is the default method for any baseline comparison expected to run
unattended for more than a couple of minutes**, regardless of which
subsystem's task is running it.

**RULED at the wave-6 close: the "short foreground run" carve-out is
DROPPED, not written down. The isolated worktree is now UNCONDITIONAL for
every baseline comparison, of any length.** The rule above said "expected
to run unattended for more than a couple of minutes", and TWO tasks read
the gap in it as a licence, three times between them: wave-6 task 1
overlaid the tree for a 2m40s prompts-file baseline, and wave-6 task 2 did
it twice more (§9.3/§9.4 of its own report, ~2m20s and ~1m50s). All three
completed intact and their evidence stands — and wave-6 task 3, next in
the same series, went isolated for every one of its own paired checks
without being told to, naming the editable-finder trap as its reason. The
practice was already converging on isolation; only the written rule
lagged. But task 2 ALSO recorded that it then hit a session usage
limit mid-task — the exact interruption class that produced wave-5's
incident — and escaped losing anything only because by then its long
sweeps were already isolated. The failure mode is live in this
environment, not historical, and it is silent: an overlay clobbered
mid-run produces no error of its own, just numbers that mean nothing.

**Why DROP rather than write the carve-out down with a bound.** The wave-6
close measured what the "expensive" alternative actually costs, rather
than assuming it: a `git worktree add` of this repo (12,866 files) plus
`uv venv` plus `uv pip install -e ".[dev]"` provisions a fully usable
isolated baseline in **5 seconds end to end (3s checkout, 2s venv + install), measured at this close**, of which uv's own share is 2.2s
(1.00s resolve, 1.01s prepare, 177ms install — 101 packages, warm cache);
disk cost is 535 MB. More decisive than the absolute number: **a task
creates ONE baseline worktree and reuses it for every check it runs**, so
the MARGINAL cost of the rule after the first check is zero. The carve-out
was therefore buying nothing that needed buying. Against that, a
five-condition exception ("≤5 minutes, foreground, targeted selection,
no session-limit exposure, verified clean after") is itself a judgement
call in every task's head — and the evidence is that an unwritten version
of exactly that judgement already drifted across two consecutive tasks
without anyone flagging it. A flat rule with no exception is cheaper to
follow than a bounded one, and here it is also nearly free.

**The rule, restated with no exception:** any baseline comparison — a
whole-root sweep, a nine-file paired battery, or a single node id — runs
against an isolated `git worktree add <scratch-path> <base-commit>` with
its own `uv venv`/`uv pip install -e ".[dev]"`. Verify the isolated venv
resolves its OWN tree (`python -c "import tldw_chatbook; print(tldw_
chatbook.__file__)"` must print a path under the scratch worktree — the
editable-finder trap makes a shared venv resolve the wrong tree), create
it once per task, reuse it, and remove it at task close. `git stash -u`
and `git checkout <base> -- <paths>` are no longer baseline methods at any
duration.

**An eighth bypass shape, found by a coordinator review of the ingest
controller PR (wave-5 task 2, fix round 1) rather than by the task's own
battery: a moved body's bare MODULE GLOBAL (an ordinary
`from module import name` in the pre-move file, not a `self.<name>`
attribute) can be patched at the `library_screen`-scoped path by a test
whose OWN ASSERTIONS still pass after the move — a green-but-vacuous
test, never a red one, because the coincidence that makes it pass is
readily available (an unrelated, independently-true condition that
would make the assertion pass whether or not the patch is actually
observed).** This is the SAME underlying mechanism the module-globals-
coupling shape (this section's own oldest entry, and the ingest
series' own `save_setting_to_cli_config`/`_library_ingest_options_for`
precedent) already describes — the free name resolves through the
DEFINING module's `__globals__`, fixed at definition time, so a move to
a different module silently repoints any test patch aimed at the old
module — but this instance was never caught by the standard "read every
test that monkeypatches this name" check the recipe's own §3 opening
already prescribes, because nobody ran that check exhaustively: the
ingest controller PR's own battery (RED→GREEN wiring, characterization,
both ratchets, `-k` sweep, full `Tests/Architecture/`, and a full
sequential xdist paired-baseline sweep) passed cleanly with the hazard
already shipped, because the ONE test it silently defeats
(`Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_
invalid_path_notifies_and_submits_nothing`) asserts a warning fires and
zero jobs submit for an INVALID path — and a bare nonexistent tmp path
fails the REAL validator identically to how it would fail the intended
STUB, so the assertions read as passing either way. Only a review pass
that mechanically re-derived the census (rather than trusting the
battery's own green result) caught it.

**The mechanical census this shape requires, and the method that found
it (recipe-timed at roughly one minute per subsystem once the mover set
is final):**

1. For every method that MOVED (not the excluded ones — their bodies
   never left their original module), walk its body's `ast.Name` nodes
   with `Load` context, excluding `self`, local variables/parameters,
   nested-function names, and builtins, and keep only the ones that
   match a name the NEW module imports at module level (`from x import
   name` / `import x`) — these are the moved body's own bare module-
   global reads, exactly the shape binding kind 1 (framework services)
   and kind 2 (named dependencies) do NOT cover, because both of those
   apply only to `self.<name>` attribute access, never a bare free name.
2. For every name that census produces, grep ALL of `Tests/` — never a
   `-k`-filtered subset, per this section's own filter-blindness rule —
   for every shape a test can spell a `library_screen`-scoped patch
   target in: the direct-attribute form (`library_screen_module.<name>`
   or `library_screen.<name>`), the fully-qualified string-patch form
   (`"tldw_chatbook.UI.Screens.library_screen.<name>"`, the target
   `unittest.mock.patch`/`patch.object` takes), AND the two-argument
   `monkeypatch.setattr`/`patch.object` form (`library_screen_module,
   "<name>"` or `library_screen, "<name>"` — a STRING literal argument,
   not a dotted attribute access, and the one shape the ingest series'
   own first attempt at this exact census missed entirely, undercounting
   3 real patch sites to 0 before the pattern was widened to catch it).
3. For every name with at least one hit, read the hitting test(s) and
   determine whether ANY of them exercises the SPECIFIC moved method's
   own call path (not just the same free name from an unrelated
   caller) — the same "read the test, don't infer from the grep alone"
   discipline every other bypass shape in this section already requires.
4. Classify each finding: an ACTIVE collision (a hitting test's own
   assertions depend, even coincidentally, on the patch reaching the
   moved body) requires excluding the method, exactly like every other
   module-globals-coupling entry in this section; a LATENT collision
   (every hit belongs to an unrelated caller/subsystem, confirmed by
   reading each one) is a documented, deliberate "keep" — record which
   test files were checked and why none applies, so the next reviewer
   does not have to re-derive it from scratch.

**The ingest series' own worked example of both outcomes from the same
census pass:** `_resolve_ingest_source` reads the bare names
`validate_path_simple`/`validate_url`; the census found exactly one
ACTIVE collision (`test_library_shell_ingest_canvas_invalid_path_
notifies_and_submits_nothing`, confirmed genuine by an EXISTING-file
probe — the reviewer's own proposed check, and the general form any
future subsystem's fix should use: temporarily make the stub reject a
path the REAL validator would accept, so the assertion can only pass if
the patch is actually observed, then confirm it fails pre-fix and
passes post-fix; the probe itself is never committed, only its result)
— excluded, reverted to the screen byte-for-byte, its one mover caller
rebound through a named late-binding dependency. `_apply_library_
ingest_backend_save` reads the bare name `_sync_library_canvas` (the
shared cross-subsystem canvas-sync dispatcher every one of the six
controllers to date imports the identical way); the census (task 2) found
~20 hits across 7 files, ALL confirmed LATENT (none reaches this mover's
own call path, each read to confirm) — kept as a mover, verdict recorded
in the controller's own module docstring, and the SAME shape's likely
presence in the other five controllers flagged as a cross-wave follow-up
census, not fixed retroactively in this task (each of those five
controllers' own battery already passed; re-opening five landed PRs to
chase a shape found by inspection, with no confirmed active collision, is
exactly the "don't fix cross-wave here" the review scoped this to).
**Correction (wave-5 task 3, the ingest cleanup PR, carried forward from
task 2's own re-review as a recorded minor):** the 7-file/~20-hit count
undercounted by 3 files whose patch sites used a variable name other than
`library_screen`/`library_screen_module` — a fixed-string grep for those
two exact identifiers cannot see an import aliased to something else
entirely. `test_library_canvas_scoped_sync.py` (`library_screen_module`
inside a multi-line `patch.object(...)` call a single-line-oriented grep
does not span), `test_library_notes_reader.py` (`library_screen_module`,
patched via `monkeypatch.context()`'s own `patcher.setattr(...)` rather
than a bare `monkeypatch.setattr(...)`), and `test_review_set_walker.py`
(imports the module under the alias `screen_module`) bring the true count
to **10 files, 38 sites** (**"site" defined, post-review fix round 1**:
one match, one line, of this section's own 3-shape pattern set — direct-
attribute form, fully-qualified string form, or the two-argument
`monkeypatch.setattr`/`patch.object` form — deduplicated by line number
within a file; the reproducible per-file breakdown summing to 38 is
recorded in `library_ingest_controller.py`'s own module docstring) — all
still confirmed LATENT with respect to
`_apply_library_ingest_backend_save`'s own call path (one test in
`test_library_canvas_scoped_sync.py` DOES exercise an Ingest-named
handler, `handle_library_ingest_option_value_changed`, but that method is
one of the 9 unbound-fake-self EXCLUSIONS — never moved — so its own bare
`_sync_library_canvas` call still resolves through `library_screen.py`'s
own globals regardless; not a collision with anything this controller
owns). No verdict changes; the correction is to the recorded evidence
only. **The lesson this adds to the census's own method (step 2 above):
a fixed-string grep for `library_screen`/`library_screen_module` is
insufficient by construction — the census must first find every name a
test file imports `tldw_chatbook.UI.Screens.library_screen` (or `from
tldw_chatbook.UI.Screens import library_screen`) AS, then search for
THAT alias, not assume the two conventional spellings are the only ones
a test will use; a `monkeypatch.context()`'s own `patcher.setattr(...)`
and a bare `mock.patch.object(...)` are both equally valid call shapes a
single `monkeypatch.setattr(` -anchored search will miss.** **The general
rule: this census is a MANDATORY step of every future subsystem's
controller-PR sweep, run to completion (not stopped at the first hit,
not skipped because the ordinary battery came back green) — an ordinary
green battery proves nothing about this shape's absence, precisely
because a green-but-vacuous test is indistinguishable from a genuinely
passing one without reading it.**

**A census SPELLING the `monkeypatch.setattr`-anchored search misses: the
patch-target TABLE row.** This is not a ninth bypass shape — the shape
catalogue above stands at eight — it is a fourth SPELLING that every one
of those censuses has to search for, and the one an obvious regex cannot
see. A test can hold its patch targets in a data structure and apply them
at a distance:

```python
for target, name, key in (
    (screen, "_load_library_reader_preference_snapshot", "snapshot"),
    (screen, "_request_library_prompts_browse", "load"),
    (rail, "refresh", "rail_refresh"),
):
    monkeypatch.setattr(target, name, probes[key])
```

The exemplar is `Tests/UI/test_library_shell.py:5146`. The method name
appears **only** as a bare string literal inside a tuple, many lines above
the single `monkeypatch.setattr(` that consumes it — so a search anchored
on `monkeypatch.setattr` (or on `patch.object`, or on `setattr(screen,`)
scores it ZERO, and so does a `<name>\s*\(` call-shaped regex. The three
spellings §3 already enumerates (direct-attribute, fully-qualified string,
two-argument `monkeypatch.setattr`) all assume the name sits on the same
line as the call. This one does not.

**The rule:** every census in this section — the monkeypatch-routing check,
the module-globals census, and the cleanup PR's delegator-prune census
alike — must search for each candidate name as a **bare quoted string
anywhere in the file**, independent of what call surrounds it, then read
the hits to see whether a distant `setattr`/`patch` consumes them. The
prompts series ran it and got a real hit
(`_request_library_prompts_browse`) which happened to already be one of
its 22 exclusions, so nothing changed — a clean result from a census that
was actually run, not a census that was skipped because the obvious
spelling came back empty.

**A cleanup-PR census this section did NOT previously require: the deleted
FIELD names' own prose sweep.** Every cleanup PR to date censused the
PRUNED METHOD names for stale prose. None censused the DELETED FIELD
names, even though the shim-block deletion makes all of them dead in
exactly the same way and at exactly the same commit. The gap produced a
review finding in the prompts series: `library_export_controller.py:375`'s
constructor docstring still said the accessor "Reads
``LibraryScreen._library_prompts_mutation_in_flight``" — an attribute that
no longer exists on any object — in a file belonging to a DIFFERENT
subsystem, which is why no prompts-scoped file sweep could reach it. This
class is invisible to every automated guard in the battery: nothing
imports or executes a docstring.

**The census, and the one distinction that makes it usable:** tokenize the
tree (COMMENT and STRING tokens only, so a live attribute access cannot
masquerade as prose), search for every deleted flat field name AND every
pruned method name, over `tldw_chatbook/`, all of `Tests/`, `Docs/`,
`backlog/`, `scripts/` and `Helper_Scripts/`. The prompts series' run
returned 54 field-name occurrences; almost all were legitimate. **A raw
count is useless without separating three classes**, and only the third is
a defect:

1. **Self-labeled relocated references** — the prose names the old flat
   spelling but immediately says so ("``_library_prompt_detail`` and
   ``_library_prompt_block_state`` ... two of the moved
   ``LibraryPromptsState`` fields", `test_library_prompts_
   characterization.py:268-269`). Correct as written; leave alone.
2. **Past-tense incident narrative, comparative prose from other
   subsystems, and the wiring test's own literal-string pins** — all
   correct by construction; leave alone.
3. **Present-tense mis-attributions** — prose that says the name lives on
   `LibraryScreen`/`library_screen` today. These are the defects, and the
   fix is always one phrase, always line-neutral.

A bare name-count cannot tell class 1 from class 3 — the two spell the
identical name — so the sweep must read each hit. Narrowing to hits with a
screen attribution within ±3 lines is what makes that reading tractable:
the wave-close run over BOTH name sets (43 fields + 39 pruned methods, vs.
the cleanup task's own field-only pass that returned the 54 above) cut
**145 raw prose lines to 8 candidates, of which 6 were real defects**.
Four of those 6 were disclosed by the cleanup task's
own review (two in-scope, two forward-noted as out-of-scope); the other
**two were found only by the wave-close sweep, in files no task's ruled
file scope had ever covered** (`tldw_chatbook/Library/library_shell_
state.py:44` and its mirror `Tests/Library/test_library_shell_state.py:371`,
both "the screen distinguishes ... by view state (`_library_prompts_view`)").
That is the argument for making this a standing census rather than a
review's incidental catch: two of six defects lived outside every file
scope any task in the series had reason to open.

## 4. The transform whitelist

An extraction PR may contain **only**:

- Verbatim block moves (method/function bodies byte-for-byte, per §1's
  canon).
- Import-path changes.
- Constructor/property bindings — the two binding kinds in §1, plus
  generated controller-local `@property` accessors for the subsystem's own
  state fields (same generator shape as the state-PR shim generator, §1
  step 1, applied to the controller instead of the screen).
- Screen-side delegator one-liners for externally-referenced names
  (`@on`-decorated / `action_*` / `on_<message-handler-name>` / anything a
  test reaches directly) — the `@on(...)` decorator line stays on the
  screen, copied verbatim. The three framework-dispatched classes are
  enumerated below; they are the ones a reference census cannot see.

Nothing else. No renames, no logic edits, no cleanups, no "while I'm here"
fixes — those land as separate, attributable changes. Receiver
normalization (direct `self._state.` access, dropping screen-routed
same-subsystem hops) is explicitly deferred to that subsystem's cleanup PR;
it never happens in a move PR.

This whitelist supersedes an earlier draft idea from this plan's own
drafting — a bespoke "receiver-rewrite transform whitelist" that would have
permitted rewriting internal call sites during the move itself. It was
dropped in favor of the stricter byte-for-byte body discipline the
dictation extraction (§1) demonstrated: bodies unedited, names rebound only
in the constructor. See the spec's "Relationship to the parent doctrine —
deltas, declared" section for the full list of drafting ideas that lost to
doctrine precedent.

### The delegator-prune whitelist — THREE members, not two

Every cleanup PR prunes the screen delegators whose names have zero
references anywhere in the repo. A **whitelist of framework-dispatched
names is exempt from that census outright**, because the framework calls
them and no source line ever spells them. Through the ingest series that
whitelist had two members. **It has three.**

1. **`@on`-decorated handlers.** The decorator registers the method in
   `cls.__dict__["_decorated_handlers"]`; nothing names it.
2. **`action_*` handlers.** Textual resolves these from binding/link
   strings, not from Python call sites.
3. **`on_<message-handler-name>` NAME-dispatched handlers.** *(New in the
   prompts series; this is the one that will bite media and notes.)*
   Textual computes `Message.handler_name` as `f"on_{name}"` at class
   creation (`textual/message.py:86`) and `MessagePump._get_dispatch_
   methods` walks `self.__class__.__mro__` looking that exact string up
   (`textual/message_pump.py:743-758`; measured against the installed
   Textual 8.2.8). Such a handler carries **no decorator at all**, so a
   decorator-driven whitelist misses it, and **no source line anywhere
   spells `screen.on_<...>`**, so a reference-count census reports it as
   zero-referenced and marks it PRUNE.

**The incident.** The prompts cleanup's census marked all six
`on_prompt_block_editor_*` handlers PRUNE on a genuine zero-reference
count. Deleting them would have silently unhooked `LibraryScreen` from six
live messages — and, worse, **the deletion would have stayed GREEN**: the
`PromptBlockEditor` canvas widget defines its own two same-named handlers
one level down and keeps handling those two messages, and a harness in
`Tests/UI/test_prompt_block_editor.py` defines one for each of the six. A
naive census that counts "the name appears in the repo" therefore finds
hits — 8 same-named `def`s across the two files — but every one is a `def`
in a DIFFERENT class, which is the exact opposite of a caller. Neither the
presence nor the absence of hits told the truth here.

**The rule:** before pruning any moved name matching `on_[a-z]`, resolve it
against the framework's NAME-based dispatch, not against the reference
census — construct the `Message` subclass it would receive and confirm
`handler_name` matches, or confirm no `Message` in the subsystem's own
widgets produces that string. Treat a same-named `def` in another class as
evidence of NOTHING; a caller is a call site, not a definition. The
symmetric question is worth asking too: the class is not Textual-specific
— any framework that dispatches by computed name (a `visit_<Node>` walker,
a `handle_<verb>` router) defeats a reference census the same way.

## 5. Rollback, not fix-forward

A landed extraction implicated in a regression is **reverted**, not fixed
forward. Pure moves revert cleanly, and single-candidate attribution — "this
one commit changed nothing but where code lives, so it is the only
candidate if behaviour changed" — is the property the pure-move policy
exists to buy. Fixing forward reintroduces exactly the ambiguity the policy
was designed to remove.

## 6. Measure after final rebase, lower budgets in the landing PR

This plan's Global Constraints state it directly: **this file churns ~14
commits/day.** Never trust a line-count or method-count number carried over
from an earlier point in planning or review — every task locates its
material and re-measures with the provided scripts at execution time.
**Rebase onto latest `origin/dev` immediately before each PR's final
measurement**; budgets (`_BUDGETS` in
`Tests/Architecture/test_screen_size_ratchet.py`) are measured *after* that
rebase and lowered *in the landing PR itself*, never in a follow-up. Console
wave 3 landed red twice from stale-base numbers before this rule was
adopted — see that file's own module docstring for the incident this
mechanism exists to prevent.

**A re-pin has THREE places the number lives INSIDE the guard file, and a
passing guard proves only ONE of them — plus a fourth, outside it, that can
never be corrected.** The enforced `_BUDGETS` row, the arithmetic
derivation in the comment above it, and the narrative prose line at the top
of that same comment block are three independent copies of the same figure;
nothing in the battery reads the last two. The fourth is the commit
message. Wave-6 got two of the four wrong, in two consecutive commits.

*Incident (wave-6 task 2).* A late fix — a 4-line function-local import
added to a static-method delegator — moved the screen's measured line count
from a first draft's 37718 to the true 37722. The row was corrected. The
`-4061 + 333 + 3 + 88 = -3637` derivation in the same comment was
corrected. The narrative line three lines above that derivation
(`test_screen_size_ratchet.py:503`, "Fresh `_measure()`: 41359/1321 ->
37718/1321") was NOT, and shipped stale eighteen lines above a row pinned
at 37722, with the guard green the whole time. Review caught it, not the
battery. **When a measurement changes mid-task, `grep` the guard file for
the OLD number before committing** — correcting two of three sites and
assuming the third feels identical to correcting all three.

*Same wave, one commit later:* `6fd2b753a`'s COMMIT MESSAGE states
"37722/1321 -> 37576/1282 … line delta -146" for a commit whose row, whose
comment, whose report and whose live `_measure()` all say **37574/1282,
delta −148**. It was written against an intermediate measurement taken
before the commit's own final edit and never re-derived. It is deliberately
NOT amended: `.git-blame-ignore-revs` records that literal hash, and
amending would orphan the entry (§10). **A commit message is a fourth place
the number lives, it is immutable once pushed, and it is the one place a
correction cannot be applied — so re-derive it LAST, after the final edit,
not from the draft you started the commit with.** The correction of record
lives in the task report and the PR body instead.

**The same discipline applies to LINE RANGES, and a "re-verified" claim
that overrides a correct value is worse than no claim at all.** *Incident
(wave-6 task 2, fix round):* a review note cited
`test_library_modal_dismissal.py:520-526` for the `_SUPPORTED_OWNER_SCOPES`
tuple; the fix round's own "re-verification" replaced it with `:518-524`,
a range that spans two blank lines, misses the fifth `_OwnerScope` entry
and misses the closing paren. The original was right. Two independent
readers had to re-check the live file to overturn the correction. A range
is a measurement — read it off the file at the moment you write it, and
never on the strength of having "just looked."

**And a range is only ever true of a stated revision.** That same tuple
sits at `:528-535` TODAY, because the prompts cleanup added a sixth
`_OwnerScope` and shifted the block. Both `:520-526` and `:528-535` are
correct; neither is correct without saying which tree it was read from. A
line range written without its revision is a number with a silent
expiry date — cite the commit, or cite the symbol and let the reader find
it.

## 7. Sweep evidence — xdist + paired baseline, not the literal CI command

The full regression net for a Library change is `Tests/UI -k "library"`
(4,200+ tests). **The literal single-process command,
`.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q`, is
CI's job, not this recipe's per-task evidence requirement** — a
single-process attempt of the full sweep runs for roughly an hour or more
(observed: ~59 minutes before hitting a session's own time budget, still
genuinely executing the whole way per stack-sample profiling, not hung).
Waiting on that per task makes the recipe unusable for the fast, frequent
PRs this plan calls for.

**The accepted per-task evidence is an xdist sweep paired with a
pristine-baseline comparison**, run at execution time:

```bash
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal
```

(the exact invocation used for PR 0a's Task 1 evidence; `-n 8 --dist
worksteal` is already available in the worktree venv). Procedure:

1. Run that command on your branch; capture the pass/fail counts and the
   set of failing test names.
2. Run the **identical** command against the pristine merge base in §3's
   isolated baseline worktree (`git worktree add <scratch-path>
   <base-commit>` with its own `uv venv`) — same `-n`/`--dist` config, so any
   parallelization-only flakiness is present in both runs equally.
3. Diff the two failure-name sets. `pytest-xdist` itself introduces
   real, non-deterministic ordering/parallelization flakiness in this
   suite (CSS-geometry and terminal-size-sensitive Pilot tests are not
   uniformly safe under heavy parallel load) — expect a handful of tests to
   flip in *both* directions between any two xdist runs, baseline included.
   **Only failures unique to your branch's run (absent from the baseline
   run) count against the task.** A failure present in both runs, or only
   in the baseline run, is pre-existing or run-to-run noise, not evidence
   of a regression.
4. For each failure unique to your branch, re-run it directly,
   single-process, in isolation and combined with the other unique
   failures, before concluding it is a real regression rather than
   xdist-specific ordering/shared-state flakiness that happened to land on
   your branch's run and not the baseline's.

This is the same technique and the same command PR 0a's Task 1 used to
surface and confirm a real, deterministic regression (7 tests, 100%
reproducible in every combination) against a backdrop of ~330+ ordinary
xdist-noise failures neither run's raw pass/fail count could have
distinguished on its own. Report both counts and the diffed unique-failure
list as the task's evidence; a CI run of the literal single-process command
remains the authoritative confirmation once time permits, but is not
required per-task.

### Documented pre-existing failures (do not re-derive these)

Tests confirmed, by at least one Library-decomposition task, to fail
identically on a pristine baseline (§3's isolated worktree at the pre-task
tree; older entries below record the `git stash -u` method §3 has since
retired) and therefore not attributable to any extraction/cleanup PR. Check this
list before spending time re-proving one of these is pre-existing; add to
it (with the task that found it) rather than letting the next series
rediscover the same red from scratch.

- `Tests/UI/test_library_content_hub.py::test_library_conversations_empty_state_is_honest_and_blocks_actions`
- `Tests/UI/test_library_shell.py::test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-conversations-#library-conversations-reader-shell]`
- `Tests/UI/test_library_shell.py::test_library_conversations_reentry_preserves_applied_page_and_query`
- `Tests/UI/test_library_shell.py::test_library_conversations_reentry_does_not_load_when_dirty_editor_vetoes`
  (all four found by Task 7, reconfirmed by Tasks 8 and 9)
- `Tests/UI/test_library_selection_updates.py::test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`
  (found by Task 9; confirmed identical on both HEAD and a pristine
  baseline via `git stash -u` + rerun. **Not selectable by the
  `-k "conversation and library"` filter** — its name contains neither
  "conversation" nor "conversations" — and invisible to the full xdist
  sweep's paired-baseline *diff* specifically because it fails on both
  sides equally, so it's absorbed into the shared ~330+-failure backdrop
  rather than surfaced as a unique-to-branch or unique-to-baseline name.
  Only a direct per-file run surfaces it; run each retargeted test file
  individually, not just the aggregate sweeps, and don't summarize that
  check as "all green" without checking every file's own result.)
- `Tests/Architecture/test_screen_size_ratchet.py`'s two `chat_screen.py`-scoped
  rows (`test_screen_does_not_grow_past_its_budget[chat_screen.py]`,
  `test_task_22507_4_does_not_worsen_chat_screen_base`) — concurrent,
  unrelated `chat_screen.py` growth from other work on `dev`; reconfirmed
  pre-existing by every task in this series via the same `git stash -u`
  method.
- `Tests/Architecture/test_library_modules_size_ratchet.py::test_controller_
  does_not_grow_past_its_budget[library_media_browse_controller.py]` — the
  file is 410 lines against a 371 pin. Found by wave 6's `origin/dev`
  reconciliation merge and confirmed pre-existing in §3's isolated baseline
  worktree at `origin/dev` (`dfe45fbe6`): the SAME 410-line file and the
  SAME 371 pin exist on both sides, and the stale pin already predates this
  wave's merge-base (`7aa048790`), so no Library task introduced it. It is
  dev-side creep, and this test's own guidance forbids re-pinning it to
  green from a passing branch — leave it red and fix it where it was
  written. The sibling failure the same merge surfaced,
  `test_every_controller_file_has_a_budget_row` (dev landed `library_
  navigation_controller.py` and `library_character_repair_controller.py`
  with no `_BUDGETS` rows), IS fixed at that merge, because governing a new
  file at its exact measured size is what that check asks for.
- Wave-2 Task 2 (export state PR) found 14 more, all reconfirmed identical
  (same 14, no more/fewer) on a `git stash -u` baseline of the pre-task
  tree via a direct node-id rerun (not the xdist sweep):
  `Tests/UI/test_library_export_cancel.py::
  test_cancel_apply_current_run_sets_cancelled_status` (an unbound-fake
  `SimpleNamespace` call into `_apply_library_export_cancelled` hits
  `AttributeError: 'types.SimpleNamespace' object has no attribute
  '_sync_library_emergency_guard_presentation'` — that call was added to
  the method by other work after this fake's flat attribute list was last
  updated); `Tests/UI/test_library_honesty_accessibility.py::
  test_escape_works_on_export_and_staging_canvases`,
  `test_rail_entry_to_export_after_media_origin_does_not_claim_media`;
  `Tests/UI/test_library_per_click_recompose_t21116.py::
  test_export_open_from_media_is_canvas_scoped`; six
  `Tests/UI/test_library_shell.py` tests (the three
  `test_library_shell_note_export_pushes_file_save_dialog` parametrizations,
  `test_library_shell_note_write_export_file_writes_expected_content`,
  `test_library_shell_note_write_export_file_rejects_invalid_path`,
  `test_library_shell_notes_selected_export_opens_exact_selection_scope`)
  and four `test_library_note_keyboard_capability_matrix` parametrizations
  (`multi_select_export`, `export_markdown_text` × two terminal sizes) —
  these all time out waiting for `#library-notes-row-0` (or a sibling
  selector) to mount within its 30s budget, consistent with machine-load-
  sensitive DOM-mount flakiness rather than a logic bug, but confirmed
  100% reproducible (same set, both runs) rather than randomly flipping.
- Wave-2 Task 3 (export controller PR) found 1 more, confirmed identical
  via `git stash -u` (same root cause as the `test_cancel_apply_current_
  run_sets_cancelled_status` row above -- both trip on the SAME stale-fake
  `_sync_library_emergency_guard_presentation` gap, in different completion
  handlers): `Tests/Library/test_library_export_roundtrip.py::
  test_library_export_success_records_a_durable_receipt_with_the_real_path`
  -- its unbound-`SimpleNamespace` fake for `_apply_library_export_success`
  never gained a `_sync_library_emergency_guard_presentation` entry, and
  the method's own `run_id`-staleness guard does not return early before
  reaching that call for this test's inputs (`run_id=1` matches the fake's
  `_library_export_run_id=1`), so it raises `AttributeError` regardless of
  whether the method is a delegator or its original body -- this task's
  own move neither causes nor fixes it. Also newly found by this task,
  affecting `Tests/Library/` specifically (a directory this recipe's own
  `-k "export and library"` per-task check had not previously needed to
  cover, since Task 2 found its 14 entirely within `Tests/UI/`): the
  "unbound fake-self" bypass shape reaches at a much larger scale here
  than the conversations exemplar's own 5-method precedent -- 9 Export
  methods, across 6 test files, 4 of them in `Tests/Library/` rather than
  `Tests/UI/`. See `library_export_controller.py`'s module docstring for
  the full per-name accounting; future subsystems' controller-PR sweeps
  should widen their `-k` search beyond `Tests/UI` before trusting a clean
  result, and should expect this shape to scale with how much a
  subsystem's tests favor unbound-`SimpleNamespace`/`Mock` unit-style
  calls over full-harness ones.
- Wave-2 Task 4 (export cleanup PR) found 1 more, confirmed identical on a
  `git stash -u` pristine baseline of the pre-task tree (same command,
  isolated to the 3 files that carried it):
  `Tests/UI/test_library_choice_strips.py::
  test_media_type_strip_works_in_both_layouts` -- times out waiting for
  the narrow-layout `library-notes-compact` CSS class within its 15s
  budget; reproduces identically on both trees, consistent with the
  machine-load-sensitive DOM-mount-timing flakiness this file's sibling
  rows already describe, not a logic regression. Task 4's own field-
  retarget pass touches none of this test's assertions or the media-type
  choice-strip code path.
- Wave-2 Task 5 (collections state PR) found 3 more, confirmed identical
  on a `git stash -u` pristine baseline of the pre-task tree via a direct
  node-id rerun of `-k "collection and library"` (Tests/UI + Tests/Library,
  361 passed/3 failed on both branch and baseline):
  `Tests/UI/test_library_shell.py::
  test_library_starter_deep_link_opens_hidden_collection_or_note_route`
  (`WorkerFailed: AttributeError("'types.SimpleNamespace' object has no
  attribute 'active_authority'")` -- an unbound-fake-scope-service test
  fixture missing a newly-added attribute the real service now carries);
  `Tests/UI/test_library_shell.py::
  test_library_landing_continue_receipt_accepts_only_authoritative_source_scopes[browse-collections-expected_scope4]`
  (`state["library_continue_receipt"]` is `None` instead of the expected
  receipt dict); `Tests/Library/test_library_collections_service.py::
  test_get_library_collection_supported_types_round_trip_public_ids`
  (`member["source_ref"]` is a real ref string instead of the expected
  `None`) -- none touch a field or line this task's move edited; this
  task's own diff is a pure field relocation plus a 2-line `_BUDGETS`
  update.
- Wave-2 Task 6 (collections controller PR) reconfirmed the same 3
  Task-5 failures identical via the narrow `-k "collection and library"`
  check (Tests/UI + Tests/Library), plus a 4th, flip-flopping name not
  attributable to either tree: `Tests/UI/test_library_prompt_
  collections.py::test_library_screen_membership_load_retry_and_apply_
  retry_are_distinct` failed on the PRISTINE baseline run but not the
  branch run of that same narrow check (360p/4f baseline vs 361p/3f
  branch) -- a Prompts "Prompt Collections" test (the unrelated feature
  this task's own 3 excluded methods belong to), confirmed pure noise by
  a second, independent reproduction in the full xdist sweep's own
  branch-unique set. **New forward note: running the branch and baseline
  full xdist sweeps CONCURRENTLY (two 8-worker invocations sharing one
  machine) measurably amplifies flakiness beyond this recipe's historical
  ~330-340 backdrop** -- this task's own concurrent run measured 349
  failed/3890 passed (branch) vs 344 failed/3895 passed (baseline), both
  above the documented range, with 12 branch-unique and 7 baseline-unique
  names (vs the export/collections-state series' own 2-and-9-ish norms).
  11 of the 12 branch-unique names passed cleanly on a single-process
  combined re-run; the 1 that reproduced,
  `Tests/UI/test_library_media_reader_traversal_t22207.py::
  test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`,
  passed in TRUE isolation and reproduced identically on the PRISTINE
  baseline under the SAME combined-invocation conditions (a shared-state/
  ordering sensitivity to which OTHER tests ran earlier in the process,
  not to which code version is loaded) -- confirmed pre-existing, not a
  regression, and not Collections-related (Media reader cluster; this
  task's diff touches zero Media-reader code).
  **Correction (fix round 1, post-review): one of the 11 that passed
  cleanly, `Tests/UI/test_library_adaptive_reader_closeout.py::
  test_closeout_single_app_route_cycle`, is NOT unrelated to Collections
  -- its own `DESTINATION_CONTRACT` includes a `"collections"` entry and
  the test cycles every destination, which for "collections" traverses
  the shared reader-shell dispatcher calling two of this task's own moved
  methods (`_sync_library_collections_reader_layout_from_shell`,
  `_mirror_library_collections_reader_preference`).** It was re-verified
  with the strongest available method for a single test: temporarily
  `git checkout bca923b4c -- tldw_chatbook` inside the task's own
  worktree (confirmed via `git diff --stat` that only `library_screen.py`
  changed, i.e. a clean pre-controller-move state), the test run in true
  isolation there, then `git checkout HEAD -- tldw_chatbook` restored
  (confirmed `git status` clean after) -- **passes identically at HEAD
  and at base**, closing the question directly rather than by the flawed
  "the test's name doesn't mention Collections" inference the task's own
  first-draft report used. Any future subsystem's sweep-triage should
  check a failing/flaky test's OWN fixture/contract content (not just its
  name or file) before asserting it is unrelated to the subsystem being
  moved -- a destination-cycling test like this one will touch every
  subsystem's own dispatchers by design. **Future tasks should
  prefer running the branch and baseline full sweeps SEQUENTIALLY, not
  concurrently, when machine time allows** -- the concurrent shortcut
  used here to save wall-clock time cost real investigation effort this
  task's own report accounts for in full.
- Wave-3 Task 4 (search+RAG cleanup PR) found 1 more via the targeted
  regression battery, confirmed identical on a `git stash -u` pristine
  baseline of the pre-task tree (`801c5375e`), reproducing in true
  isolation on both trees:
  `Tests/UI/test_library_canvas_scoped_sync.py::
  test_notes_per_click_updates_keep_screen_and_canvas_identity` -- a
  Notes-only canvas-identity characterization test with no Search/RAG
  interaction; this task's own diff touches none of its assertions or the
  Notes canvas-sync path. Not investigated further (out of this task's
  subsystem scope), but recorded per this section's own mandate rather
  than left for the next task to rediscover.
- The same task's own full sequential xdist paired-baseline sweep found 2
  more among its 5 branch-unique names, confirmed identical on a SECOND
  `git stash -u` to the same pristine `801c5375e` tree, run in the same
  true-isolation combination as the original branch-unique triage:
  `Tests/UI/test_library_shell.py::
  test_library_media_page_error_retains_rows_and_gates_unsafe_controls`
  and `Tests/UI/test_screen_navigation.py::
  test_skills_route_lands_on_library_with_skills_row_selected` -- a Media
  page-error test and a Skills-route navigation test, neither touching
  Search/RAG; this task's own diff touches neither. The other 3 of the 5
  branch-unique names passed cleanly on the same combined re-run
  (ordinary xdist noise, not investigated further per §7's own
  precedent for a name that passes on re-run).
- Wave-3 Task 5 (wave close)'s own full sequential xdist paired-baseline
  sweep (branch 357 failed/3924 passed vs. baseline `a150fc766`, `git
  stash -u`, 349 failed/3932 passed; 347 shared, 10 branch-unique, 2
  baseline-unique) found 4 more, confirmed on a SECOND `git stash -u` to
  the same pristine `a150fc766` tree, combined single-process (2 of the
  10 branch-unique names already matched Task 4's own two entries just
  above, not re-derived): `Tests/UI/test_library_notes_reader.py::
  test_wide_editor_deep_link_keeps_reader_navigation_and_local_back` and
  `Tests/UI/test_screen_navigation.py::{test_generic_library_entry_lands_
  hub_on_first_visit, test_generic_reentry_returns_to_library_landing,
  test_library_screen_round_trip_returns_to_landing_with_rag_draft}` --
  a Notes reader test and three generic screen-navigation tests, none
  touching Search/RAG logic; this task's own diff is docstring/comment-
  only. The last of these, `..._with_rag_draft`, is the SAME name Task
  4's own sweep flagged as BASELINE-unique (§18's sweep evidence) --
  flipping which side it fails on between two different runs is itself
  strong independent evidence of pure run-to-run flakiness, not a
  regression tied to either tree. The other 4 of the 10 branch-unique
  names passed cleanly on the same combined re-run (ordinary xdist
  noise). **Zero real regressions.**
- Wave-5 Task 1 (ingest state PR)'s own full sequential xdist paired-baseline
  sweep (branch `12ba4fb13`, 356 failed/3989 passed vs. an ISOLATED-worktree
  baseline at `9e62dd8f7` -- `git worktree add` + its own venv, not a
  same-tree `git stash`/path-checkout overlay, after a same-tree overlay was
  interrupted mid-run by a session usage limit and had to be discarded and
  redone isolated -- 370 failed/3971 passed; 348 shared, 22 baseline-unique,
  8 branch-unique) found 8 branch-unique names, all resolved without a
  single unexplained one: 4 passed cleanly on a combined single-process
  re-run (`test_audio_cpp_model_library_handoff.py::
  test_audio_cpp_presentation_reveals_slow_load_once_and_keeps_error_retry`,
  `test_library_media_reader_flow.py::
  test_edit_metadata_from_read_routes_to_info_form_actions`,
  `test_library_media_reader_traversal_t22207.py::
  test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke` --
  already documented above, wave-2 task 6 -- and `test_screen_navigation.py::
  test_search_route_round_trips_to_the_library_rag_row`); 2 reproduced
  identically in TRUE isolation on BOTH the branch and the isolated
  pristine-baseline worktree (`test_library_media_reader_no_change_sync_
  t22208.py::test_image_item_traversal_wall_time_probe` -- a new name for
  this list, a wall-clock timing probe by design -- and its sibling
  `test_no_change_traversal_builds_no_preview_and_copies_no_content`,
  already documented above at the wave-2-close entry as a branch-unique name
  that passed cleanly on rerun THERE; this task's own run instead reproduced
  it deterministically on both trees, a stronger and consistent escalation
  of the same "Media reader, load/timing-sensitive" characterization, not a
  contradiction); 1 more,
  `Tests/UI/test_screen_navigation.py::test_library_screen_round_trip_
  returns_to_landing_with_rag_draft`, is the SAME name already documented
  above (wave-3 task 5) as bidirectional run-to-run flakiness, reconfirmed
  reproducing identically on both trees here too; the last,
  `Tests/UI/test_library_shell.py::
  test_library_media_initial_error_is_unknown_and_retry_is_unique`, failed
  once in a combined run immediately following the 3-minute wall-time-probe
  test above but then passed 6 of 7 further isolated re-runs on the branch
  and 3 of 3 on the baseline -- ordinary load-adjacent flakiness with no
  causal link to this task's own field-move diff (Media error-retry logic
  untouched by it). None of the 8 touches Ingest code or this task's own
  diff. **Zero real regressions.** Forward note on method, not just result:
  a same-tree `git checkout <base> -- tldw_chatbook Tests` overlay is FRAGILE
  across a long-running background sweep -- an external restore-to-HEAD
  (session interruption recovery) mid-sweep silently invalidates whatever
  that sweep was reading, with no error of its own. A `git worktree add`
  (its own directory, its own venv) is immune to this class of corruption
  and should be the DEFAULT method for any baseline comparison expected to
  run for many minutes, not merely a fallback for when the overlay method is
  unavailable.
- Wave-5 Task 2 (ingest controller PR)'s own full sequential xdist paired-
  baseline sweep (branch `68a896993`, 358 failed/3992 passed vs. an
  ISOLATED-worktree baseline at `74a6f5774`, 356 failed/3989 passed; 350
  shared, 6 baseline-unique, 8 branch-unique) found 8 branch-unique names,
  all resolved: 6 passed cleanly on a combined single-process re-run
  (`test_library_media_reader_match_nav_t22209.py::test_a_new_document_
  rescans_for_the_same_query`, `test_library_prompt_collections.py::
  test_library_screen_manager_create_search_rename_and_explicit_all`,
  `test_library_prompts_canvas.py::{test_library_prompt_pager_first_and_
  filter_failure_states[size0], test_library_prompt_undo_refreshes_
  applied_page_and_preserves_basket}`, `test_library_shell.py::{test_
  library_media_durable_mutation_gates_and_refreshes_applied_scope[True],
  test_library_shell_blank_note_typed_then_deleted_all_is_gc_from_real_
  db}`); 2 reproduced identically in the same combined re-run --
  `test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_
  navigation_and_local_back` is the SAME name already documented above
  (wave-3 task 5) as bidirectional run-to-run flakiness, reconfirmed here
  too, and `test_library_media_reader_traversal_t22207.py::test_loading_
  banner_paints_in_place_without_body_rebuild` is a NEW name, confirmed
  pre-existing by reproducing identically in TRUE isolation on the
  isolated pristine-baseline worktree (a Media-reader loading-banner
  paint test; this task's own diff touches zero Media-reader code). None
  of the 8 touches Ingest code or this task's own diff. **Zero real
  regressions.**
- Wave-5 Task 3 (ingest cleanup PR)'s own targeted battery (16 of the 17
  files this task edited -- the content-grep-derived set, plus `test_
  library_url_ingest_submit.py` for its own false-positive-only mention;
  run together, single process) found 8 new failures,
  confirmed identical on a `git stash -u` pristine baseline of the
  pre-task tree via the same 8 node-ids: `Tests/UI/test_library_ingest_
  canvas.py::test_progress_detail_paints_below_row_without_obscuring_
  actions_or_neighbor[size0/size1]` (a CSS background/foreground
  color-parity assertion between a progress row and its detail line);
  `Tests/UI/test_library_ingest_retry_last.py::test_registry_ticks_only_
  reflow_footer_when_retry_availability_changes` (a footer-shortcut-list
  equality assertion, one extra `("library", ...)` entry present);
  `Tests/UI/test_library_resize_focus_gates_t23025.py::test_tab_focus_
  path_library_query_volume_is_bounded` (a DOM-query-count budget, 4
  observed vs. an asserted ceiling of 1); `Tests/UI/test_library_ingest_
  structural.py::{test_fold_hint_is_pinned_not_scrolled, test_outcome_
  lines_paint_heavier_than_the_tooling_summary, test_every_canvas_
  focusable_changes_at_the_glyph_level_on_focus}` (CSS-geometry/
  focus-styling assertions); and `Tests/Library/test_ingest_preflight_
  egress.py::test_the_probe_reports_a_redirect_as_an_answered_status_not_
  an_error` (an environment-dependent real network-egress-probe outcome).
  None touches Ingest state/controller code or this task's own diff
  (field-retarget, delegator deletion, dead-import removal, docstring
  fixes) -- all 8 reproduce byte-for-byte identically on the pristine
  pre-task tree. **Zero real regressions.**
- The same task's own full sequential xdist paired-baseline sweep
  (`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`,
  branch then an isolated `git worktree` at `e3d85ad21`): branch 356
  failed/3994 passed (1435.08s) vs. baseline 358 failed/3992 passed
  (1499.26s); 351 shared, 7 baseline-unique (not investigated, per
  precedent), 5 branch-unique, all resolved on a combined single-process
  rerun (`test_library_media_reader_traversal_t22207.py::{test_focus_
  traversal_builds_zero_bodies_for_pass_through_rows, test_one_megabyte_
  markdown_document_is_not_reparsed_per_keystroke}` -- the second already
  documented above, wave-2 task 6/wave-5 task 1, reconfirmed a third
  time; `test_library_prompts_canvas.py::{test_library_prompt_history_
  no_change_keeps_selection_and_retry_available, test_library_prompt_
  pager_first_and_filter_failure_states[size0], test_library_prompts_
  stale_search_cannot_restore_an_old_filter_caret}` -- the pager test is
  the SAME name already documented above, wave-5 task 2, as passing
  cleanly on rerun there too). None of the 5 touches Ingest code or this
  task's own diff (all Media-reader/Prompts-canvas). **Zero real
  regressions.** **New forward observation**: BOTH runs independently
  developed a massive, near-identical burst of `test_library_shell.py::
  test_library_note_*` DOM-mount-timeout failures concentrated in the
  final ~3% of each run (351 of the 356/358 total failures overlap,
  almost entirely from this cluster) -- `pytest`'s own `fd_leak_sentinel`
  plugin reported "open file descriptors grew by 274 over the test
  session (start=14, end=288, limit=200)" on an unrelated standalone
  single-file rerun of `test_library_shell.py` during this same task's
  own verification, suggesting file-descriptor exhaustion accumulating
  over a long single test session -- not this diff -- is the likely
  mechanism behind the late-run failure cascade both trees share
  identically. Recorded as a lead for whichever future task next
  investigates `Tests/UI`'s own sweep-flakiness backdrop, not chased
  further here (out of this task's own scope, and already proven
  non-diff-attributable by the baseline pairing).
- Wave-5 Task 4 (wave close)'s own full sequential xdist paired-baseline
  sweep (branch `5b9c7bdf4`+close, 356 failed/3994 passed vs. an ISOLATED
  worktree baseline at `9e62dd8f7`, 356 failed/3985 passed; 351 shared, 5
  baseline-unique, 5 branch-unique) found 1 more genuinely new name among
  its 5 branch-unique (the other 4 all matched names already documented
  above -- wave-5 task 2's own `test_loading_banner_paints_in_place_
  without_body_rebuild` entry, wave-3 task 5's `test_wide_editor_deep_
  link_keeps_reader_navigation_and_local_back`, and wave-4 close's/wave-5
  task 1's own two flaky-on-rerun entries):
  `Tests/UI/test_library_shell.py::test_library_note_compact_deep_link_
  intent_opens_notes_stage[context2-#library-note-body-editor-False]` --
  a Notes compact-layout deep-link test, confirmed pre-existing by
  reproducing identically (the same 30s DOM-mount-timeout signature) in
  TRUE isolation on BOTH the branch and the isolated pristine baseline
  worktree (not merely a combined-run artifact -- an initial combined
  baseline run alongside an unrelated Media test showed it passing, which
  a follow-up true-isolation rerun on the SAME baseline tree overturned,
  underscoring that a combined-run result is not sufficient evidence on
  its own for a timing-sensitive DOM-mount test). This task's own diff is
  doc-only (recipe, a `.git-blame-ignore-revs` addition, a backlog
  filing); it touches neither Ingest nor Notes code. **Zero real
  regressions across the whole wave-5 span.**
- **A standing row this list should have carried since task-31203 and did
  not**, named here rather than left for a seventh series to re-derive:
  `Tests/Architecture/test_library_modules_size_ratchet.py::
  test_controller_does_not_grow_past_its_budget[tldw_chatbook/UI/Library_
  Modules/library_media_browse_controller.py]`. It is the controller-ratchet
  twin of the two `chat_screen.py` screen-ratchet rows already listed above
  — concurrent, unrelated growth in a file no Library extraction series
  touches — and every wave-6 task dismissed it as "the documented
  pre-existing row" while §7 in fact documented only the `chat_screen.py`
  pair. Proven pre-existing by construction, the same way those two are:
  no Library extraction PR has ever edited that controller or its budget
  row.
- **Wave-6 Task 1 (prompts state PR) found 17 more, all in prompt-named
  test files, all reconfirmed byte-identically on a pristine baseline at
  the wave-6 RED commit (`a41cbe8c6`, screen untouched — 18 shared
  failures of which the 18th, `test_library_choice_strips.py::
  test_media_type_strip_works_in_both_layouts`, was already documented
  above at wave-2 task 4).** They are a dev-state backdrop the wave
  inherited, in four clusters:
  - **5 CSS-bundle-parity `_css_block` assertions** reading
    `tldw_chatbook/css/tldw_cli_modular.tcss` + `_agentic_terminal.tcss`
    for `#library-prompts-*`/`.library-prompt-row` blocks that now live in
    the per-screen bundle `screen_agentic_library.tcss`:
    `test_library_prompt_row_class_matches_notes_row_visual_parity`,
    `test_library_prompts_header_filter_empty_have_css_blocks`,
    `test_library_prompt_editor_field_css_blocks_match_notes_editor_parity`,
    `test_library_prompt_field_hint_css_block_matches_field_label_parity`,
    `test_library_prompts_import_row_css_blocks_match_filter_status_parity`.
    (The CSS moved; the assertions still read the old bundles. Not a
    Library-decomposition concern.)
  - **4 `test_library_prompt_history_geometry_uses_only_the_outer_editor_
    scroll[dirty-size0..3]`** — CSS-geometry parametrizations.
  - **4 `test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_
    once[…]`.**
  - **4 others**:
    `test_library_prompts_unmount_revokes_late_apply_before_workspace_
    shutdown`, `test_library_shell_create_prompt_save_creates_and_
    increments_count`, `test_library_prompt_import_blocks_undo_until_
    import_settles`, `test_cancelled_prompt_import_retains_writer_
    ownership_until_commit`.
- **Wave-6 Task 1 also proved 15 `Tests/Architecture/` failures
  pre-existing NAME-FOR-NAME at the wave-6 base**, by re-running the
  identical 15 node-ids at `a41cbe8c6` (15 failed / 166 passed / 1 skipped)
  — worth listing because three of them key on `(file, line)` tuples and
  are therefore the shape a pure move is most likely to break, so a future
  series will otherwise spend the same time re-deriving them:
  `test_console_realtime_controller_boundary` (1),
  `test_console_review_selection_controller_boundary` (1),
  `test_console_wave6_closeout_inventory` (1),
  `test_console_wave6_inventory` (3), `test_default_timeout_session_guard`
  (1), `test_persistent_diagnostic_inventory` (2),
  `test_progress_widget_clock_guard` (1),
  **`test_timer_path_static_update_inventory` (3 — the `(file, line)`-keyed
  `CLASSIFIED_SITES` census)**, `test_worker_exclusive_group_inventory` (2).
  Independently: neither `test_timer_path_static_update_inventory.py` nor
  `test_worker_exclusive_group_inventory.py` contains a single reference to
  `library_screen`, so no `(file, line)` key of theirs could have gone
  stale from a Library move at all. Wave-6 task 2 re-ran the same 15 (plus
  the 3 ratchet rows = 18) at ITS parent and `diff`ed the sorted name lists
  to **IDENTICAL**.
- **`Tests/UI/test_library_modal_dismissal.py` is a standing 1-red at every
  wave tip, and it is a BLOCKED guard rather than a failing assertion.**
  `test_library_modal_inventory_matches_declared_edges_bidirectionally`
  raises on an unresolved modal constructor for the skills-era
  `LibraryScreen._present_library_skills_import_choice_if_needed`, which
  aborts `_discover_library_modal_edges` **before** the bidirectional
  comparison runs — so the file currently proves nothing about ANY
  subsystem's rows, including rows a task has just repointed. Measured
  identically on both trees throughout wave 6 (**1 failed / 169 passed**).
  A task whose own work touches this inventory therefore cannot prove it
  green end-to-end and must prove the retarget **by construction** instead
  (run the file's own `_discover_library_modal_edges` against the new
  `_OwnerScope` alone and compare to the declared rows, matching each
  modal's concrete type — the prompts series' worked example, §21). Filed
  for repair as **TASK-31815**, together with the three other equally-stale
  rows (`handle_library_ingest_browse` and the two skill-trust passphrase
  presenters); remove this entry when that lands.
- **`Tests/Performance/test_ui_ready_module_census.py` flaps against a
  ZERO-headroom pin, identically on trees that do not contain the change.**
  The pin is 972 and the warm-boot measurement IS 972; under load it
  measures **974** and fails with byte-identical assertion text on the
  branch and on an ISOLATED baseline worktree at the parent. Wave-6 task 3
  characterized the flap properly rather than by a single lucky run: **8
  passed / 2 failed over 10 isolated single-node runs on a quiet machine**
  (3 competing processes) — so it is NOT purely load-gated. Zero of the 25
  modules the failure names is a Library module (14 are
  `tldw_chatbook.Chat.console_*`; the rest are DB, LLM_Calls, Scheduling,
  Navigation, Console_Modules and Widgets), and the Library work only ever
  REMOVES modules from the first-paint window via the born-lazy controller
  import. **Report this one as a red when it fires** — an earlier draft of
  the wave-6 task-3 battery reported only the lucky 46/3 outcome as if it
  were "the" result, which is the failure mode a flapping guard invites.

  **Read the guard's own policy before proposing a fix, because it forbids
  the obvious one.** Its module docstring states: "RATCHET (TASK-23029 /
  ADR-097, `backlog/decisions/097-boot-budget-ratchets.md`):
  `MAX_TLDW_MODULES_AT_UI_READY` **never rises**. On a breach, defer the
  new mount-leg cost or shed equivalent cost elsewhere in the same PR; the
  only other path is an explicit owner exception recorded in the ADR's
  exception ledger." So "raise the pin with some headroom" is not a
  conforming resolution at all. And the same docstring documents the
  tolerated run-to-run wobble as **±1 module** — the observed breach is
  **+2**, which is outside it. This is therefore a genuine ADR-097 breach
  owned by whatever work added those residents, not a guard that merely
  needs slack. Filed as **TASK-31816** (dev-owned; dev's own `06acf148f`,
  "perf: defer Library and Settings runtime pre-import payload", is the
  precedent for the paydown direction); remove this entry when that lands.
- **Wave-6 Task 4 (wave close) found 2 more in `Tests/Library/`**, surfaced
  by widening the paired sweep to that whole root and confirmed by the
  pairing itself — both fail on an ISOLATED worktree at the wave-6 start
  commit with byte-identical node names, in the same run that produced 22
  shared / 0 branch-unique / 0 baseline-unique:
  `Tests/Library/test_agent_chunk_student_story.py::test_student_story_
  chapter_notes_from_stored_chunks_only` and
  `Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_
  in_sync_with_real_sources`. Neither touches Prompts code. They are listed
  here because `Tests/Library/` had only ever been swept in narrow,
  subsystem-filtered slices before — the first unfiltered run of the whole
  root is what made them visible.

## 8. Subsystem order (spec, "Order of work")

Sequenced cold-to-hot so the conversations exemplar never fights rebases,
and hot subsystems migrate in short, fast series once the recipe above is
rehearsed. Churn = commits touching `library_screen.py` in the trailing 30
days whose subjects name the subsystem (measured 2026-09-01):

| Order | Subsystem | Churn | Notes |
|---|---|---|---|
| 1 | **conversations** (exemplar) — **complete** (Tasks 6–9) | 10 | 68 methods, 19 fields (2026-09-01 estimate); see §11 for the series' actual, as-landed numbers |
| 2 | **export** — **complete** (wave-2 Tasks 2–4) | 3 | 13 fields moved; 22 of 51 "export"-named method candidates moved / 29 excluded (18 other-subsystem, 2 `@work` framework-decorator self-type-assertion hazard, 9 unbound-fake-self/silent-Mock test bypasses); see §12 for the series' actual, as-landed numbers |
| 2 | **collections** — **complete** (wave-2 Tasks 5–7) | 6 | 26 fields moved (1 wiring field stayed); 64 of 67 "collection"-named method candidates moved / 3 excluded (Prompts-owned); see §13–§15 for the series' actual, as-landed numbers |
| 2 | **search + RAG** — **complete** (wave-3 Tasks 2–4, task-31203) | 6 + 16 | Deferred from wave-2 (search alone was BLOCKED at the entanglement gate, wave-2 Task 8) into ONE combined series once RAG's own pool was folded in. 20 fields moved to `LibraryRagSearchState`; 42 of 50 combined "search"+"rag"-named method candidates moved to `LibraryRagSearchController` (3 Prompts-owned + 7 Media-owned excluded from the raw 60 name matches before the 50-candidate cluster is even formed; of the 50, 8 excluded: 3 `@work` framework-decorator hazard, 1 module-globals-coupling, 4 instance-attribute-monkeypatch test bypass); 12 of 42 screen delegators pruned at cleanup. See §18 for the series' actual, as-landed numbers |
| 3 | **skills** — **complete** (wave-4 Tasks 1–3) | 15 | 36 fields moved to `LibrarySkillsState` (a three-way prefix split: 26 `_library_skill_*` singular + 9 `_library_skills_*` plural + 1 bare `_selected_skill_name`, resolved by a single `skill_state_shim_attr()` function rather than two independent frozensets); 86 of 127 "skill"-named method candidates moved to ONE `LibrarySkillsController` (41 excluded: 6 merely-delegate-to-existing-controller properties, 27 unbound-fake-self, 1 instance-attribute monkeypatch, 1 module-globals coupling, 6 bare-self-as-identity-argument hazard — plus 1 CRITICAL unbound-attribute-escape (`getattr(self, "focused", None)` with no corresponding property) found by post-landing review rather than the pre-landing battery, fixed with a fail-without/pass-with covering test); 16 of 86 screen delegators pruned at cleanup. This series' own two battery-caught regressions (§3's sixth bypass shape) and the review-found seventh instance widened that bypass catalogue for every subsequent subsystem. See §19 for the series' actual, as-landed numbers |
| 3 | **ingest** — **complete** (wave-5 Tasks 1–3) | 23 | 20 fields moved to `LibraryIngestState` (single `_library_ingest_` prefix, no plural variant); 56 of 78 "ingest"-named method candidates moved to ONE `LibraryIngestController` (22 excluded: 4 `@work` framework-decorator hazard, 3 module-globals-coupling, 9 unbound-fake-self/`object.__new__`-bypass, 6 instance-attribute-monkeypatch); 6 of 56 screen delegators pruned at cleanup. This series' own state PR found the "seventh bypass shape" (an `object.__new__`-bypassed fixture's flat-name seed breaking the instant the state shim installs, not deferrable to cleanup) — a review-found CRITICAL: 2 tests left RED at HEAD in `Tests/UI/test_parakeet_v2_install_ui.py`, the one file whose filename and test names contain neither "ingest" nor "library" and which the task's own `-k`-filtered sweep therefore could not see, a no-red-ships violation (the same task's separate 24-vs-27-site count error in its own report was a distinct Important finding, not this CRITICAL) — and its controller PR's post-landing review found a SECOND review-found CRITICAL, the "eighth" bypass shape (a moved body's bare module global patched at the OLD module path by a green-but-vacuous test, `_resolve_ingest_source`) — both widened the bypass catalogue for every subsequent subsystem. See §20 for the series' actual, as-landed numbers, and §20's own "Wave-5 close" subsection for the wave-level pin trajectory, verification battery, and lessons |
| 4 | **prompts** — **complete** (wave-6 Tasks 1–3) | 41 | 43 fields moved to `LibraryPromptsState` (a three-way prefix split, the skills precedent: 31 `_library_prompt_*` singular + 11 `_library_prompts_*` plural + 1 bare `_selected_prompt_id`, resolved by a single `prompt_state_shim_attr()`; 3 further prompt-named `__init__` attributes are WIRING — live `LibraryPromptHistoryController`/`LibraryPromptBrowseController`/`LibraryPromptCollectionsController` instances — and stayed on the screen); 139 of 161 "prompt"-named method candidates moved to ONE `LibraryPromptsController`, the largest single move of this program (22 excluded: 14 unbound-fake-self, 3 instance-attribute-monkeypatch, 2 screen-identity, 2 module-globals-coupling, 1 merely-delegate-to-existing-controller property); 39 of 139 screen delegators pruned at cleanup (~28%). This series' cleanup found a genuinely NEW delegator-prune hazard the prior five did not: Textual's `on_<Message>` NAME-dispatched handlers (`MessagePump._get_dispatch_methods` resolves them off `Message.handler_name`, not off `@on`), which a reference-count census reports as zero-referenced and whose deletion would silently unhook the screen from six messages — folded into §4's transform whitelist as its THIRD member, since media and notes both almost certainly own name-dispatched handlers too. See §21 for the series' actual, as-landed numbers, and §21's own "Wave-6 close" subsection for the wave-level pin trajectory, verification battery, sweep evidence and lessons |
| 4 | media | 55 | |
| 4 | notes | 72 | most scarred; its sync controller (`canvas_sync.py`) already lives in `UI/Library_Modules/` from PR 0a |
| 5 | final shell pass | — | residual focus/lifecycle plumbing, delegator table tidy, `compose_content` reduced to the region-yielding skeleton |

Roughly 35–50 small PRs total; every intermediate state ships (no feature
freeze, per the plan's Global Constraints — never two subsystems'
extraction PRs in flight at once).

Phase C (region ownership — moving canvas-origin `@on` handlers and state
into the already-existing canvas widgets) is a separate, later, explicitly
behaviour-changing series per subsystem, gated on that subsystem's phase-A
series being fully landed including cleanup, dense mounted coverage, and a
concrete motivating change. **First motivated candidates: media and
notes**, motivated by the measured 139–380 ms rail-mode-switch main-thread
freeze (§9's probe is that fix's before/after acceptance evidence). See the
spec's "Phase C — region ownership" section; out of scope for this recipe's
pure-move PRs.

## 9. Probe usage — before/after evidence

`Helper_Scripts/library_click_probe.py` boots the real `LibraryScreen`
headless, clicks through the rail modes, and reports per-click settle time,
max main-thread gap, recompose/full-update counts, and widget mount/removal
counts. Headless numbers exclude terminal-write cost (paint-to-terminal
bytes are not produced without a real terminal) — this is a main-thread
compute-and-DOM instrument, not end-to-end latency; that framing is honest
noise-tolerance, not a limitation to work around.

```bash
.venv/bin/python Helper_Scripts/library_click_probe.py
```

Run it and keep the report table before starting a controller-move PR
(Task 7/8-shaped work, not the state or cleanup PRs, since only a
controller move touches code that runs during a click) and again after, as
the PR's "a pure move must not move these numbers outside noise" evidence.
A pure move changing *where* code lives must not change the click-latency
numbers; a checkpoint that drifts is a signal the move was not pure and is
grounds to stop and investigate before merging, not to update the recipe.

**Run it as a PAIR on the same machine, and run the pair TWICE with the tree
order swapped** (added at the wave-6 close, §21, which was the first close to
actually do it). "Before" means the isolated baseline worktree the sweep
already requires, probed minutes apart from the branch — not a comparison
against a band some earlier close recorded on a different machine under a
different load, which is what §16/§19/§20 each fell back to and which cannot
separate a code effect from a machine effect. The order swap is not
optional: wave-6's round 1 showed the branch slower on all sixteen
wall-clock measurements, and reversing the order moved the whole penalty
onto the baseline — a first-run warm-up artifact that a single pair would
have reported as a uniform regression. Read the load-INDEPENDENT columns
(`recompose`, `full-update`, `nodes`) as the verdict and the wall-clock
columns as context; a real per-frame regression shows up in the former.

## 10. `.git-blame-ignore-revs` — one-time setup and the per-PR rule

Every pure-move commit's hash is appended to `.git-blame-ignore-revs`, in
the **same PR** that makes the move, so `git blame` keeps resolving lines to
the author who actually wrote the logic rather than to whichever PR most
recently relocated the file it lives in.

**One-time, per clone**, so `git blame` actually consults the file:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

This is a local git config setting (not committed, not inherited from the
repo) — every clone/worktree that wants blame-through-moves needs to run it
once. `git blame` and `git blame --ignore-revs-file` both work without it,
but plain `git blame`/most editor blame integrations only honor the
ignore-file automatically once this config is set.

## 11. The conversations exemplar, as landed — actual numbers and lessons

The exemplar series (Tasks 6–9: state PR, two controller PRs, cleanup PR)
is complete. This section replaces the plan's 2026-09-01 estimates with
what actually landed, and records what the rehearsal taught the recipe
above — read this before running the next subsystem (export, churn 3, per
§8).

### Methods/fields moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 6 | State | 28 fields → `LibraryConversationsState` (0 methods; a programmatic property-shim block, not per-field getters, to fit the line ratchet — see the shim's own sentinel-comment history for why) | 45134 → 45134 lines (net zero; shim added what `__init__` lost), 1300 methods (unchanged) |
| 7 | Controller 1 | 21 methods → `LibraryConversationReaderController` (5 `@on` handlers + 16 plain) | 45134 → 44715 lines, 1300 methods (pure move: 21 bodies out, 21 one-line delegators in) |
| 8 | Controller 2 | 40 methods → `LibraryConversationsController` (10 `@on` handlers + 30 plain), 7 more excluded (see below) | 44715 → 44060 → 44084 lines (a review-fix round added +24 lines of documentation, no logic change), 1300 methods (unchanged: pure move) |
| 9 | Cleanup | Shim block deleted; every remaining screen-side field reference retargeted to `self._conversations_state.<field>` (90+12 occurrences via one mechanical AST-driven pass, plus 4 methods excluded from Task 8 that had to be retargeted and then had their TEST fakes retargeted to match, per §3's monkeypatch-routing doctrine); 18 of the 61 screen delegators deleted (repo-wide census: zero references anywhere outside their own one-line body); 11 of the ledger's 12 dead imports removed + 1 dead controller import removed (the ledger's 12th, `LIBRARY_CONVERSATION_READER_MAX_CHARS`, turned out to be pinned by PR 0a's re-export contract — see the lesson below — and was restored) | 44084 → **43974 lines, 1282 methods** (18 fewer `FunctionDef`s — exactly the 18 pruned delegators) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`45134/1300 → 44715/1300 → 44060/1300 → 44084/1300 → 43974/1282` (final).

**61 screen delegators, not 68 methods**: the 68-method 2026-09-01 estimate
included the 7 methods Task 8 found could never move (shell-owned, or
reached by a test through a bypass shape a pure move can't survive — see
below); 21 (reader) + 40 (browse) = 61 delegators actually landed, of
which 15 are `@on` handlers, 6 are cross-controller wiring-lambda targets,
22 more have a genuine external reference (another screen method, a test,
or a production caller), and **18 had none** and were deleted in the
cleanup PR.

### Lessons

**Lower the ratchet in the SAME PR that moves the code, never deferred.**
Task 7's own execution first followed the plan's (wrong) instruction to
defer lowering to the cleanup task, producing a real gate slip
(`test_budget_is_not_left_slack_after_a_wave` red between Task 7 and Task
9) before a mid-series correction reversed it. Every move task in this
series ended up lowering its own pin in its own scope; Task 9's lowering
is the *last* one, not the only one.

**The `startswith` enumeration trap.** A discovery script that filters
"already-handled" names with `name.startswith(("_library_conversation",
"_conversation"))` silently swallows any OTHER name that happens to share
that prefix but isn't actually a cluster method or a state field —
Task 7's `_conversation_records`/`_conversation_record_id` (general
browse-cluster helpers) were missed this way and only caught by
re-deriving the bind list without the shortcut. Any enumeration script for
a future subsystem should cross-check its "internal, already covered"
filter against the actual state-field list and the actual cluster list,
not a prefix guess.

**Three, now four, distinct test-bypass shapes a pure move can silently
break** — the recipe's §3 documented the first (class-level
`monkeypatch.setattr`); this series found three more, each requiring a
different accommodation:

1. **Unbound fake-`self` calls** (Task 8, exclusions #2–6): a test builds a
   bare `SimpleNamespace`/hand-built fake with only the flat attribute
   names the ORIGINAL method body needed, then calls
   `LibraryScreen.<method>(fake, event)` unbound. A moved body would reach
   for a `_conversations_controller` the fake doesn't have. Fixed by
   leaving the method real and full-bodied on the screen (not moved) — and
   in Task 9, when the field-retargeting pass touched these same methods'
   *field* references (a cleanup-PR-legal edit the move PR couldn't make),
   the fakes needed a matching retarget: flat kwargs became a nested
   `_conversations_state=SimpleNamespace(...)` constructor argument. This
   is squarely inside the cleanup PR's "retarget test attribute pokes"
   mandate — not a new exception, just the mandate reaching a fixture
   builder instead of a bare attribute assignment.
2. **Instance-attribute monkeypatch** (Task 8, exclusion #7): a test does
   `screen.<method> = lambda: payload` on one REAL, fully-constructed
   instance, expecting an internal sibling call to observe the patch. Once
   both methods live on the controller, the sibling's `self.<method>()`
   resolves against the CONTROLLER instance, which never saw the patch
   applied to the SCREEN instance. Only the full paired-baseline xdist
   sweep (§7) caught this — narrower suites never touched the failing
   test file.
3. **Dynamic `getattr`/dict-string dispatch** (Task 9, new this task): a
   shared, multi-subsystem helper builds an attribute NAME as a runtime
   string (an f-string like `f"_library_{kind}_row_selection"`, or a
   `{destination: "_library_<x>_reader_preferences"}` lookup dict) and
   resolves it with plain `getattr`/`setattr`. Neither the byte-for-byte
   body diff nor an AST literal-attribute-reference retarget script can
   find this shape — the attribute name never appears as a literal
   `self.<name>` expression anywhere. Two independent instances surfaced
   in this one cleanup task: `canvas_sync.py`'s `_apply_library_row_toggle`
   (whose `AttributeError` was silently swallowed into a
   `logger.debug` + `screen.refresh(recompose=True)` fallback — a full
   recompose masquerading as normal operation, caught only by a stale
   captured-widget-reference check, not an exception) and
   `library_screen.py`'s own `_replace_library_reader_preference`/
   `_persist_library_reader_preference` (a 7-destination dict of
   attribute-name strings). Both were fixed the same way:
   `operator.attrgetter("_conversations_state.<field>")(screen)` for
   reads (it resolves a dotted path and a flat name identically, so it's
   a transparent passthrough for the other, not-yet-extracted,
   destinations) and a small `_assign_...attribute(owner, path, value)`
   helper for writes. **Any future subsystem's cleanup PR should grep for
   `getattr(screen,` / `getattr(self,` (and their `setattr` siblings) with
   an f-string or dict-literal argument before deleting that subsystem's
   shim** — a plain "does this literal name still resolve" check is not
   enough.

**"Dead within this file" is not the same question as "dead."** Task 7's
report listed `LIBRARY_CONVERSATION_READER_MAX_CHARS` among nine names it
called dead imports — true in the narrow sense that nothing in
`library_screen.py`'s own logic reads it — but PR 0a's own
`test_screen_still_re_exports_every_moved_name` (§10's sibling contract
test, `Tests/Architecture/test_library_support_layer_surface.py`) pins
`library_screen.py` to keep re-exporting every name Task 1 moved to
`Library_Modules/`, specifically so other modules can keep importing
support names FROM the screen rather than needing to know they moved.
Deleting that one import broke the contract test; it had to be restored.
**Before deleting an import a prior task's report calls "dead," check
whether the name is a member of any `_SURFACE`-shaped re-export contract
first** — a single-occurrence `grep` (the import line and nothing else)
proves the name is unused HERE, not that nothing depends on it being
importable FROM here.

**The static-method delegator pattern.** A moved cluster method that was a
bare `@staticmethod`/`@classmethod` on the screen (no `self` in its own
signature) can't dispatch through `self._controller.<name>(...)` the way
an instance method does — Task 7's first pass on
`_conversation_reader_record_version` dropped the decorator to gain a
`self` to reach the controller through (harmless there — no external
caller used it as a static method — but a latent risk). Task 8 established
the corrected shape instead: keep the decorator, and forward straight to
the CONTROLLER CLASS object (`return LibraryConversationsController.<name>(...)`),
which needs no instance at all. Task 9's strengthened wiring-test regex
(`test_screen_delegates_*_handlers`) accepts either forwarding spelling
(`self._controller.<name>(` or `ControllerClass.<name>(`) for exactly this
reason.

**The `_safe_text` class-binding pattern.** A moved `@classmethod` body
that calls `cls._safe_text(...)` needs `_safe_text` to exist on the
CONTROLLER class, but `_safe_text` is a general, non-Conversations-owned
`@staticmethod` that stays on the screen. Task 8's fix: one module-level
class-attribute assignment, `LibraryConversationsController._safe_text =
staticmethod(LibraryScreen._safe_text)`, executed from `library_screen.py`
(not the controller module, to avoid a circular import) after both
classes are defined. The gotcha a review caught: a plain class-attribute
assignment always REPLACES whatever was previously on the class under
that name — an earlier draft that also defined a `_safe_text` `@property`
on the controller (backed by an injected constructor parameter) had that
property silently and permanently overwritten the instant this module
loaded, making the property, its parameter, and its backing attribute
dead code with a misleading docstring. The corrected version carries only
the one class-level binding, documented in-place with this exact
incident.

## 12. The export series, as landed — the recipe's first non-exemplar rehearsal

Export (churn 3, §8) is the first subsystem to run this recipe after the
conversations exemplar wrote it. Its series (wave-2 tasks 2–4: state PR,
controller PR, cleanup PR) is complete. This section records what actually
landed and what the rehearsal changed or reconfirmed about the recipe
above — read this before running the next subsystem (collections/search,
also churn 6, per §8).

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 2 | State | 13 fields → `LibraryExportState` (0 methods; a programmatic property-shim block, one prefix — no plural-name split needed, unlike Conversations) | 43965 → 43930 lines (net -35: the shim block added less than `__init__` lost, unlike the conversations exemplar's net-zero), 1282 methods (unchanged) |
| 3 | Controller | 22 methods → `LibraryExportController` (of 51 "export"-named candidates: 18 belong to other subsystems, 2 excluded for a NEW bypass shape — "framework-decorator self-type assertion", `@work`'s `isinstance(self, DOMNode)` runtime assertion — and 9 excluded as unbound-fake-self/silent-Mock-auto-attribution, found only by running the battery) | 43930 → 43432 lines, 1282 methods (unchanged: pure move, 22 `FunctionDef`s out, 22 delegators in) |
| 4 | Cleanup | Shim block (13 properties) deleted; every remaining screen-side field reference retargeted to `self._export_state.<field>` (42 literal `self._library_export_<field>` occurrences via one mechanical regex pass, AST-verified against the same census the conversations exemplar used, plus 1 dynamic-dispatch site — see below); 1 of the 22 screen delegators deleted (repo-wide census: zero references anywhere outside its own one-line body); 5 named dead imports removed, each verified single-occurrence and checked against `_SURFACE` first | 43432 → **43413 lines, 1281 methods** (1 fewer `FunctionDef` — exactly the 1 pruned delegator) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`43965/1282 → 43930/1282 → 43432/1282 → 43413/1281` (final).

**22 screen delegators, not 51 methods**: the 51-method naive census
(matching name substring `"export"`) is the same trap §2's `startswith`
lesson warns about, applied to methods instead of fields — Task 3's own
report reads every one of the 51 bodies rather than trusting the name
match, and finds only 22 genuinely Export-owned AND movable. Of those 22,
**21 have a genuine external reference** (mostly the screen-resident
round-2/round-3-excluded siblings calling back into the delegator, plus 5
`@on` handlers and one `action_*`) and **1 had none**
(`_library_export_is_server_mode`, reached only by the controller's own
internal `self.<name>()` calls) and was deleted in the cleanup PR. This
21-of-22 keep ratio is sharply higher than the conversations exemplar's
43-of-61 (18 pruned) — not a sign of a shallower census, but a direct
consequence of Export's own round-2/round-3 exclusions: 11 of the 51
naive candidates stayed screen-resident specifically because a test or a
framework decorator reaches them unbound, and every one of those 11
still calls its sibling delegators internally (`self.<delegator_name>()`),
which is exactly the shape that keeps a delegator alive. Conversations
had no round-2/round-3-shaped exclusion class this large, so its moved
cluster called itself controller-to-controller far more often, orphaning
far more of the screen-side one-liners.

### Lessons

**A fourth bypass shape, new to this series: "framework-decorator
self-type assertion".** Textual's `@work(...)` decorator wraps a method in
a closure that asserts `isinstance(self, DOMNode)` at CALL time (read from
`textual/_work_decorator.py` via `inspect.getsource`, not assumed). A
plain controller object is not a `DOMNode`, so a moved `@work`-decorated
method would raise `AssertionError` on every call, not just under test —
this is not a test-bypass shape at all, but a permanent runtime contract
the framework itself enforces. Two methods
(`_run_library_export_counts_worker`, `_run_library_export_worker`) stay
on `LibraryScreen`, UNMOVED, decorator and body byte-for-byte untouched.
Add this to recipe §3's catalogue alongside the class-level monkeypatch
and the conversations exemplar's three: **before moving any `@work`- (or
similarly self-type-asserting-decorator-)wrapped method, read the
decorator's own source**, not just its name, to confirm it does not bind
`self` to a concrete type the target controller cannot satisfy.

**The "unbound fake-self" bypass shape scales with a subsystem's test
style, sometimes far past the exemplar's own precedent — and can require
widening the sweep root.** Conversations' Task 8 found 5 methods reached
this way, in one test file, entirely within `Tests/UI/`. Export's Task 3
found 9, across SIX test files, FOUR of them in `Tests/Library/` — a
directory the recipe's own canonical `-k "library"`/`-k "export and
library"` sweep commands do not cover by default (they scope to
`Tests/UI`). None of the 9 were caught by static analysis (free-name
resolution, byte-for-byte diff, or a read of the moved bodies); every one
surfaced only by running the verification battery and reading a
traceback. **Any future subsystem's controller-PR battery should
deliberately widen its `-k` search beyond `Tests/UI` before trusting a
clean result** — this recipe's §7 sweep-command documentation has been
updated with this forward note since Task 3 landed; Task 4's cleanup
confirmed the same 2 files needed touching again for the field-level
retarget (`Tests/Library/test_library_export_execution.py`,
`Tests/Library/test_library_export_roundtrip.py`).

**The dynamic-dispatch bypass shape (recipe §11 lesson 3, generalized) can
share ONE helper across two unrelated dispatch mechanisms.** Export's
cleanup found exactly one dynamic-dispatch site touching its own state:
`_library_open_choice_strip`/`_close_open_library_choice_strip` (a
FOUR-subsystem-shared, not Export-exclusive, converged Escape-handler)
returns/consumes a visibility-attribute NAME as a plain string, resolved
with `setattr(self, visibility_attr, False)`. Media/Prompts/Skills still
keep flat screen attributes for their own visibility fields; only
Export's moved to `self._export_state.quality_choices_visible` (Task 2).
Rather than write a SECOND `_assign_...attribute(owner, path, value)`
helper duplicating the shape, this task reused the conversations
exemplar's own `_assign_library_reader_preferences_attribute` (already a
generic dotted-vs-flat passthrough, `owner`/`attribute`/`value` in shape,
with no reader-preferences-specific logic in its body) for the SECOND,
unrelated call site, extending its docstring to document both callers
rather than asserting the reuse silently. **Before writing a new
`_assign_...attribute` helper for a newly-found dynamic-dispatch site,
check whether an existing one is already fully generic in behavior** — the
shape (write through a possibly-dotted path, flat-name passthrough
otherwise) is likely to recur exactly, and one documented helper serving
two call sites beats two near-identical ones.

**The screen-shim wiring test's own retirement, confirmed a second time.**
The state PR's `test_state_object_fields_match_the_shim_surface` (pinning
that every `LibraryExportState` field has a matching generated property
shim on `LibraryScreen`) was deleted wholesale in the cleanup PR, exactly
as the conversations exemplar's Task 9 first established for
`LibraryConversationsState`'s equivalent test — there is nothing left on
the screen for that assertion to check once the shim block is gone. The
controller-side `test_export_controller_exposes_every_state_field`
(already added in Task 3, unchanged by Task 4) covers the equivalent job
from the surviving side. Any future subsystem's state-PR wiring test
should expect its own screen-shim-surface assertion to be a cleanup-PR
deletion, not a retarget, from the day it's written.

**Sweep evidence came back cleaner than the exemplar's own, not worse.**
The full xdist paired-baseline sweep (recipe §7) found **zero**
branch-unique failures this time (329 shared with the pristine baseline,
2 baseline-unique — noise in the opposite direction) — cleaner than the
conversations exemplar's own 5+4 split. This is not evidence the sweep
step is skippable for a "small" cleanup PR: the field-retarget pass
touched 13 test files and one dynamic-dispatch site shared across four
subsystems, any one of which could plausibly have broken something the
narrower `-k` checks don't cover. The sweep is the check that would have
caught it if it had.

## 13. The collections series, task 1 (state PR) — as landed

Collections (churn 6, §8) is the second subsystem to run this recipe.
Wave-2 task 5 (state PR, collections series 1/3) is complete; the
controller and cleanup tasks have not yet run. This section records what
landed and what this task's own execution added to or reconfirmed about
the recipe above.

### Cluster enumeration

`ast` walk of `LibraryScreen` for method names containing "collection"
(case-insensitive): **67 methods** (2026-09-02 snapshot, matches the
wave-2 plan's estimate exactly). Of those, **3 are Prompts-owned**
(`handle_library_prompts_collection`, `_apply_library_prompt_collection`,
`_sync_library_prompt_collection_label` -- the unrelated "Prompt
Collections" feature, confirmed by reading each body: they use
`_library_prompt_collections_controller`/`_library_prompt_browse_controller`
and have nothing to do with the Library Collections/captures subsystem),
excluded per the export series' own documented substring-match trap. Of
the remaining 64, 42 carry a distinct `@on` decorator; 41 once the one
Prompts false-positive (`handle_library_prompts_collection`) is dropped.

### Field ownership (recipe §2 script, `_library_collections` prefix)

**27 `__init__`-scoped fields** (not ~28 as the wave-2 plan estimated --
re-derived, not trusted). Unlike Export's `origin_row_id`, no field was
missed by the `__init__`-scoped census: a full class-level `AnnAssign`
scan found zero collections-owned class-level-only attributes. Of the 27:

- **26 are exclusively-owned state** -- MOVE. All non-collection
  `__init__`-scoped consumers found by the script are shell/plumbing
  (`on_unmount`, `save_state`/`restore_state`, `apply_navigation_context`,
  `_build_library_shell_input`, `_select_library_rail_row_after_source_
  admission`, `_persist_library_reader_preference`,
  `request_library_reader_layout_refresh`) or a generic multi-subsystem
  dispatcher whose name happens to contain another subsystem's prefix as
  a substring without being owned by it
  (`_toggle_library_media_reader_pane`, which branches on
  `_library_selected_row_id` across Collections/Conversations/Notes/
  Media -- the same "tagging heuristic is name-based, not body-based"
  caveat the recipe's own script output already carries). No field is
  shared with a SECOND subsystem's OWN methods, so the ≥2-subsystems rule
  never triggers and none is BLOCKED.
- **1 is wiring, not state**: `_library_collections_capture_controller`
  holds a live `LibraryCollectionsCaptureController` instance (the
  `_conversation_reader_controller` precedent the task brief named in
  advance). It stays a plain `LibraryScreen` attribute, constructed at
  its original position, untouched by this move.

**The saved-searches census** (the brief's flagged contested boundary,
"collections vs the search cluster"): `_library_collections_saved_
searches` / `_library_collections_saved_searches_total` are referenced in
exactly 5 places, ALL inside `library_screen.py`, ALL inside
collection-cluster methods --
`_library_collections_capture_presentation` (reads both, for the
render-only presentation object), `_load_library_collections_capture_
entry` (writes both, from `scope.list_saved_searches(1)`), and
`select_library_collection_capture_scope` (reads `saved_searches` to
resolve a pressed saved-search scope row). A repo-wide grep confirms zero
references anywhere outside `library_screen.py`. The `SavedCaptureSearch`
type and `list_saved_searches` method live in `Library/collections_
capture_models.py` and `Library/collections_capture_service.py` --
Collections' own capture-scope service layer, not a generic "search"
module. The "search cluster" the brief warned about is a DIFFERENT,
unrelated feature living in the same file: the Library-wide search rail
and its history (`handle_library_search_changed`, `_load_library_search_
history`, `_persist_library_search_history`, `clear_library_search_
history`, `rerun_library_search_from_history`, …) -- the top search bar's
submit/history mechanism, conceptually adjacent (both are "search") but
architecturally unconnected: none of those methods reference either
saved-searches field, and Collections' saved searches are per-scope
filter presets persisted through the capture repository, not search-bar
history. **Verdict: MOVE, uncontested** -- the census resolves the
brief's flagged ambiguity outright; no BLOCKED condition arose.

**Total: 26 fields moved, 1 field (wiring) stayed, 0 fields BLOCKED.**

### Characterization spot-check

41 `@on`-bound Collections selectors checked with a per-id `grep -rn`
across `Tests/` followed by a manual read of the surrounding lines for an
actual `.press()`/`.click()`/direct-value-assignment/`Input.Submitted`
interaction. **A same-line-only grep undercounts coverage**: one selector
(`retry_library_collection_quick_capture`, `#library-collections-capture-
retry-confirm`) looked unpressed under a same-line check but is genuinely
covered once the press on the line AFTER its `query_one` call is read --
the same multi-line query-then-press shape the export series' own report
already flagged. 24 of the 41 are genuinely covered (23 direct + this
one). The remaining **17 are genuine gaps**, pinned into `Tests/UI/
test_library_collections_characterization.py` across 5 grouped test
functions (mirroring this codebase's own walkthrough-style tests, which
routinely exercise several related handlers inside one continuous Pilot
session). No live bugs found -- all 17 are coverage gaps, not behavior
bugs. See that file's own module docstring for the full per-selector
accounting.

**A new bypass-adjacent gotcha, not from a test but from writing new
characterization tests against this subsystem's OWN async recompose
shape**: pressing a button immediately after a PRIOR async transition's
`_wait_for_condition` resolves true can race that transition's own
trailing `_refresh_library_collections_capture_reader()` call --
`_run_library_collections_capture_transition` schedules its SECOND
recompose only after `await task` completes, and a condition watching
`controller.state` can observe the state mutation before that second
recompose has actually rebuilt the DOM. A widget fetched via
`_wait_for_selector` immediately after such a condition resolves can be
briefly stale, and pressing a toggle in that window can appear to silently
no-op (the flag DOES flip, but the query for the newly-revealed content
made in the same tick sees the pre-toggle tree). Symptom: `NoMatches` on
a selector that "should" exist, or a 30s `_wait_for_selector` timeout on
one that should appear instantly. Fix: a few extra `await pilot.pause()`
calls after the setup condition, before touching DOM elements that depend
on it -- not a code bug, a test-timing gap. Cost real debugging time in
this task (two of five characterization tests needed this fix before they
passed reliably); worth checking for in any future subsystem's
characterization tests that select a row/item and then immediately act on
its detail pane in the same test.

### `LibraryCollectionsState` + shims

`tldw_chatbook/UI/Library_Modules/library_collections_state.py`: a
`@dataclass` with all 26 fields, verbatim defaults, matching the export
precedent's single-prefix shape (`_library_collections_` for every
field -- Collections' own subsystem name is already plural, so unlike
Conversations no field needed a DIFFERENT prefix variant; no
`COLLECTIONS_PLURAL_STATE_FIELDS` constant exists here). Three fields
(`reader_preferences`, `reader_persistence_locks`, `reader_layout`) are
entangled with other subsystems' shared init code exactly like the
conversations exemplar's own trio and keep their original `__init__`
assignment lines untouched.

**A new deviation from the "construct at the position of the first
removed field" default, required by entanglement ordering**:
`reader_preferences` is entangled with a tuple-unpack shared with Media/
Conversations/Notes/File Notes/Prompts/Skills that executes chronologically
BEFORE Collections' other (non-entangled) fields are assigned in the
original `__init__` -- unlike the conversations exemplar and the export
series, where every entangled field's original line happened to sit AFTER
the position their state object was constructed at. Constructing
`self._collections_state` at the "first removed field" position (which
sits after the shared tuple-unpack) would raise `AttributeError` the
first time that unpack's `self._library_collections_reader_preferences`
target tried to route through the not-yet-installed property into a
not-yet-existing object. Fixed by constructing `self._collections_state =
LibraryCollectionsState()` at the SAME early point
`self._conversations_state` is constructed instead -- before the shared
tuple-unpack, not at the first non-entangled field's position. This is
still a pure, behaviorally-transparent move: every one of the 23
non-entangled fields' defaults is a static literal (no field needed a
constructor argument, unlike Export's computed `form` default), so
constructing the dataclass earlier than usual has no observable
side effect. **Any future subsystem whose entangled reader-preferences
field is NOT the chronologically-last thing assigned before its own
non-entangled block should check this ordering explicitly** before
assuming "construct at the first removed field" is safe -- it is only
safe when no entangled field's original line precedes that position.

Shim: mirrors Export's single-prefix generated-property-loop shape
exactly (`for _cos_field in dataclasses.fields(LibraryCollectionsState):
setattr(LibraryScreen, "_library_collections_" + _cos_field.name,
property(...))`), sentinel-wrapped, installed at module end.

### Size ratchet

**The stale-pin gap this task's own brief flagged (43413 pin vs 43412
true measurement) is now closed.** `git show HEAD:...|wc -l` on the
pre-task tree measured 43412, one below the recorded `_BUDGETS` pin of
43413 -- a 1-line slack that had gone unnoticed since the export cleanup
PR. This task's own net change (import +1, field-block removal -23,
early-construction line +2, shim block +20) landed at **43410 lines,
1281 methods** (measured fresh, post-edit) -- both below the stale pin
AND below the true pre-task baseline, so `_BUDGETS` is lowered to
`43410` in this same commit, closing the 1-line gap rather than carrying
it forward. Pin trajectory:
`43413 (stale) / 43412 (true) -> 43410` (methods unchanged at 1281 -- a
pure field move, zero method bodies touched).

### Wiring test -- TDD evidence

`Tests/Architecture/test_library_collections_wiring.py` was written and
run FIRST, before `library_collections_state.py`'s shim installation
existed on `LibraryScreen`:

```
FAILED Tests/Architecture/test_library_collections_wiring.py::test_state_object_fields_match_the_shim_surface
```

RED confirmed (an assertion failure -- the module already existed by the
time the test was written in this task's own execution order, but no
shim property existed on `LibraryScreen` yet). After the screen edit
landed:

```
Tests/Architecture/test_library_collections_wiring.py .            [100%]
1 passed
```

GREEN. Scope matches the export series' own Task 2 precedent exactly
(state-object fields <-> shim surface only; a controller PR in this
series will add the full-cluster/same-name-delegator-forwarding shape
later).

### Sweep evidence — zero real branch-unique failures

The full xdist paired-baseline sweep (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`) found 333 failed/3906 passed on
this task's branch versus 340 failed/3894 passed on a `git stash -u`
pristine baseline of the pre-task tree. Diffing the failure-name sets: 2
unique to branch, 9 unique to baseline only, 331 shared. Both
branch-unique failures
(`test_library_prompt_history_no_change_keeps_selection_and_retry_available`,
`test_library_starter_production_geometry_and_focus_order[size1]`) were
confirmed pure xdist ordering/shared-state noise by a direct single-process
rerun (individually and combined) — both pass cleanly every time, and
neither touches Collections or shares a fixture with this task's diff.
**Zero real regressions** — matching the export series' own Task 2 sweep
result more closely than its Task 3/4 sweeps (which each surfaced genuine
new bypass-shape exclusions), consistent with this task's pure
field-relocation shape.

## 14. The collections series, task 2 (controller PR) — as landed

Wave-2 task 6 (collections controller PR, collections series 2/3) is
complete. Full derivation, cluster table, and byte-for-byte/free-name
verification method are in `task-6-report.md`; this section records the
headline numbers and the one new finding for the next subsystem.

**64 of the 67 "collection"-named candidates move, uncontested** (3 are
Prompts-owned, reconfirmed from task 5's own field-level exclusion). This
is the first controller PR in this recipe's rehearsal with **zero method-
level exclusions** — no `@work` hazard, no class-level or instance-
attribute monkeypatch, no unbound-fake-self/silent-Mock call, found for
ANY of the 64 by a script-driven sweep of the whole `Tests/` tree (not a
sample). Contrast export (29 of 51 excluded) and see §12/§13 for why:
Collections' cluster is small, cohesive, and entirely mediated through one
pre-existing headless engine (`LibraryCollectionsCaptureController`) that
tests exercise through full-harness Pilot sessions rather than unbound
`SimpleNamespace` unit calls.

**The brief's own "browse-controller-delegation exclusion" instruction
named a file that does not exist.** `library_collections_browse_
controller.py` is not a real module; the only pre-existing Collections
controller is `library_collections_capture_controller.py` (a headless
orchestration engine, not a screen-delegation target). A guard test,
`Tests/UI/test_product_maturity_phase39_library_collections.py::
test_collections_route_has_no_generic_container_controller_or_panel`,
asserts the STRING `"LibraryCollectionsBrowseController"` (a retired
container-based design) never appears in `library_screen.py` — this is a
naming-retirement guard, not a live controller. The new controller is
named `LibraryCollectionsController` (matching the state-PR's
`LibraryCollectionsState` and the export series' own naming), which does
not collide with the guard. **Any future subsystem's controller-PR brief
should verify a named "existing controller to check delegation against"
actually exists (`find`/`grep` for the literal filename) before trusting
the brief's own naming** — this one was a stale/confused reference, caught
only by reading the actual guard test instead of assuming the file was
just not yet explored.

Pin trajectory (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`):
`43410/1281 -> 42486/1281` (methods unchanged — pure move, 64
`FunctionDef`s out, 64 one-line delegators in).

**Byte-for-byte verification by TOOLING, not manual diff-reading**: this
task extracted each of the 64 method's exact source segments with an
`ast`-driven script (using the ORIGINAL file's own line offsets, never
hand-retyped), assembled the new controller module from that extracted
text plus a hand-written header/footer, then re-verified with a SECOND
script that re-parsed both the pre-move screen file and the finished
controller module and asserted byte-for-byte identity per method. For a
move this size (64 methods, ~1300 lines), this generalizes past the
"read every body once" discipline earlier tasks used: **write the
extraction and the verification as scripts, not as a sequence of manual
Read/Edit operations**, for any future subsystem whose cluster exceeds
~40-50 methods.

**Correction (fix round 1, post-review) -- a dynamic-dispatch census gap,
new sub-shape**: this task's own module docstring and report originally
claimed a clean dynamic-dispatch sweep (no `getattr`/`setattr` call using
an f-string or dict-literal argument touches a Collections name, beyond
the one pre-existing reader-preferences hit §12 already documents). That
census script only matched a dict-literal/f-string passed DIRECTLY as the
call's own argument. It missed a two-step shape that DOES exist inside
this task's own moved cluster: `retain_library_collection_quick_capture_
input` builds a small DOM-id-keyed dict, `.get()`s a NAME into a local
variable, then calls `setattr(self, attribute, event.value)` with that
variable on a later line. **Safe by construction** (the three dispatched
names are among the 26 fields with a full `property(get, set)` shim, not
read-only), but the census's own claim of completeness was wrong until
corrected in review. **Any future subsystem's dynamic-dispatch census
should also grep for a `dict.get(...)` result assigned to a variable that
later flows into `setattr`/`getattr` within the same function** -- not
just a literal argument passed directly to the call -- to avoid
re-missing this shape.

**Sweep evidence — concurrent branch/baseline xdist runs amplify
flakiness measurably; see §7's updated documented-failures entry for the
full numbers and per-test disposition.** Zero real regressions confirmed,
but the concurrent-run shortcut (both full sweeps launched at once to
save wall-clock time) cost real investigation effort re-running 12
branch-unique names individually/combined and, for the one that
reproduced, against the pristine baseline too. **Future tasks should run
the two full sweeps sequentially when time allows.**

## 15. The collections series, task 3 (cleanup PR) — as landed

Wave-2 task 7 (collections cleanup, collections series 3/3) is complete,
closing out the collections rehearsal (§13/§14). This section records what
landed and reconfirms/extends the recipe's own guidance for the next
subsystem's cleanup PR.

### Dynamic-dispatch census (screen + tests), with the dict.get→variable→setattr guidance applied

Re-derived task 6's own dynamic-dispatch findings before deleting the
shim, plus the NEW two-step "`dict.get(...)` into a variable, then into
`setattr`/`getattr`" shape task 6's fix round added to this recipe:

- **The one screen-side dynamic-dispatch site touching Collections**:
  `_replace_library_reader_preference`/`_persist_library_reader_
  preference`'s 7-destination `{destination: attribute_name}` dicts (the
  SAME shared shell dispatcher the conversations exemplar's Task 9 and the
  export series' Task 4 already fixed for their own subsystems). The
  `"collections"` entry's value string changed from the flat
  `"_library_collections_reader_preferences"` to the dotted
  `"_collections_state.reader_preferences"`, read via the existing
  `operator.attrgetter(...)` calls and written via the existing
  `_assign_library_reader_preferences_attribute(owner, attribute, value)`
  helper -- both already fully generic dotted-vs-flat passthroughs (the
  export series' own lesson: reuse, don't re-derive), so neither needed a
  logic change, only the string value.
- **The SAME dynamic-dispatch shape, independently, inside a TEST fixture**:
  `Tests/UI/test_library_adaptive_reader_closeout.py`'s `DESTINATION_
  CONTRACT` dict (a `{destination: (rail_selector, shell_selector, ...,
  preferences_attr, layout_attr)}` table used by a battery of
  cross-destination closeout tests) carries its own `"collections"` entry
  with the same two flat strings, already consumed everywhere via
  `operator.attrgetter` (never a bare `getattr`) -- exactly mirroring the
  fix this same file's own `"conversations"` entry needed at the
  conversations exemplar's Task 9. Retargeted both strings to
  `"_collections_state.reader_preferences"`/`"_collections_state.
  reader_layout"`; zero consumption-site changes needed since every reader
  of this table already goes through `attrgetter`.
- **The controller-internal dict.get→variable→setattr site task 6's fix
  round found** (`retain_library_collection_quick_capture_input`'s
  DOM-id-keyed `attributes` dict, `.get()`-ed into a local variable, then
  `setattr(self, attribute, event.value)`): confirmed **out of this
  task's scope** -- it lives entirely INSIDE `LibraryCollectionsController`
  (moved there byte-for-byte by task 6), dispatches to 3 of the
  controller's own generated state-shim properties (full `property(get,
  set)` pairs per task 6's report), and was never routed through anything
  this cleanup task touches (the screen-side shim, the 14 pruned
  delegators, or the field-retarget pass). Per the task brief's own
  framing ("Known dynamic-dispatch site INSIDE THE CONTROLLER (safe,
  stays)"), re-verified rather than re-fixed.
- **Applying the new guidance itself** (grep for `\.get\(` results
  assigned to a variable that later flows into `setattr`/`getattr` within
  the same function, not just a literal argument): run across
  `library_screen.py` and every retargeted test file. Zero NEW instances
  found touching any of the 26 Collections state fields or the 14 pruned
  method names -- the one instance this guidance exists to catch
  (`retain_library_collection_quick_capture_input`, above) was already
  known and already out of scope.

### Screen-side field retarget

**14 literal `self._library_collections_<field>` sites**, across the
`__init__` entangled-trio lines and 2 shared-shell dispatcher methods
(`_replace_library_reader_preference`, `_persist_library_reader_
preference`) plus `restore_state` and the Collections reader-shell
`compose` call site -- a single mechanical word-boundary-anchored regex
pass (26-field mapping table, longest-match-first, verified against zero
accidental collisions with any of the 64 moved method names before
running) rewrote all 14 to `self._collections_state.<field>` in one pass;
re-verified afterward with a zero-result grep for every one of the 26
flat field names over the whole file. `_library_collections_capture_
controller` (the ONE field task 5 deliberately kept OFF the state object
as "wiring, not state") was excluded by construction -- it is not one of
the 26 mapped names, so the regex never touched its 11 occurrences.

Unlike the conversations/export cleanup PRs, the vast majority of
Collections' 26 fields have ZERO remaining screen-side literal references
after task 6's controller move -- only the three entangled fields
(`reader_preferences`, `reader_persistence_locks`, `reader_layout`) and
`requested_page` still had any screen-side literal access at all, because
task 6 moved all 64 cluster METHODS (the only other consumers) onto the
controller in one PR with zero exclusions (§14), leaving no screen-
resident method to reach the other 22 fields directly.

### Test retarget — per file, widened beyond the brief's stated `Tests/UI`/`Tests/Library` scope

The brief named `Tests/UI/` and `Tests/Library/` as the retarget scope
(matching the export series' own widened-root lesson, §12). A repo-wide
grep for `_library_collections_<field>` across the WHOLE `Tests/` tree
(not just those two directories) found a THIRD location the brief didn't
name: **`Tests/Live/test_library_collections_capture_walkthrough.py`** --
a non-network-gated, `@pytest.mark.asyncio` full-harness walkthrough file
that is collected in an ordinary pytest run (no `Tests/Live` ignore rule
in `pyproject.toml`, confirmed by reading `addopts`) and reaches 22 flat
Collections field names directly on a real `screen` instance. Left
unretargeted, these 22 references would have broken the instant the shim
was deleted, undetected by any check scoped to `Tests/UI`/`Tests/Library`
alone. **Any future subsystem's cleanup-PR test census should grep the
whole `Tests/` tree for the flat field names, not just the two directories
the recipe's prose names as the primary scope** -- this is the same
"widen the root" lesson §12 already drew for controller-PR batteries,
now shown to apply to a cleanup PR's test-retarget census too.

| File | Retargets | Notes |
|---|---|---|
| `Tests/UI/test_library_collections_characterization.py` | 13 (`screen`) | 1 docstring paragraph updated from future to past tense (the shim is gone now, not "will be gone") |
| `Tests/UI/test_library_collections_capture_reader.py` | 8 (`screen`) | |
| `Tests/UI/test_library_adaptive_reader_closeout.py` | 5 (`screen` + 2 `DESTINATION_CONTRACT` dict strings) | the dynamic-dispatch fixture fix, above |
| `Tests/UI/test_library_shell.py` | 1 (`screen`) | inside a shared multi-parametrization helper; re-run confirmed the one pre-existing `expected_scope4` failure (task 5's documented list) is unaffected in cause |
| `Tests/Live/test_library_collections_capture_walkthrough.py` | 22 (`screen`) | not named by the brief; found by widening the census root (above) |

**49 retargets total, zero assertion VALUE changes** -- every one is a
receiver-path rewrite only (`screen._library_collections_<field>` →
`screen._collections_state.<field>`), confirmed by running each file
before and after and diffing the pass/fail set (§ battery, below). No
unbound-fake-self construction touches any Collections field (task 6's
own battery already confirmed zero such fakes exist for this cluster), so
no fixture needed the "flat kwargs → nested `_collections_state=
SimpleNamespace(...)`" restructuring the conversations/export exemplars'
own cleanup PRs needed.

### Delegator census (all 64)

Per-name repo-wide grep (`tldw_chatbook/`, `Tests/`, including
`Tests/Live/`) for every one of task 6's 64 moved-cluster names, split by
decorator shape:

- **41 `@on`-decorated handlers: all KEEP, unconditionally** -- per this
  task's own brief ("`@on` always stays"), regardless of whether the name
  appears literally in any test (Pilot-driven `.press()`/`.click()` calls
  trigger these via Textual's CSS-selector message dispatch, not by
  literal method-name reference, so a zero-grep-hit `@on` handler is not
  evidence of deadness).
- **5 non-`@on` methods with a screen-resident caller beyond their own
  delegator body** (`_sync_library_collections_reader_layout_from_shell`,
  `_mirror_library_collections_reader_preference`,
  `_restore_library_collections_page`, `_library_collections_capture_
  presentation`, `_load_library_collections_capture_entry`): KEEP --
  each is called from `request_library_reader_layout_refresh`,
  `_toggle_library_media_reader_pane`, `restore_state`, the Collections
  reader-shell `compose` region, or a worker-group target, respectively.
- **4 non-`@on` methods with a direct test call on a real `screen`
  instance** (`_refresh_library_collections_capture_reader`, `_run_
  library_collections_capture_transition`, `_select_library_collection_
  capture`, `_export_library_collection_legacy_recovery`): KEEP -- each
  confirmed via `await screen.<name>(...)` in `Tests/UI/test_library_
  collections_capture_reader.py` and/or `Tests/Live/test_library_
  collections_capture_walkthrough.py`.
- **14 non-`@on` methods with ZERO consumers beyond their own delegator
  body and this task's own wiring-test shape-check list**
  (`_library_collections_capture_request`, `_ensure_library_collections_
  capture_controller`, `_notify_library_collections_warning`,
  `_capture_library_collection_quick_capture_draft`, `_reset_library_
  collection_quick_capture_draft`, `_submit_library_collection_quick_
  capture`, `_library_collection_capture_filter_request`, `_apply_
  library_collection_capture_request`, `_page_library_collection_
  captures`, `_update_selected_library_collection_capture`, `_library_
  collection_loaded_capture`, `_library_collection_capture_is_current`,
  `_load_library_collection_capture_highlights`, `_run_library_
  collection_capture_content_action`): **PRUNED** -- each confirmed by a
  repo-wide grep restricted to `*.py`/`*.md` outside the controller
  module, the screen's own delegator body, and this task's wiring-test
  list, before deletion.

**Net: 50 KEEP, 14 PRUNED.** This prune fraction (14 of 64, ~22%) is much
larger than the export series' 1-of-22 (~5%) and closer in shape to (but
still smaller than) the conversations exemplar's 18-of-61 (~30%) --
directly explained by task 6's own finding of ZERO method-level test-
bypass exclusions: because the entire 64-method cluster moved onto ONE
controller in a single PR with no screen-resident sibling class left
behind (unlike export's 11 round-2/round-3-excluded siblings, which each
kept calling their own delegator internally and so kept 21 of 22 alive),
many of Collections' internal helper methods lost their only screen-side
caller in one shot. Deleting a screen delegator whose controller-internal
callers already call the sibling directly (`self.<name>()`, `self` =
controller) is exactly the pure-move byproduct this recipe's delegator-
census step exists to find.

One type-annotation follow-on from the prune: `CapturePageRequest` (a
`Library.collections_capture_models` type) appeared in exactly 3
signatures, ALL THREE inside pruned delegators
(`_library_collections_capture_request`, `_library_collection_capture_
filter_request`, `_apply_library_collection_capture_request`) -- pruning
all three made the import newly dead, folded into the dead-import pass
below rather than treated as a separate finding. `CaptureIdentity` (used
by both a KEPT delegator, `_select_library_collection_capture`, and a
PRUNED one, `_library_collection_capture_is_current`) and
`CollectionsCaptureReaderPresentation` (used only by the KEPT `_library_
collections_capture_presentation`) both stay alive -- checked individually
rather than assumed dead-by-association with their sibling prunes.

### Shim block deletion

The task-5-generated `_library_collections_<field>` property-shim loop
(installed at module end, `dataclasses.fields(LibraryCollectionsState)`-
driven) was deleted wholesale once the census above confirmed zero
remaining consumers anywhere in `tldw_chatbook/` or `Tests/` outside
`LibraryCollectionsController`'s own PERMANENT generated shim loop
(installed by task 6, reading `self._collections_state_accessor().
<field>` -- untouched by this task, per the task brief's explicit "controller
shims STAY" instruction).

### Import verification

**10 dead imports removed**, each verified single-occurrence (import line
only) in `library_screen.py` via per-name grep, then checked against
`Tests/Architecture/test_library_support_layer_surface.py`'s `_SURFACE`
dict (the PR-0a re-export contract) before deletion -- none of the 10
belongs to any of `_SURFACE`'s 5 listed modules (`screen_constants`,
`screen_support_types`, `note_session_port`, `canvas_sync`,
`screen_helpers`), all PR-0a support modules unrelated to the two source
modules these 10 actually come from (`Library.collections_capture_
models`, `Widgets.Library`, `Library_Modules.library_collections_capture_
controller`):

- 4 flagged dead by task 5 (never became live again after the state PR):
  `CaptureCapabilities`, `CaptureHighlight`, `SavedCaptureSearch`,
  `CollectionsReaderMode`.
- 4 flagged dead by task 6 (became dead after the controller move; task 6
  deliberately left them for this cleanup PR to remove, per the export
  series' own Task 3/Task 4 split): `CAPTURE_SORTS`, `CaptureSaveRequest`,
  `CollectionsCaptureError`, `ExternalNoteReference`.
- 1 flagged dead by task 5 for a different reason (a state-object-shaped
  type never adopted): `CollectionsCaptureControllerState`.
- 1 newly dead as a DIRECT RESULT of this task's own delegator prune:
  `CapturePageRequest` (see the delegator-census section above).

Two names task 6's report explicitly warned NOT to treat as dead
(`CaptureIdentity`, `CollectionsCaptureReaderPresentation`) were
individually re-checked post-prune and confirmed still alive (2
occurrences each) -- left in place, not removed.

### Wiring test finalization

`Tests/Architecture/test_library_collections_wiring.py`:
`test_state_object_fields_match_the_shim_surface` DELETED (screen shim
gone, mirrors the conversations/export precedent exactly);
`_COLLECTIONS_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (14 names) added,
with `test_screen_delegates_collections_handlers` skipping those names and
instead asserting their genuine ABSENCE from `LibraryScreen`; module
docstring rewritten to describe the finished 3-task series. 4 of the
original 5 tests remain (the shim-surface test's removal is the only
count change) -- all 4 green post-cleanup.

### Size ratchet

Fresh measurement via the ratchet's own `_measure` semantics (`ast`-walked
line count + `LibraryScreen` method count, not `wc -l`): **42411 lines,
1267 methods** -- 1267 = 1281 (task 6's post-move count) − 14 (exactly the
pruned delegator count), confirming the prune was a pure deletion with no
replacement. Lowered in this same commit per recipe §6.

Full pin trajectory for the collections series:
`43410/1281 (pre-task-5) → 42486/1281 (task 6, controller PR) →
42411/1267 (task 7, cleanup PR, final)`.

**A note on recipe §6's rebase instruction**: by the time this task ran,
`origin/dev` had diverged from this branch's merge-base by 337 commits,
dozens touching `library_screen.py` (real, unrelated Library feature work
landed on `dev` during this wave-2 rehearsal's own execution window).
Rebasing a 3-task-deep-per-subsystem, multi-week decomposition rehearsal
branch onto 337 commits of unrelated concurrent work was judged out of
scope for a single cleanup task and not requested by the task brief
(which scopes work to this worktree's own branch, no pushes). Measured
fresh on this branch's own HEAD instead, consistent with tasks 2-6's own
practice in this series. **Flagging for whoever runs the next subsystem's
series**: §6's rebase-before-measuring rule assumes a branch that stays
close to its upstream; this rehearsal branch no longer does, and the rule
may need an explicit "how stale is too stale" amendment before the final
shell-pass task (§8's phase 5) if the gap keeps growing.

### Battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **Wiring suites**: `test_library_collections_wiring.py` (4, post-
  deletion/post-pruning) + `test_library_export_wiring.py` (5) + `test_
  library_conversations_wiring.py` (6) — 15 passed.
- **Characterization files**: `test_library_collections_characterization.py`
  (5) + `test_library_export_characterization.py` (5) + `test_library_
  conversations_characterization.py` (4) — 14 passed.
- **Recompose ratchet + support-layer surface**: `test_library_recompose_
  ratchet.py` + `test_library_support_layer_surface.py` — 14 passed
  (recompose pin unaffected: zero `refresh(recompose=True)` sites touched).
- **Both size-ratchet guards, full suite**: 3 passed, 2 failed (the two
  documented pre-existing `chat_screen.py` rows only).
- **Collections-adjacent live-functional suites**: `test_library_
  collections_capture_reader.py`, `test_library_collections_capture_
  controller.py`, `test_library_collections_reader_geometry.py`, `test_
  product_maturity_phase39_library_collections.py` — 43 passed (matches
  task 6's own count for this same battery).
- **`test_library_adaptive_reader_closeout.py`** (full file, incl. the
  dynamic-dispatch-fixed `DESTINATION_CONTRACT` table and the specific
  `test_closeout_single_app_route_cycle` test task 6's fix round proved
  DOES exercise 2 Collections methods): 14 passed.
- **`Tests/Live/test_library_collections_capture_walkthrough.py`**
  (widened-scope retarget): 2 passed, 1 skipped (network-gated, unrelated).
- **`Tests/UI/test_library_shell.py::test_library_landing_continue_
  receipt_accepts_only_authoritative_source_scopes`** (contains the
  retargeted line inside a shared parametrization helper): 5 passed, 1
  failed -- the SAME `[browse-collections-expected_scope4]` failure task
  5's own report already documented (`state["library_continue_receipt"]`
  `None` vs. expected dict), reconfirmed identical in symptom.
- **`-k "collection and library"` across `Tests/UI` + `Tests/Library`**
  (narrow per-task check): **361 passed, 3 failed** in one single-process
  run -- the SAME 3 names already documented pre-existing by task 5,
  reconfirmed by task 6 (`test_library_starter_deep_link_opens_hidden_
  collection_or_note_route`, `test_library_landing_continue_receipt_
  accepts_only_authoritative_source_scopes[browse-collections-
  expected_scope4]`, `test_get_library_collection_supported_types_round_
  trip_public_ids`). Matched the recipe's own documented list exactly
  (name-for-name and count-for-count), so no separate `git stash -u`
  re-derivation was needed for this narrow check per §7's own "check this
  list before re-deriving" guidance.
- **Full xdist paired-baseline sweep** (`Tests/UI -k "library" -p
  no:randomly -q -n 8 --dist worksteal`), run SEQUENTIALLY (branch, then a
  `git stash -u` pristine baseline of the same `91feba4a7` pre-task tree,
  restored via `git stash pop` afterward and re-verified via `ast.parse` +
  `git status` clean) per task 6's own forward note -- NOT concurrently:
  - Branch: **333 failed, 3906 passed** (1314.45s / 21:54).
  - Baseline: **337 failed, 3902 passed** (1306.25s / 21:46).
  - Diff (`comm` on the two sorted unique-failure-name sets): **4
    branch-unique**, **8 baseline-unique**, the remainder shared.
  - All 4 branch-unique names re-run single-process, combined:
    `test_audio_cpp_model_library_handoff.py::test_rapid_away_back_
    reclaims_request_after_old_operation_drains` and `test_library_media_
    reader_traversal_t22207.py::test_loading_banner_paints_in_place_
    without_body_rebuild` failed in the combined run; `test_library_
    prompts_canvas.py::test_library_prompt_history_no_change_keeps_
    selection_and_retry_available` and `test_library_shell.py::test_
    library_media_page_error_retains_rows_and_gates_unsafe_controls`
    passed. Both of the 2 that failed combined **passed individually in
    true isolation**. The SAME 4-test combined invocation, re-run against
    a pristine `git stash -u` baseline under identical conditions,
    reproduced ONE of the two (`test_rapid_away_back_reclaims_request_
    after_old_operation_drains` failed there too; `test_loading_banner_
    paints_in_place_without_body_rebuild` passed on baseline this time --
    a different subset flaking each time) -- the SAME "shared-state/
    ordering sensitivity to which OTHER tests ran earlier in the process,
    identical on both code versions" shape task 6's own report already
    established for `test_one_megabyte_markdown_document_is_not_reparsed_
    per_keystroke`. **Zero real regressions** -- none of the 4
    branch-unique names touches Collections code, this task's diff, or a
    fixture this task's diff shares (Audio/CPP model handoff, Media
    reader traversal, Prompts canvas, Media page-error), and the one
    combined-run failure that reproduces at all reproduces on BOTH trees.
- **Preflight**: all six checks green (CSS bundle, profile-owned-path
  census, diagnostic inventory, backlog task ids, chachanotes table
  allowlist, index plan pins).

### Lessons

1. **The cleanup-PR test census should widen past the brief's stated
   directories, the same way a controller-PR battery's `-k` search
   should** (§12's own lesson, now shown to generalize to a DIFFERENT
   task type): this task's own brief named `Tests/UI/` and `Tests/
   Library/`; a repo-wide grep found a third, unnamed location (`Tests/
   Live/`) with 22 real consumers of the flat field names that would have
   broken silently at shim deletion if the census had trusted the brief's
   stated scope literally.
2. **A delegator prune's type-annotation fallout is a real, checkable
   side effect, not a hypothetical**: `CapturePageRequest` went from
   3-signatures-alive to fully-dead as a direct, mechanical consequence of
   pruning the 3 delegators that carried its only usages -- checked by
   re-running the occurrence-count census AFTER the prune, not assumed
   from the pre-prune count.
3. **A cluster with zero controller-PR method-level exclusions can still
   produce a LARGE cleanup-PR prune fraction** -- the inverse of what
   might be assumed ("no test-bypass exclusions" sounds like "everything
   stays wired"). The mechanism is the opposite: exclusions are what KEEP
   screen delegators alive (an excluded, screen-resident sibling method
   calls its moved neighbors via `self.<delegator>()`); a cluster that
   moves entirely, in one PR, onto one controller has no such sibling
   left, so more of its purely-internal helper methods lose their only
   caller in the same step. Collections' 14-of-64 (22%) prune fraction,
   next to export's 1-of-22 (5%, eleven round-2/round-3 exclusions kept
   calling their delegators) and conversations' 18-of-61 (30%, a genuinely
   split Reader/Browse controller pair with less internal self-calling),
   is now a third data point for this same inverse relationship.

## 16. Wave-2 close — summary

Wave-2 (`.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio`,
branch `refactor/library-decomp-wave2-cold-trio`) is closed: the census
anti-slack guard (Task 1), the export series (Tasks 2–4, §12), and the
collections series (Tasks 5–7, §13–§15) are complete; the search series
(Tasks 8–9) is BLOCKED at the entanglement gate by design, deferred to a
combined search+RAG wave-3 series (task-31203). This section is the
wave-level pin trajectory, the full verification battery, and the wave's
own lessons — read this before starting wave 3.

### Pin trajectory — full wave-2 chain

Re-derived from `git log` (the `_BUDGETS["tldw_chatbook/UI/Screens/
library_screen.py"]` value at each commit, not carried over from any
report) and cross-checked against the ratchet file's own inline comment
history:

| Task | PR | Commit | `_BUDGETS` after |
|---|---|---|---|
| — | (foundation tip, wave-2 start) | `2b20ebbb9` | 43965 / 1282 |
| 1 | Census anti-slack guard | `477704580` | 43965 / 1282 (unchanged — guard added to the recompose ratchet file, not this one) |
| 2 | Export state | `f4e8acecf` | 43930 / 1282 |
| 3 | Export controller | `4cc9b6109` | 43432 / 1282 |
| 4 | Export cleanup (series complete) | `cdb43ebcc` | 43413 / 1281 |
| 5 | Collections state | `bca923b4c` | 43410 / 1281 |
| 6 | Collections controller | `09d238f50` | 42486 / 1281 |
| 7 | Collections cleanup (series complete) | `39a976321` | **42411 / 1267 (wave-2 final)** |
| 8 | Search census — BLOCKED | (report only) | unchanged, 42411 / 1267 (zero code touched) |
| 9 | Search cleanup | — | MOOT, no move occurred |

Full chain: `43965/1282 → 43930/1282 → 43432/1282 → 43413/1281 →
43410/1281 → 42486/1281 → 42411/1267 (final)`. Net wave-2 shrink: 1554
lines, 15 methods — all 15 fewer methods are pruned dead delegators (1
export + 14 collections), not moved bodies (a pure move is always
net-zero methods on the screen: N bodies out, N one-line delegators in).
Task 10's own fresh `_measure()` (ast-walked line count + `LibraryScreen`
method count, matching the ratchet's own semantics exactly, not `wc -l`)
gives **42411 lines / 1267 methods** — an EXACT match to the recorded pin,
zero drift, nothing to lower.

### Subsystem outcomes (§8 table, detailed)

- **Export — complete** (Tasks 2–4, §12): 13 fields moved to
  `LibraryExportState`; 22 of 51 "export"-named method candidates moved to
  `LibraryExportController` (18 other-subsystem, 2 `@work`
  framework-decorator self-type-assertion hazard, 9 unbound-fake-self/
  silent-Mock test bypasses excluded); 1 of 22 screen delegators pruned at
  cleanup.
- **Collections — complete** (Tasks 5–7, §13–§15): 26 fields moved to
  `LibraryCollectionsState` (1 field, `_library_collections_capture_
  controller`, stays screen-side as wiring, not state); 64 of 67
  "collection"-named method candidates moved to
  `LibraryCollectionsController` (3 Prompts-owned excluded, zero
  test-bypass exclusions — the first controller PR in this recipe's
  rehearsal with none); 14 of 64 screen delegators pruned at cleanup.
  The brief's flagged `_library_collections_saved_searches*` boundary
  resolved uncontested (all 5 references collections-internal; the "search
  cluster" the brief warned about is the unrelated top-search-bar feature
  — see §13).
- **Search — BLOCKED at the entanglement gate** (Task 8, no code touched):
  14 candidate methods census'd (23 raw "search"-named matches minus 3
  Prompts-owned and 6 Media-owned). Cross-call census against the 39-strong
  RAG-named method set: **8/14 (57.1%)** entangled under the direct
  reading, **5/11 (45.5%)** under the most conservative possible reading
  (stripping the 3 already-RAG-tagged candidates first) — both far past
  the wave plan's 1/3 (33.3%) gate. The top search bar's submit handler
  (`handle_library_search_submitted`) and its "rerun from history" action
  both call `_start_library_rag_query` directly: the search bar's own
  submit path *is* the RAG query entry point, not a sibling of it — the
  entanglement is structural, not coincidental. Per the wave plan's
  pre-committed contingency, search+RAG becomes ONE combined series in
  wave 3 (task-31203, filed at wave close). Full census:
  `.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/
  task-8-report.md`.

### Verification battery (Task 10, this close)

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, `-p no:randomly`.

- **Wiring suites** (3 files exist for this recipe today — `
  test_library_collections_wiring.py`, `test_library_export_wiring.py`,
  `test_library_conversations_wiring.py`; no search wiring suite exists
  since Task 8 never moved anything): **16 passed** (4 + 5 + 7 — the
  conversations file gained a 7th test since §15's own count of 6, ordinary
  suite growth, not a regression).
- **Characterization files** (collections + export + conversations): **14
  passed** (5 + 5 + 4), matching §15's own count exactly.
- **Both size-ratchet guards, full suite**
  (`Tests/Architecture/test_screen_size_ratchet.py`): **3 passed, 2
  failed** — exactly the two documented pre-existing `chat_screen.py` rows
  (`test_screen_does_not_grow_past_its_budget[chat_screen.py]`,
  `test_task_22507_4_does_not_worsen_chat_screen_base`), no others.
- **Recompose census suite** (`Tests/UI/test_library_recompose_ratchet.py`,
  home of Task 1's anti-slack guard): **6 passed**.
- **Support-layer surface**
  (`Tests/Architecture/test_library_support_layer_surface.py`): **8
  passed**.
- **Preflight** (`./scripts/preflight.sh`): all six checks green (CSS
  bundle, profile-owned-path census, diagnostic inventory, backlog task
  ids, chachanotes table allowlist, index plan pins).

### Full xdist paired-baseline sweep — sequential, per §7's own lesson (below)

Branch = this task's own tree (HEAD `1e466ffac` + this close's doc-only
edits, which touch no test or production-logic file). Baseline = a
path-scoped `git checkout 2b20ebbb9 -- tldw_chatbook Tests` overlay of the
foundation tip (wave-2's own start commit), run, then restored via
`git checkout HEAD -- tldw_chatbook Tests` (verified `_measure()` back to
42411/1267 and `git status` clean afterward) — the multi-commit-back
equivalent of the per-task `git stash -u` technique, since this
comparison spans the WHOLE wave rather than one task's uncommitted diff.
Run SEQUENTIALLY, not concurrently, per Task 6/7's own forward note (§15):

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`1e466ffac` + close) | 335 | 3904 | 1261.02s (21:01) |
| Baseline (`2b20ebbb9`) | 343 | 3895 | 1252.31s (20:52) |

Diffing the two failure-name sets: **5 branch-unique**, 13 baseline-unique,
330 shared. All 5 branch-unique names —
`test_library_media_reader_no_change_sync_t22208.py::
test_no_change_traversal_builds_no_preview_and_copies_no_content`,
`test_library_prompts_canvas.py::
test_library_prompt_history_stale_conflict_reload_refreshes_and_can_retry`,
`test_library_prompts_canvas.py::
test_library_prompts_stale_search_cannot_restore_an_old_filter_caret`,
`test_library_shell.py::
test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`,
`test_library_shell.py::test_library_shell_note_id_deeplink_opens_note_editor`
— re-run single-process, combined: **all 5 pass cleanly**. None touches
Export or Collections code, this wave's diff, or a fixture this wave's
diff shares (Media reader, Prompts canvas, Notes shell). **Zero real
regressions** across the whole wave.

### Probe run

```
.venv/bin/python Helper_Scripts/library_click_probe.py
```

| interaction | settle (ms) | max gap (ms) | recompose | full-update | mounts | nodes |
|---|---|---|---|---|---|---|
| media (switch-in) | 485 | 155 | 0 | 2 | 163 | 113 |
| media (re-click same) | 328 | 54 | 0 | 2 | 79 | 113 |
| media (re-click same, 2nd) | 329 | 56 | 0 | 2 | 79 | 113 |
| notes (switch) | 413 | 195 | 0 | 1 | 110 | 110 |
| notes (re-click same) | 264 | 56 | 0 | 1 | 38 | 110 |
| media (switch-back) | 467 | 156 | 0 | 1 | 165 | 113 |
| notes (switch, 2nd) | 356 | 131 | 0 | 1 | 110 | 110 |
| media (switch-back, 2nd) | 411 | 94 | 0 | 1 | 165 | 113 |

**No prior recorded run of this probe exists anywhere in this repo's docs
or SDD ledgers to diff against** — §9 calls for a before/after pair around
each controller-move PR, but neither Task 3 (export controller) nor Task 6
(collections controller) captured one; this is genuinely the first time
this script's output has been written down. Recorded here as the wave-2
close baseline for whoever runs the next controller-move PR. Consistent
with expectations either way: this probe exercises ONLY the Media/Notes
rail-switch path (the foundation-era freeze §8's Phase C note references,
139–380 ms), which neither the export nor the collections series touches
— every row still shows the pre-Phase-C main-thread cost the design doc
already documents, not a new regression from this wave's moves.

### Lessons

1. **The byte-for-byte canon (§1) extends to comments, and "I carried this
   verbatim" is a claim, not a default.** Task 2's export state-PR move
   retyped/paraphrased three `library_export_state.py` field comments
   instead of copy-pasting them from base `477704580` — an 8-line header
   comment explaining `_library_export_counts`/`_library_export_form`
   semantics was dropped entirely (with the `__init__` call-site comment
   then falsely claiming the detail "lives" in the state module),
   `run_id`'s comment silently renamed three field names to their new bare
   forms, and `cancel_event`'s comment dropped its trailing sentence. The
   task's own self-review had asserted byte-for-byte comment carry
   *without diffing against the base commit to check*. Review caught it;
   fixed in `264314c5f` by restoring all three verbatim (plus a fourth,
   self-caught defect on `origin_row_id`) and verifying with an automated
   normalized-substring comparison against `git show <base>:...`, not
   eyeballing. Generalizes: a "verbatim" claim about anything moved — body
   or comment — needs the same evidence discipline as a test-passing
   claim: diff against the base commit, don't trust memory of having typed
   it carefully.
2. **A cleanup-PR's test census should grep the WHOLE `Tests/` tree, not
   just the directories the recipe's own prose names — a lesson that
   recurred even after being drawn once already.** §12 already drew this
   lesson for a controller-PR battery (export's 9 unbound-fake-self
   exclusions, 4 of them outside `Tests/UI/`); Task 7's collections
   cleanup PR drew it AGAIN, one layer over: its own brief named
   `Tests/UI/` and `Tests/Library/` as the retarget scope (already the
   widened set §12 recommended), and a repo-wide grep anyway found a
   THIRD, unnamed location — `Tests/Live/test_library_collections_capture_
   walkthrough.py`, a non-network-gated, ordinarily-collected walkthrough
   file with 22 real consumers of the flat field names on a real `screen`
   instance. Left unretargeted, these would have broken silently the
   instant the shim was deleted, invisible to any check scoped to the two
   named directories. Generalizes: "the recipe says `Tests/UI` and
   `Tests/Library`" is not the same claim as "nothing outside those two
   directories references this name" — grep the whole tree, every time,
   regardless of how authoritative the stated scope looks.
3. **Sequential, not concurrent, paired-baseline sweeps — concurrency
   amplifies flakiness at a worse-than-1:1 investigation cost.** Task 6's
   controller-PR sweep ran the branch and pristine-baseline xdist
   invocations CONCURRENTLY (two 8-worker processes sharing one machine)
   to save wall-clock time. Both runs landed well above this recipe's
   historical ~330–340 backdrop (349 failed/3890 passed branch vs
   344/3895 baseline) with 12 branch-unique names — several times the
   export/collections-state series' own 2-to-9-ish norm. Confirming all 12
   as noise (11 passed cleanly on re-run; the 1 that reproduced also
   reproduced identically on the pristine baseline under the same
   combined-invocation conditions) cost real investigation time a cleaner
   run would not have needed. Task 7's own cleanup-PR sweep, run
   SEQUENTIALLY per this forward note, landed inside the historical range
   (333f/3906p branch vs 337f/3902p baseline, 4 branch-unique, all
   confirmed noise on the first single-process re-run) — and this close's
   own sequential sweep (335f/3904p vs 343f/3895p, 5 branch-unique, all
   confirmed noise) reconfirms the same pattern a third time. Run the two
   full sweeps sequentially whenever machine time allows.
4. **The RED-commit criterion's actual wording is structural (screen
   untouched, tests red at parent), not literal (which files the commit
   touches).** The recipe's controller-PR step (§1) calls for a wiring-test
   RED commit ahead of the controller move; an earlier, stricter reading
   treated this as "the RED commit contains ONLY the failing test, zero
   production code." Task 6 encountered this twice and stated the ruling
   this recipe now carries forward: the RED commit must leave the SCREEN
   untouched and its delegation tests failing at the parent — the
   controller module MAY ship in the same commit, since nothing on the
   screen delegates to it yet. What matters is that a real RED exists in
   git history (not a same-commit red+green), not which files happen to be
   present in that commit. Cost if this reading is wrong: slightly weaker
   RED purity; the criterion that actually matters — screen untouched,
   tests red at the parent commit — is preserved either way.
5. **The wiring cluster constants are not self-defending — this is a
   known, deliberately-unclosed gap, not an oversight.**
   `_EXPORT_CLUSTER_METHOD_NAMES` (`Tests/Architecture/
   test_library_export_wiring.py`, 22 entries) and
   `_COLLECTIONS_CLUSTER_METHOD_NAMES` (`Tests/Architecture/
   test_library_collections_wiring.py`, 64 entries) are hand-written
   Python tuples, frozen at the PR that derived them from a one-time
   `ast` census of every `LibraryScreen` method whose name contained
   "export"/"collection" (Task 3, Task 6) followed by reading each
   candidate's body to decide true ownership. Nothing re-runs that census
   at test time and diffs it against the tuple. Concretely: a future 23rd
   export-named method, or 65th collections-named method, added to
   `LibraryScreen` — whether genuinely subsystem-owned or a same-named
   coincidence from another subsystem, exactly the ambiguity Task 3/6 had
   to resolve by hand — is invisible to every wiring and architecture
   test this recipe has. It is not flagged as needing a cluster-
   membership decision, not required to move, not required to be
   excluded-with-reason; it simply sits on the screen, unaccounted for,
   and every existing test still passes. This is a different axis than
   the recompose census's own anti-slack guard (§16, `Tests/UI/
   test_library_recompose_ratchet.py::test_census_pin_is_not_left_slack`,
   task-27019): that guard catches the PIN drifting stale relative to a
   count that's re-measured on every run; no equivalent re-measurement
   exists for cluster membership itself. First noted as a deferred minor
   during Task 3 (export, "cluster constants not self-defending against a
   23rd export method — canon-consistent"); the wave-2 final review
   extended the observation to collections and named it Important enough
   to record durably rather than leave buried in a git-ignored ledger.
   Deferred, not fixed, by design — closing it needs an active guard (an
   AST re-census compared against the frozen tuple, failing when the two
   diverge) that no task in wave 2 was scoped to build; a candidate for
   wave 3 or a dedicated follow-up task.

## 17. Controller-file size governance (task-31203 AC#4)

The wave-2 final review named a real gap: the screens this program
decomposes FROM (`chat_screen.py`, `library_screen.py`) are governed by
`Tests/Architecture/test_screen_size_ratchet.py`, but the controller files
it decomposes INTO — `UI/Library_Modules/*_controller.py` — had no size
governance at all. By wave-2 close two of them (`library_collections_
controller.py` at 1,689 lines, `library_conversations_controller.py` at
1,738) were already screen-scale, and wave 3 (this plan) plus six more
subsystems after it will add more. Wave-3 Task 1 closes this gap. The
guard lives in `Tests/Architecture/test_library_modules_size_ratchet.py`
(a new sibling file, not a change to the screen ratchet — the two
enforce genuinely different shapes; see below).

### The decision: option (a), exact per-file `_BUDGETS` rows, discovered by glob

Three options were on the table (wave-3 plan, Task 1):

- **(a) `_BUDGETS`-style exact rows per controller**, re-pinned at each
  sanctioned landing, mirroring the screen ratchet's own flow.
- **(b) A single aggregate `Library_Modules` budget** (one number for the
  whole directory's controller-file total).
- **(c) A looser per-file ceiling with slack tolerance** (no exact pin,
  just "don't exceed N", with N chosen generously).

**Chosen: (a), with one addition beyond the screen ratchet's own
design — glob-based discovery of ungoverned files, not just a hand-kept
dict.** Reasoning:

- **(b) rejected**: an aggregate budget cannot localize which controller
  grew, so a review sees "the total went up" with no signal about
  whether that is subsystem X's sanctioned move or subsystem Y's creep —
  exactly the ambiguity per-file governance exists to remove. It also
  means every subsystem's PR touches the same shared number, which
  multiplies merge-conflict surface across ~10 eventual controllers
  landing on overlapping timelines (this wave's own Global Constraints
  already budget for "dev races" on the screen's single row; an
  aggregate would multiply that contention across every concurrent
  controller PR, not just the screen's).
- **(c) rejected**: a generous fixed tolerance either has to be loose
  enough to survive a large sanctioned move (in which case it does
  nothing to catch creep between moves, the actual failure mode this
  task exists to close) or tight enough to catch creep (in which case it
  fails on every sanctioned move exactly like a naive "only ever go
  down" ratchet would, per the wave-3 plan's own stated design tension).
  There is no single tolerance value that serves both purposes across
  controllers ranging from ~280 to ~2,000+ lines.
- **(a) chosen**: an exact per-file pin, re-measured and re-set in the
  same commit as every sanctioned move — identical in spirit to how
  `_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]` has already
  been re-pinned five times across the conversations/export/collections
  exemplar series (§11, §12, §15) — localizes every review to exactly
  the one file that changed, costs one dict entry per new controller (a
  maintenance cost linear in subsystem count, not compounding), and
  reuses a mechanism the program has already exercised repeatedly rather
  than inventing a second one.

**The one deliberate improvement over the screen ratchet's own model**:
the screen ratchet's `_BUDGETS` is a hand-maintained dict with nothing
that notices a new screen file needing a row — which is exactly how
`library_screen.py` went ungoverned for the month it tripled in size
before this program's own foundation task added its row. The new test
instead globs `UI/Library_Modules/*_controller.py` at collection time
(`test_every_controller_file_has_a_budget_row`) and fails, by name, the
instant a file matching that pattern has no `_BUDGETS` row. Wave 3's own
search+RAG controller(s) — not yet created as of this task — are
therefore **born governed**: nothing needs to remember to edit the guard
test when they land; the glob finds them and the failure message names
exactly which row to add, at what expression to measure it with.

### Resolving the byte-for-byte-canon tension: same two-check model as the screen ratchet, not a stricter one

The wave-3 plan states the tension precisely: a sanctioned move commit
inflates its destination controller BY DESIGN (the byte-for-byte canon,
§1, moves method bodies verbatim), so a ratchet that only permits a
number to go down would fail every subsystem's own controller-PR the
moment it lands. Re-reading how the screen ratchet is actually operated
in practice (not just its docstring's "may only ever go DOWN" framing)
resolves this: §16 records the screen's own `_BUDGETS` row being RAISED
twice during the wave-2 final-review fix wave, each time with a dated
justification comment explaining the increase. The screen ratchet's real
enforcement is therefore two checks, not one hard one-way rule:

1. A **ceiling** the file may not silently exceed (`test_controller_
   does_not_grow_past_its_budget` here; `test_screen_does_not_grow_
   past_its_budget` there) — this is what stops silent creep.
2. An **anti-slack** bound (`test_budget_is_not_left_slack_after_a_
   move` here; `test_budget_is_not_left_slack_after_a_wave` there) —
   this is what stops a ceiling raised "for headroom" from just sitting
   there unused, and forces the pin to track reality.

Neither check can tell "sanctioned move" from "creep" by itself — no
test can read intent — but together they make BOTH cases visible in
code review: a sanctioned move's PR diff shows the `_BUDGETS` row moving
up next to the method bodies that justify it (reviewable, exactly the
recipe's own re-pin-in-the-same-commit rule, §6); creep shows up as a
ceiling breach with no corresponding row edit in the diff, or a row
edited with no move to justify it. This is the same model the screen
ratchet already uses successfully — Task 1 did not need a stricter or
different mechanism, only to apply the existing one to a new file set.

### Re-pin-at-move flow (identical to §6, restated for controllers)

1. Land the sanctioned move (state PR, controller PR, or cleanup PR per
   §1's series).
2. Re-measure the affected controller file(s):
   `len(path.read_text(encoding="utf-8").splitlines())` — the exact
   expression `test_library_modules_size_ratchet.py`'s own `_measure`
   uses.
3. Set the `_BUDGETS` row to that exact number, in the SAME commit,
   with a one-line dated comment (mirroring the screen ratchet's comment
   trail immediately above its own `_BUDGETS`) — never deferred to a
   follow-up, per §6's rebase-then-measure rule, which applies here
   unchanged.

### Why line count only — no method-count column, unlike the screen ratchet

The screen ratchet tracks both line count and method count because a
single class (`ChatScreen`/`LibraryScreen`) filling its whole file can be
made shorter by *compressing* bodies without actually reducing
responsibility — line count alone cannot catch that on a one-class file.
Controller files under `Library_Modules/` are not shaped that way: the
byte-for-byte canon's constructor-dependency-binding pattern (§1)
deliberately produces small immutable helper classes alongside the
primary controller/coordinator in the SAME file — Protocol ports,
request "fences," result "receipts," outcome snapshots (e.g.
`CaptureRequestFence`/`CaptureArchiveReceipt` in `library_collections_
capture_controller.py`; the `*Port` protocols in `library_notes_sync_
controller.py`). There is also no reliable filename→class-name
convention the way the screen ratchet has: `library_skill_import_
controller.py`'s primary class is `LibrarySkillImportCoordinator`, not
`...Controller`. Picking "the" dominant class per file would therefore
need either a hand-maintained override table (reintroducing the
non-self-defending problem this design otherwise avoids) or summing
methods across every class in the file (which would count the
helper-class proliferation the canon itself encourages as if it were
controller-responsibility growth — punishing a pattern the recipe
recommends). File line count has neither problem and is the exact axis
the wave's own design tension is stated in, so it is the only metric
this guard tracks. A future controller file that happens to be a single
dominant class with no helper types could add a method-count column for
that row specifically without disturbing this reasoning for the rest —
none of the twelve rows pinned at this task's landing qualify.

### Measured rows at landing (task-31203 AC#4, 2026-09-03)

All twelve current `*_controller.py` files under `UI/Library_Modules/`,
pinned at their exact measured line count (zero slack against the
50-line anti-slack tolerance):

| Controller file | Lines |
|---|---|
| `library_collections_capture_controller.py` | 699 |
| `library_collections_controller.py` | 1,689 |
| `library_conversation_reader_controller.py` | 943 |
| `library_conversations_controller.py` | 1,738 |
| `library_export_controller.py` | 1,307 |
| `library_media_browse_controller.py` | 371 |
| `library_media_trash_browse_controller.py` | 319 |
| `library_note_import_controller.py` | 587 |
| `library_notes_sync_controller.py` | 2,023 |
| `library_prompt_browse_controller.py` | 281 |
| `library_skill_import_controller.py` | 760 |
| `library_skills_browse_controller.py` | 413 |

Scope note: this glob covers `*_controller.py` only, per the wave-3
plan's own framing of the gap (the wave-2 review named controller files
by example). `Library_Modules/`'s state files (`library_*_state.py`) and
other support modules (`canvas_sync.py`, `screen_helpers.py`, etc.) are
out of scope for this task — a candidate for a follow-up if the same
ungoverned-growth pattern shows up there, but not asserted here since
AC#4 scoped the review's own concern to controllers by name.

### Mutation evidence (both directions, plus the self-defending property)

All four fired correctly, verified interactively before the real
`_BUDGETS` values were committed (see task-1-report.md in this wave's
SDD ledger for full command output):

1. **Unlisted existing file** (a row deleted from `_BUDGETS` for a file
   that still exists on disk): `test_every_controller_file_has_a_budget_
   row` failed, naming exactly that one path; the file's own ceiling/
   slack parametrizations disappeared with it (25 → 22 passed, 1
   failed) since pytest parametrizes over `sorted(_BUDGETS)`.
2. **Genuinely new file** (a throwaway `_mutation_test_scratch_
   controller.py` dropped into `Library_Modules/`, matching the glob):
   same test failed, naming the new file; all 24 real rows' tests still
   passed unaffected. This is the property the screen ratchet's own
   hand-kept dict does not have.
3. **Growth trip**: one row's budget lowered 13 lines below its real
   measurement → `test_controller_does_not_grow_past_its_budget` failed
   for that row only, reporting "+13" and the guidance block; all other
   rows unaffected.
4. **Anti-slack trip**: one row's budget raised 51 lines above its real
   measurement (one over the 50-line tolerance) → `test_budget_is_not_
   left_slack_after_a_move` failed for that row only, reporting the
   exact slack and the fix ("Set it to 281"); at exactly 50 lines over
   (the tolerance boundary) the same check passes, confirmed separately.

Every mutation was reverted immediately after capturing its failure
output; the file's final committed state is a clean, zero-diff
round-trip back to the real measured values — `test_library_modules_
size_ratchet.py` itself was never left in a mutated state between
commits.
## 18. The search+RAG series, as landed — the third rehearsal, and the first combined-subsystem series

Search+RAG (wave-3 task-31203) is the third subsystem series to run this
recipe, and the first that combines two subsystems the spec originally
named separately (search, RAG) into ONE series -- forced by wave-2 Task 8's
entanglement finding (§16: 57.1% of the search cluster cross-calls
RAG-named methods; the top search bar's submit path *is* the RAG query
entry point). Its series (wave-3 Tasks 2-4: state PR, controller PR,
cleanup PR) is complete. This section records what actually landed and
what the rehearsal added to or reconfirmed about the recipe above.

### Cluster derivation

`ast`-walked `LibraryScreen` for method names containing `"search"` (24 raw
matches) or `"rag"` (39 raw matches), unioned minus the 3-name overlap
(`_apply_library_rag_search_outcome`, `_execute_library_rag_search`,
`_refresh_search_rag_panel_state_widgets`): **60 candidates**. Reading
every body (not the substring) finds **3 Prompts-owned** (the unrelated
Prompts search-box debounce trio) and **7 Media-owned** (Media's own
content/trash search boxes), leaving **50 genuinely combined-cluster
candidates** -- exactly matching wave-2 Task 8's own search-side count (14)
plus the unchanged 39-strong RAG-named set, re-derived fresh rather than
carried over (recipe §6).

### Single vs. split state/controller -- confirmed independently, twice

Task 2's field-level census found all 20 state fields consumed inside one
lock-serialized call graph rooted at
`_refresh_search_rag_panel_state_widgets`/`_library_rag_panel_state`. Task
3 re-verified the SAME conclusion independently, at the METHOD level, with
an `ast` call-graph walk of all 50 candidates: the "search"- and
"rag"-prefixed naming families call directly into each other throughout
(`handle_library_search_submitted`/`rerun_library_search_from_history`/
`submit_library_rag_query`/`run_library_rag_query` all call
`_start_library_rag_query` directly; that method calls
`_record_library_search_history` as an ordinary step). **Decision: ONE
combined `LibraryRagSearchState`/`LibraryRagSearchController`**, confirmed
by two independent derivations rather than either alone -- exactly the
combined-series contingency's own premise, borne out in practice rather
than merely asserted.

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 2 | State | 20 fields (19 `_library_rag_*` + 1 `_library_search_history`) → `LibraryRagSearchState` (0 methods; a programmatic two-prefix property-shim loop, the SAME shape the conversations exemplar's own plural-prefix split established -- `SEARCH_PREFIXED_STATE_FIELDS`, a frozenset with one name, is the single authoritative home for the one exception, applying the conversations exemplar's own task-8 fix-round lesson from the start instead of re-discovering it) | 43977 → 43923 lines, 1316 methods (unchanged) |
| 3 | Controller | 42 methods → `LibraryRagSearchController` (14 `@on` + 3 `action_*` + 25 plain; of 50 candidates, 8 excluded: 3 `@work` framework-decorator hazard, 1 module-globals-coupling exclusion found by running the battery -- not the static census -- and reverted mid-task after it broke a real `get_cli_setting` monkeypatch fixture, 4 instance-attribute-monkeypatch test-bypass) | 43923 → 43009 lines, 1316 methods (unchanged: pure move, 42 `FunctionDef`s out, 42 one-line delegators in) |
| 4 | Cleanup | Shim block (20 properties across 2 prefixes) deleted; every remaining screen-side field reference retargeted to `self._rag_search_state.<field>` (66 literal occurrences across 11 screen-resident methods via one mechanical regex pass, AST-reverified to zero remaining live consumers -- an initial count of "35 occurrences across 9 methods" undercounted and was corrected in the fix round below); a wider census also flagged 1 cross-module site outside both named scopes (`canvas_sync.py`'s `_sync_library_canvas`, writing `screen._library_rag_answer_render_key` directly) whose ONLY callers forward the CONTROLLER as `screen` -- retargeting it broke a real test (caught by this task's own sweep) and was reverted; `canvas_sync.py` needed no change (see the dedicated finding below); 12 of the 42 screen delegators deleted (repo-wide census, all three test roots plus the whole `tldw_chatbook/` tree: zero references anywhere outside their own one-line body); 5 dead imports removed (1 newly dead from the delegator prune, 3 already dead since Task 3's move but left for this cleanup PR per the export/collections split, 1 whose only screen-side consumer was the deleted shim). **Fix round 1**: 9 more cluster-caused dead imports pruned from the same `Widgets.Library` import block (`library_rag_answer_children`, `library_rag_history_children`, `library_rag_query_quiet_text`, `library_rag_query_shows_full_recovery`, `library_rag_query_status_children`, `library_rag_results_body_children`, `library_rag_scope_recovery_children`, `results_heading_text`, `scope_toggle_label`), each verified single-occurrence before deletion; the neighbour `library_rag_scope_shows_recovery` stayed live; a one-line guard comment added at `canvas_sync.py:467` documenting why the flat write there is deliberate | 43009 → 42949 lines, 1304 methods (12 fewer `FunctionDef`s — exactly the 12 pruned delegators) → **42940 lines, 1304 methods** (fix round 1, comment-only otherwise) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`43977/1316 → 43923/1316 → 43009/1316 → 42949/1304 → 42940/1304` (final,
fix round 1).

**Controller-file governance pin** (task-31203 AC#4, §17):
`library_rag_search_controller.py` was born-governed the moment it existed
(Task 3), pinned at its exact measured line count: `1857 → 1890` (Task 3
fix round 1, two false-caller-count corrections in the module docstring)
`→ 1895` (this cleanup task, the ruled moved-body-docstring correction
below -- comment-only growth both times, zero method bodies touched
either time) `→ 1897` (Task 5, wave close: a stale present-tense claim in
the same module's docstring -- "`LibraryScreen` carries [the shim]" --
corrected to past tense, +2 lines, same-commit re-pin; §18 lesson 8
below).

### Delegator census -- 30 KEEP, 12 PRUNED (~29%)

Of the 42 moved names: **14 `@on` handlers and 3 `action_*` handlers KEEP
unconditionally** per the recipe's own transform whitelist (Textual
dispatches both by CSS-selector message routing and by string-keyed action
lookup, neither of which a literal grep can see). Of the remaining 25
plain methods, **13 have a genuine external caller** beyond their own
delegator body: a screen-resident method calls it (4 of the 8 Task-3
exclusions call back into a handful of these -- `_execute_library_rag_
answer` calls `_apply_library_rag_answer`; `_execute_library_rag_search`
calls `_apply_library_rag_search_outcome`; `_refresh_search_rag_panel_
state_widgets` calls all four `_refresh_library_rag_*_widgets` plus
`_library_rag_scope_summary`; `_mirror_library_rag_scope_recovery` calls
`_apply_library_rag_scope_recovery_block` -- plus `_reconcile_library_
entry_state`, a screen-resident method that was NEVER one of the 50
candidates, calling `_sync_library_rag_scope_toggle_and_run_gate_widgets`),
a test that calls the screen delegator directly (`_apply_library_rag_
answer`, `_apply_library_rag_search_outcome`, `_library_rag_answer_chat_
kwargs`, `_start_library_rag_query`), or an instance-attribute monkeypatch
relying on the delegator's SLOT existing at all (`_focused_library_rag_
result_card_index`, `_sync_library_rag_scope_toggle_and_run_gate_widgets`)
--
and **12 have ZERO references anywhere outside their own one-line body**,
confirmed by a repo-wide census across `tldw_chatbook/`, `Tests/UI`,
`Tests/Library`, and `Tests/Live` for each of the 25 individually (not a
sample): `_focus_library_search_input`, `_open_library_rag_result_by_
index`, `_persist_library_search_history`, `_record_library_search_
history`, `_reset_library_rag_answer_state`, `_reset_library_rag_in_
flight_status`, `_reset_library_rag_retrieval_state`, `_reveal_library_
rag_results`, `_select_library_rag_result_by_index`, `_stage_library_rag_
result_in_console`, `_start_library_rag_answer`, `_use_library_rag_
result_in_console`. Every one of these 12 is still called internally,
controller-to-controller, by its own sibling movers -- the exact "no
screen-resident sibling left behind to call it back" shape the collections
series' own 14-of-64 prune (§15) first identified, at a smaller cluster
scale here (12 of 42, ~29%, between export's 1-of-22 (~5%) and
collections' 14-of-64 (~22%)/conversations' 18-of-61 (~30%)).

### A genuinely new finding: a shared dispatcher's `self`-forwarding shape makes a flat name LOOK stale when it is not — retargeting it is the bug

A repo-wide census of the 20 flat field names across the WHOLE
`tldw_chatbook/` tree (not just `library_screen.py`, per the collections
series' own "widen the root" lesson, §16 lesson 2) found exactly one hit
outside the screen: `canvas_sync.py`'s `_sync_library_canvas` (the
shared, multi-subsystem canvas-sync dispatcher already flagged twice in
this series — Task 3's own module-globals-coupling paragraph, §3b of its
docstring, documents its bare-`self`-forwarding risk), whose `"search"`
branch writes `screen._library_rag_answer_render_key = None` directly.

**The obvious-looking fix is wrong, and this task shipped it once before
catching it with the sweep.** The instinct (this report's own first
draft) is: the screen's shim is being deleted, so retarget this to
`screen._rag_search_state.answer_render_key = None`, exactly like every
other flat-name site in this cleanup. That edit passed every unit-scoped
check and the wiring suite, then failed
`Tests/UI/test_library_canvas_scoped_sync.py::
test_media_choice_and_rag_toggles_are_canvas_scoped` in the narrow
`-k "(search or rag) and library"` sweep — a test neither Task 2 nor
Task 3's own documented failure lists mention, so it demanded
investigation rather than a shrug. Tracing `_sync_library_canvas`'s
ACTUAL callers for `kind == "search"` (`grep -rn
"_sync_library_canvas(" tldw_chatbook/`, not an assumption from the
function's own `screen: "LibraryScreen"` type annotation) finds exactly
two, both `LibraryRagSearchController` methods
(`cycle_library_rag_mode`/`toggle_library_rag_scope_source`), both calling
`_sync_library_canvas(self, "search")` — `self` there is the CONTROLLER,
forwarded AS the `screen` parameter. The controller has no
`_rag_search_state` attribute at all, by design (its own permanent
shim's docstring says so explicitly: "reading/writing through the
injected `rag_search_state_accessor` instead of a direct
`self._rag_search_state` attribute (this class has none)"). The dotted
retarget therefore raised `AttributeError` on every real invocation of
this branch; the FLAT name was already correct, resolving through the
controller's own generated shim (`_library_rag_answer_render_key`,
installed permanently by Task 3, mirroring the screen's now-deleted one)
— exactly the polymorphism that shim exists to preserve. Reverted;
`canvas_sync.py` needed NO change at all for this subsystem's cleanup.

**The lesson is sharper than "grep the whole tree" (§16 lesson 2), which
this finding also reconfirms: for a flat name found OUTSIDE the screen
file, trace its actual callers before retargeting, because the receiving
object at that call site may not be a `LibraryScreen` at all.** A
parameter's static type annotation is not proof of its runtime type when
a shared, multi-subsystem dispatcher is deliberately called with `self`
forwarded from a controller (the conversations controller's own
`_sync_library_conversation_canvas` does the identical forwarding, with
no accommodation needed, precisely because ITS controller's shim ALSO
mirrors the screen's). A future subsystem whose OWN controller does not
yet carry a mirrored flat-name shim (i.e., before its own controller PR
lands) would need a different fix here — this reconfirms why the
byte-for-byte canon has every subsystem's controller carry that
permanent shim from its own Task-2/3 landing, not just as a convenience,
but as the exact mechanism that keeps a shared dispatcher's polymorphic
`self`-forwarding working after the SCREEN's copy is deleted.

### A second finding: name collisions across unrelated subsystems require checking the RECEIVER, not just the string

A repo-wide grep for the flat field name `_library_rag_query` surfaces two
files that have nothing to do with this cluster:
`Tests/UI/test_console_rag_settings_modal.py` and `Tests/UI/test_console_
library_search_modal.py`, both setting `controller._library_rag_query =
lambda: ...` where `controller` is a `ConsoleRetrievalController`
(`tldw_chatbook/UI/Console_Modules/retrieval.py`) -- a completely
unrelated Console feature that happens to use the identical attribute name
as its OWN named-dependency callable for an unrelated "current RAG query"
concept. Neither file was touched; both were confirmed unrelated by
reading the receiver's class, not by the string match alone. **A field-name
census for one subsystem's cleanup must confirm the receiver's TYPE (or at
minimum, the surrounding class/module) before assuming a hit belongs to
the subsystem being cleaned up** -- a bare grep on a common-enough name
(here, "the current query", a concept multiple unrelated subsystems each
have their own field for) will over-match.

### The ruled moved-docstring correction (Task-3-deferred, landed here)

Task 3's own review found `_sync_library_rag_scope_toggle_and_run_gate_
widgets`'s moved-body docstring carrying a false caller claim ("Called
synchronously from `_apply_local_source_snapshot`'s in-place branch") --
but that text was BYTE-FOR-BYTE ORIGINAL (present on the screen before
Task 3's own move), so fixing it inside the controller-PR would have
violated the byte-for-byte canon on a body the move PR is not allowed to
edit. Task 3's own ruling (progress.md) deferred the fix to this cleanup
PR, per the wave-2 `_apply_library_row_toggle` precedent (a moved-body
docstring correction is squarely cleanup-PR-legal, mirroring how a
cleanup PR is already allowed to retarget test attribute paths). Fixed
here: the docstring now names the actual caller,
`_reconcile_library_entry_state` (screen-resident, never a cluster
candidate), and clarifies that the call is not literally synchronous with
`_apply_local_source_snapshot`'s own stack frame -- it is scheduled via
`call_later` off every snapshot-generation bump that method (and its
siblings) triggers, so the "fires off the UI thread on every ingest
done-count growth" framing survives, just attributed to the right
intermediate caller. Matches the module docstring's own already-corrected
paragraph (Task 3 fix round 1) rather than introducing a second, possibly
divergent, correction.

### Sweep evidence

`-k "(search or rag) and library"` across `Tests/UI`+`Tests/Library`+
`Tests/Live` (single-process, matching Task 2/3's own per-task check):
**12 failed, 792 passed, 3 skipped** on the corrected tree -- every one
of the 12 matches an already-documented name (10 from Task 2's own
pre-existing list, 2 from Task 3's own "confirmed pre-existing/flaky"
bucket), zero new failures. This is the RE-RUN after the `canvas_sync.py`
near-regression above (found by the FIRST run of this same command) was
reverted -- the sweep is what caught it, exactly the discipline §7 exists
for.

The full sequential xdist paired-baseline sweep (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`, branch then a `git stash -u`
pristine baseline of the pre-task tree, per recipe §7's own "concurrent
runs amplify flakiness" lesson): **350 failed/3931 passed (branch,
1314.03s) vs 349 failed/3932 passed (baseline, 1340.34s)**, both inside
the documented ~330–355 backdrop. 345 shared, 4 baseline-unique (one of
them itself a search/RAG test, failing only on the baseline -- noise in
the opposite direction), 5 branch-unique -- 3 confirmed xdist noise on a
combined single-process re-run, 2 confirmed genuinely pre-existing via a
SECOND `git stash -u` to the same pristine tree in the same
true-isolation combination (added to this section's own list below).
Full per-test detail in this task's own report
(`.superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/
task-4-report.md`).

### Lessons

1. **The combined-series contingency (wave-2's pre-committed fallback for
   an entanglement-gate BLOCK) works cleanly across all three PR types,
   not just the controller PR the entanglement was originally measured
   at.** Every recipe mechanism designed for a SINGLE subsystem -- the
   field-ownership script (§2), the monkeypatch-routing doctrine (§3), the
   delegator census, the pin-lowering flow (§6) -- applied to the combined
   20-field/50-candidate cluster with no adaptation needed beyond widening
   the candidate set upfront (union two name substrings instead of one).
   Nothing about "two subsystems, one series" required a new mechanism;
   it required deciding ONCE, early (Task 2's field census, reconfirmed
   independently by Task 3's method census), that no split seam exists,
   and then running the ordinary three-PR series against the combined
   set.
2. **A flat name found outside the screen file needs its caller traced,
   not just retargeted -- a shared dispatcher's `self`-forwarding shape
   can make an already-correct flat name LOOK stale.** See the dedicated
   finding above: this task's own first-draft retarget of `canvas_sync.
   py`'s `_sync_library_canvas` broke a real test, caught by the sweep,
   because the "screen" parameter at that call site is actually the
   CONTROLLER (forwarded via `self`), which relies on its OWN permanent
   mirrored shim to resolve the flat name -- a dotted retarget assumed a
   receiver type the annotation implied but the actual call graph
   contradicted. Widening the census to the whole `tldw_chatbook/` tree
   (the collections series' own "grep the whole `Tests/` tree" lesson,
   §16 lesson 2, extended to production code) is necessary to FIND a site
   like this, but not sufficient to know what to do with it -- `grep -rn
   "<dispatcher-name>(" tldw_chatbook/` for every hit's actual callers,
   before editing, is the additional step this finding adds.
3. **A flat field name is not a unique key across the whole codebase --
   confirm the receiver before treating a grep hit as this subsystem's
   own.** See the name-collision finding above (`_library_rag_query` on
   an unrelated Console controller). Cost here was small (two files read
   and correctly excluded), but a less careful census could have
   "retargeted" an unrelated module's own field by string-matching alone.
4. **A "this method reads all N fields" claim needs a per-field grep
   against the actual body, not an impression of what the method is
   "basically doing."** Task 2's own state-shape section originally
   claimed `_library_rag_panel_state` reads "all 20 fields in one call"
   via a nonexistent "continuation" -- the method is a single `return
   LibraryRagPanelState.from_values(...)` statement with no continuation
   at all. Review's fix round re-derived the claim from a mechanical
   per-field grep of the method's actual body: it reads exactly 14 of 20
   directly, with the other 6 traced to their real, different consumers
   (two staleness guards, a render-skip cache, two lock primitives, one
   change-gate cache) instead of being asserted read there too. The
   one-object DECISION did not change -- all 20 fields still consume
   inside one lock-serialized call graph -- only its supporting evidence
   did. Generalizes wave-2's own "a 'verbatim' claim needs the same
   evidence discipline as a test-passing claim" lesson (§16 lesson 1) to
   fan-out claims about what a single method reads: state the count from
   a grep, not an impression, and trace every field NOT in that count to
   where it is actually used.
5. **A moved-body docstring's false claim can only be corrected in the
   CLEANUP PR, never the move PR itself.** Task 3's own review found
   `_sync_library_rag_scope_toggle_and_run_gate_widgets`'s moved-body
   docstring carrying a false caller claim, byte-for-byte ORIGINAL text
   already present on the screen before the move -- fixing it inside the
   controller PR would have violated the byte-for-byte canon (§1) on a
   body that PR is not allowed to edit. Ruled (progress.md) to defer the
   correction to the cleanup PR, mirroring an identical wave-2 precedent
   for the same shape; landed in Task 4 (see "The ruled moved-docstring
   correction" above). The byte-for-byte canon's "moved bodies are never
   edited" applies to a body's docstring exactly as it applies to its
   code -- a review finding a stale docstring claim inside a moved body is
   not licence to fix it on the spot; it is a cleanup-PR-scoped finding,
   ruled and tracked until the PR type that IS allowed to touch it lands.
6. **The no-red-ships precedent (§3) held under a second, independent
   live test.** Task 3's own path-census test
   (`test_library_screen_call_sites_never_pass_scope_kwarg`) went red the
   instant the controller PR moved its target call site off the screen --
   a hardcoded-file-path census, not a monkeypatch-routing bypass, so §3's
   exception applies: it cannot wait for the cleanup PR the way an
   ordinary test-bypass shape can, because it is red at the very commit
   boundary that moves the code. Retargeted in the SAME fix round that
   landed the controller PR (assertions preserved, only the census path
   changed), not deferred to cleanup -- confirming §3's rule holds for a
   second, independently-discovered instance of the same shape (its first
   instance is documented in §3 itself).
7. **A brand-new controller file born mid-wave was caught, pinned, and
   re-pinned entirely by the self-defending glob mechanism (§17), with
   zero manual row-adding needed.** `library_rag_search_controller.py` did
   not exist before Task 3; the moment it did,
   `test_every_controller_file_has_a_budget_row`'s glob
   (`UI/Library_Modules/*_controller.py`) found it unlisted and failed
   loudly, exactly as §17 designed -- forcing a `_BUDGETS` row at its
   exact measured line count (1857) in the SAME commit that created the
   file. It then grew twice more (Task 3's own fix round, Task 4's ruled
   docstring fix), each growth re-pinned in the SAME commit as the change
   that caused it (1857 -> 1890 -> 1895), per §17's re-pin-at-move flow
   (identical to the screen ratchet's §6 rule). This is the governance
   mechanism's first live exercise since task-31203 AC#4 landed it, and it
   worked exactly as designed on the first subsystem to need it.
8. **A battery run captured BEFORE a later edit does not verify that
   edit -- re-measure fresh after every edit, not just after the first
   one in a session.** Task 5 (wave close) ran the full ratchet battery
   green, THEN fixed a stale docstring claim in `library_rag_search_
   controller.py` (this wave-3 file's OWN module docstring, not a moved
   body -- unlike lesson 5's deferred-to-cleanup-PR mechanism, no
   byte-for-byte canon deferral applied here) as a separate, later step
   -- the rewrite needed 2 more lines to
   read naturally, growing the file from 1895 to 1897 without re-running
   the ratchet afterward. The drift sat unnoticed until a later,
   unrelated fresh-measurement pass (`_measure()` called directly, not
   through pytest) caught it; `test_controller_does_not_grow_past_its_
   budget` confirmed red on re-run. Re-pinned 1895 -> 1897 same-commit,
   battery re-confirmed green. The general lesson: "the battery was
   green" is a claim about the tree AT THE TIME it ran, not a durable
   property of the tree -- any edit after the last green run, however
   small or comment-only it looks, needs its own fresh verification,
   exactly the discipline lessons 4-5 above ask of evidence claims in
   general, now caught turning inward on this task's own work rather
   than a prior task's.

## 19. The skills series, as landed — the fourth rehearsal, and the first three-way state-field-prefix subsystem

Skills (wave-4, `.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills`)
is the fourth subsystem series to run this recipe (Tasks 1-3: state PR,
controller PR, cleanup PR). It is the largest single-controller move to
date (86 methods, ~2,000 lines) and the first subsystem whose state fields
split across THREE shim prefixes rather than one or two. This section
records what actually landed and what the rehearsal added to or
reconfirmed about the recipe above.

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 1 | State | 36 fields (26 `_library_skill_*` singular default + 9 `_library_skills_*` plural + 1 bare `_selected_skill_name`) → `LibrarySkillsState`, via `skill_state_shim_attr()` -- a single-source three-way prefix resolver, the same shape the conversations/search+RAG exemplars' own plural-prefix split established, extended one prefix further (0 methods) | 43225 → 43179 lines, 1311 methods (unchanged) |
| 2 | Controller | 86 of 127 "skill"-named methods → `LibrarySkillsController` (30 `@on` + 1 staticmethod + 55 plain; 41 exclusions: 6 merely-delegate-to-existing-controller properties, 27 unbound-fake-self, 1 instance-attribute monkeypatch, 1 module-globals coupling, 6 bare-self-as-identity-argument hazard -- 1 found by static analysis (Form A), 5 found by the verification battery after two draft rounds each broke real tests (Form B: 4 names, Form C: 1 name) -- see recipe §3's sixth bypass shape) | 43179 → 41247 lines, 1311 methods (unchanged: pure move, 86 `FunctionDef`s out, 86 one-line delegators in). Controller born-governed at 3181, fix round 1 (Form B revert) → 3113, fix round 2 (Form C revert) → 3099, post-landing-review fix (the SEVENTH bare-self-hazard-adjacent shape: an unbound-attribute escape via `getattr(self, "focused", None)` with no corresponding property) → 3131 |
| 3 | Cleanup | Shim block (36 properties across 3 prefixes) deleted; every remaining screen-side flat reference retargeted to `self._skills_state.<field>` (130 pre-existing occurrences: 121 literal `self.<flat_name>` attribute accesses + 5 dotted-vs-flat dispatch-dict string values across the `__init__` entangled-field lines and the two shared reader-preference dispatcher methods (`_replace_library_reader_preference`/`_persist_library_reader_preference`) + the skills choice-strip helper's own dispatch dict + 2 prose-comment corrections + 1 `getattr(self, "_library_skills_view", "list")` call needing a RECEIVER fix, not just a string swap, since `getattr` does not do dotted traversal); 269 test-side retargets across 9 files spanning THREE roots (`Tests/UI`: 8 files, 221 retargets; `Tests/Skills`: 2 files, 48 retargets -- `Tests/Live` had zero skills-field consumers); 16 of the 86 screen delegators deleted (repo-wide census across `tldw_chatbook/`, `Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/Skills`: zero references anywhere outside their own one-line body and the controller's own internal calls); 28 dead imports removed (1 newly dead from the shim deletion itself -- `skill_state_shim_attr` -- plus 27 left dead by task 2's own move and deliberately deferred to this cleanup PR, per the export/collections/search+RAG series' own Task 3/Task 4 split: 15 names from `Widgets.Library`, 10 from `Library.library_skills_state`, 2 from `.skills_screen`); 2 module-docstring corrections in the controller (an arithmetic error inherited from task 1's own report -- "three `@property`/`@x.setter` pairs" for a 6-match gap that is actually six pairs -- and a now-false "`LibraryScreen` keeps one-line delegators under every one of these" claim, corrected to name the 70-of-86 remaining count) | 41247 → 41155 lines, 1311 → 1295 methods (16 fewer `FunctionDef`s -- exactly the 16 pruned delegators). Controller: 3131 → 3140 (comment-only growth: the two module-docstring corrections) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`43225/1311 → 43179/1311 → 41247/1311 → 41155/1295` (final).

**Controller-file governance pin** (§17): `library_skills_controller.py` was
born-governed the moment it existed (Task 2): `3181 → 3113 → 3099 → 3131`
(Task 2 and its two fix rounds) `→ 3140` (this cleanup task, comment-only).

### Dynamic-dispatch census — zero hazard shapes found, one dotted-string update needed

Re-derived Task 2's own dynamic-dispatch findings before deleting the shim,
plus the collections/search+RAG series' own `dict.get(...)` → variable →
`setattr`/`getattr` two-step guidance: **zero** new instances of either
hazardous shape found touching any of the 36 Skills state fields. The ONE
screen-side dynamic-dispatch site touching Skills is the SAME shared shell
dispatcher every prior subsystem's cleanup has updated for its own fields --
`_replace_library_reader_preference`/`_persist_library_reader_preference`'s
7-destination `{destination: attribute_name}` dicts, plus the skills-list
choice-strip helper's own `{selector_string: visibility_attr}`/
`{visibility_attr: canvas_kind}` dict pair (mirroring the export series'
own `"export": "_export_state.quality_choices_visible"` entry exactly) --
all already fully generic dotted-vs-flat passthroughs via
`operator.attrgetter`/`_assign_library_reader_preferences_attribute`, so
none needed a logic change, only the string value(s). The SAME fixture-side
shape recurred in `Tests/UI/test_library_adaptive_reader_closeout.py`'s own
`DESTINATION_CONTRACT` dict (the fourth subsystem in a row to need this
exact fixture update -- collections, conversations, and now skills each
added one dotted entry to this same table).

**A genuinely new shape this series adds to the census, found by re-running
the recipe's own "grep the whole moved-body source for `getattr(self,
"<literal-string>"` (not just `self.<attr>` accesses)" check (§3's sixth
bypass shape, added by this wave's own Task 2 post-landing review) against
the SCREEN-RESIDENT code this time, not just the moved controller body**:
`_library_list_canvas_showing_list`'s own
`getattr(self, "_library_skills_view", "list")` call, one of several
sibling destination checks in the same method (`_library_notes_view`,
`_library_prompts_view` use the identical shape and remain flat -- those
two subsystems have not yet had their own state-PR series). Retargeting
the STRING alone (`"_library_skills_view"` → `"_skills_state.view"`) would
have been silently wrong: `getattr` performs a single attribute lookup, not
dotted-path traversal, so `getattr(self, "_skills_state.view", "list")`
would have permanently returned the literal default `"list"` -- caught by
re-reading the transformed line before running any test, not by a test
failure. Fixed by changing the RECEIVER too:
`getattr(self._skills_state, "view", "list")`. **The lesson: a mechanical
string-literal-only find/replace pass over `getattr`/`setattr` calls is not
safe by construction -- every `getattr(self, "<flat_name>", default)` site
found by a field-retarget census needs its receiver changed alongside the
string, not just the string, because `getattr` has no notion of a dotted
path.**

### Screen-side field retarget

**130 pre-existing flat-name occurrences**, across the `__init__`
entangled-field lines (the reader-preferences tuple-unpack, the computed
`reader_layout`/`reader_persistence_locks` lines, and -- a shape new to
this series -- TWO more fields, `editor_mode` and `reader_mode`, whose
original `__init__` lines also had to stay untouched per Task 1's own
"forced-early-construction-point" finding, not just the usual reader
trio), the two shared reader-preference dispatcher methods, the skills
choice-strip helper, and every remaining screen-resident (excluded, still
full-bodied) method's own field reads/writes -- a single per-field regex
pass (36-field mapping, `\bself\.<flat_name>\b` → `self._skills_state.
<field>`, plus a second pass for quoted dict-string values) rewrote 126 of
the 130 mechanically; 2 needed a prose-comment reword (not a code change);
1 needed the receiver fix described above; re-verified afterward with a
zero-result grep for every one of the 36 flat field names over the whole
file (the 1 remaining hit is this task's own explanatory comment, naming
the deleted shim's old field names in past tense, mirroring the collections/
search+RAG series' own retained-history comments at the same site).

### Test retarget — three roots, one genuinely new fixture-restructuring scale

Repo-wide census (`Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/Skills`
-- all four named test roots, not just the two the export series' own
lesson widened to) found flat-name consumers in 9 files across two roots
(`Tests/Live` had zero): **221 retargets across 8 `Tests/UI` files, 48
across 2 `Tests/Skills` files, 269 total**.

**The scale of unbound-fake-self fixture restructuring is new**: 27 of the
86 movers are unbound-fake-self exclusions (task 2's own census, roughly
triple export's prior 9-of-51 record), and a large fraction of those are
tested via `SimpleNamespace(...)` fakes carrying FLAT skills kwargs
directly (`_library_skills_sort="name"`, `_selected_skill_name=""`, etc.)
that a body now reading `self._skills_state.<field>` can no longer satisfy.
28 separate `SimpleNamespace(...)` call sites across
`Tests/UI/test_library_skills_canvas.py` (25),
`Tests/UI/test_library_canvas_scoped_sync.py` (1), and
`Tests/Skills/test_skills_import.py` (2) needed the SAME "flat kwargs →
nested `_skills_state=SimpleNamespace(...)`" restructuring the
conversations/export exemplars' own cleanup PRs first established (recipe
§11) -- written as a small line-oriented script (collect every
`<flat_name>=<value>,` kwarg line inside a `SimpleNamespace(` call block,
regardless of whether the matched kwargs are contiguous with each other,
and re-emit them as one `_skills_state=SimpleNamespace(<field>=<value>,
...)` kwarg at the position of the first match) rather than done by hand,
appropriate for this cluster's scale per the collections series' own
"write the extraction and verification as scripts" lesson, generalized
here from body-extraction to fixture-restructuring. Every non-state-field
kwarg (wiring accessors, mocked-out sibling methods) was left untouched by
construction, since the script only matches the 36 known field names.
**Zero assertion VALUE changes** -- every one of the 269 test-side edits is
a receiver-path rewrite only (`screen._library_skill_<field>` /
`fake._library_skill_<field>` → `<receiver>._skills_state.<field>`,
`_library_skill_<field>=<value>` kwarg → nested under `_skills_state=
SimpleNamespace(...)`), confirmed by running the full affected-file battery
before and after and diffing the pass/fail set (below).

### Delegator census — 70 KEEP, 16 PRUNED (~19%)

Of the 86 moved names: **30 `@on` handlers KEEP unconditionally** per the
recipe's own transform whitelist. Of the remaining 56 (55 plain + 1
`@staticmethod`), a repo-wide grep for each name across `tldw_chatbook/`
and all four test roots found **40 with a genuine external caller** (an
excluded, still-screen-resident method calling `self.<name>()` -- e.g.
`handle_library_skills_trust_action` calling
`_begin_library_skill_trust_setup`/`_unlock_library_skill_trust`/
`_refresh_library_skills_trust_posture`/`_open_first_blocked_skill`, or
`_reset_library_skill_editor_state` calling
`_invalidate_library_skill_detail_generation` -- or a test that calls/
patches the screen delegator directly) and **16 with none**, confirmed by
a repo-wide grep restricted to `*.py` outside the controller module, the
screen's own delegator body, and this task's wiring-test list, before
deletion: `_apply_library_skill_detail`, `_apply_library_skill_detail_
failure`, `_bootstrap_library_skill_trust`, `_build_library_skill_tool_
catalog`, `_claim_library_skill_detail_generation`, `_do_library_skill_
trust_reset`, `_focus_library_skills_page_control`, `_library_skill_text_
fields_match_state`, `_load_library_skill_script_grant`, `_mark_library_
skill_dirty`, `_read_library_skill_editor_fields`, `_read_library_skill_
live_name`, `_revoke_library_skill_script_grant`, `_setup_library_skill_
trust`, `_sync_library_skill_description_hint`, `_update_library_skill_
toggle_buttons`.

**A methodology trap this task's own first census draft fell into and
corrected before acting on it**: a naive `grep`-based census using a
negative lookbehind to exclude any match preceded by a word character or a
dot (intended to catch bare-word occurrences like kwarg names or string
literals, not `self.<name>(` call sites) silently EXCLUDED the single most
important signal -- `self.<name>(` call sites -- from its own "other
callers" count, because `self.` ends in a dot. This produced a FALSE
"zero external callers" verdict for `_begin_library_skill_trust_setup`
(and would have for others), when in fact `handle_library_skills_trust_
action` -- a screen-resident, still-excluded method -- calls
`self._begin_library_skill_trust_setup()` directly at its own line 27022.
Caught by manually verifying one suspicious case (a test fixture
overriding the name with a lambda, which only makes sense if SOMETHING
calls it) before trusting the census output, not by a test failure --
the flawed census would have pruned a delegator a real screen-resident
caller still needed, breaking `handle_library_skills_trust_action`'s
"setup"/"resetup" trust-action branch at runtime with no test catching it
(no test drives that exact branch through the real screen delegator; the
excluded method's own coverage mocks the target out). **The lesson: a
delegator census's grep pattern must match `receiver.name(` call sites
explicitly, not accidentally exclude them via an over-eager negative
lookbehind aimed at a different noise source (bare kwarg names/string
literals) -- verify the census script's OWN pattern against one hand-
picked, already-known-to-be-called name before trusting its "zero
callers" verdicts for the rest.**

This prune fraction (16 of 86, ~19%) sits below every prior series' own
recorded fraction (export's 1-of-22 ~5% < skills' 16-of-86 ~19% <
search+RAG's 12-of-42 ~29% < collections' 14-of-64 ~22% < conversations'
18-of-61 ~30%), consistent with the recipe's own "exclusions are what KEEP
screen delegators alive" inverse-relationship finding (§15 lesson 3): 41
of 127 candidates were excluded from the move here (a comparatively large
exclusion count), and a large share of those excluded, screen-resident
methods keep calling their moved siblings via `self.<name>()`, which is
exactly what keeps a screen delegator's reference count above zero.

### Shim block deletion

The task-1-generated `_library_skill_<field>`/`_library_skills_<field>`/
`_selected_skill_name` property-shim loop (installed at module end,
`dataclasses.fields(LibrarySkillsState)`-driven) was deleted wholesale once
the census above confirmed zero remaining consumers anywhere in
`tldw_chatbook/` or `Tests/` outside `LibrarySkillsController`'s own
PERMANENT generated shim loop (installed by task 2, reading
`self._skills_state_accessor().<field>` -- untouched by this task, per the
task brief's explicit "controller shims STAY" instruction).

### Import verification

**28 dead imports removed**, each verified single-occurrence (import line
only, via AST `Name`-node usage count, not a bare grep) in
`library_screen.py`, then checked against `Tests/Architecture/
test_library_support_layer_surface.py`'s `_SURFACE` dict (the PR-0a
re-export contract) before deletion -- none of the 28 belongs to any of
`_SURFACE`'s 5 listed modules; the 3 SKILLS constants that ARE `_SURFACE`-
pinned (`LIBRARY_SKILL_TEXT_MAX_CHARS`, `LIBRARY_SKILL_DIRTY_VETO_COPY`,
`LIBRARY_SKILL_SAVE_STATUS_COPY`) were individually re-checked and left in
place:

- 1 newly dead as a DIRECT RESULT of this task's own shim-block deletion:
  `skill_state_shim_attr` (the three-way prefix resolver function, still
  imported and used by `LibrarySkillsController`'s own permanent shim loop
  and by the wiring test, just no longer by the screen).
- 27 left dead by task 2's own controller move (each name's only
  screen-side usage lived inside one of the 86 moved method bodies), 26 of
  which were independently confirmed already re-imported and live inside
  `library_skills_controller.py` before removal from the screen: 15 from
  `Widgets.Library` (`SKILL_DISCARD_TOOLTIP_CLEAN`, `SKILL_DISCARD_
  TOOLTIP_DIRTY`, `next_skill_context`, `skill_context_toggle_label`,
  `skill_disable_model_label`, `skill_script_grant_line`, `skill_trust_
  approve_tooltip`, `skill_trust_panel_remediation_copy`, `skill_trust_
  review_enabled`, `skill_trust_review_preview`, `skill_trust_review_
  tooltip`, `skill_trust_state_line`, `skill_trust_unlock_enabled`,
  `skill_trust_unlock_tooltip`, `skill_user_invocable_label`), 9 of the 10
  from `Library.library_skills_state` (`DEFAULT_SKILL_BROWSE_PAGE_SIZE`,
  `MAX_SKILL_BROWSE_PAGE`, `build_skill_editor_state`, `classify_skill_
  save_error`, `compose_skill_markdown`, `reconcile_skill_allowed_tools`,
  `skill_allowed_tools_sequence`, `skill_invocation_copy`, `skill_review_
  identity_line`), 2 from `.skills_screen` (`SkillTrustBootstrapModal`,
  `SkillTrustPassphraseModal`). **The 1 exception**: `SkillEditorState`
  (the 10th `Library.library_skills_state` name) is NOT used in the
  controller at all (`grep -c` returns 0) -- it is live in
  `library_skills_state.py` (the `UI/Library_Modules` state object, via
  the `editor_state: SkillEditorState | None` field annotation) and in
  `Widgets/Library/library_skills_canvas.py` (three signatures), both
  already-independent imports unrelated to the controller's own move.
  Removal from `library_screen.py` was still correct either way (0
  occurrences there beyond the import line); only the blanket "all 27 are
  live in the controller" claim needed correcting to name the one
  exception and its real, different, home.

One name deliberately left alone despite a zero-`_SURFACE` check: `skill_
editor_warning_lines` (`Widgets.Library`) -- confirmed still live via a
non-`_SURFACE`, non-moved screen-resident consumer, individually re-checked
rather than assumed dead by proximity to its 15 pruned `Widgets.Library`
siblings (mirrors the collections series' own `CaptureIdentity`/
`CollectionsCaptureReaderPresentation` "checked individually, not assumed
dead-by-association" precedent).

### Moved-docstring / module-docstring corrections

Two inaccuracies in `library_skills_controller.py`'s own MODULE docstring
(not a moved method body -- freely editable by any task, no byte-for-byte
canon deferral needed, same distinction the search+RAG series' own Task 5
lesson 8 draws) were found and fixed in this cleanup task, both
comment-only:

1. **An arithmetic error inherited unfixed from task 1's own report into
   the controller's own docstring**: "the 6-match gap is three
   `@property`/`@x.setter` pairs" -- task 2's own post-landing review fix
   round (§12c) caught and corrected the IDENTICAL error in its own report
   text ("SIX pairs", not "three"; 2 raw `FunctionDef`s − 1 unique name =
   1 gap per name, 6 names = 6 gap) but never propagated that correction
   into the controller module's own docstring, which still carried the
   stale "three" claim. Fixed here.
2. **A now-false present-tense architecture claim**: "`LibraryScreen`
   keeps one-line delegators under every one of these original names" --
   true when task 2 wrote it, false for 16 of the 86 as of this task's own
   delegator prune. Fixed to name the current 70-of-86 count and point at
   `_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED` for the list, mirroring the
   collections/search+RAG wiring tests' own skip/absence-assertion
   convention.

**A precedent check worth recording**: the search+RAG controller
(`library_rag_search_controller.py`) carries the IDENTICAL "`LibraryScreen`
keeps one-line delegators under every one of these original names" claim
in its own module docstring, still present and still false today (12 of
its own 42 movers were pruned by that series' own Task 4) -- confirmed by
reading the file, not assumed. That series' own cleanup task did not fix
this claim when it pruned those 12 delegators. This task fixed the
IDENTICAL claim for skills rather than leaving it stale to match
precedent, since the task brief's canon-ruling scope explicitly authorizes
architecture-claim corrections and the fix is free (comment-only, zero
risk). Left AS a forward note, not fixed here (out of this task's own
file scope): a future pass through `library_rag_search_controller.py`
should apply the same correction there.

### Wiring test finalization

`Tests/Architecture/test_library_skills_wiring.py`:
`test_state_object_fields_match_the_shim_surface` DELETED (screen shim
gone, mirrors the conversations/export/collections/search+RAG precedent
exactly); `_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (16 names)
added, with `test_screen_delegates_skills_handlers` skipping those names
and instead asserting their genuine ABSENCE from `LibraryScreen`; module
docstring rewritten to describe the finished 3-task series. 8 of the
original 9 tests remain (the shim-surface test's removal is the only count
change) -- all 8 green post-cleanup.

### Size ratchet

Fresh measurement via the ratchet's own `_measure` semantics (`ast`-walked
line count + `LibraryScreen` method count, not `wc -l`): **41155 lines,
1295 methods** -- 1295 = 1311 (task 2's post-move count) − 16 (exactly the
pruned delegator count), confirming the prune was a pure deletion with no
replacement. Lowered in this same commit per recipe §6. Controller
re-measured at **3140 lines** (comment-only growth from the two
module-docstring corrections above), re-pinned same-commit per §17's
re-pin-at-move flow.

### Battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **Wiring suites, all five**: `test_library_skills_wiring.py` (8, post-
  deletion/post-pruning) + `test_library_collections_wiring.py` (4) +
  `test_library_conversations_wiring.py` (6) + `test_library_export_
  wiring.py` (5) + `test_library_search_rag_wiring.py` (8, controller-PR
  count unaffected by this task) — all passed alongside the 3 existing
  characterization files (collections/conversations/export) and the
  support-layer surface suite (8 passed): 51 passed total in that combined
  run.
- **Both size guards, full suite**: 32 passed, 2 failed -- both the
  documented pre-existing `chat_screen.py` ratchet rows (recipe §7's own
  standing list), unrelated to this diff.
- **`-k "skill and library"` sweep** (`Tests/UI`+`Tests/Library`, single
  process, final tree): **10 failed, 272 passed, 22073 deselected** -- all
  10 match Task 1/2's own already-documented pre-existing bucket name-for-
  name (CSS-block/geometry-parity ×5, the command-palette test, `test_
  action_library_skill_back_honors_dirty_guard`, `test_shadow_name_set_
  stays_in_sync_with_real_sources`, `test_skills_route_lands_on_library_
  with_skills_row_selected`); zero new failures; 1 more passed than Task
  2's own 271 (`test_library_skills_manual_items_priority_survives_
  compact_layout_sync` flipped to pass -- Task 1 already characterized
  this exact name as order-dependent xdist-adjacent noise).
- **`Tests/Skills/` full run** (fourth root): **537 passed, 2 failed** --
  EXACT match to Task 1/2's own documented baseline (`test_import_real_
  superpowers_skills_lands_trust_pending`, environment-dependent; `test_
  uninitialized_trust_shows_setup_state_and_bootstrap_enables_approve_
  flow`). Zero new failures.
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library"
  -p no:randomly -q -n 8 --dist worksteal`, branch then a `git stash -u`
  pristine baseline, run SEQUENTIALLY under sustained heavy CONCURRENT
  machine load from several unrelated long-running pytest processes
  already active on this machine for hours -- confirmed via `ps aux`, not
  assumed): **branch 367 failed/3937 passed (1930.5s) vs. baseline 370
  failed/3934 passed (1949.1s)**, both inside the documented ~330-371
  historical backdrop despite the elevated absolute counts this run's
  heavier-than-usual load produced. 361 shared, 9 baseline-unique
  (opposite-direction noise), 6 branch-unique -- 3 passed cleanly on a
  combined single-process re-run (ordinary xdist noise); the other 3 (all
  sharing the SAME "app never finished pushing its initial screen" generic
  Textual-startup-timeout signature, nothing skills-specific) were
  individually investigated: a fresh `git stash -u` to the identical
  pristine tree reproduced 2 of the 3 immediately, and the third settled
  by running it 3x in ISOLATION on each tree (3/3 failures on branch, 3/3
  on the SAME pristine baseline, identical signature both times). **Zero
  unexplained branch-unique failures** -- every one that reproduced at all
  reproduces identically on the pristine pre-task tree under the same
  (heavily loaded) conditions, and none touches Skills code or this task's
  diff.
- **preflight**: `./scripts/preflight.sh` -- all six checks green (no
  diagnostic-inventory drift; this task's diff touches zero
  `logger.warning`/persistent-diagnostic call sites).

### Lessons

1. **A dynamic-dispatch census's `getattr`/`setattr` fix must change the
   RECEIVER, not just the string, whenever the call uses a bare `self`
   receiver with a flat literal name.** See the dedicated finding above:
   `getattr(self, "_library_skills_view", "list")` needed BOTH
   `self` → `self._skills_state` AND the string shortened from the flat
   name to the bare field name -- retargeting only the string produces
   `getattr(self, "_skills_state.view", "list")`, which silently always
   returns the default forever (no exception, no red test by construction,
   the exact shape recipe §3's "unbound-attribute escape" bypass class
   already names for the identical reason). Caught by reading the
   transformed line, not by running anything -- this shape does not
   reliably produce a failing test (per §3's own documented false-negative
   risk for this class), so a mechanical find/replace pass over
   `getattr`/`setattr` calls needs a human (or a second, receiver-aware
   automated pass) to check every hit, not just the census that finds
   them.
2. **A delegator census's own grep pattern is itself a hazard surface --
   verify it against one known-true case before trusting its "zero
   callers" output for the rest.** See the dedicated finding above: an
   over-eager negative lookbehind, added to suppress noise from bare kwarg
   names and string literals, silently also suppressed every `self.
   <name>(` and `<receiver>.<name>(` call-site match -- exactly the
   signal a delegator-prune census exists to find. The fix was cheap once
   noticed (drop the lookbehind, classify hits by which file/context they
   land in instead of filtering them out of the pattern itself), but the
   FIRST version of this census would have silently authorized pruning a
   delegator a real screen-resident caller (`handle_library_skills_trust_
   action`) still needed. Any future subsystem's delegator census should
   sanity-check its own regex against a name already KNOWN to have a
   `self.<name>(` caller (e.g. by grepping the target file by hand first)
   before trusting a "zero other references" verdict for anything.
3. **The "flat kwargs → nested `_skills_state=SimpleNamespace(...)`"
   fixture restructuring scales cleanly to script-driven automation once a
   cluster's unbound-fake-self exclusion count crosses roughly two dozen.**
   Skills' 27 unbound-fake-self exclusions (task 2's own census) produced
   28 separate `SimpleNamespace(...)` call sites needing this exact
   restructuring in this cleanup task -- large enough that doing it by
   hand, one call site at a time, would have been both slow and
   error-prone (kwargs are not always contiguous within a call). A small,
   generically-written line-oriented script (collect matching kwarg lines
   anywhere inside a `SimpleNamespace(` block regardless of contiguity,
   re-emit them as one nested kwarg) handled all 28 correctly on the first
   pass, verified by `ast.parse` before and a full pytest run after --
   generalizes the collections series' own "write the extraction and
   verification as scripts" lesson (§15) from body-extraction to
   fixture-restructuring, and gives a rough scale threshold (a cluster
   with fewer than ~10 such fixtures is probably still faster by hand; one
   with ~28, as here, clearly was not).
4. **A stale architecture claim found in one subsystem's controller
   docstring is worth checking for in EVERY prior subsystem's own
   controller, not just fixing locally and moving on.** The search+RAG
   controller carries the identical "`LibraryScreen` keeps one-line
   delegators under every one of these original names" claim, equally
   false today (12 of its own 42 movers were pruned by that series' own
   cleanup task), and it was never corrected there. This task fixed its
   OWN copy of the claim rather than leaving it to match that precedent,
   and records the search+RAG copy as an open forward note rather than
   silently accepting "a prior series left it stale, so this one can too"
   as license.
5. **The full sequential xdist paired-baseline sweep's absolute failure
   counts move with the MACHINE's concurrent load at run time, not just
   with the code under test -- the paired-baseline methodology is what
   keeps the comparison valid regardless.** This task's own sweep ran
   under sustained heavy concurrent load from several unrelated
   long-running pytest processes already active on the machine for hours
   (confirmed via `ps aux`, not assumed) and posted 367/370 failed
   (branch/baseline) -- noticeably higher than every prior series' own
   recorded run (~330-355) but still inside the documented historical
   backdrop's outer edge. Three of the six branch-unique names needed a
   THIRD round of investigation beyond the usual "combined re-run, then
   isolated re-run": all three shared one generic "app never finished
   pushing its initial screen" Textual-startup-timeout signature (nothing
   skills-specific), and one of them passed once on an early baseline
   check before a repeat 3x-isolated run on each tree settled it as
   equally flaky on both. The lesson generalizes to a specific, quotable
   number: absolute failure counts drift with ambient machine load, but
   the PAIRED comparison (same command, same machine conditions, branch
   vs. a `git stash -u` of the identical pristine tree, sequentially) is
   what makes "zero unexplained branch-unique failures" a claim that
   survives a noisier-than-usual run rather than a claim that only holds
   on a quiet machine.

### Wave-4 close

Wave-4 (`.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills`,
branch `refactor/library-decomp-wave4-skills-ingest`) scoped itself to
skills alone at Task 1 despite the branch name (the plan's own 2026-09-04
measure found skills at 133/38 raw and ingest at 78/20; ingest was deferred
to a future wave rather than rushed alongside skills in the same series).
The skills series (Tasks 1-3, above) is complete. This section is Task 4's
own wave-level pin-trajectory re-derivation, verification battery, and
lessons.

#### Pin trajectory — full wave-4 chain

Re-derived from `git log` (the `_BUDGETS` value in
`Tests/Architecture/test_screen_size_ratchet.py` and
`Tests/Architecture/test_library_modules_size_ratchet.py` at each commit,
not carried over from any report):

| Task | PR | Commit | Screen `_BUDGETS` after | Controller pin after |
|---|---|---|---|---|
| — | (wave-4 start) | `2372ea764` | 43225 / 1311 | — (file does not exist yet) |
| 1 | Skills state (RED — wiring test + state module + characterization pins; screen untouched) | `ef289548a` | 43225 / 1311 (unchanged) | — |
| 1 | Skills state (GREEN, series complete) | `87c318d57` | 43179 / 1311 | — |
| 2 | Skills controller (RED — wiring-test additions only) | `5ecf223d4` | 43179 / 1311 (unchanged) | — |
| 2 | Skills controller (GREEN, born-governed) | `60857a2be` | 41247 / 1311 | 3099 |
| 2 | (blame-ignore, no functional change) | `679a90d1b` | 41247 / 1311 (unchanged) | 3099 (unchanged) |
| 2 | Fix round 1/5 — CRITICAL `focused` fix + recipe §3's sixth bypass shape (series complete) | `bf13b133b` + `f472f7512` | 41247 / 1311 (unchanged) | 3131 |
| 3 | Skills cleanup (GREEN, series complete) | `ed4c29d45` | 41155 / 1295 | 3140 |
| 3 | (blame-ignore, no functional change) | `2a744c434` | 41155 / 1295 (unchanged) | 3140 (unchanged) |
| 3 | Fix round 1/5 — doc-only count corrections | `f42f75d98` | 41155 / 1295 (unchanged) | 3140 (unchanged) |

Full chain, screen: `43225/1311 → 43179/1311 → 41247/1311 → 41155/1295`
(final). Full chain, controller: born `3099` at its very first commit (Task
2's own report documents a `3181 → 3113 → 3099` sequence, but all three
numbers belong to uncommitted working-tree drafts from the SAME session —
neither of the two earlier numbers was ever a landed commit, so `git log`
shows only the final, already-corrected value at the controller's birth
commit) `→ 3131` (Task 2's post-landing review fix round: the CRITICAL
`focused` property) `→ 3140` (Task 3, comment-only: two module-docstring
corrections).

Net wave-4 shrink: 2070 screen lines, 16 methods (the 16 pruned skills
delegators — a pure move is always net-zero methods on the screen; only a
delegator prune changes the method count). Task 4's own fresh `_measure()`
call (both ratchet files' own semantics, not `wc -l`) gives **41155 lines /
1295 methods** for the screen and **3140 lines** for the controller — EXACT
matches to both recorded pins, zero drift, nothing to lower.

#### Verification battery (Task 4, this close)

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, `-p no:randomly` where applicable.

- **Fresh `_measure()` on both ratchet files**: 41155/1295 (screen), 3140
  (controller) — exact match to both recorded pins, zero drift.
- **All five wiring suites + 3 characterization files + support-layer
  surface + both size guards, combined single run**: **83 passed, 2
  failed** — both the documented pre-existing `chat_screen.py` ratchet rows
  (recipe §7's own standing list), unrelated to this diff.
- **Full `Tests/Architecture/` run**: **543 passed, 1 skipped, 16 failed** —
  one FEWER than Task 2's own documented 17 (`test_persistent_diagnostic_
  inventory.py::test_task_15743_exception_types_survive_loguru_forwarding`'s
  sibling assertions now partially SKIP rather than fail, per the file's own
  skip reason: "TASK-15743 pinned commits fdee8a31f/afee9672a are not
  fetchable [...] force-pushed and deleted" — an external git-object
  availability change, unrelated to this diff). All 16 remaining failures
  match the SAME categories Task 1/2 already documented as pre-existing,
  unrelated-file churn (Console realtime/review-selection boundary ×2,
  console wave6 closeout/inventory ×4, default-timeout-session-guard ×1,
  persistent-diagnostic-inventory ×2, chat_screen ratchet ×2, timer-path-
  static-update-inventory ×3, worker-exclusive-group-inventory ×2) — zero
  Library/Skills-scoped failures.
- **`-k "skill and library"` sweep** (`Tests/UI`+`Tests/Library`, single
  process, final tree): **10 failed, 272 passed, 22073 deselected** — EXACT
  match to Task 3's own documented baseline, name-for-name. Zero new
  failures.
- **`Tests/Skills/` full run** (fourth root): **538 passed, 1 failed**
  (`test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_
  approve_flow`) — the OTHER documented pre-existing failure
  (`test_import_real_superpowers_skills_lands_trust_pending`,
  environment-dependent) passed this run, consistent with its own
  "environment-dependent" characterization across all three prior tasks.
  Zero new failures.
- **preflight**: `./scripts/preflight.sh` — all six derived-artifact checks
  pass (CSS bundle, profile-owned-path census, diagnostic inventory,
  backlog task-id sweep — 3188 task files including this close's own 2 new
  follow-up filings, chachanotes table allowlist, index plan pins).

#### Full sequential xdist paired-baseline sweep — whole-wave span

Branch = wave-4 tip (`f42f75d98` + this close task's own doc-only edits,
which touch no test or production-logic file — a comment-only docstring
correction in `library_skills_state.py`, two backlog task filings, and this
recipe's own edits). Baseline = a path-scoped `git checkout 2372ea764 --
tldw_chatbook Tests` overlay of the wave-4 START commit (the multi-commit-
back equivalent of the per-task `git stash -u` technique, per §16's own
precedent for a WAVE-level rather than per-task comparison), run, then
restored via `git checkout HEAD -- tldw_chatbook Tests` (verified
`_measure()` back to 41155/1295 and `git status` clean afterward). Run
SEQUENTIALLY, not concurrently, per §7's own "concurrent runs amplify
flakiness" lesson.

**Machine load, recorded at the start of this sweep**: load average 22.72,
21.91, and 8 concurrent `pytest` processes already running on this machine
(`ps aux`) — sustained heavy CONCURRENT load from several unrelated
long-running test processes throughout wave-4 (Task 3's own report already
recorded load average between ~19 and ~51 over its own run); recorded again
here per §19 lesson 5's own "the paired comparison is what keeps this valid
regardless of load" discipline.

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`f42f75d98` + close) | 372 | 3932 | 1699.84s (28:19) |
| Baseline (`2372ea764`) | 365 | 3934 | 1398.41s (23:18) |

Both counts sit at or just past the documented ~330–371 historical backdrop
(recipe §7) — branch's 372 is 1 over the previous recorded high of 371 —
consistent with this run's own recorded machine load being the heaviest of
any sweep in this wave. Diffing the two failure-name sets: **356 shared, 9
baseline-unique** (noise in the opposite direction, not investigated further
per §7's own precedent — one of the 9,
`test_library_adaptive_reader_closeout.py::
test_closeout_single_app_route_cycle`, is the SAME test TASK-31422 above
files a follow-up for: it failed on the BASELINE and passed on the branch
here, the opposite direction from Task 1's own observation of it, which is
itself further evidence for that filing's own premise — the failure is
direction-independent flakiness, not something either tree's code
introduces), **16 branch-unique**. All 16 resolved without a single
unexplained name: 12 passed cleanly on a combined single-process re-run
(ordinary xdist noise); the remaining 4
(`test_library_media_reader_traversal_t22207.py::
test_focus_traversal_builds_zero_bodies_for_pass_through_rows`,
`test_library_prompts_canvas.py::
test_library_prompt_undo_refreshes_applied_page_and_preserves_basket`,
`test_library_shell.py::
test_library_starter_production_geometry_and_focus_order[size0]`,
`test_library_shell.py::
test_library_starter_production_geometry_and_focus_order[size1]`) each
passed cleanly in TRUE isolation on the branch. None of the 16 touches a
Skills file, Skills-owned code, or this task's own doc-only diff — several
are Media/Notes/Prompts/Collections/Audio tests already named as
pre-existing xdist noise in wave-1/2/3's own sweeps (recipe §7,
`test_library_entry_compose_once.py::
test_source_worker_completion_during_mount_dispatch_reconciles_once` and
`test_library_shell.py::
test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
among them). **Zero unexplained branch-unique failures across the whole
wave-4 span.**

#### Probe run

```
perl -e 'alarm 150; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py
```

| interaction | settle (ms) | max gap (ms) | recompose | full-update | mounts | nodes |
|---|---|---|---|---|---|---|
| media (switch-in) | 794 | 328 | 0 | 2 | 173 | 115 |
| media (re-click same) | 442 | 118 | 0 | 2 | 85 | 115 |
| media (re-click same, 2nd) | 460 | 96 | 0 | 2 | 89 | 115 |
| notes (switch) | 539 | 279 | 0 | 1 | 110 | 110 |
| notes (re-click same) | 312 | 76 | 0 | 1 | 38 | 110 |
| media (switch-back) | 747 | 286 | 0 | 1 | 175 | 115 |
| notes (switch, 2nd) | 853 | 587 | 0 | 1 | 110 | 115 |
| media (switch-back, 2nd) | 606 | 171 | 0 | 1 | 175 | 115 |

Every row is HIGHER than the wave-2 close band (§16: settle 264-485 ms, max
gap 54-195 ms) — most sharply `notes (switch, 2nd)`'s 587 ms max gap versus
§16's own highest recorded gap of 195 ms. This is NOT evidence of a
code-level regression, but the "zero code on the probed path" framing this
section originally used (corrected in the wave-4 final review) was FALSE:
the skills cleanup (`ed4c29d45`) does touch this exact path, via mechanical
receiver swaps of skills-field reads/writes INSIDE five of the probed
path's own methods — `compose_content`,
`_select_library_rail_row_after_source_admission`,
`_toggle_library_media_reader_pane`, `restore_state`, and
`_persist_library_reader_preference` (`self._library_skill(s)_<field>` →
`self._skills_state.<field>`, dict/attribute lookups only, confirmed by
diffing each hunk against every prior/subsequent commit's own claim on
those line ranges — zero control-flow or logic changes anywhere in the
five). That is nanosecond-scale work and cannot explain a ~3x max-gap
delta on its own. The stronger evidence for "no structural regression" is
the load-INDEPENDENT probe columns, which match wave-2's row-for-row:
`recompose` is 0→0 on every row and `full-update` is identical row-for-row
(2/2/2/1/1/1/1/1, both waves) — a real per-frame cost regression on this
path would show up there, not only in wall-clock settle/gap numbers that
scale with ambient CPU contention. The load evidence remains the far more
plausible explanation for the timing delta itself — this probe measures
wall-clock settle time and main-thread gap under whatever CPU contention
exists at run time, and this session's own machine was running 8+
concurrent `pytest` processes and sitting at a load average of ~22.7 for
the ENTIRE sweep window this probe ran inside of, versus wave-2 close's own
probe run, which recorded "no prior run exists to diff against" and gives
no load context of its own either way. One further drift IS visible and
worth naming rather than smoothing over: `mounts`/`nodes` counts rose
slightly and unevenly — media mounts 163→173, nodes 113→115 (every media
row gains exactly 2 nodes); `notes (switch, 2nd)` alone gains nodes
110→115, while the other two notes rows stay at 110. This is small, does
not track the settle/gap delta's shape, and is attributable to ordinary
Media/Notes feature drift on this branch between wave-2 close (`09a5cadff`)
and wave-4 close (`ed4c29d45`), not to this wave's own skills diff: 70
non-doc commits touched `library_screen.py`/`Widgets/Library`/
`Library_Modules` in that span, including several `origin/dev` merges and
Media/Notes rail feature work with no connection to skills (e.g. `e34b11b23`
"the media items pane gets a legible multi-row action grammar", `9c128c7ee`
"the Reader carries a review-set banner", `35e9228df`/`c63e55cc0` review-set
auto-resume/build-from-read-later) — any of which plausibly adds a handful
of extra widgets/nodes to the Media pane and, specifically on a second
notes switch, the notes canvas. Recorded here as this wave's own close-time
baseline (like wave-2's), with both the load caveat and the corrected
probed-path claim attached, rather than either silently matched to the old
band or wrongly flagged as a regression.

#### Lessons

1. **A battery-found hazard's own count correction needs the SAME evidence
   discipline as any other claim — re-derive the full tuple, not just the
   one number that changed.** Wave-4 saw this pattern three separate times
   in one series: Task 1's own "three `@property`/`@x.setter` pairs"
   arithmetic error (should have been six; 2 raw `FunctionDef`s − 1 unique
   name = 1 gap per name, 6 names = 6 gap) propagated unfixed through Task
   2's own report text AND its controller's module docstring — caught in
   the report by Task 2's own post-landing review fix round but missed in
   the docstring copy until Task 3 caught it independently; Task 3's own
   delegator-census restructure count needed a review fix round to correct
   an "18 vs. 28" discrepancy; and Task 2's own mover-count tuple (91 → 87
   → 86, across two in-session battery-caught regressions, Forms B and C of
   §3's sixth bypass shape) required re-deriving every DEPENDENT count
   (exclusion tallies, constructor parameter counts) after each correction,
   not just patching the number that changed — recorded as its own standing
   rule in §3's closing paragraph ("a battery-found hazard shrinking the
   mover set legitimately amends the RED tuple"). Three independent
   incidents in one wave converging on the same failure shape (a stale
   count surviving past the point where better evidence existed) makes this
   worth stating as its own numbered lesson, not three unrelated minor-
   review findings: **any time a mechanical census's own output changes
   after the fact — a battery-found hazard, a review correction, a
   re-run — every number derived FROM that census, not just the one that
   visibly changed, needs re-deriving before the next report or docstring
   copies it forward.**
2. **A CRITICAL production bug can be invisible to the FULL verification
   battery a task brief mandates, and only surface under independent
   review — which is exactly why review is a separate, mandatory gate, not
   a redundant check on a battery that already passed.** Task 2's own
   wiring/ratchet suites, its full `Tests/Skills/` run, its `-k "skill and
   library"` sweep, and its own full sequential xdist paired-baseline sweep
   all went green with the unbound `focused` bug present (§3's own
   "unbound-attribute escape" paragraph) — no test anywhere in the standard
   battery exercises a code path where `getattr(self, "focused", None)`'s
   silent `None` default changes an assertion's outcome, because Textual's
   own generic default-focus fallback happens to land on the same widget
   the bug's own correct behavior would have. Only a coordinator review's
   own targeted bare-self/`getattr` re-sweep (§3's own closing
   recommendation: grep the whole moved-body source for `getattr(self,
   "<literal>"` calls, not just `self.<attr>` accesses) found it, after the
   commit had already landed. This generalizes past skills specifically: a
   battery that is green is evidence the KNOWN hazard shapes did not fire,
   not evidence no hazard exists — the review gate exists precisely to
   catch the shapes the battery's own design cannot see yet.
3. **A headless click-probe's absolute numbers move with ambient machine
   load exactly the way the xdist sweep's absolute failure counts do (§19
   lesson 5) — record the load alongside the numbers, or the probe's own
   "must not move outside noise" contract (§9) becomes unfalsifiable.**
   This close's own probe run posted every interaction slower than the
   wave-2 close's recorded band, under a machine sitting at a ~22.7 load
   average with 8+ concurrent `pytest` processes running throughout — a
   condition wave-2's own probe run recorded no comparable load context
   for. This task's diff only makes mechanical receiver swaps of
   skills-field reads/writes on the probe's own Media/Notes rail-switch
   path (nanosecond-scale — see the correction above, this same section),
   so the honest read is ambient load, not regression — corroborated by
   the load-independent probe columns (recompose, full-update) matching
   wave-2's row-for-row — but that conclusion is only defensible because
   the load was actually measured and recorded at run time, not asserted
   after the fact.
   A future controller-move PR's own before/after probe pair (§9) should
   capture the SAME `ps aux`/load-average snapshot each time, or a real
   regression and a noisy machine become indistinguishable from the numbers
   alone.

## 20. The ingest series, as landed — the fifth rehearsal, and the second series to trip the PR-0a re-export contract

Ingest (wave-5, `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest`)
is the fifth subsystem series to run this recipe (Tasks 1-3: state PR,
controller PR, cleanup PR). It is a single-prefix, single-controller
subsystem — closest in shape to export and collections, not to skills'
three-way prefix split or search+RAG's combined-subsystem merge — but it
is the series that found TWO of the recipe's own numbered bypass shapes
(§3's seventh and eighth), one in each of its first two tasks, and its
cleanup task's dead-import prune collided with the PR-0a re-export
contract (`Tests/Architecture/test_library_support_layer_surface.py`'s
`_SURFACE`) — the SAME shape the conversations exemplar's own Task 7
first hit (§11), not a new one; see the census section below for the
correction to an earlier, overstated "first time" draft of this claim.
This section records what actually landed.

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 1 | State | 20 fields (single `_library_ingest_` prefix, no plural variant, no wiring-field exclusion, no computed/entangled defaults) → `LibraryIngestState` (0 methods). Found the recipe's seventh bypass shape: an `object.__new__`-bypassed test fixture hand-setting a flat `_library_ingest_<field>` name breaks the INSTANT the shim installs, not deferrable to cleanup like every prior bypass shape — 27 sites across 6 files fixed in the same GREEN commit (one file, `test_parakeet_v2_install_ui.py`, missed by the first sweep because its own filename/test names/`-k` keywords contain neither "ingest" nor "library" — caught by a coordinator review's content-grep, not this task's own `-k`-filtered sweep; fixed in fix round 1) | 41574 → 41520 lines, 1302 methods (unchanged) |
| 2 | Controller | 56 of 78 "ingest"-named method candidates → `LibraryIngestController` (25 `@on` + 2 `action_*` + 1 staticmethod + 28 plain; 22 exclusions: 4 `@work` framework-decorator hazard, 3 module-globals-coupling — one, `_resolve_ingest_source`, found only by fix round 1's coordinator-mandated mechanical module-globals census (§3's eighth bypass shape, named here for the first time), not the original battery — 9 unbound-fake-self/`object.__new__`-bypass (the seventh shape recurring at the CONTROLLER level: an excluded method's OWN caller needs `_ingest_controller` seeded too, not just `_ingest_state` — 25 more call sites across 5 files), 6 instance-attribute-monkeypatch) | 41520 → 40096 lines (round 0) → 40131 lines (fix round 1, `_resolve_ingest_source` reverted), 1302 methods (unchanged: pure move, 56 `FunctionDef`s out, 56 one-line delegators in, one of them a staticmethod). Controller born-governed at 2510, fix round 1 (the eighth-bypass-shape fix + module-globals census) → 2536 |
| 3 | Cleanup | Shim block (20 properties, single prefix) deleted; every remaining screen-side flat reference retargeted to `self._ingest_state.<field>` (37 pre-existing occurrences — 31 literal `self.<flat_name>` attribute accesses + 6 `getattr(self, "_library_ingest_<field>", default)` calls needing a RECEIVER fix, not just a string swap, mirroring the skills series' own `getattr` lesson — across 12 excluded/shell-plumbing methods still full-bodied on the screen); 297 test-side attribute-path retargets + 3 `SimpleNamespace` fixture restructurings (flat kwarg → nested `_ingest_state=SimpleNamespace(...)`) across 17 files spanning `Tests/UI`, `Tests/App`, `Tests/Library`, `Tests/integration`, `Tests/Utils`, and `Tests/ProductionApp` (a repo-wide content census examined 19 files total; 2 -- `test_library_url_ingest_submit.py` and `test_library_ingest_wiring.py` -- had only false-positive moved-method-name mentions, zero real field retargets; the widest real test-root spread of any series to date); 6 of the 56 screen delegators deleted (repo-wide zero-external-reference census across `tldw_chatbook/` + every `Tests/` root, excluding the controller module and each name's own delegator body); 8 dead imports removed (all left dead by task 2's own move, each independently confirmed already re-imported and live inside the controller) — a 9th candidate, `_ingestible_file_filters`, stayed despite zero screen-side uses because it is `_SURFACE`-pinned (PR-0a's own re-export contract, the same shape the conversations exemplar's own Task 7 first hit); corrected the module-globals census evidence for `_apply_library_ingest_backend_save`/`_sync_library_canvas` from 7 files/~20 sites to the true 10 files/38 sites (3 files missed by task 2's own grep because their patch sites used a variable name other than `library_screen`/`library_screen_module` — `screen_module`, and a `monkeypatch.context()`'s own `patcher.setattr(...)`); 4 module/constructor-docstring corrections in the controller (two stray "63" counts, the now-false "LibraryScreen keeps one-line delegators under every one of these 56 original names" claim, and the census-listing correction above) | 40131 → 40094 lines, 1302 → 1296 methods (6 fewer `FunctionDef`s — exactly the 6 pruned delegators). Controller: 2536 → 2558 (comment-only growth: the 4 docstring corrections) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`41574/1302 → 41520/1302 → 40096/1302 → 40131/1302 → 40094/1296` (final).

**Controller-file governance pin** (§17): `library_ingest_controller.py`
was born-governed the moment it existed (Task 2): `2510 → 2536` (Task 2's
own fix round 1) `→ 2558` (this cleanup task, comment-only) `→ 2569`
(this cleanup task's own fix round 1, post-review: the `_sync_library_
canvas` census's "site" definition and reproducible breakdown, comment-only).

### Dynamic-dispatch census — zero hazard shapes found

Re-derived Task 2's own findings before deleting the shim: a repo-wide
content grep for every one of the 20 state fields as a `getattr`/`setattr`
string-literal second argument, and for any dict-literal dispatch table
keying off a field name, found no dynamic-dispatch hazard touching Ingest.
Unlike the skills/search+RAG/collections precedent, Ingest fields never
intersect the shared reader-preference dispatcher
(`_replace_library_reader_preference`/`_persist_library_reader_
preference`) or any choice-strip visibility dict — Ingest has no "reader"
mode and no per-type choice-strip visibility field. The only screen-side
dynamic call touching Ingest at all is `canvas_sync.py`'s own `kind ==
"ingest"` branch, which calls `screen._build_library_ingest_state()` (a
METHOD, not a flat field name) — no field-name string to retarget.

The census DID find 6 `getattr(self, "_library_ingest_<field>", default)`
call sites needing the skills series' own "receiver, not just string"
fix — 3 in the still-screen-resident `_build_library_ingest_state`, 1 in
`check_action`, and 2 in the still-screen-resident `handle_library_
ingest_backend_switch` (which ALSO has 1 direct, non-`getattr`
`self._library_ingest_backend_generation = generation` assignment
alongside them, already counted among the 31 plain-attribute retargets
below, not this 6). All 6 confirmed retargeted with the correct receiver
(`getattr(self._ingest_state, "<field>", default)`), verified by a
zero-result repo-wide grep for `getattr(self, "_library_ingest_` after
the change.

### Screen-side field retarget

**37 pre-existing flat-name occurrences**, none inside a moved method body
(the controller move already deleted those in Task 2) — all 37 live inside
12 excluded, still screen-resident methods and shell/plumbing methods:
`_build_library_ingest_state` (7: the state-snapshot builder itself, one
of the 6 instance-attribute-monkeypatch exclusions), `check_action` (1),
`handle_library_ingest_backend_switch` (4, one of the 9 unbound-fake-self
exclusions), `_save_library_ingest_backend` (2, a `@work`-hazard
exclusion), `_library_ingest_browse_location` (1, an unbound-fake-self
exclusion), `handle_library_ingest_option_value_changed` (2, an unbound-
fake-self exclusion), `handle_library_ingest_directory_browse` (1, an
unbound-fake-self exclusion), `_apply_parakeet_v2_install_result` (1, an
UNRELATED model-install completion handler that happens to read the
Ingest form field), `_run_debounced_library_ingest_preflight` (2, an
unbound-fake-self exclusion), `_on_preflight_retry` (1, shell/plumbing),
`_do_submit_ingest` (2, an unbound-fake-self exclusion),
`_apply_library_external_preparation` (2, the separate "external source"
onboarding feature that shares two fields with Ingest per task 1's own
state-module docstring), `_enqueue_library_ingest_snapshot` (4, an
unbound-fake-self exclusion), `_load_library_ingest_options_from_config`
(1, a module-globals-coupling exclusion), `_build_ingest_options_snapshot`
(2, an unbound-fake-self exclusion), and 4 more single-field-touching
`__init__`/`on_mount`/rail-collapse shell sites
(`_library_emergency_return_eligibility`, `_library_resize_layout_
signature`, `on_mount`, `_set_library_rail_collapsed`). A single per-field
regex pass (20-field mapping, `\bself\.<flat_name>\b` → `self._ingest_
state.<field>`, plus a second pass for the `getattr(self, "_library_
ingest_<field>"` receiver-fix shape above) rewrote all 37 mechanically;
re-verified with a zero-result grep for every one of the 20 flat field
names over the whole file afterward.

### Test retarget — the widest root spread to date

Content-census (never `-k`) across ALL of `Tests/` found flat-name
mentions in 19 files; 2 (`test_library_url_ingest_submit.py`,
`test_library_ingest_wiring.py`) turned out to be false-positive moved-
method-name mentions only (`_submit_library_ingest_form()` and similar --
methods, not fields), zero real retargets. The remaining 17 needed real
changes: `test_library_ingest_inline_consent.py` (95),
`test_library_shell.py` (52), `test_library_ingest_canvas.py` (40),
`test_library_ingest_retry_last.py` (29), `test_library_screen.py` (26),
`test_library_ingest_flow.py` (21, `Tests/integration`),
`test_library_ingest_characterization.py` (11), `test_submit_library_
ingest_job.py` (6, `Tests/App`), `test_library_ingest_clear_focus.py` (5),
`test_parakeet_v2_install_ui.py` (4 — the same file the state PR's own
seventh-bypass-shape fix round 1 had to add after the first sweep missed
it entirely, confirming the filter-blindness lesson generalizes to the
CLEANUP task's own census too, not just the state PR's), `test_library_
resize_focus_gates_t23025.py` (2), `test_library_ingest_structural.py`
(2), `test_retired_tldw_api_worker_pipeline.py` (2, `Tests/ProductionApp`
— a root no prior series' own cleanup task needed), `test_library_ingest_
keyboard.py` (1), `test_library_canvas_scoped_sync.py` (1 attribute-path
retarget here, PLUS its own `SimpleNamespace` fixture restructuring
below) — 297 attribute-path retargets total across 15 of the 17 files
(the other 2, `test_ingest_preflight_egress.py`/`test_config_nested_
settings.py`, needed ONLY the fixture restructuring below, zero plain
attribute-path retargets), one mechanical regex pass, zero assertion
VALUE changes (every edit is a receiver-path rewrite: `screen._library_
ingest_<field>` / `loader._library_ingest_<field>` / `library._library_
ingest_<field>` → `<receiver>._ingest_state.<field>`, confirmed safe
because every receiver in every hit is either a real, `__init__`-
constructed `LibraryScreen` or an `object.__new__`-bypass
instance task 1's own fix round already seeded with a real `_ingest_state`
— never a bare `SimpleNamespace` needing restructuring).

**3 more needed the flat-kwarg-to-nested-`SimpleNamespace` restructuring**
the conversations/skills exemplars' own cleanup PRs established, because
their receiver is a bare, ad-hoc `SimpleNamespace` fake standing in for an
unbound `self`, not an `_ingest_state`-seeded screen: `test_library_
canvas_scoped_sync.py::test_ingest_checkbox_routes_to_ingest_canvas_sync`
(`_library_ingest_form=form` → `_ingest_state=SimpleNamespace(form=form)`,
the fake `self` for the excluded `handle_library_ingest_option_value_
changed`), `test_ingest_preflight_egress.py::test_the_typing_debounce_
forbids_probing_even_when_it_is_enabled` (`_library_ingest_path_debounce_
timer=object(), _library_ingest_form=SimpleNamespace(...)` → both nested
under one `_ingest_state=SimpleNamespace(...)`, the fake `self` for the
excluded `_run_debounced_library_ingest_preflight`), and `test_config_
nested_settings.py::test_library_ingest_browse_location_audit_fix`
(`_library_ingest_form=SimpleNamespace(path="")` → nested, the fake
`self` for the excluded `_library_ingest_browse_location` — found OUTSIDE
every canonical ingest test root, the same file task 1's own seventh-
bypass-shape census had already flagged as evidence for "widen beyond the
obvious roots").

A repo-wide false-positive check (every `_library_ingest_<name>` mention
that is NOT one of the 20 field names) found only method-name substring
collisions (`_submit_library_ingest_form`, `_current_library_ingest_
start_consent`, `_restage_library_ingest_last_submission`, and similar —
call sites for MOVED method names that legitimately keep their flat
spelling since methods are not fields) and prose-comment mentions of
field names in docstrings — none required a code change; the docstring
mentions were left as historical prose except where corrected below.

### Delegator census — 50 KEEP, 6 PRUNED (~11%)

Of the 56 moved names: **25 `@on` handlers + 2 `action_*` methods KEEP
unconditionally** per the recipe's own transform whitelist. Of the
remaining 29 (28 plain + 1 staticmethod), a repo-wide grep for each name
across `tldw_chatbook/` and every `Tests/` root (excluding the controller
module's own internal calls and each name's own one-line screen-delegator
body) found **23 with a genuine external caller** and **6 with none**,
each independently re-verified with a second, broader "name appears
anywhere in the line" pass (not just call-shaped) before deletion:
`_adopt_library_ingest_path`, `_ingest_job_id_from_button` (the cluster's
one staticmethod — its only callers are internal, `self._ingest_job_id_
from_button(...)`, on the controller instance itself, which needs no
screen delegator at all), `_library_ingest_restage_discards_work`,
`_restage_library_ingest_last_submission`, `_set_library_ingest_panels_
collapsed`, `_update_library_ingest_retry_label`.

This prune fraction (6 of 56, ~11%) sits at the LOW end of every prior
series' own recorded fraction (export's 1-of-22 ~5% < ingest's 6-of-56
~11% < skills' 16-of-86 ~19% < collections' 14-of-64 ~22% < search+RAG's
12-of-42 ~29% < conversations' 18-of-61 ~30%), consistent with the
recipe's own inverse-relationship finding (§15 lesson 3): only 22 of 78
candidates were excluded here (a comparatively SMALL exclusion count next
to skills' 41-of-127), so fewer excluded, screen-resident methods exist to
keep calling their moved siblings via `self.<name>()` — the mechanism that
would otherwise keep a screen delegator's reference count above zero.

Pruning the staticmethod required its OWN wiring-test treatment, not
covered by the plain-delegator skip/absence pair alone:
`test_ingest_cluster_staticmethods_forward_to_the_controller_class`
iterated `_INGEST_CLUSTER_STATICMETHOD_NAMES` (a 1-name frozenset) and
asserted a class-forwarding shape via `inspect.getsource` — once the
delegator is gone, `getattr(LibraryScreen, name)` returns `None` and
`inspect.getsource(None)` raises `TypeError` rather than failing the
assertion cleanly. Fixed the same way `test_screen_delegates_ingest_
handlers` was: skip names in `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED` and
assert genuine absence instead.

### Shim block deletion

The task-1-generated `_library_ingest_<field>` property-shim loop
(installed at module end, `dataclasses.fields(LibraryIngestState)`-driven)
was deleted wholesale once the census above confirmed zero remaining
consumers anywhere in `tldw_chatbook/` or `Tests/` outside `LibraryIngest
Controller`'s own PERMANENT generated shim loop (installed by task 2,
reading `self._ingest_state_accessor().<field>` — untouched by this task,
per the task brief's explicit "controller shims STAY" instruction).

### Import verification

**8 dead imports removed**, each verified single-occurrence (import line
only, via AST `Name`-node usage count) in `library_screen.py`, then
checked against `Tests/Architecture/test_library_support_layer_surface.
py`'s `_SURFACE` dict (the PR-0a re-export contract) before deletion: all
8 left dead by task 2's own controller move (each name's only screen-side
usage lived inside one of the 56 moved method bodies), each independently
confirmed already re-imported and live inside `library_ingest_
controller.py`: `ACTIVE_INGEST_STATES`, `normalize_active_ingest_source`
(`Library.library_ingest_jobs`); `LibraryIngestFormState`, `build_ingest_
forecast`, `format_ingest_progress_line`, `ingest_progress_action_
signature` (`Library.library_ingest_state`); `build_type_group_title`
(`Widgets.Library.library_ingest_canvas`); `capabilities_for_backend`
(`Library.ingest_capabilities`).

**One candidate deliberately kept despite zero screen-side uses**:
`_ingestible_file_filters` (`Library_Modules.screen_helpers`) IS
`_SURFACE`-pinned — `test_screen_still_re_exports_every_moved_name`
asserts `hasattr(library_screen_module, "_ingestible_file_filters")`
regardless of whether the screen class itself still calls it, because the
PR-0a foundation task's re-export contract is about the MODULE surface,
not about live usage. **This reconfirms, rather than newly discovers, the
conversations exemplar's own lesson** (§11: "'Dead within this file' is
not the same question as 'dead'" — Task 7's own report first hit this
EXACT shape with `LIBRARY_CONVERSATION_READER_MAX_CHARS`, had to restore
it after deletion broke the same contract test) — export/collections/
search+RAG/skills' own zero-collision results are the check WORKING as
designed post-lesson, not evidence the shape stopped recurring. Caught
here only because the check was still run per-candidate on this task's
own 9, not skipped on the strength of four intervening "never collides"
results. **The lesson, restated for a fifth time now: check every
dead-import candidate against `_SURFACE` individually, every task — a
clean run of intervening series is not evidence the NEXT series' own
candidate set is also clean.**

**How to run that check, added at the wave-6 close because the prompts
series nearly failed it a second way.** "Individually" means an
**exact-name lookup of each candidate against `_SURFACE`'s own contents**
— never a subsystem-word grep. The prompts cleanup's first `_SURFACE`
check was a case-sensitive `grep prompt` over that file, which returned
**zero matches** and would have cleared all 30 of its candidates for
deletion — five of them wrongly, sending
`test_screen_still_re_exports_every_moved_name` red (loudly, not silently:
this shape costs a debugging round, not a shipped regression). The five
that are actually pinned all spell it `PROMPT`
(`LIBRARY_PROMPT_DIRTY_VETO_COPY`,
`LIBRARY_PROMPT_SAVE_STATUS_COPY`,
`_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP`,
`_LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS`,
`_LIBRARY_PROMPT_WRITE_WORKER_GROUPS`). Screaming-snake constants are
exactly what a re-export contract is most likely to pin, so the
lowercase-word grep is wrong precisely where it matters most. That run
also produced **the largest save yet — 5 of 30**, against ingest's 1 of 9,
so the shape is not decaying with repetition.

### Moved-docstring / module-docstring corrections

Four inaccuracies in `library_ingest_controller.py`'s own MODULE and
CONSTRUCTOR docstrings, plus two more of the same "63" shape in the
wiring test's own docstrings (none are moved method bodies — all freely
editable, no byte-for-byte deferral needed):

1. **A now-false present-tense architecture claim, module docstring**:
   "`LibraryScreen` keeps one-line delegators under every one of these 56
   original names" — true when task 2 wrote it, false for 6 of the 56 as
   of this task's own delegator prune. Fixed to name the current 50-of-56
   count and point at `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED` for the
   list, mirroring the skills series' own precedent (which found and
   fixed the IDENTICAL shape of claim, and left a forward note that
   `library_rag_search_controller.py` still carries it uncorrected — not
   fixed here either, out of this task's own file scope).
2. **The SAME claim, second location**: the `LibraryIngestController`
   class's own docstring repeats it verbatim ("keeps one-line delegators
   for every original name this cluster moved (56 ...)"). Same fix,
   second site.
3. **A stray "63" count, constructor docstring**: `"Every one of the 63
   method bodies..."` where every other count in the file correctly says
   56 — an arithmetic slip that never corresponded to any real derivation
   (not 78-22, not 78-15, not any other combination task 2's own report
   used). Fixed to 56.
4. **The carried census-listing correction** (task 2's own re-review
   minor): see the "IMPORTANT" entry in §3's eighth-bypass-shape section
   above — 3 files missed, true count 10 files/38 sites, not 7/~20.

Plus, in `Tests/Architecture/test_library_ingest_wiring.py` (a different
file, its own two docstrings): the SAME "63" slip, twice more
(`test_ingest_controller_owns_its_cluster`'s and `test_screen_delegates_
ingest_handlers`'s own docstrings) — fixed to 56 in both, alongside that
file's own wiring-test finalization below.

### Wiring test finalization

`Tests/Architecture/test_library_ingest_wiring.py`:
`test_state_object_fields_match_the_shim_surface` DELETED (screen shim
gone, mirrors the conversations/export/collections/search+RAG/skills
precedent exactly); `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (6
names, including the cluster's one staticmethod) added, with
`test_screen_delegates_ingest_handlers` skipping those names and instead
asserting their genuine ABSENCE from `LibraryScreen`; `test_ingest_
cluster_staticmethods_forward_to_the_controller_class` given the same
skip/absence treatment for its own 1-name set. 5 of the original 6 tests
remain (the shim-surface test's removal is the only count change) — all 5
green post-cleanup.

### Size ratchet

Fresh measurement via the ratchet's own `_measure` semantics (`ast`-walked
line count + `LibraryScreen` method count, not `wc -l`): **40094 lines,
1296 methods** — 1296 = 1302 (task 2's post-move count) − 6 (exactly the
pruned delegator count), confirming the prune was a pure deletion with no
replacement. Lowered in this same commit per recipe §6. Controller
re-measured at **2558 lines** (comment-only growth from the four
docstring corrections above), re-pinned same-commit per §17's re-pin-at-
move flow.

### Battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **Wiring suite + ratchets**: `test_library_ingest_wiring.py` (5, post-
  deletion/post-pruning) and `test_library_support_layer_surface.py` (8)
  passed together (13 passed); both size ratchets (34 passed, 2 failed —
  the documented pre-existing `chat_screen.py` rows, recipe §7's own
  standing list).
- **Directly affected ingest test files** (16 of the 17 files this task
  edited -- the content-grep-derived set): 488 passed, 9 failed
  in the first combined run — 8 of the 9 confirmed pre-existing via a
  `git stash -u` pristine-baseline rerun of the SAME 8 node-ids (all 8
  fail identically on both trees: 2 CSS-color-parity rows in `test_
  library_ingest_canvas.py`, a footer-shortcut-count assertion in `test_
  library_ingest_retry_last.py`, a DOM-query-volume budget in `test_
  library_resize_focus_gates_t23025.py`, 3 CSS-geometry/focus-styling rows
  in `test_library_ingest_structural.py`, and an environment-dependent
  network-egress-probe outcome in `test_ingest_preflight_egress.py`); the
  9th, `test_library_canvas_scoped_sync.py::test_notes_per_click_updates_
  keep_screen_and_canvas_identity`, is the SAME name already documented
  above (wave-3 task 4) as a Notes-only characterization test unrelated to
  this diff. **Zero real regressions.**
- **The 2 largest-diff files not in that 16-file set** (`test_library_
  screen.py`, 26 retargets; `test_library_shell.py`, 52 retargets) --
  `test_library_screen.py` run directly: **32 passed, 0 failed**.
  `test_library_shell.py` is too large for a standalone timed run to
  complete cleanly (a `-n 4` attempt was killed mid-run by its own time
  budget before printing a summary, itself consistent with the fd-leak-
  sentinel warning below); its coverage instead comes from the full
  sequential xdist sweep just below, which already includes every test
  in the file as part of `Tests/UI -k "library"` and found no branch-
  unique failure inside it.
- **preflight**: `./scripts/preflight.sh` — all checks green (no
  diagnostic-inventory drift; this task's diff touches zero
  `logger.warning`/persistent-diagnostic call sites).
- **Full sequential xdist paired-baseline sweep** (branch 356 failed/
  3994 passed vs. an isolated-worktree baseline at `e3d85ad21`, 358
  failed/3992 passed; 351 shared, 7 baseline-unique, 5 branch-unique, all
  5 resolved on a combined single-process rerun): see the new §7
  sweep-evidence entry this task adds.

### Lessons

1. **A `getattr(self, "<flat_name>", default)` receiver fix generalizes
   beyond skills.** The skills series' own lesson ("a mechanical
   string-literal-only find/replace pass over `getattr`/`setattr` calls is
   not safe by construction") recurred here verbatim — 6 of this task's 37
   screen-side retargets were `getattr` calls needing the SAME
   receiver-plus-string fix, not just a string swap. Any future
   subsystem's field-retarget census should grep for `getattr(self,
   "<flat_name>"` as a DISTINCT shape from `self.<flat_name>`, not assume
   the plain-attribute regex catches both.
2. **The seventh bypass shape's filter-blindness lesson applies to the
   CLEANUP task's own census too, not just the state PR's.** Task 1's own
   fix round found `test_parakeet_v2_install_ui.py` only via a
   coordinator's content-grep, after every `-k`-filtered sweep this
   series ran (including this cleanup task's own initial pass) collected
   zero tests from it. This task's own content-census (never `-k`) caught
   it again independently by construction, but the near-miss underscores
   that a `-k` sweep is a discovery AID, never the completeness proof,
   for a file whose name/test-names/keywords never mention the
   subsystem — a standing rule this series proves twice now, not once.
3. **The conversations exemplar's own `_SURFACE`-collision lesson (§11:
   "'dead within this file' is not the same question as 'dead'")
   reconfirms, not just once.** This task's initial draft of this very
   section claimed ingest was the FIRST cleanup task to hit a `_SURFACE`
   collision — wrong, and caught only by re-reading §11 before finalizing
   rather than trusting the draft's own "first" framing. The conversations
   exemplar's own Task 7 hit the IDENTICAL shape first
   (`LIBRARY_CONVERSATION_READER_MAX_CHARS`, restored after deletion broke
   the same re-export test); the four INTERVENING cleanup tasks (export,
   collections, search+RAG, skills) each recorded zero collisions BECAUSE
   they ran the check that lesson mandated, not because the shape stopped
   recurring. This task's 9-candidate set had exactly one
   (`_ingestible_file_filters`) that WAS `_SURFACE`-pinned. **The lesson,
   restated: check every dead-import candidate against `_SURFACE`
   individually, every task — and when writing up a finding as a "first,"
   grep the recipe's own prior sections for the identical shape before
   committing that framing to the record.**
4. **A module-globals census's own patch-target grep must resolve the
   test's actual import alias, not assume the two conventional
   spellings.** The carried census-listing correction (§3's eighth bypass
   shape entry above) found 3 files whose patch sites this series' own
   task 2 grep missed because they imported `library_screen` under an
   alias other than `library_screen`/`library_screen_module`
   (`screen_module`), or reached the patch through a call shape other than
   a bare `monkeypatch.setattr(...)` (`monkeypatch.context()`'s own
   `patcher.setattr(...)`, and a multi-line `patch.object(...)` a
   single-line-oriented grep did not span). **The lesson: find every
   import alias for the target module FIRST, then search for that alias
   specifically — a fixed-string search for the two spellings a prior
   series happened to see is not a completeness proof for the next one.**

### Wave-5 close

Wave-5 (`.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest`,
branch `refactor/library-decomp-wave5-ingest`) scoped itself to ingest alone
(the deferred-from-wave-4 subsystem, §19's own "Wave-4 close" note). The
ingest series (Tasks 1-3, above) is complete. This section is Task 4's own
wave-level pin-trajectory re-derivation, verification battery, and lessons.

#### Pin trajectory — full wave-5 chain

Re-derived from `git show <commit>:<path>` reads of the `_BUDGETS` row in
`Tests/Architecture/test_screen_size_ratchet.py` and the controller row in
`Tests/Architecture/test_library_modules_size_ratchet.py` at each commit
(not carried over from any report), in true chronological order (`git log`
lists newest-first; the table below is oldest-first):

| Task | PR | Commit | Screen `_BUDGETS` after | Controller pin after |
|---|---|---|---|---|
| — | (wave-5 start) | `9e62dd8f7` | 41574 / 1302 | — (file does not exist yet) |
| 1 | Ingest state (RED — wiring test + state module + characterization pins; screen untouched) | `a11220648` | 41574 / 1302 (unchanged) | — |
| 1 | Ingest state (GREEN, series continues) | `12ba4fb13` | 41520 / 1302 | — |
| 1 | Fix round 1/5 — seeded the missed `object.__new__` fixtures; recipe gains the seventh bypass shape (series complete) | `74a6f5774` | 41520 / 1302 (unchanged) | — |
| 2 | Ingest controller (RED — wiring-test + 5 new handler pins only) | `44ab7383b` | 41520 / 1302 (unchanged) | — |
| 2 | Ingest controller (GREEN, born-governed) | `68a896993` | 40096 / 1302 | 2510 |
| 2 | (blame-ignore, no functional change) | `18e9c60f7` | 40096 / 1302 (unchanged) | 2510 (unchanged) |
| 2 | Fix round 1/5 — CRITICAL `_resolve_ingest_source` exclusion; recipe gains the eighth bypass shape (series complete) | `e3d85ad21` | 40131 / 1302 | 2536 |
| 3 | Ingest cleanup (GREEN, series complete) | `e6148e29a` | 40094 / 1296 | 2558 |
| 3 | (blame-ignore, no functional change) | `3ebbab805` | 40094 / 1296 (unchanged) | 2558 (unchanged) |
| 3 | Fix round 1/5 — doc-only count corrections (dead-import count, delegator-caller relabel) | `5b9c7bdf4` | 40094 / 1296 (unchanged) | **2569 (wave-5 final)** |

Full chain, screen: `41574/1302 → 41520/1302 → 40096/1302 → 40131/1302 →
40094/1296` (final). Full chain, controller: born `2510` at its very first
commit (Task 2's own GREEN) `→ 2536` (Task 2's own post-review fix round:
the CRITICAL `_resolve_ingest_source` exclusion, comment growth from the
census documentation added alongside it) `→ 2558` (Task 3, comment-only:
four module/constructor-docstring corrections) `→ 2569` (Task 3's own fix
round 1, comment-only: the `_sync_library_canvas` census's "site"
definition and reproducible 10-file breakdown).

Net wave-5 shrink: 1480 screen lines, 6 methods (the 6 pruned ingest
delegators — a pure move is always net-zero methods on the screen; only a
delegator prune changes the method count). Task 4's own fresh `_measure()`
call (both ratchet files' own semantics, not `wc -l`) gives **40094 lines /
1296 methods** for the screen and **2569 lines** for the controller — EXACT
matches to both recorded pins, zero drift, nothing to lower.

#### Verification battery (Task 4, this close)

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, `-p no:randomly` where applicable.

- **Fresh `_measure()` on both ratchet files**: 40094/1296 (screen), 2569
  (controller) — exact match to both recorded pins, zero drift.
- **All six wiring suites** (`test_library_collections_wiring.py`,
  `test_library_conversations_wiring.py`, `test_library_export_wiring.py`,
  `test_library_ingest_wiring.py`, `test_library_search_rag_wiring.py`,
  `test_library_skills_wiring.py`) **+ 4 characterization files**
  (`test_library_ingest_characterization.py`, `test_library_collections_
  characterization.py`, `test_library_conversations_characterization.py`,
  `test_library_export_characterization.py`) **+ both size guards + the
  recompose-census guard (`test_library_recompose_ratchet.py`) + the
  support-layer surface suite (`test_library_support_layer_surface.py`),
  combined single run**: **105 passed, 2 failed** — both the documented
  pre-existing `chat_screen.py` ratchet rows (recipe §7's own standing
  list), unrelated to this diff.
- **Full `Tests/Architecture/` run**: **550 passed, 1 skipped, 16 failed** —
  identical categories to wave-4 close's own documented 16 (Console
  realtime/review-selection boundary ×2, console wave6 closeout/inventory
  ×4, default-timeout-session-guard ×1, persistent-diagnostic-inventory ×2,
  chat_screen ratchet ×2, timer-path-static-update-inventory ×3,
  worker-exclusive-group-inventory ×2 — same 1-skip reason, TASK-15743's
  unfetchable pinned commits), 7 more PASSING than wave-4 close's own 543
  (ordinary suite growth from unrelated work landed on `dev` in the
  interim). Zero Library/Ingest-scoped failures.
- **preflight**: `./scripts/preflight.sh` — all six derived-artifact checks
  pass (CSS bundle, profile-owned-path census, diagnostic inventory,
  backlog task-id sweep — 3,241 task files including this close's own new
  follow-up filing (TASK-31650), chachanotes table allowlist, index plan
  pins).

#### Full sequential xdist paired-baseline sweep — whole-wave span

Branch = wave-5 tip (`5b9c7bdf4` + this close task's own doc-only edits,
which touch no test or production-logic file — this recipe's own edits, a
`.git-blame-ignore-revs` addition, a backlog task filing, and the close
report). Baseline = an ISOLATED `git worktree add` + its own `uv venv`/
`uv pip install -e ".[dev]"` at the wave-5 START commit (`9e62dd8f7`) —
never a same-tree checkout overlay, per this wave's OWN task-1 lesson
(§3/§7): a same-tree overlay is interruption-unsafe for a long-running
background sweep, and this close applied that lesson from the start rather
than re-learning it. Run SEQUENTIALLY, not concurrently, per §7's own
"concurrent runs amplify flakiness" lesson.

**Machine load, recorded at the start of this sweep**: load average 3.03,
3.15, 3.18 (`uptime`) with 2 `pytest` processes already on the machine (one
unrelated, from a different repo checkout) — a substantially QUIETER
machine than wave-4 close's own recorded ~22.7 load average, worth naming
alongside the numbers per that close's own lesson 3.

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`5b9c7bdf4` + close) | 356 | 3994 | 1457.61s (24:17) |
| Baseline (`9e62dd8f7`) | 356 | 3985 | 1495.87s (24:55) |

**351 shared failures, 5 branch-unique, 5 baseline-unique** (baseline-unique
not investigated, per §7's own established precedent). All 5 branch-unique
names resolved, zero real regressions: `test_library_shell.py::
test_library_note_compact_deep_link_intent_opens_notes_stage[context2-
#library-note-body-editor-False]` is a genuinely NEW name, confirmed
pre-existing by reproducing identically (the same 30s DOM-mount-timeout
signature) in TRUE isolation on BOTH the branch and the isolated pristine
baseline worktree — added to §7's documented list below; the other 4 are
ALL names already documented in §7: `test_library_media_reader_
traversal_t22207.py::test_loading_banner_paints_in_place_without_body_
rebuild` (wave-5 task 2's own entry, reconfirmed reproducing here again),
`test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_
navigation_and_local_back` (wave-3 task 5), `test_library_prompts_
canvas.py::test_library_prompt_undo_refreshes_applied_page_and_preserves_
basket` (wave-4 close, passed cleanly here too) and `test_library_shell.py::
test_library_media_initial_error_is_unknown_and_retry_is_unique` (wave-5
task 1, characterized there as load-adjacent flaky, passed cleanly here).
None of the 5 touches Ingest code or this task's own doc-only diff. **Zero
real regressions.**

#### Probe run

Run separately, after both sweeps completed, to avoid CPU contention.
Machine load at run time: `uptime` reported load average 2.31, 3.84, 5.37
— substantially quieter than wave-4 close's own ~22.7, and close to
wave-2 close's own unrecorded-but-implicitly-quiet baseline.

```
perl -e 'alarm 150; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py
```

| interaction | settle (ms) | max gap (ms) | recompose | full-update | mounts | nodes |
|---|---|---|---|---|---|---|
| media (switch-in) | 476 | 119 | 0 | 2 | 173 | 115 |
| media (re-click same) | 318 | 54 | 0 | 2 | 85 | 115 |
| media (re-click same, 2nd) | 325 | 52 | 0 | 2 | 85 | 115 |
| notes (switch) | 365 | 130 | 0 | 1 | 110 | 110 |
| notes (re-click same) | 273 | 47 | 0 | 1 | 38 | 110 |
| media (switch-back) | 399 | 85 | 0 | 1 | 175 | 115 |
| notes (switch, 2nd) | 362 | 155 | 0 | 1 | 110 | 110 |
| media (switch-back, 2nd) | 379 | 81 | 0 | 1 | 175 | 115 |

Every row sits INSIDE wave-2 close's own recorded band (§16: settle
264-485 ms, max gap 54-195 ms) — the closest match of any close-time probe
run to that original baseline, and far below wave-4 close's own elevated
band (settle 312-853 ms, max gap 76-587 ms under ~22.7 load). The
load-independent columns match both prior closes row-for-row: `recompose`
is 0 on every row (three closes running); `full-update` follows the exact
same `2/2/2/1/1/1/1/1` pattern wave-2 close recorded. `mounts`/`nodes`
match wave-4 close's own numbers exactly (media 173-175/115, notes
110/110) with NO further drift on the notes-switch-2nd row this time (that
row stayed at 110/110 here, versus wave-4's own 110→115 node growth) —
consistent with wave-4 close's own explanation that the growth there was
ordinary Media/Notes feature churn between wave-2 and wave-4 close, not a
one-way ratchet; this run simply landed on the lower end of that same
window. This wave's own diff (ingest state/controller extraction, plus
this close's doc-only edits) touches none of the probed Media/Notes
rail-switch path at all — the cleanest "must not move outside noise" case
of any close to date, and the numbers bear that out: settle/gap sit
comfortably inside the ORIGINAL wave-2 band despite three more waves of
unrelated feature work landing on `dev` in between.

#### Lessons

1. **The wave's own headline structural finding is two NEW numbered bypass
   shapes — §3's seventh and eighth — cross-referenced here, not
   relitigated.** The seventh (an `object.__new__`-bypassed test fixture's
   flat-name seed breaking the INSTANT a state PR's shim installs, not
   deferrable to cleanup like every prior shape) and the eighth (a moved
   body's bare module global patched at the OLD module path by a
   green-but-vacuous test) were found one per task, both by REVIEW rather
   than by either task's own full battery, both CRITICAL. That is two-for-
   two in one series — reinforcing wave-4's own lesson 2 ("review is a
   separate, mandatory gate, not a redundant check on a battery that
   already passed") a second time, with a second wave's worth of
   independent evidence for the same claim rather than a coincidence
   specific to skills' `focused` property.
2. **Count-accuracy discipline (wave-4's own lesson 1) had to be re-earned
   three separate times in this series too — the existence of a stated
   rule does not, by itself, prevent the failure the rule describes.**
   Task 1's own fixture-seeding count went through two corrections before
   landing (24 sites/4 files → 25/5 → the true 27/6). Task 2's own mover
   count churned across FIVE correction rounds as battery-found hazards
   shrank the cluster (78 → 63 → 62 → 59 → 58 → 57), and its own
   module-globals census undercounted twice in different ways within the
   SAME task — once by missing the two-argument `monkeypatch.setattr`
   shape entirely (3 real sites read as 0 before the pattern was widened)
   and again by stopping at "~20 hits/7 files" for `_sync_library_canvas`,
   a count task 3 corrected to the true 38 sites/10 files. Task 3's own
   delegator-census table had EVERY "mover caller" label wrong in its
   first draft (re-verified via `ast`-derived containing-method lookup
   across all 23 KEEP rows, not just the one review flagged), and that
   re-verification pass itself caught two further miscounts (`_apply_
   library_ingest_preflight_result` 11→10; `_library_ingest_registry`
   33→11, the latter a literal substring-contamination bug). Three tasks,
   three independent count-instability incidents, all in the ONE series
   that already had wave-4's own numbered lesson to guard against exactly
   this. **The generalization: re-stating the rule is not the fix — the
   fix that actually worked every time here was an INDEPENDENT
   re-derivation method (a repo-wide grep re-run, an `ast`-derived
   cross-check, a byte-for-byte diff) catching what a careful-sounding
   first draft missed, not a reviewer simply asking "are you sure."**
3. **A durable-artifact hygiene gap can hide behind its own stated rule for
   four consecutive waves before an audit catches it.** §10 states
   plainly: "every pure-move commit's hash is appended to
   `.git-blame-ignore-revs`, in the same PR that makes the move." This
   wave's own close-time audit of that file found ingest task 1's own
   pure-move commit (`12ba4fb13`, the state-PR extraction) was never
   added — and, widening the check, that EVERY state-PR commit since the
   conversations exemplar's own backfilled entry (export, collections,
   search+RAG, and skills alike) has the identical gap; only controller-
   move and cleanup commits were ever appended for those four series. Not
   one review across four waves caught it, because nothing in the
   recipe's own battery checks this file's completeness — it is pure
   documentation hygiene with no test surface. Fixed here for ingest
   task 1's own commit (this wave's own responsibility); the other four
   are named, not retroactively fixed (outside this task's scope), as a
   lead for whoever next touches this file. **The generalization: a
   "same-PR" rule with no automated check behind it survives only as long
   as everyone remembers it independently every time — worth a
   self-defending guard (a CI check comparing `git log` pure-move commits
   against this file's contents) the way the recompose census and the
   controller-file size ratchet already are, not asserted here as
   in-scope for this task, but recorded as the same category of gap §16
   lesson 5 already named for the wiring cluster constants.**
4. **This wave applied its OWN interruption-unsafe-baseline lesson
   preemptively, and it worked.** Task 1 found the hard way (§3, §7) that
   an in-place `git checkout <base> -- tldw_chatbook Tests` overlay is
   fragile against a session interruption mid-sweep. Every sweep in this
   series after that incident — task 2's, task 3's, and this close's own
   whole-wave sweep above — used an isolated `git worktree add` plus its
   own venv from the start, and none of them needed a redo. A lesson
   recorded from a real incident, then actually applied by every
   subsequent task in the SAME series without having to be reminded, is
   the recipe doing its job — recorded here as confirmation, not a new
   finding.

## 21. The prompts series, as landed — the sixth rehearsal, and the largest single move of this program

Wave-6 Tasks 1–3 (`.superpowers/sdd/2026-09-05-library-decomposition-wave6-prompts`)
extracted the Prompts subsystem. At 161 method candidates and 46 prompt-named
`__init__` attributes it is the biggest cluster the recipe has processed —
skills' 127/36 was the prior high — and the first to need a delegator-prune
census that a plain reference count gets WRONG (see the `on_<Message>`
finding below).

### Fields/methods moved, per task

| Task | PR | What moved | Screen delta |
|---|---|---|---|
| 1 | State | 43 of 46 prompt-named `__init__` attributes → `LibraryPromptsState`; a THREE-way prefix split (31 `_library_prompt_*` singular + 11 `_library_prompts_*` plural + 1 bare `_selected_prompt_id`), resolved by one `prompt_state_shim_attr()` rather than two independent frozensets (the skills precedent). The other 3 are WIRING — live `LibraryPromptHistoryController`/`LibraryPromptBrowseController`/`LibraryPromptCollectionsController` instances — and stayed plain screen attributes. 4 fields keep their ORIGINAL `__init__` line (`reader_preferences`, `reader_layout`, `reader_persistence_locks`, `editor_mode`) because `self._prompts_state` must be constructed before the shared reader-preference tuple-unpack | 41393 → 41359 lines, 1321 methods (unchanged) |
| 2 | Controller | 139 of 161 "prompt"-named method candidates → `LibraryPromptsController` (44 `@on` + 6 `on_<message>` naming-convention + 1 `action_*` + 1 `@staticmethod` + 87 plain; 22 exclusions: 14 unbound-fake-self, 3 instance-attribute-monkeypatch, 2 screen-identity, 2 module-globals-coupling, 1 merely-delegate-to-existing-controller `@property`). Single controller, not split — an import-graph component analysis over all 161 candidates found one connected component of 145 names, so no clean seam existed | 41359 → 37722 lines, 1321 methods (unchanged: 139 `FunctionDef`s out, 139 delegators in). Controller born-governed at 4956, review fix round → 4991 |
| 3 | Cleanup | Shim block (43 properties, three prefixes) deleted; 128 screen-side literal `self.<flat>` retargets to `self._prompts_state.<field>` plus 4 dynamic-dispatch string retargets and 1 `getattr` RECEIVER fix; 465 test-side attribute-path retargets across 11 files spanning `Tests/UI` and `Tests/ProductionApp`, plus 13 `SimpleNamespace` fixture restructurings (27 flat kwargs → nested `_prompts_state=SimpleNamespace(...)`) and one fake-harness CLASS-ATTRIBUTE restructuring covering 5 more construction sites; 39 of 139 screen delegators pruned; 25 dead imports removed (5 further candidates SAVED by the `_SURFACE` check); the 4 stale `test_library_modal_dismissal.py` inventory rows repointed at the controller behind a new `_OwnerScope`; 10 stale docstring/comment corrections across the screen (3), the controller (2) and the state module (5) | 37722 → 37574 lines, 1321 → 1282 methods (39 fewer `FunctionDef`s — exactly the 39 pruned delegators). Controller: 4991 → 4998 (comment-only) |

**Pin trajectory** (`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
in `Tests/Architecture/test_screen_size_ratchet.py`):
`41393/1321 → 41359/1321 → 37722/1321 → 37574/1282` (final).

**Controller-file governance pin** (§17): `library_prompts_controller.py`
born-governed the moment it existed (Task 2): `4956 → 4991` (Task 2's own
review fix round) `→ 4998` (this cleanup task, comment-only).

### Delegator census — 100 KEEP, 39 PRUNED (~28%), and a genuinely new hazard

Of the 139 moved names, **51 KEEP unconditionally** per the transform
whitelist (§4) — but the whitelist needed a THIRD member this series, not
the two prior series used:

- **44 `@on` handlers** and **1 `action_*`** — the established two.
- **6 `on_<Message>` NAME-dispatched handlers** (`on_prompt_block_editor_
  back_requested` and its five siblings). **This is the new one.** Textual
  dispatches these purely by name: `Message.handler_name` is
  `f"on_{name}"` (`textual/message.py`), and `MessagePump._get_dispatch_
  methods` walks `self.__class__.__mro__` looking that name up
  (`textual/message_pump.py`). They carry no decorator, so a
  decorator-driven whitelist misses them; and no code anywhere spells
  `screen.on_prompt_block_editor_back_requested`, so a **reference-count
  census reports them as zero-referenced and marks all six PRUNE**. Deleting
  them would have silently unhooked `LibraryScreen` from six messages.
  **The rule this generalises to: before pruning any moved name matching
  `on_[a-z]`, check it against Textual's name-based dispatch, not just
  against the reference census.** (The 6 do appear to have "code
  references" in a naive census — but every hit is a `def` of the SAME name
  in a DIFFERENT class, which is the opposite of a caller: **8 such `def`s
  in all** — 2 in `Widgets/Library/library_prompts_canvas.py` (only
  `block_field_changed` and `block_action_requested`) and one for EACH of
  the six in a `Tests/UI/test_prompt_block_editor.py` harness. *(Corrected
  at the wave-6 close, re-derived by `grep -rn "def on_prompt_block_
  editor_"` over `tldw_chatbook/` + `Tests/` + `Docs/` + `Helper_Scripts/`
  + `scripts/`: this section's first draft said "4 more in a … harness",
  undercounting the harness by 2 — the harness carries 7, of which 6 match
  the moved names and the 7th, `apply_requested`, matches the one
  `on_prompt_block_editor_*` handler that is an EXCLUSION and never
  moved.)* That coincidence is also why the deletion would have stayed
  green — the canvas keeps handling its own two one level down.)

Of the remaining 88 candidates (1 staticmethod + 87 plain), **49 have a
genuine external caller** and **39 have none**. Every verdict was derived
from a tokenize-based census (NAME tokens only, so a docstring mention
cannot masquerade as a call) over `tldw_chatbook/` + all of `Tests/`,
excluding only the controller module and each name's own delegator body,
then re-checked against a broader "name appears anywhere on the line" pass.
The broad pass is what earns its keep: it caught three names
(`_arm_library_prompt_editor`, `_restore_library_prompts_focus`,
`_sync_library_prompt_memberships`) whose only callers pass them as **bare
callables** — `self.call_after_refresh(self._arm_library_prompt_editor)`,
`sync_memberships=lambda: self._sync_library_prompt_memberships`, and one
passed as a positional argument on its own line — which a `<name>\s*\(`
call-shaped regex scores as zero, marking all three PRUNE. That is the
skills series' own lesson-2 sanity check (§19) firing for real rather than
passing vacuously.

**A second novel census finding: an uncollected-but-executable script under
`Docs/`.** `_library_prompt_can_update_original` has zero references in
`tldw_chatbook/` and `Tests/` — but
`Docs/superpowers/reviews/evidence/task-22033/task22033_live_matrix_runner.py:262`
calls `screen._library_prompt_can_update_original()` on a real screen.
`pyproject.toml` sets `testpaths = ["Tests"]`, so that file is never
collected, and prior series established that frozen `Docs/` evidence
scripts are NOT retargeted (`task23019_scenarios.py` still spells
`screen._library_skill_editor_state` two waves after the skills cleanup).
Nonetheless the prune rule is "zero references **anywhere in the repo**",
and this is a reference in executable form, so the delegator was KEPT — the
first time a `Docs/` hit has changed a prune verdict. Prune fraction: 39 of
139 = **28.06%**, at the high end of the recorded range. Sorted by the
computed fraction, not by eye — an earlier draft of this line put prompts
AFTER search+RAG on a `~28%` vs `~29%` rounding, which reverses the true
order: export 1/22 = 4.55% < ingest 6/56 = 10.71% < skills 16/86 = 18.60%
< collections 14/64 = 21.88% < **prompts 39/139 = 28.06%** < search+RAG
12/42 = 28.57% < conversations 18/61 = 29.51%.

### Dynamic-dispatch census — four spellings, six real screen-side sites

Run over the 43 state fields AND the 139 mover names, in all four spellings
(attribute, quoted-string, bare-assignment/kwarg, patch-target table). The
**mover** names came back clean in every non-attribute spelling: the only
string-literal occurrences anywhere are the wiring test's own pin tuple and
the four `test_library_modal_dismissal.py` inventory rows (below). The
patch-target-table shape (`(target, "<name>", key)` rows consumed by a
distant `monkeypatch.setattr`, whose exemplar is
`Tests/UI/test_library_shell.py:5146`) DOES name a prompt method there —
but `_request_library_prompts_browse`, which is one of the 22 EXCLUSIONS,
so it never moved and needed nothing.

The **field** names produced six screen-side sites, each matching a shape a
prior series already solved:

- **The shared reader-preference dispatcher, twice**
  (`_replace_library_reader_preference` / `_persist_library_reader_
  preference`): the `"prompts"` entry's value string became
  `"_prompts_state.reader_preferences"`. Zero consumption-site changes —
  both dicts are already read through `operator.attrgetter` and written
  through `_assign_library_reader_preferences_attribute`, the dotted-path
  helper the conversations exemplar added and every series since has
  reused rather than re-derived.
- **The choice-strip visibility/canvas-kind pair**: same treatment,
  `"_prompts_state.sort_choices_visible"`, sitting directly beside the
  skills entry that already reads that way.
- **One `getattr(self, "_library_prompts_view", "list")`** needing the
  skills series' RECEIVER fix, not a string swap:
  `getattr(self._prompts_state, "view", "list")`. Task 1's census recorded
  four such sites; task 2's move took three of them into the controller,
  where they resolve through the controller's own permanent shim loop and
  correctly stay untouched.
- **`on_screen_suspend`'s flat-name timer loop**: the prompts
  search-debounce timer left the string tuple for its own explicit
  `_prompts_state` block — the ingest path-debounce timer's precedent,
  three lines above it in the same method, complete with the same
  "a `getattr` for the old flat name silently returns None" warning. Both
  of that loop's test pins (`test_library_screen_reuse.py`) got the
  matching treatment.

Test-side: `test_library_adaptive_reader_closeout.py`'s
`DESTINATION_CONTRACT` needed its two `"prompts"` strings dotted (the sixth
consecutive subsystem to need that one entry), and
`test_screen_navigation.py:3253`'s `setattr(screen, "_library_prompts_
view", "list")` needed the receiver fix.

### Import verification — the `_SURFACE` check saved 5 of 30

30 imports were left dead in `library_screen.py` by this wave, derived as a
DIFFERENCE (AST-unused set at the wave-6 start commit vs. at this commit)
rather than as an absolute unused-name list — 38 further names are unused at
BOTH ends and belong to prior series or to `__future__`/`_SURFACE`
bookkeeping, and deleting those would have been out of scope.

Each of the 30 was then checked individually against
`Tests/Architecture/test_library_support_layer_surface.py`'s `_SURFACE`
re-export contract, and **5 came back pinned**:
`LIBRARY_PROMPT_DIRTY_VETO_COPY`, `LIBRARY_PROMPT_SAVE_STATUS_COPY`,
`_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP`,
`_LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS`,
`_LIBRARY_PROMPT_WRITE_WORKER_GROUPS`. All five stayed;
`test_screen_still_re_exports_every_moved_name` asserts the MODULE surface,
not live usage. **This is the shape's third series in a row and the largest
hit yet** (ingest saved 1 of 9; this saved 5 of 30) — and it was nearly
missed a second way: the first `_SURFACE` check was a case-sensitive grep
for `prompt`, which returned zero because all five names spell it `PROMPT`.
**Do the `_SURFACE` check by exact-name lookup, not by a lowercase
subsystem-word grep.** The remaining 25 were deleted, each independently
confirmed already re-imported and live inside the controller (23 of them)
or genuinely unnecessary anywhere (`PromptSourceCapabilities`,
`local_prompt_capabilities` — both now live only in
`library_prompts_state.py`, which owns the field whose default calls them).

### The deferred modal inventory — repointed, proven by construction

`Tests/UI/test_library_modal_dismissal.py` maintains a hand-declared
`(file, class, presenter, modal-type)` inventory and rediscovers it by
AST-parsing only the files named in `_SUPPORTED_OWNER_SCOPES`. Task 2's move
made four rows stale. Repointing them required adding an
`_OwnerScope("tldw_chatbook/UI/Library_Modules/library_prompts_controller.
py", "LibraryPromptsController")` FIRST — without it a repointed edge is
never discovered and the bidirectional assertion fails the other way.

That test **cannot be proven green end-to-end**: it is pre-RED at this
task's parent for an unrelated skills-era failure
(`unresolved modal constructor … LibraryScreen._present_library_skills_
import_choice_if_needed`), which aborts discovery before the comparison.
Measured identically on both trees: **1 failed / 169 passed**. The retarget
was therefore proven by CONSTRUCTION instead — running the file's own
`_discover_library_modal_edges` against the new scope alone and comparing
to the four declared rows: 4 discovered, 4 declared, zero undeclared, zero
missing, exact match including each modal's concrete type. The fifth prompt
row, `_stage_library_prompt_for_console`, is one of the 22 exclusions, is
still screen-resident, and was correctly left alone. The cross-wave repair
(the skills blocker, plus `handle_library_ingest_browse` and the two skill-
trust passphrase presenters, all equally stale delegators) is filed
separately at wave close — it is not prompts-only work.

### Wiring test finalization

`Tests/Architecture/test_library_prompts_wiring.py`:
`test_state_object_fields_match_the_shim_surface` narrowed to
`test_state_object_declares_the_censused_field_count` (the half that never
depended on the screen);
`test_every_shim_reads_and_writes_its_own_state_field` **re-aimed at the
controller** rather than deleted, since `LibraryPromptsController`'s own
permanent shim loop is generated by the identical `dataclasses.fields` loop
and carries the identical closure-binding trap, and the surviving
`test_prompts_controller_exposes_every_state_field` is exactly the weaker
`isinstance(..., property)` check it exists to strengthen (a
deviation from the prior series' delete-it precedent, taken deliberately —
prior series had no equivalent second test to lose);
`test_the_screen_no_longer_carries_a_prompt_state_shim` added, asserting
ABSENCE; `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` filled with the 39
names, with both `test_screen_delegates_prompt_handlers` and
`test_prompts_cluster_staticmethods_forward_to_the_controller_class`
already wired (by task 2) to skip them and assert genuine absence instead.
13 tests, all green.

### Wave-6 close

Wave-6 (`.superpowers/sdd/2026-09-05-library-decomposition-wave6-prompts`,
branch `refactor/library-decomp-wave6-prompts`) scoped itself to prompts
alone — at 161 method candidates and 46 prompt-named `__init__` attributes,
the largest single cluster of the program, and the plan's own judgement that
"one subsystem in flight at a time" beats rushing media and notes alongside
it. The prompts series (Tasks 1-3, above) is complete. This section is Task
4's own wave-level pin-trajectory re-derivation, verification battery, sweep
evidence and lessons. Media and notes remain, in that order (§8).

#### Pin trajectory — full wave-6 chain

Re-derived by `git show <commit>:<path>` reads of the `_BUDGETS` row in
`Tests/Architecture/test_screen_size_ratchet.py` and the
`library_prompts_controller.py` row in
`Tests/Architecture/test_library_modules_size_ratchet.py` at every commit in
the wave — not carried over from any report — in true chronological order
(`git log` lists newest-first; this table is oldest-first):

| Task | PR | Commit | Screen `_BUDGETS` after | Controller pin after |
|---|---|---|---|---|
| — | (wave-6 start) | `e5e03846a` | 41393 / 1321 | — (file does not exist yet) |
| 1 | Prompts state (RED — wiring + characterization pins; screen untouched) | `a41cbe8c6` | 41393 / 1321 (unchanged) | — |
| 1 | Prompts state (GREEN — 43 of 46 fields → `LibraryPromptsState`) | `f59db7c94` | 41359 / 1321 | — |
| 1 | (blame-ignore, no functional change) | `1e3e0b6ef` | 41359 / 1321 (unchanged) | — |
| 1 | Fix round 1 — coverage-classification correction, wiring pin hardened (series complete) | `4ada1c55d` | 41359 / 1321 (unchanged) | — |
| 2 | Prompts controller (RED — full-cluster wiring pins only) | `8ae30f490` | 41359 / 1321 (unchanged) | — |
| 2 | Prompts controller (GREEN, born-governed — 139 of 161 movers) | `d0ec95b16` | 37722 / 1321 | **4956 (born)** |
| 2 | (blame-ignore, no functional change) | `926b0431c` | 37722 / 1321 (unchanged) | 4956 (unchanged) |
| 2 | Fix round 1 — one stale line reference, comment-only | `52268ea42` | 37722 / 1321 (unchanged) | 4956 (unchanged) |
| 2 | Fix round 2 — census-count corrections incl. the stale ratchet-comment narrative line (series complete) | `31b7d4754` | 37722 / 1321 (unchanged) | 4991 |
| 3 | Prompts cleanup (GREEN — shim deleted, 39 delegators pruned) | `6fd2b753a` | 37574 / 1282 | 4998 |
| 3 | (blame-ignore, no functional change) | `d838b7d63` | 37574 / 1282 (unchanged) | 4998 (unchanged) |
| 3 | Fix round 1 — doc accuracy only (series complete) | `d010b65a1` | 37574 / 1282 (unchanged) | 4998 (unchanged) |
| 4 | Stale-prose sweep (comment-only, line-neutral) | `33cffc4a4` | **37574 / 1282 (wave-6 final)** | **4998 (wave-6 final)** |

*(Six further commits in the wave are SDD ledger/report writes only —
`471813f21`, `9a9fa02a2`, `bcf0631f7`, `770a47736`, `3eba0592f` and this
close's own `3cf89dbe9` backlog filing — and are omitted from the table
because they move neither pin. Each was read anyway to confirm that: the
`git show` sweep above covers all 19 commits in `e5e03846a..HEAD` plus the
base itself (20 refs), not just the 14 rows listed.)*

Full chain, screen: `41393/1321 → 41359/1321 → 37722/1321 → 37574/1282`
(final). Full chain, controller: born `4956` at its very first commit (Task
2's own GREEN) `→ 4991` (Task 2's own fix round 2 — comment growth from the
census documentation added alongside the count corrections) `→ 4998` (Task
3, comment-only: two controller docstring corrections).

**Net wave-6 shrink: 3,819 screen lines and 39 methods** — the largest
single-wave reduction this recipe has recorded, by a wide margin and from a
single subsystem: wave-2 (export + collections, two subsystems) took 1,554,
wave-3 (search+RAG) 1,037, wave-4 (skills) 2,070, wave-5 (ingest) 1,480.
Prompts alone is **1.85× the prior best**. The 39 methods are exactly the 39
pruned delegators; a pure move is always net-zero on the screen's method
count, and only a delegator prune moves it.

Task 4's own fresh `_measure()` call (each ratchet file's own semantics — an
`ast`-walked line count plus a `LibraryScreen` method count, not `wc -l`)
gives **37574 lines / 1282 methods** for the screen and **4998 lines** for
the controller: EXACT matches to both recorded pins, zero drift, nothing to
lower. This close's own commits touch neither governed file (the
stale-prose sweep's six edits are all line-neutral and land in
`Tests/`, `tldw_chatbook/Library/`, `tldw_chatbook/Widgets/Library/` and
`tldw_chatbook/UI/Console_Modules/`), so no same-commit re-pin was needed.

#### Verification battery (Task 4, this close)

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, `-p no:randomly`; `timeout` is unavailable in this
environment, so long runs are bounded with `perl -e 'alarm N; exec @ARGV'`.
Machine load at battery time: **16.90, 17.43, 15.99** (`uptime`) — a
substantially BUSIER machine than wave-5 close's own ~3.0, worth naming
alongside the numbers per wave-4 close's own lesson 3.

- **Fresh `_measure()` on both ratchet files**: 37574/1282 (screen), 4998
  (controller) — exact match to both recorded pins, zero drift.
- **All SEVEN wiring suites** (`test_library_collections_wiring.py`,
  `test_library_conversations_wiring.py`, `test_library_export_wiring.py`,
  `test_library_ingest_wiring.py`, `test_library_prompts_wiring.py`,
  `test_library_search_rag_wiring.py`, `test_library_skills_wiring.py`)
  **+ the support-layer surface suite + both size ratchets + the recompose
  census + the pre-import closure + the `_ui_ready` module census + all
  FIVE characterization files + the modal-dismissal file + the screen-reuse
  file, one combined invocation**: **301 passed, 4 failed** in 253.50s.
  Every red is on §7's documented list as updated by this close: the two
  `chat_screen.py` ratchet rows, the `library_media_browse_controller.py`
  controller-ratchet row (added to §7 by this close — see below), and the
  modal-dismissal discovery blocker. **Zero `library_screen.py`- or
  `library_prompts_controller.py`-scoped failures.** The `_ui_ready` census
  passed on this run; it is a flapping guard and §7 now says so.
- **preflight**: `./scripts/preflight.sh` — all six derived-artifact checks
  pass (CSS bundle + 9 per-screen stylesheets, profile-owned-path census
  48/18/46, production diagnostic inventory 581 owners, backlog task-id
  sweep across 3,317 task files including this close's own two new filings,
  chachanotes table allowlist 105/105, index plan pins 270/270/57).

#### Full sequential xdist paired-baseline sweep — whole-wave span

Branch = wave-6 tip plus this close's own commits (a comment-only
line-neutral prose sweep, two backlog filings, this recipe's own edits and
the close report). Baseline = an ISOLATED `git worktree add` at the wave-6
START commit (`e5e03846a`) with its own `uv venv`/`uv pip install -e
".[dev]"`, verified to resolve its OWN tree
(`tldw_chatbook.__file__` prints under `.worktrees/wave6-close-baseline/`).
No stash overlay was used anywhere in this task — per the ruling this close
itself wrote into §3. Run SEQUENTIALLY, never concurrently, per §7's own
"concurrent runs amplify flakiness" lesson.

**Scope, and a deliberate deviation from this task's own brief, recorded
with the measurement that forced it.** The brief called for "the full
`Tests/UI` + `Tests/Library` + `Tests/Architecture` roots (the established
sweep scope)". That is not this recipe's established scope, and the two
differ by more than 5×: measured at this close, those three roots collect
**23,898** tests where §7's actual established net,
`Tests/UI -k "library"`, collects **4,477**. At the ~25 min/side the
established net costs on this machine, the literal reading is ~4.5 hours of
paired sweeping, 15,742 of whose tests are non-Library `Tests/UI` cases with
no relationship to a wave whose entire remaining diff is eight comment
lines — and §7 already rules the exhaustive net "CI's job, not this
recipe's per-task evidence requirement". Resolved as **two paired sweeps**
that together cover the brief's intent at a defensible cost: **Sweep A** =
`Tests/UI -k "library"` (4,477 — the established, directly wave-comparable
net) and **Sweep B** = the full `Tests/Library` + `Tests/Architecture` roots
with no filter (3,679 — the widening the brief was actually after, and the
roots §7's own list has historically under-covered).

**Machine load, recorded at the start of each half** (per wave-4 close's own
lesson 3 — a probe or sweep number without its load average is not
comparable to any other run):

| sweep half | started | load average (1 / 5 / 15 min) | wall |
|---|---|---|---|
| A — branch | 21:52 | 12.44 / 14.72 / 15.28 | 30:35 |
| A — baseline | 22:22 | 6.92 / 9.03 / 13.65 | 24:13 |
| B — branch | 22:47 | 5.61 / 7.40 / 9.37 | 3:27 |
| B — baseline | 22:57 | 3.41 / 6.15 / 8.30 | 3:21 |

**Sweep A's two halves were NOT run under comparable load**, and this is
stated up front rather than buried: the branch half ran at roughly twice the
baseline half's load average and took 26% longer in wall clock (30:35 vs
24:13). That biases the raw counts toward MORE branch failures and therefore
toward MORE spurious branch-unique names — the conservative direction for a
regression hunt, but it means the count comparison is worth nothing on its
own here and every branch-unique name had to be resolved individually. It
was.

| | Failed | Passed | Skipped | Wall |
|---|---|---|---|---|
| A — branch (`3cf89dbe9` + this close's doc edits) | 373 (+1 error) | 4104 | — | 1835.90s |
| A — baseline (`e5e03846a`) | 370 | 4103 | — | 1453.35s |
| B — branch | 22 | 3652 | 5 | 207.75s |
| B — baseline | 22 | 3637 | 5 | 201.10s |

**Both collection-count asymmetries reconcile exactly**, which is the first
thing to check before reading any failure diff:

* Sweep A: branch collects 4 more than baseline — exactly the 4 tests in
  `Tests/UI/test_library_prompts_characterization.py`, added by task 1.
* Sweep B: branch collects 15 more — exactly the 13 tests in
  `Tests/Architecture/test_library_prompts_wiring.py` plus the 2
  parametrizations the controller ratchet's own `sorted(_BUDGETS)` gains
  from the new `library_prompts_controller.py` row.

**Sweep B: 22 failed on BOTH trees, name-set IDENTICAL — zero branch-unique,
zero baseline-unique.** All 22 are documented: the 18 `Tests/Architecture/`
rows task 1 proved name-for-name at the wave-6 base (the 15 inventory reds
plus the 3 ratchet rows), and 4 in `Tests/Library/` — two already on the list
above (`test_library_export_roundtrip.py::test_library_export_success_
records_a_durable_receipt_with_the_real_path`, wave-2 task 3;
`test_ingest_preflight_egress.py::test_the_probe_reports_a_redirect_as_an_
answered_status_not_an_error`, wave-5 task 3) and two NEW names, added to the
documented list above by this close on the strength of failing byte-
identically on a tree that does not contain the change:
`Tests/Library/test_agent_chunk_student_story.py::test_student_story_chapter_
notes_from_stored_chunks_only` and `Tests/Library/test_library_skills_state.py::
test_shadow_name_set_stays_in_sync_with_real_sources`.

**Sweep A: 363 shared, 10 branch-unique, 7 baseline-unique** (baseline-unique
not investigated, per this section's own established precedent — though it is
worth noting that three of the seven are Prompts-named
(`test_library_prompt_undo_refreshes_applied_page_and_preserves_basket`,
`test_boot_with_prompts_default_tab_lands_on_library_with_prompts_row_
selected`, `test_prompts_route_lands_on_library_with_prompts_row_selected`),
i.e. the subsystem this wave moved failed MORE often on the tree that does
not contain the move).

**Not one of the 10 branch-unique names is a Prompts test.** All 10 were
resolved, none dismissed by name:

1. **Combined single-process re-run of all 10 together: 10 passed** (14.45s).
2. **True isolation, one process per node, on the BRANCH: 9 of 10 passed.**
3. **True isolation, same 10, on the isolated BASELINE worktree: 10 of 10
   passed.**

Five of the ten are already documented above (`test_focus_traversal_builds_
zero_bodies_for_pass_through_rows`, wave-5 task 3; `test_library_media_
durable_mutation_gates_and_refreshes_applied_scope[True]`, wave-5 task 2;
`test_library_media_initial_error_is_unknown_and_retry_is_unique`, wave-5
task 1; both `test_library_starter_production_geometry_and_focus_order`
parametrizations, wave-4 close). The other five passed on every re-run in
both modes and are ordinary xdist noise, not added to the list, per this
section's own precedent for a name that passes on rerun.

**The one that reproduced, and the process lesson it produced.**
`Tests/UI/test_library_media_reader_traversal_t22207.py::test_focus_
traversal_builds_zero_bodies_for_pass_through_rows` failed its isolated
branch run while passing its isolated baseline run — a one-sample asymmetry
pointing the wrong way, and the only result in this whole close that could
have been a regression. It was NOT dismissed on the citation that it is
already a documented flaky name. Instead: **matched batches of 10 isolated
single-node runs on each tree, back to back on a quiet machine (load ~4-6)
— branch 5 failed / 10, baseline 4 failed / 10.** Indistinguishable. The
failure itself is a wait-loop `TimeoutError` from
`test_library_media_reader_flow.py:141` ("Detail call for backing id 23 did
not start"), not a behavioural assertion, in a Media-reader test this wave's
diff does not touch.

**The lesson worth carrying forward is about the BASELINE half of that
check, not the branch half:** the baseline passed this name 6 times out of 6
across the first-pass isolated runs, which reads exactly like "green on the
baseline, red on the branch" — and it took a deliberate 10-run batch to show
it fails 4 times in 10 there too. **For a timing-sensitive test, a single
clean isolated run on the baseline is not a disposition; it is one sample of
a coin flip.** Wave-5 close made the converse observation (a combined-run
pass overturned by a true-isolation rerun); this is the same failure one
level further in, and the fix is the same — batch it.

**Zero real regressions across the whole wave-6 span.**

#### Probe run

Run after both sweeps completed, on a quiet machine (load average 6.61,
5.86, 7.17 at the first run; 5.26, 5.67, 6.99 at the last).

```
perl -e 'alarm 300; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py
```

**This close ran the probe as a genuine same-machine BEFORE/AFTER pair — the
first close in this program to do so.** §9 has asked for a before/after pair
around a controller move since the recipe was written; every close to date
(§16, §19, §20) instead compared a single branch run against wave-2 close's
*recorded band*, which cannot separate a code effect from a machine effect.
The isolated baseline worktree built for the sweep made the honest version
free: the identical script, from the identical commit
(`Helper_Scripts/library_click_probe.py` is byte-identical across the wave —
`git diff e5e03846a HEAD --` on it is empty), on the same machine, minutes
apart.

**Round 1 — branch first, then baseline:**

| interaction | settle (ms) br / base | max gap (ms) br / base | recompose | full-update | mounts br / base | nodes br / base |
|---|---|---|---|---|---|---|
| media (switch-in) | 564 / 503 | 183 / 158 | 0 / 0 | 2 / 2 | 177 / 177 | 119 / 119 |
| media (re-click same) | 386 / 312 | 75 / 51 | 0 / 0 | 2 / 2 | 89 / 85 | 119 / 119 |
| media (re-click same, 2nd) | 388 / 317 | 79 / 51 | 0 / 0 | 2 / 2 | 89 / 89 | 119 / 119 |
| notes (switch) | 425 / 354 | 183 / 147 | 0 / 0 | 1 / 1 | 114 / 114 | 114 / 114 |
| notes (re-click same) | 284 / 245 | 63 / 47 | 0 / 0 | 1 / 1 | 38 / 38 | 114 / 114 |
| media (switch-back) | 439 / 404 | 113 / 100 | 0 / 0 | 1 / 1 | 179 / 175 | 119 / 119 |
| notes (switch, 2nd) | 418 / 367 | 189 / 163 | 0 / 0 | 1 / 1 | 114 / 114 | 114 / 114 |
| media (switch-back, 2nd) | 455 / 401 | 111 / 91 | 0 / 0 | 1 / 1 | 179 / 179 | 119 / 119 |

Round 1 shows the branch slower on **all sixteen** wall-clock measurements
(settle +35 to +74 ms, max gap +13 to +36 ms). A uniform one-directional
shift across every row is not something to wave away as noise, so it was
tested rather than argued.

**Round 2 — the SAME two probes with the tree order REVERSED (baseline
first, branch second):**

| interaction | settle (ms) base / br | max gap (ms) base / br |
|---|---|---|
| media (switch-in) | 508 / **493** | 173 / **130** |
| media (re-click same) | 315 / **310** | 50 / **48** |
| media (re-click same, 2nd) | 311 / **310** | 50 / 51 |
| notes (switch) | 355 / **354** | 148 / **147** |
| notes (re-click same) | 246 / **245** | 46 / **42** |
| media (switch-back) | 403 / 427 | 101 / 113 |
| notes (switch, 2nd) | 390 / 396 | 175 / 179 |
| media (switch-back, 2nd) | 401 / 403 | 90 / 95 |

**The penalty followed the RUN ORDER, not the tree.** Whichever tree probed
FIRST paid it; in round 2 that was the baseline, and the branch came back at
or below it on five of eight rows and within a few ms on the rest. The round-1
gap was a first-run warm-up artifact (its own `waitms` column says so: the
delta sat almost entirely in IDLE-WAIT, +25 to +50 ms per row, while
`busyms` — actual main-thread work — was unchanged). **A single before/after
probe pair cannot distinguish a code effect from an order effect; swapping
the order is a two-minute experiment that can.**

The load-independent columns are the real verdict and they are exact:
`recompose` is **0 on every row of all four runs**; `full-update` follows the
identical `2/2/2/1/1/1/1/1` pattern on both trees and matches every close
back to wave-2; and **`nodes` is identical row-for-row between the two trees**
(119 media / 114 notes). That last one also retires a loose end from prior
closes: the mount/node counts have drifted upward across waves (wave-4 and
wave-5 both recorded media 173-175 / 115 and notes 110 / 110), and each close
attributed the drift to ordinary Media/Notes feature churn on `dev` without
being able to prove it. Here the wave-6 BASE tree already reads 177-179 / 119
and 114 / 114 — **the drift is demonstrably present before this wave's first
commit**, so the attribution is now measured rather than assumed. `mounts`
wobbles by 4 run-to-run *within* each tree (the baseline's own two identical
re-click rows read 85 then 89), which is why the 85-vs-89 cells above are not
a difference.

Round-2 branch figures sit inside wave-2 close's recorded band on every row
(settle 245-493 against 264-485, with the low end FASTER than anything wave-2
recorded; max gap 42-179 against 54-195). This wave's diff does touch the
probed path — five of the six probed-path methods changed — but every change
was proven mechanical: `ast.unparse`-ing `compose_content`,
`_select_library_rail_row_after_source_admission`,
`_toggle_library_media_reader_pane`, `restore_state` and
`_persist_library_reader_preference` at both ends and applying the 43-name
flat→`_prompts_state.<field>` rewrite to the old text makes all five
**byte-identical**. The sixth, `on_screen_suspend`, carries the one real
structural change (the prompts debounce timer lifted out of the flat-name
string loop into its own explicit block, because a plain `getattr` cannot
follow a dotted path) and is not on the probe's click path at all — the probe
clicks rail modes and never suspends the screen. That is a stronger version
of the honesty wave-4 close had to add in review, derived mechanically
instead of by reading diffs.

#### Lessons

1. **The wave's headline structural finding is a THIRD member of a
   whitelist that had been stable for five series — and the census that
   should have caught it reported the exact OPPOSITE of the truth, in both
   of its obvious forms.** The mechanism is in §4 and is not relitigated
   here; what generalises is the epistemics. The six
   `on_prompt_block_editor_*` handlers were marked PRUNE on a *genuine*
   zero-reference count — no source line anywhere spells
   `screen.on_prompt_block_editor_back_requested`, so the census was not
   buggy, it was answering the wrong question. And the obvious widening
   ("does this name appear anywhere in the repo?") would have marked them
   KEEP for entirely the wrong reason: all 8 hits are `def`s of the same
   name in a DIFFERENT class (the canvas widget's own two, plus one per
   moved name in a test harness), which is the opposite of a caller. A
   narrow census and
   its natural widening produced a wrong answer and a right-for-the-wrong-
   reason answer respectively. **The generalization: when a name is
   resolved by a FRAMEWORK rather than by a call site, no refinement of a
   reference census fixes it — you have to model the framework's own
   dispatch. Read `textual/message_pump.py`, not more of `Tests/`.**
2. **Number-verification discipline failed for a third consecutive wave —
   but the failures MIGRATED, from the counts themselves to the satellite
   copies of a number that nothing automated ever reads.** Wave-4's lesson
   1 and wave-5's lesson 2 both already state the rule, and this wave's
   reviewers plainly HAD internalised it: task 2's own errata table
   re-derived ten separate disputed figures from scratch (E1-E10), and
   task 3's report opens with a self-corrected count. What still got
   through were three copies with no reader: a ratchet comment's
   *narrative* line (`test_screen_size_ratchet.py:503`, "37718", shipped
   eighteen lines above a row pinned at 37722, caught by review); a
   *commit message* (`6fd2b753a`: "37576 … -146" for a commit whose row,
   comment, report and live `_measure()` all say 37574/−148 — shipped, and
   deliberately unamendable because `.git-blame-ignore-revs` holds that
   literal hash); and a *line range* inside a claim that explicitly said
   "re-verified" (`:518-524` overriding a correct `:520-526`). **The
   generalization, now written into §6: "verify every number" is understood
   — what is not is that a number has COPIES, and the copies with no
   automated reader are precisely the ones that survive review. Enumerate
   them (row, arithmetic, prose, commit message) and grep the old value
   before committing.** A corollary the third incident earns on its own: a
   claim carrying the word "re-verified" that overrides a correct value is
   worse than no claim, because it costs two independent readers a trip to
   the live file to undo.
3. **An unwritten process exception survived two more tasks, and was
   killed by a MEASUREMENT rather than by an argument.** §3's
   isolated-baseline rule said "expected to run unattended for more than a
   couple of minutes", and two tasks read the gap as a licence for short
   foreground stash overlays, three times between them; all three
   completed intact, and task 2 then hit the exact session-limit
   interruption the rule exists to prevent (losing nothing only because
   its long sweeps were already isolated). Two waves of "it's only two minutes" was never going to
   settle itself. What settled it in one step was pricing the alternative
   instead of debating it: uv's own share of provisioning an isolated
   baseline is **2.2s**, the tree costs 535 MB, and — decisively — a task
   creates ONE and reuses it, so the marginal cost after the first check is
   zero. The carve-out was buying nothing. **The generalization: when a
   process rule has an unwritten exception that keeps getting used, measure
   the cost the exception exists to avoid before writing a bound for it.
   Here the measurement did not justify a bound; it removed the need for
   one.**
4. **A guard that raises during DISCOVERY proves nothing about anything —
   and in a battery it is indistinguishable from a guard that failed one
   assertion.** `Tests/UI/test_library_modal_dismissal.py` reports "1
   failed / 169 passed", which reads like one bad row out of 170 checks.
   It is not: the single failure IS the bidirectional inventory
   comparison, aborted by an unresolved modal constructor before it
   compares anything, so the file's actual coverage of every subsystem's
   modal rows has been ZERO since the skills series. Five consecutive
   waves read that line, correctly concluded "pre-existing, not mine", and
   moved on. Nobody asked what the failure was BLOCKING — which is why
   wave-6 task 3 had to prove its own four repointed rows by construction
   instead. **The generalization: when triaging a pre-existing red,
   separate "an assertion that fails" from "a discovery or collection step
   that raises". The second silently voids every assertion downstream of
   it, and the passing count printed beside it is meaningless.** Filed as
   TASK-31815.
5. **A shared documented-reds list only works if WRITING to it is part of
   dismissing a red.** Three consecutive wave-6 tasks dismissed
   `test_library_modules_size_ratchet.py`'s `library_media_browse_
   controller.py` row as "the documented pre-existing row" — while §7
   documented only the two `chat_screen.py` rows and had never mentioned
   that one. The habit of CHECKING the list had been learned across five
   waves; the habit of ADDING to it had not, for a row uncontroversial
   enough that nobody looked it up. Fixed at this close, along with the
   wave's own 17 prompt-file reds and its 15 `(file, line)`-keyed
   Architecture inventory reds (both proven name-for-name at the wave-6
   base). **The generalization: the phrase "the documented X" in a report
   is worth grepping the document for. The cost of it being false is that
   the next series re-derives the same red from scratch — exactly what §7
   exists to prevent.**
6. **Two of six stale-prose defects lived outside every file scope any
   task in the series had reason to open — and the census that found them
   first returned a confident, wrong ZERO.** The four in-scope defects were
   disclosed by task 3's own review; the other two sat in
   `tldw_chatbook/Library/library_shell_state.py` and its own test, files
   no prompts task ever needed to touch. That asymmetry is the argument for
   §3's new deleted-FIELD-name prose census being a standing wave-close
   step rather than a per-task file-scoped sweep. **And the instrument
   itself nearly hid them:** the census's first run reported 0 hits repo-
   wide, because its skip-list (`.worktrees`, `.venv`, …) was matched
   against absolute `path.parts` and this worktree's own path CONTAINS
   `.worktrees` — so it skipped every file in the repository and reported a
   clean sweep over nothing, which looks identical to a clean sweep over
   3,000 files. It was caught by asserting a known-positive (a file that
   demonstrably contains one of the names) rather than by reading the
   output. **The generalization: a census that returns zero must be proven
   capable of returning non-zero before the zero is believed — the same
   "prove the instrument fires" discipline this recipe already demands of
   mutation evidence, applied to a grep.**

7. **The probe has been run as a single-tree measurement for four closes,
   and running it as a real PAIR — same machine, minutes apart — changed
   what it could conclude.** §9 has specified a before/after pair since the
   recipe was written; §16, §19 and §20 each ran ONE probe and compared it
   to wave-2 close's *recorded band*, a comparison that cannot separate a
   code effect from a machine effect, which is exactly why wave-4 close had
   to spend a paragraph arguing that its elevated numbers were ambient
   load. The isolated baseline worktree that the sweep already requires
   makes the honest version free: two runs, two minutes. **And the pair
   immediately produced a result no single run could have**: round 1 showed
   the branch slower on all sixteen wall-clock measurements — a uniform,
   one-directional shift that looks like a regression — and reversing the
   tree ORDER in round 2 moved the penalty onto the baseline. It was a
   first-run warm-up artifact, and its own `waitms` column said so (the
   delta sat in IDLE-WAIT, not in `busyms`). **The generalization: when a
   paired measurement shows a consistent directional difference, re-run it
   with the order swapped before believing the direction.** The pairing
   also retired a loose end three closes carried on assumption — the
   mount/node growth each of them attributed to "ordinary Media/Notes churn
   on dev" is now MEASURED, because the wave-6 base tree already reads the
   higher counts.
## 22. Media analysis and Reader interactions (tasks 31648, 31649)

The rebased review tree had grown to 42,558 Library lines and 1,319 screen
methods, beyond the existing 41,574/1,302 ceilings. Two separately attributable
controllers move coherent ownership out of the screen:

- `library_media_analysis_controller.py`: Reader analysis generation/saving,
  bulk partition/overwrite/retry, and receipt state. The in-flight flag remains
  shared shell state because Import also reads it; per-item Import outcomes
  remain on Import. Provider dispatch and provider-reason rendering retain their
  screen seams.
- `library_media_reader_controller.py`: content search, display-state memo,
  reading-position coalescing/draining, and read-later operations. The Reader
  session identity and detail remain screen-provided named ports; the controller
  owns the transient search, progress, and memo fields.

Moved bodies keep their original operations and names. Explicit keyword wiring
binds sibling callbacks and shell reads at call time; only real framework state
is read through the screen handle. The stable app identity is documented as the
existing snapshot exception. No DOM ids, nesting, or CSS moved.

Compatibility state declarations use `ControllerState`, the exact read/write
descriptor shape already used by Console. Each declaration names one controller
and one field; it installs no catch-all proxy. This avoids inflating the screen
method census with duplicated getter/setter bodies while preserving assignment
behavior. Private method callers move to their owner; DOM handlers remain screen
entry points. The shared canvas checkbox updater still calls
`_library_media_analyze_reason`, so that named screen delegator is retained.

The new controllers are born governed at their measured sizes. The screen is
41,325 lines / 1,301 methods after the extraction and private-call cleanup, a
reduction of 1,233 lines / 18 methods from the actual rebased starting point.
No existing ceiling increases. New ADR required: no; this directly applies the
approved screen decomposition design and `DESIGN.md` section 7.

The separate task 31658 follow-up fixes a pre-existing Media entry-focus race
exposed by the combined characterization run. A facet settlement queued an
entry-focus callback with guard cleanup, then a page settlement replaced that
callback with ordinary control focus before recomposition. The latest-wins
callback queue therefore dropped the cleanup as well as the obsolete intent;
the lingering guard suppressed real arrow selection indefinitely. Three focused
regressions fail before the fix and cover accepted replacement, suppressed
replacement, and an initially suppressed entry sync. Each new browse sync now
owns its current guard, releasing it when synchronization is suppressed. No stale
callback is replayed. The screen measures 41,324 lines / 1,301 methods afterward.

## 21. Keep controller execution payload behind construction (task 31720)

Controller extraction alone does not reduce route-discovery import cost if the
screen still imports every implementation at module scope. The Notes sync
controller imported its execution runtime, executor, coordinator and support
modules during the whole-registry preimport pass, even before Library opened.
Moving its existing import block into `LibraryScreen.__init__` preserves both
original classes, all constructor ports and the app-owned runtime lifecycle.
There is no new lazy proxy or lifecycle owner.

Fresh-process Library marginal payload fell from 188 modules / 146,804 lines to
178 modules / 130,426 lines: 10 modules / 16,378 lines deferred. Screen size stays
41,324 lines / 1,301 methods. The isolated packaging regression first proves
the execution closure absent after app, Chat and Library route imports, then
constructs a real screen and checks exact controller and runtime identity.
This applies existing ADR-097; global budget tightening and snapshot refresh
must use the final combined census, because concurrent route changes affect
the aggregate and marginal attribution is route-order dependent.

## 22. Retained action policy and blocking snapshot adapters (tasks 31667, 31668)

The destination tests exposed two runtime defects after minimal timeout fixtures
were given their required `app_config`. Conversation snapshot listing lacked the
worker-isolation option already used by Notes and Media: a blocking async adapter
took 0.2075 seconds against the unchanged 0.05-second timeout check. Supplying that
same option restored the deadline; a thread-identity regression also checks that
all three source families preserve their result records and counts off the UI
thread.

Separately, retained rail synchronization updated error text but left the
workspace handoff button's initial empty-source tooltip behind. The policy itself
was already correct. Its pure derivation now lives in workspace display state,
with the existing screen adapter used by both initial composition and the three
retained rail-sync callsites. Only the derived blocked/tooltip value crosses into
the rail; it patches the existing button without composing temporary widgets or
replacing DOM. A regression checks available, failure, custom-policy and restored
availability states while retaining exact button identity. The original failure
and custom-tooltip assertions remain unchanged; the screen stays within its
41,325-line / 1,301-method ceilings (measured 41,324 / 1,301 after both fixes).

## 23. Evidence settlement is not layout settlement (task 31682)

Both production-CSS Starter geometry sizes reached settled empty evidence while
the replacement rail button still had a zero-sized region. The test now waits
for the current widgets at its existing required selectors to reach the
compositor before reading geometry. Every original screen-bound, visibility,
keyboard-order and painted-copy assertion remains in place. No production timing,
DOM or CSS changed; 21 Starter/graduation checks and an independent two-size
geometry rerun pass.

## 24. Re-query test controls after yielding to the Pilot (task 31689)

The shared selector helper captured a widget, awaited `pilot.pause()`, then
returned the old owner even when a same-ID replacement had mounted during that
await. Prompt history retry could therefore press an orphaned button. A real
remove/mount regression proves the stale return before the fix; the helper now
requires the post-pause query to resolve to the same attached owner before it
returns. `is_mounted` alone is insufficient: Textual keeps it true after removal.

The stale-search caret test also began its newer action while loading recompose
had cleared focus. It now gates that real recompose and waits for the current
filter's loading-phase caret before editing. Final stale-request, caret, retry,
collapse and no-change assertions remain unchanged. A conflict restore waits for
detail adoption and notification, not merely the earlier SQLite version write.

## 25. Source admission must not repaint the retained authority (task 31724)

Files admission called Notes navigation invalidation with its default rendering
enabled. That reached the shared canvas synchronizer and recomposed the outgoing
Database Work pane before it was hidden, replacing the editor, its undo history
and the remembered focus target. A refresh-stack probe located this exact call;
the regression now counts outgoing Work-pane recompositions and checks original
editor identity immediately after Files opens, before any return.

The source handoff uses the existing `render=False` invalidation option. Pending
navigation still becomes stale, while both retained authorities keep their own
widgets. Eight identity, typing, focus and guarded-return cases pass. Responsive
scroll receipts are independently verified; this change does not claim to fix
their outstanding offset mismatches.

## 26. Finish private ownership and keep assembly explicit (tasks 31734, 31735)

The review rebase restored Library to 41,651 lines / 1,309 methods, over both
existing ceilings. Eight Conversations methods were already pure forwards into
their controller. Their remaining screen callers now address that owner directly;
the sole direct UI-test invocation was retargeted with every behavioral assertion
unchanged. The existing pruned-name guard checks all 17 removed private names are
absent while their controller implementations remain present. No event handler,
public action, descriptor semantics, or controller body changed.

The next move puts the contiguous six-controller assembly in
`UI/Library_Modules/wiring.py`, following DESIGN section 7's Console precedent.
All six constructor call ASTs match the pre-move source after only normalizing
`self` to `screen`; order, named keyword lambdas, initialization position, and
canonical controller identities stay unchanged. Real-screen characterization
checks replacement state and sibling lookups remain late-bound. The new helper
is imported during construction, so relocation does not add a module to the
guarded preimport phase; the original canonical controller imports retain their
existing phase and compatibility names.

The resulting screen is 41,303 lines / 1,301 methods; assembly is governed at
338 lines. Statement-level diagnostic comparison is unchanged: 100 screen
statements before/after, zero in the new assembly. Unrelated Notes/Skills/RAG
failures found in wider verification are kept separate from the pure move.
