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

Each of the eleven subsystems below (see §7) ships as a small series of
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
    print(f"{f}: non-conversation users={non_conv or 'NONE'}")
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
  (`@on`/`action_*`/anything a test reaches directly) — the `@on(...)`
  decorator line stays on the screen, copied verbatim.

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

## 7. Subsystem order (spec, "Order of work")

Sequenced cold-to-hot so the conversations exemplar never fights rebases,
and hot subsystems migrate in short, fast series once the recipe above is
rehearsed. Churn = commits touching `library_screen.py` in the trailing 30
days whose subjects name the subsystem (measured 2026-09-01):

| Order | Subsystem | Churn | Notes |
|---|---|---|---|
| 1 | **conversations** (exemplar) | 10 | 68 methods, 19 fields; lowest cross-coupling (3 notes refs plus shared fields handled by §2's rule) |
| 2 | export | 3 | recipe rehearsal |
| 2 | collections | 6 | recipe rehearsal |
| 2 | search | 6 | recipe rehearsal |
| 3 | skills | 15 | |
| 3 | RAG / onboarding plumbing | 16 | |
| 3 | ingest | 23 | |
| 4 | prompts | 41 | |
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
freeze (§8's probe is that fix's before/after acceptance evidence). See the
spec's "Phase C — region ownership" section; out of scope for this recipe's
pure-move PRs.

## 8. Probe usage — before/after evidence

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

## 9. `.git-blame-ignore-revs` — one-time setup and the per-PR rule

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
