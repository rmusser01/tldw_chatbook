# Wave-6 prompts — final fix wave + dev reconciliation

**Status: COMPLETE.** The two on-branch review conditions are fixed in one
commit, and the `origin/dev` reconciliation merge landed as a true merge
commit with the specified semantic port plus the ratchet governance the
review required.

| | |
|---|---|
| Fix commit | `ad2d6dac7` |
| Merge commit | `441cc0e1c` (parents `ad2d6dac7`, `dfe45fbe6`) |
| Merge-base | `7aa048790` |
| dev commits merged | 266 |
| Working tree | clean except `progress.md`, which was already modified at session start and was deliberately never staged |

Both commits were verified to exclude `progress.md`.

---

## Step 1 — on-branch fixes (`ad2d6dac7`)

### I1 — the guard gap

`Tests/Packaging/test_library_preimport_closure.py` did not name the prompts
controller in its deferred-suffix tuple, so the wave's own central claim —
`library_prompts_controller` stays off the registry's import-only walk — was
ungoverned by the test written to govern exactly that.

Two edits:

```python
     for suffix in (
-        "collections", "conversation_reader", "note_import",
-        "notes_sync", "rag_search", "skills",
+        "collections", "conversation_reader", "ingest", "note_import",
+        "notes_sync", "prompts", "rag_search", "skills",
     )
```

```python
 assert LibraryScreen._restore_library_skills_scope({}).query == ""
+assert LibraryScreen._restore_library_prompts_scope({}).query == ""
```

The static probe sits **after** the `sys.modules` intersection assert, matching
the existing collections/skills probes — calling it necessarily imports the
controller, so probing before the assert would defeat the assert. Verified
against the live file before writing: the staticmethod is
`LibraryScreen._restore_library_prompts_scope(state: Mapping[str, Any]) ->
PromptBrowseScope` at `library_screen.py:9927-9933`, whose body is a
function-local `from ..Library_Modules.library_prompts_controller import
LibraryPromptsController` — so the probe does exercise the delegator's local
import. `PromptBrowseScope`'s `query` defaults to `""`
(`Library/library_prompts_state.py:83`), so the assertion is the skills probe's
exact shape.

**"ingest" was added on evidence, not on the review's say-so.** The review
claimed it was also missing; before adding it I ran the test's own route walk
and dumped `sys.modules`:

```
LOADED CONTROLLERS (after every route.load_screen_class()):
  library_collections_capture_controller, library_conversations_controller,
  library_export_controller, library_media_browse_controller,
  library_media_trash_browse_controller, library_prompt_browse_controller,
  library_skills_browse_controller
```

`library_ingest_controller` is absent — its only import site in the screen is
function-local at `library_screen.py:2044`. Laziness holds, so the row is real
governance rather than a pin on a lie. (`library_skill_import_controller` is
also lazy and also unlisted, but it is outside this brief's scope and was left
alone rather than silently widened.)

Re-run after the merge: still green, so dev's 266 commits did not make either
controller eager.

### M1 — recipe §7 vs §3

§3 now states the isolated-worktree rule **with no exception** and says
outright that "`git stash -u` and `git checkout <base> -- <paths>` are no
longer baseline methods at any duration" (`library-decomposition-recipe.md`
:396-403). §7 still instructed readers to use `git stash -u`. Two prescriptive
spots reconciled:

- the paired-sweep step 2 (`:799-802`) → "against the pristine merge base in
  §3's isolated baseline worktree (`git worktree add <scratch-path>
  <base-commit>` with its own `uv venv`)"
- the pre-existing-failures preamble (`:830`) → "(§3's isolated worktree at the
  pre-task tree; older entries below record the `git stash -u` method §3 has
  since retired)"

The other ~20 `stash -u` mentions in §7 were checked and **deliberately left**:
every one is a record of what a specific past task actually ran ("confirmed
identical on a `git stash -u` baseline", "reconfirmed pre-existing by every
task in this series via the same `git stash -u` method"). Rewriting those would
falsify the evidence trail. Only the two instructions were changed; the second
edit says so explicitly so a reader does not think the records are drift.

### M2 — task-31816 AC order

ACs ran `#1 #2 #3 #4 #6 #5`. The content order already reads correctly (name
the two residents, then remove the recipe entry), so the two labels were
renumbered in place rather than the paragraphs being reordered. Now sequential
`#1`…`#6`.

---

## Step 2 — the reconciliation merge (`441cc0e1c`)

`git fetch origin && git merge origin/dev`. Merge commit, never rebase or
squash — `.git-blame-ignore-revs` pins this series' literal SHAs. Two parents
confirmed on the resulting commit. dev had advanced past the review's snapshot
(`f32a16839..dfe45fbe6`) during the session; the merge is against `dfe45fbe6`.

### Conflict 1 — `apply_navigation_context` (the expected one)

`tldw_chatbook/UI/Screens/library_screen.py:11120`. Exactly as predicted: HEAD
carried the old 58-line body with this branch's retargeted gate line; dev's
`6b5285986` had replaced the whole body with a delegator.

**Resolution: dev's delegator, taken verbatim.**

```python
     def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
         """Admit route context through the Library-owned navigation controller."""
         self._navigation_controller.apply_navigation_context(context)
```

### The port

Verified live before writing (the brief's line 75 was accurate; it is line 75
on dev and became 75-78 after the comment):

`tldw_chatbook/UI/Library_Modules/library_navigation_controller.py`

```diff
-        if self.screen._library_prompts_mutation_in_flight:
+        # wave-6 (prompts) retarget: the flat `_library_prompts_mutation_in_
+        # flight` attribute this line read on dev was deleted by the prompts
+        # cleanup PR; the field now lives on the Prompts state object.
+        if self.screen._prompts_state.mutation_in_flight:
             return
```

Target verified rather than assumed: `mutation_in_flight: bool = False` is a
real field of `LibraryPromptsState`
(`UI/Library_Modules/library_prompts_state.py:272`), `_prompts_state` is
constructed in `LibraryScreen.__init__` at `library_screen.py:2200`, and the
same expression is already the screen's own spelling at 9 sites
(`:2319`, `:5470`, `:11062`, …). Left unported this was an `AttributeError` on
the first line of every Library deep link.

### Conflict 2 — `backlog/docs/lessons-testing-evidence.md`

Both sides appended new sections at the same offset — this branch's two
(zero-reference census vs Textual name dispatch; `_SURFACE` exact-name
checking) and dev's ten. Purely additive on both sides, verified by listing
the `##` headers of each side. Resolution: **kept both**, markers removed by
line number after asserting each was a marker line.

### Conflict 3 — `Docs/security/production-diagnostic-inventory.json`

`git checkout --theirs`, then the drift rows were **read before** regenerating:

```
summary: owner_files: 583 -> 584
owners:
  + library_prompts_controller.py [TASK-494] count=9
  ~ library_screen.py 104 -> 95  (-9 diagnostic call(s))
```

The script's own `--statements --since origin/dev` recovery was run to answer
the interpolation question the aggregate digest cannot. All nine statements
move with **identical digests** (`cc59466be0a8f19f`, `5e2f25c207d2fd04`,
`9174f1ce8d63f125`, `84f0027746ce9369`, `b9b9aa23176903d3`,
`edcbecd3396b8694`, `7923b031d5af0f8a`, `ff336b6ca3c5c552`,
`c97296a224d7d058`) — removed from the screen, added to the controller,
byte-for-byte, no rewording, no re-levelling, no new interpolation of user
content, secrets or paths. Only then `--write`. Re-verified: `584 owners, 1347
TASK-492, 30 TASK-31551, 7646 TASK-494, 12 sink files`.

*(A first `--statements` attempt used the script's suggested `git log -1` base,
which during a conflicted merge resolves to the branch's own last pin commit
and reported "nothing changed". The base was widened to `origin/dev`, which is
the revision dev's pin actually describes.)*

### The two auto-merged regions the review flagged (verified, not trusted)

git merged both without a conflict, which is exactly when a semantic loss goes
unnoticed. Both were read by hand:

- **Timer region.** `on_screen_suspend` keeps BOTH semantics: dev's flat
  `"_library_prompts_debounce_timer"` string-loop entry is gone (this branch
  removed it) and this branch's explicit `_prompts_state.debounce_timer` block
  is present at `:9525-9531`, while dev's `call_after_refresh(self._navigation_
  controller.present_pending_repair)` survives in `on_screen_resume` (`:9543`)
  and `on_mount` (`:9684`).
- **`__init__`.** Dev's 15-line `_navigation_controller` construction
  (`:2170-2184`) and this branch's `self._prompts_state = LibraryPromptsState()`
  (`:2200`) are both present.

---

## Census over the merged tree

All **43** deleted flat prompt field names were derived programmatically from
`LibraryPromptsState`'s dataclass fields through `prompt_state_shim_attr()`
(the single authoritative mapping) — not hand-listed. A whole-word regex over
every `.py` file under `tldw_chatbook/` + `Tests/` covers all four spellings at
once (attribute, quoted-string, bare-assignment, patch-target table). 474 hits
across 9 files. Non-`.py` files under both roots: zero hits.

| File | Hits | Disposition |
|---|---|---|
| `UI/Library_Modules/library_navigation_controller.py` | 1 | **RETARGETED** — dev's new gate, the only live reader dev's 266 commits added. See the port above. |
| `UI/Library_Modules/library_prompts_controller.py` | 453 | Correct — the controller's own generated flat-name shims, read by its byte-for-byte moved bodies. |
| `UI/Library_Modules/library_export_controller.py` | 3 | Correct — its own accessor-backed `_library_prompts_mutation_in_flight` property (`:545`), wired at `library_screen.py:2319` to `lambda: self._prompts_state.mutation_in_flight`. Pre-existing, reviewed. |
| `UI/Library_Modules/library_prompts_state.py` | 3 | Comments/docstrings. |
| `UI/Library_Modules/library_skills_controller.py` | 1 | Comment. |
| `UI/Library_Modules/library_skills_state.py` | 4 | Comments. |
| `UI/Screens/library_screen.py` | 1 | Comment recording the retarget. |
| `Tests/Architecture/test_library_prompts_wiring.py` | 4 | Assertions on `prompt_state_shim_attr()`'s mapping — testing the function, not a screen attribute. |
| `Tests/UI/test_library_prompts_canvas.py` | 3 | Comments/docstrings. |
| `Tests/UI/test_library_prompts_characterization.py` | 2 | Comments. |

After the port, zero live readers of any deleted flat name remain outside the
two files that legitimately own one.

**Extra check the brief did not ask for, run because dev's cleanup pruned 39 of
139 screen delegators and a dev-side caller of a pruned name would be invisible
to the flat-name census:** an AST walk collected every `self.screen.<attr>` in
dev's two new controllers (10 distinct attributes) and resolved each against
the merged `LibraryScreen`. All 10 resolve — the three that fail a class-level
`hasattr` (`_prompts_state`, `_library_selected_row_id`,
`_library_navigation_context_generation`) are instance attributes assigned in
`__init__` at `:2200`, `:2164` and `:2169`.

---

## Pins set in the merge commit

Every number below was measured with the owning file's **own** convention, on
the merged tree, immediately before writing.

### I2 — dev's two ungoverned controllers

`Tests/Architecture/test_library_modules_size_ratchet.py`, measured with that
file's `_measure` expression `len(path.read_text(encoding="utf-8")
.splitlines())` (cross-checked against `wc -l`; they agree, both files end in a
newline):

| Row | Pin |
|---|---|
| `library_character_repair_controller.py` | **502** |
| `library_navigation_controller.py` | **198** |

Both carry a comment recording that these are dev-side controllers governed at
the wave-6 merge because dev landed them without rows. The navigation row's
comment notes the 195→198 delta is this merge's own three-line port comment,
so the number is not mistaken for dev's.

This is not cosmetic: `test_every_controller_file_has_a_budget_row` — the
self-defending check this file exists for — is **RED on `origin/dev`** and
green here.

### §6 re-pin

`Tests/Architecture/test_screen_size_ratchet.py`, measured with that file's own
AST `_measure(rel_path, class_name)`:

| Row | Before | After |
|---|---|---|
| `library_screen.py` | `("LibraryScreen", 37574, 1282)` | `("LibraryScreen", 37537, 1282)` |

The −37 reconciles exactly, and reconciles to **dev's** delta rather than to an
estimate: dev moved this file 41393 → 41356 over the same range (−37), and the
branch contributed zero lines because the one conflict took dev's side
verbatim. Term by term off dev's own hunks: +15 `_navigation_controller`
construction, +1 and +1 for the two `call_after_refresh` dispatches, +1 blank
separator, −55 for `apply_navigation_context` becoming a delegator (63 → 8).
The method count is unchanged because dev moved bodies only — the AST method-
name set on `LibraryScreen` is identical at the merge-base and at `origin/dev`
(measured, both directions of the set difference empty).

The prompts controller row needed no change: `library_prompts_controller.py`
measures **4998**, its existing pin, because dev never touched it.

---

## Battery

All on the merged tree, `.venv/bin/python -m pytest … -p no:randomly`.

| Suite | Result |
|---|---|
| 7 wiring suites + support-layer surface | **55 passed, 0 failed** |
| recompose ratchet + preimport closure + ui_ready census + prompts characterization | **15 passed, 0 failed** |
| `Tests/UI/test_library_character_repair.py` (dev's) | **15 passed, 0 failed** |
| Both size ratchets | 39 passed, **3 failed** — all pre-existing (below) |
| `Tests/UI/test_screen_navigation.py` | 126 passed, 30 failed — all pre-existing (below) |
| `Tests/UI/test_library_prompts_canvas.py` | 320 passed, 20 failed — **zero unique to the merge** (below) |
| `./scripts/preflight.sh` | **all six checks passed** |

Notable: the `ui_ready` module census — the documented ±1-wobble flake that
task-31816 exists for — passed on the merged tree.

### Every red, proven

Baselines were isolated worktrees with their own `uv venv`, per the §3 rule
this same fix wave was reconciling §7 against. Both were verified to resolve
their **own** tree (`tldw_chatbook.__file__` under the scratch path) before any
number was taken — the editable-finder trap.

**1. `library_media_browse_controller.py` ceiling (410 vs pin 371).** Surfaced
by the merge, not caused by it. The file is 410 lines and the pin is 371 on
**both** sides, and the stale pin already predates the merge-base `7aa048790` —
so no Library task introduced it. Confirmed by running the ratchets in the
`origin/dev` worktree, where it fails identically. **Deliberately not
re-pinned**: this test's own guidance forbids raising a number to silence
creep, and doing it from a passing branch would launder dev's debt. Recorded in
recipe §7's documented-pre-existing list instead, with the numbers and the
reasoning, because that list exists to stop the next series rediscovering it.

**2–3. The two `chat_screen.py` ratchet rows.** Already in recipe §7's
documented list; reconfirmed red in the `origin/dev` worktree this round.

For the record, `origin/dev` fails **four** of these; the merged branch fails
three. The one that flipped to green is
`test_every_controller_file_has_a_budget_row`, fixed by I2.

**4. `test_screen_navigation.py`, 30 failures.** Matched-batch comparison
against the `origin/dev` worktree: branch 30 failed / 126 passed, baseline 31
failed / 126 passed — **29 shared, 1 unique to branch, 2 unique to baseline**.
The file is demonstrably flaky (two branch runs of the same command gave 31
then 30, with different membership).

The one branch-unique name,
`test_search_route_lands_on_library_rag_canvas`, is the one that exercises the
ported line, so it got the full 10-run matched-batch treatment rather than a
verdict:

| | fail rate, 10 isolated runs |
|---|---|
| merged branch | 10 / 10 |
| `origin/dev` baseline | 9 / 10 |

Identical failure text on both sides — `AssertionError: assert 'Screen' ==
'LibraryScreen'` (and a `'ChatScreen'` variant on the baseline, i.e. a
nondeterministic route-landing race, not an attribute error). Pre-existing on
dev. Had the port been wrong, this is the file that would have shown an
`AttributeError` on `_library_prompts_mutation_in_flight`; it does not, on
either side.

**5. `test_library_prompts_canvas.py`, 20 failures.** Not in the brief's
battery; run anyway because the port touches the prompts gate. Compared against
an isolated worktree at the **pre-merge branch tip** `ad2d6dac7` — the more
decisive comparison than dev, since it isolates the merge itself: pre-merge 21
failed / 320 passed, merged 20 failed / 320 passed, **20 shared, ZERO unique to
the merged branch**, one unique to pre-merge (noise). The merge introduced
nothing here.

---

## Concerns

1. **`library_media_browse_controller.py` is red and stays red**, by choice.
   It needs an owner on dev; re-pinning it here would hide dev-side creep
   behind a Library merge. Documented in recipe §7 with both numbers.
2. **`test_screen_navigation.py` is badly flaky** — ~30 of 156 failing on
   `origin/dev` itself, with churning membership. Nothing in this wave, but any
   future task that treats a single run of it as a signal will be misled.
3. **`library_skill_import_controller` is lazy and unlisted** in the preimport
   suffix tuple. Verified lazy today; left out because it is outside this
   brief. Same one-word fix as "prompts"/"ingest" whenever someone owns it.
4. **dev's `library_navigation_controller.py` now carries a wave-6-specific
   comment.** If dev's copy is ever edited concurrently, that three-line
   comment is the likely conflict site — noted so the next reconciliation
   recognises it.
