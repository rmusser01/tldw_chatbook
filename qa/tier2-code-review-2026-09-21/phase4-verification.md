# Phase 4 — verification log

Every P0, every P1, and every S-sized "helper exists, ignored" claim, reproduced or
demoted. Commands run in the worktree at `origin/dev` `3722a857480b94b30fd4755f3f8e3002bd163ec3`
with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12.11).

---

## S01-P1a — `Notes/file_notes_service.py:576-594 save_file` writes without `fsync`
**VERDICT: CONFIRMED.**

```
$ sed -n '574,596p' tldw_chatbook/Notes/file_notes_service.py
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(new_bytes)
                temporary.flush()            # <-- flush only; no os.fsync
                ...
            os.replace(temporary_path, path)

$ grep -c fsync tldw_chatbook/Notes/file_notes_service.py   -> 0
$ grep -c fsync tldw_chatbook/Notes/sync_paths.py           -> 12
$ grep -c fsync tldw_chatbook/Utils/atomic_file_ops.py      -> 3
```
The sibling module in the same package fsyncs 12 times and the shared helper fsyncs
before every rename. This path does neither. The crash window is not reproducible
read-only (see "Left UNVERIFIED"), but the *absence of the fsync* — which is the claim —
is verified, and POSIX `os.replace` semantics do the rest.

---

## S01-P1b — the reconciler's `stale_observation` gate is a tautology and can never fire
**VERDICT: CONFIRMED.**

```
$ grep -rn --include='*.py' 'ReconciliationInput(' tldw_chatbook
Notes/notes_sync_reconciler.py:195:   return f"ReconciliationInput(root_id={self.root_id!r}, <private>)"   # __repr__
Notes/notes_sync_runtime.py:1079:     request = ReconciliationInput(                                       # the ONLY construction

$ sed -n '1079,1090p' tldw_chatbook/Notes/notes_sync_runtime.py
        request = ReconciliationInput(
            ...
            observation_generation=max(
                (item.note_version for item in observed), default=0
            ),
            expected_generation=max(
                (item.note_version for item in observed), default=0
            ),

$ sed -n '700p' tldw_chatbook/Notes/notes_sync_reconciler.py
    if request.observation_generation != request.expected_generation:
```
One production construction site; both fields are the identical expression over the
identical list. The gate is `x != x`. The only tests that reach it
(`Tests/Notes/test_notes_sync_conflict_runtime.py:130,526,1182`) build the input by hand
with unequal generations — i.e. they pin a shape production never produces.

---

## S01-P2 — `_ProductionRuntimeAdapter._bundles` leak window and hard cap
**VERDICT: CONFIRMED** (promoted evidence; kept at P2 because recovery is a restart, not data loss).

```
$ grep -n '_bundles' tldw_chatbook/Notes/notes_sync_runtime.py
704:  self._bundles: dict[...] = {}
1092: if len(self._bundles) >= _OBSERVATION_BUNDLE_LIMIT:   # raise, no eviction
1094: self._bundles[token] = MappingProxyType(bundle)       # INSERT
1181/1267/1332/1356/1390: reads
1322: self._bundles.pop(observation_token, None)            # the ONLY removal

$ grep -n '_OBSERVATION_BUNDLE_LIMIT' tldw_chatbook/Notes/notes_sync_runtime.py
114: _OBSERVATION_BUNDLE_LIMIT = 8

$ sed -n '1135p' tldw_chatbook/Notes/notes_sync_runtime.py
        self._observation_reuse[root.root_id] = await asyncio.to_thread(build_reuse)
```
Insert at 1094 precedes the `await` at 1135. Cancellation or an exception in that window
leaves an unreferenced token behind; the only removal (1322) runs from `finally` blocks
guarded on values computed *after* `observe_root` returns. Eight such events and every
root raises `observation_capacity_exceeded` for the process lifetime.

---

## Repo-wide D4-1 — `_coerce_int` is a NEW 4-copy cluster created two days after the review that removed its sibling
**VERDICT: CONFIRMED.**

```
$ git log -1 --format='%ad %h %s' --date=short 0ea2906e99
2026-09-19 0ea2906e99 feat(llm): migrate deepseek and mistral onto the hosted_chat engine (TASK-32852)
   (baseline SHA d8fb4053f9 is dated 2026-09-17)

$ grep -P '_coerce_int@' qa/tier2-code-review-2026-09-21/candidates/dup_verbatim.tsv
4  57f0590b969f  _coerce_int@LLM_Calls/mistral.py:317  _coerce_int@LLM_Calls/openrouter.py:205
                 _coerce_int@LLM_Calls/groq.py:199     _coerce_int@LLM_Calls/deepseek.py:206

$ for f in mistral openrouter groq deepseek; do
    printf "%-12s coerce_bool_flag:%s _coerce_bool:%s _coerce_int:%s\n" $f \
      $(grep -c coerce_bool_flag  tldw_chatbook/LLM_Calls/$f.py) \
      $(grep -c 'def _coerce_bool' tldw_chatbook/LLM_Calls/$f.py) \
      $(grep -c 'def _coerce_int'  tldw_chatbook/LLM_Calls/$f.py); done
mistral      coerce_bool_flag:0 _coerce_bool:0 _coerce_int:1
openrouter   coerce_bool_flag:0 _coerce_bool:0 _coerce_int:1
groq         coerce_bool_flag:0 _coerce_bool:0 _coerce_int:1
deepseek     coerce_bool_flag:0 _coerce_bool:0 _coerce_int:1

$ grep -rn 'def coerce_int' tldw_chatbook/Utils/   -> (no match)
$ grep -rn 'def coerce_bool_flag' tldw_chatbook/Utils/Utils.py -> 618
```
`TASK-32808.4` (Done) landed `coerce_bool_flag` in `Utils/Utils.py` and removed `_coerce_bool`
from these four modules. No integer counterpart was added. Two days later a *consolidation*
PR (TASK-32852) shipped four byte-identical `_coerce_int` copies into the same four modules.
Nothing mechanical could object.

---

## Repo-wide D4-2 — `Utils/ui_helpers.py` and `Utils/pagination.py` (the two 0-importer helpers in the 2026-09-17 seed table) are gone
**VERDICT: CONFIRMED CLOSED.**

```
$ ls tldw_chatbook/Utils/ui_helpers.py tldw_chatbook/Utils/pagination.py
ls: ...: No such file or directory   (both)
$ git log --oneline -1 --diff-filter=D -- tldw_chatbook/Widgets/base_components.py
5f3adeca33 chore(css): delete remaining dead modules with their dedicated tests; ...
```
`Widgets/base_components.py` went with them. TASK-32807.6's sweep landed for these three.

---

## Repo-wide D4-3 — `_perform_safe_cancel` / `_initialize_schema` / `_get_connection`: census clusters RETIRED, not findings
**VERDICT: RETIRED with evidence. A consolidation here would be a regression.**

```
$ .venv/bin/python qa/tier2-code-review-2026-09-21/candidates/variants.py _perform_safe_cancel
### _perform_safe_cancel: 45 defs, 28 shapes
```
28 distinct shapes over 45 definitions: this is a template-method override against the
shared `request_safe_cancel` / `dismiss_safe_once` / `run_cancel_effect_once` base, with
each modal's own cancel semantics (dirty guards, in-flight mutations, partial results).
The 2026-09-17 seed table listed `_cancel`/`_cancel_safe`/`_perform_safe_cancel` at 24/15/39
as a duplication cluster — that reading was by **name**, not by body.

```
$ sed -n '809,816p' tldw_chatbook/DB/base_db.py
    @abstractmethod
    def _initialize_schema(self):
        """Initialize the database schema. Must be implemented by subclasses."""
```
The 16 `_initialize_schema` definitions (2,120 lines — the largest duplication mass by LOC
in the whole census) implement an abstract hook. Each is a different schema.

```
$ sed -n '818,825p' tldw_chatbook/DB/base_db.py
    def _get_connection(self) -> sqlite3.Connection:
        """... Can be overridden by subclasses for custom connection handling."""
        conn = connect_private_sqlite("db.base", self.db_path_str)
```
The 9 subclass overrides all call `super()._get_connection()` (so they keep
`connect_private_sqlite`) and their PRAGMA divergences carry per-store task references and
documented exceptions (`Scheduling/db/scheduled_tasks_db.py` names task-22224 and says
"Do NOT copy this pattern into a store that holds connections"). A grep for
`connect_private_sqlite` returning 0 in four of them is the rule-of-evidence-#1 trap: they
inherit it.

---

## Repo-wide D4-4 — `_maybe_await` sync-on-loop escalation does NOT reproduce on the persona path
**VERDICT: RETIRED for the path tested. The D4 cluster stands; the D1 escalation does not.**

The seed table asked whether the other 59 `_maybe_await` copies still carry the shape that
task-283 fixed in `Chat/chat_conversation_scope_service.py` (a sync sqlite call evaluated as
the *argument* to `_maybe_await`, so the deferral is a no-op and the query runs on the loop).

```
$ .venv/bin/python qa/.../variants.py _maybe_await
### _maybe_await: 65 defs, 4 distinct bodies, 3 distinct shapes  (59 byte-identical)

$ grep -rl --include='*_scope_service.py' 'to_thread' tldw_chatbook | wc -l   -> 9
$ ls tldw_chatbook/*/*_scope_service.py | wc -l                              -> 47
```
So 38 of 47 scope services lack the offload. Traced the largest candidate end to end:

- `Character_Chat/character_persona_scope_service.py:339 list_characters`
  → `await self._maybe_await(backend.list_characters(...))`
  → `local_character_persona_service.py:751 def list_characters` (sync)
  → `self._require_db().list_character_cards(...)` → sqlite. **Shape confirmed.**
- But: `grep -rn '\.list_characters(\|\.search_characters(' tldw_chatbook | grep -v Character_Chat/`
  → **no callers**. The method is not reached from any UI path.
- `list_persona_profiles`, which *is* reached (`UI/CCP_Modules/ccp_persona_handler.py:140`
  ← `UI/Screens/personas_screen.py:3935`, inside `@work(exclusive=True, group=...)`), resolves
  to `local_character_persona_service.py:1114`, which filters an **in-memory list**
  (`self._persona_profiles`) — no sqlite, no blocking I/O.
- The neighbouring UI path that *does* read the whole character library
  (`ccp_character_handler.py:399 refresh_character_list`) already wraps it:
  `self.character_list = await asyncio.to_thread(fetch_all_characters)` — TASK-1320.

The `to_thread`-vs-not census is a real signal, but the module-level regex I used to rank it
(`.execute(` present anywhere in the local service module) is a weak one: it scored
`character_persona` at "105 sync DB methods" when the reached method touches no DB at all.
No live sync-on-loop instance found through `_maybe_await`. `TASK-32804.12` ("close the
remaining synchronous-work-on-the-loop findings", To Do) owns the class if one is found.

---

## Baseline — `Audio/meeting_owner.py` F821 is annotation-only, not a crash
**VERDICT: CONFIRMED but DEMOTED to P3.**

```
$ .venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/Audio
tldw_chatbook/Audio/meeting_owner.py:657:41: F821 Undefined name `MeetingCapture`
tldw_chatbook/Audio/meeting_owner.py:738:38: F821 Undefined name `MeetingCapture`

$ .venv/bin/python -c "
import typing, tldw_chatbook.Audio.meeting_owner as m
try: typing.get_type_hints(m._default_dictation_factory); print('hints resolved OK')
except Exception as e: print('get_type_hints FAILS:', type(e).__name__, e)
print('module imports fine:', m.__name__)"
get_type_hints FAILS: NameError name 'MeetingCapture' is not defined
module imports fine: tldw_chatbook.Audio.meeting_owner
```
`from __future__ import annotations` is at line 15, so the annotations are strings and the
module imports and runs. The name is imported only inside a function body
(`meeting_owner.py:1184`). It is the only ruff fatal in 890k lines of Tier-2 code, and it is
a type-checking defect, not a crash. Fix: add a `TYPE_CHECKING` import.

---

## S05-P0 — Console builds NO Library tool provider on the default configuration
**VERDICT: CONFIRMED end to end. The review's headline finding.**

The reviewer reported it; I re-derived every link independently.

**1. The constructor does not accept the argument.**
```
$ .venv/bin/python -c "import inspect; from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService; print(inspect.signature(LocalLibraryToolService.__init__))"
(self, *, media_service=None, notes_service=None, prompt_service=None, skills_service=None,
 conversation_service=None, media_chunk_service=None, notes_user_id='local_library',
 notes_scope_service=None, policy_enforcer=None) -> None
   accepts collections_service: False      has **kwargs: False
```

**2. Constructing it the way the factory does raises.**
```
$ .venv/bin/python -c "... LocalLibraryToolService(..., collections_service=None, ...)"
TypeError: LocalLibraryToolService.__init__() got an unexpected keyword argument 'collections_service'
```

**3. The factory passes it.** `Chat/console_runtime.py:675` — `collections_service=getattr(app, "local_library_collections_service", None)`.

**4. The parameter was deleted 20 days before this review, and only one of the two copies was updated.**
```
$ git log -1 --format='%ad %h %s' --date=short -S'collections_service' -- tldw_chatbook/Library/local_library_tool_service.py
2026-09-01 5dd1077df6 feat(collections): retire generic containers from current surfaces

$ grep -n 'collections_service' tldw_chatbook/UI/Console_Modules/library_activity.py
(no output — the sibling factory WAS updated)
```

**5. The broken factory is the one that wins.** `Chat/console_runtime.py:3647` uses `kwargs.update(...)`, not
`setdefault`, so `library_provider_factory=partial(_library_provider_for_app, self._app)` **overwrites** the
screen's `library_provider_factory=self._library_activity.build_provider` (`UI/Screens/chat_screen.py:10161`) —
i.e. the maintained copy is discarded in favour of the stale one.

**6. The failing branch is the default, and there is no fallback.**
```
$ grep -n 'direct_library_tools' tldw_chatbook/UI/Screens/settings_library_rag_defaults.py
94:    direct_library_tools: bool = True
126:        raw = get_cli_setting("console", "direct_library_tools", True)
```
`_library_provider_for_app` returns `LibraryRagToolProvider` only in the `if not …direct_library_tools:` branch.
With the default `True` it falls through to the `LocalLibraryToolService(...)` that raises — so the RAG provider is
never reached either.

**7. The exception is swallowed into a warning.** `Chat/console_chat_controller.py:27191-27198`:
```python
except Exception:  # noqa: BLE001 -- never block a send
    logger.opt(exception=True).warning(
        "library_provider_factory failed; running without Library tools")
```
Same shape at `:20767-20785`.

**Net:** on stock configuration every Console agent run silently loses all 18 `library_*` tools, and the only trace is
one WARNING line. No test catches it because
`Tests/integration/test_console_library_control_integration.py:177` and
`Tests/Chat/test_console_library_runtime_policy.py:101` both inject a **fake** `library_provider_factory` — the
"mocked tests never catch these" shape the brief names.

**Not verified:** that a live Console run produces no Library tool calls (the brief forbids running the app). Every
link in the static chain is verified. Settling command is in "Left UNVERIFIED".

---

## S18-P1a — `rich.markup.escape` survivors after TASK-32802.1
**VERDICT: CONFIRMED.**
```
$ .venv/bin/python -c "
from rich.markup import escape; from tldw_chatbook.Utils.input_validation import escape_markup
from textual.content import Content
n='[TODO] Q3 plan'
print('rich.escape ->', repr(Content.from_markup('Run '+escape(n)).plain))
print('repo escape ->', repr(Content.from_markup('Run '+escape_markup(n)).plain))"
rich.escape -> 'Run  Q3 plan'        <-- the token is DELETED
repo escape -> 'Run [TODO] Q3 plan'
```
Nine files still import `rich.markup.escape`. Classifying each by whether it is the deliberate
markup-off-sink case (which TASK-32802.4 tagged):

| File | `markup=False`/sink hits | TASK-32802 marker | Verdict |
|---|---:|---:|---|
| `UI/Library_Modules/library_export_controller.py` | 2 | 2 | deliberate |
| `UI/Screens/chat_screen.py` | 5 | 2 | deliberate |
| `Event_Handlers/STTS_Events/stts_events.py` | 2 | 2 | deliberate |
| `UI/Screens/evals_screen.py` | 42 | 0 | **broken escaper on markup-ON surfaces** |
| `UI/Screens/home_screen.py` | 0 | 0 | **broken** |
| `UI/Screens/study_screen.py` | 0 | 0 | **broken** |
| `Library/library_rag_state.py` | 0 | 0 | **broken** |
| `Chat/console_provider_gateway.py` | 0 | 0 | **broken** |
| `TTS/audio_cpp_supervisor.py` | 0 | 0 | **broken** |

TASK-32802.1's AC#2 is "the correct escaper is used at every site that currently calls `rich.markup.escape` for a
Textual surface". Six files still do. Two of the six (`Library/`, `TTS/`) are Tier-2 and outside what a Tier-1-scoped
sweep would have seen; four are Tier-1 and were missed.

---

## S11 — the three eval findings are real code on an UNREACHABLE path
**VERDICT: demoted P1 → P2 (lead).**
```
$ grep -rn --include='*.py' '\.run_evaluation(\|quick_eval\|ABTestOrchestrator\|handle_start_evaluation' tldw_chatbook
Evals/eval_orchestrator.py:407,424,434,637,1178   (definitions + internal)
Evals/ab_testing.py:182,192                        (ABTestRunner)
Evals/ab_testing.py:489,495                        (ABTestOrchestrator — defined, never constructed)
Evals/eval_runner.py:2720                          (definition)
   handle_start_evaluation: 0 hits anywhere in tldw_chatbook/ or Tests/
$ grep -n 'orchestrator' tldw_chatbook/UI/Screens/evals_screen.py | head
455:  orchestrator = getattr(app_instance, "evaluation_orchestrator", None)
456:  return getattr(orchestrator, "db", None)          <-- .db only
```
`EvaluationOrchestrator.run_evaluation` is reachable only from `ABTestRunner`, which only `ABTestOrchestrator`
constructs, which nothing constructs. The screen touches the orchestrator only for `.db`. The three defects
(1000-row false-failure, truncated export, truncated A/B) are correctly diagnosed and currently unreachable.

---

## S25 / repo-wide — `Utils/atomic_file_ops.py` is atomic but NOT durable, at all 17 adopter sites
**VERDICT: CONFIRMED. This inverts the obvious D4 recommendation and is a finding in its own right.**

The S25 reviewer retired the "7 `Backup_Recovery` files call `os.replace` without `Utils/atomic_file_ops.py`"
candidate on the grounds that non-adoption is *correct there*, because the shared helper is weaker. I checked.

```
$ grep -n 'fsync\|os.replace\|dir_fd\|O_DIRECTORY' tldw_chatbook/Utils/atomic_file_ops.py
103:            os.fsync(f.fileno())        # the FILE
110:            os.replace(temp_path, str(file_path))
178:            os.fsync(f.fileno())        # the FILE
184:        os.replace(temp_path, str(file_path))
284:        os.replace(temp_path, str(dst_path))
   -> no os.fsync of a parent-directory fd anywhere; no dir_fd; no O_DIRECTORY
```
versus the same repo's own recovery layer:
```
$ sed -n '12,25p' tldw_chatbook/Backup_Recovery/native_platform.py
def flush_file(fd: int) -> None:
    """Persist file contents and metadata using the host's native barrier."""
    if os.name == "nt": ... native_flush(fd); return
    os.fsync(fd)
    if platform.system() == "Darwin":
        import fcntl
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
```
Two gaps in the shared helper, both known to this repo because it fixed them elsewhere:
1. **No parent-directory fsync.** `os.replace` creates a new directory entry. Without an fsync on the parent
   directory fd, that entry can be lost on a crash even though the file's own data was fsynced — the rename is
   undone and the *old* file comes back (or nothing does). `Backup_Recovery` pairs every rename with
   `flush_directory(...)`; `atomic_file_ops` never does.
2. **Plain `os.fsync` on Darwin does not flush the drive write cache.** `F_FULLFSYNC` is the macOS barrier, which is
   exactly why `native_platform.flush_file` adds it. This user's platform is darwin.

So `atomic_file_ops` delivers atomicity (no torn file is ever visible) but not durability (the write can vanish).
At 17 importer modules. TASK-32808.5 (**Done**) drove adoption *toward* this helper.

**Consequence for the D4 recommendation:** these are two tiers, not one cluster. Do **not** consolidate
`Backup_Recovery`'s writes down onto `atomic_file_ops`. The repo-wide question is the reverse — whether
`atomic_file_ops` should gain a parent-directory fsync and the Darwin `F_FULLFSYNC` barrier for its 17 importers.
Filed as a repo-wide finding in `report.md`.

**Separately confirmed:** `Backup_Recovery/raw_participants.py` has **zero** fsync/flush calls
(`grep -c 'fsync\|flush_file\|flush_directory'` → `0`) while writing the user's live `config.toml`, settings TOML,
MCP `targets.json` and chat-dictionary files. That is the one place in the package with no barrier at all, and it is
outside TASK-32808.5's named sweep.

---

## S08-P0 — A sync while Schedules shows "This device" mirrors server-owned reminders and automation definitions under `owner_id="local"`, arming them for local execution alongside the server's own
**VERDICT: CONFIRMED.** The reviewer reproduced it against a real `ScheduledTasksDB`; I re-traced every link.

```
$ sed -n '265,271p' tldw_chatbook/Scheduling/services/sync_engine.py
    async def pull(self, owner_id: str | None = None) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id      # <-- the UI VIEW
        ...
        await self._pull_reminders(target_owner)

$ grep -n '_apply_pulled_reminders' tldw_chatbook/Scheduling/services/sync_engine.py
321:  pull_conflicts = self.db._apply_pulled_reminders(conn, target_owner, pulled_items, set())
514:  pull_conflicts = self.db._apply_pulled_reminders(conn, target_owner, pulled_items, pending_local_ids)
      -> server rows are STORED under target_owner

$ sed -n '5400,5409p' tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
            owner_id = service.owner_id          # <-- the view toggle, passed straight through
            outcome = await service.sync_now(owner_id)

$ sed -n '33,38p' tldw_chatbook/Scheduling/scheduler/queue.py
    return isinstance(owner_id, str) and owner_id.startswith(_SERVER_OWNER_PREFIX)
      -> every ADR-077 execution guard keys on the "server:" PREFIX
```
The pull is the one writer that never puts the prefix on, so a mirrored row with `owner_id="local"` passes
`is_server_scoped_owner` as *not* server-scoped and is armed by the local queue.

**The codebase already knows the rule and states it in a docstring two files away:**
```
$ sed -n '1162,1176p' tldw_chatbook/Scheduling/services/scheduling_service.py
    def _active_server_owner_id(self) -> str | None:
        """The single connected server's owner scope ("server:<id>") …
        the app's `active_server_id` property, NOT `self.owner_id` — `self.owner_id` is a
        UI-togglable VIEW (the user can flip to "This device" while a server stays connected),
        so it cannot stand in for "which server this session is connected to" …"""
```
`_active_server_owner_id` exists, computes the right value, and is not threaded into the pull.

ADR file: `backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md` (note the number collides with
`077-console-bounded-rail-section-scrolling.md` — cite by full filename). Its decision 1 is "execution follows
ownership" and its **rejected** alternative is named as "Both sides execute, dedupe at delivery … for agent work
this is double execution with nondeterministic ordering, and dedupe after the fact cannot un-run side effects."

**Consequence:** duplicate reminder notifications, and for `recurring_question` automation definitions a second
unattended LLM run per occurrence at the user's expense.

**Why no test caught it:** `Tests/Scheduling/test_sync_engine.py` constructs `SyncEngine(..., owner_id="server:1")`
in **every** pull test (lines 22, 45, 67, 82, 95, 110, 128, 144, 159, …); the single `owner_id="local"` case
(line 56) passes `server_client=None`, so it never pulls.

**Not verified:** that the server concurrently fires the same reminder (requires a live tldw_server). The client
side — mirroring under a local owner and arming it — is proven. Settling command is in "Left UNVERIFIED".

---

## S12-P1 — the live `_maybe_await` sync-on-loop instance (the one the 2026-09-17 seed table asked for)
**VERDICT: CONFIRMED.** This closes the open question in "Repo-wide D4-4" above.

The seed table asked whether the other 59 `_maybe_await` copies still carry the shape task-283 fixed. I traced the
`Character_Chat` scope service and found **no live instance** (the sqlite method has no callers; the reached method
is in-memory). S12 found one in `Media/`. Re-verified:

```
$ grep -c '_call_local_leaf(' tldw_chatbook/Media/media_reading_scope_service.py   -> 22  (21 calls + 1 def)
$ grep -c '_maybe_await('     tldw_chatbook/Media/media_reading_scope_service.py   -> 119 (117 calls + def + 1 inside the seam)
```
So **21 of 138 local-mode call sites use the threading seam; 117 bypass it.** The seam is not incidental — it was
built for exactly this and says so:
```
$ grep -n 'def _call_local_leaf' -A 12 tldw_chatbook/Media/media_reading_scope_service.py
155:  """Call one backend leaf, threading a confirmed local sync call (task-15467).
157:   ``run_worker(coroutine)`` does NOT leave the event loop, so every
158:   local-mode call that previously went straight through
159:   ``_maybe_await`` to a plain synchronous ``LocalMediaReadingService`` …
```

The worst bypass is not slow sqlite — it is a network round-trip:
```
media_reading_scope_service.py:1582   await self._maybe_await(service.save_reading_item(request_data))
                                              ^ the argument is evaluated FIRST, on the loop
local_media_reading_service.py:2091   def save_reading_item(          <-- plain def, synchronous
local_media_reading_service.py:2127     scraper = self.url_article_scraper or self._default_url_article_scraper
                        :2128           scraped = scraper(normalized_url, custom_cookies=None)
_default_url_article_scraper (~4276):   asyncio.get_running_loop()            # detects the loop
                                        with ThreadPoolExecutor(max_workers=1) as executor:
                                            …
                                            return future.result()            # NO timeout
```
The thread offload does not help: the *caller* blocks on `future.result()`, and the caller is the event-loop
thread. So a `save_reading_item` on a slow host freezes the UI for the full duration of the article fetch, with no
timeout bounding it.

`_maybe_await` cannot defer any of this — Python evaluates the argument before the call. That is the mechanism the
whole 65-copy cluster hides: the helper looks like a deferral and is a no-op.

**Consequence for the D4 recommendation:** consolidating `_maybe_await` to one home is correct but does **not** fix
this. The fix is adopting `_call_local_leaf` at the 117 sites (mechanical, same signature) plus a timeout on
`future.result()`. `TASK-32804.12` (To Do) owns the class; this is its highest-value member.

**Not verified:** that a shipped flow reaches `:1582` with an empty `content` (which is what triggers the scrape).
The chain is verified; the trigger condition is read-only.

---

## Repo-wide — 71 `logger.*` calls pass structured fields that reach no sink
**VERDICT: CONFIRMED and sized.** (Surfaced by S02; census run by the lead.)

```
$ grep -n 'format=' tldw_chatbook/Logging_Config.py
611:  format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
      -> the ONLY sink format in the file, and it has no {extra}

$ .venv/bin/python -c "loguru with format='{level} | {message}' vs '... || extra={extra}'"
INFO | Failed to enqueue
INFO | Failed to enqueue || extra={'note_id': 42, 'profile': 'p1'}
```
loguru puts `logger.info(msg, key=value)` kwargs into `record["extra"]`. With no `{extra}` in the format, every one
is discarded at every sink. AST census over all 6,637 `logger.<level>(` call sites, counting those that pass a kwarg
appearing in **no** `{placeholder}` of a literal message:

```
calls passing kwargs that appear in NO placeholder: 71
   UI 47 · Chunking 11 · Chat 6 · Notes 2 · Skills_Interop 1 · Web_Scraping 1
   Workspaces 1 · TTS 1 · RAG_Search 1
   e.g. Chat/console_agent_bridge.py:7626 ['conversation_id','session_id','outcome_status','final_text_len','step_count']
        Chat/console_chat_controller.py:26772 ['assistant_message_id','variant_mode','prepare_retry']
        Chat/Chat_Functions.py:1162,1184 ['exc_info']          <-- a stdlib-logging kwarg, inert in loguru
```
Two defect classes in one census: fields intended as structured context that vanish, and `exc_info=` (a stdlib
`logging` kwarg loguru ignores — it uses `logger.opt(exception=True)`). One-line sink fix for the first.

---

## S24 — two claims verified
**(a) Policy-gate gaps are real.**
```
$ for m in import_flashcards_tsv import_flashcards_json_file; do ... done
  import_flashcards_tsv              line 1489   _enforce_policy in next 30 lines: 0
  import_flashcards_json_file        line 1609   _enforce_policy in next 30 lines: 0
$ grep -n '_study_flashcard_import_action_id' tldw_chatbook/Study_Interop/study_scope_service.py
1263: …(normalized_mode, "import")      <-- gated sibling
1292: …(normalized_mode, "import")      <-- gated sibling
1474: …(normalized_mode, "preview")     <-- even the PREVIEW is gated
1643: …(normalized_mode, "import")      <-- gated sibling
```
Three siblings of the same operation gate; two do not. `runtime_policy/engine.py:22-30` is fail-closed (an
unregistered `action_id` denies), so a method that never calls the enforcer bypasses the boundary entirely rather
than getting a permissive default.

**(b) The `mode=None` default is inverted across all four `Interop` / `server_*` twin pairs.**
```
Outputs_Interop/outputs_scope_service.py               Backend.SERVER
Outputs/server_outputs_scope_service.py                Backend.LOCAL
Sharing_Interop/sharing_scope_service.py               Backend.SERVER
Sharing/server_sharing_scope_service.py                Backend.LOCAL
Web_Clipper_Interop/web_clipper_scope_service.py       Backend.SERVER
WebClipper/server_web_clipper_scope_service.py         Backend.LOCAL
Notifications/notifications_scope_service.py           Backend.SERVER
Notifications/server_notifications_scope_service.py    Backend.LOCAL
```
Four for four: **every `server_*_scope_service.py` defaults its server domain to LOCAL while its non-server twin
defaults to SERVER.** That reads as a copy-paste inversion, not four independent decisions — and it is exactly what
"41 byte-identical `_normalize_mode` bodies" hides, because the substitution that makes them identical substitutes
away the default. **A shared base cannot hard-code this default; it needs a class attribute, and the ADR has to say
which of each pair is right.**

---

## S03-P0 — `SimpleAudioPlayer.stop()` SIGKILLs the whole tldw_chatbook process group
**VERDICT: CONFIRMED (mechanism verified; trigger inferred).**

```
$ .venv/bin/python -c "import subprocess, os; p = subprocess.Popen(['sleep','3']); ..."
app  pgid: 59634
child pgid: 59634
killpg would target the app itself: True
```
A `subprocess.Popen` child with no `start_new_session` **inherits the parent's process group**. Then:
```
$ sed -n '541,560p' tldw_chatbook/TTS/audio_player.py
    self._current.process.kill()
    try:  self._current.process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        if self._system == "Darwin":
            os.killpg(os.getpgid(self._current.process.pid), signal.SIGKILL)
$ # do the three Popen calls pass start_new_session?
  :348  0     :353  0     :360  0
```
So `killpg` resolves the child's pgid, which **is** the app's pgid, and SIGKILLs the TUI — plus everything else in
the terminal's foreground process group. No shutdown, no `close_tts_resources()`, no DB quiesce.

Guard chain: `terminate()` → `wait(2)` → `kill()` → `wait(0.5)` → on `TimeoutExpired` **and** Darwin → `killpg`.
**Trigger not reproduced:** it needs a player wedged in uninterruptible I/O for >0.5 s after SIGKILL (a network
mount or a busy CoreAudio device). The pgid identity — the part that makes the consequence catastrophic rather than
correct — is proven.

**The repo already ships the correct pairing six times**, each with an explanatory comment:
`Agents/run_hooks.py:342,365`, `Skills_Interop/skill_script_runner.py:299,425-427`,
`Notes/git_process_containment.py:200`, `Agents/agent_worktree_git.py:90`, `Tools/git_tool_impls.py:254`,
`Web_Server/artifact_share.py:169`. **TTS is the only site that calls `killpg` without it.**

Not covered by TASK-32806.5, which names only `Event_Handlers/LLM_Management_Events/server_lifecycle.py` — and the
defect is worse here: that task's site *leaks* a child, this one *kills the parent*.

---

## S09 — two claims checked, one demoted
**(a) `Evaluations_Interop/evaluation_scope_service.py` arity mismatch — REAL but LATENT, not a live P0.**
```
$ grep -n 'def _enforce_policy' tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py
106:  def _enforce_policy(self, mode: EvaluationBackend, resource: str, action: str) -> None:   # SHADOWED
137:  def _enforce_policy(self, action_id: str) -> None:                                        # LIVE
$ # :129 calls  self._enforce_policy(normalized_mode, resource, action)   -> 3 args
$ .venv/bin/python -c "... s._enforce_policy(E(), 'runs', 'create')"
TypeError: EvaluationScopeService._enforce_policy() takes 2 positional arguments but 4 were given
```
The `TypeError` is real and reproducible. **But the containing method has no caller:**
`grep -n 'self\._server_only_service' …/evaluation_scope_service.py` → **no hits** (the cross-file hits are
`Study_Interop`'s own same-named method on a different class). So the path is dead. **Demoted from the implied P0
to a P3 latent defect** — armed, not live. `EvaluationScopeService` itself *is* constructed (`app.py:10605`), so the
class ships; only this method is unreachable.

**(b) `ChatScreen.on_button_pressed` is defined twice — CONFIRMED.**
```
$ grep -n 'def on_button_pressed' tldw_chatbook/UI/Screens/chat_screen.py
24578:    async def on_button_pressed(...)    # 201 lines — SHADOWED, unreachable
25009:    async def on_button_pressed(...)    # 186 lines — LIVE
```
Python keeps the last definition, so the entire first handler is dead. Comparing the bodies: the live one is a
superset in the branches I sampled (it adds a `console-transcript-library-activity` class check and a
`console-prompt-improvement-undo` branch ahead of the shared ones), and an AST diff of string constants found
**zero constants present only in the shadowed body** — so this reads as a duplicated-then-extended handler rather
than lost functionality. **Not a live user-facing defect; a 201-line dead method on the main chat screen, and a
trap for the next editor.** `chat_screen.py` is Tier-1, so this is reported as a cross-tier observation.

**The generalisable result is the census, not either instance.** An AST scan of every class body found **11 classes
with a shadowed method definition**, and `ruff --select F811` flags **none** of them: a two-def probe *is* flagged,
but inserting any intervening method between the two defs suppresses it (reproduced in an isolated 22-line file).
A ~20-line AST check wired into `preflight.sh` closes the whole class.

---

## Cross-slice conflict resolved — persona-visual vs visual-identity locking
**S25 claimed a behavioural drift. S15/S16 retired it. S15/S16 is right.**

S25 (reviewing `Backup_Recovery/`) filed, as the sharp half of its near-clone finding: *"`persona_visual_participants._Source` carries `lock: object = field(default_factory=threading.RLock)` … `visual_identity_participants._Source` has **no such field**. The two modules mutate live image files under different serialisation rules."*

S15/S16 (reviewing `Persona_Visual/`) retired it: *"Both lock."* Checked:
```
$ grep -n 'lock' tldw_chatbook/Backup_Recovery/persona_visual_participants.py
35:     lock: object = field(default_factory=threading.RLock)     # per-SOURCE
410:        while not source.lock.acquire(timeout=0.05):
508:    source.lock.release()

$ grep -n 'lock\|_lock' tldw_chatbook/Backup_Recovery/visual_identity_participants.py
276:  # Stage methods own the mutable edit fields under the original candidate lock.
294:        "_lock",                                               # per-CANDIDATE, in __slots__
981:        with request(), candidate._lock:
```
**Verdict: S25's observation is literally true and its conclusion does not follow.** `visual_identity_participants`
locks per *candidate* rather than per *source*; the granularities differ because the structures differ, and the
comment at `:276` says so. There is no unserialised path.

**What survives from S25's finding:** the *duplication* half — 12 byte-identical functions, identical module-level
state blocks, and **936 differing lines out of 1,956** between the two modules. That stands on its own and is
unaffected. **What is struck:** "settle the lock question explicitly in the same change" as a *defect* to fix; it
becomes "preserve both locking schemes when extracting the shared base", which is a different and easier
instruction.

Recorded because rule 6 requires reporting retired findings as findings — and because a consolidation PR acting on
S25's text as written would have gone looking for a bug that is not there.

---

## The endpoint-probe egress gap — severity ruled, with the reasoning stated
**VERDICT: CONFIRMED. Ruled P1, not P0. This is the lead's call, made explicitly rather than deferred.**

Three slices converged on `UI/Screens/settings_endpoint_probe.py`. S19 found it; S21 escalated it with a restore
vector and asked the lead to rule. Verified:

```
$ grep -n "check_url_or_raise_async" tldw_chatbook/UI/Screens/settings_endpoint_probe.py
   -> imports at 44-46; the ONLY use is line 514, inside _probe_openai_tts_catalog (the TTS branch).
      The chat branch (probe_settings_endpoint -> _request_models) has none.

$ grep -rn 'probe_settings_endpoint' tldw_chatbook | grep -v 'def probe_settings_endpoint'
UI/Speech/speech_catalog_mixin.py:145        <- the GUARDED (TTS) branch
UI/Screens/chat_screen.py:3241
UI/Screens/settings_screen.py:15174
UI/Wizards/FirstRunSetupWizard.py:934
   -> FOUR distinct callers, not five. S21 counted an import line alongside its call; S19's count was right.
      None of the four contains `check_url_or_raise` or `egress` — no caller compensates.

$ grep -n '_write_raw_cli_config_unlocked' tldw_chatbook/Backup_Recovery/config_participants.py
385, 412     -> config.toml IS a guarded backup/restore participant

$ grep -n 'def _initial_endpoint_for' -A 6 tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py
2243:    configured = first_configured_endpoint(self._provider_settings(provider_key))
   -> the wizard prefills the endpoint straight from api_settings.<provider>.{base_url,api_url,…}
```

**The chain holds:** a restored profile archive can set `api_settings.<p>.base_url`; the first-run wizard prefills
it; the user types their own API key and presses Test; `FirstRunSetupWizard.py:922-924` builds
`httpx.AsyncClient(headers={"Authorization": f"Bearer {credential_value}"})` and hands it to the unguarded branch.
That is a credential-exfiltration primitive needing no user-authored URL.

**Why P1 and not P0:**
1. It requires the user to accept and restore an attacker-supplied backup archive — itself a high-trust action, and
   one a user performs knowingly. A P0 in this review's severity table is a boundary crossed on a *shipped path*
   the user is already on; this one needs a prior compromise.
2. The exposure is bounded to the first hop: `follow_redirects=False` at `:302` and the origin re-check at `:426`
   block redirect pivots.
3. I did not execute a restore to prove the prefill lands — the static chain is verified, the end-to-end is not.

**What would promote it to P0:** showing that any *non-restore* path can set `base_url` (a chatbook import, a
synced profile, a URL handler). S21 checked chatbook import and cleared it —
`ChatbookImportWizard.py:79-83` scopes content to CONVERSATION/NOTE/CHARACTER/MEDIA/PROMPT/KEPT_BRIEFING, **no
config**. The settling command is in "Left UNVERIFIED".

**The fix is one guard in the shared function, not four at call sites** — and the file's own TTS branch at `:514` is
the working model, which is what makes this an omission rather than a design choice.

---

## S13 — `check_canvas_mermaid_assets` is a tautology for 2 of its 6 declared outputs
**VERDICT: CONFIRMED. A third defective guard, wrong in a third way.**

```
$ sed -n '349,353p' scripts/vendor_canvas_mermaid.py          # the "build"
    outputs = {
        "canvas_runtime_worker_v2.js": (STATIC / "canvas_runtime_worker_v2.js").read_bytes(),
        "canvas_renderer_v2.js":       (STATIC / "canvas_renderer_v2.js").read_bytes(),
        ...
    for name, data in outputs.items(): (output_dir / name).write_bytes(data)

$ sed -n '200,236p' scripts/check_canvas_mermaid_assets.py     # the "check"
    committed_dir: Path = STATIC,
    ...
    generated = build(isolated_inputs, rebuilt)
    compare_generated_outputs(rebuilt, committed_dir, generated)

$ grep -rn 'canvas_runtime_worker_v2|canvas_renderer_v2' scripts/ Tests/
   -> no generator anywhere; the only producers are those two read_bytes() calls
```
`build()` **copies the two files out of `STATIC`**, and `check_assets` then compares the copy **against `STATIC`**.
For those two outputs the assertion is `STATIC/x == STATIC/x`. The check's success line says *"Canvas Mermaid assets
reproduce: 6 outputs"*; four reproduce from hash-pinned inputs, **two are checked-in source laundered through a
copy** — and they are the largest executables in the Canvas runtime closure (85 KB + 59 KB).
`Tests/CI/test_canvas_mermaid_asset_checker.py` duplicates `EXPECTED_OUTPUTS` verbatim, so it cannot notice.

### The three guards, and the three different ways they are wrong

| Guard | Failure mode |
|---|---|
| `check_timestamp_writers.py` | **Too narrow.** Matches two idioms, not the contract. Reports `OK` with an empty census while six slices independently found writers it cannot see. |
| `check_textual_worker_contract.py` (W002) | **Wrong predicate.** Counts a `query_one` in a `finally:` body as *guarded* — hiding 59 sites, and `finally` is the variant that runs during cancellation. |
| `check_canvas_mermaid_assets.py` | **Circular.** Two of six outputs are compared against the directory they were copied from. |

**All three are wired into `preflight.sh` and the required CI job, and all three are green.** That is the argument
for reviewing guards as code — a green guard is a claim, and these three claim more than they prove.

---

## Correction to this review's own Phase 3 — the `_get_connection` ruling
**S17 contested one line of my ruling. S17 is right.**

I wrote, in the "clusters that are NOT real" table: *"All call `super()._get_connection()` (so they keep
`connect_private_sqlite`)."* Checked:

```
$ for each BaseDB subclass overriding _get_connection: count super()._get_connection in the body
  Scheduling/db/scheduled_tasks_db.py        super()=1
  Sync_Interop/sync_state_repository.py      super()=1
  DB/AgentRuns_DB.py                         super()=1
  DB/Library_Collections_DB.py               super()=1
  DB/Library_Ingest_Jobs_DB.py               super()=0   <--
  DB/Workspace_DB.py                         super()=1
  DB/Subscriptions_DB.py                     super()=1
  Notifications/client_notifications_db.py   super()=1
  Notifications/event_state_repository.py    super()=0   <--
```
Both exceptions call `connect_private_sqlite(...)` **directly**, and both pass `check_same_thread=False` — which is
the reason, and which `BaseDB` cannot pass. `event_state_repository.py:207-215` documents it in nine lines
(TASK-21131: *"Held connections are handed to `close()` on another thread; sqlite3's default guard would refuse
that and leave every worker-pool connection open for the life of the process."*).

**The ruling itself stands** — all nine still route through `connect_private_sqlite`, so the safety envelope is
intact and consolidating them would still be wrong. **What was wrong was my stated mechanism**, and it matters
practically: an auditor grepping for `super()._get_connection()` to confirm the envelope would get a false negative
on exactly the two stores that had a documented reason to differ. Report corrected in place.

---

## S22/S23 — TASK-32802.1's finding text names a consumer that does not consume
**VERDICT: CONFIRMED. A correction to an In-Progress task, not a new finding.**

TASK-32802.1 states: *"the shared escaper `library_rail.py:367-377 _visible_row_title` =
`escape_markup(_truncate_row_title(...))`, consumed by `library_media_canvas.py:204` …, and
**`Widgets/Home/home_rail.py:127,170`**."*

```
$ sed -n '21,27p' tldw_chatbook/Widgets/Home/home_rail.py
def _visible_row_title(title: str) -> str:
    """Return a rail-safe visible title that does not clip in narrow panes."""
    readable = str(title).strip()
    if len(readable) <= _MAX_HOME_ROW_TITLE: return readable
    return f"{readable[: _MAX_HOME_ROW_TITLE - 3].rstrip()}..."        # truncate only
$ grep -c escape_markup tldw_chatbook/Widgets/Home/home_rail.py    -> 0
$ grep -c escape_markup tldw_chatbook/Widgets/Library/library_rail.py -> 4
```
`home_rail.py` has its **own same-named** `_visible_row_title` that truncates and does not escape, and the module
never imports `input_validation` at all. **It is not a consumer of the shared escaper — it is a non-escaping twin.**

**Consequence:** the .1 fix (swap `rich.markup.escape` → `escape_markup` inside the Library helper) will land, AC#1
will pass on Library rows, and **Home rail titles stay unescaped** — `Button.label = "▸ 📄 [Draft] plan"` renders as
`"▸ 📄  plan"`, and a title containing `[/]` raises `MarkupError`.

### The sharper half of the same slice's finding

The `MarkupError` from an unescaped Select option is raised **inside `textual/_compositor.py:387 reflow`**, reached
from `screen.py:1352 _refresh_layout` — **not in `compose()`, not in a handler.** TASK-32802.2/.3 document
compose()-time raises in one dialog, which a `try` around the mount can contain. **A raise during reflow is not
confined to the widget that caused it.** That is a different and worse shape than the task documents, and it is
reachable by typing a deck name, a folder name, or by pointing the vLLM setup at a server whose `/v1/models`
returns a bracketed id — `_is_admissible_model_id('model[/]x')` → **True** (ADR-114's boundary bans control chars,
backslashes, path shapes and `.gguf`, but **allows brackets**).
