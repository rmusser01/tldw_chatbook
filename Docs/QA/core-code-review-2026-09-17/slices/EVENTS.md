# EVENTS — tldw_chatbook/Event_Handlers/ (40 files), 16,816 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. All `$PY` runs use `source <SCRATCH>/env.sh`.
Resumed run: the reachability census below was produced by the first (usage-limit-killed) pass and has been RE-VERIFIED in this pass with exact-name `rg` outside Tests, relative-import forms, and `rg` over `<SCRATCH>/collect_only.txt` (97,863 collected tests).

## Legacy-reachability census

Command used for every row (example for `tab_events`):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && rg -n --no-heading '\btab_events\b' --glob '!Tests/**' --glob '*.py' . ; rg -n 'tab_events' <SCRATCH>/collect_only.txt
```

| module | prod importers (exact name, outside pkg) | intra-pkg | Tests files | collect-only rows naming it | verdict |
|---|---|---|---|---|---|
| `app_lifecycle.py` (141) | **0** | 0 | 0 | 2, **both false positives** (`Tests/App/test_app_lifecycle_events.py::test_mounting_the_app_records_app_started` tests `app.py`; `Tests/TTS/test_tts_app_ownership.py::test_app_lifecycle_shutdown_...` is a test *name*) | **dead** |
| `ingest_events.py` (28) | **0** | 0 | 1 (`Tests/test_application_state_ownership.py:1232` reads the file as *text*) | 2, both `note_ingest_events` substring hits | **dead re-export shim, pinned by a source-text absence test** |
| `ingest_utils.py` (39) | 0 | 2 (`ingest_events`, `note_ingest_events`) | 1 (same source-text test) | 0 | live via `note_ingest_events` |
| `ingest_status_helper.py` (123) | **0** | 0 | 0 | 0 | **dead** |
| `tab_events.py` (48) | **0** | 0 | 0 | 0 | **dead — ADR-014's stated reason to keep it is stale (see F4)** |
| `Chat_Events/chat_messages.py` (445) | **0** | 0 | 0 | 0 (all 0 `chat_messages` rows; the 8 prod hits are a local variable in `LLM_Calls/qwencloud.py`) | **dead** |
| `Audio_Events/dictation_integration_events.py` (106) | **0** | 0 (NOT re-exported by `Audio_Events/__init__.py`) | 0 | 0 | **dead** |
| `Media_Creation_Events/swarmui_events.py` (236) | 0 direct; re-exported by `Media_Creation_Events/__init__.py` — and **nothing imports that package** (`rg -n Media_Creation_Events --glob '*.py' .` → only the `__init__` itself) | — | 0 | 0 | **dead package (2 files)** |
| `LLM_Management_Events/llm_management_events_{llamacpp,llamafile}.py` | **0** | 0 | 0 | 853/24 hits are the *providers* (`local_llamafile`, llama.cpp server tests), none the module | **0-byte files, dead** |
| `eval_db_operations.py` (336) | **0** | 0 | 1 (`Tests/Event_Handlers/test_eval_db_operations_path.py` imports `EvalDBOperations` directly) | 3 | **dead production code kept alive only by its own test** |
| `worker_events.py` | `app.py:424` | 0 | 3 | — | live |
| `worker_handlers/*` | `app.py:408` | — | 5 | — | live |
| `Chat_Events/chat_events_console_dictionaries` | `UI/Screens/chat_screen.py:259` | 0 | 1 | — | live |
| `Chat_Events/chat_image_events` | `Utils/file_handlers.py` +1 | 0 | 5 | — | live |
| `Chat_Events/chat_rag_events` | `UI/Console_Modules/retrieval.py:31`, `RAG_Search/pipeline_integration.py:105` | 0 | 10 | — | live |
| `Audio_Events/{recording,dictation}_events` | `Widgets/voice_input_widget.py:21`, `UI/Dictation_Window_Improved.py:35` | — | 2 | — | live |
| `LLM_Management_Events/llm_management_events{,_mlx_lm,_ollama,_onnx,_transformers,_vllm}`, `gguf_source_modes` | `UI/LLM_Management_Window.py:32-60` | — | 1-4 each | — | live |
| `LLM_Management_Events/server_lifecycle` | 5 (`app.py:15472`, `llm_screen.py:27`, `LLM_Management_Window.py:60`, …) | 1 | 13 | — | live |
| `media_events` | `UI/MediaWindow_v2.py:31`, `Widgets/Media/*` | 0 | 1 | — | live |
| `note_ingest_events` | `Backup_Recovery/async_file_participants.py:110` (private helper `_import_template_files`) | `ingest_events` | 3 | — | live |
| `notes_events` | 2 | 0 | 2 | — | live |
| `STTS_Events/stts_events` | 8 (`app.py:434`, …) | 0 | 27 | — | live |
| `TTS_Events/tts_events` | 7 (`app.py:425`, …) | 0 | 24 | — | live |

**Total dead: 9 files / ~1,127 lines** (`app_lifecycle` 141, `ingest_events` 28, `ingest_status_helper` 123, `tab_events` 48, `chat_messages` 445, `dictation_integration_events` 106, `swarmui_events` 236 + its `__init__` 14, two 0-byte llama files). Deleting any of them also requires an entry removal in the derived artifact `Docs/security/production-diagnostic-inventory.json` (rows 1064/1085/1099/1120 name four of them) — that is a `./scripts/preflight.sh` gate, not a blocker.

### Deletion-candidate verdicts (one line each)

Every candidate was re-verified three ways in this pass: exact dotted-path + bare-name `rg` outside `Tests/` (including relative-import forms `from .X` / `from ..Event_Handlers.X`), `rg` over `<SCRATCH>/collect_only.txt` (97,863 collected tests), and an actual `importlib.import_module` of each.

| candidate | verdict |
|---|---|
| `app_lifecycle.py` (141) | **DELETE** — 0 importers, 0 tests; the 2 collect-only hits are test *names* containing the string, not this module. Imports cleanly. |
| `ingest_events.py` (28) | **DELETE with its pinning test** — a pure re-export shim with 0 importers. `Tests/test_application_state_ownership.py:1232` reads it as *text* to assert retired names are absent; that assertion must move or go in the same commit or it fails with `FileNotFoundError`. |
| `ingest_status_helper.py` (123) | **DELETE** — 0 importers, 0 tests, 0 collect-only rows. Its only cross-reference is the retired `#ingest-notes-import-status-area` widget id, which exists nowhere in the UI. |
| `tab_events.py` (48) | **DELETE** — 0 importers, 0 tests. `backlog/decisions/014-retire-legacy-navigation-chrome.md:19` kept it in 2026-07 because "upstream `app.py` still routes legacy window buttons through `tab_events.handle_tab_button_pressed`"; `rg -n "tab_events|handle_tab_button_pressed" tldw_chatbook/app.py` now returns **nothing**. The ADR's stated reason is stale — note that in the deleting PR rather than re-opening the ADR. |
| `Chat_Events/chat_messages.py` (445) | **DELETE** — 0 importers, 0 tests, 0 collect-only rows; all 9 of its `Message` classes are unreferenced. The 8 prod `chat_messages` hits are a local variable in `LLM_Calls/qwencloud.py`. |
| `Audio_Events/dictation_integration_events.py` (106) | **DELETE** — 0 importers, 0 tests, and deliberately *not* re-exported by `Audio_Events/__init__.py`; all 5 of its `Message` classes unreferenced. |
| `Media_Creation_Events/swarmui_events.py` (236) + its `__init__.py` (14) | **DELETE the package** — 0 importers of the package or the module, **and it is unimportable**: `importlib.import_module("tldw_chatbook.Event_Handlers.Media_Creation_Events.swarmui_events")` → `ImportError: cannot import name 'GenerationResult' from 'tldw_chatbook.Media_Creation'` (`swarmui_events.py:10`). It has been broken long enough for its dependency to drop the symbol, and nothing noticed. |
| `LLM_Management_Events/llm_management_events_{llamacpp,llamafile}.py` | **DELETE** — 0 bytes each. The 853/24 `llamacpp`/`llamafile` collect-only hits are the *providers* (`local_llamafile`, llama.cpp server tests), not these files. |
| `eval_db_operations.py` (336) | **DELETE with its test** — 0 production importers; alive only because `Tests/Event_Handlers/test_eval_db_operations_path.py` imports `EvalDBOperations` directly (3 collected tests). Same shape as `notes_events._parse_note_from_file_content`. |


## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| `LLM_Management_Events/server_lifecycle.py` | 645 | **read in full** |
| `LLM_Management_Events/llm_management_events.py` | 700 | **read in full** |
| `LLM_Management_Events/llm_management_events_vllm.py` | 344 | **read in full** |
| `LLM_Management_Events/llm_management_events_mlx_lm.py` | 173 | **read in full** |
| `Chat_Events/chat_image_events.py` | 299 | **read in full** |
| `Chat_Events/chat_events_console_dictionaries.py` | 163 | **read in full** (+ both call sites in `chat_screen.py`) |
| `media_events.py` | 267 | **read in full** |
| `notes_events.py` | 228 | **read in full** |
| `Audio_Events/recording_events.py` | 55 | **read in full** |
| `Audio_Events/dictation_events.py` | 87 | **read in full** |
| `Audio_Events/__init__.py`, `Media_Creation_Events/__init__.py`, `worker_handlers/__init__.py`, `STTS_Events/__init__.py`, `TTS_Events/__init__.py` | 3+14+15+16+1 | **read in full** |
| `worker_events.py` | 44 | **read in full** |
| `ingest_events.py`, `ingest_utils.py` | 28+39 | **read in full** |
| `app_lifecycle.py` | 141 | **read in full** (prior run) |
| `ingest_status_helper.py` | 123 | **read in full** (prior run) |
| `worker_handlers/base_handler.py` | 181 | **read in full** (prior run) |
| `worker_handlers/misc_worker_handler.py` | 126 | **read in full** (prior run) |
| `Chat_Events/chat_rag_events.py` | 2055 | **sampled**: 1-180, 383-500, 600-920 read in full; full symbol outline (33 defs); every import, every silent `except`, the 2 dynamic-SQL `fetchall` sites and the top-3 largest functions inspected. NOT read line-by-line: 180-383, 920-2055 |
| `LLM_Management_Events/llm_management_events_ollama.py` | 1053 | **sampled**: 1-140 and 273-430 read in full; full symbol outline (16 defs); the repeated guard shape counted mechanically across all 9 async handlers. NOT read line-by-line: 430-1053 |
| `STTS_Events/stts_events.py` | 2968 | **sampled**: 669-790, 985-1010, 2593-2700 read in full; full method outline (62 methods); every import, silent `except`, class-body attribute, `run_worker`, config read and `%`-format logger call swept mechanically over the whole file. NOT read line-by-line: 1-669, 790-985, 1010-2593, 2700-2968 |
| `TTS_Events/tts_events.py` | 4455 | **sampled**: 575-660, 700-800, 1630-1690, 1798-1900, 2660-2700 read in full; full outline (116 defs, 65 of them `TTSEventHandler` methods) with per-function line counts; same mechanical sweeps as above over the whole file. NOT read line-by-line: 1-575, 800-1630, 1900-2660, 2700-4455 |
| `note_ingest_events.py` | 693 | **sampled**: 60-180 and 300-560 read in full; symbol-level reachability resolved for all 7 top-level defs; every widget id it queries resolved against the UI. NOT read line-by-line: 180-300, 560-693 |
| `LLM_Management_Events/gguf_source_modes.py` | 325 | **sampled**: 1-120 and 260-300 read in full; outline + the `BaseException` handler inspected |
| `LLM_Management_Events/llm_management_events_onnx.py` | 203 | **sampled**: 1-60, 130-200 read in full |
| `LLM_Management_Events/llm_management_events_transformers.py` | 201 | **sampled**: 130-201 read in full; the broken import executed |
| `Chat_Events/chat_messages.py` | 445 | **mechanical only** — dead (census); class-level reference count + importability probe |
| `Audio_Events/dictation_integration_events.py` | 106 | **mechanical only** — dead (census) |
| `Media_Creation_Events/swarmui_events.py` | 236 | **mechanical only** — dead (census); import probe shows it raises `ImportError` |
| `tab_events.py` | 48 | **mechanical only** — dead (census) |
| `eval_db_operations.py` | 336 | **mechanical only** — dead but for its own test; logging sweep only |
| `LLM_Management_Events/llm_management_events_{llamacpp,llamafile}.py` | 0+0 | 0-byte |
| `Event_Handlers/__init__.py`, `Chat_Events/__init__.py`, `LLM_Management_Events/__init__.py` | 0 | empty |

Slice-wide mechanical passes run over **all 40 files** (these back the per-file rows above): `ast` sweeps for mutable class-body attributes, silent `except` handlers with context, `run_worker(exclusive=…)`/`group=`, discarded constructor calls, per-function line counts, `Message` subclass reference counts, and import-target resolution against the real interpreter; plus `rg` sweeps for `get_cli_setting`/`load_settings`, loguru-vs-stdlib logging, `%`-style loguru format strings, raw-exception f-string logging, and function-body imports. `ruff check --select E9,F63,F7,F82 tldw_chatbook/Event_Handlers/` → **All checks passed** (0 fatal, matching the Tier-1 baseline).

## Findings


### P1 [D1] — the Transformers "Browse models dir" button can never open a picker: it imports the *top-level* `textual_fspicker`, which is not installed and is not a dependency (the library is vendored)

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py:152-159` — `try: from textual_fspicker import FileOpen / except ImportError: app.notify("File picker utility (textual-fspicker) not available.", severity="error"); return`. Wired at `llm_management_events_transformers.py:196` → `UI/LLM_Management_Window.py:52,528`, button composed at `UI/LLM_Management_Window.py:1395`.
- Evidence (the handler run against a fake app):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import asyncio, inspect
from loguru import logger
from tldw_chatbook.Event_Handlers.LLM_Management_Events import llm_management_events_transformers as tr
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen as VendoredFileOpen
class FakeApp:
    loguru_logger = logger
    def __init__(self): self.notes=[]; self.pushed=[]
    def notify(self, msg, severity="information"): self.notes.append((severity,msg))
    async def push_screen(self, screen, callback=None): self.pushed.append(screen)
app = FakeApp()
asyncio.run(tr.handle_transformers_browse_models_dir_button_pressed(object(), app, None))
print("NOTIFICATIONS:", app.notes); print("PICKERS PUSHED:", app.pushed)
print("vendored FileOpen accepts select_dirs?:", "select_dirs" in inspect.signature(VendoredFileOpen.__init__).parameters)
EOF
```
→
```
ERROR … textual_fspicker not found for Transformers model dir browsing.
NOTIFICATIONS: [('error', 'File picker utility (textual-fspicker) not available.')]
PICKERS PUSHED: []
vendored FileOpen accepts select_dirs?: False
```
and the module is genuinely absent, not merely unimported:
```
$PY -c "import importlib.util as u; print(u.find_spec('textual_fspicker'))"   → None
rg -n "fspicker" pyproject.toml                                               → only the vendored package-data line (:621)
```
`rg -n "^from textual_fspicker" --glob '*.py' tldw_chatbook/` finds exactly two: this line, and `Third_Party/textual_fspicker/__main__.py:20` (the vendored library's own demo entry point, which is allowed to).
- Why it matters: the button ships, is composed, and `Tests/ProductionApp/test_llm_destination_actions.py:101,121,175` asserts it exists **and** carries the tooltip "Choose the local Transformers models root directory." — a promise the handler cannot keep. Every press produces an error toast and no picker, with no way for the user to fix it (installing `textual-fspicker` is not a documented extra). The intended fallback is wrong twice over: the vendored `FileOpen` does not accept the `select_dirs=True` the handler passes at `:182`, so even importing the right module would `TypeError`.
- Recommended correction: use what every sibling in this package already uses — `from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedSelectDirectory` and push it with `callback=_make_path_update_callback(window, app, "transformers-models-dir-path")`, exactly as `llm_management_events_vllm.py:306-318 handle_vllm_local_directory_browse_requested` does. Drop the `try/except ImportError` entirely; the vendored package is not optional.
- Size: S · ADR: no · Confidence: **verified**.
- Pinning test: `Tests/ProductionApp/test_llm_destination_actions.py` pins the button's *presence and tooltip*, not its behaviour — which is why this survived. Adding one assertion that a picker is pushed would have caught it.
- Already covered: none.

### P1 [D4a] — `server_lifecycle.py` spawns every local LLM server without a process group and stops it with a bare `Popen.terminate()`, so a forked worker child survives and the app still reports "stopped"

- Where: `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py:482-501` (`terminate_process_bounded`) and `:559-570` (the `Popen` kwargs in `run_server_subprocess` — no `start_new_session`). Consumers of the stop path: `llm_management_events.py:546,680` (llamafile, llama.cpp), `llm_management_events_ollama.py:256`, `_onnx.py:188`, `_mlx_lm.py:143`, `_vllm.py:239,294`, `UI/Screens/llm_screen.py:3748,4150`.
- Evidence (mechanism reproduced against the shipped function):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import subprocess, sys, os, time
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import terminate_process_bounded
launcher = ("import subprocess,sys,time;"
            "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
            "print(c.pid,flush=True);time.sleep(120)")
p = subprocess.Popen([sys.executable,"-c",launcher], stdout=subprocess.PIPE, text=True)
g = int(p.stdout.readline().strip())
print("child",p.pid,"grandchild",g)
print("terminate_process_bounded ->", terminate_process_bounded(p, timeout=2.0))
time.sleep(0.5)
def alive(pid):
    try: os.kill(pid,0); return True
    except OSError: return False
print("child alive:",alive(p.pid)," GRANDCHILD alive:",alive(g)); os.kill(g,9)
EOF
```
→ `child 48697 grandchild 48698` / `terminate_process_bounded -> True` / `child alive after stop: False  GRANDCHILD alive after stop: True`
- The comparison that makes this a D4a, not a design choice: **nine other subprocess sites in this repo already do the containment** — `Web_Server/artifact_share.py:150,217,226`, `Tools/git_tool_impls.py:254,316`, `Tools/workspace_tool_executor.py:180`, `Chat/console_trace_regex_worker.py:209`, `Chat/console_voice_process.py:361`, `Skills_Interop/skill_script_runner.py:425,305`, `Notes/git_process_containment.py:200,250,1172`, `Agents/run_hooks.py:361,340`, `STT/executor_process_tree.py:395,402` — all `start_new_session=True` + `os.killpg`. The one spawner of *long-running servers* is the one that skips it. `rg -n "start_new_session|killpg" --glob '*.py' tldw_chatbook/` returns zero hits inside `Event_Handlers/`.
- Why it matters: `terminate_process_bounded` returns `True`, so `stop_server_process` clears `app.<provider>_server_process`, calls `_notify_snapshot_stopped`, and notifies `"<label> stopped."` — while a forked engine/runner child still holds the listening port. The next Start then fails on a port already in use with no handle left to kill, and the app's own state says nothing is running. The launchers that fork are the shipped ones: vLLM is launched as the venv console script (`UI/LLM_Management/vllm_setup.py:714  candidate = python_path.with_name("vllm")`), which forks engine-core workers, and `ollama serve` forks a runner per loaded model.
- Recommended correction: in `run_server_subprocess`, add `start_new_session=(os.name == "posix")` (Windows: `creationflags=CREATE_NEW_PROCESS_GROUP`, the shape already written in `Notes/git_process_containment.py:840,1172`); in `terminate_process_bounded`, escalate to `os.killpg(os.getpgid(process.pid), SIGTERM)` then `SIGKILL`, exactly as `Web_Server/artifact_share.py:217-226` does. Canonical home: keep it in `server_lifecycle.py` (it is already the single lifecycle primitive for all six providers) and copy the artifact_share shape rather than inventing a third.
- Size: M · ADR: no (`rg -in "server|subprocess|process group|lifecycle" <SCRATCH>/adr_list.txt` → no ADR covers local-server subprocess containment) · Confidence: verified (mechanism); the per-launcher fork shape is the inferred half — see UNVERIFIED.
- Pinning test: none. `Tests/LLM_Management/test_server_lifecycle_resources.py` has `test_successful_stop_closes_resource_once_after_process_death` and `test_stop_failure_notification_is_actionable_without_process_id`; neither asserts anything about children, and `rg -n start_new_session Tests/LLM_Management/` is empty. Nothing pins the current behaviour.
- Already covered: none.

### P2 [D1] — `TTSEventHandler._request_cooldown` is a class-body dict mutated in place, so TTS cooldown state is process-global and outlives the handler that wrote it

- Where: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:600` (`_request_cooldown: Dict[str, float] = {}`), never reassigned in `__init__` (`:606-697`). Every access mutates the class object: `:756-760` (`del`), `:767-773` (`del`), `:1656` (`.pop`), `:1658-1659` (read), `:1680` (write).
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler as H
a, b = H(), H()
print("same dict object:", a._request_cooldown is b._request_cooldown is H._request_cooldown)
a._request_cooldown["msg-42"] = 1234.0
print("written on a, visible on b:", b._request_cooldown)
print("a fresh third handler starts with:", H()._request_cooldown)
EOF
```
→ `same dict object: True` / `written on a, visible on b: {'msg-42': 1234.0}` / `a fresh third handler starts with: {'msg-42': 1234.0}`
- The repo-wide check that this is the only one in the slice: an `ast` sweep of every class body under `Event_Handlers/` for a mutable literal/`dict()`/`list()`/`set()` default found three — `chat_image_events.py:38 ChatImageHandler.SUPPORTED_FORMATS` (never mutated; in fact never *read* — `rg -n SUPPORTED_FORMATS tldw_chatbook/` shows no use inside that file), `tts_events.py:483 CostTracker.DEFAULT_COSTS` (correctly `.copy()`-ed at `:492`), and this one.
- Why it matters: `app.py:18406-18424` recreates the handler whenever `_initialize_tts_service_owned` raised (it sets `self._tts_handler = None` at `:18424`, and `_ensure_tts_handler` at `:18461` then constructs a fresh one). The fresh handler inherits the old handler's cooldown map, so a message the user already asked to speak is silently refused for up to `COOLDOWN_SECONDS` with "Please wait N seconds…" against a handler that has never spoken anything. It is also why four test files (`Tests/TTS/test_console_audio_cpp_native.py:762,1128`, `test_console_speak_autoplay.py:528,550`, `test_console_speech_snapshot_admission.py:131`, `Tests/UI/test_uat_first_time_character_chat.py:900`) have to hand-reset `handler._request_cooldown = {}` — the tests are working around the leak, and that assignment shadows the class attribute rather than fixing it.
- Recommended correction: one line — move it into `__init__` beside its sibling `self._last_cooldown_cleanup = 0.0` (already correctly per-instance at `:697`): `self._request_cooldown: Dict[str, float] = {}`. Keep the class-level annotation without a value if the type is wanted.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none asserts sharing. `Tests/TTS/test_tts_improvements.py:88-114` exercises cleanup/limit on one handler and passes either way.
- Already covered: none.

### P2 [D1] — a loguru call uses `%s` formatting, so the field name is silently dropped from the log line

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events.py:155` — `logger.info("File/Directory selection cancelled for #%s.", input_widget_id)`. The same function's other two calls (`:147`, `:150`) correctly use `{}`.
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY - <<'EOF'
import sys
from loguru import logger
logger.remove(); logger.add(sys.stdout, format="{message}")
logger.info("File/Directory selection cancelled for #%s.", "llamafile-exec-path")
logger.info("Correct loguru style: {}", "llamafile-exec-path")
EOF
```
→ `File/Directory selection cancelled for #%s.` / `Correct loguru style: llamafile-exec-path`
- Why it matters: every cancelled file-picker in LLM Management logs an identical, field-less line, so the log cannot say *which* picker was cancelled — the exact thing the argument was added for. Repo-wide there are 12 more in 8 other files (`rg -c 'logger\.(debug|info|warning|error|critical|exception)\("[^"]*%[sdrf]' --glob '*.py' tldw_chatbook/` → `Embeddings/Embeddings_Lib.py:1, UI/Wizards/FirstRunSetupWizard.py:1, Chat/custom_endpoint_registry.py:2, UI/Screens/settings_screen.py:4, Chat/prompt_template_manager.py:1, UI/Screens/scheduling/conflicts_tab.py:1, Library/local_media_chunk_tool_service.py:1, TTS/kokoro_pytorch.py:1`) — a defect class, not a one-off.
- Recommended correction: `%s` → `{}` at `:155`; the repo-wide fix is the same mechanical swap plus a `ruff`/grep guard, since this is invisible in review and in tests.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.
- Already covered: none.

### P2 [D3/D4a] — all five "Security functions for input validation" in `llm_management_events_vllm.py` are dead; one of them re-rolls `Utils/input_validation.validate_port`

- Where: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py:47-149` — `validate_python_path` (50), `validate_host` (81), `validate_port` (104), `validate_model_path` (116), `validate_additional_args` (131), under the banner comment `# Security functions for input validation`.
- Evidence: `rg -n "\bvalidate_python_path\b|\bvalidate_host\b|\bvalidate_model_path\b|\bvalidate_additional_args\b" --glob '*.py' .` → only the five `def` lines; zero callers, zero tests, including under `Tests/`. `rg -n "\bvalidate_port\b" --glob '*.py' .` → the dead one here plus the live canonical `tldw_chatbook/Utils/input_validation.py:1221`.
- Why it matters: the banner asserts the vLLM launch inputs are injection-checked. They are not checked by anything — the real vLLM input gate is `Utils/input_validation.validate_vllm_draft_input` (`input_validation.py:313`). Dead security code that reads as live is worse than none: the next reader adds a field and assumes the file's validators cover it. (The three `re.compile`-per-call sites the mechanical scan flagged at `:57/:87/:91/:122` are therefore *not* a hot-path cost — they are never called at all. Retired as a D2, see Retired.)
- Recommended correction: delete lines 47-149. If any of it is wanted, `Utils/input_validation.py` already has `validate_port` (1221) and `validate_ip_address` (1093); that is the canonical home.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.
- Already covered: none.

### P2 [D3] — 25 of the 62 `Message` classes this package defines are never referenced outside their own file; 10 of those are in *live* modules

- Where / evidence: an `ast` pass that collects every class in the slice whose base is `Message` (or a `*Event` base), then counts bare-name references across all of `tldw_chatbook/` excluding the defining file:
```
62 Message/Event classes defined in the slice; 25 with ZERO production references outside their own file
```
Broken down: 9 live in the dead `Chat_Events/chat_messages.py` and 5 in the dead `Audio_Events/dictation_integration_events.py` (both covered by the census), leaving **11 in live modules**:
  - `Audio_Events/recording_events.py` — `AudioRecordingEvent:10`, `RecordingStartedEvent:16`, `RecordingStoppedEvent:24`, `RecordingErrorEvent:33`, `AudioDeviceChangedEvent:49`. The file's *only* live export is `AudioLevelUpdateEvent:41`; the one consumer (`Widgets/voice_input_widget.py:21-28`) imports seven names and takes exactly one from this file. Confirmed: `rg -c "\bRecordingStartedEvent\b" --glob '*.py' --glob '!Tests/**' tldw_chatbook/` → 1 file (the definition), 0 test files, 0 rows in `collect_only.txt`.
  - `Audio_Events/dictation_events.py` — `DictationEvent:10` (base), `DictationPausedEvent:35`, `DictationResumedEvent:41`, `DictationStateChangeEvent:64`.
  - `TTS_Events/tts_events.py:240 TTSStreamingEvent` — never constructed or posted in production; the only references are two **negative** assertions, `Tests/TTS/test_console_audio_cpp_native.py:835` and `:1595`, both `assert not any(isinstance(message, TTSStreamingEvent) …)`. Per the brief's rule 4 that makes its absence a stated requirement — so it is a deliberately-retired contract, not an oversight. Report only; if it is deleted the two assertions go with it.
- Why it matters: `recording_events.py` (55 lines) is 5/6 dead and `dictation_events.py` (87 lines) is 4/10 dead, both behind `from .x import *` in `Audio_Events/__init__.py:2-3` — so a star import is what keeps them nominally "exported". A reader adding a recording feature will wire into `RecordingStartedEvent` and find nothing listens.
- Recommended correction: delete the 9 unused classes in the two Audio_Events modules (`recording_events.py` collapses to the single `AudioLevelUpdateEvent`), and replace the two `import *` lines with explicit names so the next dead export is visible in review. Leave `TTSStreamingEvent` alone unless its two pinning assertions go too.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: only the two negative `TTSStreamingEvent` assertions; none for the Audio_Events nine.
- Already covered: none.

### P2 [D3] — `TTSEventHandler` is a 3,606-line, 65-method class with a single 721-line method, and neither it nor `STTSEventHandler` is on any size ratchet

- Where: `Event_Handlers/TTS_Events/tts_events.py:590-4196` (`TTSEventHandler`, 3,606 lines, 65 methods, 30 instance attributes set in `__init__`); `Event_Handlers/STTS_Events/stts_events.py:669-2963` (`STTSEventHandler`, 2,294 lines, 62 methods, 20 attributes).
- Evidence (`ast` measurement over the slice; the 15 largest functions):
```
 721  Event_Handlers/TTS_Events/tts_events.py:1798 _generate_tts
 386  Event_Handlers/TTS_Events/tts_events.py:3652 handle_tts_playback
 343  Event_Handlers/note_ingest_events.py:308 handle_ingest_notes_import_now_button_pressed
 245  Event_Handlers/TTS_Events/tts_events.py:2521 _stream_response_via_sink
 237  Event_Handlers/Chat_Events/chat_rag_events.py:1811 get_rag_context_capture_for_chat
 236  Event_Handlers/Chat_Events/chat_rag_events.py:956  resolve_scope_for_session
 232  Event_Handlers/STTS_Events/stts_events.py:1956 _persist_settings
 185  Event_Handlers/STTS_Events/stts_events.py:2679 handle_audiobook_generate
 170  Event_Handlers/TTS_Events/tts_events.py:2846 _play_utterance_legacy_artifact
```
(7 functions >200 lines in the slice.) And the ratchet that exists covers only two files: `Tests/Architecture/test_screen_size_ratchet.py:77` (`chat_screen.py`) and `:887` (`library_screen.py`) — `rg -ln tts_events scripts/ Tests/Architecture/` finds only `scripts/validate_live_tts.py` and the diagnostic-inventory test.
- Responsibilities inside the one class, from the method list: message-level TTS requests, guarded hands-free utterances, console speech-destination resolution and fingerprinting, global voice overrides (issue/peek/consume/prune, 4 methods), per-message cooldown rate limiting (3), generation admission + rate limit, the streaming sink, legacy clip playback and its timeout poll, artifact lifecycle (create/append/secure-delete/cache/release/discard/drain, 9 methods), playback lifecycle, user-facing error-copy mapping, and the maintenance protocol. `CostTracker` (`:480-560`) is already a separate class in the same file and has no dependency on the handler.
- Why it matters: `_generate_tts` alone carries the destination authorization closure, the outcome-reporting closure, the service call, the streaming branch, the legacy-artifact branch, and the metrics — 721 lines with one `try` covering nearly all of it. Nothing stops either file growing further, unlike the two screens that are budgeted.
- Recommended correction: **do not redesign** — follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) and §17 (controller-file size governance). The cheapest first step that buys the ratchet: add `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` and `.../STTS_Events/stts_events.py` rows to `Tests/Architecture/test_screen_size_ratchet.py`'s budget table at their measured current values, so the files can only shrink. The natural first extraction is the artifact lifecycle (9 methods, no TTS-request state) into `TTS/`.
- Size: L (needs the recipe's PR series) · ADR: no (recipe covers it) · Confidence: verified.
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` is the mechanism, but it does not cover these files today.
- Already covered: task-1378/task-31202 are the settings_screen equivalents; nothing covers these two.

### P2 [D3] — quitting the app orphans every running local LLM server

- Where: `tldw_chatbook/app.py:19203 on_unmount` (the whole teardown, ~800 lines, tears down recovery service, backup monitor, speech, canvas control, app-owned lifecycles, responsiveness monitor, ingest pools and workers) never touches the six `*_server_process` handles declared at `app.py:7801-7806`.
- Evidence: `rg -n "terminate_process_bounded|stop_server_process" --glob '*.py' tldw_chatbook/` → 20 hits, **every one** inside `Event_Handlers/LLM_Management_Events/` or `UI/Screens/llm_screen.py` (explicit user Stop actions). Zero in `app.py`, zero in any shutdown path.
- Why it matters: `q`/quit leaves llama.cpp / vLLM / Ollama running and holding GPU memory and their port; the next app launch's readiness check sees a server it does not own and cannot stop. (Closing the terminal happens to kill them only because of the *missing* `start_new_session` above — so fixing the P1 above without also adding a shutdown sweep makes this strictly worse.)
- Recommended correction: one `await asyncio.gather(*(stop_server_process(self, p, l) ...))` sweep in `on_unmount`, before `arm_exit_watchdog`'s deadline bites — or an explicit ADR that says servers deliberately outlive the TUI. The two fixes are coupled; ship them in one PR.
- Size: M · ADR: yes if the answer is "deliberate" (new) · Confidence: verified (the grep is exhaustive over the package).
- Pinning test: none found.
- Already covered: none.

### P2 [D3] — ~610 of `note_ingest_events.py`'s 693 lines are handlers for a retired Ingest UI: every widget id they query exists nowhere, and one calls a `TldwCli` method that no longer exists

- Where: `Event_Handlers/note_ingest_events.py` — `_update_note_preview_display:60`, `_parse_single_note_file_for_preview:108`, `_handle_note_file_selected_callback:213`, `handle_ingest_notes_select_file_button_pressed:~260`, `handle_ingest_notes_clear_files_button_pressed:~290`, `handle_ingest_notes_import_now_button_pressed:308` (343 lines — the third-largest function in the slice). Only `_import_template_files` is live.
- Evidence, symbol by symbol (script walks every top-level def and greps all of `tldw_chatbook/` and `Tests/`):
```
_update_note_preview_display                   prod=['Event_Handlers/ingest_events.py'] tests=0
_parse_single_note_file_for_preview            prod=['Event_Handlers/ingest_events.py'] tests=2
_handle_note_file_selected_callback            prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_select_file_button_pressed prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_clear_files_button_pressed prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_import_now_button_pressed  prod=['Event_Handlers/ingest_events.py'] tests=2
_import_template_files                         prod=['Backup_Recovery/async_file_participants.py'] tests=1
```
…and `ingest_events.py` itself has **zero** importers (census above), so six of the seven are reachable from nothing.
- The UI they target is gone. Every widget id in the file:
```
for wid in ingest-notes-import-status-area import-as-templates-radio chat-notes-collapsible \
           ingest-notes-preview-area ingest-notes-selected-files-list ; do
  rg -l "\"$wid\"" --glob '*.py' --glob '*.tcss' tldw_chatbook/ ; done
```
→ only one hit in total, `ingest_status_helper.py` (itself dead). `#ingest-notes-import-type` likewise. CLAUDE.md records why: "ingestion lives in the Library screen's Import rail path; `media_ingest_screen.py` was removed".
- And `note_ingest_events.py:534` calls `app.call_later(app.on_chat_notes_collapsible_toggle, …)` — `rg -c on_chat_notes_collapsible_toggle tldw_chatbook/app.py` → **0**. That is an `AttributeError`, not a `QueryError`, so the `except QueryError` two lines up would not catch it; it is unreachable only because the `query_one("#chat-notes-collapsible")` on the line above fails first. Two dead things covering for each other.
- Why it matters: 343 lines of carefully-commented concurrency work (TASK-15468's snapshot-the-list and cancel-event fix rounds) maintained on a path no user can reach, and two tests (`Tests/Event_Handlers/test_note_ingest_events.py::test_successful_note_import_refreshes_mounted_library_screen`, `…_does_not_touch_non_library_screen`) keep it green so the rot is invisible. `app.py:7792,8058` still declares `selected_note_files_for_import` for it.
- Recommended correction: delete `ingest_events.py`, `ingest_status_helper.py` and the six retired handlers, keeping `_import_template_files` (+ `ingest_utils`, which it needs) — ideally moved next to its one consumer in `Backup_Recovery/`. The `Tests/test_application_state_ownership.py:1232-1244` source-text assertions on `ingest_events.py`/`ingest_utils.py` must move or go in the same commit, and `app.py`'s `selected_note_files_for_import` with them.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: `Tests/Event_Handlers/test_note_ingest_events.py` (2 collected tests) exercises the dead handler; `Tests/test_application_state_ownership.py::test_legacy_ccp_prompt_handlers_and_compatibility_exports_are_absent` reads `ingest_events.py` as text and would fail with `FileNotFoundError` on deletion.
- Already covered: none. Related: task-30019 (legacy Collections migration/retirement) is the nearest in spirit but does not cover this.

### P2 [D4a] — the MLX-LM server is launched with a bare `"python"` resolved from `PATH`, and its failure is invisible because the subprocess's stderr goes to `DEVNULL`

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py:88-98` — `command = ["python", "-m", "mlx_lm.server", "--model", …]`.
- Evidence: the three sibling providers do not do this. ONNX takes an explicit interpreter from the form (`llm_management_events_onnx.py:138  command = [python_path, script_path]`, fed by `#onnx-python-path`); vLLM makes the user pick one (`#vllm-python-path`, `llm_management_events_vllm.py:175-190`, and `UI/LLM_Management/vllm_setup.py:714  python_path.with_name("vllm")`). MLX has no interpreter control at all — `rg -n "mlx-python" tldw_chatbook/UI/LLM_Management_Window.py` → nothing; the only MLX path input is `#mlx-model-path:1432`. And the repo already has the pinned-interpreter idiom in six places, documented at `Notes/file_notes_git_network.py:1440` ("the running interpreter (`sys.executable`) is pinned") — `Tools/file_operation_tools.py:1437`, `Tools/workspace_tool_executor.py:172`, `Audio/system_audio_tap.py:166,204`, `Audio/diarizer_local.py:296`.
- Why it matters: macOS — the only platform MLX runs on — ships no `/usr/bin/python`; `python` exists only if a venv is active in the shell that launched the TUI or the user installed one. A pipx/uv-tool install therefore gets `FileNotFoundError`, and a system-python hit gets `No module named mlx_lm`. Either way `run_server_subprocess` (`server_lifecycle.py:559-562`) sets `stdout=DEVNULL, stderr=DEVNULL`, so the only user-visible trace is the optimistic `app.notify("MLX-LM server starting…")` at `:121` followed by silence — the actual reason never reaches the UI or the log. (On this review box `command -v python` happens to resolve into the checkout's venv, which is exactly why this would pass a developer's manual test.)
- Recommended correction: `sys.executable` in place of `"python"` (one word, matching the six precedents), or add an `#mlx-python-path` input like ONNX's. Separately worth considering: `run_server_subprocess` capturing the child's first N stderr bytes for the destination log — today a launch failure of *any* provider is indistinguishable from a silent one.
- Size: S · ADR: no · Confidence: verified (the code fact and the DEVNULL consequence); **inferred** for "the user hits it", since that depends on their `PATH` — see UNVERIFIED for the check.
- Pinning test: none asserts the argv's first element.
- Already covered: none.

### P2 [D4a] — the chat-attachment image decode path does not escalate PIL's decompression-bomb warning, while five other modules in this repo do — including the one that re-validates the same payload

- Where: `Event_Handlers/Chat_Events/chat_image_events.py:204-238` (`ChatImageHandler.prepare_image_payload`: `PILImage.open` → `thumbnail()` → `save()`), reached from `process_image_file:88,94` on every attached image.
- Evidence (reproduced; `MAX_IMAGE_PIXELS` lowered so a harmless 900×900 lands in PIL's warn-only band rather than allocating anything):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import asyncio, io, warnings
from PIL import Image as PILImage
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
print("PIL default MAX_IMAGE_PIXELS:", PILImage.MAX_IMAGE_PIXELS)
buf = io.BytesIO(); PILImage.new("RGB",(900,900),"red").save(buf,format="PNG"); src = buf.getvalue()
PILImage.MAX_IMAGE_PIXELS = 700_000          # 810k px = between 1x and 2x -> warn-only band
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out, mime = asyncio.run(ChatImageHandler.prepare_image_payload(src, ".png"))
print("RETURNED normally:", len(out), mime)
print("warnings seen, not escalated:", [w.category.__name__ for w in caught])
EOF
```
→ `PIL default MAX_IMAGE_PIXELS: 89478485` / `source PNG bytes: 4360` / `RETURNED normally: 4360 image/png` / `warnings seen, not escalated: ['RequestsDependencyWarning', 'DecompressionBombWarning']`
- The helper that is ignored: `Chat/console_chat_fork.py:798-805` does exactly the right thing for the *same* payload class — `with warnings.catch_warnings(): warnings.simplefilter("error", PILImage.DecompressionBombWarning)` around `open`/`verify`/`load` — and it imports `PAYLOAD_FORMAT_MIME` from this very module. Four more precedents: `Character_Chat/visual_identity.py:1891,1927-1928`, `Character_Chat/expression_set_io.py:563`, `Actor_Packs/contracts.py:713-739`, `Petdex/sources.py:195`. (`Tools/web_tool_impls.py:503-508` deliberately leaves the default *because that path never decodes pixels* — the opposite of this one.)
- Why it matters: PIL raises `DecompressionBombError` only past **2×** `MAX_IMAGE_PIXELS`; between 1× and 2× it emits a warning and decodes anyway. A flat-colour PNG of 12,000×8,000 (96 MP) is a few hundred KB — comfortably inside the 10 MB source cap enforced at `:75-81` — and decodes to ~288 MB before `thumbnail()` and `save()` add their own buffers. The attachment path is the first thing that touches a file the user did not create (downloaded, shared, pasted). The "Processed image too large" check at `:107` runs *after* the allocation and cannot prevent it.
- Recommended correction: wrap the `PILImage.open` … `save` block in `chat_image_events.prepare_image_payload` in the identical two lines `console_chat_fork.py:798-799` already uses, and let the existing `except Exception` at `process_image_file:97` turn it into the same user-visible rejection as any other processing failure. Canonical home: leave it inline (five sites now, all two-liners), or lift a `Utils` `bomb_guard()` context manager if a sixth appears.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none. `Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py` uses Hypothesis on small images and would stay green.
- Already covered: none.

### P2 [D4b] — the Ollama handlers hand-write the same async-staleness guard 28 times across 9 handlers (~750 of the file's 1,053 lines)

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py:273-1053` — `handle_ollama_{list_models,show_model,delete_model,copy_model,pull_model,create_model,push_model,embeddings,ps}_button_pressed`.
- Evidence (counts over the one file):
```
rg -c '_begin_async_presentation'  → 9
rg -c '_owns_async_presentation'   → 28
rg -c '"generation" in locals'     → 18
rg -c 'asyncio.to_thread'          → 9
```
and repo-wide `rg -c "_owns_async_presentation" --glob '*.py' tldw_chatbook/` → this file 28, `llm_management_events_transformers.py` 2, `UI/LLM_Management_Window.py` 1 (the definition, `:2082`).
- The shape repeated verbatim in every one of the nine: `query_one` the inputs → validate-empty + `focus()` + `notify` → `clear()` → `generation = window._begin_async_presentation(channel)` → `del` the widget refs → `await asyncio.to_thread(<ollama api fn>, …)` → `if not window._owns_async_presentation(channel, generation): return` → re-`query_one` → `if error / elif data / else` → **two** `except` blocks (`QueryError`, `Exception`) each re-testing `if "generation" in locals() and not window._owns_async_presentation(channel, generation): return` before logging a category and notifying.
- Why it matters: the guard is the mechanism that stops a stale API reply from writing into a re-navigated destination. Hand-written 28 times, one miss is a silent wrong-pane write, and `"generation" in locals()` is exactly the idiom that breaks quietly when someone reorders a line. `_begin_async_presentation`/`_owns_async_presentation` (`UI/LLM_Management_Window.py:2075-2088`) are two lines each and already the right primitives — what is missing is the wrapper that uses them.
- Recommended correction: one `@asynccontextmanager async def _owned_presentation(window, channel)` (or an `async def _run_ollama_action(window, app, channel, fn, /, **kwargs)`) beside them in `UI/LLM_Management_Window.py`, yielding the generation and swallowing the stale case; each handler keeps only its input reads and its three result branches. Follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR) if this is split out.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: `Tests/ProductionApp/test_llm_destination_actions.py` covers the destination-ownership behaviour these guards implement; a shared helper is the safe refactor target.
- Already covered: none.

### P2 [D4b] — the `maintenance_drain` polling loop is hand-rolled in 10 participants (5 byte-identical), with the poll interval already drifted 0.02 vs 0.01

- Where (my slice's two): `Event_Handlers/STTS_Events/stts_events.py:709`, `Event_Handlers/TTS_Events/tts_events.py:718`. The other eight: `Chat/console_chat_controller.py:5550`, `Chat/console_prompt_queue_coordinator.py:137` (partial), `STT/dispatch_coordinator.py:247`, `TTS/adapter_registry.py:194`, `TTS/audio_cpp_artifact_dependencies.py:583`, `TTS/TTS_Generation.py:1127`, `UI/Console_Modules/dictation.py:893`, `UI/Navigation/audio_cpp_model_handoff.py:169`.
- Evidence: an `ast` sweep that normalises away docstrings and string literals and hashes the remaining source of every `async def maintenance_drain` in `tldw_chatbook/`:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY - <<'EOF'
import ast, pathlib, hashlib, re
root = pathlib.Path("/Users/macbook-dev/Documents/GitHub/tldw-review/tldw_chatbook")
bodies = {}
for p in root.rglob("*.py"):
    try: src = p.read_text(); tree = ast.parse(src)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "maintenance_drain":
            seg = ast.get_source_segment(src, n)
            norm = re.sub(r'"[^"]*"', '"S"', re.sub(r"'''.*?'''|\"\"\".*?\"\"\"", "", seg, flags=re.S))
            bodies.setdefault(hashlib.sha1(re.sub(r"\s+"," ",norm).strip().encode()).hexdigest()[:8], []).append(f"{p.relative_to(root.parent)}:{n.lineno}")
for h, locs in sorted(bodies.items(), key=lambda kv: -len(kv[1])): print(f"[{h}] {len(locs)}"); [print("   ",l) for l in locs]
EOF
```
→ `[17034996] 5 copies` (adapter_registry:194, TTS_Generation:1127, dictation:893, stts_events:709, tts_events:718) plus four one-off variants.
- The drift: `await asyncio.sleep(min(remaining, 0.02))` in seven copies, **`0.01`** in `TTS/audio_cpp_artifact_dependencies.py:583` and `UI/Navigation/audio_cpp_model_handoff.py:169`; the not-paused guard raises `RuntimeError` in eight and `AudioCppArtifactDependencyError` in one; `STT/dispatch_coordinator.py:247` alone takes `self._lock` around the guard. `maintenance_close_admission` and `maintenance_resume` are copy-pasted alongside it in all ten — the "copy-pasted three-method block" shape verbatim.
- Why it matters: the consumer, `Backup_Recovery/runtime_maintenance.py:199 RuntimeMaintenance`, binds ~25 participants by duck-typing and declares **no `Protocol` or ABC** (`rg -n "Protocol|class " Backup_Recovery/runtime_maintenance.py` → only `_Hook` and `RuntimeMaintenance`). A participant that mis-implements the contract fails silently at backup/quiesce time, and the next drain-loop author picks whichever interval they happened to copy.
- Recommended correction: one `async def drain_until(ready: Callable[[], bool], deadline: float, *, interval: float = 0.02) -> bool` plus a `MaintenanceParticipant` `Protocol`, both in `Backup_Recovery/runtime_maintenance.py` (the module that already owns the contract and imports every participant by dotted name). Each site keeps only its own `maintenance_ready` predicate and its own guard message.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: none defines the protocol; the participants' individual drain tests would keep passing through the refactor.
- Already covered: none.

### P3 [D1-shaped, retired to P3] — `validate_host` is annotated `-> bool` but returns a `re.Match`

- Where: `llm_management_events_vllm.py:95-101` — `return host == "localhost" or ... or ipv4_pattern.match(host) or hostname_pattern.match(host)`.
- Evidence: read only; the function is dead (finding above), so no caller can observe it.
- Why it matters: only if the dead code is ever revived — `is True` / JSON-serialising the result would then misbehave.
- Recommended correction: covered by deleting the block.
- Size: S · ADR: no · Confidence: verified (dead) / inferred (impact).
- Pinning test: none.

### P3 [D1] — `Filters(…)` is constructed and thrown away, so the Llamafile executable picker has no filter while its Llama.cpp twin does

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events.py:171` — `Filters(("Executables", lambda p: p.is_file()))` as a bare expression statement; the `FileOpen` two lines below is given no `filters=`. The equivalent Llama.cpp handler at `:554` binds `exec_filters = Filters(("Executables", lambda p: p.is_file()))` and passes it.
- Evidence: an `ast` sweep of the slice for discarded calls to a capitalised callee returns exactly one row:
```
… for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call): …
```
→ `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py:171  Filters(("Executables", lambda p: p.is_file()))`
- Why it matters: a user browsing for the Llamafile executable sees directories and non-executables; the identical Llama.cpp flow does not. A one-word difference in two copy-pasted handlers.
- Recommended correction: bind it and pass `filters=exec_filters`, matching `:554`.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.

### P3 [D3] — 17 function-body imports in `tts_events.py` for two modules that are neither optional nor cycle-breaking

- Where: `Event_Handlers/TTS_Events/tts_events.py` — `tldw_chatbook.TTS.audio_player` at `:2824, :2928, :3581, :3603, :4345, :4412, :4444`; `tldw_chatbook.Metrics.metrics_logger` at `:1834, :2688`; `TTS.playback_capability` at `:1955, :2932, :3031`; plus `Chat.console_speech_text:1267`, `TTS.pcm_playback:2260`.
- Evidence: `sed -n '1,40p' tldw_chatbook/TTS/audio_player.py` → module-level imports are `asyncio, subprocess, platform, shutil, threading, time, pathlib, typing, enum, dataclasses, loguru` — stdlib plus loguru, no optional dependency and no import of `Event_Handlers`, so neither an `optional_deps` guard nor a cycle break applies. For `metrics_logger`, `rg -n "^from .*Metrics.metrics_logger import" --glob '*.py' tldw_chatbook/` → 63 module-level importers against 69 total, i.e. the module-level form is the house style.
- Why it matters: cost is negligible (each is a `sys.modules` hit, and the one on a streaming path at `:2688` fires once per utterance, not per chunk — I checked the enclosing block) — this is a readability/consistency item, and it hides the module's real dependency surface from anyone reading the imports.
- Recommended correction: hoist `audio_player`, `metrics_logger`, `playback_capability` to module scope; leave any that a cycle actually requires and say so in a comment.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.

### P3 [D3] — `ChatImageHandler.SUPPORTED_FORMATS` is a dead constant

- Where: `tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py:38`.
- Evidence: `rg -n "SUPPORTED_FORMATS" tldw_chatbook/` → the definition, plus an unrelated module-level `SUPPORTED_FORMATS` in `Local_Ingestion/Document_Processing_Lib.py:93` and a `SUPPORTED_FORMATS_COPY` in `Library/library_ingest_state.py:86`. Zero reads of the class attribute.
- Why it matters: three different "supported image/doc formats" tables with no shared owner; a reader editing this one changes nothing.
- Recommended correction: delete it, or make `chat_image_events` actually use it in its validation path (see the extension check inside the same file).
- Size: S · ADR: no · Confidence: verified.

### P3 [D3] — `RAG_Search/pipeline_integration.py:122` imports `chat_rag_events_simplified`, a module that does not exist (in a module with zero importers)

- Where: `tldw_chatbook/RAG_Search/pipeline_integration.py:120-127` — the `pipeline_id.endswith("_v2")` branch of `PipelineManager.execute_pipeline`. Root cause is in my slice: `Event_Handlers/Chat_Events/chat_rag_events.py` was renamed from `chat_rag_events_simplified.py` and the fossil is still visible at its own line 1 (`# chat_rag_events_simplified.py`) and line 69 (`logger.bind(module="chat_rag_events_simplified")`).
- Evidence: `ls tldw_chatbook/Event_Handlers/Chat_Events/` → `__init__.py  chat_events_console_dictionaries.py  chat_image_events.py  chat_messages.py  chat_rag_events.py  MIGRATION_GUIDE.md`. No `chat_rag_events_simplified.py`. `rg -n "pipeline_integration" --glob '*.py' .` → **zero importers**, and `rg -c pipeline_integration <SCRATCH>/collect_only.txt` → zero collected tests.
- Why it matters: it is the brief's "function-body import of a module that no longer exists" exactly — but the containing module is unreachable, so nothing can hit it today. Kept at P3 for that reason. The live cost is the log tag: every line this 2,055-line module emits is bound to `module="chat_rag_events_simplified"`, a filename that does not exist, so a log search by module name finds nothing.
- Recommended correction: for the RAG slice — delete `pipeline_integration.py` (0 importers, 0 tests) or fix the import. In my slice — fix `chat_rag_events.py:1` and `:69` to say `chat_rag_events`.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.
- Already covered: none. **Hand-off: this belongs to whoever reviews `RAG_Search/`.**

### P3 [D3] — `VLLM_BUTTON_HANDLERS` is a permanently empty dict that is still imported, splatted, and pinned empty by two tests

- Where: `llm_management_events_vllm.py:340` (`VLLM_BUTTON_HANDLERS: dict[str, object] = {}`), consumed at `UI/LLM_Management_Window.py:55,529` (`**VLLM_BUTTON_HANDLERS`).
- Evidence: `rg -n "\bVLLM_BUTTON_HANDLERS\b" --glob '*.py' .` → definition, two import/splat sites, and `Tests/LLM_Management/test_vllm_setup.py:1118` / `Tests/ProductionApp/test_llm_destination_actions.py:804`, both `assert vllm_events.VLLM_BUTTON_HANDLERS == {}`.
- Why it matters: nothing breaks; it is a retirement that was left half-finished. The two tests state the emptiness as a *requirement*, so per the brief's rule 4 this is a decision, not a bug — report only.
- Recommended correction: leave it, or retire the symbol and the two assertions together in one commit. Not worth a PR on its own.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: `Tests/LLM_Management/test_vllm_setup.py::…:1118` and `Tests/ProductionApp/test_llm_destination_actions.py::…:804` — both assert the current behaviour as a requirement.
- Already covered: none.

### P3 [D3] — `notes_events.py` does file I/O, JSON parsing and a config-path lookup at module import, and re-imports `pathlib.Path` at line 138

- Where: `Event_Handlers/notes_events.py:219  NOTE_TEMPLATES = load_note_templates()` (module scope) and `:138  from pathlib import Path  # noqa: E402` (already imported at `:6`).
- Evidence: `load_note_templates` (`:140-216`) calls `_get_effective_config_path()`, stats two paths, reads and JSON-parses whichever exists, and emits `logger.info(...)` — all at import. Mitigating: both production consumers late-import the name (`UI/Library_Modules/library_notes_controller.py:6209`, `Widgets/Library/library_notes_canvas.py:3727`, the latter commented "imported locally to match the existing…"), so the cost lands at first template use, not at startup.
- Why it matters: the module has no lazy accessor, so templates are frozen for the process; `Notes/template_store.read_templates()` (its own single caller is this function) is the real reader. There is no in-app writer for `note_templates.json`, which is why this is P3 rather than a staleness bug. The `chat_rag_events.py:110-125` lazy-probe comment (TASK-21731) is the pattern the package already chose for exactly this.
- Recommended correction: turn `NOTE_TEMPLATES` into a `@lru_cache`-backed `note_templates()` function (one line), and delete the duplicate `Path` import.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: `Tests/UI/test_library_shell.py:23264` does `monkeypatch.setitem(notes_events.NOTE_TEMPLATES, …)` — a mutation of the module global that a function accessor would need to accommodate.

### P3 [D4a] — 22 sites in this package log raw exception text/filenames by f-string while the rest of the package logs bounded `category=` codes

- Where: `eval_db_operations.py` ×9, `note_ingest_events.py` ×4 (`:101, :152, :304, :689`), `stts_events.py` ×3 (`:2655, :2846, :2851`), `ingest_status_helper.py` ×2, `worker_handlers/base_handler.py` ×2 (`:116, :130`), `notes_events.py` ×1, `swarmui_events.py` ×1.
- Evidence: `rg -c 'logger\.(error|warning|exception|info|debug)\(\s*f?"[^"]*\{(e|err|error|exc|exception|ex)\}' tldw_chatbook/Event_Handlers/` → the per-file counts above, 22 total. Contrast the modernized convention in the same package: `server_lifecycle.py:321`, `llm_management_events.py:392`, `llm_management_events_ollama.py` (`logger.error("… (category={}).", type(e).__name__)`) and `llm_management_events_ollama.py:80` which routes payloads through `Utils.log_sanitizer.sanitize_dict`.
- Why it matters: `note_ingest_events.py:152 logger.error(f"Error parsing {file_path.name}: {e}")` puts a user's note filename and a raw OS error message in the log; `stts_events.py:2655` renders an ffmpeg failure verbatim including the full input/output paths. None of the 22 carries a credential (I checked each), which is why this is P3 and not higher — but the package has already decided on the bounded form and these are the stragglers. 13 of the 22 are in modules this report finds dead (`eval_db_operations`, `ingest_status_helper`, `swarmui_events`), so deleting those closes most of it.
- Recommended correction: the 9 live sites take the same `category={type(e).__name__}` form the package already uses; `Utils/log_sanitizer.py` (already imported by `llm_management_events_ollama.py:44`) is the canonical home for anything that must keep a payload.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.

### P3 [D4a] — `_sensitive_fetchall` reaches around `execute_query` to avoid parameter logging that `ChaChaNotes_DB` already solved with `redact_params=True`

- Where: `Event_Handlers/Chat_Events/chat_rag_events.py:678-705`; its three callers at `:711` (conversation metadata), `:792` (scope existence), and through them `:724`, `:753`.
- Evidence: the docstring's premise is "their public `execute_query` helpers DEBUG-log parameters and therefore must not be used for prompt-boundary identity reads". `DB/ChaChaNotes_DB.py:3989` signature is `execute_query(self, query, params=None, *, commit=False, script=False, redact_params: bool = False)` and `:4030` logs `f"…Params: {'<redacted>' if redact_params else preview_params(params)}"` under `logger.opt(lazy=True)`. `DB/Client_Media_DB_v2.py:1195` has **no** `redact_params` — its signature is `(self, query, params=None, *, commit=False)`.
- Why it matters: the bypass is safe but it skips `execute_query`'s error translation (`ConflictError`/`CharactersRAGDBError`) and its `chachanotes_db_query_duration` / `chachanotes_db_query_count` metrics, so RAG scope reads are invisible in DB telemetry. And it only *has* to exist because the two DB classes' redaction capability has drifted.
- Recommended correction: add `redact_params` to `Client_Media_DB_v2.execute_query` (identical three-line change, canonical home `DB/base_db.py` if a shared base is wanted), then replace `_sensitive_fetchall` with `db.execute_query(query, params, redact_params=True).fetchall()`. That also deletes the module-name-sniffing test-double branch at `:695-703`, which is production code that inspects `type(db).__module__` for `"Tests."`.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: `Tests/RAG/test_scope_pipeline_enforcement.py` exercises the callers with doubles; the module-sniffing branch exists to serve those doubles.
- Already covered: none.

### P3 [D4a] — `notes_events._parse_note_from_file_content` is a superseded 85-line parser kept alive only by its own test file

- Where: `Event_Handlers/notes_events.py:39-124`.
- Evidence: `rg -n "_parse_note_from_file_content" --glob '*.py' .` → the definition plus `Tests/Notes/test_notes_events_parse_note_from_file_content.py` (6 collected tests) and nothing else. The live parser is the registry: `note_ingest_events.py:120  note_importer_registry.parse_file(file_path, import_as_template=…)`.
- Why it matters: two note parsers with different behaviour (this one is a hand-rolled JSON→YAML→plaintext ladder with a filename-stem fallback; the registry is pluggable), and the dead one has the test coverage.
- Recommended correction: delete the function and its test module; `note_importer_registry` is the canonical home.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: `Tests/Notes/test_notes_events_parse_note_from_file_content.py` — 6 tests that state the dead behaviour as a requirement.

### P3 [D4b] — `ChatImageHandler.MAX_IMAGE_SIZE` is a dead constant that `attachment_core.MAX_IMAGE_BYTES` claims to mirror, plus three dead public methods

- Where: `chat_image_events.py:37` (`MAX_IMAGE_SIZE`), `:38` (`SUPPORTED_FORMATS`, see the earlier P3), `:240 _process_image_data`, `:254 validate_image_data`, `:272 get_image_info`.
- Evidence: `rg -n "MAX_IMAGE_SIZE" --glob '*.py' .` → the definition, two test assertions, and `Chat/attachment_core.py:26  MAX_IMAGE_BYTES = 10 * 1024 * 1024  # matches ChatImageHandler.MAX_IMAGE_SIZE`. The live code path reads `max_image_bytes()` from `attachment_core` (`chat_image_events.py:76`), never the class constant. `rg -n "validate_image_data|get_image_info|_process_image_data" --glob '*.py' .` → zero production callers; only `Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py` and a `Tests/Chat/test_attachment_core.py:239` comment that reads "…which process_attachment_bytes stopped [using]".
- Why it matters: two copies of the 10 MB number with a comment asserting a link nothing enforces — change `max_image_bytes`'s default and the constant plus its two tests silently disagree. `_process_image_data`'s docstring claims its shape is "pinned by existing callers"; there are none.
- Recommended correction: delete `MAX_IMAGE_SIZE`, `SUPPORTED_FORMATS`, and the three methods with their tests in one commit; `attachment_core` is the single owner of both the cap and the format allowlist.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py:379` asserts the constant's value as a requirement — a decision to retire together with it.

### P3 [D4b] — `handle_start_{llamafile,llamacpp}_server_button_pressed` are ~70-line copies that have drifted in widget type and default port

- Where: `llm_management_events.py:456-536` (llamafile) and `:581-661` (llamacpp).
- Evidence: read; the two bodies differ only in the six widget ids, the default port (`"8000"` vs `"8001"`), the provider literal, the "Llamafile executable path is required." vs "Executable path is required." copy — and, materially, the additional-args widget: llamafile reads a `TextArea` (`additional_args_input.text`), llamacpp reads an `Input` (`additional_args_input.value`).
- Why it matters: both flows are one `_run_gguf_server_worker` underneath; only the form-reading differs. Any future fix to one (the `Filters` bug above is already an instance) has to be remembered for the other.
- Recommended correction: one `async def _handle_start_gguf_server(window, app, provider, ids: _GgufFormIds, default_port: str)` in this module; the two public handlers become two-line adapters, the same shape the four `handle_stop_*` handlers already have over `stop_server_process`.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: `Tests/LLM_Management/test_gguf_server_sources.py` monkeypatches `run_server_subprocess` around both flows; a shared helper keeps those green.

### P3 [D4b] — `media_events.py` writes `self.record_id = record_id if record_id is not None else media_id` in 12 of its 13 message classes

- Where: `Event_Handlers/media_events.py:32, 55, 67, 99, 124, 141, 158, 176, 194, 216, 243, 265`.
- Evidence: `rg -c "record_id if record_id is not None else media_id" tldw_chatbook/` → `media_events.py:12` (and nowhere else in the repo).
- Why it matters: it is the file's only logic, and it is the identity rule every Media message must agree on — twelve chances to type `media_id` where `record_id` belongs. The file is otherwise a clean contract module (267 lines, no I/O, no dispatch, exactly as its docstring promises).
- Recommended correction: one `class _MediaRecordEvent(Message)` base in the same file that sets `media_id`/`record_id`/`backing_media_id`; each subclass calls `super().__init__(media_id, record_id=…)` and adds its own fields.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none specific.

### P3 [D4b] — `stts_events._terminate_conversion_process` ends with an unbounded `await process.wait()` where the package's own `terminate_process_bounded` bounds both waits

- Where: `Event_Handlers/STTS_Events/stts_events.py:2659-2678`, final line `await process.wait()` after `process.kill()`.
- Evidence: read; compare `Event_Handlers/LLM_Management_Events/server_lifecycle.py:482-501`, which wraps both `wait()` calls in `timeout=timeout` and returns a boolean rather than blocking.
- Why it matters: a ffmpeg child that cannot be reaped (uninterruptible state) parks the audiobook/format-conversion coroutine forever, and this is the cancellation path, so the caller is already trying to give up. Practically near-impossible after SIGKILL, hence P3 — it is listed because it is the same job as `terminate_process_bounded` written a second time with a different bound.
- Recommended correction: `await asyncio.wait_for(process.wait(), timeout=2)` inside a `contextlib.suppress(TimeoutError)`, or reuse the lifecycle primitive.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.

## Candidate dispositions

Every row from `<SCRATCH>/excerpts/EVENTS.md`. Mechanical rows are hints; each was resolved by reading the code.

| candidate (file:line pattern) | disposition |
|---|---|
| **dup_shape** `handle_stop_{mlx,vllm,llamacpp,onnx}_server_button_pressed` ×4 | **retired** — these are 4-line adapters over the shared `stop_server_process`, one per button id in the four `*_BUTTON_HANDLERS` maps. The shared helper is already used; a table-driven form would save ~20 lines and lose the per-id docstring. Not duplication worth a PR. |
| **dup_shape** `maintenance_drain` ×3 (stts:709, tts:718, dictation:893) | **confirmed and widened** — an `ast` sweep found **10** implementers, 5 byte-identical after normalising strings, with the poll interval already drifted 0.02 vs 0.01. See the P2 D4b finding. |
| **except_exception_pass** `chat_rag_events.py:897` | **retired** — `try: app._console_rag_scope_cache = cache / except: pass` in `_scope_cache_for`, documented at `:873-886` as a deliberate fallback to an unattached cache when `app` refuses attribute assignment. |
| **except_exception_pass** `gguf_source_modes.py:279` | **retired** — it is the *inner* handler of `except BaseException: try: leased.close() / except BaseException: pass; raise`. Cleanup failure must not mask the original; the outer handler re-raises. Correct. |
| **except_exception_pass** `llm_management_events_transformers.py:174` | **retired** — swallows a failed HF-cache-dir probe and keeps `Path.home()` as the picker default. Comment present. (The *enclosing handler* is broken for an unrelated reason — see the P1.) |
| **except_exception_pass** `server_lifecycle.py:493, 499` | **retired** — both inside `terminate_process_bounded`, which re-checks `process_is_running` afterwards and returns the answer; a failed `terminate()`/`kill()` correctly falls through. (The function has a real defect, but it is the missing process group, not these handlers.) |
| **except_exception_pass** `server_lifecycle.py:526` | **retired** — `notify_state_change`'s `call_from_thread` on a closed app loop. Purely presentational. |
| **except_exception_pass** `server_lifecycle.py:643` | **retired** — carries `# noqa: BLE001,S110` and a two-line explanation ("No worker-thread service/UI mutation after the app loop closes"). |
| **except_exception_pass** `stts_events.py:1000` | **retired** — `task.exception()` inside a done-callback; `except BaseException` is the correct width there because a cancelled task raises `CancelledError` from `.exception()`. Standard "retrieve so asyncio does not warn" idiom. |
| **except_exception_pass** `stts_events.py:2744, 2794, 2828` | **retired** — all three carry `# UI element not found, continue without UI updates` and wrap only `RichLog.write`/attribute sets on a possibly-unmounted audiobook widget. |
| **except_exception_pass** `stts_events.py:2863` | **confirmed as noted, low** — same shape as the three above but **without** the explanatory comment, and it swallows the failure of `audiobook_widget.audiobook_generation_complete(False)`, i.e. the one that tells the user generation failed. Add the comment or narrow to `QueryError`. Folded into the P3 logging/consistency row rather than raised separately. |
| **except_exception_return** ×7 files (chat_image, chat_rag, llm_management_events, server_lifecycle ×4, stts ×2, tts ×3, eval_db_operations) | **retired as a class** — enumerated with an `ast` pass that printed each handler plus its surrounding four lines. Every one is a narrow predicate returning a safe default (`process_is_running` → `True` on an unreadable handle; `current_llm_destination` → `None`; `validate_image_data` → `False`; JSON parse → `None`). None is on a write/data path. The two that matter for other reasons are reported separately (`validate_host`'s wrong return type; `_settle_source_preparation`'s nested fallback, which is deliberate). |
| **fetchall_dynamic_sql** `chat_rag_events.py:693, 703` (`_sensitive_fetchall`) | **retired as SQL risk, confirmed as duplication** — the only interpolation is `f"SELECT id FROM {table} …"` where `table` comes from a two-branch `if source_type == SOURCE_TYPE_MEDIA … elif … SOURCE_TYPE_NOTE` (`:778-786`), never user input; ids go through `json_each(?)` as one bound parameter, and the result set is bounded by the caller's scope allowlist. No injection, no unbounded scan. The *helper* question is real — see the P3 D4a on `redact_params`. |
| **function_body_import** ×8 files (chat_image 3, chat_rag 4, llm_management_events 4, stts 3, **tts 11**, eval_db 1, note_ingest 4, notes_events 1) | **partly confirmed** — `chat_rag_events.py:142` is the documented TASK-21731 lazy probe (verified-fine); `note_ingest_events.py:529` and `chat_rag_events.py:626` break real `Event_Handlers ↔ UI.Screens` cycles (verified-fine); `llm_management_events.py:230,288-291` load `LLM_Management.snapshot_*` only on the llama.cpp snapshot path (fine). The 17 in `tts_events.py` are **confirmed** — see the P3 D3 row; `audio_player` is stdlib-only and `metrics_logger` is imported at module scope by 63 of its 69 importers. |
| **legacy_markers** ×8 files (tts 36, chat_rag 12, stts 6, …) | **retired as a signal** — spot-read ~20 of them; in this package "legacy" marks *live, deliberately-retained* paths (`_play_utterance_legacy_artifact`, `_generate_legacy`, the "legacy(#0)+table(≥1)" attachment read contract), not rot. The genuine rot in this slice was found by reachability, not by the word. |
| **mutable_class_attr** `tts_events.py:600 TTSEventHandler._request_cooldown` | **confirmed — mutated in place, shared across instances.** See the P2 D1 finding with the reproduction. The `ast` sweep found only two other mutable class-body attributes in the slice; both are correct (`CostTracker.DEFAULT_COSTS` is `.copy()`-ed at `:492`; `ChatImageHandler.SUPPORTED_FORMATS` is never read at all, reported as dead). |
| **raw_1024x1024** `chat_image_events.py:37`, `tts_events.py:98` | **retired as magic-number rows, one confirmed dead** — `chat_image_events.py:37` is `MAX_IMAGE_SIZE = 10 * 1024 * 1024`, a **dead** constant duplicated by `attachment_core.MAX_IMAGE_BYTES` (reported P3). `tts_events.py:98` is `MAX_TTS_*` sizing; live and used. |
| **re_compile_in_def** `llm_management_events_vllm.py:57, 87, 91, 122` | **retired as a D2, confirmed as dead code** — all four are inside `validate_python_path`/`validate_host`/`validate_model_path`, which have **zero callers anywhere** (`rg` over the whole repo returns only the `def` lines). There is no hot path because there is no path. Reported as the P2 dead-security-helpers finding instead. |
| **try_import_guard** `chat_image_events.py:125` | **retired** — `defusedxml.ElementTree` inside `_svg_raster_kwargs`, guarding an aspect-ratio parse that falls back to a hard square bound; documented at `:116-123`. |
| **try_import_guard** `chat_rag_events.py:141` | **retired** — the TASK-21731 lazy RAG probe, documented at `:110-125` and explicitly listed as known-deliberate in the brief's spirit (cache-backed, computed once). |
| **try_import_guard** `chat_rag_events.py:625` | **retired** — `from ...UI.Screens.chat_screen import ChatScreen` inside `_active_console_session`, breaking a real cycle; documented at `:611-618`. |
| **try_import_guard** `llm_management_events_ollama.py:424` | **retired** — `except (QueryError, Exception)` around widget reads in the delete-model handler, part of the 28-guard pattern reported as D4b. |
| **try_import_guard** `llm_management_events_transformers.py:152` (`optional_deps_imported=True`) | **CONFIRMED — this is the P1.** The guarded module `textual_fspicker` is not installed, is not a declared dependency, and is vendored under `Third_Party/`; the handler therefore always takes the failure branch. |
| **try_import_guard** `stts_events.py:1958, 2686, 2732, 2751, 2804, 2855` | **retired** — `:1958` guards `asyncio.CancelledError` around a settings write; `:2686` guards the genuinely optional `TTS.audiobook_generator`; the rest are the commented "UI element not found" widget writes covered above. |
| **try_import_guard** `tts_events.py:2687` | **retired** — wraps a single `log_counter` metric publication; failure degrades to `logger.debug` and cannot affect playback. |
| **try_import_guard** `notes_events.py:189` | **retired as a guard, confirmed for import-time I/O** — the `try/except` around `Notes.template_store.read_templates()` is a reasonable fallback chain; what is worth changing is that the whole thing runs at module import (`:219`). Reported P3. |
| **try_import_guard** `worker_handlers/base_handler.py:110` | **retired** — `update_button_state` swallowing a missing-widget failure; logged with a warning at `:116`. |

## Verified-fine

- **No `get_cli_setting` / `load_settings` call anywhere in the package.** `rg -n "get_cli_setting\(|load_settings\(|get_cli_providers_and_models\(" tldw_chatbook/Event_Handlers/` → **0**. The sibling's ~11 ms-per-call P1 D2 pattern simply does not apply to this slice. The one config read is `stts_events.py:753 get_runtime_config_snapshot()`, inside `_capture_sample_evidence_candidate`, i.e. once per Studio playground synthesis request — not a tick, compose or keystroke path.
- **No credential reaches a subprocess argv, env or log from this slice.** `rg -n "env=|api_key|API_KEY" tldw_chatbook/Event_Handlers/LLM_Management_Events/*.py` → **0 hits**. The only `env=` that `run_server_subprocess` ever receives is `descriptor.child_env` from `LLM_Management/snapshot_admission.prepare_launch`, and `LaunchDescriptor` declares both `child_env` and `bearer_token` as `field(repr=False)` (`snapshot_models.py:139,153`). No module in the slice logs `command` or `argv` (`rg -n "command|argv" …/LLM_Management_Events/*.py | rg -i "logger|log_output|write\(|notify"` → empty). The ENTRY-config placeholder-key concern does not reach here.
- **Every `run_worker(exclusive=True)` in the slice carries `group=`.** An `ast` pass over all 40 files found 6 `run_worker` calls; the 5 exclusive ones (`llm_management_events.py:520,653`, `_mlx_lm:114`, `_ollama:215`, `_onnx:158`) all pass `group=`, and the sixth (`note_ingest_events.py:646`) is not exclusive.
- **No file in the slice mixes loguru and stdlib `logging`.** Three use stdlib exclusively (`chat_image_events.py`, `server_lifecycle.py`, the dead `tab_events.py`); the rest use loguru. Stdlib is not a dead end here: `Logging_Config.py:410,712,747` attaches the file handler, the Textual console handler and the in-app `_rich_log_handler` to the stdlib **root** logger, so `server_lifecycle.py:321`'s error does reach the user's Logs window. P3 consistency at most, not a lost-error.
- **`_resolve_scope_with_current_ids` (`chat_rag_events.py:823-833`) is the *safe* direction of the negative-predicate offload.** `if bool(getattr(db, "is_memory_db", False)): inline … else: await asyncio.to_thread(...)` — the default when the attribute is absent is `False`, i.e. **thread it**. Unknown DB shapes are offloaded, not inlined. Both DB classes hand out thread-local connections (`ChaChaNotes_DB.py:3576 → _get_thread_connection`, `Client_Media_DB_v2.py:1182` likewise), so the thread hop is safe.
- **`console_attachable_dictionaries` / `console_attached_dictionaries` are sync DB reads dispatched correctly.** Both call sites wrap them in `asyncio.to_thread` (`UI/Screens/chat_screen.py:14461, 14648`), and `list_chat_dictionaries(db, limit=1000, …)` is bounded. The N+1 `load_chat_dictionary` loop in `console_attached_dictionaries` is bounded by the attachments of one conversation.
- **`media_events.py` does exactly what its docstring promises** — 13 `Message` contracts, no dispatch, no I/O, no presentation. (Its one repetition is reported as a P3.)
- **`_snapshot_listener_exists` (`llm_management_events.py:273-282`) deliberately lets non-`ConnectionRefusedError` failures propagate**, with the comment "Ambiguous network failures are preflight failures, never proof of ownership". The 5-second blocking connect runs on a worker thread, not the loop.
- **`ruff check --select E9,F63,F7,F82 tldw_chatbook/Event_Handlers/` → All checks passed.** Matches the Tier-1 baseline of 0 fatal.
- **Ollama's outbound HTTP does not go through `Utils/egress.py`, and that is defensible here** — `Local_Inference/ollama_model_mgmt.py:95` validates the URL scheme and rejects non-http(s) before building `full_url`, and the URL is the user's own typed server address, not attacker-supplied. Noted for the `Local_Inference` reviewer rather than raised: that module is outside this slice, and 47 modules elsewhere do use `Utils/egress`.

## Retired

- **`llm_management_events_vllm.py`'s four `re.compile`-per-call sites as a D2.** Symptom real (regexes are compiled inside function bodies), cause wrong: the functions have zero callers, so there is no cost. Retired as an efficiency finding and re-raised as dead code. Evidence: `rg -n "\bvalidate_python_path\b|\bvalidate_host\b|\bvalidate_model_path\b|\bvalidate_additional_args\b" --glob '*.py' .` → only the four `def` lines.
- **"`server_lifecycle.py`'s `except Exception: pass` around `kill()` leaks the server process."** Symptom plausible, cause wrong: the swallow is harmless because the function immediately re-checks `process_is_running` and returns that. The actual leak is one level up — no process group — and is reported with a reproduction.
- **"`_sensitive_fetchall` is dynamic SQL."** Retired: the interpolated `{table}` is one of two literals chosen by an `if/elif` on a module constant, and the id list is a single bound `json_each(?)` parameter.
- **"`RAG_Search/pipeline_integration.py`'s broken import is a crash path."** Symptom real (the module does not exist), severity wrong: the containing module has zero importers and zero collected tests, so nothing can reach it. Kept at P3 and handed to the RAG slice.
- **"`app.py`'s six `*_server_process` class attributes are shared mutable state."** Retired: `app.py:7801-7806` are `Optional[subprocess.Popen] = None` annotations reassigned per instance at `:8062-8069`. Not the `_request_cooldown` failure mode.
- **"`TTSStreamingEvent` is dead and should be deleted."** Retired to report-only: `Tests/TTS/test_console_audio_cpp_native.py:835,1595` assert `not any(isinstance(message, TTSStreamingEvent) …)`, i.e. its *absence from the message stream* is a stated requirement. Per the brief's rule 4 that makes it a decision.
- **"`Audio_Events` star imports hide a broken re-export."** Retired: `Audio_Events/__init__.py` imports only `recording_events` and `dictation_events`, both of which import cleanly; `dictation_integration_events` is deliberately *not* re-exported, which is precisely why it is unreachable.

## Left UNVERIFIED

| claim | why not verified | literal command to run |
|---|---|---|
| A real `vllm serve` / `ollama serve` forks worker children, so the P1 process-group leak is hit in practice (the primitive's leak is reproduced; the per-launcher shape is not) | the brief forbids spawning real servers | with a real install: `ollama serve & sleep 3; pgrep -P $(pgrep -n 'ollama serve')` — any output is the grandchild `terminate_process_bounded` would orphan |
| The MLX "bare `python`" failure is user-visible on a clean install (verified in code; depends on the user's `PATH`) | this box has a venv on `PATH`, which masks it | `env -i PATH=/usr/bin:/bin /usr/bin/which python` → empty on a stock macOS; then press MLX ▸ Start in the TUI and confirm only "MLX-LM server starting…" appears with no error |
| The Transformers "Browse models dir" button shows the error toast in the live TUI (the handler is verified to always take the failure branch; the button is verified composed and mapped) | the brief forbids running the app | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d 'python3 -m tldw_chatbook.app'`, navigate to Models ▸ Transformers, click `#transformers-browse-models-dir-button`, `tmux -L verify capture-pane -p` and look for "File picker utility (textual-fspicker) not available." |
| Deleting the 9 dead modules is safe against the derived-artifact gate | `preflight.sh` regenerates artifacts and the brief is read-only | `./scripts/preflight.sh` after the deletion, then inspect the rows it names in `Docs/security/production-diagnostic-inventory.json` (1064/1085/1099/1120) before any `--write` |
| Ranges I did not read line-by-line (`chat_rag_events` 180-383 & 920-2055, `ollama` 430-1053, `stts_events` ~2,200 lines, `tts_events` ~3,000 lines, `note_ingest_events` 180-300 & 560-693) contain no further defects | slice is 16,816 lines; depth was spent on the largest symbols and on slice-wide mechanical passes that do cover those ranges for the specific defect classes | re-run this report's `ast`/`rg` sweeps against those ranges, or read `tts_events.py:1798-2521` (`_generate_tts`, 721 lines) and `stts_events.py:1956-2188` (`_persist_settings`, 232 lines) in full — the two largest unread bodies |
