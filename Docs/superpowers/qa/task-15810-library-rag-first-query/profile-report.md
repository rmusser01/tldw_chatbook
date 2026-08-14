# TASK-15810 cold Library RAG first-query profile

## Result

The first real Library RAG Answer query reproduced as a CPU-bound UI loop. The hottest repeated Python frame was Textual's `textual.css.match._check_selectors`: 6,694,754 calls, 11.616 seconds internal and 30.277 seconds cumulative. Its Library-side callers were repeated query-input mirroring and panel refreshes, not the retrieval or answer-provider boundary.

The leading hypothesis is a two-way input synchronization loop. `update_library_rag_query()` patches the sibling Library search input; the resulting `Input.Changed` is handled by `handle_library_search_changed()`, which patches back. Queued stale values can defeat the current-value equality guard, so the values ping-pong and rebuild RAG panel widgets. Textual then reapplies CSS millions of times.

This is a profile-first finding only. No production or test file was changed.

## Provenance and frozen corpus

- Evidence commit before the QA commit: `d4e0f66013a4d72d1d631971ca2e6167b589a6b6` (`docs(rag): harden concurrent profile isolation`), directly descended from the reviewed TASK-15810 safety amendments. At evidence close the branch was `codex/task-15810-rag-first-query`, ahead 6 and behind 19 relative to the moving `origin/dev`.
- Import provenance command:

  ```text
  PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import pathlib, tldw_chatbook; p=pathlib.Path(tldw_chatbook.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd().resolve())'
  ```

  It resolved inside `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query`.
- `Docs/User_Guide/**/*.md` count: exactly 36.
- Frozen manifest: `fixture-manifest.sha256`, 36 sorted repository-relative entries.
- Manifest file SHA-256: `e0d5880faeee0f9fe28e6dffce918b57d37cee8525d8ed3696ed313c25643283`.

## Isolation fixture

Scratch root: `/tmp/tldw-task15810.9Skaet`.

The clean environment contained only these names and values:

```text
TLDW_TEST_MODE=1
HOME=/tmp/tldw-task15810.9Skaet/home
XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data
XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config
TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models
HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings
NO_PROXY=127.0.0.1,localhost
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query
PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
LANG=en_US.UTF-8
```

All stub, seed, profile, TUI, and acceptance launches began with `/usr/bin/env -i`. No cloud credential, cloud-provider, or proxy environment variable was inherited.

The scratch TOML resolved `[paths].data_dir` to `/private/tmp/tldw-task15810.9Skaet/data`, distinct from and confined away from the real profile. It selected RAG profile `hybrid_basic`, loopback model `task15810-loopback`, non-streaming custom OpenAI-compatible answers, and disabled model-catalog auto-refresh.

Clean preflight before DB open and again after every relevant boot proved:

```text
data_dir=/private/tmp/tldw-task15810.9Skaet/data
model_cache_dir=/private/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings
provider=custom-openai-api
credential_recovery_empty=true
run_gate_enabled=true
```

The preflight also exposed fixture plumbing: on this path `config.py`'s legacy bridge read `[API].default_api`, while the directly authored `[llm_api_settings].default_api` value alone was ignored. The scratch TOML therefore retained the mandated `[llm_api_settings]` table and added `[API].default_api = "custom-openai-api"`. This follows `backlog/docs/lessons-live-verification.md`: verify the effective runtime configuration, not merely the file. It is not the TASK-15810 root cause.

### Offline model cache

- Source: `/Users/macbook-dev/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`.
- Scratch clone: `/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2`.
- Source and clone manifests matched exactly: 21 entries including relative path, entry type, symlink target, and regular-file SHA-256.
- Six source symlinks and six clone symlinks were enumerated; every target stayed within its respective model root. The copy preserved links without dereferencing.
- The clone contained the required snapshot, was made non-writable, and a normal write probe failed without creating a file.

### Corpus seed

The separate clean seed process used the production constructors and shared runtime, added and read all 36 notes, converted them through `note_index_entry()`, and awaited production `index_entries()`.

```text
notes=36
indexed=36
skipped=0
failed=0
vector_chunks=275
elapsed=13.582s
```

The metadata-only vector fetch had a `source_id` set exactly equal to the 36 retained note IDs. The seed process closed its DB and service owners. Its harness printed only aggregates, paths, IDs, and timing.

### Loopback answer stub

The stub used `ThreadingHTTPServer(("127.0.0.1", 19090), Handler)`, refused an occupied port, accepted only body-free `/health` and `/v1/chat/completions`, rejected bodies over 1 MiB, read and discarded request bytes without parsing or logging, returned one static non-streaming OpenAI-compatible JSON answer, and disabled `log_message()`.

The normal sandbox rejected the strict loopback bind with `PermissionError: [Errno 1] Operation not permitted`. After the reviewed safety amendment, only the exact discard-only stub, body-free health call, and exact clean-env TUI shared a narrowly escalated namespace. An un-escalated health call returned curl exit 7 across the namespace boundary; the escalated body-free health check succeeded. No seed, DB, cache preparation, or profile-artifact work was escalated.

The sole final stub PID was `57585`. Cleanup sent it `SIGINT`; final `lsof` found no listener on port 19090, and the body-free health call failed with expected curl exit 7.

## Real TUI reproduction and isolation

The real application ran in a true 100x30 PTY. The Library pane showed RAG Answer mode, Notes (36) as the sole usable and selected source, top-k 15, citations enabled, active `hybrid_basic` / hybrid disclosure, provider `custom-openai-api`, and an enabled Run action.

An unprofiled corroborating run activated the fixed query at `2026-08-14T02:52:56.941840Z`. At +16.193 seconds and +65.066 seconds the pane still showed `Evidence · top 15` and `searching · Notes…` with no evidence row; the process was `Rs+` at 102.1% CPU.

The final owning-cProfile TUI was PID `77532`, 100x30. Before query submission, its own `lsof` proved:

- zero handles beneath the real config or real data roots;
- active handles beneath the scratch data root;
- every DB, WAL, SHM, Chroma, and vector handle was beneath the scratch root.

An unrelated app, PID `41306`, had previously been identified as Python 3.13 from `/Users/macbook-dev/.local/share/uv/python/cpython-3.13.6-macos-aarch64-none/bin/python3.13`, cwd `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook`, with real-profile DB/log handles. A candidate profile was hard-rejected when the real config hash changed while that process exited. For the final accepted isolation interval PID `41306` was down both before and after. The real config fingerprint was byte-for-byte unchanged, and the sorted real-data fingerprint was also byte-for-byte unchanged (zero changed paths, hence an empty external-writer attribution set).

The final query-input Enter activation was:

```text
activation_monotonic_ns=1993309717515833
activation_wall_ns=1786678081462105000
```

The immediate capture showed exactly `Evidence · top 15` and `searching · Notes…`; no modal or other control activated. The final stop sample was:

```text
stop_monotonic_ns=1993386526632291
stop_wall_ns=1786678158272920000
elapsed=76.809s
process=77532 Rs+ 99.5% CPU
```

The pane was still searching with no evidence row. An intended background 18-second Ctrl+Q timer was terminated with its tool shell and never produced its timestamp; this made the accepted interval longer than the requested 15–20 seconds. The failure was detected rather than inferred, Ctrl+Q was then sent immediately, and the wrapper dumped successfully. This timing overshoot is a transparent harness deviation; it does not weaken the sustained-spin or caller evidence.

## Profiler method and accepted caller evidence

Artifact: `/tmp/tldw-task15810.9Skaet/live-first-query-final.pstats` (3,462,138 bytes).

The owning wrapper constructed `cProfile.Profile()`, enabled it before calling production `main_cli_runner()`, and in `finally` disabled and dumped the profile. The exact clean-environment TUI was launched under tmux as a 100x30 PTY. This preserved the real UI event path while naming Python frames. The final artifact privacy scan found no fixed-query text, guide path, authorization value, private-key marker, or token-like `sk-` value; it contains symbols and timings only.

The Library service boundary is present:

| Frame | Profile entries | Meaning |
|---|---:|---|
| `LibraryScreen._start_library_rag_query` | 12 | coroutine resumes for one logical submit |
| `LibraryScreen._execute_library_rag_search` | 2 | one logical Textual worker |
| `run_library_rag_search` | 2 | one logical production service call |

The repeated Library/UI chain is:

| Frame | Calls / entries | Internal | Cumulative |
|---|---:|---:|---:|
| `LibraryScreen.update_library_rag_query` | 14,488 total / 6,315 primitive | 0.048s | 26.245s |
| `LibraryScreen.handle_library_search_changed` | 4,830 | 0.004s | 0.579s |
| `LibraryScreen._patch_sibling_library_search_input` | 9,661 | 0.020s | 1.784s |
| `LibraryScreen._library_rag_panel_state` | 4,835 | 0.069s | 2.791s |
| `LibraryScreen._refresh_search_rag_panel_state_widgets` | 14,501 total / 6,327 primitive | 0.116s | 24.337s |
| `LibraryScreen._refresh_library_rag_query_status_widgets` | 14,494 total / 7,495 primitive | 0.142s | 13.981s |
| `LibraryScreen._apply_library_rag_scope_recovery_block` | 4,831 | 0.039s | 11.795s |
| `LibraryScreen._refresh_library_rag_answer_widgets` | 4,831 | 0.018s | 5.533s |
| `textual.css.stylesheet.Stylesheet.apply` | 17,482 total / 10,223 primitive | 7.518s | 19.409s |
| `textual.css.match._check_selectors` | 6,694,754 | 11.616s | 30.277s |

For `_patch_sibling_library_search_input`, cProfile attributes 4,803 call entries to `update_library_rag_query` and 4,809 to `handle_library_search_changed`, directly naming both sides of the mirror loop. The millions of `_check_selectors` calls are downstream of the repeated widget refresh/style application.

These counts are repeated query work, not merely initialization: the two mirror callers recur thousands of times and refresh the same query-status, scope, and answer widgets. One-time Torch/transformer import and registration frames also appear in the full-session profile, but do not have this symmetric, repeated Library caller chain.

## Rejected evidence and alternatives

- A bounded service-only cProfile used the production `run_library_rag_search()` → `LibraryLocalRagSearchService` → resolved `EnhancedRAGServiceV2` path, completed with 14 results in 5.453 seconds, and reported about 0.06 seconds in the runtime search itself. Its leading frames were one-time imports/Torch initialization, so it was rejected as root-cause evidence for the live spin.
- A TUI automation attempt opened Import media instead of firing Run and was rejected immediately.
- A combined-input attempt raised `NoMatches` before query activation and was rejected.
- One owning profile reproduced the caller chain over 26.491 seconds, but the real config fingerprint changed while unrelated PID `41306` exited; the isolation amendment required a hard rejection, so it was preserved only as rejected evidence.
- Loopback answer latency is rejected: the UI remained at the searching stage and the production service boundary was suspended while the main thread consumed CPU in the repeated UI/style chain.
- Corpus/index failure is rejected: the frozen 36-file manifest seeded 36/0/0 and the vector metadata exactly matched all note IDs.
- A generic embedding attribution is rejected: the service-only run completed, while the accepted live profile names the immediate Library mirror callers and the repeated `_check_selectors` frame.

## Root-cause hypothesis

The likely defect is stale queued `Input.Changed` values in the bidirectional RAG-query/Library-search-input mirror. Each side compares against the widget's current value, but an older queued event can arrive after the other side changed it. The handler then writes the stale value back, emits another change, and repeats. Each cycle calls `_refresh_search_rag_panel_state_widgets()` and its query-status, recovery-scope, and answer refreshes, driving Textual style reapplication and millions of selector checks.

The fix should be tested at the mirror boundary and in the real Library submit path, but no fix is proposed or applied in this profiling commit.
