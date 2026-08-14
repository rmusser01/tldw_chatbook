# TASK-15810 cold Library RAG first-query profile

## Result

The first real Library RAG Answer query reproduced as a CPU-bound UI loop. In the accepted, watchdog-bounded 15.001-second profile, the hottest repeated Python frame was Textual's `textual.css.match._check_selectors`: 1,099,317 calls, 1.661 seconds internal and 4.276 seconds cumulative. Its Library-side callers were repeated query-input mirroring and panel refreshes, not the retrieval or answer-provider boundary.

The leading hypothesis is a two-way input synchronization loop. `update_library_rag_query()` patches the sibling Library search input; the resulting `Input.Changed` is handled by `handle_library_search_changed()`, which patches back. Queued stale values can defeat the current-value equality guard, so the values ping-pong and rebuild RAG panel widgets. Textual then reapplies CSS millions of times.

This is a profile-first finding only. No production or test file was changed.

## Provenance and frozen corpus

- Evidence commit before this repair commit: `5e2c65f298a8f917b1759e15f527037bc061a4a6`, on branch `codex/task-15810-rag-first-query` exactly based on `origin/dev` commit `bb91fef739e88570fa5689afd3a02887add558b7` and ahead by the seven reviewed TASK-15810 evidence/plan commits.
- Import provenance command:

  ```text
  PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import pathlib, tldw_chatbook; p=pathlib.Path(tldw_chatbook.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd().resolve())'
  ```

  It resolved to `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query/tldw_chatbook/__init__.py`.
- `Docs/User_Guide/**/*.md` count: exactly 36.
- Frozen manifest: `fixture-manifest.sha256`, 36 sorted repository-relative entries.
- Manifest file SHA-256: `e0d5880faeee0f9fe28e6dffce918b57d37cee8525d8ed3696ed313c25643283`.
- The manifest, import path, model-clone manifest, symlink confinement, read-only write denial, effective config, provider gate, and origin-dev ancestry were all revalidated immediately before the two repair runs.

## Isolation fixture

Scratch root: `/tmp/tldw-task15810.9Skaet`.

Every stub, seed, profile, TUI, and acceptance command began with `/usr/bin/env -i` and exactly these names and values:

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

No cloud credential, cloud-provider, or inherited proxy variable was present.

The scratch TOML resolved `[paths].data_dir` to `/private/tmp/tldw-task15810.9Skaet/data`, distinct from and confined away from the real profile. It selected RAG profile `hybrid_basic`, loopback model `task15810-loopback`, non-streaming custom OpenAI-compatible answers, and disabled model-catalog auto-refresh.

Clean preflight before DB open and after the TUI boots proved:

```text
data_dir=/private/tmp/tldw-task15810.9Skaet/data
model_cache_dir=/private/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings
provider=custom-openai-api
credential_recovery_empty=true
run_gate_enabled=true
```

The preflight also exposed fixture plumbing: on this path `config.py`'s legacy bridge read `[API].default_api`, while the directly authored `[llm_api_settings].default_api` value alone was ignored. The scratch TOML therefore retained the mandated `[llm_api_settings]` table and added `[API].default_api = "custom-openai-api"`. This follows `backlog/docs/lessons-live-verification.md`: verify effective runtime configuration, not merely the file. It is not the TASK-15810 root cause.

### Offline model cache

- Source: `/Users/macbook-dev/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`.
- Scratch clone: `/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2`.
- Source and clone manifests matched exactly: 21 entries including relative path, entry type, symlink target, and regular-file SHA-256.
- Six source symlinks and six clone symlinks were enumerated; every target stayed within its respective model root. The copy preserved links without dereferencing.
- The clone contained the required snapshot, was non-writable, and the fresh repair-run write probe failed without creating a file.

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

The normal sandbox rejected the strict loopback bind with `PermissionError: [Errno 1] Operation not permitted`. After the reviewed safety amendment, only the exact discard-only stub, body-free health call, and exact clean-env TUI shared a narrowly escalated namespace. No seed, DB, cache preparation, profile-artifact, or report work was escalated.

The repair-run stub PID was `88756`, the sole listener on strict `127.0.0.1:19090`; the same-namespace body-free health check passed. Cleanup sent `SIGINT`; final `lsof` found no listener on port 19090, and the body-free health call failed with expected curl exit 7.

## Fresh unprofiled reproduction

The exact 100x30 unprofiled launch was:

```text
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task15810.9Skaet/home XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings NO_PROXY=127.0.0.1,localhost PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 tmux -L tldw15810-review-a new-session -d -x 100 -y 30 -s tldw15810-review-a '/bin/zsh -c "stty cols 100 rows 30; exec /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app"'
```

PID `89201` was resolved from `tmux display-message`; tmux reported `100x30` and `python3.12`. Its own pre-query handle capture was exactly:

```text
lsof -nP -p 89201 > /tmp/tldw-task15810.9Skaet/review-unprofiled-tui.before-query.lsof
```

That PID-specific capture proved zero real-config/data handles, active scratch-data handles, and all DB/WAL/SHM/Chroma/vector handles beneath the scratch root.

The Library pane showed RAG Answer mode, Notes (36) as the sole usable and selected source, top-k 15, citations enabled, active `hybrid_basic` / hybrid disclosure, provider `custom-openai-api`, and enabled Run. Query-input Enter activation was:

```text
activation_monotonic_ns=1994413106699166
activation_wall_ns=1786679184875683000
```

The immediate capture showed exactly `Evidence · top 15` and `searching · Notes…`; no modal or other control activated. The >30-second sample was:

```text
sample_monotonic_ns=1994466433327375
sample_wall_ns=1786679238203491000
elapsed=53.326628209s
process=89201 Rs+ 92.4% CPU
```

The pane was unchanged and had no Evidence row. Ctrl+Q was sent cleanly and the PID exited. External PID `41306` was absent before and after. The real config SHA-256 manifest and the sorted 417-file real-data SHA-256 manifest were each byte-for-byte identical before and after this exact unprofiled process. The before/after manifest-file hashes were respectively `58d1890eefb00c9eb671ffd2a4ae640d0a4fc1ecbc8e5e785b8d126473d3bbeb` and `ee10bc03a53dd7d5ecec82900e83fa7d30d07fce3895ca0ec0efb7ec11611469`.

## Watchdog-bounded owning-cProfile TUI

The scratch-only wrapper was:

```python
import cProfile
from pathlib import Path
import signal
import time

from tldw_chatbook.app import main_cli_runner

PROFILE = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s.pstats")
ARMED = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s.armed")
TIMING = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s.timing")

class _ProfileDeadline(KeyboardInterrupt):
    pass

profiler = cProfile.Profile()
armed = False
arm_monotonic_ns = 0
arm_wall_ns = 0
deadline_monotonic_ns = 0
deadline_wall_ns = 0
deadline_caught = False

def _deadline(_signum, _frame):
    global deadline_caught, deadline_monotonic_ns, deadline_wall_ns
    deadline_caught = True
    deadline_monotonic_ns = time.monotonic_ns()
    deadline_wall_ns = time.time_ns()
    raise _ProfileDeadline

def _arm_profile(_signum, _frame):
    global armed, arm_monotonic_ns, arm_wall_ns
    if armed:
        raise RuntimeError("profile already armed")
    armed = True
    arm_monotonic_ns = time.monotonic_ns()
    arm_wall_ns = time.time_ns()
    profiler.enable()
    signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, 15.0)
    ARMED.write_text(
        f"arm_monotonic_ns={arm_monotonic_ns}\narm_wall_ns={arm_wall_ns}\n",
        encoding="utf-8",
    )

signal.signal(signal.SIGUSR1, _arm_profile)
try:
    main_cli_runner()
except _ProfileDeadline:
    deadline_caught = True
finally:
    remaining_seconds, interval_seconds = signal.setitimer(signal.ITIMER_REAL, 0.0)
    if armed:
        profiler.disable()
        profiler.dump_stats(PROFILE)
    TIMING.write_text(
        "\n".join(
            (
                f"armed={str(armed).lower()}",
                f"deadline_caught={str(deadline_caught).lower()}",
                f"arm_monotonic_ns={arm_monotonic_ns}",
                f"arm_wall_ns={arm_wall_ns}",
                f"deadline_monotonic_ns={deadline_monotonic_ns}",
                f"deadline_wall_ns={deadline_wall_ns}",
                f"elapsed_seconds={(deadline_monotonic_ns - arm_monotonic_ns) / 1_000_000_000:.9f}",
                f"remaining_seconds={remaining_seconds:.9f}",
                f"interval_seconds={interval_seconds:.9f}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
```

The accepted owning-wrapper/tmux launch was exactly:

```text
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task15810.9Skaet/home XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings NO_PROXY=127.0.0.1,localhost PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 tmux -L tldw15810-review-c new-session -d -x 100 -y 30 -s tldw15810-review-c '/bin/zsh -c "stty cols 100 rows 30; exec /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/tldw-task15810.9Skaet/profile_tui_watchdog.py"'
```

PID `95177` was resolved from that pane. Its independent handle capture was exactly:

```text
lsof -nP -p 95177 > /tmp/tldw-task15810.9Skaet/review-profile-accepted-tui.before-query.lsof
```

It passed the same zero-real/active-scratch/all-DB-vector-scratch assertions. After the verified query was ready, the exact clean-env arm command sent `SIGUSR1`, waited for the in-process armed marker, recorded activation, and sent Enter. The profile was armed only 42.818 milliseconds before activation:

```text
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task15810.9Skaet/home XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings NO_PROXY=127.0.0.1,localhost PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 /bin/zsh -c 'kill -USR1 95177; for i in {1..100}; do test -s /tmp/tldw-task15810.9Skaet/review-first-query-15s.armed && break; sleep 0.01; done; test -s /tmp/tldw-task15810.9Skaet/review-first-query-15s.armed; cat /tmp/tldw-task15810.9Skaet/review-first-query-15s.armed; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c '\''import time; print(f"activation_monotonic_ns={time.monotonic_ns()} activation_wall_ns={time.time_ns()}")'\''; tmux -L tldw15810-review-c send-keys -t tldw15810-review-c:0.0 Enter'
```

```text
arm_monotonic_ns=1994960695082458
activation_monotonic_ns=1994960737900916
deadline_monotonic_ns=1994975696496916
elapsed_seconds=15.001414458
armed=true
deadline_caught=true
remaining_seconds=0.000000000
```

The immediate post-Enter pane showed `Evidence · top 15` and `searching · Notes…`. At the deadline the private `KeyboardInterrupt` subclass followed production `main_cli_runner()`'s clean interrupt path, returned through wrapper `finally`, canceled the timer, disabled cProfile, dumped stats, and exited without any manual/background-shell stop.

Artifact: `/tmp/tldw-task15810.9Skaet/review-first-query-15s.pstats` (1,106,982 bytes). Its privacy scan found no fixed-query text, User Guide path, authorization value, private-key marker, or token-like `sk-` value; it contains symbols and timings only. External PID `41306` was absent before and after, and both the real config and sorted 417-file real-data manifests were byte-for-byte unchanged. Their before/after manifest-file hashes again matched exactly at `58d1890eefb00c9eb671ffd2a4ae640d0a4fc1ecbc8e5e785b8d126473d3bbeb` and `ee10bc03a53dd7d5ecec82900e83fa7d30d07fce3895ca0ec0efb7ec11611469`.

### Accepted 15-second caller evidence

The Library service boundary is present:

| Frame | Profile entries | Meaning |
|---|---:|---|
| `LibraryScreen._start_library_rag_query` | 12 | coroutine resumes for one logical submit |
| `LibraryScreen._execute_library_rag_search` | 2 | one logical Textual worker |
| `run_library_rag_search` | 2 | one logical production service call |

The repeated Library/UI chain is:

| Frame | Calls | Internal | Cumulative |
|---|---:|---:|---:|
| `LibraryScreen.update_library_rag_query` | 2,476 | 0.005s | 8.703s |
| `LibraryScreen.handle_library_search_changed` | 805 | 0.001s | 0.080s |
| `LibraryScreen._patch_sibling_library_search_input` | 1,630 | 0.002s | 0.235s |
| `LibraryScreen._library_rag_panel_state` | 827 | 0.007s | 0.526s |
| `LibraryScreen._refresh_search_rag_panel_state_widgets` | 2,488 | 0.011s | 8.489s |
| `LibraryScreen._refresh_library_rag_query_status_widgets` | 2,479 | 0.015s | 4.311s |
| `LibraryScreen._apply_library_rag_scope_recovery_block` | 826 | 0.004s | 1.682s |
| `LibraryScreen._refresh_library_rag_answer_widgets` | 826 | 0.002s | 0.822s |
| `textual.css.stylesheet.Stylesheet.apply` | 2,491 | 0.851s | 3.821s |
| `textual.css.match._check_selectors` | 1,099,317 | 1.661s | 4.276s |

For `_patch_sibling_library_search_input`, cProfile attributes 825 call entries to `update_library_rag_query` and 794 to `handle_library_search_changed`, directly naming both sides of the mirror loop. The millions-per-minute `_check_selectors` rate is downstream of the repeated widget refresh/style application.

These counts are repeated query work, not merely initialization: the two mirror callers recur hundreds of times inside only 15 seconds and refresh the same query-status, scope, and answer widgets. A one-time Torch custom-op registration frame appears for 1.142 seconds internal, but it is smaller than `_check_selectors` and lacks the symmetric repeated Library caller chain.

## Rejected evidence and alternatives

- The previously reported 76.809-second owning profile violated the enforced 15-second bound and is rejected, regardless of its similar symbols.
- The first signal-watchdog repair attempt used a private `BaseException`; Textual stopped the hot loop but retained `app.run()`, so wrapper `finally` did not execute and no artifact existed. That run was rejected. Changing only the scratch exception to a private `KeyboardInterrupt` subclass matched production's existing clean-interrupt boundary and produced the accepted self-terminating dump.
- A bounded service-only cProfile used the production `run_library_rag_search()` → `LibraryLocalRagSearchService` → resolved `EnhancedRAGServiceV2` path, completed with 14 results in 5.453 seconds, and reported about 0.06 seconds in the runtime search itself. Its leading frames were one-time imports/Torch initialization, so it was rejected as root-cause evidence for the live spin.
- A TUI automation attempt opened Import media instead of firing Run and was rejected immediately.
- A combined-input attempt raised `NoMatches` before query activation and was rejected.
- One owning profile reproduced the caller chain over 26.491 seconds, but the real config fingerprint changed while unrelated PID `41306` exited; the isolation amendment required a hard rejection.
- Loopback answer latency is rejected: the UI remained at the searching stage and the production service boundary was suspended while the main thread consumed CPU in the repeated UI/style chain.
- Corpus/index failure is rejected: the frozen 36-file manifest seeded 36/0/0 and the vector metadata exactly matched all note IDs.
- A generic embedding attribution is rejected: the service-only run completed, while the accepted live profile names the immediate Library mirror callers and repeated `_check_selectors` frame.

## Root-cause hypothesis

The likely defect is stale queued `Input.Changed` values in the bidirectional RAG-query/Library-search-input mirror. Each side compares against the widget's current value, but an older queued event can arrive after the other side changed it. The handler then writes the stale value back, emits another change, and repeats. Each cycle calls `_refresh_search_rag_panel_state_widgets()` and its query-status, recovery-scope, and answer refreshes, driving Textual style reapplication and millions of selector checks.

The fix should be tested at the mirror boundary and in the real Library submit path, but no fix is proposed or applied in this profiling commit.
