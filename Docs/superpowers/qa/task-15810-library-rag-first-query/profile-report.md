# TASK-15810 cold Library RAG first-query profile

## Result

The first real Library RAG Answer query reproduced as a CPU-bound UI loop. In the replacement, watchdog-bounded 15.003-second profile, the hottest repeated Python frame was Textual's `textual.css.match._check_selectors`: 692,228 calls, 1.516 seconds internal and 3.948 seconds cumulative. Its Library-side callers were repeated query-input mirroring and panel refreshes, not the retrieval or answer-provider boundary.

Measured evidence and the root-cause hypothesis are distinct. The profile measures repeated calls through both input handlers, `_patch_sibling_library_search_input()`, panel refreshes, and Textual selector matching. Source inspection shows an asymmetric event path: `handle_library_search_changed()` always overwrites shared `_library_rag_query`, while `update_library_rag_query()` returns when its event value already equals that shared state; `_patch_sibling_library_search_input()` separately compares the target widget's current value before assigning it. The hypothesis is that stale queued `Input.Changed` values exploit that asymmetry and sustain alternating assignments. The profiler does not itself reveal event payloads or queue order.

This is a profile-first finding only. No production or test file was changed.

## Provenance and frozen corpus

- Evidence commit before this quality-review repair: `c99d9ecb397c5f8f4978d2c5fd982014eeab6b2a`, on branch `codex/task-15810-rag-first-query`. Its merge base with the moving `origin/dev` remained `bb91fef739e88570fa5689afd3a02887add558b7`; at closeout `origin/dev` was `309646d5d7e088b81be7272c0e8c37ad3704de3a` and the branch was ahead 8, behind 26. The replacement profile therefore describes the exact committed branch under review, not unverified newer code.
- Import provenance command:

  ```text
  PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import pathlib, tldw_chatbook; p=pathlib.Path(tldw_chatbook.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd().resolve())'
  ```

  It resolved to `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query/tldw_chatbook/__init__.py`.
- `Docs/User_Guide/**/*.md` count: exactly 36.
- Frozen manifest: `fixture-manifest.sha256`, 36 sorted repository-relative entries.
- Manifest file SHA-256: `e0d5880faeee0f9fe28e6dffce918b57d37cee8525d8ed3696ed313c25643283`.
- The manifest, import path, model-clone manifest, symlink confinement, read-only write denial, effective config, and provider gate were revalidated immediately before the quality-review replacement run.

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

The stub used `ThreadingHTTPServer(("127.0.0.1", 19090), Handler)` and refused an occupied port. It accepted body-free `GET /health` and bounded `POST /v1/chat/completions` only. The POST handler rejected `Content-Length` over 1 MiB, read and discarded the request bytes without parsing or logging, returned one static non-streaming OpenAI-compatible JSON answer, and disabled `log_message()`.

The normal sandbox rejected the strict loopback bind with `PermissionError: [Errno 1] Operation not permitted`. After the reviewed safety amendment, only the exact discard-only stub, body-free health call, and exact clean-env TUI shared a narrowly escalated namespace. No seed, DB, cache preparation, profile-artifact, or report work was escalated.

The quality-review replacement reused one sole strict listener, PID `12751`, on `127.0.0.1:19090`; the same-namespace body-free `GET /health` passed before both replacement attempts. Cleanup sent `SIGINT`; final `lsof` found no listener on port 19090, both the stub PID and accepted TUI PID were absent, and body-free `GET /health` failed as expected.

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

The pane was unchanged and had no Evidence row. Ctrl+Q was sent cleanly and the PID exited. External PID `41306` was absent before and after. The real config SHA-256 manifest and the sorted 417-file real-data SHA-256 manifest were each byte-for-byte identical before and after this exact unprofiled process: real config before = after, and real data before = after. The before/after manifest-file hashes were respectively `58d1890eefb00c9eb671ffd2a4ae640d0a4fc1ecbc8e5e785b8d126473d3bbeb` and `ee10bc03a53dd7d5ecec82900e83fa7d30d07fce3895ca0ec0efb7ec11611469`.

## Replacement watchdog-bounded owning-cProfile TUI

The scratch-only wrapper had SHA-256 `9e51fa2f8e2320f60bbd4cd50e2a415b8a91417475bd36a8e33cc6b120544888` and was:

```python
import cProfile
from pathlib import Path
import signal
import time

from tldw_chatbook.app import main_cli_runner

PROFILE = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s-safe.pstats")
ARMED = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s-safe.armed")
TIMING = Path("/tmp/tldw-task15810.9Skaet/review-first-query-15s-safe.timing")

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

The replacement owning-wrapper/tmux launch was exactly:

```text
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task15810.9Skaet/home XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings NO_PROXY=127.0.0.1,localhost PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 tmux -L tldw15810-quality-safe new-session -d -x 100 -y 30 -s tldw15810-quality-safe '/bin/zsh -c "stty cols 100 rows 30; exec /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/tldw-task15810.9Skaet/profile_tui_watchdog_safe.py"'
```

The exact foreground validation/arm command was:

```text
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task15810.9Skaet/home XDG_DATA_HOME=/tmp/tldw-task15810.9Skaet/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task15810.9Skaet/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task15810.9Skaet/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/tldw-task15810.9Skaet/data/default_user/models HF_HUB_CACHE=/tmp/tldw-task15810.9Skaet/data/default_user/models/embeddings NO_PROXY=127.0.0.1,localhost PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 /bin/zsh /tmp/tldw-task15810.9Skaet/atomic_arm_safe.sh
```

`atomic_arm_safe.sh` had SHA-256 `316bb6cb7071d147a2d73e2341ebbeef2eb306c4740f685d123cd6a046f01fc0`. The following is a compact transcript of its safety-critical sequence in execution order; the exact foreground command and script hash above bind the preserved source artifact:

```zsh
set -euo pipefail
session=tldw15810-quality-safe
scratch=/tmp/tldw-task15810.9Skaet
worktree=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query

test ! -e "$scratch/review-first-query-15s-safe.armed"
test ! -e "$scratch/review-first-query-15s-safe.pstats"
test ! -e "$scratch/review-first-query-15s-safe.timing"

pane_record="$(tmux -L "$session" display-message -p -t "$session":0.0 '#{pane_pid}|#{pane_current_command}|#{pane_current_path}|#{pane_width}|#{pane_height}')"
validated_pid="$(printf '%s\n' "$pane_record" | awk -F '|' '{print $1}')"
pane_command="$(printf '%s\n' "$pane_record" | awk -F '|' '{print $2}')"
pane_cwd="$(printf '%s\n' "$pane_record" | awk -F '|' '{print $3}')"
pane_width="$(printf '%s\n' "$pane_record" | awk -F '|' '{print $4}')"
pane_height="$(printf '%s\n' "$pane_record" | awk -F '|' '{print $5}')"
case "$validated_pid" in ""|*[!0-9]*) exit 41 ;; esac
test "$pane_command" = python3.12
test "$pane_cwd" = "$worktree"
test "$pane_width" = 100
test "$pane_height" = 30

ps_comm="$(ps -p "$validated_pid" -o comm= | awk '{$1=$1; print}')"
test "$ps_comm" = /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
ps_command="$(ps -p "$validated_pid" -o command=)"
printf '%s\n' "$ps_command" | grep -F "/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python $scratch/profile_tui_watchdog_safe.py" >/dev/null
process_cwd="$(lsof -a -p "$validated_pid" -d cwd -Fn | sed -n 's/^n//p')"
test "$process_cwd" = "$worktree"
lsof -nP -p "$validated_pid" > "$scratch/review-profile-safe-tui.before-query.lsof"
test -s "$scratch/review-profile-safe-tui.before-query.lsof"
! grep -F "/Users/macbook-dev/.config/tldw_cli" "$scratch/review-profile-safe-tui.before-query.lsof" >/dev/null
! grep -F "/Users/macbook-dev/.local/share/tldw_cli" "$scratch/review-profile-safe-tui.before-query.lsof" >/dev/null
grep -F "/private/tmp/tldw-task15810.9Skaet/data/" "$scratch/review-profile-safe-tui.before-query.lsof" > "$scratch/review-profile-safe-tui.scratch-handles.txt"
test -s "$scratch/review-profile-safe-tui.scratch-handles.txt"
grep -E '\.db(-wal|-shm)?$' "$scratch/review-profile-safe-tui.before-query.lsof" > "$scratch/review-profile-safe-tui.db-vector-handles.txt"
test -s "$scratch/review-profile-safe-tui.db-vector-handles.txt"
! grep -v -F "/private/tmp/tldw-task15810.9Skaet/" "$scratch/review-profile-safe-tui.db-vector-handles.txt" >/dev/null

kill -USR1 -- "$validated_pid"
for _index in {1..200}; do
    test -s "$scratch/review-first-query-15s-safe.armed" && break
    sleep 0.01
done
test -s "$scratch/review-first-query-15s-safe.armed"
activation_record="$(/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import time; print(f"activation_monotonic_ns={time.monotonic_ns()} activation_wall_ns={time.time_ns()}")')"
printf '%s\n' "$activation_record" >> "$scratch/review-profile-safe.atomic-validation.txt"
tmux -L "$session" send-keys -t "$session":0.0 Enter
sleep 2
tmux -L "$session" capture-pane -p -t "$session":0.0 -S -30 > "$scratch/review-profile-safe.immediate-pane.txt"
grep -F "Evidence" "$scratch/review-profile-safe.immediate-pane.txt" >/dev/null
grep -F "searching · Notes" "$scratch/review-profile-safe.immediate-pane.txt" >/dev/null
```

The guarded prefix resolved PID `19930` itself and validated it before `kill -USR1 -- "$validated_pid"`. Both tmux and `lsof` reported cwd `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query`; tmux reported `python3.12`, `100x30`, and `ps` reported the exact venv interpreter and watchdog command. PID-specific `lsof` found zero real-profile handles, 57 scratch-data handles, and 56 DB/vector-pattern handle entries, all beneath the scratch root. Its persisted pre-signal/activation transcript was:

```text
validated_pid=19930
pane_command=python3.12
ps_comm=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
pane_cwd=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query
process_cwd=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-15810-rag-first-query
width=100
height=30
real_handle_count=0
scratch_handle_count=57
db_vector_handle_count=56
activation_monotonic_ns=1996840384788583
activation_wall_ns=1786681612155845000
```

The profile was armed 71.333 milliseconds before query-input Enter activation:

```text
arm_monotonic_ns=1996840313455791
activation_monotonic_ns=1996840384788583
deadline_monotonic_ns=1996855316297333
elapsed_seconds=15.002841542
armed=true
deadline_caught=true
remaining_seconds=0.000000000
```

The visible post-Enter capture showed RAG Answer mode and `Searching…`, proving activation. It did not show the lower, off-viewport `Evidence` or `searching · Notes…` text, so the two extra post-signal `grep` assertions returned shell exit 1. This nonzero status occurred only after PID validation, `SIGUSR1`, armed-marker confirmation, Enter activation, and pane capture; it did not weaken or bypass any safety check. The in-process deadline still returned through wrapper `finally`, canceled the timer, disabled cProfile, dumped stats, and exited. This artifact replaces the earlier literal-PID profile for accepted caller evidence.

Artifact: `/tmp/tldw-task15810.9Skaet/review-first-query-15s-safe.pstats` (1,072,488 bytes), SHA-256 `e9f11548b3b1a7d95e43294271455be098df67fb721719c67fbebe4768491332`. Its privacy scan found no fixed-query text, User Guide path, authorization value, private-key marker, or token-like `sk-` value; it contains symbols and timings only. External PID `41306` was absent before and after. The real config manifest before = after, both with manifest-file SHA-256 `58d1890eefb00c9eb671ffd2a4ae640d0a4fc1ecbc8e5e785b8d126473d3bbeb`. The sorted 417-file real-data manifest before = after, both with manifest-file SHA-256 `ee10bc03a53dd7d5ecec82900e83fa7d30d07fce3895ca0ec0efb7ec11611469`.

### Replacement 15-second caller evidence

The Library service boundary is present:

| Frame | Profile entries | Meaning |
|---|---:|---|
| `LibraryScreen._start_library_rag_query` | 12 | coroutine resumes for one logical submit |
| `LibraryScreen._execute_library_rag_search` | 2 | one logical Textual worker |
| `run_library_rag_search` | 2 | one logical production service call |

The repeated Library/UI chain is:

| Frame | Calls | Internal | Cumulative |
|---|---:|---:|---:|
| `LibraryScreen.update_library_rag_query` | 1,556 | 0.006s | 8.757s |
| `LibraryScreen.handle_library_search_changed` | 525 | 0.001s | 0.151s |
| `LibraryScreen._patch_sibling_library_search_input` | 1,043 | 0.003s | 0.362s |
| `LibraryScreen._library_rag_panel_state` | 520 | 0.008s | 0.321s |
| `LibraryScreen._refresh_search_rag_panel_state_widgets` | 1,568 | 0.015s | 8.511s |
| `LibraryScreen._refresh_library_rag_query_status_widgets` | 1,559 | 0.018s | 3.626s |
| `LibraryScreen._apply_library_rag_scope_recovery_block` | 519 | 0.005s | 1.597s |
| `LibraryScreen._refresh_library_rag_answer_widgets` | 519 | 0.003s | 0.832s |
| `textual.css.stylesheet.Stylesheet.apply` | 1,570 | 0.800s | 3.459s |
| `textual.css.match._check_selectors` | 692,228 | 1.516s | 3.948s |

For `_patch_sibling_library_search_input`, cProfile attributes 496 call entries to `update_library_rag_query` and 517 to `handle_library_search_changed`, directly naming both asymmetric event paths. The millions-per-minute `_check_selectors` rate is downstream of repeated widget refresh/style application.

These counts are measured repeated query work, not merely initialization: both handlers recur hundreds of times inside only 15 seconds and the guarded canvas path repeatedly refreshes the same query-status, scope, and answer widgets. One-time Torch/transformer setup lacks this repeated Library caller chain.

## Rejected evidence and alternatives

- The predecessor 15.001-second profile named the same hot chain, but its arm transcript used a literal `kill -USR1 95177` rather than resolving and validating the live pane PID in the same foreground command. It is replaced, not combined with the accepted counts above.
- The first atomic quality-review attempt resolved and validated PID `12981` safely and dumped the same caller chain at 15.004 seconds. Its post-Enter pane was sampled after only 0.25 seconds and had not rendered the visible searching state, so that artifact was preserved as rejected replacement evidence. Real config before = after and real data before = after for that attempt.
- The previously reported 76.809-second owning profile violated the enforced 15-second bound and is rejected, regardless of its similar symbols.
- The first signal-watchdog repair attempt used a private `BaseException`; Textual stopped the hot loop but retained `app.run()`, so wrapper `finally` did not execute and no artifact existed. That run was rejected. Changing only the scratch exception to a private `KeyboardInterrupt` subclass matched production's existing clean-interrupt boundary and enabled the later self-terminating dumps; PID safety and isolation were evaluated separately.
- A bounded service-only cProfile used the production `run_library_rag_search()` → `LibraryLocalRagSearchService` → resolved `EnhancedRAGServiceV2` path, completed with 14 results in 5.453 seconds, and reported about 0.06 seconds in the runtime search itself. Its leading frames were one-time imports/Torch initialization, so it was rejected as root-cause evidence for the live spin.
- A TUI automation attempt opened Import media instead of firing Run and was rejected immediately.
- A combined-input attempt raised `NoMatches` before query activation and was rejected.
- One owning profile reproduced the caller chain over 26.491 seconds, but the real config fingerprint changed while unrelated PID `41306` exited; the isolation amendment required a hard rejection.
- Loopback answer latency is rejected: the UI remained at the searching stage and the production service boundary was suspended while the main thread consumed CPU in the repeated UI/style chain.
- Corpus/index failure is rejected: the frozen 36-file manifest seeded 36/0/0 and the vector metadata exactly matched all note IDs.
- A generic embedding attribution is rejected: the service-only run completed, while the replacement live profile names both Library input handlers, the guarded sibling patch, repeated panel refresh, and `_check_selectors`.

## Root-cause hypothesis

The source-verified mechanism is asymmetric. Let `R` be the rail input, `C` the canvas RAG input, and `S` the shared `_library_rag_query` state:

1. `handle_library_search_changed(Rx)` unconditionally assigns `S = Rx`, then asks `_patch_sibling_library_search_input()` to assign `C = Rx` only when the canvas widget currently differs.
2. `update_library_rag_query(Cx)` first returns when `Cx == S`. Otherwise it assigns `S = Cx`, asks the same helper to assign `R = Cx` only when the rail widget currently differs, resets in-flight status, and refreshes the RAG panel.
3. `_patch_sibling_library_search_input()` guards only the target widget assignment; it does not compare the incoming event with shared state or cancel older queued events.

The following stale-event sequence is the explicit root-cause hypothesis, not a measured queue trace. Suppose the widgets and state currently contain `B`, while the queue still contains rail events `Changed(A)` then `Changed(B)` from earlier assignments:

1. Rail `Changed(A)` sets `S=A`, changes `C` from `B` to `A`, and queues canvas `Changed(A)`.
2. The already queued rail `Changed(B)` sets `S=B`, changes `C` back to `B`, and queues canvas `Changed(B)`.
3. Canvas `Changed(A)` now sees `A != S(B)`, sets `S=A`, changes `R` from `B` to `A`, queues rail `Changed(A)`, and refreshes the panel.
4. Canvas `Changed(B)` sees `B != S(A)`, sets `S=B`, changes `R` back to `B`, queues rail `Changed(B)`, and refreshes again.
5. The replenished rail pair repeats the cycle.

What is measured is the resulting recurrence: 525 rail-handler calls, 1,556 canvas-handler calls, 1,043 guarded sibling-patch calls split across both handlers, 1,568 panel refreshes, and 692,228 selector checks in 15.003 seconds. The exact stale values and queue order remain a hypothesis to test at the mirror boundary.

The fix should be tested at the mirror boundary and in the real Library submit path, but no fix is proposed or applied in this profiling commit.
