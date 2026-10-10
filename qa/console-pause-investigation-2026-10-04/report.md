# Console pause investigation — 2026-10-04

The delays reproduce on Windows, Linux and macOS. Windows amplifies shared
blocking work considerably. [PR #3017](https://github.com/rmusser01/tldw_chatbook/pull/3017)
contains the DeepSeek contract, failed-call settlement/retry and redundant
store-getter fixes. It does not resolve these broader pauses. This investigation
changes diagnostics only, on the separate `codex/console-pause-evidence` branch.

## Method and evidence

Production source: PR head `16e5f0270e1ee02ed807abde940b25843a5b4c1d`, based on
dev `8f83422dde2a5b95da648882b8f7e09da5b41f08`.
During this investigation, concurrent retry-guard changes and a newer dev merge
advanced PR #3017 to `b825d974d78306bc4a6d9db754fedbaac27c6985` (dev
`a7d9bca5da`). The production diff from the measured commit changes only
`Chat/console_trace_service.py`. The UI, storage admission, native-file, config,
SQLite and CSS paths examined below are unchanged. Measurements remain pinned
to the earlier source; they are not a fresh whole-head timing run.
Native matrix [37228072972](https://github.com/rmusser01/tldw_chatbook/actions/runs/37228072972)
ran diagnostic revision `badbc8dd17457f1943a5f3447cb5390ef130c0d1` on Python 3.12.10.
Its OS-specific JSON timing/stacks and JUnit receipts are retained as artifacts.

The probe mounts the complete TldwCli in a fresh private profile with a real
file-backed DB. Three messages pass through the visible action, controller,
agent runtime and normalized trace capture. Only the final provider adapter
returns a fixed reply immediately. Storage, qualification, locking, permissions
and capture checks remain real. Eight credential ticks and eight keystrokes
precede the sends. The profile uses the census fixture's OpenAI settings and
synthetic key to isolate shared Console work; live DeepSeek UAT timings are
reported separately. Normal app timers remain enabled; the known independent
startup trace-GC revision race is held beyond the probe, as in existing captured
send tests. No real provider, keyring or model download is used. Targeted probe
only; no full test suite. The pytest environment also suppresses the app's
normal background screen pre-importing; this is a mounted full-app probe,
not an exact reproduction of every background activity in an interactive launch.

Observers call through, separate main/worker timings, and sample frame metadata
without reading source lines or content. **Nested seam times overlap and must
not be added.** Headless Pilot dispatch adds wall time. Machine, path depth and
instrumentation differ; this establishes reproduced delays and contributors,
not a controlled OS speed ratio.

| Native host | Send to provider entry, three sends | Maximum send-phase heartbeat lag | UI-thread storage admissions per send |
| --- | --- | --- | --- |
| Local Windows | 50.53 / 48.83 / 41.42 s | 8.27 / 8.76 / 9.13 s | 346 / 343 / 326 |
| Ubuntu 24.04 | 4.36 / 3.72 / 2.80 s | 0.53 / 0.54 / 0.33 s | 276 / 283 / 275 |
| macOS 15 | 5.92 / 5.75 / 4.93 s | 0.58 / 0.63 / 0.55 s | 316 / 331 / 298 |

Linux/macOS receipts: three users, three replies, three complete trace calls,
three response links, zero dispatch checkpoints. Local Windows also completed
three exchanges. The Windows Server runner's separate refusal is below.

The actual **uninstrumented browser UAT** on local Windows had already recorded
29.125 and 21.391 seconds to provider entry, with 5.463 and 4.839 second stalls
ending in guarded native file access. The instrumented fresh-profile numbers
above must not replace those as an estimate of ordinary user latency.

## Confirmed causes and platform scope

### Refresh storage fan-out — shared

`ChatScreen._start_console_transcript_sync_timer` runs a full Console sync every
0.2 seconds while runs are active. Recorded UI-thread callers include rail and
workspace builders reading subagent counts, settings/transcript context controls
reading branch memory, and transcript rendering acknowledging terminal receipts.
Spend refreshes also read context-policy config. Workspace availability workers
publish results by rebuilding workspace presentation and its subagent counts.
These callers appear on every native platform and incur hundreds of main-thread
admissions per short reply. A Textual async worker still runs its synchronous
parts on the event loop.

Source: [chat_screen.py](../../tldw_chatbook/UI/Screens/chat_screen.py),
`_sync_native_console_chat_ui`, `_start_console_transcript_sync_timer`,
`_run_console_config_sync`; [workspace.py](../../tldw_chatbook/UI/Console_Modules/workspace.py),
`_refresh_workspace_files_availability_snapshot`.

### Full verification and serialized admission — Windows amplification

`_acquire_storage` explicitly excludes Windows from reusable evidence. Each
admission performs full startup, qualification and scope derivation; `_scope`
does disk work twice, under the shared coordinator lock. `_Acquisition.initializing`
also serializes authority preparation per root. Both UI and executor threads
wait at these boundaries.

Over three local Windows sends, main-thread `_acquire_storage` took **88.05 s
inclusive across 1,015 calls**. Linux's equivalent was 0.76 s; macOS's 1.14 s,
with similar call counts and predominantly reusable-evidence paths. These are
different native paths and machines, not a benchmark of the operating systems.

Windows stats perform real handle, identity, NTFS/remote-volume and security
checks. Even a config-cache hit re-observes every ancestor through
`_config_file_posture`. The local probe counted **857,078 UI-thread native handle
opens** in the three sends, taking 45.81 s inclusive. Individual opens were short;
repetition and contention create the pauses. Python's `os.open` audit counted
only 72: it does not see the facade's NtCreateFile calls.

Windows exclusion is deliberate under [ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md):
NTFS change-time and provenance require proof before reuse. Do not enable the
POSIX shortcut without that evidence.

Source: [storage_admission.py](../../tldw_chatbook/Backup_Recovery/storage_admission.py),
`_acquire_storage`, `_scope`, `_Acquisition.initializing`;
[windows_files.py](../../tldw_chatbook/Utils/windows_files.py), `open_handle`,
`ntfs`, `security`; [config.py](../../tldw_chatbook/config.py), `_config_file_posture`.

### Short worker reads and repeated connection verification — shared, POSIX process cost

Character scope snapshots read authority ID and search revision before and after
each snapshot. Each `run_owned_db_call` retires its newly opened worker connection;
the next snapshot opens and verifies another. Availability/browser refreshes add
similar work, competing for shared locks even when moved off the event loop.

POSIX private SQLite preparation starts verification subprocesses: **39 / 35 / 28
helper starts** per Linux send, **36 / 29 / 25** on macOS. Their aggregate worker
times were 6.47 and 9.14 s, overlapping other tasks. Windows started no helper in
these send phases; repeated native verification dominated instead.

Source: [character_context.py](../../tldw_chatbook/UI/Console_Modules/character_context.py),
`_capture_scope`, `_read_database_scope_metadata`;
[base_db.py](../../tldw_chatbook/DB/base_db.py), `run_owned_db_call`,
`operation_owned_connection`; [private_sqlite.py](../../tldw_chatbook/DB/private_sqlite.py),
`prepare_in_helper`.

### Unchanged layout requests stylesheet work — shared, isolated control confirmed

Linux/macOS samples repeatedly reach `sync_auto_speak` → `_set_recovery_height`
→ stylesheet `apply`. The height setter removes and re-adds its class even when
visibility is unchanged. A follow-up probe compared eight unchanged calls on
the same mounted bar with an equality-guarded diagnostic control, asserting
equal classes, height, minimum height and maximum height. It does not compare
painted geometry after an awaited render. Eight ordinary calls took **689.1 ms on local Windows,
262.6 ms on Linux and 383.9 ms on macOS**. Equality checks took 18 microseconds
on Windows and approximately 9 microseconds on POSIX. This control tests the
hidden recovery row at rest after the three sends.
This directly demonstrates avoidable UI-thread work, without claiming to
account for every whole-app stall. No widget implementation was changed.

The [control run](https://github.com/rmusser01/tldw_chatbook/actions/runs/37228455116)
used revision `48a6d1e7cec8d9079657279f2442170c8cf67282`. Both POSIX hosts and the
local Windows repeat completed all three captured replies and exact receipts.
Helper-start callers explicitly identify character scope → authority metadata
→ new connection → `prepare_in_helper`. Character-context work started 43 of
Linux's 96 send-phase helpers and 51 of macOS's 100: 40 and 48 were direct
scope-metadata reads, plus three navigation/search authority initializations
on each host. Availability worker connection preparation was another measured
caller. These counts are background work overlapping sends, not exclusively
provider-preflight work.

Source: [console_control_bar.py](../../tldw_chatbook/Widgets/Console/console_control_bar.py),
`_set_recovery_height`; [textual_css_fastpath.py](../../tldw_chatbook/Utils/textual_css_fastpath.py),
`install_stylesheet_fastpath`.

### Verification misses combined sends and Windows native costs — shared test gap

The PR performance workflow is Ubuntu-only. Its storage-unit census holds
wall-clock timers still and measures typing, ticks, maintenance and visits
separately, without a three-turn captured conversation competing with normal
refreshes. Its Windows `os.open` ceiling is unset; that audit cannot count the
native opens above. [Latest dev's Linux performance check](https://github.com/rmusser01/tldw_chatbook/actions/runs/37222782278)
passed, consistently with these gaps.

Source: [perf-guard.yml](../../.github/workflows/perf-guard.yml),
[test_console_keystroke_work_census.py](../../Tests/Performance/test_console_keystroke_work_census.py),
`_census`, `_ceiling`, `test_console_storage_units_stay_within_their_ratchets`.

## Separate Windows Server refusal

The Windows 2022 runner failed before first provider entry with
`wrong_owner: unsafe_sqlite_artifact`. The traceback identifies
`tldw_chatbook_workspaces.db-wal`, mode 0600, projected owner UID 0. The Windows
shim maps UID 0 to SYSTEM/Administrators/TrustedInstaller and UID 1000 to the
token user. The privacy gate consequently refused the WAL. This is not a
successful Windows CI conversation. The exact owner SID/default-token-owner
mechanism was not collected; elevated-runner default ownership is a hypothesis.
Local Windows completed the conversation and reproduced pauses. Investigate
sidecar ownership separately and preserve the ownership gate.

## Recommended implementation order

1. Avoid unchanged layout updates, with layout-equivalence and live validation.
2. Reduce refresh storage fan-out and batch finite metadata reads. Publish
   snapshots off-loop with authority/revision/session fencing.
3. Reduce connection preparation within one finite snapshot, preserving
   retirement and maintenance ownership.
4. Qualify Windows reusable evidence under ADR-126, preserving fresh detection
   of ACL, owner, path replacement, pauses and provenance.
5. Add complete captured sends, native handle counts and process costs to bounded
   performance checks, retaining each native platform's evidence separately.

ADR required for diagnostics: **no**. ADR path: **N/A**; existing ADR-126 applies.
Runtime, storage and permissions remain unchanged. Changes to evidence reuse or
connection ownership require a separate ADR assessment.
