# Windows interaction profiling — 2026-10-18

The app's own UI thread is blocking on storage/configuration work. This reproduces
with an empty, isolated profile and without a terminal renderer or provider call.
Windows native path/handle validation is prominent in those paths. Terminal choice
alone cannot explain this reproduction. There is no matched Linux/macOS run here,
so this does not quantify a Windows-versus-other-OS penalty.

Checkout: `69771c1811e734421f0bdf31ed90969f8c1f0c63`. Windows 10 build 19045,
Python 3.12.10, Textual 8.2.8, Rich 15.0.0, viewport 170 × 48. Production source
was not changed. Pre-existing working-tree changes were left in place.

## What was measured

The completed `plain-ready.json` run used the real `TldwCli`, fresh temporary
configuration/databases, splash disabled, and a dummy configured llama.cpp endpoint.
No messages were sent. The repository's network guard recorded zero attempts and
its real-profile guard recorded zero refusals. No UI test factory or test CSS cache
was used. Textual Pilot supplied the headless driver; navigation was dispatched
through the application's normal navigation message.

Times below are navigation request → the target screen's `call_after_refresh`
callback. They include UI-thread work preceding that callback, but do not prove
all deferred work has completed or measure pixels appearing in Windows Terminal.
There were three visits per destination; these are diagnostic observations, not
statistical latency guarantees.

| Destination | First measured visit | Subsequent visits |
| --- | ---: | ---: |
| Home | 1,404 ms | 52 / 40 ms |
| Console | 317 ms | 938 / 324 ms |
| Library | 2,519 ms | 110 / 120 ms |
| Settings | 2,279 ms | 2,258 / 1,862 ms |

The Console had already mounted at startup, so its first measured visit is a
resume, not a cold mount. Settings is reconstructed on each visit; Home, Console,
and Library are reusable routes. Console had 525 widgets, Settings 146, and the
tour ended with 31 CSS sources. This tour did not cross the documented 64-source
CSS parse-cache threshold.

Twelve synthetic draft keystrokes took 2.4–37.4 ms to become observable in the
composer, median 3.9 ms. This measures draft state with polling, not terminal paint
or a loaded/streaming conversation. A three-second idle sample used 0.375 process
CPU seconds and had a maximum observed 20 ms loop-timer overshoot of 153 ms.

Fresh-profile startup: import 3.19 s, constructor 13.36 s, then about 8.98 s from
constructor return to UI-ready (25.53 s combined). Migrations and first-run resource
creation are included. These are not returning-user startup timings.

## Hotspots and why they matter

### 1. Settings counts customized prompts synchronously while composing

`UI/Screens/settings_screen.py::_get_internal_prompts_customized_count` calls
`Internal_Prompts/authoring.py::customized_count`. The latter enters
`Backup_Recovery/config_participants.py::operation`, which acquires the settings
rebuild and config-file locks and enters the guarded storage lifetime.

During the Settings stages, 239 of 398 sampled main-thread stacks included
`Backup_Recovery`; 176 included `customized_count`. Many samples were lock or
condition waits, not CPU computation. These stages include the short post-refresh
settle interval, and sampling can be delayed by the GIL, so sample fractions must
not be presented as exact CPU-time percentages.

The existing screen-level memoization avoids repeated work within an instance,
but Settings gets a new instance on every visit. The source comment describing
this count as approximately 2.5 ms does not describe the measured current path.

**Measured A/B:** In a separate ready-profile session, alternate three normal
Settings visits with three visits supplied the already-computed actual count
(`0`, calculated off-thread before the tour). Only the diagnostic process patched
the display helper; no application source changed and no configuration changed
during the experiment.

| Variant | Refresh times | Median |
| --- | --- | ---: |
| Normal count | 811 / 1,040 / 931 ms | 931 ms |
| Prepared count | 707 / 838 / 665 ms | 707 ms |

This reduced the median by **223 ms / 24%**. It supports moving count acquisition
out of synchronous composition. It does not demonstrate that a permanent cache
without invalidation would be correct, or that this change alone removes all lag.
This Home/Settings-only tour was already substantially faster than the broader
tour, showing sensitivity to overlapping work and warm state.

**First fix to implement:** prepare the summary in an exclusive worker, publish a
small immutable display value, and invalidate it on prompt edits, configuration
replacement and profile/runtime changes. Keep the existing checked read inside
the worker. Show a pending state before the first result. Re-run the same A/B and
verify edits/reset/navigation still update the count.

### 2. Console attach/refresh still performs synchronous reads

Main-thread samples during Console stages included:

- `load_settings → _provider_readiness_app_config → _sync_console_chat_core_state`
  during attach reconciliation, waiting in the guarded config path.
- `list_console_unseen_marks → recompute_console_attention` during pending-decision
  projection. The marks service opens a database transaction even for an empty
  result; it runs from shell mount paths too.
- `run_console_config_sync` underneath cost-chip/rail/control refreshes.
- Active-workspace lookup during session reconciliation and skill context reads.

**Next fix area:** use the existing worker/projection patterns for display-only
reads and coalesce duplicated refresh requests. Preserve owner/revision checks
when publishing results. A warm config getter alone is insufficient when a caller
explicitly enters a guarded operation or misses the cache. Do not move database
connections across threads: create/use/retire them within their owning worker.

### 3. Library does storage work during cold screen construction

Sampled main-thread path:
`LibraryScreen.__init__ → notes_sync_runtime_owner → _construct_notes_sync_runtime_owner`
`→ get_notes_sync_state_db_path/get_chachanotes_db_path → guarded config/storage`.
The two path-helper chains accounted for 38 recovery-containing samples in Library
stages. Workspace lookups and Console background refreshes also overlapped.

**Next fix area:** defer runtime-owner/path preparation from the constructor to
the existing worker lifecycle, or prepare it before navigation when needed.
The reusable Library route already makes repeat visits much cheaper.

### 4. Windows magnifies repeated storage validation

The native adapter walks ancestors, opens Windows handles with `NtCreateFile`,
reads file identity/security metadata, and closes those handles. Recovery admission,
registry and qualification checks repeatedly reach this machinery. The broad
cProfile discovery run recorded roughly 197,000 `open_handle` calls during startup
and another 170,000 in the later navigation/cleanup window. These are process-wide
discovery counts, not calls attributable solely to the UI thread.

**Longer-term fix area:** reduce duplicate acquisition within an already-proven
operation lifetime, with the existing recovery/native ownership contracts intact.
Changing native validation lifetimes requires review against ADR-126 and its
amendments. Removing ACL, replacement or recovery checks is not a performance fix.
Start with callers that needlessly block the UI; that is a smaller intervention.

## Measurement caveats and retained evidence

- `plain-ready.json` and `plain-ready-settings-ab.json` are the completed runs.
  Their matching `*-samples.json` files contain sampled code locations, not frame
  locals, prompts or credentials. `*-stacks.txt` contains watchdog snapshots.
- `plain.json` and `profiled.json` came from earlier unconfigured-profile tours.
  All twelve route measurements completed, but their later return-to-Console
  condition failed; no typing/idle claims are based on those runs.
- `startup.prof`, `main-thread.prof`, and `hotspots.txt` are discovery aids only.
  The name `main-thread.prof` is historical: a separate minimal thread experiment
  verified that this Python 3.12.10 build records a worker-only function when
  cProfile is enabled only on main. Its call graph contains cross-thread anomalies
  and overlapping cumulative totals. Do not use its cumulative times as a budget,
  sum them, or call them main-thread CPU time. The independent sampler explicitly
  selects the main thread and corroborates the paths described above.
- Deterministic profiling also changes performance. Python's
  [profiler documentation](https://docs.python.org/3.12/library/profile.html)
  explicitly distinguishes profiling from benchmarking. The reported A/B numbers
  have cProfile disabled; the low-rate stack sampler remains enabled in both arms.
- Network/provider latency, a large real conversation history, terminal drawing,
  antivirus overhead and cross-platform differences remain unmeasured. No evidence
  here justifies antivirus exclusions or a terminal/Python upgrade as a fix.

## Reproduce

From the repository root in an ordinary user PowerShell session:

```powershell
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --settings-ab
```

The script creates a new `.tmp-windows-perf-*` directory directly under the original
home, uses the repository's Windows fixture-owner context, redirects config/data/home
and the keyring, and blocks network access. The system Temp directory and this
checkout have shared-write ancestry that the app correctly rejects for private
storage; the sandbox account also cannot traverse some native ancestors. This is
why the runs used the normal Windows account and a private temporary home ancestor.
The script retains its scratch profile for inspection and reports its exact path;
remove that exact directory only after the process exits. Re-runs overwrite the
same-named evidence files, so copy earlier evidence first if comparing revisions.
The four temporary profiles used by the recorded runs and the failed checkout
fixture were removed after completion; their measurements remain here.

Validation: two completed diagnostic runs (ready tour and alternating A/B), zero
network attempts/profile-guard refusals, plus targeted Ruff format/check and Python
compilation for the probe. No full test suite or application change was needed.

ADR required: no. ADR path: N/A. This is a diagnostic artifact, not an architecture
change. Proposed display offloading should follow existing worker/projection
contracts; any storage authority/lifetime redesign requires its own ADR review.
