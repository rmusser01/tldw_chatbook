# Bounded Library overview diagnosis

No product correction is justified by this comparison. The actual replacement UAT failure remains preserved and unexplained at the individual callback level.

The prior real UAT log records a 6.636-second event-loop stall followed by the Library aggregate five-second source deadline. Its traceback identifies the gather/to_thread cancellation boundary, not an individual failing source. Successful incoming-note visibility and subsequent note writes do not invalidate the overview failure.

Two wholly disposable synthetic profiles used the existing installed native package at `/private/tmp/uat-windows-observation-native/f9-native-package0/installed`. One was an ordinary initialized profile; the other used the existing native Notes/Media seed plus actual isolated archive restoration and ordinary recovery-profile CLI selection. Both mounted actual TldwCli Console and navigated to an actual LibraryScreen using the normal navigation message. No service results, guards, application deadlines, or product files were changed. Network access was blocked by the existing test guard; the existing headless native test scaffolding was used, so these are diagnostic mounted runs, not keyboard acceptance.

Final comparison:

- Ordinary: source snapshot 0.3605 seconds, no failure; actual Library loaded; child exited 0.
- Native isolated restore: source snapshot 0.7764 seconds, no failure; Notes 0.6411s, Media 0.2761s, conversations 0.6399s; all required calls returned; actual Library loaded; child exited 0.
- All six relevant installed modules match current source byte hashes: Library, ChatScreen, Console runtime, config, config binding, runtime maintenance. `source-hashes.json` records exact SHA256 values.

The first comparison also completed both snapshots without timeout (ordinary 1.9878s; restored 2.2322s), then hit an observer-only cleanup error: `observe_threads.stop()` tried to restore faulthandler while Textual's redirected stderr had no usable fileno. That attempt is preserved at `/private/tmp/uat-library-overview-comparison`. Moving only disposable observer cleanup outside the mounted context produced the clean final comparison. There were two diagnostic attempts; no further retries.

Content-free thread samples identify synchronous UI-thread work during Library entry in both ordinary/restored runs: ChatScreen `_sync_native_console_chat_ui` → rail/provider/cost refresh → `_ensure_console_chat_store` → `ConsoleRuntime.bind_canvas_native_view` → `_canvas_enabled` → `get_canvas_execution_enabled` → guarded config loading/native admission. Source locations: `UI/Screens/chat_screen.py:18493`, `:10016`; `Chat/console_runtime.py:2665`, `:2770`; `config.py:401`, `:6261`. Those are real sampled calls, but these successful runs do not prove they caused the earlier 6.636-second stall. Other Library first-mount configuration work remains possible.

The aggregate deadline is in `UI/Screens/library_screen.py:12904`; `_run_library_service_call` is at `:12508`. Its awaited elapsed time includes delayed event-loop resumption, so a long wrapper duration alone would not prove a slow database operation.

Smallest next diagnostic, if further investigation is needed: use a separately restored replacement/later-rollback lifecycle fixture with its full control journal history, retaining these per-source timings and bounded main-thread samples. Observe worker-side completion separately from loop-side return and record fixed-name durations for the sampled Console sync/config entry points. This would distinguish a slow source from an already-finished worker whose UI continuation cannot run. The current isolated fixture has less journal/history than the failing replacement fixture; it does not establish baseline equivalence. Do not increase the five-second deadline, cache policy/authority checks, or broaden Library/Canvas implementation work on this evidence.

Evidence: `summary.json`, `ordinary/home/library-phases.json.log`, `restored/home/library-phases.json.log`, each profile's `library-stacks.json.log`, `mount.log`, and restored `seed-restore.log`; disposable driver `driver.py`. Only fixed phases, fixed service names, elapsed times, bounded code/frame metadata, and error classes were emitted by observers. No source values or user UAT data were read or modified.
