# Diagnostic exit and Console presentation stalls

TASK-35612; user-authorized continuation of PR3166.

ADR required: no. ADR path: N/A. Routine fixes preserve existing ADR-126/230/231
storage, source ownership, finite-reader retirement and display boundaries.

1. Compare bounded storage-free timed/synchronous traceback stress. Preserve the
   interrupted run and distinguish a suspected diagnostic race from proof.
2. Test hidden Home completion and next-visit replay. Suppress only invisible
   presentation; keep refresh workers and native checks unchanged.
3. Test checked-display coordinator contention and fresh replay. Defer on failed
   nonblocking acquisition, never grant acceptance or fall through to native IO.
4. Scope repeated inspector derivation to one synchronous build; verify fresh
   config/session state next call and teardown on exceptions. Keep FULL's fresh
   post-await rail projection and hidden inspector update.
5. Make periodic diagnostics GIL-owned if the timer investigation warrants it;
   keep synchronous timeout dumps, record exact child exit codes, and document
   that a GIL-held hang needs an external debugger.
6. Run targeted checks and serial original/current Windows return profiles with
   identical diagnostic machinery and the Windows clock-safe polling floor.
7. Review the patch, record measurements/limitations, update PR3166, close task.

No full suite; no storage admission cache, TTL extension, reduced ownership or
marker checks, or changes to action/send/maintenance fences.
