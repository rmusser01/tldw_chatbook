# ADR-253: Try compact discovery before plain chat

Status: Accepted
Date: 2026-10-28
Related task: TASK-37234
Amends: ADR-104 (token-budgeted disclosure)

## Evidence and decision

The Fedora trace advertised only read_agent_messages. A later guessed load_tools call succeeded, then list_directory accepted fs_* arguments and listed scratch. A production planner probe with a 4096-token verified window or a conservatively clamped unknown local window removed every tool, although discovery plus the progress reader fit.

After full direct/deferred disclosure fails its whole-request budget, try a compact deferred plan before plain chat. Retain find_tools/load_tools and already-gated progress schemas; omit optional runtime features and their operating/run-log preamble. Preserve the caller's plain/session system prompt and append concise discovery guidance and only the retained progress instructions. Measure the complete provider request with the same context window, response reserve and workspace note. If that still fails, keep the existing plain fallback. The compact plan retains its input-token ceiling for subsequent load_tools admission, which may only narrow the existing runtime budget. No limit is raised, catalog working-set replacement and all dispatch/permission/folder gates remain unchanged.

The runtime must not add progress schemas or their instructions to a frozen plain plan. Compact plans already reserve progress schemas when available; ordinary plans retain existing late progress support.

## Alternatives

Raising the unverified 4096-token limit would recreate real small-server overflow. Automatically executing guessed catalog tools would bypass disclosure. Keeping the all-or-nothing fallback unnecessarily removes usable discovery. Changing the model or requiring a larger server is not necessary when the compact request fits.

## Verification

Exercise actual Console planning and service requests with unknown and verified windows, native/fenced protocols, a real progress inbox, too-small/history-heavy requests, and direct large-window controls. Recover through find/load/execute without widening the allowed set. Separately reject cross-family file arguments before filesystem access; never reinterpret root_alias as an absolute path.
