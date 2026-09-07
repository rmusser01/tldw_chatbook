# Console send and refresh diagnostics

Console send failures and excessive UI refresh activity are recorded through the
normal metadata-only diagnostic sink. No separate diagnostic launcher is needed.
Use the Logs screen's **Copy all (redacted)** action after reproducing a problem.
The normal persistent application log carries the same structured events when
file logging is enabled. The descriptive **Copy visible** action has a different
privacy contract; use Copy all for the support artifact described here.

At INFO, look for `event=console_send_stage` under `diagnostics.console`:

- `ui_action`: the visible Send action was received, before parsing and gating.
- `ui_dispatch` and `ui_submit`: dispatch to the Console worker and its entry.
- `controller_submit`: a controller submission began or returned an outcome.
- `provider_resolution`: provider readiness/validation began.
- `capture_policy`: whether durable capture was enabled for this attempt.
- `durable_commit`: saving the accepted turn began, succeeded or failed.
- `trace_reservation`: pre-provider trace reservation began, succeeded or failed.
- `trace_dispatch_commit`: the trace's dispatch record was being committed.
- `provider_entry`: the provider adapter was entered. This is not proof that a
  remote server received the request or that a response completed.

A random `attempt_token` connects the UI's first submission with its worker and
provider stages. Further submissions in a queued prompt chain receive separate
tokens and event allowances. Tokens are unrelated to stored conversation,
message, workspace or provider-request identifiers. Durations are relative to
the diagnostic attempt. `dispatched` means queued for execution;
`not_dispatched` may be an empty input or a handled command, not a send error.

Failed stages are ERROR events, preserving their phase, application/Python/SQLite
versions, capture mode when already resolved, exception class, and a safe failure
category even at WARNING logging thresholds. Categories distinguish database,
storage, validation, timeout and other internal errors. Recognized fixed trace
validation codes, such as `trace_owner_unavailable`, retain that exact code;
arbitrary exception messages are never recorded. SQLite failures also carry a
numeric `sqlite_code`. App version is the running package's source version; it
is not a verified Git commit or proof that an installation is latest dev.

`event=ui_refresh_churn` under `diagnostics.ui` records excessive activity even
when the event loop remains responsive. It identifies `console_sync` or
`screen_recompose`, the count/window duration, active timers/workers and the last
observed send phase. Detection starts at 20 Console syncs/second or 4 whole-screen
recompositions/second, evaluated over the existing heartbeat window. These are
investigation thresholds, not proof of a bug. Ordinary five-per-second polling
is below the threshold. A continuous episode produces one warning and a quiet
window rearms it. This observation never cancels providers or skips UI updates.
`[diagnostics] ui_responsiveness_enabled = false` disables UI churn/stall
monitoring; ordinary Console failure diagnostics still use normal logging.

Each attempt has a 64-event allowance, and the existing monitor drain has a
64-record queue. Overflow is disclosed by `diagnostic_events_dropped`; an
incomplete log must not be interpreted as proof that the provider was never
entered. Queueing is nonblocking, writes occur off the UI loop, and normal
shutdown makes a bounded drain attempt. A process kill, unavailable sink, or
logging threshold above an event's severity can prevent recording. Diagnostics
do not change send, retry, cancellation or capture policy.

For a useful report, reproduce one fresh-conversation send and copy the logs.
State whether the capture card appeared, whether the message completed, and
whether flicker continued after the blocked state. A log with no send-stage
entries does not establish a successful send.

Governance: ADR-029's TASK-31977 amendment. Trace admission remains governed by
ADR-097 (the semantic trace ledger), section 15. No prompts, responses,
credentials, exception messages, traceback locals or private identifiers are
added to these diagnostics.
