# Report a bug — design

Date: 2026-10-28
Status: Approved following the requested pre-implementation design review. Implementation plan prepared; product implementation has not started.
Baseline: freshly fetched `origin/dev`, `798f8fca75a7acba56da5f523329efa52c4a78f0`.
Builder check: completed; no important unanswered planning questions found.
Design review: checked against the same freshly fetched dev head on 2026-10-28; privacy, transport-size, source-scope, and read-bound corrections incorporated below.
Implementation task: [TASK-37173](../../../backlog/tasks/task-37173%20-%20Prepare-copyable-bug-reports-from-retained-diagnostics.md). Plan: [2026-10-28-report-a-bug.md](../plans/2026-10-28-report-a-bug.md).

## Intent and agreed requirements

A user who encounters a problem can prepare one readable, copyable report containing their description and existing diagnostic evidence. They can paste it into a GitHub issue, message, or paste site and share it themselves.

The user selected reporting from **existing evidence**, rather than starting a recording and reproducing the problem. Both exception stack traces and Console agent/tool history are in scope. Console **events, timing, and errors are included by default**; **redacted tool content is optional and off by default**.

The remaining choices are design defaults, approved in principle with this document and refined during review; they are not additional requirements independently supplied by the user.

## Goals, non-goals, and failure conditions

Goals:

- Reporting is useful for bugs anywhere in the application, including bugs with no exception or Console conversation.
- Output is a single self-contained Markdown document that also reads sensibly as plain text.
- The user can inspect and remove evidence before requesting a clipboard copy.
- Missing, incomplete, redacted, and truncated evidence is stated honestly.
- Report creation remains usable if one diagnostic source fails.

Out of scope for v1: automatic submission, GitHub authentication, paste-site integrations, uploading, reproduction recording, screenshots, database/config archives, whole-machine inventories, report history, and a separate reporting CLI. Reporting after restarting the app uses current-session logs only; recovering previous-session or native crash-dump files is deferred. Existing persistent Console history may still be available after restart, and its timestamps distinguish it from current-session logs.

Failure conditions: exporting an unrelated conversation silently; exporting structured tool bodies without opt-in; calling an arbitrary text preview a payload-free event summary; blocking the UI while collecting diagnostics; or claiming a missing traceback, timing measurement, or clipboard acknowledgement exists.

## Approaches considered

1. **Assemble an incident report from retained evidence — selected.** Reuse existing logs, tracebacks, and Console trace sources. It works after a problem occurs and requires little extra effort. Evidence is limited to what was already captured and retained.
2. **Record a reproduction.** Mark a capture interval before the user repeats the problem. This provides a clearer incident boundary but adds a recording workflow and cannot help an unreproducible incident retrospectively. Deferred.

The user selected approach 1. There is no change to ordinary logging levels, trace capture policy, or retention to support reporting.

## Proposed user experience

Use one shared report dialog, reached through `Ctrl+P` → **Report a bug…**, a **Report a bug…** action in Logs, and the same action in Settings → About. These entry points call one owner; no new shell destination or global shortcut is added. Reporting does not navigate away from or reset the underlying screen.

The dialog has two stages:

1. **Describe and include evidence.** A required description and optional reproduction steps, expected behavior, and actual behavior. **Application-wide session logs** and Console event facts are selected by default when available. **Include redacted tool inputs/results** is unchecked. Its explanatory text says that tool results can contain file, document, and other-conversation excerpts. The evidence inventory shows the source, actual range, counts, and any limitations. The user can exclude either logs or Console trace entirely.
2. **Review and copy.** Show the complete generated Markdown as editable text, with **Back**, **Copy report**, and **Close** actions. Collection does not update the review behind the user's back. Editing or removing text changes the copied report. Returning to change evidence settings explicitly regenerates the report and first warns if that would replace manual review edits.

Initial focus is Description. Required-field validation stays at that field. Escape, backdrop, Close, and app Quit use the existing safe dismissal/discard contract for user-entered text and review edits. Returning to the underlying screen restores eligible opener focus. A duplicate invocation raises the existing dialog rather than starting another collection.

### Evidence context

Freeze the originating route, runtime/workspace identity, report collection timestamp, and the currently selected local Console conversation identity when reporting starts. This is the current conversation in the active runtime, even when the user launches reporting from Logs or About. It must be displayed explicitly in the dialog so the user can exclude it. Do not select an arbitrary historical conversation if no current conversation exists. To report another conversation, open it in Console before reporting. Do not gather imported shared traces or conversations from another runtime/workspace.

Resolve this context only from already-existing runtime/session state. Reporting must not call `ensure_session()`, initialize a Console controller/agent bridge, create a conversation, reopen a database, or run schema/bootstrap work just to discover evidence. Unsaved, capture-disabled, or uninitialized sessions may have no durable trace; retain available metadata and say why details are unavailable.

Conversation titles are shown locally to identify the selection but are omitted from automatic report output. Export uses report-local aliases for conversation/run/turn/event IDs. Application logs are different: the existing buffer has no per-conversation/workspace ownership labels and includes background activity across the app session. Display and export that app-wide scope explicitly; do not present these logs as isolated to, or proven related to, the selected Console incident. Do not filter by guessing at identifiers inside log prose.

A runtime/workspace change invalidates pending collection rather than changing its owner silently. Record source collection start/end times and the log sequence watermark independently from report generation time. Once collected, the evidence is frozen. Toggling tool content only reads governed fields for those same frozen event references; it does not silently capture a newer event tail. Missing or changed sources remain unavailable. A bounded trace read may observe an in-progress run; label it as such and do not imply that all source owners were read in one atomic database snapshot.

## Report contents and limits

The report has a format marker such as `Chatbook bug report v1`, a UTC generation time, and these sections:

| Section | Contents |
| --- | --- |
| Description | User description; optional reproduction steps and expected/actual behavior. |
| Environment | Installed app version; Python, Textual, and SQLite versions; OS family/release and architecture; native terminal or served mode; originating route. In served mode these are app-server facts, not browser/client-machine facts. Provider/model only where known for included trace events. |
| Evidence inventory | Included sources and ranges; tool-content opt-in; retained/included counts where known; omitted, unavailable, truncated, or failed sources. Do not manufacture total counts. |
| Exceptions | Recognizable Python tracebacks from the selected log tail, with sanitized diagnostic text. If none were captured, say so. This does not create a live thread dump. |
| Console trace | Observed events in the existing deterministic causal order, causal aliases, event/tool names, state, provider/model, observed timestamps/durations, and diagnostic errors. |
| Logs | Bounded recent session-log entries in chronological order, including normal messages, warnings, and errors rather than only error-level entries. |
| Optional tool content | Redacted captured tool arguments/results, associated with the same event aliases; present only after opt-in. |

Limits are named constants rather than user settings: newest 500 retained log entries and newest 500 projected Console events. The **complete copied report**, including user edits, has a **60,000-byte UTF-8 ceiling**. Section budgets are 24,000 bytes for logs, 12,000 for trace facts, 8,000 for optional tool content, 4,000 for separately displayed exceptions, 8,000 for user-authored fields, and 4,000 for environment/inventory/formatting. Budget the rendered, redacted UTF-8 output, including fences and omission notices; preserve valid Unicode. Section budgets sum to the final limit. Do not silently borrow space from the user's description or omit limitation notices.

The smaller final cap supports the stated GitHub paste destination: issue creation has rejected bodies above 65,536 characters in the [Go project's documented integration incident](https://go.googlesource.com/build/+/823a5f284329c82522056e701c5b21cce25cc411). A 60,000-byte UTF-8 cap is a conservative product choice below that observed limit, not a promise about every message/paste service. A second larger report format or automatic report splitting is unnecessary for v1.

Keep the newest evidence. Render logs in captured append order and preserve the Trace projection's source/causal order under ADR-080; do not sort all records by wall-clock time. Preserve missing/non-monotonic timestamps and mark parents outside the excerpt rather than inventing them. Show observed durations at available precision; do not use the Safe-summary profile's one-second timing buckets to turn a short failure into a zero-duration event. For log timestamps, retain the captured clock text and state its app-server timezone/UTC offset when known; never relabel naive log timestamps as UTC. Report the earliest/latest included observed times when known. Never claim a specific time window merely because a record limit was applied.

Preserve whole entries/events where possible and identify any per-entry clipping; do not silently cut a traceback. A shorter dedicated exception section can point to a complete included log entry. If that source entry was already truncated at capture, retain that fact rather than attempting recovery from other files. An incident older than the retained or included tail can be absent even though the user can describe it; say so, without substituting the latest unrelated error as its cause.

The existing source-level log cap of 2,000 characters and retained-session cap of 10,000 records still apply. Report limits cannot recover information those caps or capture policies already removed. The input/read and projection work must also be bounded. The existing `UI/Console_Modules/review_selection.py::_build_trajectory_snapshot` reads up to 1,000,000 messages and other full-history sources; a small on-screen viewport does not make that reader bounded. Do not call it and discard most output afterward. Read metadata-first tails through repository-owned, read-only queries with explicit source and aggregate work limits, then project at most 500 selected events. Materialize only the selected, opted-in tool fields through governed readers. Any missing bounded query is a focused adapter in its existing repository, not a new event store. Partial/missing ancestors and source coverage must be disclosed rather than expanding the read without limit.

Use headings and fenced text/JSON blocks that remain valid even when user/log/tool text contains backticks or tildes; choose delimiters that cannot terminate the generated block. Plain text inspection never interprets diagnostic content as Rich markup or executes/opens its links.

## Privacy and exact-copy behavior

Automatic log and exception sections reuse the application's credential/PII sanitization policy under ADR-029. Console evidence first resolves through the governed trace/semantic readers, including `console_semantic_revision.py::project_semantic_revision_trace_message` where source revision references are involved, and corresponding governed artifact readers. They must honor immutable capture-time credential/PII masks and source omissions under ADR-097, including stored custom-PII provenance. The generic `trajectory_export.py` preflight alone does not establish these guarantees. Missing/invalid policy, ownership, masks, or equivalent legacy governance produces an unavailable field; there is no fallback to canonical unmasked messages, old sidecar payloads, or raw artifact tables. Metadata can still be reported when governed content is unavailable.

After source governance, a reporting projection applies a strict field allowlist and the existing path/identity export rules. Apply the existing built-in diagnostic credential/PII sanitizer to all resulting text, even when optional PII capture was Off for a historical run. Report-time built-in redaction supplements source masks; it does not change stored data or retroactively apply today's custom regex rules to old application logs or manually entered descriptions. A required sanitizer failure makes that field unavailable rather than copying it verbatim. Never present the word "redacted" as universal anonymization.

The Console default is narrower than the general Trace export profiles: it excludes message bodies, variants, reasoning, raw provider request/response bodies, tool arguments/results, and arbitrary `content_preview`/payload-derived summaries. Build event labels from known event metadata. Errors may contain sanitized diagnostic class/code/message text but never use an entire payload as the error summary. Tool names come from the governed diagnostic projection.

Opt-in adds only captured, redacted **tool arguments and tool results**. It never adds source message-body, variant, provider-exchange, or reasoning fields. This is a field-selection guarantee, not semantic isolation: a tool result can itself quote local files, documents, or other conversations (for example `search_library_rag` and `expand_document`). State that at the opt-in and in the exported inventory; the user must review those excerpts before sharing. Respect source fields already omitted, redacted, truncated, capture-failed, or unavailable; never rehydrate them through raw tables or files. If an existing profile cannot express the tool-only addition safely, add a small reporting adapter over its governed field decisions rather than weakening the general export profiles or inventing a second redactor.

Do not export configuration files, environment-variable dumps, credentials, hostnames, usernames, workspace roots, arbitrary endpoint URLs, conversation titles, or local absolute paths as environment facts. No diagnostic/report content is written to app logs, telemetry, a report database, or automatic draft files.

The inventory includes a short, concrete notice: recognized credentials and PII are masked, but pattern matching can miss private text; review before sharing. Turning tool content off governs the structured trace section; it cannot promise that ordinary existing log prose contains no incidental tool or conversation text. Logs can be excluded and all report text can be edited.

Generated user-authored description fields also pass through the existing sanitizer. Manual edits in Review remain editable. **Copy report** sanitizes the final candidate again and validates its final UTF-8 size: if text changes, update Review, explain that recognized sensitive text was masked, and require a second Copy press so the reviewed bytes are the copied bytes. Repeated sanitization must stabilize; it cannot cause an endless sequence of changed reviews. If the final text exceeds 60,000 bytes, show the byte count/limit and retain all edits for the user to shorten; do not silently trim user text. If sanitization fails, keep the draft and disable copying with an actionable error. Never silently copy different text from the visible review.

Apply the shared uncapped redaction primitives such as `Utils/log_sanitizer.py::sanitize_string` to the complete bounded candidate so multiline credentials remain detectable; preserve report structure. Do not feed the entire Markdown report into the capped single-log-record formatter: its 2,000-character capture limit must not silently truncate a complete report. Sanitize complete source units before clipping/escaping so a credential is not cut into an unrecognizable fragment. Any report-level clipping remains explicit under the report limits above.

Clipboard delivery uses the existing client-facing clipboard route in served/SSH contexts. Reuse a confirmed native delivery capability if practical, but do not introduce a new clipboard dependency or copy to the server desktop. An unacknowledged terminal/browser copy says **Copy requested**, not **Copied**; a failure leaves the selectable report available for manual copy. Clipboard transport is the only external effect, and only follows the user's Copy action.

## Integration and ownership

Keep the feature small: a focused report assembly module with plain immutable evidence input and a pure Markdown renderer, a shared modal, and thin entry-point wiring. Existing logging and trace repositories remain the evidence owners. Heavy reads, sanitization, and rendering run in an exclusive background worker. Each source has a five-second presentation deadline; a locked/unavailable database produces a partial report while description and logs remain usable. Timeout does not imply that an arbitrary thread stopped: source operations need bounded queries, cooperative retirement, and late-result rejection, with no overlapping retry against an unretired read. Do not change a shared database's global timeout settings to implement this. Closing the dialog cancels or retires the request; a late completion cannot update a closed dialog, another report, or another workspace. Description/review text survives partial source failures and retries.

Relevant existing seams verified on the baseline:

- `app_command_providers.py` and `TldwCli.COMMANDS`: command discovery and shared app actions.
- `UI/Logs_Window.py`, `Logging_Config.py::LogsBufferHandler`, and app `_log_records`/`_log_buffer`: bounded already-sanitized session evidence. Capture a consistent immutable tail through the logging owner; do not iterate a concurrently mutating deque unsafely or add another long-lived buffer.
- `UI/Screens/settings_screen.py` About category: canonical Settings entry point.
- `Chat/console_semantic_revision.py`, governed `console_trace_service.py` readers, `console_trace_redaction.py`, and `console_trace_custom_pii.py`: immutable source-governance boundary. `Chat/trajectory.py`, `trajectory_export.py`, and `trace_export_profiles.py` provide event projection, identity/path policy, and field decisions after that boundary; they are not substitutes for it. Keep heavyweight trajectory imports deferred until reporting requests Console evidence, per existing packaging guards. The new metadata-tail query must avoid the existing unbounded Trace-screen snapshot loader.
- `Utils/about_text.py::get_app_version`: installed-version lookup. Unknown metadata remains unknown; do not run Git or external commands against a user's workspace to discover a build revision.
- `Widgets/modal_dismissal.py`, `Widgets/confirmation_dialog.py`: guarded dismissal, dirty-state handling, and focus restoration.
- `Utils/install_clipboard.py`: existing precedent for distinguishing confirmed clipboard delivery from an unacknowledged request; avoid install-specific user copy in the new feature.

Use the design language and ADR-161 component catalog: a scrollable modal body with `.dialog-title`, labelled `.form-input`/`.form-textarea` fields, `.form-checkbox` evidence controls, `.status-area`, and a docked `.dialog-buttons` action row. At narrow terminal sizes, stack fields and keep actions reachable; at wide sizes, use the same reading order. Inherit rest/hover/focus/disabled states from canonical components. All visual values use existing `$ds-*` tokens; any genuinely missing value is added to `_variables.tcss` first. Rebuild generated CSS from source modules only. No existing token values need to change.

## ADR and task governance

ADR required: **yes**.

ADR path: [backlog/decisions/358-bug-report-evidence-and-sharing.md](../../../backlog/decisions/358-bug-report-evidence-and-sharing.md), allocated after checking current dev, recent local Git history, and open-PR reservations.

Reason: this introduces a user-controlled export contract spanning logs, trace sources, privacy policy, and clipboard output. Existing ADR-029, ADR-080, and ADR-097 (`097-console-reference-backed-semantic-trace-ledger.md`) constrain it but do not define this narrower report format, evidence-selection rule, or tool-only opt-in. The new ADR records those choices without changing their underlying capture/retention contracts. ADR-031, ADR-150, and ADR-161 govern the UI.

TASK-37173 is the atomic implementation task and is In Progress for planning. Its Implementation Plan was added after that status transition and links this design, ADR-358, and the detailed plan. The `superpowers:writing-plans` workflow prepares the execution handoff; no product code, scaffolding, or dependencies are added during planning.

## Acceptance and verification expectations

- A report opened from a non-Console screen works with no conversation and no captured exception.
- Captured exceptions and normal log context appear, in original observed order, within disclosed limits.
- Default Console evidence includes observed events, timing, and errors while excluding payloads, content previews, conversation text, and reasoning.
- Tool opt-in adds only governed tool inputs/results; disabling it removes them. Sensitive/source-omitted fields cannot be recovered by switching the option.
- A source whose PII capture was Off still receives report-time built-in credential/PII sanitization. Frozen custom masks remain effective, unavailable/legacy governance has no raw fallback, and tool results containing retrieved conversation excerpts carry the stated scope warning.
- Review is editable and copy uses exactly the reviewed, sanitized text; source failures, Unicode limits, adversarial delimiters, and clipboard failures retain a usable draft.
- The complete rendered and manually edited report is at most 60,000 UTF-8 bytes, including fences/notices, and post-edit oversize text is preserved with actionable validation. Multiline secrets are scrubbed before clipping, with stable repeat-copy behavior.
- Concurrent logging, an active run, dialog close/reopen, and runtime/workspace changes cannot mix evidence or apply stale results.
- App-wide log scope is explicit, source clocks are labelled honestly, causal event order survives clock skew, and trace reads remain bounded even for very large conversations or a locked database.
- Entry points, keyboard-only navigation, narrow layouts, dismissal/discard, and focus restoration work through real app paths.
- No reporting action performs an upload, LLM call, config/database mutation, or automatic report-file write.

During implementation, use targeted unit/integration tests for evidence selection, privacy, bounds, rendering, and worker lifetimes, plus focused Textual UI and design-token/packaging checks. Verify the report flow live with a disposable profile and a controlled failure, then inspect the pasted bytes and missing-evidence behavior. Run the relevant formatter/linter. Full-suite testing requires the user's opt-in under repository instructions. Update Logs/user guidance and the app-shell/Console architecture references where the implementation changes their described behavior.

## Builder discretion

The builder chooses module/class names, existing bounded-reader details, exact captions, and token-backed responsive geometry. These choices must preserve the agreed output, source scope, default exclusions, byte limits, explicit tool opt-in, and review/copy contract. No additional backend service, dependency, schema migration, settings category, event recorder, or general export framework is required.
