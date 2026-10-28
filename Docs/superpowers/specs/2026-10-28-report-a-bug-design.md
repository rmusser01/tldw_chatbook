# Report a bug — design

Date: 2026-10-28
Status: Draft for user approval; product implementation has not started.
Baseline: freshly fetched `origin/dev`, `798f8fca75a7acba56da5f523329efa52c4a78f0`.
Builder check: completed; no important unanswered planning questions found.

## Intent and agreed requirements

A user who encounters a problem can prepare one readable, copyable report containing their description and existing diagnostic evidence. They can paste it into a GitHub issue, message, or paste site and share it themselves.

The user selected reporting from **existing evidence**, rather than starting a recording and reproducing the problem. Both exception stack traces and Console agent/tool history are in scope. Console **events, timing, and errors are included by default**; **redacted tool content is optional and off by default**.

The choices below that were not explicitly selected in conversation are design proposals, to be reviewed with this document.

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

1. **Describe and include evidence.** A required description and optional reproduction steps, expected behavior, and actual behavior. Logs and Console event facts are selected by default when available. **Include redacted tool inputs/results** is unchecked. The evidence inventory shows the source, actual range, counts, and any limitations. The user can exclude either logs or Console trace entirely.
2. **Review and copy.** Show the complete generated Markdown as editable text, with **Back**, **Copy report**, and **Close** actions. Collection does not update the review behind the user's back. Editing or removing text changes the copied report. Returning to change evidence settings explicitly regenerates the report and first warns if that would replace manual review edits.

Initial focus is Description. Required-field validation stays at that field. Escape, backdrop, Close, and app Quit use the existing safe dismissal/discard contract for user-entered text and review edits. Returning to the underlying screen restores eligible opener focus. A duplicate invocation raises the existing dialog rather than starting another collection.

No visual companion is used; the user declined it.

### Evidence context

Freeze the originating route, runtime/workspace identity, report collection timestamp, and the currently selected local Console conversation identity when reporting starts. This is the current conversation in the active runtime, even when the user launches reporting from Logs or About. It must be displayed explicitly in the dialog so the user can exclude it. Do not select an arbitrary historical conversation if no current conversation exists. To report another conversation, open it in Console before reporting. Do not gather imported shared traces or conversations from another runtime/workspace.

Conversation titles are shown locally to identify the selection but are omitted from automatic report output. Export uses report-local aliases for conversation/run/turn/event IDs. A runtime/workspace change invalidates pending collection rather than changing its owner silently. A bounded trace read may observe an in-progress run; label it as such and do not imply that all source owners were read in one atomic database snapshot.

## Report contents and limits

The report has a format marker such as `Chatbook bug report v1`, a UTC generation time, and these sections:

| Section | Contents |
| --- | --- |
| Description | User description; optional reproduction steps and expected/actual behavior. |
| Environment | Installed app version; Python, Textual, and SQLite versions; OS family/release and architecture; native terminal or served mode; originating route. Provider/model only where known for included trace events. |
| Evidence inventory | Included sources and ranges; tool-content opt-in; retained/included counts where known; omitted, unavailable, truncated, or failed sources. Do not manufacture total counts. |
| Exceptions | Recognizable Python tracebacks from the selected log tail, with sanitized diagnostic text. If none were captured, say so. This does not create a live thread dump. |
| Console trace | Chronological observed events, causal aliases, event/tool names, state, provider/model, observed timestamps/durations, and diagnostic errors. |
| Logs | Bounded recent session-log entries in chronological order, including normal messages, warnings, and errors rather than only error-level entries. |
| Optional tool content | Redacted captured tool arguments/results, associated with the same event aliases; present only after opt-in. |

Proposed limits, as named constants rather than user settings: newest 500 retained log entries, newest 500 projected Console events, at most 128 KiB of log text, 64 KiB of trace facts, 32 KiB of optional tool content, and 16 KiB of separately displayed exception text. User-authored fields together are limited to 32 KiB. The generated report has a 256 KiB UTF-8 ceiling, with room reserved for headings, environment, and limitation messages. These are byte limits; preserve valid Unicode.

Keep the newest evidence but render it oldest-to-newest. Report the earliest/latest included observed times when known. Never claim a specific time window merely because a record limit was applied. Preserve whole entries/events where possible and identify any per-entry clipping; do not silently cut a traceback. A shorter dedicated exception section can point to a complete included log entry. If that source entry was already truncated at capture, retain that fact rather than attempting recovery from other files.

The existing source-level log cap of 2,000 characters and retained-session cap of 10,000 records still apply. Report limits cannot recover information those caps or capture policies already removed. The input/read and projection work must also be bounded: do not call the legacy million-message export reader simply to discard most of its output afterward. Reuse existing bounded Console/Trace read seams; any necessary tail-reader adapter remains read-only and within existing source ownership.

Use headings and fenced text/JSON blocks that remain valid even when user/log/tool text contains backticks or tildes; choose delimiters that cannot terminate the generated block. Plain text inspection never interprets diagnostic content as Rich markup or executes/opens its links.

## Privacy and exact-copy behavior

Automatic log and exception sections reuse the application's credential/PII sanitization policy under ADR-029. Console evidence reuses the credential, PII, path, identity, source-omission, and custom-PII rules that govern existing trace collaboration. A reporting projection applies a strict field allowlist after those rules.

The Console default is narrower than the general Trace export profiles: it excludes message bodies, variants, reasoning, raw provider request/response bodies, tool arguments/results, and arbitrary `content_preview`/payload-derived summaries. Build event labels from known event metadata. Errors may contain sanitized diagnostic class/code/message text but never use an entire payload as the error summary. Tool names come from the governed diagnostic projection.

Opt-in adds only captured, redacted **tool arguments and tool results**. It does not opt in conversation messages, provider exchanges, or reasoning. Respect source fields already omitted, redacted, truncated, capture-failed, or unavailable; never rehydrate them through raw tables or files. If an existing profile cannot express the tool-only addition safely, add a small reporting adapter over its governed field decisions rather than weakening the general export profiles or inventing a second redactor.

Do not export configuration files, environment-variable dumps, credentials, hostnames, usernames, workspace roots, arbitrary endpoint URLs, conversation titles, or local absolute paths as environment facts. No diagnostic/report content is written to app logs, telemetry, a report database, or automatic draft files.

The inventory includes a short, concrete notice: recognized credentials and PII are masked, but pattern matching can miss private text; review before sharing. Turning tool content off governs the structured trace section; it cannot promise that ordinary existing log prose contains no incidental tool or conversation text. Logs can be excluded and all report text can be edited.

Generated user-authored description fields also pass through the existing sanitizer. Manual edits in Review remain editable. **Copy report** sanitizes the final candidate again: if text changes, update Review, explain that recognized sensitive text was masked, and require a second Copy press so the reviewed bytes are the copied bytes. If sanitization fails, keep the draft and disable copying with an actionable error. Never silently copy different text from the visible review.

Apply the shared redaction primitives to appropriately bounded text units while preserving report structure. Do not feed the entire Markdown report into the capped single-log-record formatter: its 2,000-character capture limit must not silently truncate a complete report. Any report-level clipping remains explicit under the report limits above.

Clipboard delivery uses the existing client-facing clipboard route in served/SSH contexts. Reuse a confirmed native delivery capability if practical, but do not introduce a new clipboard dependency or copy to the server desktop. An unacknowledged terminal/browser copy says **Copy requested**, not **Copied**; a failure leaves the selectable report available for manual copy. Clipboard transport is the only external effect, and only follows the user's Copy action.

## Integration and ownership

Keep the feature small: a focused report assembly module with plain immutable evidence input and a pure Markdown renderer, a shared modal, and thin entry-point wiring. Existing logging and trace repositories remain the evidence owners. Heavy reads, sanitization, and rendering run in an exclusive background worker. Closing the dialog cancels or retires the request; a late completion cannot update a closed dialog, another report, or another workspace. Description/review text survives partial source failures and retries.

Relevant existing seams verified on the baseline:

- `app_command_providers.py` and `TldwCli.COMMANDS`: command discovery and shared app actions.
- `UI/Logs_Window.py`, `Logging_Config.py::LogsBufferHandler`, and app `_log_records`/`_log_buffer`: bounded already-sanitized session evidence. Capture a consistent immutable tail through the logging owner; do not iterate a concurrently mutating deque unsafely or add another long-lived buffer.
- `UI/Screens/settings_screen.py` About category: canonical Settings entry point.
- `Chat/trajectory.py`, `Chat/trajectory_export.py`, `Chat/trace_export_profiles.py`, and Console/Trace snapshot readers: existing event projection and privacy decisions. Keep heavyweight trajectory imports deferred until reporting requests Console evidence, per existing packaging guards.
- `Utils/about_text.py::get_app_version`: installed-version lookup. Unknown metadata remains unknown; do not run Git or external commands against a user's workspace to discover a build revision.
- `Widgets/modal_dismissal.py`, `Widgets/confirmation_dialog.py`: guarded dismissal, dirty-state handling, and focus restoration.
- `Utils/install_clipboard.py`: existing precedent for distinguishing confirmed clipboard delivery from an unacknowledged request; avoid install-specific user copy in the new feature.

Use the design language and ADR-161 component catalog: a scrollable modal body with `.dialog-title`, labelled `.form-input`/`.form-textarea` fields, `.form-checkbox` evidence controls, `.status-area`, and a docked `.dialog-buttons` action row. At narrow terminal sizes, stack fields and keep actions reachable; at wide sizes, use the same reading order. Inherit rest/hover/focus/disabled states from canonical components. All visual values use existing `$ds-*` tokens; any genuinely missing value is added to `_variables.tcss` first. Rebuild generated CSS from source modules only. No existing token values need to change.

## ADR and task governance

ADR required: **yes**.

ADR path: a new `backlog/decisions/NNN-bug-report-evidence-and-sharing.md`, with the ID allocated from current repository history before implementation planning.

Reason: this introduces a user-controlled export contract spanning logs, trace sources, privacy policy, and clipboard output. Existing ADR-029 and ADR-080 constrain it but do not define this narrower report format, evidence-selection rule, or tool-only opt-in. The new ADR records those choices without changing their underlying capture/retention contracts. ADR-031, ADR-150, and ADR-161 govern the UI.

After design approval, create the atomic Backlog implementation task with these outcomes and link this design and the new ADR. Its Implementation Plan is added after moving it to In Progress. The next workflow is `superpowers:writing-plans`; no product code, scaffolding, or dependencies are added during this design step.

## Acceptance and verification expectations

- A report opened from a non-Console screen works with no conversation and no captured exception.
- Captured exceptions and normal log context appear, in original observed order, within disclosed limits.
- Default Console evidence includes observed events, timing, and errors while excluding payloads, content previews, conversation text, and reasoning.
- Tool opt-in adds only governed tool inputs/results; disabling it removes them. Sensitive/source-omitted fields cannot be recovered by switching the option.
- Review is editable and copy uses exactly the reviewed, sanitized text; source failures, Unicode limits, adversarial delimiters, and clipboard failures retain a usable draft.
- Concurrent logging, an active run, dialog close/reopen, and runtime/workspace changes cannot mix evidence or apply stale results.
- Entry points, keyboard-only navigation, narrow layouts, dismissal/discard, and focus restoration work through real app paths.
- No reporting action performs an upload, LLM call, config/database mutation, or automatic report-file write.

During implementation, use targeted unit/integration tests for evidence selection, privacy, bounds, rendering, and worker lifetimes, plus focused Textual UI and design-token/packaging checks. Verify the report flow live with a disposable profile and a controlled failure, then inspect the pasted bytes and missing-evidence behavior. Run the relevant formatter/linter. Full-suite testing requires the user's opt-in under repository instructions. Update Logs/user guidance and the app-shell/Console architecture references where the implementation changes their described behavior.

## Builder discretion

The builder chooses module/class names, existing bounded-reader details, exact captions, and token-backed responsive geometry. These choices must preserve the agreed output, source scope, default exclusions, byte limits, explicit tool opt-in, and review/copy contract. No additional backend service, dependency, schema migration, settings category, event recorder, or general export framework is required.
