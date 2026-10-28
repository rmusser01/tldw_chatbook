# ADR-252: Bug report evidence and sharing

Status: Accepted for implementation planning
Date: 2026-10-28
Related Task: [TASK-37173](../tasks/task-37173%20-%20Prepare-copyable-bug-reports-from-retained-diagnostics.md)
Supersedes: N/A

## Decision

Offer one user-reviewed Markdown bug report assembled from retained diagnostic evidence, with explicit source scope and a tool-content opt-in, and deliver it only through a user-requested clipboard copy.

1. **Retained sources remain authoritative.** Reuse app-session logs and existing Console trace/agent metadata. Reporting does not change capture, retention, or logging levels; add no report database, duplicate recorder, schema migration, upload service, or dependency.
2. **Default to diagnostic facts.** Include description, environment facts, recent logs, captured exception tracebacks, and available Console events/timing/errors. Tool arguments/results are off by default and require the explicit **Include redacted tool inputs/results** option. Source message bodies, variants, provider exchanges, and reasoning fields are never selected by the option.
3. **Source governance precedes export.** Honor immutable capture-time credential/PII masks, ownership, and omission rules under [ADR-097](097-console-reference-backed-semantic-trace-ledger.md). Use governed semantic/native trace readers; generic trajectory-export redaction is not a replacement. Unverifiable content becomes unavailable, with no fallback to raw canonical messages or old sidecar payloads. Apply the existing built-in diagnostic credential/PII sanitizer at reporting time even when capture-time PII was Off. Never weaken stored masks or claim universal anonymization.
4. **Tell the truth about scope.** Console evidence belongs to the already-selected local session and runtime/workspace. App-session logs are app-wide and cannot be represented as conversation-isolated. Opted-in tool results can quote files, documents, or other conversations; disclose this possibility. Preserve observed times and source/causal order, not a fabricated globally chronological sequence.
5. **Collect bounded immutable excerpts.** Keep at most 500 log entries and 500 projected Console events. Use repository-owned metadata-tail queries; the existing million-message Trace-screen snapshot reader is unsuitable. Bound source reads, lineage traversal, content materialization, and presentation waits. Disclose missing ancestors, source coverage, timing, and truncation. A timed-out or late read cannot replace another report's data.
6. **Keep the whole output pasteable.** Cap the complete rendered/copied document at 60,000 UTF-8 bytes, including user edits, formatting, and limitation notices. Budgets: logs 24,000; trace facts 12,000; optional tool content 8,000; exception section 4,000; user-authored fields 8,000; environment/inventory/formatting 4,000 bytes. Sanitize before clipping. Preserve valid Unicode and valid Markdown fences. Oversized manual edits remain visible with a clear copy-blocking error instead of silent deletion.
7. **Review the bytes that will be copied.** Generate an editable preview. Final sanitization that changes it updates the preview and requires another Copy press; an unchanged, validated preview is the exact clipboard payload. A sanitizer failure preserves the draft and blocks unsafe copying. Clipboard delivery that cannot be acknowledged says **Copy requested**, not **Copied**. Served/SSH sessions target the user's client rather than the server desktop.
8. **Keep reporting read-only and local.** Do not create sessions, initialize Console/agent services, bootstrap/reopen storage, mutate config/domain data, call an LLM, write report drafts automatically, or log report contents. Reuse the canonical command palette, Logs, and Settings About as entry points to one modal. Existing safe-dismissal, unsaved-edit, focus, keybinding, and design-token contracts apply.

## Context

Users need a useful description plus traces/logs they can paste into GitHub, a message, or a paste site. They selected reporting from evidence already retained, not reproduction recording. General Trace sharing profiles do not express a tool-only opt-in and do not alone apply the newer semantic trace masks. App logs also differ from a selected conversation's trace in ownership and retention.

This is an explicit user-controlled export contract. [ADR-029](029-local-private-data-boundary.md), [ADR-080](080-trace-v2-exhaustive-event-projection-and-collaboration.md), and ADR-097 continue to govern the underlying evidence. This ADR narrows selection and disclosure for reports without changing those sources.

## Alternatives considered

| Option | Why rejected or deferred |
| --- | --- |
| Start recording, reproduce, then report | Requires reproduction and a new capture workflow. The user selected retained evidence. |
| Attach the complete existing Trace export | Includes fields outside the requested default and does not supply the required tool-only choice. |
| Read whole histories and trim the rendered report | Bounds output but leaves database, memory, and privacy exposure unbounded. |
| Automatically file issues or upload to a paste service | Adds credentials, service dependencies, and transmission behavior the user did not request. |
| Export a large archive or separate compact/full reports | Adds a second format and workflow when one bounded Markdown report meets v1. |

## Consequences

- Large or legacy conversations may produce explicitly partial metadata and unavailable content; safe source governance takes precedence over detail completeness.
- V1 uses current-session logs. Previous-session/native-crash-dump recovery and reporting while the app is hard-hung are not delivered here.
- Reporting cannot infer which incident a user means from their description or prove that a recent error caused it. Included ranges and scope stay visible.
- Presentation deadlines require cooperative source retirement and stale-result rejection; a timeout alone cannot stop an arbitrary Python thread.
- No general export framework or new configuration surface is needed. Implementation lives in focused report helpers, existing evidence owners, and one shared modal.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-10-28-report-a-bug-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-10-28-report-a-bug.md)
- [ADR-031](031-tui-keybinding-and-footer-hint-conventions.md), [design language](../docs/design-language.md), and [component catalog](../docs/component-patterns.md)
