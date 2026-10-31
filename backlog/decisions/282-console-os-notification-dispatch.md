# ADR-282: Opt-in Console OS notification dispatch

Status: Accepted
Date: 2026-10-31
Related: [TASK-18925](../tasks/task-18925%20-%20Opt-in-OS-notifications-for-background-Console-events.md), [repair plan](../../Docs/superpowers/plans/2026-10-31-task-18925-os-notifications-review-repair.md), [ADR-148](148-console-run-hooks.md)

## Context

Console already deduplicates in-app completion, failure, and parked-approval toasts. Users may want a desktop signal while another application covers the terminal. This is an additional presentation channel, with a smaller content budget than ordinary notifications or configurable run hooks.

The 2026-10-31 PR #3155 review reproduced a lifetime gap: a daemon notification thread vanished at interpreter exit before its subprocess timeout could kill and reap a slow child. It also identified that Linux notification bodies accept XML markup and that ignored exit codes made delivery failures invisible to debug diagnostics.

## Decision

- The channel is opt-in through `[console.os_notifications]`, default `enabled=false`, with independent `run_finished`, `run_failed`, and `needs_approval` flags defaulting true. Canonical Settings -> Console Behavior drafts and persists those values.
- Existing Console outcome/failure/park bridges own eligibility and cadence, including queued-run coalescing and approval-round deduplication. No second event ledger, window-focus detector, replay, or retry queue is introduced. A viewed successful completion remains silent, as its existing toast contract requires.
- macOS uses `osascript`; Linux uses `notify-send`. Windows and other platforms are unsupported and skip with debug metadata. No optional package or configurable command is added.
- Fixed argv only, without a shell. AppleScript string literals escape backslashes and quotes. Linux body text uses standard-library XML-compatible escaping: desktop notification markup supports links and images, so names must remain literal text. Reference: [freedesktop notification markup](https://specifications.freedesktop.org/notification/1.3/markup.html).
- The body contains bounded, sanitized conversation/workspace display names and a fixed outcome category. Message bodies, tool outputs, error copy, and credentials are not inputs. Display names themselves are user-controlled identity and may contain sensitive text; enabling this channel permits showing those names on the desktop.
- The subprocess inherits only PATH and, on Linux, DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR. Provider credential environment variables are excluded. Failures log category and return code or exception type; argv, display identity, stdout/stderr, and exception text are not logged.
- A non-daemon one-shot thread owns each invocation. It holds only copied identity/settings and no UI/profile resources. The UI loop does not wait. `subprocess.run` retains its 3-second timeout and kill/reap path while the interpreter exits normally. Orderly process exit can wait for the remainder of an admitted invocation; concurrent notifications wait concurrently. Forced process termination is outside that guarantee, and platform process creation itself is not interruptible by Python's timeout.
- This channel is separate from the persisted notification inbox and its policy/settings/store. It creates no inbox record and does not change that subsystem's global or per-category preferences. It also does not execute user-configured run hooks or inherit their payload authority.

## Alternatives

- A daemon thread was rejected because interpreter exit abandons its timeout owner and can strand a hung notifier.
- A process manager, durable dispatcher, or worker pool was rejected for three small, existing guarded presentation events. The standard subprocess timeout and interpreter-owned thread lifetime cover the observed defect without another service.
- Reusing the inbox dispatch path was rejected because the Console cadence, content budget, opt-in switch, and persistence semantics differ. This decision does not claim the inbox is a desktop-notification service.
- Passing raw Linux text was rejected because body markup can interpret a conversation name as formatting, a link, or an image.

## Verification and limits

Command construction and private diagnostic tests cover content boundaries. A real slow child verifies off-thread dispatch and timeout cleanup through orderly parent-interpreter exit. Mounted tests cover existing dedupe, gating, and Settings persistence. These tests do not establish that a particular desktop notification daemon or macOS notification permission setting displays a notification; native desktop delivery remains an explicit platform-validation limitation on the Windows review host.
