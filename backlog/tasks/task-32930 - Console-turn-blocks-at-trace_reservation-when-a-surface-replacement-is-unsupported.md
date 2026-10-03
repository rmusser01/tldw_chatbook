---
id: TASK-32930
title: Console turn blocks at trace_reservation when a surface replacement is unsupported
status: To Do
assignee: []
created_date: '2026-09-24 20:07'
updated_date: '2026-09-24 20:07'
labels:
  - console
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An ordinary Console turn intermittently blocks at trace reservation: the user's message is durably accepted and committed, the provider is never contacted, and the turn is lost with nothing on screen naming a cause.

Observed live on 2026-09-24 at 19:48:42, in a chat with Capture On and after agent tool work in earlier turns:

```
event=console_send_stage phase=trace_reservation status=failed
  error_category=unsupported_surface_change exception_type=ValueError
event=console_send_stage phase=controller_submit status=blocked
```

The failure is per-turn, not a stuck chat: the send at 19:43:04 completed normally, the 19:48:40 turn was refused, and the next turn at 19:49:16 completed again. The user's own recovery route is blind: `Capture Off` for that chat (Console -> `c` -> Capture Off) takes the capture-off admission path, which never reserves a trace call.

The refusal itself is deliberate. `Chat/console_trace_service.py` fails closed when the incoming provider message surface differs from the recorded surface in a shape the reference-backed ledger cannot express: more than one changed entry, a change whose span exceeds `MAX_SURFACE_REPLACEMENT_SPAN`, a non-`messages_payload` domain change, or a gap in the recorded sequences. TASK-32926 intentionally did not widen that fence; it only made the refusing invariant nameable, which is why the log now carries the token instead of the generic `validation` bucket. The stopped-tool-run case that fence was shaped around is documented in TASK-32075/32184/32197.

What is missing is the reason and the way out. The refusal carries one code token, so nothing reports which entries changed, how many, or in which domains; and a refused turn leaves the accepted message with no provider reply and no actionable recovery. This task scopes making the refusal diagnosable and the lost turn recoverable, without widening the fail-closed boundary until the cause is known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refused surface replacement reports what changed - changed-entry count, replacement span, and the domains involved - through the existing content-free send diagnostic, never the message text.
- [ ] #2 A turn refused at trace reservation leaves the user an explicit recovery (retry, send without capture, cancel) instead of a silently lost turn with no provider reply.
- [ ] #3 The known stopped-tool-run case is either representable by the ledger or refused with a reason that names the surface suffix that cannot be represented, so the fence's real width is documented rather than inferred.
- [ ] #4 A regression test drives a refusal through the real reservation path and asserts the reported reason; pre-existing failures in the touched files are reproduced at baseline.
- [ ] #5 Recovery from a refused turn neither duplicates nor drops the already-committed user message.
<!-- AC:END -->
