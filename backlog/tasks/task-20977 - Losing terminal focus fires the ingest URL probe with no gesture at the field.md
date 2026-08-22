---
id: TASK-20977
title: >-
  Losing terminal focus fires the ingest URL probe with no gesture at the field
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - egress
  - privacy
  - library-ingest
  - ui
priority: low
dependencies:
  - TASK-19556
---

## Description

Source: found live by **TASK-19556**'s reviewer while verifying that the
typing-triggered probe oracle was closed. Re-verified at `684c6aba4`.

TASK-19556 closed the case where a staged URL was probed on a timer as the user
typed. The remaining triggers are deliberate gestures: blur, Enter, Browse…, and
an explicit retry. Blur is a reasonable proxy for "the user has finished
entering this URL and moved on".

It is not only that. Textual posts `Input.Blurred` on `events.AppBlur` —
terminal focus loss — so switching away from the terminal entirely counts as
leaving the field. `handle_library_ingest_path_blurred`
(`UI/Screens/library_screen.py:26586-26612`) calls
`_trigger_library_ingest_preflight(path)` with no `allow_probe=False`, unlike
the typing path which passes it unconditionally (`:27520`). So for a user who
has opted in, alt-tabbing away with a URL staged in the ingest field causes the
application to contact that host, with no gesture aimed at the field and
nothing on screen to prompt it. Demonstrated live.

This is the same category as the oracle TASK-19556 closed — network activity
the user did not ask for at that moment — at far lower frequency, and it is
inert under the shipped default: `[library] ingest_url_preflight_probe` defaults
to `false` (`config.py:2749-2750`), so nobody is affected without opting in.

**Read the handler's history before changing it.** Its docstring
(`library_screen.py:26588-26608`) records a prior reversal of TASK-3314 AC#4
after an xhigh review and live-verify round: blur deliberately does *not* disarm
a pending Start consent, because the Start click that the confirmation copy asks
for itself blurs the path field on its way in, so the original rule made the
prescribed route unable to ever submit. That is a different concern from
probing, but it is the same handler and the same event, and it is exactly the
kind of correction that gets silently undone by someone treating blur as simple.

## Acceptance Criteria

- [ ] Losing terminal focus with a URL staged does not by itself cause a network
      probe, for an opted-in user
- [ ] A genuine within-app blur — moving to another field or control — still
      probes, so the useful trigger is not lost along with the unwanted one
- [ ] Enter, Browse… and explicit retry continue to probe
- [ ] The consent behaviour the handler's docstring documents is unchanged: a
      blur still does not disarm a pending Start consent, and the Start-click
      route still submits
- [ ] A test distinguishes an application-level focus loss from a within-app
      blur, so the two cannot be conflated again
- [ ] The default-off gate is unchanged and still verified to issue no transport
      calls and no DNS lookups when off

## Notes

Low, and only reachable behind an opt-in — but it is a real instance of the
class TASK-19556 exists to eliminate, and leaving it unrecorded would mean the
next audit of that gate re-derives it from scratch.

Whoever takes this should read the docstring history first. The handler has
already been changed once on a plausible-sounding rule that turned out to break
the flow the UI copy instructs the user to follow.
