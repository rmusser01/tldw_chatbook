# Console transcript viewport ownership implementation plan

Goal. Make the active viewport an explicit property of the transcript surface
so a session switch can reuse an already rendered view.

Architecture. Keep the existing Console shell, transcript renderer and
store-owned activation. Route viewport consumers through the surface and
verify the lookup with a mounted viewport whose ID differs from the original.
Treat retention and the residual event-loop stalls as separately measurable
work.

Tech stack. Python 3.12, Textual 8, pytest and the existing Terminal harness.

Spec. [ADR-365](../../../backlog/decisions/365-console-transcript-viewport-ownership.md)
and the TASK-31966 acceptance criteria.

ADR required: yes
ADR path: backlog/decisions/365-console-transcript-viewport-ownership.md
Reason: separating the active viewport from its DOM identifier changes the
projection boundary shared by the screen, transcript region and controllers.

## Active viewport lookup

- [x] Mount the real transcript surface and a second viewport in a test.
  Verify region reads and native focus address the surface's active viewport
  while the original ID remains attached to the hidden viewport. Run against
  the current lookup and capture the failure.
- [x] Add the shared lookup beside the surface. Convert the existing screen,
  region, action readiness, voice and message-projection consumers. Keep the
  standalone widget lookup and CSS for the single-viewport path.
- [x] Run the mounted regression, original switcher and transcript contracts,
  and affected size checks. The mounted transcript tests use the existing
  bootstrap-profile fixture. See the QA report for broader results.

## Retention

- [x] Measure the disposable two-view control on all 12 native journeys.
  Measure a warm switch with zero row registrations. Check queued input
  while the previous viewport is hidden and disabled.
- [x] Add mounted REDs for a quote queued at the screen, A → B → A
  delivery and reuse of the exact mounted message row after a background edit.
- [x] Define an activation-scoped input boundary for transcript actions and
  focus. Test delayed actions across A → B → A, including actions already
  queued above the transcript. Prove the inactive viewport can release
  selection, capture and media work before keeping it mounted.
- [x] Keep one settled text viewport in the surface, use the existing
  projection on a hit, and publish only after rechecking the exact store
  activation. Keep stable geometry and unchanged window sizes.
- [x] Exercise background edit, close, rebind, resize and cancelled staging
  with mounted regressions and exact row identity.
- [x] Repeat all 12 Terminal journeys, including cold open and ordinary quit,
  with the original timing and resource assertions. Ordinary passed 6 of 6.
  Scale exceeded the 50 ms loop limit in all six journeys.

## Scale viewport restyles

ADR required: no new ADR
ADR path: backlog/decisions/365-console-transcript-viewport-ownership.md
Reason: a parked viewport still has no layout, focus, pointer or queued
action authority. Its native renderers, ownership and size limits stay fixed.

- [x] Reproduce the warm swap's full disabled-state restyles in a mounted
  test. Check parked focus and immediate hit-testing before the next paint.
- [x] Hide the viewport from layout and hit-testing at publication, release
  outgoing focus, and keep the activation check on queued effects. Preserve
  the incoming renderer's enabled styles and native controls.
- [x] Verify the retained interaction, exact activation and retirement tests.
  Repeat 12 Terminal journeys. Four scale intervals remain over 50 ms.

## Markdown styling during cold opens

ADR required: no new ADR
ADR path: N/A
Reason: use native registration for initial CSS in the existing Markdown
block types. Preserve roleplay components, tokens and mounted restyles.

- [x] Count native style calls on detached flavor blocks during Markdown
  construction and verify the same live block class and rendered content.
- [x] Assign the flavor class before native mount without an eager detached
  restyle. Keep immediate style updates for any attached conversion.
- [x] Run the roleplay paint, streaming and link regressions.
- [x] Remeasure the scale journeys that build new Markdown. Five scale
  intervals still exceed 50 ms in this sample.

## Synchronous readiness during activation

ADR required: no new ADR
ADR path: N/A
Reason: reuse the existing call-scoped provider-readiness memo within one
derivation. Config freshness on independent calls and send admission remain live.

- [x] Count real config reads in the existing guidance subprocess across
  session convergence, a settings save and a deliberately failing projection.
- [x] Share the existing derivation memo for the synchronous readiness call.
  Release it before publishing guidance.
- [x] Run the existing provider guidance, activation, and lifetime contracts
  before repeating the native journeys.

## Retained-view eligibility checks

ADR required: no new ADR
ADR path: backlog/decisions/365-console-transcript-viewport-ownership.md
Reason: check the same native controls and pending Markdown under the same
mounted viewport. Keep the same maximum retention and validation points.

- [x] Pin the mounted typed checks for native buttons and rendered links,
  including a plain text viewport that is eligible for reuse.
- [x] Avoid CSS selector matching while scanning the actual native controls
  and waiting for Markdown. Check exact reuse, focus, and cancellation.
- [x] Rerun the original 12 Terminal journeys with their existing limits.
  The two scale Active switches pass. Two cold opens still exceed 50 ms.

## Model readiness recovery lookup

ADR required: no new ADR
ADR path: N/A
Reason: the existing Model rail has one readiness row. Resolve that mounted
row by ID as for its sibling values. Preserve copy, error class, and live
repair when the row is remounted.

- [x] Pin a zero whole-screen selector scan for the existing Model update.
- [x] Use the native ID lookup for the existing recovery row.
- [x] Run the mounted Model tests and the same Terminal journeys. The
  Model and size tests passed. The held-send assertion also failed with
  the original callbacks. Four saved-chat journeys exceeded 50 ms.

## Saved Character presentation

ADR required: no new ADR
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: the existing committed owner finishes transcript preparation and
yields before its final readiness and synchronous dismissal checks. The
existing target, cancellation and rollback authority stays with that owner.

- [x] Compare the same six scale journeys with and without a timer yield
  immediately after transcript projection. Check the live target and
  composer, the modal's dismissal, and ordinary quit.
- [x] In a mounted test, queue a due timer at that boundary. Verify the
  switcher still owns presentation when the timer runs, and that an
  intervening activation returns through exact-runtime rollback.
- [x] Yield before moving focus and handing the committed result back.
  Recheck that the screen, store and activation are still current after
  the yield. Keep dismissal and the exact-visible proof synchronous.
- [x] Run the Character activation contracts and the same 12 Terminal
  journeys with the existing limits. The first run passed 10 of 12.
  The identical repeat passed all 12. Reproduced baseline failures and
  remaining qualification gaps are recorded in the QA report.
