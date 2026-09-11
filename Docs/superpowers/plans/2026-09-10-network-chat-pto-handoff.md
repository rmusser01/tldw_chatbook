# Network Chat / tldw-pydle — PTO handoff

Date: 2026-09-10
Reviewed Chatbook base: `3afa68f1b99103ec8b5b5f19921ac480ad9344ac` (`origin/dev`)
Scope: Plan reassessment and contributor handoff only; no runtime implementation.

## Start here

The fork remains the selected direction. Maintaining it is a core reliability
project, not a small dependency bump. Start with **TASK-21601**, then take one
task at a time from the [eight-task execution plan](2026-08-23-tldw-pydle-first-release.md#task-sequence).
All eight tasks remain **To Do**, unassigned, with unchecked acceptance criteria.
No fork release, Chatbook IRC adapter, Network Chat screen, or tldw_server IRC
service is delivered by this PR.

Read in this order:

1. This handoff: current facts, blockers and first actions.
2. [ADR-148](../../../backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md): approved product and fork boundary.
3. [ADR-149](../../../backlog/decisions/149-network-chat-handoff-reliability-amendments.md): proposed corrections for PR review, not silently accepted policy.
4. [Design](../specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md): full ownership and feature contracts.
5. Your task and its section in the [execution plan](2026-08-23-tldw-pydle-first-release.md).

ADR required: yes

ADR path: `backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md`;
`backlog/decisions/149-network-chat-handoff-reliability-amendments.md`

Reason: Preserve the approved dependency/runtime boundary and separately review
corrections to async ownership, correlation, security and release contracts.

## What the reassessment found

| Finding | Change in this handoff |
| --- | --- |
| Callback backpressure could deadlock a callback awaiting WHOIS | Reducer never waits for callback capacity; overload closes visibly. |
| A timed-out query's late replies could resolve a new query | Bounded tombstones quarantine correlation keys until terminal replies or teardown. |
| A lock alone does not bound queued outbound work | Bound admission bytes/entries and reserve control capacity for PONG. |
| Arbitrary callbacks cannot be forcibly stopped inside asyncio | Bound yielding stragglers honestly; fresh client per connection; no sandbox claim. |
| CAP draft assigned continuation/value grammar to ACK/NAK | Correct LS/LIST and LS/NEW grammar; authenticate before ready. |
| Identical messages could be mistaken for duplicate echoes | Do not correlate by text/time; history cannot change live presence. |
| Current dev reuses some screens and now requires Python 3.12 | Explicit suspend/resume tests; keep the fork's independent 3.11 floor. |
| Mechanical packaging prematurely removed SASL dependencies | Defer dependency removal to the tested PLAIN replacement. |
| Recording a publishing blocker could count as programme completion | Publication remains unchecked and unfinished until retrievable artifacts exist. |
| ADR-084 was claimed by other work while this draft was unlanded | Preserve the decision as ADR-148; proposed changes are ADR-149. |

## Verified upstream state and limits of the evidence

Canonical upstream is **Codeberg**, not the GitHub mirror. Read-only ref checks
and the Codeberg comparison API on 2026-09-10 established:

- Approved fork point: `4efcc3b5096536668dfe772461f19a17f1ddd84e`.
- Upstream v1.1.0 tag: `cbc18112e141d357abfb5828262cd98d65628096` (unchanged).
- Current develop: `e27b6a2138c81061c2e6a937526fceaedb94d31a`.
- Eight commits after the approved point include a stalled-write fix,
  fixture cleanup, formatting/lint/workflow/link changes and a commit titled
  `Version 1.2.0`. That title alone is not evidence of a published artifact.
- The stalled-write fix is commit `7c7f5c73bf14b8207a551a6cbf38627984cb7f4e`;
  do not tell contributors PR 196 is still merely an open proposal.

The [exact comparison](https://codeberg.org/shiz/pydle/compare/4efcc3b5096536668dfe772461f19a17f1ddd84e...e27b6a2138c81061c2e6a937526fceaedb94d31a)
is inventory evidence, **not** a fresh behavioral audit. This PR does not claim
that the new develop tip passes the old spike or fixes every known problem.
Begin at the approved SHA; propose a base bump separately if the audit warrants it.

The August draft reports 53 upstream tests, 9 selection passes/2 skipped live
cases, and 2 Ergo passes. Those were historical editable-checkout feasibility
results; they have not been rerun here. Their unpublished spike files are not
required to follow this plan. Rebuild deterministic cases in the fork from the
listed scenarios and prove them against the actual release artifact. Do not
use old task IDs 15700/15701 or ADRs 057/058: current dev assigns them elsewhere.

## Prerequisites and ownership during PTO

| Item | Current evidence | Owner/action |
| --- | --- | --- |
| GitHub fork `rmusser01/tldw-pydle` | Authenticated lookup could not resolve it on 2026-09-10 | Repository owner checks again, creates it if absent, grants maintainers access and configures develop protection. |
| Behavioral amendments | ADR-149 proposed | Reviewer accepts or records changes before the affected implementation task starts. |
| PyPI project/environment | Not inspected or configured by this PR | Release maintainer obtains project access and configures Trusted Publishing before TASK-21608. Never paste a token into a task. |
| Artifact compatibility fixtures | Historical prototypes only | Task owners implement deterministic/TLS fixtures and pin exact AgentIRC/Ergo versions and hashes. |
| tldw_server IRC endpoint | No implemented contract verified here | Server owner designs service/auth contract later; not a blocker for public IRC or fork work. |

Assignees are intentionally not invented. Claim a task in Backlog and link the
fork PR in that task; check for an existing claim/PR first. At handoff, record
branch, commit, exact failing/passing commands and the next uncompleted step.
Do not depend on the original author's checkout, credentials, stash or memory.

## First contributor session

- [ ] Confirm the handoff PR and applicable ADR decisions have been reviewed.
- [ ] Check the task's status, assignee and existing fork PRs before taking it.
- [ ] Have an authorized owner establish the remote/access if needed. Check
      before creating anything; never replace a repository another contributor
      has already initialized.
- [ ] Clone canonical Codeberg with full history outside Chatbook source. Keep
      Codeberg as upstream and add the GitHub fork as origin.
- [ ] Verify the pinned SHA and branch fork develop explicitly from it, not
      from moving upstream develop. Do not mirror-push every branch/tag.
- [ ] Disable/review copied automation before enabling workflows on the fork.
- [ ] Mark only TASK-21601 In Progress and add its task-local Implementation
      Plan, including ADR required/path/reason, before implementation.
- [ ] Preserve LICENSE, record UPSTREAM_BASE and NOTICE, make the namespace
      migration separately, then build/inspect/install both distributions.
- [ ] Open one fork PR for the task and update this repository's task with its
      evidence. Never mark later tasks Done based on a predecessor's tests.

Backlog tasks are the source of truth in Chatbook; code and fork CI live in the
fork repository. Do not copy the fork into Chatbook or duplicate competing task
boards. The [Backlog tooling lessons](../../../backlog/docs/lessons-backlog-hygiene.md)
document a five-digit task editing bug: verify the printed path and resulting
diff after CLI edits, and use the documented direct-file fallback if affected.

## Required regression scenarios added by this review

Use explicit events/barriers rather than arbitrary sleeps. Each task retains
the focused test commands and file owners in the execution plan.

| Task | Deterministic setup | Required observation |
| --- | --- | --- |
| 21602 | Fill application outbound admission; inject server PING | Capacity stays bounded; PONG/control processing succeeds or a typed terminal failure occurs, never a hidden indefinite wait. |
| 21603 | Two callers await close; cancel one during cleanup | Cleanup remains owned; the other receives the truthful final result; no detached protocol work. |
| 21603 | Callback suppresses cancellation but yields; separately attempt a public command after close | Close reports retained callback; command is refused; replacement Client remains untouched. |
| 21604 | Callback awaits WHOIS; fill callback queue before delivering 318 | Query succeeds before overflow or fails via explicit slow-consumer close; never deadlocks. |
| 21604 | Timeout WHOIS A; attempt A again; deliver old 311/318 | Second A is refused until the old terminal, and old results never reach a new caller. |
| 21604 | Coalesce presence at sequences 1 and 3 with a message at 2 | New state at 3 does not appear ahead of message 2. |
| 21605 | Send 001 before mandatory SASL completes; repeat without mandatory capabilities on a no-CAP server | First attempt fails closed; legacy attempt can become ready; readiness emits once. |
| 21605 | Cancel one of two wait_ready callers | Other caller still receives readiness or the same terminal failure. |
| 21606 | Cancel LIST; omit 323 past drain deadline; interleave chat and request another LIST | Caller settles; chat works; second LIST refused; eventual 323 clears the tombstone. |
| 21607 | Send identical messages twice and receive two echoes; deliver historical membership frames | Both legitimate messages survive; history does not rewrite live membership. |
| 21608 | Build once, distribute wheel to clean target environments | Recorded hashes identify the exact tested/published bytes; no editable-import leakage. |

Tag budgets follow the [IRCv3 specification](https://ircv3.net/specs/extensions/message-tags.html):
incoming tag section is at most 8191 bytes including `@` and its trailing space;
client-sent tag data is at most 4094 bytes excluding those delimiters. The
non-tag message keeps its independent 512-byte budget including CRLF. Test
encoded boundaries and fragmented input before semantic processing.

## Scope checkpoints and next work

After TASK-21604, review the real diff and deterministic evidence against
ADR-148's reconsideration triggers. If ordering requires pervasive feature
rewrites, record the concrete cost and reconsider the core before investing in
history. Do not promise a release date before this checkpoint. The first
release still includes the approved IRCv3 semantics; an intermediate wheel is
not permission to integrate an unstable client into Chatbook.

The next **separately filed and reviewed** Chatbook tranche is:

1. Protocol-neutral models and fake adapter; no fork imports in UI or models.
2. App-owned session manager, generation fencing, bounded buffers, reconnect
   policy and shutdown through the existing composition root.
3. Private non-secret profiles/favorites and credential references; follow
   current local-private-data rules rather than placing secrets in TOML.
4. Fake-adapter Network Chat route, transcript/composer, channel/direct buffers,
   members, browse/join/favorite; mounted navigation and suspend/resume tests.
5. Exact released fork adapter and two-client live verification, then history.
6. Explicit tldw_server service descriptor and IRC-scoped credential exchange,
   coordinated with the server owner. Never reuse an HTTP token implicitly.

These are scope boundaries, **not nonexistent future task IDs**. File atomic
tasks before starting them. Public IRC is the first integration target; the
client fork does not supply an IRC server.

Current Chatbook seams to read before that tranche:

- [screen_registry.py](../../../tldw_chatbook/UI/Navigation/screen_registry.py),
  [shell_destinations.py](../../../tldw_chatbook/UI/Navigation/shell_destinations.py)
  and [route_inventory.py](../../../tldw_chatbook/UI/Workbench/route_inventory.py).
- [app.py](../../../tldw_chatbook/app.py) and
  [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md).
- [settings_screen.py](../../../tldw_chatbook/UI/Screens/settings_screen.py),
  [optional_deps.py](../../../tldw_chatbook/Utils/optional_deps.py) and
  [pyproject.toml](../../../pyproject.toml).

## Documentation PR verification

Run the Backlog ID/Windows-path guard, check all new relative links and task
dependencies, check whitespace, and inspect the staged diff for docs-only scope.
No application/fork test run is claimed. All implementation acceptance criteria
remain open, and PR creation does not create a fork repository or publish a package.
