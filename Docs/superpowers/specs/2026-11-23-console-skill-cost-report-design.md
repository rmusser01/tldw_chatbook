# Console skill cost and usage report

Status: Approved for current-session observation only

Task: [TASK-38931](../../../backlog/tasks/task-38931%20-%20Report-skill-and-extension-context-costs-and-observed-session-usage.md)

## User experience

Open a read-only report from Console with `/skill-cost`. Show the age and
source of the request sample, the existing request total, and separate figures
for advertised skills, other tool schemas, and identifiable loaded skill bodies.
Show skill usage as accepted, observed invocations in the current live
session, with a count and last use. Show unknown when no observation exists.
Point users to Library → Skills or Settings → Tools & MCP for relevant conflicts,
blocked skills, and disabled components. Use the existing Console command worker
and display the result outside the transcript.

## Approaches

- **Recommend observing the live session.** Reuse request accounting and the
  skill invocation paths. This works with Capture off and adds no saved
  history. Reopening the app starts a fresh observation window.
- **Recover activity from existing saved runs.** This could cover some older
  activity, but Capture is optional. Saved records do not consistently
  distinguish direct skill expansions, agent skill calls, and retries. A
  historical report would need separate coverage rules.
- **Persist a new usage ledger.** A ledger could support complete history after
  release, but it would introduce a retention policy, storage lifecycle, and
  migration. The task asks for observed session usage.

## Attribution and ownership

- Read the same request accounting snapshot used by the Context breakdown.
  Attribute only material present in its prepared request. Use schema identity
  from the planner rather than matching bare names. Native skill schemas are
  part of the tool schema bucket. Fenced skill schemas are part of the system
  prompt. Classify those shares within their owning buckets and show any
  unallocated remainder. Treat derived token shares as estimates. A catalog
  entry that wasn't sent, or a remembered plan that can't be matched to the
  request sample, can't establish a charged cost.
- Observe successful manual expansions at accepted request handoff and
  agent-picked skills at the skill runner's accepted launch. Count each accepted
  invocation once. Preparation, preview, refusal, and retransmission are not
  invocations. Identify loaded-body cost only where the request's provenance
  supports it. A body loaded into a child request is outside the parent
  request's total. Show unavailable attribution when the body can't be located in
  a prepared request.
- Keep bounded metadata in the existing Console runtime, scoped to the session
  incarnation and workspace. Retain eligible display names, counts, observed
  times, token attribution and status codes. Label catalog changes and any
  incomplete observation window. Use diagnostics from the existing catalog and
  skill owners for conflict and availability reasons. If an observation is
  stale or its owner no longer matches, withhold it.
- The report does not retain skill bodies, descriptions, arguments, paths, or
  project instructions. It does not launch providers, tools, plugins, or MCP
  servers, change settings, or grant permissions. The report command does not
  add its results to chat history or future model context.

## ADR and verification

Amend [ADR-230](../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md)
before implementation. It already governs Console request ownership. The
amendment should govern who publishes report metadata, its lifetime, and
its display boundary. Preserve
[ADR-317](../../../backlog/decisions/317-console-project-instruction-local-state-and-preflight.md)
for automatic project-instruction visibility. This proposal does not
authorize a durable ledger or reads of protected bodies.

Cover native and fenced schemas, reconciled totals, advertised unused skills,
both accepted invocation paths, retry deduplication, unavailable attribution,
and an absent snapshot. Exercise the Console command and catalog, workspace,
and session changes in the mounted app. Verify that the report performs no
provider call and that no report or hidden body enters the transcript.
