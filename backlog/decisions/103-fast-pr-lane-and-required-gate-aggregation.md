# ADR-103: Fast PR lane and required gate aggregation

Status: Proposed
Date: 2026-08-29
Related Task: [TASK-24403](../tasks/task-24403%20-%20Fast-PR-lane-preserves-required-gate-and-full-coverage-cadence.md)
Supersedes: N/A

## Decision

Pull requests into `dev` use one serial, minimal-dependency fast-test job whose
result is aggregated by the existing required `Derived artifacts reproduce from
their sources` context; comprehensive test coverage runs on `main`, nightly, and
manual events instead of every pull-request update.

## Context

The account-wide CI investigation in TASK-22250 found that `tldw_chatbook`
generated approximately 93% of the account's runner use. One ordinary pull
request can request about thirteen concurrent runners after its core/UI shard
limits and focused guards are counted. Two active pull requests can therefore
exceed the former twenty-runner account ceiling, and several active pull
requests can still create avoidable queueing after the account's GitHub Pro
upgrade raised that ceiling to forty.

Trigger deduplication removed redundant `dev` push and permanent promotion-PR
runs, but each real pull-request update still launches the complete six-shard
core and twelve-shard UI matrices plus specialized jobs. Raising account
capacity does not bound that demand.

The `dev` protection rule currently requires only the stable job context
`Derived artifacts reproduce from their sources`. Adding a new required context
would be operationally unsafe: existing pull-request head commits may never
have reported it and can remain waiting for an expected check. Folding package
installation and tests directly into the existing artifact job would avoid
that rollout problem but would destroy the job's intentionally install-free,
roughly ninety-second diagnostic contract.

Repository paths cannot safely select which PRs need tests. Tests consume
files under `Docs/` and `backlog/`, including directory globs, so documentation-
only heuristics can silently skip load-bearing inputs.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add `PR Fast Lane` as a new required branch-protection context immediately | Existing PR heads may not report the new context, leaving them permanently waiting unless branches are rewritten or PRs are reopened. |
| Install dependencies and run tests directly inside the existing derived-artifacts job | It mixes unrelated ownership, removes the install-free diagnostic path, and makes artifact drift harder to distinguish from environment/test failures. |
| Keep full matrices on every PR and rely on the forty-runner Pro ceiling | The workload remains unbounded across active PRs and can still consume the whole account pool during ordinary bursts. |
| Select tests from changed paths | Documentation and Backlog files are test inputs; path heuristics have already been reviewed and rejected as unsound in this repository. |
| Reduce only matrix shard counts | Fewer shards reduce instantaneous fan-out but retain hours of setup and execution per PR, so several PRs still monopolize the shared pool. |

## Consequences

- The existing required job name remains unchanged, so branch protection needs
  no migration and existing PRs are not stranded.
- The fast-test job and derived-artifact job remain separate. The required job
  declares the fast lane as a prerequisite, runs with `always()`, and fails
  explicitly on pull requests unless the prerequisite result is `success`.
- The required workflow consumes at most one runner at a time: the fast lane
  runs first, followed by the install-free derived checks. With four focused
  guard workflows, one PR has a five-runner peak instead of approximately
  thirteen.
- Existing unchanged PR heads can retain a previously reported required result
  until their next pull-request event. This bounded grandfathering is accepted
  instead of rewriting branches or close/reopening unrelated PRs.
- The fast lane runs at the supported Python and Textual floor with only core
  application dependencies and explicit test utilities. Optional ML, document,
  and browser stacks remain outside the PR gate.
- Full-tree coverage remains mandatory, but moves to events with bounded
  cadence: `main` pushes, the scheduled deep run against `dev`, and explicit
  manual dispatch.
- The candidate target list must remain non-overlapping. Pytest has silently
  collapsed a directory argument when a file inside that directory was also
  listed, so collection count is part of verification evidence.
- A future change to the required context name, prerequisite relationship,
  event cadence, or fast-lane dependency boundary must update ADR-103 or
  supersede it.

## Links

- [TASK-24403](../tasks/task-24403%20-%20Fast-PR-lane-preserves-required-gate-and-full-coverage-cadence.md)
- [Fast PR lane design](../../Docs/superpowers/specs/2026-08-29-fast-pr-lane-design.md)
- [TASK-22250](../tasks/task-22250%20-%20CI%20runs%20are%20swept%20by%20simultaneous%20burst%20cancellations.md)
