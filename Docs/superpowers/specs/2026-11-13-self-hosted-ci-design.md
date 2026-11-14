# Self-hosted CI capacity

Goal: reduce runner queue delay without changing required test coverage.

Design and constraints: [ADR-359](../../../backlog/decisions/359-isolated-self-hosted-ci-runners.md). Task: TASK-40721.

Routing is opt-in and keeps the original hosted label when unset. PR fork jobs always use hosted runners. Test and control pools have distinct labels, and bulk workflows do not use either pool. No repository variable is enabled until a real representative shard and disposable renewal are verified. First rollout is eight test workers plus one control worker on the inspected host.

Success evidence: existing CI contract tests, new routing checks, image boot and isolation checks, a timed existing shard, and two sequential jobs proving disposal and renewal. Record unmet activation prerequisites explicitly.
