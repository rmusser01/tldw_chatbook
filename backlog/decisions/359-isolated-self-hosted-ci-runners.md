# ADR-359: Isolated self-hosted Linux CI runners

Status: Accepted; host qualified, workflow integration merged in PR #3529
Date: 2026-11-13
Task: TASK-40721

## Context

GitHub run 43398660691 took 7h52m. Its 49 executed jobs used 615 runner-minutes, with median job queue delay 183.4 minutes and maximum 314.4 minutes. The longest execution was 33 minutes. Owner requests working, timely CI and authorizes the server at 192.168.2.236. SSH inspection finds Ryzen 7700X (8 cores/16 threads), 124 GiB RAM (66 GiB available), 156 GiB free root NVMe, existing services and low CPU load.

## Decision

Keep GitHub orchestration and ADR-103 required gate/test selections. Add reversible repository-variable routing for same-repository PRs and control work, with hosted defaults and fork fallback. The initial rollout kept bulk/nightly jobs hosted so they could not take the local PR pool; the dev-push amendment below extends that placement explicitly.

Start with eight 2-vCPU/6-GiB test VMs and one 1-vCPU/2-GiB control VM. Disposable Ubuntu 24.04 KVM guests run one official GitHub JIT runner job and are destroyed. No host mounts or host Docker socket. A dedicated unprivileged host controller holds a repository-scoped GitHub App key; guests receive only single-use JIT configuration. Block guest/controller private-network egress except necessary DNS. Bound resource use and retain disk headroom.

Evaluate and pin an existing QEMU controller rather than write an autoscaler. Carry a focused tested controller patch to reclaim deregistered/offline guests and enforce busy-job deadlines; guest shutdown cannot be the disposal authority. Use systemd `KillMode=mixed` so controller shutdown drains active jobs. Qualify image tooling, network isolation, successful job execution and renewal before enabling repository variables. Hosted rollback is clearing those variables, affecting newly scheduled jobs; existing local jobs must finish or be explicitly rerun after switching.

## Alternatives

Moving to another CI platform does not remove the measured execution workload and adds migration of GitHub merge semantics. Bare-host or ordinary Docker runners share too much with the existing server. Unlimited concurrency risks moving contention onto the host. A copied personal CLI token has broader access than necessary; use a repository-installed App for renewal.

## Consequences

### Dev-push amendment (2026-11-14, TASK-40998)

The owner requested completing local execution of the remaining dev jobs before establishing a new optimization baseline. Route the fourteen Library shell shards and the Perf Guard job to `CI_TEST_RUNNER` on pushes to `refs/heads/dev`. Retain the hosted fallback when the variable is unset and retain all other event routing. Lint, preflight, aggregate and merge-control jobs already use the control pool.

Use the existing eight disposable test guests, image, network isolation and resource bounds. Keep Library parallelism at seven, both workflows' twenty-minute job caps, every selector and latency budget, and the existing non-required Library policy. These jobs share capacity with PRs; placement does not reserve a slot or increase throughput. The required dev aggregate can finish before the non-required Library tail, so qualify those real jobs separately after merge.

After the actual local dev jobs pass, measure a fresh full run of the final migration head's normal PR gate in a clear local workload window. A rerun retains the original pull_request event; never dispatch the required workflow to bypass PR association. Verify its tested source is equivalent to the deployed source, and preserve every attempt's job and environment evidence for the subsequent comparison. PR #3529's recorded run remains historical evidence for the earlier deployment. Adding a separate pool or raising limits is not part of this amendment.

One server is not highly available. Eight workers may still need more than an hour for the present gate; measure instead of claiming a sub-hour guarantee. Additional pool capacity is a measured follow-up. Test failures remain failures. No coverage reduction or merge-protection weakening is part of this change.

Full required PR qualification passed on 2026-11-14: run 43612634465 completed its required aggregate in 1h28m09s, with all 51 executed jobs passing on local runners. PR #3529 merged as d46f753d8287cffd48706f912bfd838dc8756edb. Exact timing, image, dependencies, queue delays and comparison limits are retained in `Docs/QA/2026-11-14-self-hosted-ci-baseline/README.md`.

Initial capacity qualification: the representative shard used 84% of one CPU and 619,888 KiB test-process RSS; after host preparation, 105 GiB RAM remained available. Increase the initial conservative four-slot proposal to eight test slots, retaining 2-vCPU/6-GiB guests. Bound the fleet to 64 GiB memory and 12 logical CPUs (of 16), with one separate control slot. The successful full run above supplies the subsequent whole-gate measurement.
