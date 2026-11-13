# Local CI runners (TASK-40721)

Status: host qualified and enabled; routing variables set, workflow integration pending PR merge. See [ADR-359](../../../backlog/decisions/359-isolated-self-hosted-ci-runners.md).

## Host and capacity

Owner-approved host: `ml-server-1` (`192.168.2.236`), Debian 13, Ryzen 7700X (8 cores / 16 threads), 124 GiB RAM. Initial inspection found 66 GiB available RAM and 156 GiB free root NVMe. Existing services remain in place.

Eight test guests: 2 vCPU, 6 GiB RAM, 20 GiB virtual disk each. One control guest: 1 vCPU, 2 GiB RAM, 20 GiB disk. Controller service: `MemoryMax=64G`, `CPUQuota=1200%`. Sparse 100 GiB ext4 backing file `/var/lib/tldw-ci-storage.img` bounds aggregate image, overlay, and console-log storage. Its systemd mount is `/var/lib/github-qemu-runner`; no existing disk is reformatted. Monitor free space on both host and CI filesystem.

## Implementation and provenance

Controller: [a1678991/github-runner](https://github.com/a1678991/github-runner), pinned commit `b1dadbc247419019d1c62c3ee9e293f0cdede895` (MIT). Apply the checked-in `controller-reclamation.patch` to that exact commit with `git apply` before building; `controller.LICENSE` preserves the upstream MIT notice. The patch adds host-side cleanup of deregistered/offline runners and a busy-job deadline so guest shutdown is not trusted. Source reviewed at the QEMU argv, guest seed, image-download, GitHub authentication and slot lifecycle boundaries; this is not an exhaustive third-party security audit.

Built with Go 1.26.2, `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o github-qemu-runner ./cmd/github-qemu-runner`. Binary SHA256: `a4b4018a0937fda0ede125e3599125e71273cbe9b11cfd125f9c4f1b6602234b`. Upstream config, GitHub, pool, QEMU, seed, controller, imagebake and imageprep tests passed. Host packages: `qemu-system-x86 qemu-utils cloud-image-utils` from Debian repositories. Dedicated service user `gh-runner` belongs to `kvm`, has no login shell and no host Docker access.

The upstream image builder verifies the Ubuntu cloud-image SHA256 and the published Actions runner digest when available. Its metadata under `images/base.json` records the actual runner version, cloud checksum and bake time. Each job boots a fresh qcow2 overlay with a single-use JIT registration, then powers off; the controller removes the overlay and renews that slot. No host directories are shared into guests. Keep the pinned controller source separate from the application; no runner binary is committed here.

## Credential and network boundary

Use a private GitHub App installed on **only `rmusser01/tldw_chatbook`**, with repository `administration: write` (required for runner registration) and implicit metadata read. Administration is broad repository-settings authority; no personal CLI token is copied to the host. The private key is root-owned mode 0600 at `/etc/github-qemu-runner/app-key.pem`; systemd `LoadCredential` supplies it to the controller, never the guests. Set the numeric App/installation IDs in `config.yaml` from `config.example.yaml`.

`github-qemu-runner.service` blocks loopback, RFC1918, link-local, multicast, Tailscale CGNAT and IPv6 ULA egress for the controller and QEMU children. A service-private resolver uses public DNS so this does not depend on Tailscale DNS. No incoming host forwards exist. A malicious fork can edit workflow YAML, so the hosted-fork expression is routing policy, not a security boundary; VM disposal and host-level egress restrictions must remain effective even for hostile jobs.

## Activation and rollback

1. Verify image toolchain includes Git, Python 3.12, working setup-python prerequisites, and `gh` for merge-control jobs.
2. Verify public HTTPS/DNS succeeds and known reachable local/LAN services become unreachable under the actual service policy. Test the guest path as well as the controller process.
3. Install App credentials and verify the installation is restricted to this repository. Start the service and require both pool labels online: `tldw-ci-test`, `tldw-ci-control`.
4. Run an existing UI shard and two sequential runner jobs to prove correctness, job isolation/disposal and automatic renewal. Record timing and peak resources; preserve genuine failures.
5. Land the routing patch with both variables unset. Set `CI_CONTROL_RUNNER=tldw-ci-control` and `CI_TEST_RUNNER=tldw-ci-test` only after qualification. Both PR lint jobs use the control label too. Bulk/nightly workflows and fork PRs retain hosted labels. Never use generic `ubuntu-latest` labels on these local workers.

Clear the two repository variables to restore hosted selection for newly scheduled jobs. An already queued/running job keeps its selected runner: let it finish or explicitly rerun it after rollback. Drain with `systemctl stop github-qemu-runner` (up to 45 minutes for test jobs); do not force-stop active jobs. Do not delete the backing filesystem, key or App as part of ordinary rollback. No branch-protection or required-check name change is needed.

Refresh guest images deliberately with the upstream `refresh-image` command; preserve the old base until a fresh image is verified. The upstream image needs an additional trusted image-bake step: execute the checked-in `customize-guest.sh` inside a fresh overlay, run `cloud-init clean --logs --machine-id`, power it off, and flatten that overlay with `qemu-img convert -O qcow2` before replacing the base. Never promote an overlay that has executed repository code. Upstream refresh alone loses these additions; automatic refresh is disabled. Configure credentials for the command or supply the service credential via a transient systemd unit. The controller must be kept current enough for GitHub's runner-version requirements.

## Evidence

Live GitHub run [43398660691](https://github.com/rmusser01/tldw_chatbook/actions/runs/43398660691): 49 executed jobs, 615.0 total runner-minutes, median job queue delay 183.4 min, maximum 314.4 min, longest execution 33.0 min. This diagnoses queue pressure; it is not a promise that eight local workers finish the whole gate in under an hour.

Routing validation: 586 targeted CI cases passed with a restricted harness; the one harness-dependent storage-lease case passed separately with normal conftest loading. Independent routing review found only `runs-on` changes in parsed workflows and no actionable issues. Full application suite was not run.

Deployment on 2026-11-13: QEMU, the dedicated service account, bounded storage mount, controller binary, service unit and customized Ubuntu image are installed. The private App installation is verified for this repository only; its root-owned key is deployed. The controller is running with all nine runners registered online; `CI_CONTROL_RUNNER=tldw-ci-control` and `CI_TEST_RUNNER=tldw-ci-test` are set after qualification, with workflow integration pending PR merge. Base metadata: runner `2.338.2`, Ubuntu cloud-image SHA256 `04e8f2461a16a44e78a370284471f29616cdd9baf1893df25807b71f985b6c94`, upstream bake at `2026-11-13T17:44:02Z`. Added GitHub CLI 2.45.0 and Python 3.12.3 venv support in a trusted bake overlay. Original base is retained as `base-upstream.qcow2`.

Network verification: host SSH was reachable as a positive control; the service policy blocked loopback and LAN SSH while GitHub HTTPS succeeded. The actual disposable guest also reached GitHub HTTPS and could not connect to host gateway `10.0.2.2` or LAN host `192.168.2.236`. These checks exercise the configured boundary; they are not an exhaustive penetration test.

Qualification workflow: `.github/workflows/self-hosted-runner-qualification.yml` runs two sequential jobs on the single control slot, checks that the second runner has a new name and no first-job marker, and checks checkout/setup-python and guest network isolation on a test worker. It is manual or triggered by setup-file changes on `codex/self-hosted-ci`; it does not run the full application suite.

Representative existing UI shard 36/36 at commit `1455ceeb96c7c113a569a49a126fd287d4ac593b`: **116 passed, 1 existing Darwin-only skip, 2 warnings in 714.60 seconds** (11m54s). The subprocess wall time was 11m59s, maximum test-process RSS 619,888 KiB, and zero swaps. Guest start through completion was about 13 minutes, including clone/install. The guest used 2 vCPU and 6 GiB RAM. Host service cgroup peak was 7 GiB including cache, not test-process RSS. No test skips were added. Raw evidence is retained at `/var/lib/github-qemu-runner/probe/console.log` on the host.

Initial GitHub qualification [43478237685](https://github.com/rmusser01/tldw_chatbook/actions/runs/43478237685) passed checkout, setup-python 3.12.11, control tools, actual guest network checks, and two sequential control jobs. Control job 1 ran 53 seconds, the renewed job started 19 seconds after its completion and ran 4 seconds. Both completed runner identities disappeared from the repository roster and replacement slots returned online. Review then required host-side reclamation even if guest shutdown fails; the stronger qualification intentionally disables guest poweroff. The stronger check passed in [43480488554](https://github.com/rmusser01/tldw_chatbook/actions/runs/43480488554): the controller logged host-side reclamation at 18:14:43Z, and the next control job started 23 seconds after the first completed. The service is enabled at boot.

Controller patch SHA256: `d881d91bb348fa8f3371c51068f97fb2b6dffecf8aabe9962fd9b474412bacec`. Race-enabled pool/config/controller/GitHub/QEMU tests and Go vet passed. Busy-job deadlines are 45 minutes for tests and 15 minutes for control; idle waiting does not consume that budget. Continuously offline runners are reclaimed after the existing five-minute liveness grace; transient API errors are not treated as proof of being offline. At nine slots, ten-second status polling uses about 3,240 App requests/hour before registration/cleanup: check rate limits before expanding further.

Integration status: the routing changes must land on `dev` to migrate its existing control workflow. A PR containing this patch uses the local labels immediately through its updated workflow. Required merge checks and review policy still apply. The first complete eight-worker PR gate has not yet established sustained-load timing.

Final routing coverage includes both lint jobs; all 380 routing-matrix cases passed after that extension, with Ruff and formatting checks clean.

Python tool-cache correction (TASK-40723): the first full gate failed when project-chat UAT removed `LD_LIBRARY_PATH`; the relocated setup-python interpreter could no longer import its installed pytest. A trusted bake VM reproduced that failure with official Python 3.12.16 archive SHA256 `dec61ed20cd3e71143f1642ef9966945dea8ae4f3e96b306205a710da4f2191a`, then passed with the same installation at `/opt/hostedtoolcache` and a changed HOME. The guest runner `.env` now sets `AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache`, with that directory owned by runner. The base retains only the canonical cache configuration; setup-python performs each complete interpreter installation, including executable entry points. The relocation-proof interpreter was removed after its missing pip executable caused the next gate to fail before testing. Qualification now tests sanitized child imports for both pinned 3.12.11 and the test lanes' 3.12. The application test's environment isolation is unchanged. The pre-correction base and before/after console evidence remain on the host; the first failed/cancelled gate 43482084755 is not a successful full-run timing.

Qualification now enables `cache: pip` on the test-worker setup-python step, matching the real PR test lanes. Import-only qualification had missed the incomplete preloaded installation; do not mark tool-cache completion manually or promote a relocated probe installation as a preloaded runtime.


Sustained-load qualification (TASK-40724): attempt [43487945840](https://github.com/rmusser01/tldw_chatbook/actions/runs/43487945840) passed the repaired project-chat UAT and its negative control, but Console P0 exhausted its 20-minute job cap during the sandboxed invocation. Dreams/Guardian reached 91% of its final test group when its 20-minute cap expired. Both annotations explicitly report the job limit. The remaining run was cancelled after 37m49s; this is **not** a successful full-gate duration. Console P0 now runs sandboxed, restore and admission-sensitive tests as separate required matrix profiles with fail-fast disabled. Those jobs and Dreams use the existing 35-minute long-lane budget, below the host's 45-minute busy-job deadline. Per-test timeouts and test selections are unchanged; sustained-load verification is pending.


Attempt 43496035687 passed Dreams/Guardian in 21m08s and the P0 admission profile in 13m27s, but sandboxed P0 still hit 35 minutes. Its three response-rule restore selectors completed in 25m46s before the remaining Canvas workload. Those exact selectors now have their own restore profile; all P0 selectors still occur exactly once, and no per-test or job limit was raised further. The same attempt exposed an ingest crash-probe startup cost, repaired separately by TASK-40725. Full-gate timing remains unqualified until the next complete successful attempt.
