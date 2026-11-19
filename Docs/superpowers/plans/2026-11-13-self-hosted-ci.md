# Self-hosted CI implementation plan

Goal: relieve measured runner starvation with isolated local capacity.
Architecture: GitHub orchestrates disposable KVM runners on the existing Linux server. Separate test and control pools; hosted defaults remain until qualification.
Tech stack: GitHub Actions, Ubuntu 24.04, QEMU/KVM, systemd, existing runner controller.
Spec: Docs/superpowers/specs/2026-11-13-self-hosted-ci-design.md
ADR required: yes
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: CI execution, credential and isolation boundary.

## Tasks

- [x] Inspect live timings and host; record design and rollback.
- [x] Add opt-in routing to derived-artifacts.yml and merge-queue.yml, including hosted fork fallback; preserve selectors and job graph. Add focused tests before changing routing, update runner-label contract assertions, run targeted CI tests.
- [x] Review and pin existing disposable runner controller, build Linux executable, prepare eight-worker and one-control config, install bounded services and private-network isolation.
- [x] Build and boot runner image; run representative existing shard, verify disposal and renewal.
- [x] Record exact deployment state and test evidence; activate only after successful live qualification.

## Execution ledger

2026-11-13: Task in progress. Worktree is isolated from user's unrelated edits. No routing or protection settings changed. QEMU packages installed on approved host. GitHub App login/installation is pending; no personal CLI credential is copied to server.

Routing task review: no actionable issues. 586 targeted cases passed; one conftest-dependent case passed separately under the normal harness. Only runner routing changed in parsed workflow comparison.

Host preparation and actual guest network checks passed. The selected 117-test UI shard is running. App creation completed, but installation verification found all-repository access; the user has been asked to restrict it before credential deployment. No runner routing has been activated.

App installation is now verified for tldw_chatbook only. Credentials deployed via root-owned systemd credential source; five runners online. Qualification workflow pushed to the isolated branch and running before any routing variable is set.

Representative shard passed (116 passed, 1 existing Darwin-only skip in 714.60s); normal GitHub renewal qualification passed (run 43478237685). Review corrected systemd KillMode to mixed, preserving controller drain. Before activation, strengthen upstream controller reclamation and verify a guest that refuses shutdown; this is required to make disposable-slot capacity reliable.

Final preparation: eight test slots plus one control slot are online, service enabled at boot, App scoped only to this repository. Host reclamation with guest poweroff disabled passed in run 43480488554. Routing variables are set after qualification; workflow integration and the first complete PR gate remain pending. TASK-40722 separately corrected the two reproduced plugin backup census failures (5 affected cases passed).
