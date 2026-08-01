# Guarded File Notes push acceptance

Verdict: **PASS** for the production-app flows exercised on macOS, including the retained uncertain-result reopen remediation.

## Build and environment

- Primary full-app source: `ad0285f96c8cb7b8cbc8c1d06d7f28be7fd16d33`
- Remediation retest source: `913d02fd764168247600bff017a7fb6d9d09de80`
- Launch: `python -m tldw_chatbook.app`
- Operator: agent-operated real PTY input
- Platform: macOS 15.6 (24G84), arm64
- Python: 3.12.11
- Git: 2.39.5 (Apple Git-154)
- OpenSSH: 9.9p2, LibreSSL 3.3.6
- tmux: 3.6a
- Wide viewport: 120x40
- Compact viewport: 40x20

The fixture used standard OpenSSH, a loopback server, a standard user
`known_hosts` file, and a disposable key loaded into an isolated agent. The
private client key was deleted before Chatbook launched. Retained artifacts do
not contain the key, host-key material, agent socket, credentials, note body,
real user paths, or raw server diagnostics.

## Accepted behavior

- File Notes linked an existing Git-backed notes root and preserved exact
  frontmatter while the body was edited separately and autosaved.
- Session-only staging and commit controls changed only the tracked session
  path and showed the promise/count copy.
- Canceling authorization made no connection. Authorizing made one read-only
  connection before review.
- The review showed the exact candidate, parent transition, full destination
  ref, sanitized endpoint, expected-parent lease, and frozen SSH policy.
- Host-trust drift after review blocked locally before a receive operation.
- A fresh review completed one exact ref transition in the success scenario.
- A divergent destination blocked before receive and was not overwritten.
- A deliberately delayed receive produced an original `Uncertain` result and
  query-only recovery action.
- After the remediation, Back kept the result dismissed through ordinary
  poll/refresh cycles; Files -> Session Git explicitly restored the same
  `Uncertain` result and `Check remote again — no push` action without any new
  connection.
- The private trust snapshot was observed with mode `0400`; no disk identity,
  askpass, credential helper, or proxy path executed.

## Fixture corrections and scope

The first launch used a non-canonical temporary-path alias for the agent
socket. Production correctly rejected it locally with zero network activity.
The fixture was relaunched with the canonical socket path; this was a fixture
correction, not a product defect.

The success and divergence destination baselines were prepared locally inside
the disposable fixture. Those baseline preparations did not use SSH and are
not counted as Chatbook network operations.

Process-tree settlement is retained from the focused native automated lane,
not claimed from terminal observation. HTTPS is automated-lane evidence only.
The Windows Job Object case was not native on this macOS host and remains the
focused suite's explicit skip.
