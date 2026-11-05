# Windows retained Eval recovery cost correction

The original retained-rebackup restore keeps its 45-second subprocess limit.
At source publication, hosted Windows qualification and final mobile UAT remain
pending; current exact-head results and review evidence belong to PR #3345 and
the QA README. No instrumented pass is a release qualification.

Windows run41935498768/job126942809784 reproduced the restore timeout after the
fixture lifecycle correction: 1 failed, 82 passed, one existing platform skip.
The staged boundary arrived at30.5904s. Its profile measured4,948,151 calls to
`SchemaPolicy`'s recursive leaf helper (0.955s self,2.609s cumulative). File writes
and Windows handle/security work also remain substantial; the cumulative helper
time is not an assertion that all2.609s can be recovered.

The correction checks primitive values inside the tuple traversal, avoiding a
separate recursive call per SQL literal. Exact tuple roots, primitive type
comparison, traversal order and refusals remain unchanged. No validation verdict,
policy or permission is cached, and no buffer, memory fee, deadline, native DACL,
file flush or directory barrier changes.

After removing temporary profiling, the exact affected Eval selection passed
all110 cases locally (epoch396,174.13s); the original complete retained journey
passed in42.38s with its original45-second restore child limit. The child limit
is distinct from total multi-process journey time. This is a Mac control and
cannot qualify the hosted Windows deadline. Physical reap and unchanged source
were verified.

The real installed ChaChaNotes policy regression first failed at27,592 helper
calls for125 tuple containers. The correction passes that unchanged bound and
all178 targeted policy, real-SQLite, migration, forged-policy and refusal cases
(epoch395,63.07s, physically reaped/source_changes empty). All25 semantic controls
already passed in the preceding RED. New controls run in the existing Eval lane
on all three platforms with its original limits.

A separate fixture correction awaits the actual retained constructor attention
read before each shared preview. Its final native regression fails on original
ordering (391), then passes with the original retained/lifetime/refusal controls
(392:10 passed74.60s). The other four original capture/reopen/deletion callers
pass (393:4 passed137.25s). Every run physically reaped with unchanged source.
Intermediate incorrect empty-mark test expectations and their failures are
retained; the final check observes the actual successful native tuple return and
read/lease retirement. All three existing fixture marks remain.

Temporary capture and restore profiling wrappers were removed before final
qualification. Their source and original Windows artifacts remain under
`.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/landing-2026-11-02/ci-repair-2026-11-04/windows-eval-timeout-2026-11-04`.
The two existing model lint findings remain unchanged; scoped verification
requires zero introduced findings and the existing hosted gates.

Existing ADR125/126 apply. Actual phones/VPN, live providers, real profiles and
native-Console phone attachment phase2 are outside this software-UAT scope.
