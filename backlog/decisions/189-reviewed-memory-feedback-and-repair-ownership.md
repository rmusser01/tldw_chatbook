# ADR-189: Reviewed memory feedback and repair through existing native owners

Status: Accepted — design direction explicitly approved; no runtime/schema rollout
Date: 2026-09-25
Task: TASK-25907.9
Extends: [ADR-182](182-personal-context-memory-evolution.md), [ADR-201](201-versioned-profile-evidence-and-temporal-claims.md), [ADR-202](202-dependency-aware-personal-context-forgetting.md), [ADR-203](203-personal-context-provider-disclosure-authority.md)
Related: [ADR-105](105-portable-notes-organization-and-agent-lessons.md), [ADR-106](106-human-reviewed-agent-lesson-promotion.md), [ADR-196](196-dreams-daily-discovery-and-tracking.md), [ADR-188](188-opt-in-proposal-only-memory-consolidation.md)
Design: [Feedback and repair contract](../../Docs/superpowers/specs/2026-09-25-personal-context-feedback-and-repair-design.md)

## Context

Explicit feedback currently has local mutable JSON and separate server routing; default unspecified mode is SERVER and remote detail may be unsupported. Those events do not supply immutable evidence or durable approval. Personal Context owns facts/preferences; Agent Lessons use ordinary Notes with exact foreground approval and independently verified procedures. Companion goals are remote-only; proposed Dreams owns forward-looking discovery, not memory repair. A separate alignment/repair memory store would duplicate those authorities.

## Decision

Propose foreground-only repair through existing owners. Current feedback/message owners retain attributed observations. Exact facts/preferences become canonical Personal Context proposals, accepted only by the user. Independently verified reusable procedures become Agent Lessons under ADR-105/106. Unresolved neutral issues may become ordinary user-owned Notes only through a separately exact foreground save. Default previews and rejected/abandoned lesson drafts remain ephemeral. No inferred personality/emotion, duplicate fact store, raw-log corpus or narrative reflection becomes memory truth.

Separate observed feedback, proposed response change, owner-approved change, user-reported resolution and independently verified resolution. Approval does not prove success; a quiet conversation or agent claim does not prove resolution. Each linked change needs its own owner receipt, while issue resolution derives separately from exact reviewed outcome evidence. An ephemeral response fix may resolve an issue despite declined/unnecessary lasting profile or lesson changes; resolution cannot approve them. Native exact source/target/approval receipts and current scope/version/authority determine state. A Notes-owned bounded typed operational projection references the current issue Note and permitted canonical owner receipts; it retains no separate prose or permissions. Future trusted one-use foreground transitions bind exact Note/organization/issue/target/evidence versions and policy epochs before Notes COMMIT. Broad ordinary Note tool allow, imported text/status fields or caller actor=USER cannot forge that transition. Other direct/imported Note edits invalidate the typed projection until review.

The workflow has no cross-owner transaction. Profile acceptance, issue saves and Agent Lesson saves are distinct operations with separate approval, permission and conflict results. Recovery reconciles exact native owner receipts and never rolls back accepted changes or replays approval. Legacy mutable feedback and inline Note labels remain unverified; the first local slice excludes unqualified sources and explicitly chooses LOCAL routes without lookup fallback. Source/custody/current head controls require shipped ADR-201/202/203 adapters.

Issue scope is an explicit native conversation/workspace/global binding, not folder text or server dataset names. Scope promotion requires separate review and preserves restrictions. A retained issue defaults to a finite 30-day review deadline; expiry makes it review due without deleting the user Note or changing an independently valid approved fact. Reopening requires new explicit permitted evidence and exact review; old resolutions remain historical. Echoes of one source are one evidence root.

Standing guidance is a disposable bounded view from the existing eligible approved Personal Context selection pass, with current scope overrides, destination/purpose filters and 12 KiB/ten-percent limits. Pending issues, raw feedback, inferred feelings and reflection summaries are excluded. Verified permitted lessons remain untrusted tool results under existing native capability/approval rules, never system/project instructions or permission/goal mutation authority. No separate synthesis, ranking pass or guidance store is added.

Source-derived Notes/repair metadata require qualified inherited retention, disclosure and local custody across FTS, Notes dispatcher/file Sync, export, captures and recovery. A private path or keyword is not encryption or Sync suppression. The first slice refuses persistence when owner coverage is missing and permits an authorized ephemeral preview. Forgetting uses ADR-202 native review/suppression/retirement; independent originals need separate exact consent. Unknown legacy/offline/external coverage remains honest, and no hidden locator/hash preserves the secret.

Proposed ADR-196 Dreams interests/goals/query angles remain discovery-owned. No dream prose or inferred interest auto-imports into memory; distilled queries still need distinct reviewed disclosure/source/custody authority for any future integration. Companion goals stay server-owned; no local replica, automatic goal write or server request is added. Personas retain their existing user-selected owner boundaries.

## Alternatives and consequences

Ephemeral-only handoffs remain available and avoid persistence. Reviewed issue Notes make unresolved work inspectable while preserving fact/procedure ownership; they require future Notes-native typed transition/custody qualification rather than treating current text fields as receipts. A dedicated alignment/repair/Dreams store was rejected because it creates competing truth, retention and authority. This design adds no nightly work, model/search/server budget or self-modifying instructions.

## Release boundary

The first future slice is one foreground primary, unlocked native local profile/scope, exact current direct-user feedback source and one reviewed correction/issue/lesson handoff at a time. No model/server calls, automatic collection or background jobs. Future source/Notes metadata schemas, migrations, compatibility, current-owner receipts and privacy adapters must be shipped and verified through healthy/refused native controls before enabling persistence/use. Design-task completion does not qualify them. No UI, schema, worker, real profile, provider permission or data changed here.

ADR-189 is provisionally allocated against available offline refs/worktrees; recheck newly fetched refs/open PRs at integration. The user explicitly approved the reviewed written contract. This accepts design direction only; no runtime/schema rollout, Notes mutation or provider permission is granted.
