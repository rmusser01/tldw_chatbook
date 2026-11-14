# ADR-217: First-run setup shape and surfaces

Status: Accepted 2026-10-03 (approved by the owner; approval recorded in the design spec's §15)
Date: 2026-10-03
Related Task: [TASK-34100.17](../tasks/task-34100.17%20-%20Owner-approved-design-spec-for-the-setup-flow-Quick-track-tldw-server-re-run-dashboard-Say-hello.md)
Design: [First-run setup shape](../../Docs/superpowers/specs/2026-10-03-first-run-setup-shape-design.md) (revision 3)
Evidence: [First-run wizard UX review, 2026-10-02](../../Docs/superpowers/qa/first-run-wizard-ux-review-2026-10-02/README.md)
Supersedes: N/A. It replaces TASK-21148 AC#5's mechanism ("Protect always on Quick") and keeps that AC's guarantee. It amends the wording of ADR-076 through ADR-076's own 2026-10-03 amendment.

## Context

No ADR owned the shape of setup. ADR-012, ADR-029 (`029-local-private-data-boundary.md`), ADR-033 (`033-application-session-state-ownership.md`), ADR-076 and ADR-126 each constrained a part of it. Setup was one fixed corridor: six Quick steps, two of which do nothing for most people, no tldw server offer, no path for someone who came for their documents, a re-run that replays first run, and a Summary that says "done" when config was saved rather than when a chat can be sent. Setup also existed only as a full-screen Textual app, which screen readers cannot use.

The 2026-10-02 review's structural fix SF10 and enhancements E1, E5, E8, E9 and E12 asked for a shape decision. The owner ruled that it be written as a spec and approved before any build. The spec was revised after independent HCI and engineering critiques and approved on 2026-10-03, with one amendment: the owner ruled that keychain storage for provider keys must be optional, not the default.

## Decision

One setup owner serves four surfaces: the first-run corridor, the re-run dashboard ("Review your setup"), single-step sheets, and `tldw-cli setup --plain`. The owner is the setup core in `tldw_chatbook/Setup/` (tracks, the `SetupSession` reducer, the step states and the commit builders), hosted through the `SetupStepHost` protocol. Single-step sheets serve dashboard Change, palette commands, Ready next steps and Console Speak/Dictate. Every surface uses the same track definitions, step states, commit builders and readiness verdict, and the same `has_usable_chat_provider` predicate to choose its first screen.

1. **Tracks.** Quick = Welcome, Connect, Model, Ready. Full = Welcome, Connect, Model, tldw server, Search, Tools, Spoken replies, Dictation, Appearance, Keys, Ready. Documents-first finishes on Welcome and records that AI setup was deferred.
2. **Stable total.** Once the user leaves Welcome, the step list does not change until they return to it. Conditional offers are options on Ready, never steps. (This supersedes TASK-21148 AC#5's "Protect always on Quick" and keeps its guarantee.)
3. **Optional steps.** Every optional step defaults to "not now" and writes nothing when untouched. "Finish with defaults" is offered on Full from Model onward.
4. **The verdict.** Ready's verdict is computed through Console's shared readiness verdict, never a wizard copy, combined with this run's test outcome, with one mark per region. First run docks at most three exits and keeps both Library exits visible.
5. **The test message.** The optional test is a Console probe turn: Console's admission and dispatch, a turn-scoped profile with no tools, retrieval, history or persona, and a saved conversation. It runs automatically only on first run, for allowlisted local engines on loopback, with no key and nothing to load. For every other provider it runs only on an explicit press, with the computed cost on screen. Start chatting opens a new conversation.
6. **Re-runs.** A setup entry with a usable chat provider opens the dashboard. Only a step's own Save writes. Done returns to the origin. Console readiness links go to the control that fixes each reason.
7. **Secrets.** Keys are never read from argv by any setup surface. An environment key is never stored. A new provider key is stored where it is today, in config.toml, unless the user chooses the system keychain for it; the keychain is never the default (owner ruling, 2026-10-03). A keychain-held key is never written to config.toml by any writer, and a keychain failure never writes it there.
8. **Not offered.** Setup does not offer a Welcome router, a master tool switch, an embedding-model picker, a multi-tick provider list, per-provider base-URL overrides, an undo-this-session ledger, or an inline TLS-verification toggle. Non-interactive setup is deferred until users ask.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep six Quick steps, making Voice and Protect skippable | The steps become honest but not relevant: two extra Nexts on every Quick run. |
| A conditional Protect step | Brings back the mid-flight step-count change TASK-21148 fixed. |
| A setup hub of cards for first run | A newcomer needs a sequence, not a menu (report §4.3). The hub is the re-run dashboard only. |
| A wizard-built test request | It would drift from Console's request path; the probe is a turn-scoped override inside Console's own path. |
| A second, app-free implementation for plain-text setup | Two implementations drift; plain mode renders the same step states. |
| The system keychain as the default key store | Rejected by the owner on 2026-10-03: keychain storage must be optional. |
| Non-interactive setup now | The report's verifiers advised deferral; the copy-config route works, and the CLI cannot host the bind or the test. Reopen on the first user request for scripted setup, or the first support case from a copied config carrying keys. |

## Consequences

- The wizard spec's "re-run and first-run are one code path" (2026-07-28 §1) is replaced by "one setup core, several surfaces".
- ADR-076's "the only startup/setup owner" now refers to this owner and all its surfaces (ADR-076 amendment, 2026-10-03).
- ADR-012 and `029-local-private-data-boundary.md` gain dated amendments that admit the OS keychain as an optional, per-key store for provider keys. ADR-033 (`033-application-session-state-ownership.md`) gains one: one runtime-source coordinator for every binding surface, with rollback on disk, and server-token storage unchanged. ADR-126 gains one for keychain-held provider keys in backups.
- Steps may reach the outside world only through `SetupStepHost`. The setup core must stay importable without Textual.
- The design is delivered as the follow-up tasks listed in the spec's §10, starting with the enabler that creates the setup core and the step-host protocol.

## Links

- [First-run setup shape design (revision 3)](../../Docs/superpowers/specs/2026-10-03-first-run-setup-shape-design.md)
- [First-run setup wizard design (2026-07-28)](../../Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md)
- [ADR-012: Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-076: Library Lifecycle Progressive Disclosure](076-library-lifecycle-progressive-disclosure.md)
- [ADR-126: Complete local backup and recovery](126-complete-local-backup-and-recovery.md)
