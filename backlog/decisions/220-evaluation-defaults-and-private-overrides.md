# ADR-220: Evaluation Defaults and Private Overrides

Status: Accepted
Date: 2026-10-04
Related Task: TASK-34407
Extends: ADR-029, ADR-032, ADR-040, ADR-126

## Decision

Read `Evals/config/eval_config.yaml` as shipped application data. Keep
`_default_config_path()` as the dependency-light packaged-resource selector.
Store sparse profile overrides in `eval_overrides.yaml` beside the effective
`config.toml`, resolved through the existing configuration path policy.

Default `EvalConfigLoader` instances merge current shipped defaults with private
overrides. Reuse `config.deep_merge_dicts`: dictionaries merge recursively;
lists and scalar values replace. Explicit updates remain overrides even when
equal to a shipped value. Mutable values returned by the existing `get()` API
remain drafts and are captured as sparse changes on save. A missing override
file is a clean default configuration and is not created during a read.

Keep explicit `EvalConfigLoader(config_path)` inputs as standalone YAML
configurations, and explicit save destinations as full effective exports.
Reuse the existing raw participant, atomic publication, draft, pause/drain,
and persistence failure mechanisms. Only private overrides enter that raw
file lifecycle; package files are never written or privately hardened.

Recovery discovers the same profile override selector, marks an absent
override `unused`, and retains the existing `eval.definitions` owner, YAML
mapping policy, inactive destinations, provenance validation and activation
requirements. Legacy recovered YAML remains retained under those existing
rules; startup does not automatically adopt or rewrite it.

## Evidence and Alternatives

The repository's Codex sandbox ACL grants project modification to additional
principals. Its native Windows facade correctly projects those grants as
`0o766`. Treating the package YAML as private mutable state refuses that parent
and substitutes four fallback task types for the twelve shipped definitions.

The existing loader already owns atomic YAML saves, reload, dirty drafts and
recovery lifetimes. Introducing replacement storage or recovery infrastructure
would duplicate it. The existing merge helper supplies the required merge
semantics. Internal Prompts is a precedent for sparse overrides over shipped
defaults, but moving arbitrary evaluation YAML into TOML would lose supported
null values: the current TOML encoder drops `None`, while YAML preserves it.

Copying complete shipped defaults on first run was rejected because those
copies would stop inheriting updated defaults. Relaxing private parent checks
was rejected because private user files still require the existing boundary.

## Verification

Regressions exercise shipped defaults from the writable checkout, clean missing
overrides, sparse and explicit-equal updates, mutable drafts, YAML nulls,
failed save/reload under maintenance, full exports, profile selector isolation,
and recovery discovery. The final Eval-focused native Windows run passes 29
tests, including both review regressions. Full retained-definition publication
and rollback fixtures stop during setup in unchanged directory-flush code with
`WinError 5`; their end-to-end outcomes remain unverified on this host. Older
subprocess maintenance tests use POSIX pipe selection unavailable on Windows.
