# Evaluation defaults and private overrides

The user chose automatic shipped defaults with private overrides and authorized
verification followed by implementation in this chat. The implementation reuses
the existing Evals loader, configuration merge helper and recovery owner.

## Verified existing behavior

- `config.deep_merge_dicts` recursively merges dictionaries, replaces lists,
  and does not mutate the baseline.
- `EvalConfigLoader` already implements reload, atomic YAML publication,
  mutable drafts and persistence failures.
- `_default_config_path()` selects shipped YAML; current raw participant
  binding and recovery discovery incorrectly give that resource user ownership.
- Current TOML serialization drops nulls; the YAML mapping policy supports them.
- Shipped YAML defines twelve task types, while the exception fallback has four.

## Required change

Preserve `_default_config_path()` for packaged resources. Add
`_override_config_path(config_selector=None)` returning `eval_overrides.yaml`
beside the effective config; an explicit selector keeps recovery discovery
free of runtime config bootstrap. Default loaders merge package defaults and
sparse overrides. Explicit custom-path loaders keep their full-file contract.

Default saves persist only overrides and actual mutable-draft changes; explicit
updates equal to a default stay pinned. Missing overrides are clean and create
no file. Failed IO preserves the draft and persisted baseline. Explicit exports
contain the full effective mapping and do not mark the local draft saved.

Runtime raw binding and recovery discovery use the same override selector.
Recovery declares missing overrides unused and continues retaining legacy
definitions with authenticated provenance and inactive activation behavior.
Package resources never enter the private mutable-file lifecycle.

## Boundaries

Python >=3.12; retain YAML; add no dependencies or UI. Preserve existing private
path checks and maintenance gates. Do not alter per-run configuration overrides.
Use targeted tests only. Existing ADRs govern the infrastructure; ADR-220 records
the changed ownership and rejected alternative formats.
