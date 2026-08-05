# ADR-026: Keep Chatbook brand masters authoritative and commit reviewed exports

Status: Accepted
Date: 2026-07-27
Related Tasks: Future core brand asset-pack task
Related Spec: `Docs/superpowers/specs/2026-07-27-chatbook-brand-mascot-design.md`
Related Plan: `Docs/superpowers/plans/2026-07-27-chatbook-core-brand-assets.md`
Supersedes: N/A

## Decision

Chatbook will keep its core brand identity in a repository-root `brand/`
directory that is separate from application package assets.

The source-of-truth boundary is:

- `brand/brand.json` defines identity metadata, palette roles, assets, and
  export dimensions.
- `brand/source/` contains self-contained master SVG geometry. Masters contain
  no fonts, raster images, scripts, external references, or ungoverned colors.
- `brand/licenses/` records the exact typeface source, input binary digest,
  selected axes or static face, extraction method, modifications, and license.
- `brand/approvals/` records dated human decisions against SHA-256 digests of
  the reviewed source and review-board files.
- `brand/dist/` contains generated SVG and PNG delivery assets.
- `brand/review/` contains generated validation boards and a retained manual
  wordmark-comparison artifact.

Master SVGs and the manifest are authoritative. Generated delivery assets are
also committed so documentation, downstream consumers, and reviewers do not
need a native Cairo toolchain merely to inspect or use an approved logo.
Generated files must never be hand-edited.

The development exporter uses Python, standard-library XML/JSON handling,
CairoSVG, and Pillow. FontTools is used only to produce the initial outlined
wordmark from an approved, hash-pinned font binary; the production wordmark
does not require a font at display or export time. These are design/development
tools and do not become Chatbook runtime or package dependencies in this
tranche.

The exporter validates the complete manifest and all master SVGs before
writing. It builds into a temporary sibling directory, verifies the exact
expected output set, and replaces `brand/dist/` only after the candidate build
passes. A failed build leaves the previously reviewed distribution intact.

Byte determinism is claimed only for repeated runs inside the same recorded
export environment. The committed build record captures Python, Pillow,
CairoSVG, Cairo binding, native Cairo, operating-system, and architecture
versions. Cross-platform byte identity is not claimed. Review and tests instead
enforce source geometry, palette roles, dimensions, padding, output inventory,
and same-environment repeatability.

Repository tests that validate masters and committed artifacts run in the
default test environment. Tests that invoke CairoSVG use `pytest.importorskip`
when the optional exporter is unavailable. A dedicated brand rebuild installs
the existing `svg` extra and must run before changing generated assets. This
keeps ordinary CI compatible without weakening validation of committed files.

This decision does not place brand assets inside `tldw_chatbook/assets`, add
package data, or load the identity at runtime. Application, package, splash,
README, website, and campaign integration require a separate plan and a fresh
ADR check.

Because the canonical GitHub remote is public, every branch or commit that
contains final identity sources, generated exports, approval boards, or
clearance evidence must remain local or on an access-controlled private remote.
It must not be pushed to the public canonical remote, used to open a public
pull request, or merged into a public branch until `brand/clearance.md` records
that every public-release gate has passed. Local commits are required for
reviewability but do not grant publication authority.

## Context

The approved Chatbook identity uses one emblem and wordmark across one-color,
dark-editorial, and cyber-noir variants. Hand-maintained copies would drift in
geometry, palette, dimensions, and small-size behavior. Source-only delivery
would instead make every reviewer and downstream consumer install a native
SVG rasterization stack.

The repository already exposes CairoSVG through the optional `svg` extra, but
its default CI dependency files do not install that extra. Unconditionally
invoking the exporter from the normal test suite would therefore break the
existing CI contract. Brand production also requires auditable human visual,
originality, type, and clearance gates that ordinary unit tests cannot replace.

This work requires a canonical ADR because it establishes long-lived ownership
between master and derived files, chooses committed generated artifacts, and
defines a development toolchain and optional-dependency test boundary.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Commit only hand-authored final SVG/PNG files | Allows geometry and palette variants to drift and provides no repeatable export or validation path. |
| Commit masters but generate all delivery assets downstream | Makes basic inspection and use depend on CairoSVG and native Cairo, including environments that do not otherwise need them. |
| Add CairoSVG, Pillow, and FontTools to the application runtime | Expands installation and native-library burden for files that the runtime does not yet load. |
| Add CairoSVG to every default CI job | Changes the repository-wide CI/native dependency contract for an isolated design workflow. |
| Claim cross-platform byte-identical PNG output | Cairo and image-encoder behavior can vary by platform and version; the claim would exceed the evidence. |
| Commit the selected font binary as a permanent build input | The outlined wordmark is the reviewed master; retaining the exact digest, upstream source, license, axes, and extraction script is sufficient for audit and avoids unnecessary font redistribution. |

## Consequences

- Reviewers can inspect and use committed logo files without installing export
  tooling.
- Contributors change masters and regenerate all derived outputs together.
- Generated-output diffs remain reviewable and must match recorded approvals.
- Normal CI validates the source and committed distribution without requiring
  CairoSVG.
- Exporter-dependent tests skip only when the optional exporter is absent; a
  brand-changing task cannot complete without running them in the dedicated
  export environment.
- Rebuild reports are honest about their toolchain and platform envelope.
- Public release remains independently blocked by the identity specification's
  legal, typography, and vendor-production gates.
- Asset-bearing commits remain local/private until that release record
  explicitly permits publication; normal branch completion does not include a
  push or public pull request.
- A later application-integration tranche decides package location, loading,
  fallback behavior, and runtime ownership.

## Rollback Plan

Before application integration, this decision can be rolled back by removing
the root-level `brand/` pack, its isolated tests, and its development-only
documentation. No runtime, schema, configuration, or package-data migration is
required.

After downstream consumers adopt the committed exports, preserve the last
approved `brand/dist/` release while replacing the exporter or source policy
through a superseding ADR. Do not silently regenerate or delete previously
released assets.
