# Chatbook Core Brand Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a production-grade internal Chatbook logo pack that rebuilds repeatably inside a recorded export environment and contains Sammy Vadem's vector emblem, the outlined Chatbook wordmark, horizontal and stacked lockups, dark-editorial and cyber-noir variants, raster exports, technical validation, and usage documentation while keeping public release explicitly blocked on the remaining release gates.

**Architecture:** Following ADR-026, a new root-level `brand/` directory is the source of truth for master SVG geometry, palette/export metadata, clearance, approval, and font-license records, recorded-environment export tooling, derived SVG/PNG files, and review boards. The application does not load these files in this plan; package-data changes, README replacement, splash integration, campaign art, character model sheets, and copy remain separate follow-on work. Python tooling uses the existing optional `svg` extra (CairoSVG), Pillow, FontTools for repeatable wordmark outlining, and standard-library XML/JSON handling.

**Tech Stack:** SVG, JSON, Python 3.11, CairoSVG via `.[svg]`, Pillow, FontTools (design time), pytest

---

## Scope Boundary

This plan covers one independently completable internal unit: the core logo
asset pack. It is not a public-release tranche.

Included:

- Preliminary identity-collision checkpoint
- One-color side-profile emblem
- Outlined `Chatbook` wordmark
- Horizontal and stacked lockups
- One-color, reversed, dark-editorial, and cyber-noir variants
- Recorded-environment repeatable SVG/PNG exports
- Small-size, palette, structure, and clipping validation
- Visual review boards and brand-pack documentation
- Typeface provenance and license record

Deferred to separate plans:

- Research-mode and launch-formal wardrobe design
- Full front/profile/three-quarter character model sheet
- Expression sheet and campaign illustration production
- Campaign copy
- Technical companion typeface and broader typography system
- Motion behavior
- README, website, social, installer, package, or Textual application integration
- Changes to `tldw_chatbook/assets`, `pyproject.toml`, splash screens, or application code

## ADR Check

- ADR required: yes
- ADR path: `backlog/decisions/026-brand-asset-source-and-export-boundary.md`
- Reason: The plan establishes long-lived master/derived ownership, commits
  generated exports, and chooses an optional design toolchain plus its default
  CI boundary. Application/package integration remains outside ADR-026 and
  must repeat the ADR check.

## Governance Acceptance Gate

- The user approved proceeding with the core Chatbook identity work on
  2026-07-27; ADR-026 is `Accepted`.
- This plan, ADR-026, and the ADR index must be committed together before
  implementation begins.
- That governance commit and every later asset-bearing commit remain
  local/private. Because the canonical GitHub remote is public, do not push the
  branch, open a public pull request, or merge it into a public branch while
  `Public release permitted` remains `no`.
- Selecting an execution mode authorizes local implementation only; it does not
  authorize publication.

## Prerequisites and Human Gates

1. Verify the accepted plan/ADR/index governance commit exists locally and has
   not been pushed to the public remote.
2. Execute in a dedicated git worktree; do not use the current dirty working tree.
3. Before Task 1, create one atomic Backlog.md task for this core asset pack,
   move it to `In Progress`, and add this plan as its Implementation Plan.
   Record the ADR decision exactly as shown above. Use the task ID returned by
   `backlog task create` in later `backlog task edit <id> ...` commands.

   ```bash
   backlog task create "Produce Chatbook core brand asset pack" \
     -s "To Do" -a @codex --priority medium --plain \
     --no-dod-defaults \
     -d "Turn the approved Chatbook and Sammy Vadem identity direction into a recorded-environment, repeatably built internal logo asset pack while preserving the public-release clearance gate." \
     --ac "Preliminary collision screen is recorded and material collisions block production." \
     --ac "Approved one-color Sammy emblem remains recognizable at 16, 24, and 32 pixels." \
     --ac "Outlined Chatbook wordmark and horizontal and stacked lockups pass human visual review." \
     --ac "All four declared color variants rebuild repeatably inside the recorded export environment from validated master SVGs." \
     --ac "Typeface provenance, license, usage rules, and public-release status are documented." \
     --ac "Brand tests, scoped static analysis, and the full repository test suite pass." \
     --dod "All implementation-plan tasks and human review gates are complete." \
     --dod "Acceptance criteria are checked and implementation notes are present." \
     --dod "Automated tests and scoped static analysis pass with no regressions." \
     --dod "Relevant brand documentation and license records are current." \
     --dod "Self-review confirms generated assets match committed masters." \
     --dod "ADR decision is recorded and remains accurate." \
     --dod "Public-release clearance status is explicit and truthful." \
     --ref "Docs/superpowers/specs/2026-07-27-chatbook-brand-mascot-design.md" \
     --ref "Docs/superpowers/plans/2026-07-27-chatbook-core-brand-assets.md" \
     --ref "backlog/decisions/026-brand-asset-source-and-export-boundary.md"
   backlog task edit <id> -s "In Progress" --plain \
     --plan "Execute Docs/superpowers/plans/2026-07-27-chatbook-core-brand-assets.md. ADR required: yes. ADR path: backlog/decisions/026-brand-asset-source-and-export-boundary.md. Reason: master/derived ownership, committed exports, and optional design-tooling policy."
   backlog task <id> --plain
   ```

   Expected: one `In Progress` task with six unchecked acceptance criteria,
   seven unchecked Definition-of-Done items, all three document references,
   the implementation plan, and the ADR record.

   Replace ADR-026's `Related Tasks` metadata with the exact new task ID (an
   allowed metadata-only update under the ADR immutability rule). Resolve the
   exact task filename with `rg -l "^id: <id>$" backlog/tasks`, then stage and
   commit only that file and ADR-026 before Task 1:

   ```bash
   git add "backlog/tasks/<resolved-task-file>.md" \
     backlog/decisions/026-brand-asset-source-and-export-boundary.md
   git commit -m "chore(backlog): track Chatbook core brand assets"
   ```
4. Use the approved design spec:
   `Docs/superpowers/specs/2026-07-27-chatbook-brand-mascot-design.md`.
5. Install development and SVG export dependencies:

   ```bash
   python -m pip install -e ".[dev,svg]"
   python -m pip install fonttools mypy
   ```

   CairoSVG also needs the platform Cairo library. If `python -c "import cairosvg"` raises a dynamic-library error, install Cairo using the platform's normal package manager before continuing.
   `fonttools` and `mypy` are design/development-time utilities for outlining
   the approved typeface and performing scoped static analysis; they are not
   added to the application's runtime dependencies.
6. Task 1 is a human-owned clearance gate. Do not represent search notes as legal advice.
7. Tasks 2, 3, and 4 each end with visual approval. Do not mark those tasks complete from automated tests alone.

## File Map

### New source and documentation

- `brand/README.md` — source-of-truth rules, variant matrix, usage, rebuild commands, and deferred integration boundary
- `brand/brand.json` — palette tokens, master asset list, export dimensions, and identity metadata
- `brand/clearance.md` — dated preliminary collision review and explicit formal-clearance status
- `brand/approvals/emblem.md` — hashed small-size, stress, originality, and emblem approval
- `brand/approvals/wordmark.md` — hashed type-direction and lockup approval
- `brand/approvals/dual-mode.md` — hashed palette/contrast and dual-mode approval
- `brand/licenses/README.md` — selected wordmark typeface provenance, version/source, modification notes, and license obligations
- `brand/licenses/upstream-license.txt` — exact upstream license file content for the selected typeface
- `brand/wordmark.json` — selected font digest, static face or axes, advances, and approved optical offsets
- `brand/source/chatbook-emblem.svg` — master side-profile emblem geometry
- `brand/source/chatbook-wordmark.svg` — outlined `Chatbook` wordmark geometry
- `brand/source/chatbook-lockup-horizontal.svg` — emblem facing horizontal wordmark
- `brand/source/chatbook-lockup-stacked.svg` — compact stacked lockup

### New tooling and tests

- `brand/scripts/build_brand_assets.py` — validate master SVG roles, apply palettes, write SVG/PNG variants, and create review boards
- `brand/scripts/outline_wordmark.py` — verify and instantiate the selected font, apply recorded metrics/offsets, and emit the outlined wordmark
- `Tests/Brand/test_brand_assets.py` — manifest, SVG safety, palette, raster size, padding, inventory, approval, and recorded-environment repeatability tests

### Derived, committed outputs

- `brand/dist/svg/<variant>/chatbook-emblem.svg`
- `brand/dist/svg/<variant>/chatbook-wordmark.svg`
- `brand/dist/svg/<variant>/chatbook-lockup-horizontal.svg`
- `brand/dist/svg/<variant>/chatbook-lockup-stacked.svg`
- `brand/dist/png/<variant>/chatbook-emblem-<size>.png`
- `brand/dist/png/<variant>/chatbook-wordmark-<width>w.png`
- `brand/dist/png/<variant>/chatbook-lockup-horizontal-<width>w.png`
- `brand/dist/png/<variant>/chatbook-lockup-stacked-<width>w.png`
- `brand/dist/build-info.json` — exact export environment and generated-file hashes
- `brand/review/generated/emblem-small-size-sweep.png`
- `brand/review/generated/emblem-production-stress-test.png`
- `brand/review/generated/lockup-mode-comparison.png`
- `brand/review/manual/wordmark-candidate-comparison.png`

The `<variant>` set is:

- `one-color-ink`
- `one-color-reverse`
- `dark-editorial`
- `cyber-noir`

## Asset Contract

Every master SVG must:

- Be self-contained and use a `viewBox`.
- Start with a canvas-sized background shape using
  `data-brand-fill="background"`; transparent variants resolve it to
  `fill="none"`.
- Contain only `<svg>`, `<title>`, `<desc>`, `<g>`, and basic vector-shape
  elements; no embedded raster `<image>`, `<use>`, filter, mask, or font.
- Contain no `<text>`, `<style>`, `<script>`, `foreignObject`, `href`, external
  URL, inline CSS, or external stylesheet.
- Include a `<title>` and `<desc>`.
- Use `data-brand-fill` and `data-brand-stroke` attributes for palette roles.
- Use only these roles: `background`, `foreground`, `accent`, `secondary`.
- Preserve identical geometry across generated color variants.

The build script owns `brand/dist/` and `brand/review/generated/`. Do not
hand-edit those outputs. The wordmark candidate comparison under
`brand/review/manual/` is a retained design-decision artifact, not a generated
release asset.

Every visual approval file uses this auditable contract:

```markdown
# <Gate> Approval

- Reviewer: <name>
- Review date (UTC): <YYYY-MM-DD>
- Decision: `approved`

## Criteria

- [x] <criterion>

## Reviewed SHA-256

| Repository-relative path | SHA-256 |
| --- | --- |
| `path/to/file` | `<64 lowercase hexadecimal characters>` |
```

The approval test recomputes every table digest and fails after any reviewed
file changes. A rejected review is recorded in work notes, corrected, and
re-presented; only an explicit approved decision enters the committed approval
record.

---

### Task 1: Record the Identity-Clearance Gate

**Files:**
- Create: `brand/clearance.md`

- [ ] **Step 1: Perform and record a preliminary collision search**

Search the relevant trademark registries, general web, major software catalogs, app stores, package indexes, social platforms, and character databases for:

- `Chatbook`
- `Chatbook AI`
- `Samira Vadem`
- `Sammy Vadem`
- Visually similar woman-profile AI marks

Record search date, searcher, jurisdictions/catalogs checked, exact queries, material results, and follow-up owner. This is a collision screen, not legal clearance.

- [ ] **Step 2: Write the clearance record**

Create `brand/clearance.md` with this exact structure:

```markdown
# Chatbook Identity Clearance

## Status

- Preliminary collision review: `clear` | `blocked`
- Formal legal clearance: `pending`
- Public release permitted: `no`
- Reviewer:
- Review date:
- Approved design-spec commit:

This document is a preliminary collision screen, not legal advice or formal
legal clearance.

## Search Record

| Date | Searcher | Source | Query | Material result | Follow-up |
| --- | --- | --- | --- | --- | --- |

## Decision

State why internal production may continue or why work is blocked.

## Release Gate

Formal legal clearance is required before any public release of the final
Chatbook/Samira “Sammy” Vadem logo or campaign identity.
```

- [ ] **Step 3: Stop if the preliminary review is blocked**

Pause until the named human reviewer records either `clear` or `blocked`.

If the result is `blocked`, stop this plan and return to naming/identity
design. If it is `clear`, internal production may continue while formal legal
clearance and public release remain pending. Do not work around a blocked gate
by changing spelling or visual details without user approval.

- [ ] **Step 4: Verify the record contains no unsupported legal claim**

Run:

```bash
rg -n "Preliminary collision review|Formal legal clearance|Public release permitted|Reviewer|Review date|Approved design-spec commit|not legal advice|collision screen" brand/clearance.md
```

Expected: the status and limitation are explicit; the document does not claim trademark registration or attorney approval.

- [ ] **Step 5: Commit**

```bash
git add brand/clearance.md
git commit -m "docs(brand): record identity clearance gate"
```

---

### Task 2: Build the One-Color Emblem and Export Foundation

**Files:**
- Create: `brand/brand.json`
- Create: `brand/source/chatbook-emblem.svg`
- Create: `brand/scripts/build_brand_assets.py`
- Create: `Tests/Brand/test_brand_assets.py`
- Create: `brand/dist/svg/one-color-ink/chatbook-emblem.svg`
- Create: `brand/dist/svg/one-color-reverse/chatbook-emblem.svg`
- Create: `brand/dist/png/one-color-ink/chatbook-emblem-{16,24,32,64,128,256,512}.png`
- Create: `brand/dist/png/one-color-reverse/chatbook-emblem-{16,24,32,64,128,256,512}.png`
- Create: `brand/dist/build-info.json`
- Create: `brand/review/generated/emblem-small-size-sweep.png`
- Create: `brand/review/generated/emblem-production-stress-test.png`
- Create: `brand/approvals/emblem.md`

- [ ] **Step 1: Write failing manifest and SVG-safety tests**

Create `Tests/Brand/test_brand_assets.py` with these foundations:

```python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_ROOT = REPO_ROOT / "brand"
MANIFEST_PATH = BRAND_ROOT / "brand.json"
SVG_NS = "{http://www.w3.org/2000/svg}"
ALLOWED_TAGS = {
    "svg", "title", "desc", "g", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon",
}
ALLOWED_ROLES = {"background", "foreground", "accent", "secondary"}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_svg(relative_path: str) -> ET.Element:
    return ET.parse(BRAND_ROOT / relative_path).getroot()


def test_manifest_identity_and_directional_tokens() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["identity"] == {
        "brand": "Chatbook",
        "mascot": 'Samira "Sammy" Vadem',
    }
    assert manifest["palettes"]["dark-editorial"] == {
        "background": "#0B0B0D",
        "foreground": "#E8DDC9",
        "accent": "#6D1F2B",
        "secondary": "#B8B3A9",
    }
    assert manifest["palettes"]["cyber-noir"] == {
        "background": "#07111F",
        "foreground": "#D8E7F0",
        "accent": "#42D9FF",
        "secondary": "#8B5CFF",
    }


def test_master_svgs_are_self_contained_vectors() -> None:
    for asset in load_manifest()["assets"]:
        root = parse_svg(asset["source"])
        assert root.tag == f"{SVG_NS}svg"
        assert root.get("viewBox")
        assert root.find(f"{SVG_NS}title") is not None
        assert root.find(f"{SVG_NS}desc") is not None
        for element in root.iter():
            assert local_name(element.tag) in ALLOWED_TAGS
            assert "style" not in element.attrib
            for attribute, value in element.attrib.items():
                assert "http://" not in value
                assert "https://" not in value
                assert local_name(attribute) != "href"
                if attribute in {"data-brand-fill", "data-brand-stroke"}:
                    assert value in ALLOWED_ROLES
            if element.get("fill") not in {None, "none"}:
                assert element.get("data-brand-fill") in ALLOWED_ROLES
            if element.get("stroke") not in {None, "none"}:
                assert element.get("data-brand-stroke") in ALLOWED_ROLES
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py -v
```

Expected: FAIL because `brand/brand.json` and the master SVG do not exist.

- [ ] **Step 3: Create the initial manifest**

Create `brand/brand.json`:

```json
{
  "schema_version": 1,
  "identity": {
    "brand": "Chatbook",
    "mascot": "Samira \"Sammy\" Vadem"
  },
  "enabled_variants": [
    "one-color-ink",
    "one-color-reverse"
  ],
  "palettes": {
    "one-color-ink": {
      "background": null,
      "foreground": "#0B0B0D",
      "accent": "#0B0B0D",
      "secondary": "#0B0B0D"
    },
    "one-color-reverse": {
      "background": null,
      "foreground": "#E8DDC9",
      "accent": "#E8DDC9",
      "secondary": "#E8DDC9"
    },
    "dark-editorial": {
      "background": "#0B0B0D",
      "foreground": "#E8DDC9",
      "accent": "#6D1F2B",
      "secondary": "#B8B3A9"
    },
    "cyber-noir": {
      "background": "#07111F",
      "foreground": "#D8E7F0",
      "accent": "#42D9FF",
      "secondary": "#8B5CFF"
    }
  },
  "assets": [
    {
      "id": "emblem",
      "source": "source/chatbook-emblem.svg",
      "square_sizes": [16, 24, 32, 64, 128, 256, 512]
    }
  ]
}
```

- [ ] **Step 4: Establish the emblem canvas and profile silhouette**

Create `brand/source/chatbook-emblem.svg` as clean vector geometry based on the approved side-profile concept. Preserve:

- Right-facing calm profile

Refine:

- Forehead/nose/lip contour
- Neck termination
- Padding inside the square `viewBox`

Start with the canvas background, profile, ear, and neck only. The file must
use authored path geometry, not an automatic raster trace or embedded PNG.

- [ ] **Step 5: Add Sammy's identifying geometry**

Add the approved identity anchors without changing the facial silhouette:

- Compact three-section page-ribbon low knot
- Horizontal bookmark pin
- Index-tab earring
- Minimal temple-to-jaw seam

Refine the earring scale and negative space between knot sections. Keep the
details sparse enough to survive the 16-pixel export.

- [ ] **Step 6: Apply palette roles and validate the master contract**

Apply palette roles with attributes such as:

```xml
<rect
  width="100%"
  height="100%"
  data-brand-fill="background"
  fill="none"
/>
<path
  d="..."
  data-brand-fill="foreground"
  fill="#0B0B0D"
/>
```

Do not add the wordmark in this task.

Run:

```bash
pytest Tests/Brand/test_brand_assets.py \
  -k "manifest_identity or master_svgs" -v
```

Expected: PASS for the manifest and emblem source contract.

- [ ] **Step 7: Write failing exporter, inventory, and board tests**

Before creating the exporter, add these executable tests:

- `test_committed_emblem_output_inventory_is_exact` computes the expected two
  SVGs, fourteen PNGs, `build-info.json`, and two generated review boards from
  `brand.json`; it fails on a missing or extra generated file.
- `test_build_info_hashes_match_committed_generated_files` runs without
  CairoSVG, requires a digest entry for every generated file except
  `build-info.json` itself, rejects missing/extra entries, and recomputes each
  committed digest.
- `test_emblem_png_exports_have_exact_dimensions_and_padding` opens every
  committed PNG and asserts the declared square size plus nonempty transparent
  bounds inset from all four edges.
- `test_emblem_review_boards_exist_and_are_nonempty` opens both generated
  boards and asserts nonzero width, height, and visible content.
- `test_failed_validation_preserves_installed_distribution` hashes the current
  distribution and `brand/review/generated/`, calls the build module with an
  invalid copied manifest, and asserts the call fails before any installed
  hash changes.
- `test_recorded_environment_rebuild_is_repeatable` uses
  `pytest.importorskip("cairosvg")`, runs the exporter twice, and compares
  SHA-256 values for every file under `brand/dist/` and
  `brand/review/generated/`. Do not hash `brand/review/manual/`.

The committed-artifact tests must not import CairoSVG or the build module at
module import time, so they continue to run in the repository's default CI
environment.

- [ ] **Step 8: Run the exporter tests to verify they fail**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py \
  -k "output_inventory or build_info or png_exports or review_boards or failed_validation or recorded_environment" -v
```

Expected: committed-artifact tests fail because outputs do not exist; the
rebuild test either fails because the script is absent or skips only if
CairoSVG is unavailable.

- [ ] **Step 9: Implement validate-stage-install exporting**

Create `brand/scripts/build_brand_assets.py` with these boundaries:

1. `validate_brand_root()` loads the manifest and validates every declared
   master before any output mutation. It enforces the SVG contract, palette
   roles, unique IDs, declared dimensions, source existence, and exact expected
   inventory.
2. `apply_palette()` deep-copies a parsed master, resolves every role, strips
   role attributes in derived SVGs, and never changes a source file.
3. `render_png()` imports CairoSVG lazily and raises an actionable message
   directing the contributor to `python -m pip install -e ".[svg]"`.
4. `build_candidate()` writes `dist/` and `review/generated/` beneath a
   `tempfile.TemporaryDirectory(dir=BRAND_ROOT, prefix=".brand-build-")`.
5. After all non-manifest outputs exist, `build-info.json` records Python,
   Pillow, CairoSVG, cairocffi, native Cairo,
   operating-system, and architecture versions plus SHA-256 for every other
   generated file. It contains no timestamp or volatile absolute path and
   states that byte repeatability is scoped to that recorded environment.
6. Candidate validation then reparses every SVG, opens every PNG, verifies the
   exact inventory including `build-info.json`, recomputes every recorded
   digest, and checks dimensions and transparent validation bounds.
7. Only after every candidate check passes does `install_candidate()` replace
   `brand/dist/` and `brand/review/generated/`, retaining a temporary backup
   until both replacements succeed and restoring it on failure.

The two Pillow boards must composite actual generated PNGs; they never redraw
the emblem. The small-size board shows native and nearest-neighbor-enlarged 16,
24, and 32 pixel ink/reverse exports. The stress board shows the actual
32-pixel ink export in black-on-white, grayscale, 1-bit threshold, and a
16-pixel coarse nearest-neighbor approximation. The latter is not a vendor
embroidery proof.

- [ ] **Step 10: Build and run technical validation**

Run in the export-enabled environment:

```bash
python brand/scripts/build_brand_assets.py --write
pytest Tests/Brand/test_brand_assets.py -v
python brand/scripts/build_brand_assets.py --write
git diff --exit-code -- brand/dist brand/review/generated
git diff --check
```

Expected: all brand tests pass, a same-environment second build changes no
generated output, and no whitespace errors are present.

- [ ] **Step 11: Perform and record the emblem visual gate**

Open both generated emblem boards and compare them with the approved spec
references. Required human criteria:

- Profile still reads as Sammy rather than a generic woman.
- Knot, pin, and earring remain distinguishable.
- Expression does not become severe.
- 16/24/32 pixel exports remain recognizable.
- Neck and negative space feel intentional.
- Grayscale and 1-bit treatments preserve the profile.
- The coarse approximation keeps the anchors separable enough for a later
  vendor embroidery proof.
- The result passes every originality guardrail in the approved spec.

Pause for a named human decision. If rejected, edit only the master, rebuild,
rerun tests, and repeat. If approved, create `brand/approvals/emblem.md` with
reviewer, UTC date, decision, design-spec commit, originality decision, every
criterion, and SHA-256 values for the master SVG and both reviewed boards.
Verify the recorded hashes against the current files with `shasum -a 256`.

- [ ] **Step 12: Commit**

```bash
git add brand/brand.json brand/source/chatbook-emblem.svg \
  brand/scripts/build_brand_assets.py brand/dist brand/review/generated \
  brand/approvals/emblem.md Tests/Brand/test_brand_assets.py
git commit -m "feat(brand): add validated one-color emblem pack"
```

---

### Task 3: Select, License, and Outline the Wordmark

**Files:**
- Create: `brand/licenses/README.md`
- Create: `brand/licenses/upstream-license.txt`
- Create: `brand/wordmark.json`
- Create: `brand/scripts/outline_wordmark.py`
- Create: `brand/source/chatbook-wordmark.svg`
- Create: `brand/source/chatbook-lockup-horizontal.svg`
- Create: `brand/source/chatbook-lockup-stacked.svg`
- Create: `brand/review/manual/wordmark-candidate-comparison.png`
- Create: `brand/approvals/wordmark.md`
- Modify: `brand/brand.json`
- Modify: `brand/scripts/build_brand_assets.py`
- Modify: `Tests/Brand/test_brand_assets.py`

- [ ] **Step 1: Add failing source, provenance, and output tests**

Extend the source contract:

```python
def test_wordmark_and_lockups_are_outlined_and_labeled() -> None:
    manifest = load_manifest()
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    assert {"emblem", "wordmark", "lockup-horizontal", "lockup-stacked"} <= set(assets)
    for asset_id in ("wordmark", "lockup-horizontal", "lockup-stacked"):
        root = parse_svg(assets[asset_id]["source"])
        assert root.get("data-brand-name") == "Chatbook"
        assert all(local_name(element.tag) != "text" for element in root.iter())


def test_selected_typeface_license_is_recorded() -> None:
    provenance = (BRAND_ROOT / "licenses" / "README.md").read_text(
        encoding="utf-8"
    )
    license_text = (BRAND_ROOT / "licenses" / "upstream-license.txt").read_text(
        encoding="utf-8"
    )
    required = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in provenance.splitlines()
        if line.startswith("- ") and ":" in line
    }
    for field in (
        "- Typeface", "- Upstream source", "- Version or commit",
        "- Input filename", "- Input SHA-256", "- Selected face or axes",
        "- FontTools version", "- Upstream license filename",
        "- Wordmark modifications",
    ):
        assert required[field]
    assert len(required["- Input SHA-256"]) == 64
    normalized_license = license_text.upper()
    assert "SIL OPEN FONT LICENSE" in normalized_license
    assert "VERSION 1.1" in normalized_license
```

Also add failing tests that:

- `brand/wordmark.json` contains the same 64-character input digest, exact
  upstream commit, source filename, either a static face or explicit variable
  axes, units-per-em, glyph advances, explicit per-glyph optical offsets, and
  a nonempty selection reviewer/date/decision.
- The expected wordmark and both lockup SVGs plus every declared PNG width are
  included in the exact generated inventory.
- The smallest wordmark and lockup PNGs open successfully and their
  transparent validation bounds remain inset.
- `brand/scripts/outline_wordmark.py` exists and the recorded source SVG hash
  matches `brand/source/chatbook-wordmark.svg`.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py \
  -k "wordmark or selected_typeface" -v
```

Expected: FAIL because wordmark sources, config, outputs, and license records
do not exist.

- [ ] **Step 3: Produce an editorial-serif comparison**

Start with two officially maintained, OFL-1.1 candidates:

- [Newsreader](https://github.com/productiontype/Newsreader)
- [Source Serif 4](https://github.com/adobe-fonts/source-serif)

Retrieve each font from its official repository into a temporary directory;
do not commit font binaries. Record the exact upstream commit, font filename,
and SHA-256 on the comparison board. Use Pillow for the raster comparison and
FontTools only after selection to extract the chosen glyph outlines.

Render `Chatbook` in each candidate at the intended lockup scale beside the approved emblem. Compare:

- Moderate contrast rather than extreme Didone fragility
- Open counters at small sizes
- Shape of uppercase `C`
- Readability of `tb` and `oo`
- Compatibility with the emblem's curved profile and sharp index details

Save the board as
`brand/review/manual/wordmark-candidate-comparison.png`.

- [ ] **Step 4: Obtain human type-direction approval**

Pause until the user chooses one exact font binary and either a static face or
every variable axis value, including weight and optical size where applicable.
Record the reviewer, date, upstream commit, filename, and SHA-256 before
outlining.

Do not infer approval from the original raster sample; it established a direction, not a font license or exact letterforms.

- [ ] **Step 5: Record provenance and license**

Write `brand/licenses/README.md`:

```markdown
# Chatbook Wordmark Typeface Provenance

- Typeface:
- Designer/foundry:
- Upstream source:
- Version or commit:
- Input filename:
- Input SHA-256:
- Selected face or axes:
- FontTools version:
- Upstream license filename:
- License: SIL Open Font License 1.1
- Date retrieved:
- Wordmark modifications:

The distributed Chatbook wordmark is outlined vector geometry. The font binary
is not required to display the production logo.
```

Copy the exact upstream license content, whether the source repository names it
`OFL.txt`, `LICENSE.md`, or another filename, into
`brand/licenses/upstream-license.txt`. Record the original filename.

- [ ] **Step 6: Implement repeatable wordmark extraction**

Create `brand/wordmark.json` and `brand/scripts/outline_wordmark.py`. The
script accepts `--font <temporary-font-path> --write`, verifies the input
SHA-256, and:

1. Opens the exact binary with `fontTools.ttLib.TTFont`.
2. Uses `fontTools.varLib.instancer.instantiateVariableFont` with every
   recorded axis when the input is variable; otherwise verifies the recorded
   static face.
3. Resolves `Chatbook` through the best character map.
4. Positions glyphs from recorded `hmtx` advances plus explicit optical
   offsets in `wordmark.json`; it does not depend on implicit host shaping.
5. Extracts paths with `fontTools.pens.svgPathPen.SVGPathPen`, normalizes the
   coordinate system and `viewBox`, and writes stable path ordering.
6. Writes the output SHA-256 back to `wordmark.json` only with `--write`.

Run the script twice against the unchanged temporary font and verify the source
SVG hash is identical. Then inspect `brand/source/chatbook-wordmark.svg`:

- Exact visible spelling: `Chatbook`
- All letters converted to paths
- `data-brand-name="Chatbook"` on the root SVG
- `data-brand-fill="foreground"` on color-bearing geometry
- No `<text>` or font dependency
- Optical spacing corrected around `C-h`, `t-b`, and `oo`
- No decorative modifications that reduce readability

- [ ] **Step 7: Assemble horizontal and stacked master lockups**

Create both master lockups from the approved emblem and outlined wordmark geometry:

- Horizontal: emblem left, facing the wordmark; emblem approximately cap height; clear separation
- Stacked: emblem centered above wordmark; designed for square-ish placements

Do not independently redraw the emblem. Copy its approved paths exactly into
each lockup master and preserve their internal geometry.

- [ ] **Step 8: Extend the manifest and exporter**

Add:

```json
{
  "id": "wordmark",
  "source": "source/chatbook-wordmark.svg",
  "widths": [256, 512, 1024]
},
{
  "id": "lockup-horizontal",
  "source": "source/chatbook-lockup-horizontal.svg",
  "widths": [512, 1024, 2048]
},
{
  "id": "lockup-stacked",
  "source": "source/chatbook-lockup-stacked.svg",
  "widths": [256, 512, 1024]
}
```

Update the build script so `widths` produce `<asset-id>-<width>w.png`; candidate
validation and exact-inventory checks must include every new SVG and PNG.

- [ ] **Step 9: Build and run technical validation**

Run:

```bash
python brand/scripts/build_brand_assets.py --write
pytest Tests/Brand/test_brand_assets.py -v
python brand/scripts/build_brand_assets.py --write
git diff --exit-code -- brand/dist brand/review/generated
git diff --check
```

Expected: all tests pass and the second same-environment build produces no
generated diff.

- [ ] **Step 10: Perform and record the wordmark/lockup visual gate**

Human approval must confirm:

- Exact wordmark spelling
- Editorial/technical balance
- Emblem-to-wordmark scale
- Horizontal spacing
- Stacked lockup balance
- Readability at the smallest exported width

Pause for a named decision. If approved, create
`brand/approvals/wordmark.md` with reviewer, UTC date, decision, exact font
choice/axes, every criterion, and SHA-256 values for `wordmark.json`, the three
master SVGs, the manual comparison board, and the smallest reviewed PNG for
each lockup. Verify the hashes with `shasum -a 256`.

- [ ] **Step 11: Commit**

```bash
git add brand/brand.json brand/wordmark.json brand/licenses brand/source \
  brand/scripts brand/dist brand/review brand/approvals/wordmark.md \
  Tests/Brand/test_brand_assets.py
git commit -m "feat(brand): add outlined wordmark and lockups"
```

---

### Task 4: Generate and Validate the Dual-Mode Variant Matrix

**Files:**
- Modify: `brand/brand.json`
- Modify: `brand/scripts/build_brand_assets.py`
- Modify: `Tests/Brand/test_brand_assets.py`
- Create: `brand/dist/svg/dark-editorial/*`
- Create: `brand/dist/svg/cyber-noir/*`
- Create: `brand/dist/png/dark-editorial/*`
- Create: `brand/dist/png/cyber-noir/*`
- Create: `brand/review/generated/lockup-mode-comparison.png`
- Create: `brand/approvals/dual-mode.md`

- [ ] **Step 1: Add failing palette-role and variant tests**

Add these executable tests before enabling the two color modes:

- `test_generated_svg_palette_contract` parses every output; each `fill` and
  `stroke` is `none` or belongs to that variant, no palette-role attribute
  remains, and no forbidden tag/attribute was introduced.
- `test_geometry_and_viewbox_are_identical_across_variants` compares the root
  `viewBox` and a shape/geometry signature for the same asset in all modes.
- `test_generated_output_inventory_matches_manifest_exactly` covers all four
  variants, asserts `enabled_variants` is the exact four-mode list, and fails
  on missing or stale files.
- `test_essential_role_contrast_meets_policy` computes WCAG relative-luminance
  contrast and asserts `foreground` and `secondary` are at least `4.5:1`
  against an opaque mode background. `accent` is declared decorative-only and
  may never be the sole carrier of a silhouette or identity anchor. Record the
  measured ratios in the assertion message and later in the README.
- `test_transparent_validation_bounds_are_inset_at_every_size` uses
  `pytest.importorskip("cairosvg")`, renders a validation copy with the
  background role forced to `none`, and asserts nonempty alpha bounds remain
  inset from all four edges at every declared size/width.
- `test_mode_comparison_board_is_generated` opens the generated board and
  verifies nonzero dimensions/content.

Geometry helper:

```python
def geometry_signature(
    root: ET.Element,
) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    geometry_attributes = {
        "d", "points", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy",
        "r", "rx", "ry", "width", "height", "transform",
    }
    signature = []
    for element in root.iter():
        attrs = tuple(
            sorted(
                (name, value)
                for name, value in element.attrib.items()
                if name in geometry_attributes
            )
        )
        if attrs:
            signature.append((local_name(element.tag), attrs))
    return signature
```

- [ ] **Step 2: Run variant tests to verify they fail**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py \
  -k "palette or variant or geometry or contrast or validation_bounds or mode_comparison" -v
```

Expected: FAIL because the color modes, contrast policy, validation render, and
mode board are not implemented.

- [ ] **Step 3: Add the role/contrast policy and finish palette application**

Update `apply_palette()` to:

- Replace every palette role in stable document order.
- Set `background` elements to `fill="none"` for transparent variants.
- Remove `data-brand-fill` and `data-brand-stroke` in derived SVGs.
- Preserve master geometry and `viewBox`.
- Never alter the source SVGs.

Update `brand/brand.json`:

```json
"enabled_variants": [
  "one-color-ink",
  "one-color-reverse",
  "dark-editorial",
  "cyber-noir"
],
"contrast_policy": {
  "minimum_essential_ratio": 4.5,
  "essential_roles": ["foreground", "secondary"],
  "decorative_only_roles": ["accent"]
}
```

Validate and record the expected ratios:

```text
dark-editorial foreground/background: 14.62
dark-editorial secondary/background: 9.42
dark-editorial accent/background: 1.76 (decorative only)
cyber-noir foreground/background: 14.98
cyber-noir secondary/background: 4.57
cyber-noir accent/background: 11.36 (still optional/decorative)
```

The emblem silhouette and its knot, pin, earring, and profile may not rely
solely on `accent`.

- [ ] **Step 4: Implement the mode-comparison board**

Implement a Pillow review-board function in
`brand/scripts/build_brand_assets.py` that composites the generated PNGs; it
must not redraw either the emblem or wordmark. Call it from `build_all()` so a
normal rebuild writes
`brand/review/generated/lockup-mode-comparison.png`.

The board must show:

- Horizontal and stacked lockups
- One-color ink on ivory
- One-color reverse on ink
- Dark-editorial lockup on ink
- Cyber-noir lockup on midnight
- Grayscale previews of both color-mode lockups
- Labels outside the logo clear-space region

- [ ] **Step 5: Generate the complete variant matrix**

Run:

```bash
python brand/scripts/build_brand_assets.py --write
```

Expected:

- One-color ink and reverse variants use one foreground color.
- Dark editorial uses ink, ivory, oxblood, and chrome only.
- Cyber-noir uses midnight, silver-white, cyan, and violet only.
- Emblem and lockup geometry do not move between modes.

- [ ] **Step 6: Run technical validation**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py -v
python brand/scripts/build_brand_assets.py --write
git diff --exit-code -- brand/dist brand/review/generated
git diff --check
```

Expected: all tests PASS; rebuild produces no diff.

- [ ] **Step 7: Perform and record the dual-mode visual gate**

Pause until a named human confirms that the modes are unmistakably the same
logo, cyber-noir remains restrained, measured contrast usage is acceptable,
oxblood carries no essential detail, grayscale remains legible, and the result
passes the originality guardrails. Create `brand/approvals/dual-mode.md` with
reviewer, UTC date, decision, criteria, measured ratios, and SHA-256 values for
`brand.json`, all four master SVGs, and the reviewed mode board. Verify hashes
with `shasum -a 256`.

- [ ] **Step 8: Commit**

```bash
git add brand/brand.json brand/scripts/build_brand_assets.py \
  brand/dist brand/review/generated brand/approvals/dual-mode.md \
  Tests/Brand/test_brand_assets.py
git commit -m "feat(brand): add dual-mode logo exports"
```

---

### Task 5: Publish the Core Brand Pack Contract

**Files:**
- Create: `brand/README.md`
- Modify: `Tests/Brand/test_brand_assets.py`
- Modify: `brand/clearance.md`

- [ ] **Step 1: Write failing documentation and approval-integrity tests**

Add:

```python
def test_brand_readme_documents_required_boundaries() -> None:
    readme = (BRAND_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "Source of truth",
        "Do not edit `brand/dist/`",
        "Dark editorial",
        "Cyber-noir",
        "Minimum sizes",
        "Clear space",
        "Application integration is out of scope",
        "Formal legal clearance",
        "companion typeface",
        "embroidery proof",
    ):
        assert required in readme
```

Add `test_approval_records_match_current_files`. It must parse the three
approval files using the contract above, require nonempty reviewer/date,
`Decision: approved`, all criteria checked, and at least one hash. Recompute
every repository-relative SHA-256 and compare it to the recorded lowercase
digest.

Add `test_preliminary_clearance_record_is_auditable`. It requires a named
reviewer, ISO date, 40-character approved-spec commit, at least one populated
search row, an explicit `clear` decision for this completed path, formal
clearance still `pending`, and public release still `no`.

- [ ] **Step 2: Run the documentation/integrity tests to verify they fail**

Run:

```bash
pytest Tests/Brand/test_brand_assets.py \
  -k "readme_documents_required_boundaries or approval_records or clearance_record" -v
```

Expected: the README test fails because `brand/README.md` does not exist.
Approval integrity must already pass before writing documentation.

- [ ] **Step 3: Write the brand-pack README**

Document:

- Source of truth and directory structure
- Rebuild command
- Which files are master vs derived
- Variant selection matrix
- Minimum sizes established by the approved review board
- Clear-space rule expressed as a multiple of the emblem's earring width
- One-color fallback
- Background guidance
- Misuse rules
- Typeface provenance link
- Clearance status and public-release gate
- Remaining public-release work: formal legal clearance, technical companion
  typeface selection, and vendor-level embroidery proof
- Explicit statement that application/package integration is out of scope
- Links to the design spec and concept references

Do not add campaign headlines or invent application placements.

- [ ] **Step 4: Update the clearance status honestly**

Keep:

```markdown
- Formal legal clearance: `pending`
- Public release permitted: `no`
```

until the user supplies formal clearance evidence. The technical asset pack can be complete for internal use while public release remains blocked.

- [ ] **Step 5: Run the full brand validation**

Run:

```bash
python brand/scripts/build_brand_assets.py --write
pytest Tests/Brand/test_brand_assets.py -v
git diff --exit-code -- brand/dist brand/review/generated
git diff --check
```

Expected: all tests pass; generated outputs repeat within the recorded
environment; no whitespace errors.

- [ ] **Step 6: Perform final visual review**

Review:

- `brand/review/generated/emblem-small-size-sweep.png`
- `brand/review/generated/emblem-production-stress-test.png`
- `brand/review/manual/wordmark-candidate-comparison.png`
- `brand/review/generated/lockup-mode-comparison.png`

Confirm every approval from Tasks 2–4 is represented by the current committed sources.

- [ ] **Step 7: Commit**

```bash
git add brand/README.md brand/clearance.md Tests/Brand/test_brand_assets.py
git commit -m "docs(brand): publish core identity usage guide"
```

---

## Final Verification

Run from the dedicated worktree:

```bash
python brand/scripts/build_brand_assets.py --write
pytest Tests/Brand/test_brand_assets.py -v
python -m mypy brand/scripts/build_brand_assets.py \
  brand/scripts/outline_wordmark.py Tests/Brand/test_brand_assets.py
pytest
git diff --exit-code -- brand/dist brand/review/generated
git diff --check
git status --short -- brand Tests/Brand Docs backlog
```

Expected:

- Brand tests report zero failures.
- Scoped static analysis and the full repository test suite report zero
  failures.
- A second build changes no derived output.
- No whitespace errors.
- Scoped status for brand, tests, plans, and Backlog records is empty before
  task closeout.
- `brand/clearance.md` still blocks public release unless formal legal clearance has been recorded.
- `brand/README.md` also records the technical companion typeface and
  vendor-level embroidery proof as unresolved public-release work.
- The asset-bearing branch remains local/private; no public push, pull request,
  or merge has occurred.

After these checks and the final human visual review pass, update every
Backlog task acceptance-criteria checkbox, add concise Implementation Notes
covering the approach, files, tests, trade-offs, and ADR-026, then run:

```bash
backlog task edit <id> --plain \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-ac 4 --check-ac 5 --check-ac 6 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 \
  --check-dod 5 --check-dod 6 --check-dod 7 \
  --notes "<concise implementation notes with approach, files, tests, trade-offs, and ADR required: yes; ADR path: backlog/decisions/026-brand-asset-source-and-export-boundary.md>" \
  -s Done
backlog task <id> --plain
git add "backlog/tasks/<resolved-task-file>.md"
git commit -m "chore(backlog): close Chatbook core brand assets"
git status --short
```

Expected: the task is `Done`, all acceptance criteria are checked, and
Implementation Notes are present; the closeout commit succeeds; unscoped
status is empty. Do not mark it done while a test, review gate, or acceptance
criterion remains incomplete.

Stop after the local closeout commit. Do not run `git push`, create a pull
request, or merge to the public canonical remote until the user supplies
evidence for every public-release gate and explicitly authorizes publication.

## Follow-On Plans

After this core pack is complete, write separate specs/plans for:

1. Character model sheet and the remaining two wardrobe looks
2. Campaign copy and curator-voice system
3. Application/package/splash/README integration
4. Motion and active-mode transitions
5. Public-release tranche: formal legal clearance, technical companion
   typeface selection, and vendor-level embroidery proof
