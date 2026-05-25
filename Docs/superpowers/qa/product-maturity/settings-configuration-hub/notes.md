# Settings Configuration Hub QA Notes

Date: 2026-05-25
Branch: `codex/settings-config-next-slice`
Screen: Settings
Spec: `Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md`
Evidence method: actual textual-web/CDP screenshots driven with Playwright browser automation.

## Approved Screenshots

- Overview: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/01-overview.png`
- Providers & Models: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/02-providers-models.png`
- Appearance: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/03-appearance.png`
- Storage: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/04-storage.png`
- Privacy & Security: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/05-privacy-security.png`
- Console Behavior: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/06-console-behavior.png`
- Diagnostics: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/07-diagnostics.png`
- Advanced Config: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/08-advanced-config.png`
- Focused Settings button final fix: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/11-focused-settings-button-no-heavy-outline.png`
- Category search polish: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/13-category-search-polish.png`

User approval: approved after actual rendered screenshot review.

## Rejected Or Superseded Evidence

- `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/09-focused-button-cdp.png`
- `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/10-clicked-save-button-cdp.png`

These captures showed that one-line Settings buttons inherited the global heavy focus outline, obscuring selected button labels. The final approved capture replaces that state with readable raised/underlined focus styling.

## Workflow Coverage

- Category switching was exercised through keyboard navigation in textual-web.
- Overview and each first-slice category rendered with the three-column Settings workbench.
- Advanced Config rendered guarded raw TOML controls with validation-first save policy.
- Focused Settings buttons no longer render as empty heavy-outline rectangles.
- Category search filters Settings sections, reports match count, distinguishes primary and secondary matches, and opens the top-ranked category on Enter.
- The active search field remains visually identifiable while filtering, and the wider Settings section column keeps filtered category labels readable.
- Scope Inspector preserves the boundary that runtime MCP, ACP, and tool control stay in their own destinations.

## Automated Verification

Commands were run with the project virtual environment active:

```bash
python tldw_chatbook/css/build_css.py
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Result: `56 passed`, one existing requests dependency warning, and `git diff --check` clean.

## Residual Risks

- Settings is now a first-slice configuration hub, not a complete replacement for every legacy config surface.
- MCP, ACP, Skills, Personas, Schedules, and Workflows ownership is intentionally documented as out-of-Settings runtime control.
- Additional persisted controls should be added category-by-category with the same validation, save/revert, and screenshot approval workflow.
