# CCP Destination-Native Route Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy CCP sidebar route with the approved Personas destination-native Behavior Profile Workbench layout.

**Architecture:** Keep the existing CCP handlers and editor/detail widgets as the behavior layer, but replace the route shell with destination header, mode strip, character/persona list pane, detail/editor pane, and attachment/readiness inspector. Characters mode is the first working slice because imported character-card selection/load/edit is the broken workflow currently under test.

**Tech Stack:** Python 3.11, Textual Screen widgets, existing CCP handlers, mounted Textual tests, actual textual-web/CDP screenshot approval.

---

### Task 1: Replace CCP Route Chrome With Destination Workbench

**Files:**
- Modify: `Tests/UI/test_ccp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`

- [x] **Step 1: Write the failing route-layout regression**
  Add a mounted test proving `CCPScreen` renders a Personas destination header, mode strip, left character/persona library pane, center detail/editor pane, and right attachment/readiness inspector, and no longer renders the legacy `#ccp-sidebar`.

- [x] **Step 2: Verify red**
  Run: `python -m pytest -q Tests/UI/test_ccp_screen.py::TestCCPScreenIntegration::test_ccp_route_uses_destination_native_personas_workbench --tb=short`
  Expected: FAIL because the current route still renders legacy CCP sidebar chrome.

- [x] **Step 3: Implement the destination-native shell**
  Replace `CCPScreen.compose_content()` with the destination grammar while keeping existing CCP detail/editor widgets mounted in the center pane.

- [x] **Step 4: Wire character list selection**
  Refresh the new character list pane from `CCPCharacterHandler.refresh_character_list()` and route character buttons to `load_character()`.

- [x] **Step 5: Verify green**
  Run the new focused test, then the existing CCP screen test file.

### Task 2: Capture Actual Screenshot For Approval

**Files:**
- Add screenshot evidence under `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/`

- [x] **Step 1: Launch/refresh textual-web**
  Use the existing CDP/textual-web workflow, not SVGs or code-rendered mockups.

- [x] **Step 2: Capture CCP/Personas route screenshot**
  Open the real route, capture the rendered screen, and present the image for approval.

- [x] **Step 3: Only proceed after approval**
  Do not expand beyond Characters mode until the actual rendered screen is approved.

## QA Evidence

- Focused route regression: `python -m pytest -q Tests/UI/test_ccp_screen.py::TestCCPScreenIntegration::test_ccp_route_uses_destination_native_personas_workbench --tb=short`
- Full CCP screen suite: `python -m pytest -q Tests/UI/test_ccp_screen.py --tb=short`
- Destination shell suite: `python -m pytest -q Tests/UI/test_destination_shells.py --tb=short`
- Navigation suite: `python -m pytest -q Tests/UI/test_screen_navigation.py --tb=short`
- Approved actual screenshot: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-20-task-60-5-ccp-native-00-started.png`
