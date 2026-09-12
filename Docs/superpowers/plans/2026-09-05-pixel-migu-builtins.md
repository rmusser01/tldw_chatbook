# Pixel-migu built-in implementation plan

Goal: fresh-install users can select pixel-migu as either a character or Buddy.
Architecture: use the existing Shared Visual Identity character seed and Persona Visual / Actor Pack ownership boundaries. Keep current selections unchanged. Source: approved output/pixel-migu artifact kit in the maintainer workspace.

ADR required: yes
ADR path: backlog/decisions/122-bundled-pixel-migu-character-and-buddy.md
Reason: bundled content and first-install Persona ownership need an explicit decision; existing storage/runtime contracts remain unchanged.

1. Add targeted real-database failing tests for fresh character and Buddy creation, expression/runtime resolution, restarts, tombstones, and rollback.
2. Include the approved final raster assets and provenance. Implement create-only character seed and a coordinated Buddy Persona seed through existing startup readiness.
3. Add explicit package-data and artifact checks. Build wheel/sdist, then verify a fresh isolated profile against installed resources.
4. Run focused tests, formatting/lint, import provenance and self-review. Record evidence in task-31758; commit and open a PR to dev.
