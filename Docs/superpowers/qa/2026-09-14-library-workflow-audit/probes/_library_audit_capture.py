"""Disposable audit capture; service fixtures, production Library screen/styles."""

import json
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    StaticLibraryRagSearchService,
    _wait_for_query_ready,
)

OUT = (
    Path(__file__).resolve().parents[2]
    / ".superpowers/sdd/2026-09-14-library-ui-audit/captures"
)


def capture(host, name):
    OUT.mkdir(parents=True, exist_ok=True)
    screen = host.screen
    strips = screen._compositor.render_strips()
    text = "\n".join(strip.text for strip in strips)
    (OUT / f"{name}.txt").write_text(text)
    (OUT / f"{name}.svg").write_text(host.export_screenshot())
    widgets = []
    for widget in screen.query("*"):
        if not widget.id:
            continue
        region = widget.region
        painted = (
            "\n".join(
                strips[y]
                .crop(max(0, region.x), min(screen.size.width, region.right))
                .text
                for y in range(max(0, region.y), min(len(strips), region.bottom))
            )
            if region.x < screen.size.width and region.right > 0
            else ""
        )
        widgets.append(
            {
                "id": widget.id,
                "kind": type(widget).__name__,
                "region": tuple(region),
                "content": tuple(widget.content_region),
                "display": widget.display,
                "visible": widget.visible,
                "focus": widget.has_focus,
                "label": str(getattr(widget, "label", "")),
                "painted": painted,
            }
        )
    (OUT / f"{name}.json").write_text(json.dumps(widgets, indent=2))


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize(
    "mode", ["notes", "media", "conversations", "search", "workspaces"]
)
async def test_capture_library(theme, populated, mode):
    app = _build_test_app()
    notes = [
        {
            "id": "note-1",
            "title": "Design review [draft] — café",
            "content": "# Review\n\nThe library must keep source context visible.",
            "last_modified": "2026-09-14T10:00:00Z",
        },
        {
            "id": "note-2",
            "title": "Meeting decisions",
            "content": "Keep the editor focused.",
        },
    ]
    media = [
        {
            "id": 1,
            "title": "Interview transcript — design review",
            "type": "article",
            "content": "Speaker: Source evidence remains available after filtering.\n"
            * 20,
            "author": "Audit fixture",
            "last_modified": "2026-09-14T10:00:00Z",
        }
    ]
    _seed_conversations(
        app,
        _two_conversations() if populated else [],
        notes=notes if populated else [],
        media=media if populated else [],
    )
    if populated:
        app.workspace_registry_service.create_workspace(
            workspace_id="design-review", name="Design review workspace"
        )
        app.workspace_registry_service.set_active_workspace("design-review")
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Design review note",
                    "snippet": "Keep source context visible.",
                    "source_id": "note-1",
                    "source_type": "note",
                    "score": "0.93",
                    "provenance": {
                        "source_type": "note",
                        "runtime_backend": "local-fts",
                    },
                }
            ]
            if populated
            else [],
            "runtime_backend": "local-fts",
        }
    )
    host = LibraryGlobalKeyProductionCSSHarness(app)
    name = f"{mode}-{theme}-{'populated' if populated else 'empty'}"
    async with host.run_test(size=(120, 45)) as pilot:
        host.theme = theme
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()
        if mode == "workspaces":
            screen.query_one(
                "#console-rail-section-toggle-library-details", Button
            ).press()
            await pilot.pause()
            await pilot.pause()
            screen.query_one("#library-workspaces-depth-panel").scroll_visible(
                animate=False
            )
        else:
            screen.query_one(f"#library-row-browse-{mode}", Button).press()
            target = (
                "#library-search-rag-panel"
                if mode == "search"
                else f"#library-{mode}-canvas"
            )
            await _wait_for_selector(screen, pilot, target)
        await pilot.pause()
        await pilot.pause()
        capture(host, name + "-120-browse")
        if mode == "search" and populated:
            query = screen.query_one("#library-rag-query-input", Input)
            query.value = "design review"
            await _wait_for_query_ready(screen, pilot, query.value)
            query.focus()
            await pilot.press("enter")
            if populated:
                await _wait_for_selector(screen, pilot, "#library-rag-result-0")
            await pilot.pause(0.1)
        elif populated and mode in ("notes", "media", "conversations"):
            prefix = {
                "notes": "library-notes-row-",
                "media": "library-media-row-",
                "conversations": "library-conversation-row-",
            }[mode]
            rows = [b for b in screen.query(Button) if b.id and b.id.startswith(prefix)]
            if not rows and mode == "notes":
                rows = [b for b in screen.query(Button) if getattr(b, "note_id", None)]
            if rows:
                rows[0].focus()
                await pilot.press("enter")
                await pilot.pause(0.2)
        await pilot.pause()
        capture(host, name + "-120-active")
        for size in ((80, 24), (120, 45)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            await pilot.pause()
            capture(host, name + f"-{size[0]}-resize")
        if theme == "textual-dark" and populated:
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            seen = set()
            focus = []
            for _ in range(70):
                await pilot.press("tab")
                await pilot.pause()
                widget = host.focused
                if widget is None:
                    focus.append({"id": None})
                    break
                identity = (type(widget).__name__, widget.id)
                if identity in seen:
                    break
                seen.add(identity)
                region = widget.region
                strips = screen._compositor.render_strips()
                painted = (
                    "\n".join(
                        strips[y].crop(max(0, region.x), min(80, region.right)).text
                        for y in range(max(0, region.y), min(24, region.bottom))
                    )
                    if region.x < 80 and region.right > 0
                    else ""
                )
                focus.append(
                    {
                        "id": widget.id,
                        "kind": type(widget).__name__,
                        "region": tuple(region),
                        "label": str(getattr(widget, "label", "")),
                        "painted": painted,
                    }
                )
            (OUT / f"{name}-80-focus-traversal.json").write_text(
                json.dumps(focus, indent=2)
            )
