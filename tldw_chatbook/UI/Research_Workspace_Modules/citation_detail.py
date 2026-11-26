"""Inspect the citations stored with one Research answer."""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .chat_message import context_rows, research_message_context


class ResearchCitationDetail(SafeModalDismissMixin, ModalScreen[None]):
    AUTO_FOCUS = "#research-chat-citations-close"
    SAFE_MODAL_CONTENT = "#research-chat-citations-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]

    def __init__(self, answer) -> None:
        super().__init__()
        self.context = research_message_context(answer)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="research-chat-citations-dialog", classes="research-wide-dialog"):
            yield Static("Citations", classes="dialog-title", markup=False)
            citations = context_rows(self.context.get("citations"))
            sources = {
                str(source.get("catalog_item_id")): source
                for source in context_rows(self.context.get("sources"))
            }
            for citation in citations:
                media_id = citation.get("catalog_item_id") or citation.get("media_id")
                source = citation.get("title") or f"Library media {media_id}"
                marker = (
                    citation.get("marker")
                    or citation.get("citation_id")
                    or citation.get("id")
                )
                version = sources.get(str(media_id), {}).get("catalog_item_version")
                detail = f"{marker} · {source}"
                if version is not None:
                    detail += f" · Revision {version}"
                yield Static(detail, markup=False)
            if not citations:
                yield Static(
                    "No citations were stored with this message.", markup=False
                )
            yield Button("Close", id="research-chat-citations-close")

    @on(Button.Pressed, "#research-chat-citations-close")
    def close(self) -> None:
        self.dismiss_safe_once(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)
