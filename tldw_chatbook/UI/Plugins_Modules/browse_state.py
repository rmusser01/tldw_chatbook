"""In-session browsing state; identities never derive from row positions."""

from dataclasses import dataclass


def accept_response(request_generation: int, current_generation: int) -> bool:
    return request_generation == current_generation


@dataclass
class PluginBrowseState:
    view: str = "installed"
    query: str = ""
    page: int = 0
    selected_source_id: str | None = None
    selected_installation_id: str | None = None
    selected_entry_id: str | None = None
    request_generation: int = 0
    workspace_id: str | None = None
    focus_id: str = "plugins-list"
    details_open: bool = False
    entry_action: str | None = None
    location: str = ""
    ref: str = "HEAD"
    import_app: str = "cursor"
    catalog_path: str = "auto"
    preview: object | None = None
    preview_workspace_id: str | None = None
