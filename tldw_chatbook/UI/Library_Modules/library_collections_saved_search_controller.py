"""Bounded saved-search window loading under the existing capture authority."""

from ...Library.collections_capture_models import CollectionsCaptureError
from ...Library.collections_capture_service import CollectionsCaptureScopeService
from .library_collections_state import LibraryCollectionsState


async def load_saved_search_page(
    state: LibraryCollectionsState,
    scope: CollectionsCaptureScopeService,
    page: int,
) -> bool:
    """Replace one window after a current request, retaining good rows on failure."""
    state.saved_searches_generation += 1
    generation = state.saved_searches_generation
    authority = scope.active_authority
    key = authority.key if authority else None
    if state.saved_searches_authority != key:
        state.saved_searches = ()
        state.saved_searches_total = 0
        state.saved_searches_page = 1
    state.saved_searches_authority = key
    state.saved_searches_requested_page = page
    state.saved_searches_error = ""
    state.saved_searches_loading = True

    def current() -> bool:
        active = scope.active_authority
        return state.saved_searches_generation == generation and key == (
            active.key if active else None
        )

    try:
        result = await scope.list_saved_searches(page)
        if result.page != page or any(s.authority_key != key for s in result.items):
            raise CollectionsCaptureError("saved_search_page_mismatch")
    except Exception:  # noqa: BLE001 - keep backend failures in the retryable rail
        if current():
            state.saved_searches_error = "Saved searches could not load. Retry below."
        return False
    finally:
        if state.saved_searches_generation == generation:
            state.saved_searches_loading = False
    if not current():
        return False
    state.saved_searches = tuple(result.items)
    state.saved_searches_total = result.total
    state.saved_searches_page = result.page
    return True
