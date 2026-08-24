"""Console prompts controller.

Extracted out of `ChatScreen` (wave-3 console decomposition, task 3): the
Console shell's PROMPT cluster -- the source-aware Prompt Library modal
(`_open_console_prompts_modal`, 397 lines at the time of the move and the
largest non-`__init__` method the pre-move class had; 82 since task 2766
split its 17 nested closures into the two collaborators below), `/prompt`
and `/system` name resolution
and their shared refuse-a-Recipe guard, both prompt-picker launches, the
system-prompt editor and its save-to-Library flow, the Library "Use in
Console" staged-insert handoff, and the shared prompt-history store.

This module follows the SAME binding rule waves 1-2 established (see
`dictation.py`'s `ConsoleDictationController.__init__` docstring for the
canonical statement, and `message.py` for the closest analogue in size and
fan-in; restated briefly here):

1. **Framework services** (`push_screen`) live-read from the
   screen via `@property` on every access -- never snapshotted.
2. **Everything else this cluster depends on that is not its own state** is
   a NAMED keyword-only constructor callable, matching the design spec's
   rule that "a controller's dependencies are its signature". Each is a
   callable the CALLER (`ChatScreen.__init__`) constructs as a late-binding
   lambda closing over `self` -- never a bound method passed directly. That
   matters concretely here: `Tests/UI/test_console_workbench_contract.py`
   replaces `_open_console_provider_recovery` and `_console_provider_
   blocker_copy` on the SCREEN INSTANCE and then calls the modal opener,
   and twelve of the seventeen dependencies below are monkeypatched by name
   somewhere in the pre-existing suite. A constructor snapshot would
   silently stop observing every one of those.
3. `app_instance` is a plain snapshot, for the same reason `dictation.py`
   and `message.py` snapshot it: every read in the moved bodies is a
   bare-attribute or `notify()` call, never one that could observe a later
   `screen.app_instance` reassignment.

**Zero DOM.** No `query_one`/`query` traffic reaches through `screen` here.
That is why `_insert_prompt_text_into_composer` -- which does a raw
`self.query_one("#console-native-composer", ConsoleComposerBar)` -- did NOT
move and is instead one of the seventeen injected callables, even though it
is the single most prompt-shaped helper on the screen. (Six pre-existing
test sites also replace it on the screen by name, so it has to keep
resolving through the screen either way.)

**Controller-to-controller seam** (session): three moved bodies called
`self._session.X(...)` in the pre-move source -- `_ensure_active_console_
session_settings`, `_apply_console_session_system_prompt`, `_sync_console_
session_draft`. Each drops that prefix in favour of a same-named property
here, wrapping a constructor-injected callable `ChatScreen.__init__` points
at `self._session.X`, exactly as `message.py` documents for its own session
seam. Python resolves those lambdas at CALL time, so construction order
between controllers never matters. There is no reverse seam: nothing on
`ConsoleSessionController` reaches back into this cluster.

**Delegation table.** `ChatScreen` keeps a thin (<=4-line) delegation under
the ORIGINAL private name for every moved method that is still reached from
outside the cluster -- either by a staying screen method, by Textual's own
`action_*` resolution, by the `/prompt`+`/system` command-registry dict, or
by a pre-existing test that monkeypatches or calls that exact name:

- `_open_console_prompts_modal` -- `_handle_console_composer_menu_choice`'s
  `ACTION_PROMPTS` branch, and `test_console_composer_menu.py` replaces the
  screen attribute wholesale.
- `_ensure_console_prompt_history` -- `_ensure_console_chat_controller` and
  the composer's `set_prompt_history` mount wiring, plus
  `test_console_composer_history.py`.
- `_console_command_insert_prompt` / `_console_command_apply_system` -- the
  `/prompt` and `/system` rows of the command-handler dict, which
  `test_console_command_composer.py` replaces per-instance.
- `_open_console_system_prompt_editor` -- `action_open_console_system_
  prompt_editor` (stays: Textual resolves `action_*` by name on the
  Screen), the setup-modal recovery branch, and six sites in
  `test_console_system_prompt.py`.
- `_consume_pending_console_prompt_insert` -- two `set_timer` schedules
  (`on_mount`, `on_screen_resume`) plus five test sites.

The remaining moved methods (`_is_recipe_prompt_record`, `_console_prompt_
prefix_fts_query`, `_console_prompt_search`, `_resolve_console_prompt_by_
name`, `_open_console_prompt_picker_for_insert`, `_open_console_prompt_
picker_for_apply_system`, `_save_console_system_prompt_to_library`) have no
staying caller other than `action_open_console_prompt_insert`, which got a
direct `self._prompts.X(...)` call-site edit instead of a delegation.

**Stays on `ChatScreen`, despite matching the `*prompt*` name pattern**:
- `action_open_console_prompt_insert` / `action_open_console_system_prompt_
  editor` -- Textual resolves `action_*` by name on the Screen.
- `_console_system_prompt_chip_activated` -- an `@on`-decorated message
  handler; Textual's message dispatch requires the decorator on the class
  that receives the message.
- `_insert_prompt_text_into_composer` -- real `query_one` DOM access (see
  "Zero DOM" above).
- `_console_prompted_source_count` -- a name-pattern false positive: it
  counts RAG evidence entries "put in front of the model", i.e. it belongs
  to the live-work/citation cluster, and touches no prompt record at all.
- `_insert_console_caption_prompt` -- appends one canned caption string to
  the composer draft; composer cluster, not the Prompt Library.
- `_console_rewind_prompt_rows` -- builds `RewindPromptRow`s from a
  session's USER turns for the `/rewind` menu; rewind cluster.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional, TYPE_CHECKING
import asyncio
import uuid

from loguru import logger

from ..Navigation.pending_handoff_store import HandoffChannel
from ..Navigation.screen_state_store import ConsolePromptTargetProjection
from ...Chat.console_command_grammar import CommandParse
from ...Utils.fts5_match_forms import quote_fts5_prefix
from ...Chat.console_provider_endpoints import safe_endpoint_display
from ...Chat.prompt_history import PromptHistory, default_prompt_history_path
from ...Library.library_prompts_state import classify_prompt_save_error
from ...Prompt_Management.prompt_artifact_codec import decode_prompt_artifact
from ...Prompt_Management.prompt_improvement_models import (
    PromptImprovementRequestSnapshot,
    fingerprint_block_definition,
    fingerprint_text,
)
from ...Prompt_Management.prompt_improvement_service import PromptImprovementService
from ...Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    compile_prompt_variables,
    fingerprint_system_text,
)
from ...Widgets.Console.console_composer_bar import (
    ComposerDraftSnapshot,
    ComposerTransactionValidationError,
)
from ...Widgets.Console.console_prompt_improve_view import (
    ConsolePromptImprovementContext,
)
from ...Widgets.Console.console_prompt_picker_modal import (
    MODE_APPLY_SYSTEM as CONSOLE_PROMPT_PICKER_MODE_APPLY_SYSTEM,
    ConsolePromptPickerModal,
)
from ...Widgets.Console.console_prompts_modal import (
    ConsolePromptsApplyOutcome,
    ConsolePromptsModal,
    ConsolePromptsResult,
    ConsoleRecipeApplyGuard,
    ConsoleSavedPromptApplyGuard,
)
from ...Widgets.Console.console_system_prompt_modal import ConsoleSystemPromptModal
from ...Widgets.Console.prompt_variables_dialog import (
    PromptVariablesDialog,
    PromptVariablesDialogRequest,
)

#: One browse page of the Console prompt picker. Behaviour-defining: the
#: modal's paging controls, its "no more results" state, and the Library's
#: own page size are all in terms of this number.
CONSOLE_PROMPT_PAGE_SIZE = 10

#: Row cap for a prompt SEARCH, as distinct from a browse page. The modal
#: renders a single unpaged result list for a search, so this bounds what a
#: user can be shown at once rather than what one page holds. The controller
#: aliases this as `_CONSOLE_PROMPT_SEARCH_LIMIT` for `/prompt` resolution --
#: one number, two readers.
CONSOLE_PROMPT_SEARCH_LIMIT = 25

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")


@dataclass(frozen=True, slots=True)
class _ConsolePromptReplaceTarget:
    """Exact Console mutation target captured before asynchronous selection."""

    composer_snapshot: ComposerDraftSnapshot = field(repr=False)
    session_id: str
    system_fingerprint: str = field(repr=False)


# -- Module-level copy this cluster owns exclusively -- moved verbatim from
# `chat_screen.py`, which no longer references either name for itself.

#: Console's `/system` editor only ever performs a Library prompt CREATE
#: (never an update to an existing one, unlike the Library prompt editor's
#: own save flow), but the outcome copy the user sees must read identically
#: either way -- duplicated here rather than imported across screens to
#: avoid a Screen-to-Screen import.
CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY = {
    "ok": "Saved.",
    "name-in-use": "Name already in use — pick another or open the existing prompt.",
    "soft-deleted-name": "A deleted prompt holds this name — restore it or choose another.",
    "error": "Couldn't save this prompt. Try again.",
}
CONSOLE_SYSTEM_PROMPT_NO_SYSTEM_PART_TEMPLATE = 'Prompt "{name}" has no system part.'


def _provider_resolution_identity(resolution: Any) -> tuple[str, str, str, str, str]:
    """The five fields that decide whether two resolutions are one target.

    Equal identities mean the user is still spending against the target the
    modal disclosed; any difference means they were shown one thing and would
    get another, which every guard in the improvement flow treats as stale.

    Args:
        resolution: A `ConsoleProviderResolution`, or anything shaped like
            one -- the guards compare whatever the gateway handed back.

    Returns:
        tuple[str, str, str, str, str]: provider, model, base URL, readiness
        key and execution key, each coerced to `str` (missing fields empty).
    """
    return (
        str(getattr(resolution, "provider", "")),
        str(getattr(resolution, "model", "")),
        str(getattr(resolution, "base_url", "")),
        str(getattr(resolution, "readiness_key", "")),
        str(getattr(resolution, "execution_key", "")),
    )


class _ConsolePromptSource:
    """The Prompt Library modal's data adapter over one prompt scope service.

    Five of `_open_console_prompts_modal`'s closures captured exactly one
    thing -- `app_instance.prompt_scope_service` -- and did exactly one thing
    with it: look up a method, refuse in user-visible copy when the source
    cannot serve, and forward the call with this cluster's own contract
    (browse pages of 10, searches capped at 25, `save`'s `source` routed into
    the service's `mode`). That is a collaborator with one field, so it is
    one here rather than five closures over a 397-line scope (task 2766).

    One instance per modal open; its bound methods are what
    `ConsolePromptsModal` receives, and `detail`/`record_usage` are also what
    the improvement flow's saved-artifact guards run against.

    The service is duck-typed deliberately: `getattr` + `callable` per call,
    so an absent or partially-implemented source degrades into the refusal
    copy the modal has always rendered instead of an `AttributeError`.
    """

    def __init__(self, service: Any) -> None:
        """Bind the scope service this modal open will read.

        Args:
            service: `app_instance.prompt_scope_service`, or `None` when the
                app exposes none -- every method below then refuses.
        """
        self._service = service

    def _require(self, attribute: str, unavailable_copy: str) -> Any:
        """Return a callable service method, or refuse in user-visible copy.

        Args:
            attribute: The scope-service method name to resolve.
            unavailable_copy: What the user is told when it is missing.

        Returns:
            Any: The bound service method.

        Raises:
            ValueError: When the service has no callable under that name.
        """
        method = getattr(self._service, attribute, None)
        if not callable(method):
            raise ValueError(unavailable_copy)
        return method

    async def capabilities(self, source: str) -> Any:
        """Report what `source` can store, so the editor can gate its kinds."""
        method = self._require(
            "get_capabilities", f"{source.title()} Prompt source is unavailable."
        )
        return await method(mode=source)

    async def list_page(self, source: str, page: int) -> Any:
        """Return one browse page of `source` at `CONSOLE_PROMPT_PAGE_SIZE`."""
        method = self._require(
            "list_prompts", f"{source.title()} Prompt source is unavailable."
        )
        return await method(mode=source, page=page, per_page=CONSOLE_PROMPT_PAGE_SIZE)

    async def search(self, source: str, query: str) -> Any:
        """Search `source`, bounded to `CONSOLE_PROMPT_SEARCH_LIMIT`."""
        method = self._require(
            "search_prompts", f"{source.title()} Prompt search is unavailable."
        )
        return await method(mode=source, query=query, limit=CONSOLE_PROMPT_SEARCH_LIMIT)

    async def detail(self, source: str, identifier: str) -> Any:
        """Fetch one record -- also the freshness probe the apply guards use."""
        method = self._require(
            "get_prompt", f"{source.title()} Prompt source is unavailable."
        )
        return await method(mode=source, prompt_identifier=identifier)

    async def save(self, **payload: Any) -> Any:
        """Persist the editor's working copy to the payload's own source."""
        method = self._require("save_prompt", "The selected Prompt source cannot save.")
        source = str(payload.pop("source", "local"))
        return await method(mode=source, **payload)

    async def record_usage(self, source: str, identifier: str) -> None:
        """Record one Library usage, skipping a source that cannot track it.

        A source with no recorder is not a failure -- it is a source that
        does not count usage -- so this returns quietly. A recorder that
        raises propagates: the caller owns what the user is told about it.

        Args:
            source: The scope the applied record came from.
            identifier: That record's source-scoped identity.
        """
        recorder = getattr(self._service, "record_prompt_usage", None)
        if not callable(recorder):
            return
        await recorder(mode=source, prompt_identifier=identifier)


class _ConsolePromptImprovementFlow:
    """The Prompt Library modal's improvement sub-flow, scoped to ONE open.

    Eleven of `_open_console_prompts_modal`'s closures were a single
    stateful conversation: pin and disclose a provider target, build a
    request snapshot against it, validate a reviewed result against the
    artifact it came from, apply it to the live session, and retry a
    System-prompt persist that failed. They shared eight captured values
    plus one `nonlocal` -- the pinned resolution -- that nothing else in that
    scope touched. That is an object with state, so it is one here
    (task 2766); the `nonlocal` becomes `self._pinned_resolution`.

    One instance per modal open, built by
    `ConsolePromptsController._open_console_prompts_modal`; six of its bound
    methods are what `ConsolePromptsModal` receives. Everything it needs from
    outside this cluster arrives as a named constructor argument and stays a
    CALLABLE, following the rule the controller itself follows (see the
    module docstring), so a screen-level replacement made after the modal
    opened is still observed.
    """

    def __init__(
        self,
        *,
        source: _ConsolePromptSource,
        store: Any,
        session_id: Any,
        composer: Any,
        composer_snapshot: Any,
        current_system: str,
        current_system_fingerprint: str,
        gateway: Any,
        improvement_context: Any,
        app_instance: Any,
        active_session_settings: Callable[[], Any],
        build_provider_selection: Callable[[], Any],
        sync_system_prompt_surfaces: Callable[[], None],
    ) -> None:
        """Capture the facts this open was disclosed against.

        Args:
            source: The open's `_ConsolePromptSource`; the saved-artifact
                guards re-fetch through it and usage is recorded through it.
            store: The Console chat store, live -- `active_session_id` is
                re-read on every guard, never trusted from open time.
            session_id: The session the modal was opened over.
            composer: The Console composer widget the draft belongs to.
            composer_snapshot: That draft, captured at open, and the thing
                every projection and validation runs against.
            current_system: The System prompt as disclosed at open.
            current_system_fingerprint: Its fingerprint, the guard the live
                System prompt is re-checked against.
            gateway: The Console provider gateway that resolves targets.
            improvement_context: The opening disclosure the activation
                returns a `replace`d copy of.
            app_instance: Held (not snapshotted into a bound `notify`) so a
                later `app.notify` replacement is still observed.
            active_session_settings: Returns the LIVE session settings --
                `ConsoleSessionController._ensure_active_console_session_
                settings`, through the controller's own named dependency.
            build_provider_selection: Rebuilds the provider selection per
                resolve, so staleness compares against a LIVE selection.
            sync_system_prompt_surfaces: Re-syncs the three screen surfaces
                that display the System prompt, after it is persisted.
        """
        self._source = source
        self._store = store
        self._session_id = session_id
        self._composer = composer
        self._composer_snapshot = composer_snapshot
        self._current_system = current_system
        self._current_system_fingerprint = current_system_fingerprint
        self._gateway = gateway
        self._improvement_context = improvement_context
        self._app_instance = app_instance
        self._active_session_settings = active_session_settings
        self._build_provider_selection = build_provider_selection
        self._sync_system_prompt_surfaces = sync_system_prompt_surfaces
        #: The disclosed target, pinned by `activate_improvement_context` (or
        #: by a model-free `capture_manual_resolution`) and compared against a
        #: freshly resolved one before anything is sent or applied.
        self._pinned_resolution: Any | None = None

    # -- Staleness -----------------------------------------------------------

    def _active_system_fingerprint(self) -> str:
        """Fingerprint the LIVE System prompt, re-read on every guard."""
        live_settings = self._active_session_settings()
        return fingerprint_text(str(live_settings.system_prompt or ""))

    def _stale_reason(self) -> str:
        """Why this open can no longer act, or `""` while it still can.

        Both facts were fixed when the modal opened -- the session it was
        opened over and the System prompt it disclosed. Either moving means
        the user is deciding about a state that no longer exists.

        Returns:
            str: User-visible copy naming what moved, or `""`.
        """
        if self._store.active_session_id != self._session_id:
            return "The active Console session changed."
        if self._active_system_fingerprint() != self._current_system_fingerprint:
            return "The Console System prompt changed."
        return ""

    # -- Pinning the disclosed target ---------------------------------------

    async def activate_improvement_context(self) -> Any:
        """Pin and disclose the exact target before any model path can run."""

        stale = self._stale_reason()
        if stale:
            raise ValueError(stale)
        projection = None
        projection_blocker = ""
        try:
            projection = self._composer.project_snapshot_for_model(
                self._composer_snapshot,
                request_nonce=f"prompt-preview-{uuid.uuid4().hex}",
            )
        except ValueError:
            projection_blocker = (
                "Model improvement is unavailable because the draft contains "
                "reserved protected-placeholder text. Remove or rename that "
                "literal token, then reopen Improve."
            )
        try:
            resolution = await self._gateway.resolve_for_send(
                self._build_provider_selection()
            )
        except Exception:
            self._pinned_resolution = None
            return replace(
                self._improvement_context,
                current_user_projection=projection,
                endpoint_label="Unavailable",
                model_unavailable_reason=(
                    projection_blocker
                    or "Prompt improvement could not resolve the current provider "
                    "target. Review Console provider settings and reopen Improve."
                ),
            )
        self._pinned_resolution = resolution
        blocker = projection_blocker
        if not blocker and (
            not resolution.ready or not str(resolution.model or "").strip()
        ):
            blocker = str(
                resolution.visible_copy
                or "Choose a ready provider and model, then reopen Improve."
            )
        return replace(
            self._improvement_context,
            current_user_projection=projection,
            provider_label=str(resolution.provider or "Not configured"),
            model_label=str(resolution.model or "Not configured"),
            endpoint_label=(
                safe_endpoint_display(resolution.base_url) or "Provider default"
            ),
            model_unavailable_reason=blocker,
            pinned_resolution=resolution,
        )

    async def capture_manual_resolution(self) -> Any:
        """Capture the effective target once for a later model-free Apply."""

        if self._pinned_resolution is None:
            self._pinned_resolution = await self._gateway.resolve_for_send(
                self._build_provider_selection()
            )
        return self._pinned_resolution

    async def build_improvement_snapshot(self, **values: Any) -> Any:
        """Build the request snapshot, against the pinned target only."""
        stale = self._stale_reason()
        if stale:
            raise ValueError(stale)
        request_id = str(values["request_id"])
        projection = self._composer.project_snapshot_for_model(
            self._composer_snapshot,
            request_nonce=request_id,
        )
        self._composer.validate_improvement(self._composer_snapshot, projection.text)
        pinned_resolution = self._pinned_resolution
        if pinned_resolution is None:
            raise ValueError(
                "The provider target is no longer pinned. Reopen Improve to refresh disclosure."
            )
        live_resolution = await self._gateway.resolve_for_send(
            self._build_provider_selection()
        )
        if _provider_resolution_identity(
            live_resolution
        ) != _provider_resolution_identity(pinned_resolution):
            raise ValueError(
                "The provider, model, or endpoint changed. Reopen Improve to refresh disclosure."
            )
        if (
            not pinned_resolution.ready
            or not str(pinned_resolution.model or "").strip()
        ):
            raise ValueError(
                pinned_resolution.visible_copy or "Provider is unavailable."
            )
        include_system = bool(values.get("include_system"))
        recipe_definition = values.get("recipe_definition")
        return PromptImprovementRequestSnapshot(
            request_id=request_id,
            mode=values["mode"],
            session_id=self._session_id,
            composer_snapshot=self._composer_snapshot,
            projection=projection,
            system_prompt=self._current_system if include_system else None,
            system_fingerprint=(
                self._current_system_fingerprint if include_system else None
            ),
            resolution=pinned_resolution,
            provider_label=pinned_resolution.provider,
            model_label=str(pinned_resolution.model),
            recipe_source=values.get("recipe_source"),
            recipe_source_id=values.get("recipe_source_id"),
            recipe_version=values.get("recipe_version"),
            recipe_definition=recipe_definition,
            recipe_fingerprint=(
                fingerprint_block_definition(recipe_definition)
                if recipe_definition is not None
                else None
            ),
        )

    # -- Validating what came back ------------------------------------------

    def validate_improvement(self, captured: Any, text: str) -> None:
        """Validate reviewed text against the snapshot the request captured."""
        snapshot = getattr(captured, "composer_snapshot", self._composer_snapshot)
        self._composer.validate_improvement(snapshot, text)

    async def _validate_saved_recipe(self, captured: Any) -> None:
        """Refuse to apply a Recipe that moved since it was selected.

        Raises:
            ValueError: Naming what drifted -- source, identity, version,
                compatibility or content.
        """
        recipe_source_id = str(getattr(captured, "recipe_source_id", "") or "")
        if not recipe_source_id or recipe_source_id.startswith("builtin:"):
            return
        source = getattr(captured, "recipe_source", None)
        if source not in {"local", "server"}:
            raise ValueError("The selected Recipe source changed.")
        latest = await self._source.detail(source, recipe_source_id)
        if not isinstance(latest, Mapping):
            raise ValueError("The selected Recipe is no longer available.")
        latest_identity = (
            latest.get("source_id") or latest.get("id") or latest.get("uuid")
        )
        if str(latest_identity or "") != recipe_source_id:
            raise ValueError("The selected Recipe identity changed.")
        latest_version = latest.get("version", latest.get("optimistic_version"))
        if latest_version != getattr(captured, "recipe_version", None):
            raise ValueError("The selected Recipe version changed.")
        decoded = decode_prompt_artifact(latest)
        if decoded.artifact_type != "recipe" or decoded.definition is None:
            raise ValueError("The selected Recipe is no longer compatible.")
        if fingerprint_block_definition(decoded.definition) != getattr(
            captured, "recipe_fingerprint", None
        ):
            raise ValueError("The selected Recipe changed.")

    async def _validate_saved_prompt(self, captured: Any) -> None:
        """Refuse to apply a saved Prompt that moved since it was selected.

        Raises:
            ValueError: Naming what drifted -- availability, identity,
                version, compatibility or content.
        """
        if not isinstance(captured, ConsoleSavedPromptApplyGuard):
            return
        latest = await self._source.detail(captured.source, captured.prompt_source_id)
        if not isinstance(latest, Mapping):
            raise ValueError("The selected Prompt is no longer available.")
        latest_identity = (
            latest.get("source_id") or latest.get("id") or latest.get("uuid")
        )
        if str(latest_identity or "") != captured.prompt_source_id:
            raise ValueError("The selected Prompt identity changed.")
        latest_version = latest.get("version", latest.get("optimistic_version"))
        if latest_version != captured.prompt_version:
            raise ValueError("The selected Prompt version changed.")
        decoded = decode_prompt_artifact(latest)
        if decoded.artifact_type != "prompt" or decoded.definition is None:
            raise ValueError("The selected Prompt is no longer compatible.")
        if (
            fingerprint_block_definition(decoded.definition)
            != captured.prompt_fingerprint
        ):
            raise ValueError("The selected Prompt changed.")

    async def _record_applied_usage(self, captured: Any) -> None:
        """Record Library usage for an applied saved Prompt, never fatally.

        The apply already landed by the time this runs, so a recorder failure
        is a warning about bookkeeping -- never a rollback.
        """
        if (
            not isinstance(captured, ConsoleSavedPromptApplyGuard)
            or not captured.record_usage
        ):
            return
        try:
            await self._source.record_usage(
                captured.source,
                captured.prompt_source_id,
            )
        except Exception:
            self._app_instance.notify(
                "The prompt was applied, but Library usage could not be recorded.",
                severity="warning",
            )

    # -- Applying it --------------------------------------------------------

    async def apply_improvement_result(
        self, result: ConsolePromptsResult, captured: Any
    ) -> ConsolePromptsApplyOutcome:
        """Apply a reviewed result to the live session, or refuse as stale."""
        stale = self._stale_reason()
        if stale:
            return ConsolePromptsApplyOutcome("stale", stale)
        if isinstance(captured, PromptImprovementRequestSnapshot):
            live_resolution = await self._gateway.resolve_for_send(
                self._build_provider_selection()
            )
            if _provider_resolution_identity(
                live_resolution
            ) != _provider_resolution_identity(captured.resolution):
                return ConsolePromptsApplyOutcome(
                    "stale", "The provider, model, or endpoint changed."
                )
        elif isinstance(
            captured, (ConsoleRecipeApplyGuard, ConsoleSavedPromptApplyGuard)
        ):
            captured_resolution = captured.provider_resolution
            if captured_resolution is None:
                return ConsolePromptsApplyOutcome(
                    "stale",
                    "The provider target was not captured. Reopen the Prompt and retry.",
                )
            live_resolution = await self._gateway.resolve_for_send(
                self._build_provider_selection()
            )
            if _provider_resolution_identity(
                live_resolution
            ) != _provider_resolution_identity(captured_resolution):
                return ConsolePromptsApplyOutcome(
                    "stale", "The provider, model, or endpoint changed."
                )
        try:
            await self._validate_saved_recipe(captured)
            await self._validate_saved_prompt(captured)
            if result.apply_user:
                if result.user_text is None:
                    raise ValueError("The reviewed User prompt is missing.")
                self._composer.validate_improvement(
                    result.composer_snapshot, result.user_text
                )
            elif self._composer.capture_draft_snapshot() != result.composer_snapshot:
                raise ValueError("The Console draft changed.")
        except Exception:
            return ConsolePromptsApplyOutcome(
                "stale",
                "The draft or Recipe changed. Review the working copy and retry.",
            )

        if result.apply_user and result.user_text is not None:
            self._composer.apply_improvement(
                result.composer_snapshot,
                result.user_text,
            )
            self._store.set_session_draft(self._session_id, self._composer.draft_text())
        persisted = True
        if result.apply_system:
            _session, persisted = self._store.set_session_system_prompt(
                self._session_id, result.system_text
            )
            self._sync_system_prompt_surfaces()
        await self._record_applied_usage(captured)
        if not persisted:
            return ConsolePromptsApplyOutcome("persistence_failed")
        return ConsolePromptsApplyOutcome("applied")

    async def retry_improvement_persistence(
        self, result: ConsolePromptsResult
    ) -> ConsolePromptsApplyOutcome:
        """Re-attempt a System-prompt persist that failed, and only that."""
        if self._store.active_session_id != self._session_id or not result.apply_system:
            return ConsolePromptsApplyOutcome(
                "stale", "The active Console session changed."
            )
        live_settings = self._active_session_settings()
        if str(live_settings.system_prompt or "") != str(result.system_text or ""):
            return ConsolePromptsApplyOutcome(
                "stale", "The live System prompt changed."
            )
        _session, persisted = self._store.set_session_system_prompt(
            self._session_id, result.system_text
        )
        self._sync_system_prompt_surfaces()
        return ConsolePromptsApplyOutcome(
            "applied" if persisted else "persistence_failed"
        )


class ConsolePromptsController:
    """Owns the Console shell's prompt cluster: the Prompt Library modal,
    `/prompt` + `/system` resolution and their pickers, the system-prompt
    editor and its save-to-Library flow, the Library staged-insert handoff,
    and the shared prompt-history store.

    `ChatScreen` constructs exactly one of these, in `__init__`, and keeps a
    `self._prompts` reference plus the delegation table described in the
    module docstring.
    """

    # Bounded prompt-search page size for `/prompt` resolution and the
    # picker's own search callable -- mirrors Task 11's picker contract
    # (PromptScopeService.search_prompts, FTS-ranked, <= 25 rows). Aliases
    # the module-level constant rather than repeating the number: the two
    # were briefly separate copies, which is one tuning edit away from the
    # picker and `/prompt` disagreeing about how many rows a search returns.
    _CONSOLE_PROMPT_SEARCH_LIMIT = CONSOLE_PROMPT_SEARCH_LIMIT

    _LIBRARY_PROMPT_INSERT_BLOCKED_COPY = "Finish provider setup to insert prompts."
    _RECIPE_EXECUTION_BLOCKED_COPY = (
        "Recipes cannot be applied directly. Open Prompts and edit the Recipe "
        "as an unsaved Prompt copy first."
    )
    _PROMPT_APPLICATION_EXPIRED_COPY = (
        "This Prompt insertion expired. Open the Prompt and retry."
    )
    _PROMPT_APPLICATION_STALE_COPY = (
        "The Console draft, session, or System prompt changed. "
        "Open the Prompt and retry."
    )
    _PROMPT_APPEND_STALE_COPY = (
        "The Console session or System prompt changed. Open the Prompt and retry."
    )
    _PROMPT_APPEND_EMPTY_COPY = "This Prompt has no text to append."
    _PROMPT_APPLICATION_FAILED_COPY = (
        "The Prompt could not be applied. The Console draft was restored."
    )
    _PROMPT_SYSTEM_PERSISTENCE_FAILED_COPY = (
        "System prompt applied for this session, but the change could not be "
        "saved -- it may not survive a reload."
    )
    _PROMPT_DISPLAY_SYNC_FAILED_COPY = (
        "Prompt applied, but some Console displays could not be refreshed."
    )

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        composer_accessor: Callable[[], Any],
        chat_store_accessor: Callable[[], Any],
        ensure_active_console_session_settings: Callable[[], Any],
        apply_console_session_system_prompt: Callable[[str], None],
        sync_console_session_draft: Callable[[], None],
        active_console_provider_model_display: Callable[[], tuple],
        build_console_provider_selection: Callable[[], Any],
        ensure_console_provider_gateway: Callable[[], Any],
        console_provider_blocker_copy: Callable[[], str],
        open_console_provider_recovery_accessor: Callable[[], Any],
        console_setup_blocked_reason: Callable[[], str],
        focus_console_composer_if_needed: Callable[..., None],
        insert_prompt_text_into_composer: Callable[..., bool],
        clear_console_composer_draft: Callable[[], None],
        append_native_console_system_message: Callable[[str], Any],
        sync_console_system_prompt_surfaces: Callable[[], None],
        sync_console_command_popup: Callable[[], None],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the method bodies below is a byte-for-byte copy of the
        pre-extraction `ChatScreen` method, EXCEPT the three documented
        `self._session.X(...)` seam call sites (dropping that prefix for a
        same-named property here, per the module docstring's
        "Controller-to-controller seam" section), every
        `self.app.push_screen(...)` call site (dropping the `.app.` for the
        `push_screen` framework property below, the same transformation
        `session.py` and `message.py`'s own moved bodies already made), and
        `_open_console_prompts_modal`, whose closures task 2766 moved into
        `_ConsolePromptSource` and `_ConsolePromptImprovementFlow`.

        Seventeen named dependencies is squarely in the band waves 1-3 have
        established (dictation: 12; message: 19). Thirteen of the seventeen
        are needed by `_open_console_prompts_modal` alone -- it is the
        cluster's whole fan-in, not a sprawl across many methods. The count
        was eighteen until task 2766 collapsed the post-apply re-sync trio
        (see `sync_console_system_prompt_surfaces` below).

        Args:
            screen: The Console screen. Used ONLY for the framework
                services (`push_screen`) below. Zero
                `query_one`/`query` traffic reaches through `screen` here --
                see the module docstring's "Zero DOM" section.
            app_instance: Snapshotted once, not re-read through `screen` --
                every reference in the moved bodies is a bare-attribute
                read (`prompt_scope_service`, `pending_handoffs`,
                `console_prompt_history_factory`) or a `notify()` call,
                never one that could observe a later reassignment.
            composer_accessor: `ChatScreen._console_composer_or_none`, the
                general screen helper (33 call sites screen-wide, not
                prompt-specific), replaced by name at 9 test sites.
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`,
                the general store accessor already shared by every other
                controller (see `session.py`'s identical parameter).
            ensure_active_console_session_settings: `ConsoleSession
                Controller._ensure_active_console_session_settings` --
                session seam, read by the modal opener (twice, inside
                closures that must see LIVE settings) and the system-prompt
                editor.
            apply_console_session_system_prompt: `ConsoleSessionController.
                _apply_console_session_system_prompt` -- session seam, the
                one write path both `/system` and the apply-system picker
                use.
            sync_console_session_draft: `ConsoleSessionController._sync_
                console_session_draft` -- session seam; the staged-insert
                handoff calls it to settle `_console_visible_draft_session_
                id` immediately before inserting, so a racing session
                switch cannot clobber the insert.
            active_console_provider_model_display: `ChatScreen._active_
                console_provider_model_display`, a general screen helper
                (10 call sites across clusters), used for the modal's
                opening disclosure labels.
            build_console_provider_selection: `ChatScreen._build_console_
                provider_selection` -- rebuilt (not captured) at every
                resolve so the modal's staleness guards compare against a
                LIVE selection; replaced by name at 3 test sites.
            ensure_console_provider_gateway: `ChatScreen._ensure_console_
                provider_gateway`, the shared provider gateway the
                improvement service runs against.
            console_provider_blocker_copy: `ChatScreen._console_provider_
                blocker_copy`, the "why Improve is unavailable" copy; a
                CALL, evaluated at modal-build time.
            open_console_provider_recovery_accessor: The one
                bare-attribute-read shape here. `ChatScreen._open_console_
                provider_recovery` is passed to the modal as a CALLABLE
                (`configure_provider=...`), not called, so this accessor
                returns the screen's current attribute rather than wrapping
                it -- which is what keeps
                `test_console_workbench_contract.py`'s
                `console._open_console_provider_recovery = AsyncMock()`
                landing on the modal, exactly as it did pre-move.
            console_setup_blocked_reason: `ChatScreen._console_setup_
                blocked_reason` -- the first-run readiness gate the staged
                Library insert honours.
            focus_console_composer_if_needed: `ChatScreen._focus_console_
                composer_if_needed` (DOM: focuses a queried widget) --
                stays screen-owned, called on every modal/picker dismissal.
            insert_prompt_text_into_composer: `ChatScreen._insert_prompt_
                text_into_composer` -- raw `query_one` DOM access, so it
                stays on the screen (module docstring, "Zero DOM"); also
                replaced by name at 6 pre-existing test sites.
            clear_console_composer_draft: `ChatScreen._clear_console_
                composer_draft` (DOM via the composer accessor) -- `/system`
                clears its own invocation text after a successful apply.
            append_native_console_system_message: `ChatScreen._append_
                native_console_system_message` -- the inline transcript
                error channel both command handlers use to refuse a Recipe.
            sync_console_system_prompt_surfaces: The post-apply re-sync
                bridge. Three separate screen methods (`_sync_console_chat_
                core_state`, `_sync_console_rail_system_line`, `_sync_
                console_settings_summary`, all DOM-adjacent and all staying
                screen-owned) sit behind ONE dependency because this cluster
                only ever needs the trio, in that order, at the two moments
                the store accepted a new System prompt -- never any of them
                individually. Task 2766 collapsed them; `wiring.py` still
                reaches each screen method by name at CALL time, so the
                late-binding property every one of them had is unchanged.
            sync_console_command_popup: The existing screen-owned command
                suggestion refresh, called after a draft replacement or
                rollback so programmatic mutations cannot leave stale rows.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._composer_accessor = composer_accessor
        self._chat_store_accessor = chat_store_accessor
        self._ensure_active_console_session_settings_fn = (
            ensure_active_console_session_settings
        )
        self._apply_console_session_system_prompt_fn = (
            apply_console_session_system_prompt
        )
        self._sync_console_session_draft_fn = sync_console_session_draft
        self._active_console_provider_model_display_fn = (
            active_console_provider_model_display
        )
        self._build_console_provider_selection_fn = build_console_provider_selection
        self._ensure_console_provider_gateway_fn = ensure_console_provider_gateway
        self._console_provider_blocker_copy_fn = console_provider_blocker_copy
        self._open_console_provider_recovery_accessor = (
            open_console_provider_recovery_accessor
        )
        self._console_setup_blocked_reason_fn = console_setup_blocked_reason
        self._focus_console_composer_if_needed_fn = focus_console_composer_if_needed
        self._insert_prompt_text_into_composer_fn = insert_prompt_text_into_composer
        self._clear_console_composer_draft_fn = clear_console_composer_draft
        self._append_native_console_system_message_fn = (
            append_native_console_system_message
        )
        self._sync_console_system_prompt_surfaces_fn = (
            sync_console_system_prompt_surfaces
        )
        self._sync_console_command_popup_fn = sync_console_command_popup

        # This cluster's own state, moved verbatim from `ChatScreen.__init__`.
        # Nothing outside this cluster ever read the attribute directly (only
        # `_ensure_console_prompt_history()`), so `ChatScreen` keeps no proxy
        # property for it -- unlike the message/dictation clusters.
        self._console_prompt_history: Any | None = None

    # -- Framework services (live-read via `@property`) --------------------

    @property
    def push_screen(self) -> Any:
        """`Screen.app.push_screen`, live-read rather than snapshotted.

        See `__init__`'s docstring for why a framework service is reached
        through a property on every access instead of captured once.

        Returns:
            Any: The screen's app-level `push_screen`, bound.
        """
        return self._screen.app.push_screen

    # -- Named constructor dependencies -------------------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method used -- see
    # `__init__`'s docstring. `_open_console_provider_recovery` is the one
    # bare-attribute-read shape (calls its accessor immediately and returns
    # the value, because the moved body passes it on as a callable rather
    # than calling it); every other property returns the callable itself.

    @property
    def _console_composer_or_none(self) -> Any:
        return self._composer_accessor

    @property
    def _ensure_console_chat_store(self) -> Any:
        return self._chat_store_accessor

    @property
    def _ensure_active_console_session_settings(self) -> Any:
        return self._ensure_active_console_session_settings_fn

    @property
    def _apply_console_session_system_prompt(self) -> Any:
        return self._apply_console_session_system_prompt_fn

    @property
    def _sync_console_session_draft(self) -> Any:
        return self._sync_console_session_draft_fn

    @property
    def _active_console_provider_model_display(self) -> Any:
        return self._active_console_provider_model_display_fn

    @property
    def _build_console_provider_selection(self) -> Any:
        return self._build_console_provider_selection_fn

    @property
    def _ensure_console_provider_gateway(self) -> Any:
        return self._ensure_console_provider_gateway_fn

    @property
    def _console_provider_blocker_copy(self) -> Any:
        return self._console_provider_blocker_copy_fn

    @property
    def _open_console_provider_recovery(self) -> Any:
        """The screen's CURRENT `_open_console_provider_recovery` attribute.

        Calls the accessor and returns the value, not the callable wrapping
        it: `_open_console_prompts_modal` hands this straight to the modal
        as `configure_provider=`, so the modal must receive the same object
        identity the pre-move code gave it. See `__init__`'s docstring.
        """
        return self._open_console_provider_recovery_accessor()

    @property
    def _console_setup_blocked_reason(self) -> Any:
        return self._console_setup_blocked_reason_fn

    @property
    def _focus_console_composer_if_needed(self) -> Any:
        return self._focus_console_composer_if_needed_fn

    @property
    def _insert_prompt_text_into_composer(self) -> Any:
        return self._insert_prompt_text_into_composer_fn

    @property
    def _clear_console_composer_draft(self) -> Any:
        return self._clear_console_composer_draft_fn

    @property
    def _append_native_console_system_message(self) -> Any:
        return self._append_native_console_system_message_fn

    @property
    def _sync_console_system_prompt_surfaces(self) -> Any:
        return self._sync_console_system_prompt_surfaces_fn

    @property
    def _sync_console_command_popup(self) -> Any:
        return self._sync_console_command_popup_fn

    # -- Moved bodies -------------------------------------------------------

    def _ensure_console_prompt_history(self) -> PromptHistory:
        """Return the shared JSONL prompt-history store (TASK-1364).

        One instance feeds both the composer (ghost text, Up/Down recall)
        and the controller (recording accepted sends). Creation is lazy and
        IO-free -- the store self-loads on first awaited use, and the
        composer kicks a background `load()` on mount so ghost text works on
        the first keystroke. `console_prompt_history_factory` on the app is
        the test seam, mirroring `console_provider_gateway_factory`.
        """
        history = getattr(self, "_console_prompt_history", None)
        if history is None:
            factory = getattr(self.app_instance, "console_prompt_history_factory", None)
            history = (
                factory()
                if callable(factory)
                else PromptHistory(default_prompt_history_path())
            )
            self._console_prompt_history = history
        return history

    def _restore_console_composer_focus(
        self, _result: ConsolePromptsResult | None = None
    ) -> None:
        """Return focus to the composer when a prompt modal closes.

        The `push_screen` callback for the Prompt Library modal: the modal's
        result is irrelevant to focus, so it is accepted and ignored.

        Args:
            _result: Whatever the modal dismissed with; unused.
        """
        self._focus_console_composer_if_needed(force=True)

    def _open_console_prompts_modal(self) -> None:
        """Open the source-aware Prompt Library without changing the draft.

        Builds the modal's two collaborators for this one open -- a
        `_ConsolePromptSource` over the app's prompt scope service, and a
        `_ConsolePromptImprovementFlow` holding everything the improvement
        sub-flow was disclosed against -- and hands their bound methods to
        `ConsolePromptsModal`. Before task 2766 those were 17 closures over
        this method's scope; see each class's docstring for why they split
        where they do.
        """
        source = _ConsolePromptSource(
            getattr(self.app_instance, "prompt_scope_service", None)
        )
        composer = self._console_composer_or_none()
        if composer is None:
            return
        settings = self._ensure_active_console_session_settings()
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return
        composer_snapshot = composer.capture_draft_snapshot()
        current_system = str(settings.system_prompt or "")
        current_system_fingerprint = fingerprint_text(current_system)
        provider_display, model_display, _settings = (
            self._active_console_provider_model_display()
        )
        opening_selection = self._build_console_provider_selection()
        gateway = self._ensure_console_provider_gateway()
        improvement_service = PromptImprovementService(gateway=gateway)
        improvement_context = ConsolePromptImprovementContext(
            session_id=session_id,
            composer_snapshot=composer_snapshot,
            current_user_projection=None,
            current_system_prompt=current_system,
            current_system_fingerprint=current_system_fingerprint,
            provider_label=str(provider_display or "Not configured"),
            model_label=str(model_display or "Not configured"),
            endpoint_label=(
                safe_endpoint_display(opening_selection.base_url)
                or "Resolve on Improve"
            ),
            model_unavailable_reason=self._console_provider_blocker_copy(),
        )

        flow = _ConsolePromptImprovementFlow(
            source=source,
            store=store,
            session_id=session_id,
            composer=composer,
            composer_snapshot=composer_snapshot,
            current_system=current_system,
            current_system_fingerprint=current_system_fingerprint,
            gateway=gateway,
            improvement_context=improvement_context,
            app_instance=self.app_instance,
            active_session_settings=self._ensure_active_console_session_settings,
            build_provider_selection=self._build_console_provider_selection,
            sync_system_prompt_surfaces=self._sync_console_system_prompt_surfaces,
        )

        self.push_screen(
            ConsolePromptsModal(
                capabilities=source.capabilities,
                list_page=source.list_page,
                search=source.search,
                detail=source.detail,
                save=source.save,
                improve_unavailable_reason=self._console_provider_blocker_copy(),
                configure_provider=self._open_console_provider_recovery,
                improvement_context=improvement_context,
                activate_improvement_context=flow.activate_improvement_context,
                capture_manual_resolution=flow.capture_manual_resolution,
                build_improvement_snapshot=flow.build_improvement_snapshot,
                improve=improvement_service.improve,
                validate_improvement=flow.validate_improvement,
                apply_improvement_result=flow.apply_improvement_result,
                retry_improvement_persistence=flow.retry_improvement_persistence,
            ),
            callback=self._restore_console_composer_focus,
        )

    @staticmethod
    def _is_recipe_prompt_record(record: Mapping[str, Any]) -> bool:
        """Return whether a normalized or raw record is a non-executable Recipe."""
        return str(record.get("artifact_type") or "prompt").casefold() == "recipe"

    @staticmethod
    def _console_prompt_prefix_fts_query(text: str) -> str:
        """Build an FTS5 phrase-prefix MATCH expression for ``text``.

        Plain FTS5 MATCH requires a full token, which would defeat both the
        `/prompt` prefix-match resolution stage (a query like "Summ" would
        never match a stored name "Summarize") and a picker that is supposed
        to filter results as the user is still mid-word. Quoting the whole
        query as a phrase with a trailing ``*`` makes FTS5 match names whose
        tokens *start with* the typed text instead -- a prefix match trivially
        covers an exact match too, so one query shape serves both. Embedded
        quotes are doubled per FTS5 string-literal escaping (mirrors
        ``library_fts_query._quote_fts_term``), so user text can never break
        out of the quoted phrase to inject MATCH operators.
        """
        return quote_fts5_prefix(text)

    async def _console_prompt_search(self, query: str) -> list:
        """Bounded FTS prompt search bound to the active scope service.

        Shared by `/prompt` resolution and the picker's ``prompt_search``
        callable so both always read a fresh page rather than any cached
        boot-time snapshot.
        """
        service = getattr(self.app_instance, "prompt_scope_service", None)
        search_prompts = getattr(service, "search_prompts", None)
        if not callable(search_prompts):
            return []
        stripped_query = query.strip()
        fts_kwargs = (
            {"fts_match_query": self._console_prompt_prefix_fts_query(stripped_query)}
            if stripped_query
            else {}
        )
        try:
            records = await search_prompts(
                mode="local",
                query=query,
                limit=self._CONSOLE_PROMPT_SEARCH_LIMIT,
                **fts_kwargs,
            )
            return [
                record
                for record in records
                if isinstance(record, Mapping)
                and not self._is_recipe_prompt_record(record)
            ]
        except Exception:
            logger.opt(exception=True).warning(
                f"Console prompt search failed for query {query!r}."
            )
            return []

    async def _resolve_console_prompt_by_name(
        self, query: str
    ) -> Optional[Mapping[str, Any]]:
        """Resolve `/prompt <name>` to a single prompt record, or ``None``.

        ``None`` means the caller should fall back to the picker: no
        candidates, an ambiguous (2+) exact case-insensitive name match, or
        no/ambiguous unique prefix match either.
        """
        candidates = [
            record
            for record in await self._console_prompt_search(query)
            if isinstance(record, Mapping) and not self._is_recipe_prompt_record(record)
        ]
        normalized_query = query.strip().casefold()
        exact_matches = [
            record
            for record in candidates
            if str(record.get("name") or "").strip().casefold() == normalized_query
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return None
        prefix_matches = [
            record
            for record in candidates
            if str(record.get("name") or "")
            .strip()
            .casefold()
            .startswith(normalized_query)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def _capture_prompt_replace_target(self) -> _ConsolePromptReplaceTarget | None:
        """Capture the exact active Console target before asynchronous work."""
        composer = self._console_composer_or_none()
        if composer is None:
            return None
        settings = self._ensure_active_console_session_settings()
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return None
        return _ConsolePromptReplaceTarget(
            composer_snapshot=composer.capture_draft_snapshot(),
            session_id=session_id,
            system_fingerprint=fingerprint_system_text(
                str(settings.system_prompt or "")
            ),
        )

    def _warn_prompt_application(self, copy: str) -> None:
        self.app_instance.notify(copy, severity="warning")
        self._focus_console_composer_if_needed(force=True)

    def _apply_prompt_application(
        self,
        application: PromptVariableApplication,
        *,
        captured_snapshot: ComposerDraftSnapshot | None,
    ) -> bool:
        """Apply one guarded replacement request through the shared transaction."""
        if (
            not isinstance(application, PromptVariableApplication)
            or application.destination != "replace_snapshot"
        ):
            self._warn_prompt_application(self._PROMPT_APPLICATION_STALE_COPY)
            return False
        return self._apply_guarded_prompt_application(
            application,
            captured_snapshot=captured_snapshot,
        )

    def _apply_guarded_prompt_application(
        self,
        application: PromptVariableApplication,
        *,
        captured_snapshot: ComposerDraftSnapshot | None,
    ) -> bool:
        """Guard, mutate, roll back, and report either Prompt destination."""
        destination = application.destination
        stale_copy = (
            self._PROMPT_APPEND_STALE_COPY
            if destination == "append_active"
            else self._PROMPT_APPLICATION_STALE_COPY
        )
        if destination not in ("replace_snapshot", "append_active"):
            self._warn_prompt_application(stale_copy)
            return False
        if application.is_expired():
            self._warn_prompt_application(self._PROMPT_APPLICATION_EXPIRED_COPY)
            return False

        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        composer = self._console_composer_or_none()
        if session_id != application.target_session_id or composer is None:
            self._warn_prompt_application(stale_copy)
            return False
        settings = store.session_settings(session_id)
        if settings is None:
            self._warn_prompt_application(stale_copy)
            return False
        if application.apply_system and (
            fingerprint_system_text(str(settings.system_prompt or ""))
            != application.system_fingerprint
        ):
            self._warn_prompt_application(stale_copy)
            return False

        if destination == "replace_snapshot":
            if (
                not isinstance(captured_snapshot, ComposerDraftSnapshot)
                or captured_snapshot.fingerprint != application.composer_fingerprint
                or composer.capture_draft_snapshot() != captured_snapshot
            ):
                self._warn_prompt_application(stale_copy)
                return False

        checkpoint = composer.capture_transaction_checkpoint()
        composer_changed = False
        persisted = True
        try:
            if destination == "replace_snapshot":
                replacement_text = (
                    application.user_text
                    if application.apply_user and application.user_text is not None
                    else ""
                )
                composer_changed = (
                    composer.replace_snapshot_as_paste(
                        captured_snapshot,
                        replacement_text,
                    )
                    is not None
                )
            elif application.apply_user and application.user_text:
                if composer.draft_text():
                    composer.move_cursor_end()
                    composer.insert_text_as_paste(f"\n{application.user_text}")
                else:
                    composer.insert_text_as_paste(application.user_text)
                composer_changed = True

            store.set_session_draft(session_id, composer.draft_text())
            if application.apply_system:
                _session, persisted = store.set_session_system_prompt(
                    session_id,
                    application.system_text,
                )
            if not composer_changed:
                composer.invalidate_improvement_undo()
        except Exception as exc:
            try:
                composer.rollback_transaction(checkpoint)
            except Exception:
                pass
            try:
                store.set_session_draft(session_id, composer.draft_text())
            except Exception:
                pass
            if application.apply_system:
                try:
                    store.replace_session_settings(session_id, settings)
                    self._sync_console_system_prompt_surfaces()
                except Exception:
                    pass
            try:
                self._sync_console_command_popup()
            except Exception:
                pass
            self._warn_prompt_application(
                stale_copy
                if isinstance(exc, ComposerTransactionValidationError)
                else self._PROMPT_APPLICATION_FAILED_COPY
            )
            return False

        display_sync_failed = False
        if application.apply_system:
            try:
                self._sync_console_system_prompt_surfaces()
            except Exception:
                display_sync_failed = True

        try:
            self._sync_console_command_popup()
        except Exception:
            display_sync_failed = True
        if display_sync_failed:
            self._warn_prompt_application(self._PROMPT_DISPLAY_SYNC_FAILED_COPY)
        if not persisted:
            self._warn_prompt_application(self._PROMPT_SYSTEM_PERSISTENCE_FAILED_COPY)
        if not display_sync_failed and persisted:
            self._focus_console_composer_if_needed(force=True)
        return True

    def _launch_prompt_application(
        self,
        record: Mapping[str, Any],
        target: _ConsolePromptReplaceTarget,
    ) -> None:
        """Route one resolved Prompt through the shared dialog or fast path."""
        raw_system = record.get("system_prompt")
        system_text = (
            raw_system if isinstance(raw_system, str) and raw_system.strip() else None
        )
        raw_user = record.get("user_prompt")
        user_source = raw_user if isinstance(raw_user, str) else ""
        user_text = (
            None if system_text is not None and not user_source.strip() else user_source
        )
        plan = compile_prompt_variables(
            system_text=system_text,
            user_text=user_text,
        )

        if system_text is None and plan.is_valid and not plan.variables:
            application = PromptVariableApplication(
                system_text=None,
                user_text=user_text,
                apply_system=False,
                apply_user=True,
                destination="replace_snapshot",
                target_session_id=target.session_id,
                composer_fingerprint=target.composer_snapshot.fingerprint,
                system_fingerprint=None,
            )
            self._apply_prompt_application(
                application,
                captured_snapshot=target.composer_snapshot,
            )
            return

        request = PromptVariablesDialogRequest(
            system_text=system_text,
            user_text=user_text,
            destination="replace_snapshot",
            target_session_id=target.session_id,
            composer_fingerprint=target.composer_snapshot.fingerprint,
            system_fingerprint=(
                target.system_fingerprint if system_text is not None else None
            ),
        )

        def _apply_dialog_result(
            application: PromptVariableApplication | None,
        ) -> None:
            if application is not None:
                self._apply_prompt_application(
                    application,
                    captured_snapshot=target.composer_snapshot,
                )
            else:
                self._focus_console_composer_if_needed(force=True)

        self.push_screen(
            PromptVariablesDialog(request),
            callback=_apply_dialog_result,
        )

    async def _console_command_insert_prompt(self, parse: CommandParse) -> None:
        """Resolve and insert a saved prompt's ``user_prompt`` for `/prompt`.

        Resolution order (brief): exact case-insensitive name match over a
        bounded search page; else a unique case-insensitive name-prefix
        match over that same page; else (no args, 0 matches, or an
        ambiguous 2+ match at either stage) open the picker prefilled with
        the typed args. A resolved match REPLACES the composer draft
        wholesale (the draft IS the `/prompt ...` command being replaced by
        its result) via paste semantics, so an oversized body still
        collapses to a token exactly like a real paste would.
        """
        target = self._capture_prompt_replace_target()
        if target is None:
            return
        query = parse.args.strip()
        resolved = await self._resolve_console_prompt_by_name(query) if query else None
        if resolved is not None:
            if self._is_recipe_prompt_record(resolved):
                await self._append_native_console_system_message(
                    self._RECIPE_EXECUTION_BLOCKED_COPY
                )
                return
            self._launch_prompt_application(resolved, target)
            return
        await self._open_console_prompt_picker_for_insert(query, target=target)

    async def _open_console_prompt_picker_for_insert(
        self,
        initial_query: str,
        *,
        target: _ConsolePromptReplaceTarget | None = None,
    ) -> None:
        """Open the prompt picker for `/prompt`, inserting whatever is chosen."""

        captured_target = target or self._capture_prompt_replace_target()
        if captured_target is None:
            return

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            if self._is_recipe_prompt_record(record):
                self.app_instance.notify(
                    self._RECIPE_EXECUTION_BLOCKED_COPY,
                    severity="warning",
                )
                return
            self._launch_prompt_application(record, captured_target)

        self.push_screen(
            ConsolePromptPickerModal(
                mode="insert",
                initial_query=initial_query,
                prompt_search=self._console_prompt_search,
            ),
            callback=_apply_picker_choice,
        )

    async def _consume_pending_console_prompt_insert(self) -> None:
        """Settle and apply one typed Library append request, if pending."""
        handoffs = self.app_instance.pending_handoffs
        claim = handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
        if claim is None:
            return
        if claim.status == "expired":
            handoffs.acknowledge(claim)
            self._warn_prompt_application(self._PROMPT_APPLICATION_EXPIRED_COPY)
            return
        application = claim.value
        if (
            not isinstance(application, PromptVariableApplication)
            or application.destination != "append_active"
        ):
            handoffs.acknowledge(claim)
            self._warn_prompt_application(self._PROMPT_APPEND_STALE_COPY)
            return
        if (
            application.apply_user
            and not application.user_text
            and not application.apply_system
        ):
            handoffs.acknowledge(claim)
            self._warn_prompt_application(self._PROMPT_APPEND_EMPTY_COPY)
            return
        if self._console_setup_blocked_reason():
            handoffs.acknowledge(claim)
            self.app_instance.notify(
                self._LIBRARY_PROMPT_INSERT_BLOCKED_COPY,
                severity="warning",
            )
            return

        def release_for_retry() -> None:
            if handoffs.release_prompt_claim(claim) == "expired":
                self._warn_prompt_application(self._PROMPT_APPLICATION_EXPIRED_COPY)

        try:
            self._sync_console_session_draft()
        except asyncio.CancelledError:
            release_for_retry()
            raise
        except Exception:
            release_for_retry()
            return

        composer = self._console_composer_or_none()
        if composer is None:
            release_for_retry()
            return
        self._apply_guarded_prompt_application(
            application,
            captured_snapshot=None,
        )
        handoffs.acknowledge(claim)

    def console_prompt_target_projection(
        self,
    ) -> ConsolePromptTargetProjection | None:
        """Return the sanitized live Console target for app-owned publication.

        Returns:
            The active session and one-way System fingerprint, or ``None``
            when Console has no complete active target.
        """
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        settings = (
            store.session_settings(session_id) if session_id is not None else None
        )
        if session_id is None or settings is None:
            return None
        return ConsolePromptTargetProjection(
            target_session_id=session_id,
            system_fingerprint=fingerprint_system_text(
                str(settings.system_prompt or "")
            ),
        )

    async def _console_command_apply_system(self, parse: CommandParse) -> None:
        """Resolve and apply a saved prompt's ``system_prompt`` for `/system`.

        Bare `/system` (no args) opens the system prompt editor modal seeded
        with the active session's current system prompt. With args,
        resolution mirrors `/prompt` (Task 12): exact case-insensitive name
        match over a bounded search page, else a unique case-insensitive
        name-prefix match; a resolved match with a blank ``system_prompt``
        shows an inline transcript error (the session is left unchanged,
        and the draft is deliberately left in place so the user can correct
        it) rather than silently clearing it, since that is very likely not
        what the user meant by naming that specific prompt. A resolved match
        WITH a system part applies it and clears the `/system <name>`
        command text from the composer -- mirrors `/prompt`'s successful
        insert always replacing its own draft (Task 12) -- so a handled
        command never leaves its own invocation text behind. 0 or 2+
        matches at either stage fall back to the apply-system picker mode
        (Task 11), prefilled with the typed args.
        """
        args = parse.args.strip()
        if not args:
            await self._open_console_system_prompt_editor()
            return
        resolved = await self._resolve_console_prompt_by_name(args)
        if resolved is not None:
            if self._is_recipe_prompt_record(resolved):
                await self._append_native_console_system_message(
                    self._RECIPE_EXECUTION_BLOCKED_COPY
                )
                return
            # Blank check only via strip(); the applied value below is the
            # raw prompt text so leading/trailing whitespace and internal
            # formatting survive verbatim.
            raw_system_prompt = resolved.get("system_prompt")
            system_prompt = (
                raw_system_prompt if isinstance(raw_system_prompt, str) else ""
            )
            if not system_prompt.strip():
                name = str(resolved.get("name") or args)
                await self._append_native_console_system_message(
                    CONSOLE_SYSTEM_PROMPT_NO_SYSTEM_PART_TEMPLATE.format(name=name)
                )
                return
            self._apply_console_session_system_prompt(system_prompt)
            self._clear_console_composer_draft()
            return
        await self._open_console_prompt_picker_for_apply_system(args)

    async def _open_console_prompt_picker_for_apply_system(
        self, initial_query: str
    ) -> None:
        """Open the prompt picker in apply-system mode for `/system`.

        Rows without a ``system_prompt`` render dimmed and refuse selection
        (``ConsolePromptPickerModal``'s own ``MODE_APPLY_SYSTEM`` behavior,
        Task 11) -- this caller only needs to apply whatever record the
        picker actually dismisses with.
        """

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            if self._is_recipe_prompt_record(record):
                self.app_instance.notify(
                    self._RECIPE_EXECUTION_BLOCKED_COPY,
                    severity="warning",
                )
                return
            # Blank check only via strip(); the applied value is the raw
            # prompt text so formatting survives verbatim.
            raw_system_prompt = record.get("system_prompt")
            system_prompt = (
                raw_system_prompt if isinstance(raw_system_prompt, str) else ""
            )
            if not system_prompt.strip():
                return
            self._apply_console_session_system_prompt(system_prompt)

        self.push_screen(
            ConsolePromptPickerModal(
                mode=CONSOLE_PROMPT_PICKER_MODE_APPLY_SYSTEM,
                initial_query=initial_query,
                prompt_search=self._console_prompt_search,
            ),
            callback=_apply_picker_choice,
        )

    async def _open_console_system_prompt_editor(self) -> None:
        """Open the system prompt editor modal for the active Console session."""
        settings = self._ensure_active_console_session_settings()

        def _apply_modal_result(result: Optional[str]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if result is None:
                return
            self._apply_console_session_system_prompt(result)

        self.push_screen(
            ConsoleSystemPromptModal(
                system_prompt=settings.system_prompt,
                save_to_library=self._save_console_system_prompt_to_library,
            ),
            callback=_apply_modal_result,
        )

    async def _save_console_system_prompt_to_library(self, name: str, text: str) -> str:
        """Save the system-prompt editor's text as a brand-new Library prompt.

        Always a CREATE (the Console `/system` editor never edits an
        existing Library prompt): pre-checks the name for a collision the
        same way ``library_screen._save_library_prompt``'s own create path
        does, so a genuine duplicate is classified via
        ``classify_prompt_save_error`` -- with ``exc=None`` and a manually
        built message -- rather than racing the DB's raw ``ConflictError``,
        and reports the SAME outcome copy that screen's own save flow shows.

        Args:
            name: Name for the new Library prompt.
            text: The prompt's ``system_prompt`` body (the modal's current
                editor text).

        Returns:
            User-facing outcome copy to display inline in the modal.
        """
        name = name.strip()
        if not name:
            return "Enter a name to save this system prompt to Library."
        text = text.strip()
        if not text:
            return "Enter a system prompt to save."
        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        save_prompt = getattr(service, "save_prompt", None)
        if not callable(get_prompt) or not callable(save_prompt):
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
        try:
            candidate = await get_prompt(
                mode="local", prompt_identifier=name, include_deleted=True
            )
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping) and candidate:
            if candidate.get("deleted"):
                outcome = classify_prompt_save_error(
                    None, f"Prompt '{name}' exists but is soft-deleted.", None
                )
            else:
                outcome = classify_prompt_save_error(
                    None, f"Prompt '{name}' already exists.", None
                )
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
                outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
            )
        try:
            result = await save_prompt(
                mode="local", name=name, system_prompt=text, user_prompt=""
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Console system-prompt save-to-library failed for name {name!r}."
            )
            outcome = classify_prompt_save_error(None, str(exc), exc)
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
                outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
            )
        result_id = (
            result.get("local_id")
            if isinstance(result, Mapping)
            else (1 if result else None)
        )
        outcome = classify_prompt_save_error(result_id, "", None)
        return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
            outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
        )
