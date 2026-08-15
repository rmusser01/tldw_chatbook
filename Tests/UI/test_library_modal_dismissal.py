"""Library focus contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_prompts_state import (
    begin_prompt_collection_catalog,
)
from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
    PromptCollectionManagerModal,
)
from tldw_chatbook.UI.Screens.skills_screen import (
    SkillTrustBootstrapModal,
    SkillTrustPassphraseModal,
)
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderNameDialog,
    LibraryNoteFolderTargetDialog,
)
from tldw_chatbook.Notes.file_notes_conflict_compare import (
    ConflictSide,
    build_conflict_comparison,
)
from tldw_chatbook.Notes.file_notes_git_push import (
    PushAuthorizationProjection,
    PushCandidateProjection,
    PushDestinationProjection,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    PushDestinationAuthorizationDialog,
    PushEndpointDetailsDialog,
    SessionGitTrustDialog,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FileNotesConflictCompareDialog,
    FileNotesRootDetailsDialog,
)
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
)
from tldw_chatbook.Widgets.ModelArtifacts.install_modal import ModelInstallModal
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


_STABLE_OPENER_ID = "library-stable-opener"


ORDINARY_LIBRARY_MODAL_CONTRACTS = (
    (SkillTrustPassphraseModal, "#skill-trust-passphrase-modal"),
    (SkillTrustBootstrapModal, "#skill-trust-bootstrap-modal"),
    (ModelInstallModal, ".model-install-modal"),
    (PromptDeleteConfirmationModal, "#prompt-delete-modal"),
    (LibraryNoteFolderNameDialog, "#library-note-folder-name-dialog"),
    (LibraryNoteFolderTargetDialog, "#library-note-folder-target-dialog"),
)


def _push_destination() -> PushDestinationProjection:
    return PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/session-notes",
    )


def _push_authorization_dialog() -> PushDestinationAuthorizationDialog:
    candidate = PushCandidateProjection(
        local_branch_ref="refs/heads/main",
        parent_oid="a" * 40,
        candidate_oid="b" * 40,
        subject="Publish notes",
        included_notes=(),
    )
    return PushDestinationAuthorizationDialog(
        candidate,
        PushAuthorizationProjection(_push_destination()),
    )


def _conflict_compare_dialog() -> FileNotesConflictCompareDialog:
    comparison = build_conflict_comparison(
        ConflictSide.from_text("Base", "base"),
        ConflictSide.from_text("Draft", "draft"),
        ConflictSide.from_text("Disk", "disk"),
    )
    return FileNotesConflictCompareDialog("note.md", comparison)


@dataclass(frozen=True)
class _FileNotesModalContract:
    name: str
    modal_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    visible_cancel: str
    cancel_result: object
    positive_action: str | None = None


FILE_NOTES_MODAL_CONTRACTS = (
    _FileNotesModalContract(
        "root-details",
        FileNotesRootDetailsDialog,
        lambda: FileNotesRootDetailsDialog("/notes"),
        "#file-notes-root-details-dialog",
        "#file-notes-root-details-close",
        None,
    ),
    _FileNotesModalContract(
        "conflict-compare",
        FileNotesConflictCompareDialog,
        _conflict_compare_dialog,
        "#file-notes-conflict-dialog",
        "#file-notes-conflict-close",
        None,
    ),
    _FileNotesModalContract(
        "trust",
        SessionGitTrustDialog,
        lambda: SessionGitTrustDialog("/notes"),
        "#confirmation-dialog",
        "#cancel-button",
        False,
        "#confirm-button",
    ),
    _FileNotesModalContract(
        "endpoint-details",
        PushEndpointDetailsDialog,
        lambda: PushEndpointDetailsDialog(_push_destination()),
        "#file-notes-push-endpoint-details-dialog",
        "#file-notes-push-endpoint-details-close",
        None,
    ),
    _FileNotesModalContract(
        "authorization",
        PushDestinationAuthorizationDialog,
        _push_authorization_dialog,
        "#file-notes-push-auth-dialog",
        "#file-notes-push-auth-cancel",
        False,
        "#file-notes-push-auth-confirm",
    ),
)


class _FileNotesModalHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


def _binding_key_action(binding: object) -> tuple[str, str]:
    if isinstance(binding, tuple):
        return str(binding[0]), str(binding[1])
    return str(binding.key), str(binding.action)  # type: ignore[attr-defined]


def test_file_notes_modal_dismissal_contracts_adopt_shared_boundary() -> None:
    for contract in FILE_NOTES_MODAL_CONTRACTS:
        assert issubclass(contract.modal_type, SafeModalDismissMixin)
        assert contract.modal_type.SAFE_MODAL_CONTENT == contract.content_selector
        escape_actions = [
            action
            for binding in contract.modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == ["request_safe_cancel"]


@pytest.mark.parametrize(
    "contract", FILE_NOTES_MODAL_CONTRACTS, ids=lambda row: row.name
)
@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_file_notes_modal_dismissal_returns_exact_negative_once(
    contract: _FileNotesModalContract,
    source: str,
) -> None:
    app = _FileNotesModalHarness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert modal.query_one(contract.content_selector)
        if source == "visible":
            await pilot.click(contract.visible_cancel)
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert len(app.results) == 1
    assert app.results[0] is contract.cancel_result


@pytest.mark.parametrize(
    "contract", FILE_NOTES_MODAL_CONTRACTS, ids=lambda row: row.name
)
@pytest.mark.asyncio
async def test_file_notes_modal_dismissal_inside_and_non_primary_stay_open(
    contract: _FileNotesModalContract,
) -> None:
    app = _FileNotesModalHarness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click(contract.content_selector)
        non_primary = events.Click(
            modal,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=3,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=0,
            screen_y=0,
        )
        await modal._dispatch_message(non_primary)
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []


@pytest.mark.parametrize(
    "contract",
    [row for row in FILE_NOTES_MODAL_CONTRACTS if row.positive_action is not None],
    ids=lambda row: row.name,
)
@pytest.mark.asyncio
async def test_file_notes_modal_dismissal_positive_behavior_is_unchanged(
    contract: _FileNotesModalContract,
) -> None:
    app = _FileNotesModalHarness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        assert contract.positive_action is not None
        await pilot.click(contract.positive_action)
        await pilot.pause()

    assert app.results == [True]


@pytest.mark.parametrize(
    "contract", FILE_NOTES_MODAL_CONTRACTS, ids=lambda row: row.name
)
@pytest.mark.asyncio
async def test_file_notes_modal_dismissal_lifecycle_runs_mixin_once(
    contract: _FileNotesModalContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount_calls = 0
    unmount_calls = 0
    original_mount = SafeModalDismissMixin.on_mount
    original_unmount = SafeModalDismissMixin.on_unmount

    def count_mount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mount_calls
        mount_calls += 1
        original_mount(self)

    def count_unmount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal unmount_calls
        unmount_calls += 1
        original_unmount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_mount", count_mount)
    monkeypatch.setattr(SafeModalDismissMixin, "on_unmount", count_unmount)
    app = _FileNotesModalHarness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        assert mount_calls == 1

        await pilot.press("escape")
        await pilot.pause()
        assert unmount_calls == 1


@pytest.mark.asyncio
async def test_authorization_endpoint_details_nested_modal_dismissal_restores_focus_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FileNotesModalHarness()
    authorization = _push_authorization_dialog()
    details_focus_calls = 0
    original_focus = Button.focus

    def count_details_focus(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal details_focus_calls
        if self.id == "file-notes-push-auth-details":
            details_focus_calls += 1
        return original_focus(self, *args, **kwargs)

    monkeypatch.setattr(Button, "focus", count_details_focus)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(authorization, callback=app.results.append)
        await pilot.pause()
        opener = authorization.query_one("#file-notes-push-auth-details", Button)
        await pilot.click(opener)
        await pilot.pause()
        endpoint_details = app.screen
        assert isinstance(endpoint_details, PushEndpointDetailsDialog)
        details_focus_calls = 0

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert app.screen is authorization
        assert authorization.focused is opener
        assert details_focus_calls == 1
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [False]


def test_library_modal_contract_ordinary_modals_adopt_safe_dismissal() -> None:
    assert len(ORDINARY_LIBRARY_MODAL_CONTRACTS) == 6
    for modal_type, content_selector in ORDINARY_LIBRARY_MODAL_CONTRACTS:
        assert issubclass(modal_type, SafeModalDismissMixin)
        assert modal_type.SAFE_MODAL_CONTENT == content_selector
        escape_actions = [
            action
            for binding in modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == ["request_safe_cancel"]


def _prompt_collection_manager_modal() -> PromptCollectionManagerModal:
    async def load(*, query: str, offset: int):
        del offset
        return begin_prompt_collection_catalog(query=query, request_token=1)

    async def unused(*_args, **_kwargs):
        raise AssertionError("collection mutation was not requested")

    return PromptCollectionManagerModal(
        mode="browse",
        selected_collection_id=None,
        staged_collection_ids=(),
        load_catalog=load,
        create_collection=unused,
        rename_collection=unused,
    )


def test_prompt_collection_modal_contract_adopts_exact_safe_boundary() -> None:
    assert issubclass(PromptCollectionManagerModal, SafeModalDismissMixin)
    assert (
        PromptCollectionManagerModal.SAFE_MODAL_CONTENT == "#prompt-collection-manager"
    )
    assert [
        action
        for binding in PromptCollectionManagerModal.BINDINGS
        for key, action in [_binding_key_action(binding)]
        if key == "escape"
    ] == ["request_safe_cancel"]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_prompt_collection_modal_contract_idle_cancel_returns_none(
    source: str,
) -> None:
    app = _FileNotesModalHarness()
    modal = _prompt_collection_manager_modal()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        if source == "visible":
            await pilot.click("#prompt-collection-manager-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert app.results == [None]


@pytest.mark.parametrize(
    "modal_type",
    [SkillTrustPassphraseModal, SkillTrustBootstrapModal],
    ids=["passphrase", "bootstrap"],
)
@pytest.mark.asyncio
async def test_ordinary_modal_lifecycle_runs_skill_and_mixin_handlers_once(
    modal_type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixin_mount_calls = 0
    mixin_unmount_calls = 0
    original_mount = SafeModalDismissMixin.on_mount
    original_unmount = SafeModalDismissMixin.on_unmount

    def count_mount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_mount_calls
        mixin_mount_calls += 1
        original_mount(self)

    def count_unmount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_unmount_calls
        mixin_unmount_calls += 1
        original_unmount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_mount", count_mount)
    monkeypatch.setattr(SafeModalDismissMixin, "on_unmount", count_unmount)
    app = App()
    modal = (
        modal_type(confirm_bootstrap=False)
        if modal_type is SkillTrustPassphraseModal
        else modal_type()
    )

    async with app.run_test(size=(100, 36)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        assert mixin_mount_calls == 1
        assert isinstance(modal.focused, Input)

        modal.dismiss(None)
        await pilot.pause()
        assert mixin_unmount_calls == 1


class _LibraryLikeHost(Screen[None]):
    """Small revealed screen whose opener can be replaced by recomposition."""

    def __init__(self, *, initial_id: str | None = _STABLE_OPENER_ID) -> None:
        super().__init__()
        self.initial_id = initial_id
        self.replacement_ids: tuple[str, ...] | None = None

    def compose(self) -> ComposeResult:
        opener_ids = (
            (self.initial_id,) if self.replacement_ids is None else self.replacement_ids
        )
        for index, opener_id in enumerate(opener_ids):
            with Container(id=f"library-opener-slot-{index}"):
                yield Input(
                    id=opener_id,
                    classes="library-focus-opener",
                )
        yield Container(id="library-extra-opener-slot")
        yield Input(id="library-normal-focus-policy")
        yield Input(id="library-unrelated-action")

    async def recompose_openers(self, *opener_ids: str) -> None:
        self.replacement_ids = opener_ids
        await self.recompose()


class _LibraryFocusHarness(App[None]):
    def __init__(self, *, initial_id: str | None = _STABLE_OPENER_ID) -> None:
        super().__init__()
        self.host = _LibraryLikeHost(initial_id=initial_id)

    async def on_mount(self) -> None:
        await self.push_screen(self.host)


class _LibraryFocusModal(SafeModalDismissMixin, ModalScreen[None]):
    SAFE_MODAL_CONTENT = "#library-focus-modal-content"

    def compose(self) -> ComposeResult:
        with Vertical(id="library-focus-modal-content"):
            yield Static("Library modal")


async def _mount_focus_modal(
    app: _LibraryFocusHarness,
    pilot,
    *,
    empty_id: bool = False,
) -> tuple[_LibraryFocusModal, Input]:
    opener = app.host.query_one(".library-focus-opener", Input)
    if empty_id:
        opener._id = ""
    opener.focus()
    await pilot.pause()
    assert app.host.focused is opener

    modal = _LibraryFocusModal()
    app.push_screen(modal)
    await pilot.pause()
    assert app.screen is modal
    return modal, opener


async def _dismiss_focus_modal(modal: _LibraryFocusModal, pilot) -> None:
    await modal.action_request_safe_cancel()
    await pilot.pause()
    await pilot.pause()


def _make_ineligible(widget: Widget, reason: str) -> None:
    if reason == "not-displayed":
        widget.display = False
    elif reason == "not-visible":
        widget.visible = False
    elif reason == "disabled":
        widget.disabled = True
    else:
        assert reason == "not-focusable"
        widget.can_focus = False


@pytest.mark.asyncio
async def test_opener_focus_restores_the_exact_eligible_library_opener() -> None:
    app = _LibraryFocusHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal, opener = await _mount_focus_modal(app, pilot)
        assert opener.is_mounted
        assert opener.is_attached
        assert opener.display
        assert opener.visible
        assert not opener.disabled
        assert opener.can_focus

        app.host.set_focus(app.host.query_one("#library-normal-focus-policy", Input))
        await _dismiss_focus_modal(modal, pilot)

        assert app.screen is app.host
        assert app.host.focused is opener


@pytest.mark.asyncio
async def test_stable_focus_restores_one_eligible_recomposed_opener() -> None:
    app = _LibraryFocusHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal, original = await _mount_focus_modal(app, pilot)
        await app.host.recompose_openers(_STABLE_OPENER_ID)
        await pilot.pause()
        replacement = app.host.query_one(f"#{_STABLE_OPENER_ID}", Input)
        assert replacement is not original
        assert not original.is_attached

        app.host.set_focus(app.host.query_one("#library-normal-focus-policy", Input))
        await _dismiss_focus_modal(modal, pilot)

        assert app.host.focused is replacement


@pytest.mark.parametrize(
    "reason",
    ["not-displayed", "not-visible", "disabled", "not-focusable"],
)
@pytest.mark.asyncio
async def test_stable_focus_rejects_an_ineligible_exact_opener_for_replacement(
    reason: str,
) -> None:
    app = _LibraryFocusHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal, original = await _mount_focus_modal(app, pilot)
        replacement = Input(id=_STABLE_OPENER_ID)
        await app.host.query_one("#library-extra-opener-slot", Container).mount(
            replacement
        )
        _make_ineligible(original, reason)
        await pilot.pause()

        app.host.set_focus(app.host.query_one("#library-normal-focus-policy", Input))
        await _dismiss_focus_modal(modal, pilot)

        assert original.is_mounted
        assert app.host.focused is replacement


@pytest.mark.parametrize("ancestor_state", ["not-displayed", "disabled"])
@pytest.mark.asyncio
async def test_stable_focus_rejects_an_inaccessible_opener_ancestor_for_replacement(
    ancestor_state: str,
) -> None:
    app = _LibraryFocusHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal, original = await _mount_focus_modal(app, pilot)
        replacement = Input(id=_STABLE_OPENER_ID)
        await app.host.query_one("#library-extra-opener-slot", Container).mount(
            replacement
        )
        opener_ancestor = original.parent
        assert isinstance(opener_ancestor, Container)
        if ancestor_state == "not-displayed":
            opener_ancestor.display = False
        else:
            opener_ancestor.disabled = True
        await pilot.pause()

        app.host.set_focus(app.host.query_one("#library-normal-focus-policy", Input))
        await _dismiss_focus_modal(modal, pilot)

        assert original.is_mounted
        assert app.host.focused is replacement


@pytest.mark.parametrize(
    ("identity_case", "replacement_ids", "ineligible_reasons"),
    [
        pytest.param("missing", (), (), id="missing-id"),
        pytest.param(
            "duplicated",
            (_STABLE_OPENER_ID, _STABLE_OPENER_ID),
            (),
            id="duplicated-id",
        ),
        pytest.param("empty", (), (), id="empty-id"),
        pytest.param(
            "ineligible",
            (_STABLE_OPENER_ID,),
            ("not-displayed",),
            id="only-not-displayed",
        ),
        pytest.param(
            "ineligible",
            (_STABLE_OPENER_ID,),
            ("not-visible",),
            id="only-not-visible",
        ),
        pytest.param(
            "ineligible",
            (_STABLE_OPENER_ID,),
            ("disabled",),
            id="only-disabled",
        ),
        pytest.param(
            "ineligible",
            (_STABLE_OPENER_ID,),
            ("not-focusable",),
            id="only-not-focusable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_stable_focus_leaves_unusable_identity_to_revealed_screen_policy(
    identity_case: str,
    replacement_ids: tuple[str, ...],
    ineligible_reasons: tuple[str, ...],
) -> None:
    initial_id = None if identity_case in {"missing", "empty"} else _STABLE_OPENER_ID
    app = _LibraryFocusHarness(initial_id=initial_id)

    async with app.run_test(size=(80, 24)) as pilot:
        modal, _original = await _mount_focus_modal(
            app,
            pilot,
            empty_id=identity_case == "empty",
        )

        await app.host.recompose_openers(*replacement_ids)
        await pilot.pause()
        replacements = list(app.host.query(f"#{_STABLE_OPENER_ID}"))
        for index, reason in enumerate(ineligible_reasons):
            _make_ineligible(replacements[index], reason)

        policy_target = app.host.query_one("#library-normal-focus-policy", Input)
        app.host.set_focus(policy_target)
        await _dismiss_focus_modal(modal, pilot)

        assert app.screen is app.host
        assert app.host.focused is policy_target
        assert app.host.focused is not app.host.query_one(
            "#library-unrelated-action", Input
        )
