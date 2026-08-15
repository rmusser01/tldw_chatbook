"""Library focus contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Input, Static

from tldw_chatbook.UI.Screens.skills_screen import (
    SkillTrustBootstrapModal,
    SkillTrustPassphraseModal,
)
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderNameDialog,
    LibraryNoteFolderTargetDialog,
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


def _binding_key_action(binding: object) -> tuple[str, str]:
    if isinstance(binding, tuple):
        return str(binding[0]), str(binding[1])
    return str(binding.key), str(binding.action)  # type: ignore[attr-defined]


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
