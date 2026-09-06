"""Library focus contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import ArtifactRef, ProvenanceClass
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
)
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    FileSave,
    SelectDirectory,
)
from tldw_chatbook.Library.library_prompts_state import (
    begin_prompt_collection_catalog,
)
from tldw_chatbook.UI.Library_Modules.prompt_collections import (
    PromptCollectionManagerResult,
)
from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
    PromptCollectionManagerModal,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
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
    PromptDeleteDecision,
    PromptDeleteConfirmationModal,
    PromptDeleteItem,
    PromptDeleteRequest,
)
from tldw_chatbook.Widgets.ModelArtifacts.install_modal import ModelInstallModal
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    PromptVariablesDialog,
    PromptVariablesDialogRequest,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedFileSave,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


_STABLE_OPENER_ID = "library-stable-opener"


@dataclass(frozen=True)
class LibraryModalContract:
    """One concrete Library modal's independently mounted behavior contract."""

    concrete_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    visible_negative_selector: str
    negative_assertion: Callable[[object], None]
    positive_type: type[object] | tuple[type[object], ...]
    positive_assertion: Callable[[object], None] | None
    active_guard: str | None
    focus_postcondition: str
    non_dismissible_reason: str | None


@dataclass(frozen=True)
class LibraryModalLaunchEdge:
    """One supported production presenter edge to a concrete modal type."""

    owner_file: str
    owner_class: str
    presenter_name: str
    concrete_type: type[ModalScreen[Any]]


def _assert_none(result: object) -> None:
    assert result is None


def _assert_false(result: object) -> None:
    assert result is False


def _assert_prompt_delete_negative(result: object) -> None:
    assert result == PromptDeleteDecision(False, "library-contract")


def _assert_exact(expected: object) -> Callable[[object], None]:
    def assertion(result: object) -> None:
        assert type(result) is type(expected)
        assert result == expected

    return assertion


def _assert_prompt_variables_positive(result: object) -> None:
    assert type(result) is PromptVariableApplication
    assert result.system_text is None
    assert result.user_text == "Hello {name}"
    assert result.apply_system is False
    assert result.apply_user is True
    assert result.destination == "replace_snapshot"
    assert result.target_session_id == "library-contract"
    assert result.composer_fingerprint == "a" * 64
    assert result.system_fingerprint is None
    assert result.created_monotonic < result.expires_monotonic


def _model_install_modal() -> ModelInstallModal:
    reference = ArtifactRef("library-contract", "revision", "int8")
    report = PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=reference,
                source_url="https://example.test/model",
                repository="publisher/library-contract",
                revision="revision",
                license_id="CC-BY-4.0",
                license_url="https://example.test/license",
                precision="int8",
                total_bytes=1,
                file_count=1,
                already_installed=False,
                provenance=(ProvenanceClass.CHATBOOK_CURATED,),
            ),
        ),
        download_bytes=1,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=Path("/tmp/library-modal-contract-model"),
        free_bytes=2,
        required_bytes=1,
        sufficient_space=True,
        gating_errors=(),
    )
    return ModelInstallModal(report, model_label="Library contract")


def _contract_prompt_collection_modal() -> PromptCollectionManagerModal:
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


def _prompt_delete_modal() -> PromptDeleteConfirmationModal:
    return PromptDeleteConfirmationModal(
        PromptDeleteRequest(
            items=(PromptDeleteItem("Contract prompt", "prompt"),),
            fingerprint="library-contract",
        )
    )


def _prompt_variables_modal() -> PromptVariablesDialog:
    return PromptVariablesDialog(
        PromptVariablesDialogRequest(
            system_text=None,
            user_text="Hello {name}",
            destination="replace_snapshot",
            target_session_id="library-contract",
            composer_fingerprint="a" * 64,
            system_fingerprint=None,
        )
    )


def _contract_push_destination() -> PushDestinationProjection:
    return PushDestinationProjection(
        "https",
        "push.example.test",
        443,
        "/team/notes.git",
        "refs/heads/session-notes",
    )


def _contract_push_authorization_dialog() -> PushDestinationAuthorizationDialog:
    candidate = PushCandidateProjection(
        local_branch_ref="refs/heads/main",
        parent_oid="a" * 40,
        candidate_oid="b" * 40,
        subject="Publish notes",
        included_notes=(),
    )
    return PushDestinationAuthorizationDialog(
        candidate,
        PushAuthorizationProjection(_contract_push_destination()),
    )


def _contract_conflict_compare_dialog() -> FileNotesConflictCompareDialog:
    comparison = build_conflict_comparison(
        ConflictSide.from_text("Base", "base"),
        ConflictSide.from_text("Draft", "draft"),
        ConflictSide.from_text("Disk", "disk"),
    )
    return FileNotesConflictCompareDialog("note.md", comparison)


_FOCUS_POSTCONDITION = "restore the exact eligible opener identity"


LIBRARY_MODAL_CONTRACTS = (
    LibraryModalContract(
        SkillTrustPassphraseModal,
        lambda: SkillTrustPassphraseModal(confirm_bootstrap=False),
        "#skill-trust-passphrase-modal",
        "#skill-trust-passphrase-cancel",
        _assert_none,
        str,
        _assert_exact("secret"),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        SkillTrustBootstrapModal,
        SkillTrustBootstrapModal,
        "#skill-trust-bootstrap-modal",
        "#skill-trust-bootstrap-cancel",
        _assert_none,
        str,
        _assert_exact("secret"),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        ModelInstallModal,
        _model_install_modal,
        ".model-install-modal",
        "#model-install-cancel",
        _assert_false,
        bool,
        _assert_exact(True),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        FileOpen,
        lambda: FileOpen("/tmp", must_exist=False, default_file="open.txt"),
        "#file-system-picker-dialog",
        "#cancel",
        _assert_none,
        Path,
        _assert_exact(Path("/tmp/open.txt").resolve()),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        FileSave,
        lambda: FileSave("/tmp", default_file="save.txt"),
        "#file-system-picker-dialog",
        "#cancel",
        _assert_none,
        Path,
        _assert_exact(Path("/tmp/save.txt").resolve()),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        SelectDirectory,
        lambda: SelectDirectory("/tmp"),
        "#file-system-picker-dialog",
        "#cancel",
        _assert_none,
        Path,
        _assert_exact(Path("/tmp")),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        PromptDeleteConfirmationModal,
        _prompt_delete_modal,
        "#prompt-delete-modal",
        "#prompt-delete-cancel",
        _assert_prompt_delete_negative,
        PromptDeleteDecision,
        _assert_exact(PromptDeleteDecision(True, "library-contract")),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        LibraryNoteFolderNameDialog,
        lambda: LibraryNoteFolderNameDialog(title="New folder"),
        "#library-note-folder-name-dialog",
        "#library-note-folder-dialog-cancel",
        _assert_none,
        str,
        _assert_exact("Folder"),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        LibraryNoteFolderTargetDialog,
        lambda: LibraryNoteFolderTargetDialog(
            title="Move note", folders=(), include_root=True
        ),
        "#library-note-folder-target-dialog",
        "#library-note-folder-target-cancel",
        _assert_none,
        str,
        _assert_exact(""),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        PromptCollectionManagerModal,
        _contract_prompt_collection_modal,
        "#prompt-collection-manager",
        "#prompt-collection-manager-cancel",
        _assert_none,
        PromptCollectionManagerResult,
        _assert_exact(PromptCollectionManagerResult("browse", 1, None, ())),
        "_mutation_in_flight",
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        FileNotesRootDetailsDialog,
        lambda: FileNotesRootDetailsDialog("/notes"),
        "#file-notes-root-details-dialog",
        "#file-notes-root-details-close",
        _assert_none,
        type(None),
        None,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        FileNotesConflictCompareDialog,
        _contract_conflict_compare_dialog,
        "#file-notes-conflict-dialog",
        "#file-notes-conflict-close",
        _assert_none,
        type(None),
        None,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        SessionGitTrustDialog,
        lambda: SessionGitTrustDialog("/notes"),
        "#confirmation-dialog",
        "#cancel-button",
        _assert_false,
        bool,
        _assert_exact(True),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        PushEndpointDetailsDialog,
        lambda: PushEndpointDetailsDialog(_contract_push_destination()),
        "#file-notes-push-endpoint-details-dialog",
        "#file-notes-push-endpoint-details-close",
        _assert_none,
        type(None),
        None,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        PushDestinationAuthorizationDialog,
        _contract_push_authorization_dialog,
        "#file-notes-push-auth-dialog",
        "#file-notes-push-auth-cancel",
        _assert_false,
        bool,
        _assert_exact(True),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        WorkbenchHelpPanel,
        lambda: WorkbenchHelpPanel(
            WorkbenchHelpState(route_id="library", title="Library")
        ),
        "#workbench-help-panel",
        "#workbench-help-close",
        _assert_none,
        type(None),
        None,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        PromptVariablesDialog,
        _prompt_variables_modal,
        "#prompt-variables-dialog",
        "#prompt-variables-cancel",
        _assert_none,
        PromptVariableApplication,
        _assert_prompt_variables_positive,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        ConfirmationDialog,
        ConfirmationDialog,
        "#confirmation-dialog",
        "#cancel-button",
        _assert_false,
        bool,
        _assert_exact(True),
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
    LibraryModalContract(
        WorkspaceCreateModal,
        lambda: WorkspaceCreateModal(
            registry_service=SimpleNamespace(
                list_workspaces=lambda **_kwargs: (),
            )
        ),
        "#workspace-create-modal",
        "#workspace-create-cancel",
        _assert_none,
        type(None),
        None,
        None,
        _FOCUS_POSTCONDITION,
        None,
    ),
)


ENHANCED_PICKER_COMPATIBILITY_TYPES = (EnhancedFileOpen, EnhancedFileSave)


_LIBRARY_SCREEN_FILE = "tldw_chatbook/UI/Screens/library_screen.py"
_COLLECTIONS_FILE = "tldw_chatbook/UI/Library_Modules/prompt_collections.py"
#: Wave-6 task 2 moved 139 prompt-cluster methods off `LibraryScreen` into
#: `LibraryPromptsController`; four of the edges below launch their modal
#: from a body that now lives there, so discovery has to parse that file
#: too or a repointed edge is simply never found (and the bidirectional
#: assertion fails the other way).
_PROMPTS_CONTROLLER_FILE = (
    "tldw_chatbook/UI/Library_Modules/library_prompts_controller.py"
)
_FILE_NOTES_WORKSPACE_FILE = (
    "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py"
)
_FILE_NOTES_GIT_FILE = "tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py"


@dataclass(frozen=True)
class _OwnerScope:
    owner_file: str
    owner_class: str


_SUPPORTED_OWNER_SCOPES = (
    _OwnerScope(_LIBRARY_SCREEN_FILE, "LibraryScreen"),
    _OwnerScope(_COLLECTIONS_FILE, "LibraryPromptCollectionsController"),
    _OwnerScope(_PROMPTS_CONTROLLER_FILE, "LibraryPromptsController"),
    _OwnerScope(_FILE_NOTES_WORKSPACE_FILE, "LibraryFileNotesWorkspace"),
    _OwnerScope(_FILE_NOTES_GIT_FILE, "LibraryFileNotesGitPanel"),
    _OwnerScope(_FILE_NOTES_GIT_FILE, "PushDestinationAuthorizationDialog"),
)


def _edge(
    owner_file: str,
    owner_class: str,
    presenter_name: str,
    concrete_type: type[ModalScreen[Any]],
) -> LibraryModalLaunchEdge:
    return LibraryModalLaunchEdge(
        owner_file, owner_class, presenter_name, concrete_type
    )


LIBRARY_MODAL_LAUNCH_EDGES = (
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "create_local_workspace",
        WorkspaceCreateModal,
    ),
    _edge(_LIBRARY_SCREEN_FILE, "LibraryScreen", "_export_library_note", FileSave),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_skills_import_browse",
        FileOpen,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_skills_import_browse_folder",
        SelectDirectory,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "action_show_workbench_help",
        WorkbenchHelpPanel,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_request_library_skill_trust_passphrase",
        SkillTrustPassphraseModal,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_request_library_skill_trust_bootstrap_passphrase",
        SkillTrustBootstrapModal,
    ),
    _edge(
        _PROMPTS_CONTROLLER_FILE,
        "LibraryPromptsController",
        "handle_library_prompts_import_browse",
        FileOpen,
    ),
    _edge(
        _PROMPTS_CONTROLLER_FILE,
        "LibraryPromptsController",
        "handle_library_prompt_history_restore",
        ConfirmationDialog,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_stage_library_prompt_for_console",
        PromptVariablesDialog,
    ),
    _edge(
        _PROMPTS_CONTROLLER_FILE,
        "LibraryPromptsController",
        "_export_library_prompt",
        FileSave,
    ),
    _edge(
        _PROMPTS_CONTROLLER_FILE,
        "LibraryPromptsController",
        "_open_library_prompt_delete_confirmation",
        PromptDeleteConfirmationModal,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_push_library_note_import_picker",
        FileOpen,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_notes_lasting_folder_requested",
        FileOpen,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_ingest_browse",
        FileOpen,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_ingest_directory_browse",
        SelectDirectory,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_open_transcribe_cpp_gguf_picker",
        FileOpen,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_apply_parakeet_v2_preflight_result",
        ModelInstallModal,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_apply_library_external_preparation",
        ModelInstallModal,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_export_choose_destination",
        FileSave,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_notes_folder_new",
        LibraryNoteFolderNameDialog,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_notes_folder_rename",
        LibraryNoteFolderNameDialog,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_notes_folder_move",
        LibraryNoteFolderTargetDialog,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "handle_library_notes_folder_remove",
        ConfirmationDialog,
    ),
    _edge(
        _LIBRARY_SCREEN_FILE,
        "LibraryScreen",
        "_choose_library_notes_placement_target",
        LibraryNoteFolderTargetDialog,
    ),
    _edge(
        _COLLECTIONS_FILE,
        "LibraryPromptCollectionsController",
        "open_manager",
        PromptCollectionManagerModal,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_open_push_authorization",
        PushDestinationAuthorizationDialog,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_open_session_git",
        SessionGitTrustDialog,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_choose_root",
        SelectDirectory,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_show_root_details",
        FileNotesRootDetailsDialog,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_session_git_push_endpoint_details",
        PushEndpointDetailsDialog,
    ),
    _edge(
        _FILE_NOTES_WORKSPACE_FILE,
        "LibraryFileNotesWorkspace",
        "_compare_conflict",
        FileNotesConflictCompareDialog,
    ),
    _edge(
        _FILE_NOTES_GIT_FILE,
        "PushDestinationAuthorizationDialog",
        "_details_pressed",
        PushEndpointDetailsDialog,
    ),
)


_MODAL_TYPES_BY_NAME = {
    contract.concrete_type.__name__: contract.concrete_type
    for contract in LIBRARY_MODAL_CONTRACTS
}


def _import_aliases(
    tree: ast.Module, presenter: ast.FunctionDef | ast.AsyncFunctionDef
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    top_level_imports = (
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    for node in (*top_level_imports, *ast.walk(presenter)):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                aliases[imported.asname or imported.name] = imported.name
    return aliases


def _expression_name(expression: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        return aliases.get(expression.attr, expression.attr)
    if isinstance(expression, ast.Call):
        return _expression_name(expression.func, aliases)
    return None


def _is_modal_presenter(call: ast.Call) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id in {"push_screen", "push_screen_wait"}
    if isinstance(function, ast.Attribute):
        return function.attr in {"push_screen", "push_screen_wait"}
    return (
        isinstance(function, ast.Call)
        and isinstance(function.func, ast.Attribute)
        and function.func.attr == "_push_modal"
    )


def _modal_argument_type(
    expression: ast.expr,
    *,
    aliases: dict[str, str],
    assignments: dict[str, ast.expr],
) -> type[ModalScreen[Any]] | None:
    if isinstance(expression, ast.Name) and expression.id in assignments:
        return _modal_argument_type(
            assignments[expression.id], aliases=aliases, assignments=assignments
        )
    name = _expression_name(expression, aliases)
    return _MODAL_TYPES_BY_NAME.get(name or "")


def _discover_library_modal_edges(
    sources: dict[str, str],
    scopes: tuple[_OwnerScope, ...] = _SUPPORTED_OWNER_SCOPES,
) -> set[LibraryModalLaunchEdge]:
    """Resolve modal constructors only inside the explicitly supported owners."""
    discovered: set[LibraryModalLaunchEdge] = set()
    for scope in scopes:
        tree = ast.parse(sources[scope.owner_file])
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == scope.owner_class
        )
        methods = (
            node
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for presenter in methods:
            presenter_name = presenter.name
            aliases = _import_aliases(tree, presenter)
            assignments = {
                target.id: node.value
                for node in ast.walk(presenter)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and (
                    targets := (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                )
                for target in targets
                if isinstance(target, ast.Name) and node.value is not None
            }
            for call in (
                node for node in ast.walk(presenter) if isinstance(node, ast.Call)
            ):
                if not call.args or not _is_modal_presenter(call):
                    continue
                concrete_type = _modal_argument_type(
                    call.args[0], aliases=aliases, assignments=assignments
                )
                assert concrete_type is not None, (
                    "unresolved modal constructor in supported presenter: "
                    f"{scope.owner_file}:{scope.owner_class}.{presenter_name} "
                    f"({ast.unparse(call.args[0])})"
                )
                discovered.add(
                    _edge(
                        scope.owner_file,
                        scope.owner_class,
                        presenter_name,
                        concrete_type,
                    )
                )
    return discovered


def _assert_exact_library_modal_inventory(
    discovered: set[LibraryModalLaunchEdge],
    declared: set[LibraryModalLaunchEdge],
) -> None:
    undeclared = discovered - declared
    missing = declared - discovered
    assert not undeclared and not missing, (
        f"undeclared Library modal edges: {sorted(map(repr, undeclared))}; "
        f"missing Library modal edges: {sorted(map(repr, missing))}"
    )


ORDINARY_LIBRARY_MODAL_CONTRACTS = (
    (SkillTrustPassphraseModal, "#skill-trust-passphrase-modal"),
    (SkillTrustBootstrapModal, "#skill-trust-bootstrap-modal"),
    (ModelInstallModal, ".model-install-modal"),
    (PromptDeleteConfirmationModal, "#prompt-delete-modal"),
    (LibraryNoteFolderNameDialog, "#library-note-folder-name-dialog"),
    (LibraryNoteFolderTargetDialog, "#library-note-folder-target-dialog"),
    (WorkspaceCreateModal, "#workspace-create-modal"),
)


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
        _contract_conflict_compare_dialog,
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
        lambda: PushEndpointDetailsDialog(_contract_push_destination()),
        "#file-notes-push-endpoint-details-dialog",
        "#file-notes-push-endpoint-details-close",
        None,
    ),
    _FileNotesModalContract(
        "authorization",
        PushDestinationAuthorizationDialog,
        _contract_push_authorization_dialog,
        "#file-notes-push-auth-dialog",
        "#file-notes-push-auth-cancel",
        False,
        "#file-notes-push-auth-confirm",
    ),
)


class _FileNotesModalHarness(ConsolidatedCSSApp):
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
    authorization = _contract_push_authorization_dialog()
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
    assert len(ORDINARY_LIBRARY_MODAL_CONTRACTS) == 7
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
    modal = _contract_prompt_collection_modal()

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


class _CompetingFocusModal(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Input(id=_STABLE_OPENER_ID)
        yield Input(id="new-modal-current-focus")


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


@pytest.mark.asyncio
async def test_stable_focus_ignores_a_callback_after_another_modal_is_pushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _LibraryFocusHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal, original = await _mount_focus_modal(app, pilot)
        original_call_after_refresh = app.host.call_after_refresh
        deferred: list[
            tuple[Callable[..., object], tuple[object, ...], dict[str, object]]
        ] = []

        def delay_focus_restoration(
            callback: Callable[..., object],
            *args: object,
            **kwargs: object,
        ) -> bool:
            if callback.__name__ == "_restore_focus_after_dismissal":
                deferred.append((callback, args, kwargs))
                return True
            return original_call_after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(
            app.host,
            "call_after_refresh",
            delay_focus_restoration,
        )
        await modal.action_request_safe_cancel()
        await pilot.pause()
        assert app.screen is app.host
        assert len(deferred) == 1

        original.disabled = True
        competing_modal = _CompetingFocusModal()
        app.push_screen(competing_modal)
        await pilot.pause()
        current_focus = competing_modal.query_one("#new-modal-current-focus", Input)
        current_focus.focus()
        await pilot.pause()
        assert competing_modal.focused is current_focus

        callback, args, kwargs = deferred.pop()
        callback(*args, **kwargs)
        await pilot.pause()

        assert app.screen is competing_modal
        assert competing_modal.focused is current_focus


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


def _production_owner_sources() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    return {
        scope.owner_file: (repository_root / scope.owner_file).read_text()
        for scope in _SUPPORTED_OWNER_SCOPES
    }


class _LibraryContractHarness(_FileNotesModalHarness):
    CSS_PATH = TldwCli.CSS_PATH


def _assert_contract_factory_exact(
    contract: LibraryModalContract,
) -> ModalScreen[Any]:
    modal = contract.factory()
    assert type(modal) is contract.concrete_type, (
        f"factory returned {type(modal).__name__}, expected "
        f"exactly {contract.concrete_type.__name__}"
    )
    return modal


def test_library_modal_contract_table_covers_every_discovered_concrete_type() -> None:
    contract_types = [row.concrete_type for row in LIBRARY_MODAL_CONTRACTS]
    edge_types = {edge.concrete_type for edge in LIBRARY_MODAL_LAUNCH_EDGES}

    assert len(contract_types) == len(set(contract_types)) == 19
    assert edge_types == set(contract_types)
    assert set(ENHANCED_PICKER_COMPATIBILITY_TYPES).isdisjoint(contract_types)
    assert {SessionGitTrustDialog}.issubset(contract_types)
    assert all(
        issubclass(row.concrete_type, SafeModalDismissMixin)
        for row in LIBRARY_MODAL_CONTRACTS
    )
    assert all(
        row.concrete_type.SAFE_MODAL_CONTENT == row.content_selector
        for row in LIBRARY_MODAL_CONTRACTS
    )
    assert {
        row.concrete_type: row.active_guard
        for row in LIBRARY_MODAL_CONTRACTS
        if row.active_guard is not None
    } == {PromptCollectionManagerModal: "_mutation_in_flight"}
    assert all(
        row.focus_postcondition == _FOCUS_POSTCONDITION
        for row in LIBRARY_MODAL_CONTRACTS
    )
    assert all(row.non_dismissible_reason is None for row in LIBRARY_MODAL_CONTRACTS)


def test_library_modal_contract_table_rejects_factory_returning_sibling_type() -> None:
    contract = next(
        row
        for row in LIBRARY_MODAL_CONTRACTS
        if row.concrete_type is ConfirmationDialog
    )
    mutated = replace(
        contract,
        factory=lambda: SessionGitTrustDialog("/notes"),
    )

    with pytest.raises(AssertionError, match="factory returned"):
        _assert_contract_factory_exact(mutated)


def test_library_modal_contract_table_positive_oracle_rejects_negative_result() -> None:
    contract = next(
        row for row in LIBRARY_MODAL_CONTRACTS if row.concrete_type is ModelInstallModal
    )

    assert contract.positive_assertion is not None
    with pytest.raises(AssertionError):
        contract.positive_assertion(False)


def test_library_modal_contract_harness_uses_exact_production_css_stack() -> None:
    assert _LibraryContractHarness.CSS_PATH == TldwCli.CSS_PATH


def test_library_modal_inventory_matches_declared_edges_bidirectionally() -> None:
    discovered = _discover_library_modal_edges(_production_owner_sources())
    declared = set(LIBRARY_MODAL_LAUNCH_EDGES)

    assert len(discovered) == len(declared) == 33
    _assert_exact_library_modal_inventory(discovered, declared)


def test_library_modal_inventory_controller_route_uses_app_push_screen() -> None:
    tree = ast.parse(_production_owner_sources()[_LIBRARY_SCREEN_FILE])
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LibraryScreen"
    )
    initializer = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    controller_call = next(
        node
        for node in ast.walk(initializer)
        if isinstance(node, ast.Call)
        and _expression_name(node.func, {}) == "LibraryPromptCollectionsController"
    )
    push_modal = next(
        keyword.value
        for keyword in controller_call.keywords
        if keyword.arg == "push_modal"
    )

    assert isinstance(push_modal, ast.Lambda)
    assert ast.unparse(push_modal.body) == "self.app.push_screen"


def _assert_synthetic_edge_is_rejected(
    *, source: str, scope: _OwnerScope, expected_type: type[ModalScreen[Any]]
) -> None:
    discovered = _discover_library_modal_edges(
        {scope.owner_file: source}, scopes=(scope,)
    )
    assert {edge.concrete_type for edge in discovered} == {expected_type}
    with pytest.raises(AssertionError, match="undeclared Library modal edges"):
        _assert_exact_library_modal_inventory(discovered, set())


def _production_owner_scope(owner_class: str) -> _OwnerScope:
    return next(
        scope for scope in _SUPPORTED_OWNER_SCOPES if scope.owner_class == owner_class
    )


def test_library_modal_inventory_detects_controller_injected_edge() -> None:
    scope = _production_owner_scope("LibraryPromptCollectionsController")
    _assert_synthetic_edge_is_rejected(
        source="""
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog as Hidden
class LibraryPromptCollectionsController:
    def open_manager(self):
        pass
    def _unexpected_presenter(self):
        self._push_modal()(Hidden())
""",
        scope=scope,
        expected_type=ConfirmationDialog,
    )


def test_library_modal_inventory_detects_nested_edge() -> None:
    scope = _production_owner_scope("LibraryFileNotesGitPanel")
    _assert_synthetic_edge_is_rejected(
        source="""
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog as Hidden
class LibraryFileNotesGitPanel:
    def _unexpected_presenter(self):
        self.app.push_screen(Hidden())
""",
        scope=scope,
        expected_type=ConfirmationDialog,
    )


def test_library_modal_inventory_detects_modal_to_modal_edge() -> None:
    scope = _production_owner_scope("PushDestinationAuthorizationDialog")
    _assert_synthetic_edge_is_rejected(
        source="""
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog as Hidden
class PushDestinationAuthorizationDialog:
    def _details_pressed(self):
        pass
    def _unexpected_presenter(self):
        self.app.push_screen(Hidden())
""",
        scope=scope,
        expected_type=ConfirmationDialog,
    )


@pytest.mark.parametrize(
    "contract",
    LIBRARY_MODAL_CONTRACTS,
    ids=lambda row: row.concrete_type.__name__,
)
@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_concrete_library_modal_exact_negative_for_every_gesture(
    contract: LibraryModalContract,
    source: str,
) -> None:
    app = _LibraryContractHarness()
    modal = _assert_contract_factory_exact(contract)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        assert modal.query_one(contract.content_selector).is_mounted
        if source == "visible":
            await pilot.click(contract.visible_negative_selector)
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0), button=1)
        await pilot.pause()

    assert len(app.results) == 1
    contract.negative_assertion(app.results[0])


@pytest.mark.parametrize(
    "contract",
    LIBRARY_MODAL_CONTRACTS,
    ids=lambda row: row.concrete_type.__name__,
)
@pytest.mark.asyncio
async def test_concrete_library_modal_inside_and_non_primary_clicks_stay_open(
    contract: LibraryModalContract,
) -> None:
    app = _LibraryContractHarness()
    modal = _assert_contract_factory_exact(contract)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        await pilot.click(contract.content_selector, offset=(1, 1), button=1)
        await pilot.click(offset=(0, 0), button=3)
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()


async def _drive_public_positive(modal: ModalScreen[Any], pilot: Any) -> None:
    if isinstance(modal, SkillTrustPassphraseModal):
        modal.query_one("#skill-trust-passphrase-input", Input).value = "secret"
        await pilot.click("#skill-trust-passphrase-submit")
    elif isinstance(modal, SkillTrustBootstrapModal):
        modal.query_one("#skill-trust-bootstrap-input", Input).value = "secret"
        modal.query_one("#skill-trust-bootstrap-confirm-input", Input).value = "secret"
        await pilot.click("#skill-trust-bootstrap-submit")
    elif isinstance(modal, ModelInstallModal):
        await pilot.click("#model-install-confirm")
    elif isinstance(modal, (FileOpen, FileSave, SelectDirectory)):
        await pilot.click("#select")
    elif isinstance(modal, PromptDeleteConfirmationModal):
        await pilot.click("#prompt-delete-confirm")
    elif isinstance(modal, LibraryNoteFolderNameDialog):
        modal.query_one("#library-note-folder-name", Input).value = "Folder"
        await pilot.click("#library-note-folder-dialog-confirm")
    elif isinstance(modal, LibraryNoteFolderTargetDialog):
        await pilot.click("#library-note-folder-target-confirm")
    elif isinstance(modal, PromptCollectionManagerModal):
        await pilot.click("#prompt-collection-manager-done")
    elif isinstance(modal, PushDestinationAuthorizationDialog):
        await pilot.click("#file-notes-push-auth-confirm")
    elif isinstance(modal, PromptVariablesDialog):
        await pilot.click("#prompt-variables-original")
    elif isinstance(modal, ConfirmationDialog):
        await pilot.click("#confirm-button")
    else:
        raise AssertionError(f"No feasible positive action for {type(modal).__name__}")


@pytest.mark.parametrize(
    "contract",
    [row for row in LIBRARY_MODAL_CONTRACTS if row.positive_assertion is not None],
    ids=lambda row: row.concrete_type.__name__,
)
@pytest.mark.asyncio
async def test_concrete_library_modal_public_positive_result_type(
    contract: LibraryModalContract,
) -> None:
    app = _LibraryContractHarness()
    modal = _assert_contract_factory_exact(contract)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        await _drive_public_positive(modal, pilot)
        await pilot.pause()

    assert len(app.results) == 1
    assert contract.positive_assertion is not None
    contract.positive_assertion(app.results[0])


@pytest.mark.parametrize(
    "contract",
    LIBRARY_MODAL_CONTRACTS,
    ids=lambda row: row.concrete_type.__name__,
)
@pytest.mark.asyncio
async def test_library_modal_lifecycle_runs_shared_handlers_exactly_once(
    contract: LibraryModalContract,
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
    app = _LibraryContractHarness()
    modal = _assert_contract_factory_exact(contract)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        assert mount_calls == 1
        await pilot.press("escape")
        await pilot.pause()
        assert unmount_calls == 1
