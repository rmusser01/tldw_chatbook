"""Thin canonical backup view over the application's recovery service."""

from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Input, Select, Static

from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    FileSave,
    SelectDirectory,
)

from .backup_restore_state import result_label


class BackupRestoreScreen(Screen):
    """Navigation releases widgets, while the app retains accepted native work."""

    BINDINGS = (Binding("escape", "back", "Back"),)
    DEFAULT_CSS = """
    BackupRestoreScreen { background: $background; }
    BackupRestoreScreen #backup-title { height: auto; padding: 1 2; text-style: bold; }
    BackupRestoreScreen #backup-actions { layout: grid; grid-size: 2; grid-columns: 1fr 1fr; height: 6; padding: 0 1; }
    BackupRestoreScreen #backup-actions Button { width: 100%; }
    BackupRestoreScreen #backup-status { height: auto; max-height: 5; padding: 0 2; }
    BackupRestoreScreen #backup-body { height: 1fr; padding: 1 2; }
    BackupRestoreScreen .backup-form { height: auto; }
    BackupRestoreScreen .backup-form Static { height: auto; margin-top: 1; }
    BackupRestoreScreen .backup-form Input { width: 100%; }
    BackupRestoreScreen .backup-form Checkbox { height: auto; width: 100%; }
    BackupRestoreScreen #backup-message { height: auto; padding: 0 2; max-height: 4; }
    BackupRestoreScreen #backup-footer-actions { height: auto; padding: 0 1; }
    BackupRestoreScreen #backup-footer-actions Button { min-width: 10; width: 1fr; }
    """

    def __init__(self, service, *, config_paths=(), include_known_profiles=False, restart_request=None):
        super().__init__()
        self.service = service
        self.config_paths = tuple(Path(path) for path in config_paths)
        self.include_known_profiles = include_known_profiles
        self._restart_request = restart_request
        self.external_roots = ()
        self._mode = "home"
        self._revision = 0
        self._preview = None
        self._reviewed = None
        self._poller = None
        self._summary_requested = None
        self._inspection_id = None
        self._dismissed_inspection_id = None
        self._inspection_summary = None
        self._restore_plan = None
        self._extraction_plan = None
        self._delete_copy_id = None
        self._last_terminal = None
        self._review_codes_seen = ()
        self._requested_backup_operation = None
        self._requested_restore_operation = None
        self._restore_review_codes_seen = ()
        self._safety_scope_seen = ()
        self._rollback_copy_id = None
        self._rollback_plan = None
        self._requested_rollback_operation = None
        self._rollback_selection = None
        self._later_review_codes_seen = ()
        self._media_review = None
        self._media_offset = 0

    def compose(self) -> ComposeResult:
        yield Static("Backup & Restore", id="backup-title")
        with Container(id="backup-actions"):
            yield Button("Create backup", id="backup-open-create")
            yield Button("Inspect / restore", id="backup-open-inspect")
            yield Button("Recovery copies", id="backup-open-copies")
            yield Button("Restored profiles", id="backup-open-profiles")
        yield Static("No operation started", id="backup-status", markup=False)
        with VerticalScroll(id="backup-body"):
            yield Static(
                "Create a verified local backup, or inspect an archive before choosing where to restore it.",
                id="backup-home",
                markup=False,
            )
            with Vertical(id="backup-create-form", classes="backup-form"):
                yield Static("Profile configurations", classes="destination-section")
                yield Static(self._profile_text(), id="backup-profiles", markup=False)
                yield Button("Add profile configuration", id="backup-add-profile")
                yield Static("New backup file")
                yield Input(placeholder="Choose a new file", id="backup-destination")
                yield Button("Choose output file", id="backup-pick-destination")
                yield Static("Optional additions", classes="destination-section")
                yield Static(
                    "No external folders selected",
                    id="backup-external-roots",
                    markup=False,
                )
                yield Button("Add external folder", id="backup-add-external")
                yield Input(
                    placeholder="Optional model IDs, separated by commas",
                    id="backup-models",
                )
                yield Checkbox("Include temporary media", id="backup-temporary")
                yield Checkbox("Include diagnostic history", id="backup-diagnostics")
                yield Checkbox(
                    "Acknowledge partial coverage if sources are unavailable",
                    id="backup-partial",
                )
                yield Checkbox(
                    "Include supported credentials (requires encryption)",
                    id="backup-credentials",
                )
                yield Checkbox("Encrypt this backup", id="backup-encrypted")
                yield Input(placeholder="Password", password=True, id="backup-password")
                yield Input(
                    placeholder="Confirm password",
                    password=True,
                    id="backup-password-confirm",
                )
                yield Static(
                    "Review coverage before creating the backup.",
                    id="backup-coverage",
                    markup=False,
                )
                yield Vertical(id="backup-credential-review", classes="backup-form")
            with Vertical(id="backup-inspect-form", classes="backup-form"):
                yield Static("Backup archive", classes="destination-section")
                yield Input(placeholder="Choose an archive", id="backup-source")
                yield Button("Choose archive", id="backup-pick-source")
                yield Input(
                    placeholder="Archive password, if encrypted",
                    password=True,
                    id="backup-inspect-password",
                )
                yield Static(
                    "Inspect first. Restore controls require a verified archive and a reviewed local destination plan.",
                    id="backup-inspection-summary",
                    markup=False,
                )
                with Vertical(id="backup-restore-form", classes="backup-form"):
                    yield Select(
                        [
                            ("Restore as isolated profile", "isolated"),
                            ("Replace selected stored data", "replace"),
                        ],
                        value="isolated",
                        allow_blank=False,
                        id="backup-restore-mode",
                    )
                    yield Static(
                        "Replacement continues after Chatbook closes. Review the archive and destinations again in recovery mode.",
                        id="backup-restart-note",
                    )
                    yield Button("Continue in recovery mode", id="backup-restart")
                    yield Static(
                        "Choose local destination roots. Archived paths are never defaults."
                    )
                    yield Vertical(id="backup-destination-slots", classes="backup-form")
                    yield Static(
                        "For replacement: selected existing profile configuration"
                    )
                    yield Input(
                        placeholder="Existing local config.toml",
                        id="backup-target-config",
                    )
                    yield Input(
                        placeholder="Replacement rollback password",
                        password=True,
                        id="backup-rollback-password",
                    )
                    yield Input(
                        placeholder="Confirm rollback password",
                        password=True,
                        id="backup-rollback-confirm",
                    )
                    yield Vertical(id="backup-restore-credential-review", classes="backup-form")
                    yield Vertical(id="backup-safety-scope", classes="backup-form")
                    yield Static(
                        "Review the actual restore, retirement and preservation plan.",
                        id="backup-restore-preview",
                        markup=False,
                    )
                    yield Button("Review restore", id="backup-review-restore")
                    yield Button(
                        "Confirm reviewed restore",
                        id="backup-start-restore",
                        variant="warning",
                        disabled=True,
                    )
                with Vertical(id="backup-inert-form", classes="backup-form"):
                    yield Static(
                        "Manual extraction: copy selected groups as inert files. No profile is restored or opened."
                    )
                    yield Vertical(id="backup-inert-groups", classes="backup-form")
                    yield Input(
                        placeholder="New absolute directory for extracted files",
                        id="backup-inert-destination",
                    )
                    yield Static(
                        "Select groups and review the extraction first.",
                        id="backup-inert-preview",
                        markup=False,
                    )
                    yield Button("Review extraction", id="backup-review-extraction")
                    yield Button(
                        "Confirm inert extraction",
                        id="backup-start-extraction",
                        variant="warning",
                        disabled=True,
                    )
            with Vertical(id="backup-dependent", classes="backup-form"):
                yield Static("", id="backup-list-title", markup=False)
                yield Input(
                    placeholder="Recovery archive password, if required",
                    password=True,
                    id="backup-copy-password",
                )
                yield Vertical(id="backup-list", classes="backup-form")
                yield Static("", id="backup-delete-preview", markup=False)
                yield Checkbox(
                    "I understand deletion removes this later-rollback option",
                    id="backup-delete-confirm",
                )
                yield Button(
                    "Delete selected recovery copy",
                    id="backup-delete-copy",
                    variant="error",
                    disabled=True,
                )
                with Vertical(id="backup-later-form", classes="backup-form"):
                    yield Static(
                        "Later rollback replaces changes made since the selected recovery copy. A new verified encrypted recovery copy preserves those current changes first.",
                        markup=False,
                    )
                    yield Input(
                        str(self.config_paths[0])
                        if len(self.config_paths) == 1
                        else "",
                        placeholder="Current profile configuration to review",
                        id="backup-later-target",
                    )
                    yield Button("Review later rollback", id="backup-later-review")
                    yield Static("", id="backup-later-preview", markup=False)
                    yield Vertical(id="backup-later-credential-review")
                    yield Input(
                        placeholder="New safety-copy password",
                        password=True,
                        id="backup-safety-password",
                    )
                    yield Input(
                        placeholder="Repeat new safety-copy password",
                        password=True,
                        id="backup-safety-confirm",
                    )
                    yield Checkbox(
                        "I reviewed the affected data and understand current changes will be replaced",
                        id="backup-later-confirm",
                    )
                    yield Button(
                        "Confirm reviewed rollback",
                        id="backup-later-start",
                        variant="warning",
                        disabled=True,
                    )
            with Vertical(id="backup-media-form", classes="backup-form"):
                yield Static("Recovered media in the current profile", markup=False)
                yield Static(
                    "Recorded sizes remain visible after deletion. References from earlier restores or other profiles can prevent cleanup.",
                    markup=False,
                )
                yield Static("", id="backup-media-summary", markup=False)
                yield Vertical(id="backup-media-list", classes="backup-form")
                with Horizontal():
                    yield Button("Previous page", id="backup-media-previous", disabled=True)
                    yield Button("Next page", id="backup-media-next", disabled=True)
                yield Static("", id="backup-media-review", markup=False)
                yield Checkbox("I reviewed every affected reference and hold", id="backup-media-confirm")
                yield Button("Delete reviewed asset", id="backup-media-delete", variant="error", disabled=True)
                yield Button("Clean up reviewed orphan", id="backup-media-cleanup", variant="warning", disabled=True)
                yield Button("Cancel review", id="backup-media-cancel")
        yield Static("", id="backup-message", markup=False)
        with Horizontal(id="backup-footer-actions"):
            yield Button("Review", id="backup-review")
            yield Button(
                "Create backup", id="backup-create", variant="primary", disabled=True
            )
            yield Button("Inspect", id="backup-inspect", variant="primary")
            yield Button("Cancel operation", id="backup-cancel", disabled=True)
            yield Button("Back", id="backup-back")
        yield Footer()

    def _profile_text(self):
        if self.include_known_profiles:
            return "All known local profiles, plus selected configurations:\n" + "\n".join(
                str(path) for path in self.config_paths
            )
        return (
            "\n".join(str(path) for path in self.config_paths)
            or "Add a profile configuration."
        )

    def on_mount(self):
        self._show_mode("home")
        if self._restart_request is not None:
            self._show_mode("inspect")
            if self._restart_request.archive is not None:
                self.query_one("#backup-source", Input).value = str(self._restart_request.archive)
            self.query_one("#backup-target-config", Input).value = str(self._restart_request.target_config)
            self.query_one("#backup-restore-mode", Select).value = "replace"
        self._sync_replacement_host()
        self._poller = self.set_interval(0.2, self._refresh_status)
        self._refresh_status()

    def on_unmount(self):
        self._revision += 1
        if self._poller is not None:
            self._poller.stop()
        self._clear_passwords()

    def _clear_passwords(self):
        for field in (
            "backup-password",
            "backup-password-confirm",
            "backup-inspect-password",
            "backup-rollback-password",
            "backup-rollback-confirm",
            "backup-copy-password",
            "backup-safety-password",
            "backup-safety-confirm",
        ):
            for widget in self.query("#" + field):
                widget.value = ""

    def action_back(self):
        self._clear_passwords()
        self.app.pop_screen()

    @on(Button.Pressed, "#backup-back")
    def _back(self):
        self.action_back()

    def _show_mode(self, mode):
        self._mode = mode
        self._revision += 1
        self._preview = self._reviewed = None
        self._media_review = None
        self.query_one("#backup-media-confirm", Checkbox).value = False
        self.query_one("#backup-media-form").display = mode == "media"
        self.query_one("#backup-home").display = mode == "home"
        self.query_one("#backup-create-form").display = mode == "create"
        self.query_one("#backup-inspect-form").display = mode == "inspect"
        self.query_one("#backup-restore-form").display = (
            mode == "inspect" and self._inspection_id is not None
        )
        self.query_one("#backup-inert-form").display = (
            mode == "inspect" and self._inspection_id is not None
        )
        self.query_one("#backup-later-form").display = False
        self._rollback_copy_id = self._rollback_plan = None
        self._forget_later_credential_review()
        for identifier, visible in (
            ("backup-review", mode == "create"),
            ("backup-create", mode == "create"),
            ("backup-inspect", mode == "inspect"),
        ):
            self.query_one("#" + identifier).display = visible
        self.query_one("#backup-create", Button).disabled = True
        self.query_one("#backup-dependent").display = mode in ("copies", "profiles")
        self.query_one("#backup-message", Static).update("")
        self._clear_passwords()

    @on(Button.Pressed, "#backup-open-create")
    def _open_create(self):
        self._show_mode("create")

    @on(Button.Pressed, "#backup-open-inspect")
    def _open_inspect(self):
        self._show_mode("inspect")

    @on(Button.Pressed, "#backup-open-copies")
    def _open_copies(self):
        self._show_mode("copies")
        self._refresh_list(self.app, self._revision, "copies")

    @on(Button.Pressed, "#backup-open-profiles")
    def _open_profiles(self):
        self._show_mode("profiles")
        self._refresh_list(self.app, self._revision, "profiles")

    @on(Button.Pressed, "#backup-open-media")
    def _open_media(self):
        self._show_mode("media")
        self._media_offset = 0
        self._load_media(self.app, self._revision, 0)

    @work(exclusive=True, thread=True, group="backup-media")
    def _load_media(self, app, revision, offset):
        try:
            page = self.service.recovered_media_details(limit=20, offset=offset)
        except (OSError, ValueError, RuntimeError):
            page = None
        self._deliver(app, self._media_ready, revision, page)

    async def _media_ready(self, revision, page):
        if not self.is_mounted or self._mode != "media" or revision != self._revision:
            return
        self._clear_media_review()
        listing = self.query_one("#backup-media-list", Vertical)
        await listing.remove_children()
        details = page["details"] if page is not None else None
        text = "Recovered media unavailable. Recheck the selected profile and local recovery state."
        if page is not None:
            text = "Current profile: " + page["config"] + "\n"
            text += (
                "No recovered-media catalog."
                if details is None
                else (
                    f"Showing {details.offset + 1}–{details.offset + len(details.assets)} of {details.total} assets"
                    if details.assets
                    else "No assets on this page."
                )
            )
        self.query_one("#backup-media-summary", Static).update(text)
        self.query_one("#backup-media-previous", Button).disabled = (
            details is None or not details.offset
        )
        self.query_one("#backup-media-next", Button).disabled = (
            details is None or not details.has_more
        )
        if details is not None:
            for asset in details.assets:
                await listing.mount(
                    Static(
                        f"{asset.asset_id}\nCatalog state: {asset.state} · {asset.size} recorded bytes · {asset.media_type}\n{asset.reference_count} references · {asset.hold_count} recovery holds",
                        markup=False,
                    ),
                    Button(
                        "Review asset details",
                        name=asset.asset_id,
                        classes="backup-review-media",
                    ),
                )

    @on(Button.Pressed, "#backup-media-previous")
    @on(Button.Pressed, "#backup-media-next")
    def _media_page(self, event):
        self._media_offset = max(
            0,
            self._media_offset
            + (20 if event.button.id == "backup-media-next" else -20),
        )
        self._revision += 1
        self._clear_media_review()
        self._load_media(self.app, self._revision, self._media_offset)

    @on(Button.Pressed, ".backup-review-media")
    def _review_media(self, event):
        self._revision += 1
        self._clear_media_review()
        self._load_media_review(self.app, self._revision, event.button.name)

    @work(exclusive=True, thread=True, group="backup-media-review")
    def _load_media_review(self, app, revision, asset_id):
        try:
            review = self.service.review_recovered_media(asset_id)
        except (OSError, ValueError, RuntimeError, KeyError):
            review = None
        self._deliver(app, self._media_review_ready, revision, review)

    def _media_review_ready(self, revision, review):
        if not self.is_mounted or self._mode != "media" or revision != self._revision:
            return
        self._media_review = review
        text = "Asset review unavailable; refresh current-profile details."
        if review is not None:
            owner = review.asset
            asset = owner.asset
            refs = "\n".join(
                f"Profile: {profile} · message: {message} · slug: {slug} · type: {kind}"
                for profile, message, slug, kind in owner.references
            )
            holds = "\n".join(owner.holds)
            text = (
                f"Current profile: {review.source.config}\nAsset: {asset.asset_id}\nCatalog state: {asset.state} · {asset.size} recorded bytes\n"
                "Deletion removes the payload and keeps deleted-reference history; all listed aliases are affected. Recovery holds block deletion. Orphan cleanup requires zero references and holds, including historical aliases.\n"
                "Affected references (including historical/other profiles):\n"
                + (refs or "None")
                + "\nRecovery holds:\n"
                + (holds or "None")
            )
        self.query_one("#backup-media-review", Static).update(text)
        self._sync_media_actions()

    def _clear_media_review(self):
        self._media_review = None
        self.query_one("#backup-media-confirm", Checkbox).value = False
        self.query_one("#backup-media-review", Static).update("")
        self._sync_media_actions()

    @on(Button.Pressed, "#backup-media-cancel")
    def _cancel_media_review(self):
        self._revision += 1
        self._clear_media_review()

    @on(Checkbox.Changed, "#backup-media-confirm")
    def _sync_media_actions(self):
        review = self._media_review
        ready = review is not None and review.asset.asset.state == "ready"
        permitted = ready and not review.asset.holds
        confirmed = self.query_one("#backup-media-confirm", Checkbox).value
        for identifier, eligible in (
            ("backup-media-delete", permitted),
            ("backup-media-cleanup", ready and review.asset.asset.orphan_eligible),
        ):
            button = self.query_one("#" + identifier, Button)
            button.display = bool(eligible)
            button.disabled = not (eligible and confirmed)

    @on(Button.Pressed, "#backup-media-delete")
    @on(Button.Pressed, "#backup-media-cleanup")
    def _mutate_media(self, event):
        if (
            self._media_review is None
            or not self.query_one("#backup-media-confirm", Checkbox).value
        ):
            return
        try:
            self.service.start_recovered_media_action(
                self._media_review,
                action="delete"
                if event.button.id == "backup-media-delete"
                else "cleanup",
                user_selected=True,
            )
        except (ValueError, RuntimeError):
            self.query_one("#backup-message", Static).update(
                "Asset action unavailable; review current details again."
            )
        self._revision += 1
        self._clear_media_review()
        self._refresh_status()

    @work(exclusive=True, thread=True, group="backup-list")
    def _refresh_list(self, app, revision, mode):
        current = self.service.current_requirements() if mode == "profiles" else None
        try:
            entries = (
                self.service.recovery_copies()
                if mode == "copies"
                else self.service.profiles()
            )
            pending = (
                tuple(
                    (row, self.service.status(row["operation_id"]))
                    for row in self.service.pending_operations()
                )
                if mode == "copies"
                else ()
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._deliver(
                app,
                self._list_ready,
                revision,
                mode,
                (),
                (),
                self.service.issue_code(error),
                current,
            )
        else:
            self._deliver(app, self._list_ready, revision, mode, entries, pending, None, current)

    async def _list_ready(self, revision, mode, entries, pending, issue, current=None):
        if not self.is_mounted or mode != self._mode or revision != self._revision:
            return
        self.query_one("#backup-list-title", Static).update(
            ("Recovery copies" if mode == "copies" else "Restored profiles")
            + (": " + issue if issue else "")
        )
        for identifier in (
            "backup-copy-password",
            "backup-delete-confirm",
            "backup-delete-copy",
        ):
            self.query_one("#" + identifier).display = mode == "copies"
        self._delete_copy_id = None
        self.query_one("#backup-delete-copy", Button).disabled = True
        self.query_one("#backup-delete-preview", Static).update("")
        listing = self.query_one("#backup-list", Vertical)
        await listing.remove_children()
        if current is not None:
            if current["status"] == "no_verified_generation":
                setup = "No verified recovery generation for this current profile. Optional features may still require setup."
            elif not current["requirements_checked"]:
                setup = "Setup requirements unavailable. Local recovery evidence needs attention."
            elif current["needs_setup"]:
                setup = "Needs setup\nReview these features in their settings before enabling them:\n" + "\n".join(
                    "• " + owner.replace(".", " / ").replace("_", " ")
                    for owner in current["pending_owners"]
                )
            else:
                setup = "Owner reviews complete. Optional features may still require setup."
            generation = "\nGeneration: " + current["generation"] if current["generation"] else ""
            await listing.mount(Static(
                f"Current profile\n{current['config']}{generation}\n{setup}",
                id="backup-current-requirements", markup=False,
            ))
            await listing.mount(Button("Recovered media details", id="backup-open-media"))
        if not entries and not pending and current is None:
            await listing.mount(Static("No local entries.", markup=False))
        for entry in entries:
            if mode == "copies":
                await listing.mount(
                    Static(
                        f"{entry.operation_id}\n{entry.status} · {entry.size} bytes\n{entry.path or 'No retained payload'}\nCoverage: {entry.coverage}",
                        markup=False,
                    ),
                    Button(
                        "Inspect recovery copy",
                        name=entry.operation_id,
                        classes="backup-inspect-copy",
                        disabled=entry.status != "verified",
                    ),
                    Button(
                        "Review deletion",
                        name=entry.operation_id,
                        classes="backup-review-delete",
                        disabled=entry.pending_operation or entry.status != "verified",
                    ),
                    Button(
                        "Review later rollback",
                        name=entry.operation_id,
                        classes="backup-review-rollback",
                        disabled=entry.pending_operation or entry.status != "verified",
                    ),
                )
            else:
                if not entry.get("requirements_checked"):
                    setup = "Setup requirements unavailable. Recover this profile before opening it."
                elif entry["needs_setup"]:
                    setup = "Needs setup\nReview these features in their settings before enabling them:\n" + "\n".join(
                        "• " + owner.replace(".", " / ").replace("_", " ")
                        for owner in entry["pending_owners"]
                    )
                else:
                    setup = "Owner reviews complete. Optional features may still require setup."
                await listing.mount(
                    Static(
                        f"{entry['profile_id']}\n{entry['config']}\n{entry['data']}\n{entry['status']}\n{setup}",
                        markup=False,
                    ),
                    Button(
                        "Open in new process",
                        name=entry["profile_id"],
                        classes="backup-open-profile",
                        disabled=not entry.get("requirements_checked"),
                    ),
                )
        for row, state in pending:
            await listing.mount(
                Static(
                    f"Pending recovery: {row['operation_id']}\n{state['phase']}\nSelected configurations: {row['config_paths']}\nFinish completes this reviewed recovery. Roll back restores its verified prior data. Abort releases an untouched replacement without restoring a safety copy.",
                    markup=False,
                )
            )
            for action in state["actions"]:
                label = {
                    "finish": "Finish recovery",
                    "rollback": "Roll back interrupted replacement",
                    "abort": "Abort untouched replacement",
                }.get(action)
                if label is None:
                    continue
                await listing.mount(
                    Button(
                        label,
                        name=row["operation_id"],
                        classes="backup-recover-" + action,
                    )
                )

    @on(Button.Pressed, ".backup-review-delete")
    def _review_delete(self, event):
        self._delete_copy_id = event.button.name
        self.query_one("#backup-delete-preview", Static).update(
            f"Delete recovery copy {self._delete_copy_id}. This permanently removes its retained archive and its later-rollback option."
        )
        self.query_one("#backup-delete-confirm", Checkbox).value = False
        self.query_one("#backup-delete-copy", Button).disabled = False

    @on(Button.Pressed, ".backup-review-rollback")
    def _select_rollback(self, event):
        self._invalidate()
        self._forget_later_credential_review()
        self._rollback_copy_id = event.button.name
        self.query_one("#backup-later-form").display = True
        self.query_one("#backup-later-preview", Static).update(
            "Selected recovery copy: " + self._rollback_copy_id
        )
        self.query_one("#backup-later-confirm", Checkbox).value = False
        self.query_one("#backup-later-form").scroll_visible(immediate=True)

    @on(Button.Pressed, "#backup-later-review")
    def _review_rollback(self):
        if self._rollback_copy_id is None:
            return
        target = Path(self._input("backup-later-target")).expanduser()
        password = self._input("backup-copy-password")
        if not target.is_absolute() or not password:
            self.query_one("#backup-message", Static).update(
                "Select the current configuration and enter the old recovery-copy password."
            )
            return
        acknowledged = tuple(
            box.name
            for box in self.query(".backup-acknowledge-later-credential")
            if box.value
            and not box.disabled
            and box.name in self._later_review_codes_seen
        )
        self._requested_rollback_operation = None
        self._invalidate()
        self._preview_rollback(
            self.app,
            self._revision,
            self._rollback_copy_id,
            target,
            password.encode(),
            acknowledged,
        )
        self._clear_passwords()

    @work(exclusive=True, thread=True, group="backup-later-review")
    def _preview_rollback(
        self, app, revision, operation, target, password, acknowledged
    ):
        from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired

        try:
            inventory = self.service.preview_backup((target,), options={})
            plan = self.service.preview_rollback(
                operation,
                old_password=password,
                target=inventory,
                acknowledged_credential_issues=acknowledged,
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._deliver(
                app,
                self._rollback_ready,
                revision,
                operation,
                None,
                self.service.issue_code(error),
                error.issues if isinstance(error, CaptureReviewRequired) else (),
            )
        else:
            self._deliver(app, self._rollback_ready, revision, operation, plan, None)

    def _rollback_ready(self, revision, operation, plan, issue, review_issues=()):
        if (
            not self.is_mounted
            or self._mode != "copies"
            or revision != self._revision
            or operation != self._rollback_copy_id
        ):
            return
        self._rollback_plan = plan
        if issue:
            message = "Rollback review refused: " + issue
        else:
            message = (
                f"Recovery copy: {operation}\nRestore: {plan.restore}\nRetire: {plan.retire}\nPreserve: {plan.preserve}\nIssues: {plan.issues}\n"
                "Re-enter the old archive password above and choose a new safety-copy password. Review and confirmation do not activate providers or scheduled work."
            )
        self.query_one("#backup-later-preview", Static).update(message)
        self.query_one("#backup-later-start", Button).disabled = plan is None
        codes = tuple(code for code in review_issues if code.startswith("credential_"))
        if codes:
            self._later_review_codes_seen = codes
            self.run_worker(
                self._show_later_credential_review(
                    revision, self._later_selection(), codes, pending=False
                ),
                exclusive=True,
                group="later-credential-review",
            )

    @on(Button.Pressed, "#backup-later-start")
    def _start_rollback(self):
        old = self._input("backup-copy-password")
        new = self._input("backup-safety-password")
        if (
            self._rollback_copy_id is None
            or self._rollback_plan is None
            or not self.query_one("#backup-later-confirm", Checkbox).value
            or not old
            or not new
            or new != self._input("backup-safety-confirm")
        ):
            self.query_one("#backup-message", Static).update(
                "Confirm the reviewed consequences, re-enter the old password, and enter matching new safety-copy passwords."
            )
            return
        try:
            self._rollback_selection = self._later_selection()
            self._requested_rollback_operation = self.service.start_rollback(
                self._rollback_copy_id,
                self._rollback_plan,
                old_password=old.encode(),
                new_password=new.encode(),
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-message", Static).update(
                "Rollback refused: " + self.service.issue_code(error)
            )
        finally:
            self._clear_passwords()
        self._rollback_plan = None
        self.query_one("#backup-later-start", Button).disabled = True
        self._refresh_status()

    @on(Button.Pressed, "#backup-delete-copy")
    def _delete_copy(self):
        if (
            self._delete_copy_id is None
            or not self.query_one("#backup-delete-confirm", Checkbox).value
        ):
            self.query_one("#backup-message", Static).update(
                "Select the copy and acknowledge the deletion consequence first."
            )
            return
        try:
            self.service.start_delete_copy(self._delete_copy_id, user_selected=True)
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-message", Static).update(
                "Deletion refused: " + self.service.issue_code(error)
            )
        self._delete_copy_id = None
        self.query_one("#backup-delete-copy", Button).disabled = True
        self._refresh_status()

    @on(Button.Pressed, ".backup-inspect-copy")
    def _inspect_copy(self, event):
        password = self._input("backup-copy-password")
        try:
            self.service.start_copy_inspection(
                event.button.name, password=password.encode()
            )
            self._clear_inspection()
            self._show_mode("inspect")
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-message", Static).update(
                "Inspection refused: " + self.service.issue_code(error)
            )
        finally:
            self._clear_passwords()
        self._refresh_status()

    @on(Button.Pressed, ".backup-recover-finish")
    @on(Button.Pressed, ".backup-recover-rollback")
    @on(Button.Pressed, ".backup-recover-abort")
    def _recover(self, event):
        password = self._input("backup-copy-password")
        action = next(
            (
                name
                for name in ("finish", "rollback", "abort")
                if event.button.has_class("backup-recover-" + name)
            ),
            None,
        )
        if action is None:
            return
        try:
            self.service.start_recovery(
                event.button.name,
                action=action,
                rollback_password=password.encode()
                if password and action != "abort"
                else None,
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-message", Static).update(
                "Recovery refused: " + self.service.issue_code(error)
            )
        finally:
            self._clear_passwords()
        self._refresh_status()

    @on(Button.Pressed, ".backup-open-profile")
    def _open_profile(self, event):
        self.app.open_recovery_profile(event.button.name)

    def _input(self, identifier):
        return self.query_one("#" + identifier, Input).value

    def _options(self):
        credentials = self.query_one("#backup-credentials", Checkbox).value
        return {
            "external_roots": self.external_roots,
            "model_ids": tuple(
                value.strip()
                for value in self._input("backup-models").split(",")
                if value.strip()
            ),
            "temporary_media": self.query_one("#backup-temporary", Checkbox).value,
            "diagnostics": self.query_one("#backup-diagnostics", Checkbox).value,
            "allow_partial": self.query_one("#backup-partial", Checkbox).value,
            "credential_mode": "include" if credentials else "exclude",
            "acknowledged_credential_issues": tuple(
                box.name
                for box in self.query(".backup-acknowledge-credential")
                if credentials and box.value and box.name in self._review_codes_seen
            ),
            "encrypted": credentials
            or self.query_one("#backup-encrypted", Checkbox).value,
        }

    def _validate_password(self, options):
        password, repeated = (
            self._input("backup-password"),
            self._input("backup-password-confirm"),
        )
        if password != repeated:
            raise ValueError("Passwords must match.")
        if options["encrypted"] and not password:
            raise ValueError("Enter and confirm an encryption password.")
        if password and not options["encrypted"]:
            raise ValueError("Select encryption to use this password.")

    def _invalidate(self):
        self._revision += 1
        self._preview = self._reviewed = None
        self.query_one("#backup-create", Button).disabled = True
        self._restore_plan = None
        self._extraction_plan = None
        self.query_one("#backup-start-extraction", Button).disabled = True
        self.query_one("#backup-start-restore", Button).disabled = True
        self._rollback_plan = None
        self.query_one("#backup-later-start", Button).disabled = True
        self.query_one("#backup-later-confirm", Checkbox).value = False

    @on(Input.Changed)
    @on(Checkbox.Changed)
    @on(Select.Changed)
    def _form_changed(self, event):
        if self.is_mounted:
            if event.control.id in {
                "backup-copy-password",
                "backup-safety-password",
                "backup-safety-confirm",
                "backup-later-confirm",
                "backup-media-confirm",
            }:
                return
            self._invalidate()
            self._sync_replacement_host()
            if event.control.id == "backup-source":
                self._clear_inspection(dismiss_current=True)
            if event.control.id == "backup-later-target":
                self._forget_later_credential_review()
            control = event.control
            if not control.has_class("backup-acknowledge-credential") and not (
                isinstance(control, Input) and control.password
            ):
                self._forget_credential_review()
            if not control.has_class("backup-acknowledge-restore-credential") and not (
                isinstance(control, Input) and control.password
            ):
                self._forget_restore_credential_review()
            if isinstance(control, (Input, Select)) and not (
                isinstance(control, Input) and control.password
            ):
                self._forget_safety_scope()

    def _forget_safety_scope(self):
        self._safety_scope_seen = ()
        self.query_one("#backup-safety-scope").display = False
        for box in self.query(".backup-safety-member"):
            box.value = False
            box.disabled = True

    def _later_selection(self):
        return self._rollback_copy_id, str(Path(self._input("backup-later-target")).expanduser())

    def _forget_later_credential_review(self):
        """Omission decisions belong to the selected copy and current target."""
        self._requested_rollback_operation = None
        self._rollback_selection = None
        self._later_review_codes_seen = ()
        self.query_one("#backup-later-credential-review").display = False
        for box in self.query(".backup-acknowledge-later-credential"):
            box.value = False
            box.disabled = True

    def _forget_restore_credential_review(self):
        """A rollback omission belongs to the unchanged replacement choices."""
        self._requested_restore_operation = None
        self._restore_review_codes_seen = ()
        self.query_one("#backup-restore-credential-review").display = False
        for box in self.query(".backup-acknowledge-restore-credential"):
            box.value = False
            box.disabled = True

    def _forget_credential_review(self):
        """An omission decision belongs only to the source choices that failed."""
        self._requested_backup_operation = None
        self._review_codes_seen = ()
        for box in self.query(".backup-acknowledge-credential"):
            box.value = False
            box.disabled = True

    @on(Button.Pressed, "#backup-review")
    def _review(self):
        options = self._options()
        try:
            self._validate_password(options)
            destination = Path(self._input("backup-destination")).expanduser()
            if not destination.is_absolute() or (
                not self.config_paths and not self.include_known_profiles
            ):
                raise ValueError(
                    "Choose a full output path and at least one profile configuration."
                )
        except ValueError as error:
            self.query_one("#backup-message", Static).update(str(error))
            return
        self._invalidate()
        self.query_one("#backup-message", Static).update(
            "Discovering selected sources…"
        )
        self._preview_sources(
            self.app, self._revision, self.config_paths, destination, options
        )

    @work(exclusive=True, thread=True, group="backup-preview")
    def _preview_sources(self, app, revision, profiles, destination, options):
        try:
            preview = self.service.preview_backup_details(
                profiles, options=options, destination=destination,
                include_known_profiles=self.include_known_profiles,
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._deliver(
                app, self._show_preview, revision, None, self.service.issue_code(error)
            )
        else:
            self._deliver(
                app,
                self._show_preview,
                revision,
                preview,
                (profiles, destination, options),
            )

    def _show_preview(self, revision, preview, reviewed):
        if not self.is_mounted or revision != self._revision:
            return
        if preview is None:
            self.query_one("#backup-message", Static).update(
                "Source review failed: " + reviewed
            )
            return
        details = preview
        preview = details["inventory"]
        self._preview, self._reviewed = preview, reviewed
        self.query_one("#backup-profiles", Static).update(
            "Reviewed profile configurations:\n" + "\n".join(
                dict.fromkeys(
                    str(item.path) for item in preview.items
                    if item.owner == "config" and item.path is not None
                )
            )
        )
        rows = [
            "Complete coverage" if preview.complete else "Partial coverage",
            details["maintenance"],
        ]
        rows.extend(
            f"Volume {row['path']}: {row['required_bytes']} bytes required; {row['available_bytes']} bytes available"
            for row in details["capacity"]
        )
        if details["credential_mode"] == "include":
            rows.append(
                "Supported credential coverage is checked during capture. Any unavailable scopes require a new explicit review."
            )
        rows.extend(
            f"{item.owner}: {item.status}" + (f" — {item.path}" if item.path else "")
            for item in preview.items
        )
        rows.extend(preview.issues)
        self.query_one("#backup-coverage", Static).update("\n".join(rows))
        self.query_one("#backup-create", Button).disabled = not (
            preview.complete or reviewed[2]["allow_partial"]
        ) or not all(row["sufficient"] for row in details["capacity"])
        self.query_one("#backup-message", Static).update(
            "Review the displayed coverage. Backup pauses writers for capture, then resumes them before packaging."
        )

    @on(Button.Pressed, "#backup-create")
    def _create(self):
        if self._preview is None or self._reviewed is None:
            return
        profiles, destination, options = self._reviewed
        try:
            self._validate_password(options)
            password = (
                self._input("backup-password").encode()
                if options["encrypted"]
                else None
            )
            self._requested_backup_operation = self.service.start_backup(
                profiles,
                self._preview.scope_digest,
                destination,
                options=options,
                password=password,
                include_known_profiles=self.include_known_profiles,
            )
        except (OSError, ValueError, RuntimeError):
            self.query_one("#backup-message", Static).update(
                "Backup could not start. Review the source selection and output path again."
            )
        finally:
            self._clear_passwords()
        self._invalidate()
        self._refresh_status()

    @on(Button.Pressed, "#backup-inspect")
    def _inspect(self):
        source = Path(self._input("backup-source")).expanduser()
        if not source.is_absolute():
            self.query_one("#backup-message", Static).update(
                "Choose a full archive path."
            )
            return
        password = self._input("backup-inspect-password")
        try:
            self.service.start_inspection(
                source, password=password.encode() if password else None
            )
            self._clear_inspection()
        except (OSError, ValueError, RuntimeError):
            self.query_one("#backup-message", Static).update(
                "Inspection could not start. Another recovery operation may still be running."
            )
        finally:
            self._clear_passwords()
        self._refresh_status()

    def _clear_inspection(self, *, dismiss_current=False):
        self._forget_restore_credential_review()
        self._forget_safety_scope()
        if dismiss_current:
            current = self.service.current()
            self._dismissed_inspection_id = (
                current["operation_id"]
                if current is not None
                else self._inspection_id or self._summary_requested
            )
        self._inspection_id = self._inspection_summary = self._summary_requested = None
        self.query_one("#backup-inspection-summary", Static).update(
            "Inspect the selected archive before reviewing a restore."
        )
        self._restore_plan = None
        self.query_one("#backup-restore-form").display = False
        self._extraction_plan = None
        self.query_one("#backup-inert-form").display = False
        self.query_one("#backup-start-extraction", Button).disabled = True
        self.query_one("#backup-start-restore", Button).disabled = True

    def _refresh_status(self):
        current = self.service.current()
        if current is None:
            return
        result = current["result"]
        if (
            current["kind"].startswith("inspect")
            and current["state"] == "succeeded"
            and self._summary_requested != current["operation_id"]
            and self._dismissed_inspection_id != current["operation_id"]
        ):
            self._summary_requested = current["operation_id"]
            self._load_summary(self.app, self._summary_requested)
        label = result_label(
            archive_verified=bool(result.get("archive_verified")),
            restoration_validated=bool(result.get("restoration_validated")),
            opened=bool(result.get("opened_successfully")),
            needs_setup=bool(result.get("needs_setup")),
        )
        phase = current["phase"].replace("_", " ")
        if current["state"] == "succeeded" and result.get("inert_extracted"):
            label = "Inert files extracted"
        text = (
            label
            if current["state"] == "succeeded" and label != "Not verified"
            else f"{current['state'].capitalize()}: {phase}"
        )
        if result.get("path"):
            text += "\n" + str(result["path"])
        if result.get("profile_id"):
            text += "\nProfile: " + result["profile_id"]
        if result.get("journal_operation_id"):
            text += "\nRecovery operation: " + result["journal_operation_id"]
        if current["issues"]:
            text += "\n" + ", ".join(current["issues"])
        review_issues = tuple(current.get("review_issues", ()))
        if review_issues:
            text += "\nReview: " + ", ".join(review_issues)
            codes = tuple(
                code for code in review_issues if code.startswith("credential_")
            )
            if (
                current["kind"] == "backup"
                and current["operation_id"] == self._requested_backup_operation
                and codes
                and codes != self._review_codes_seen
            ):
                self._review_codes_seen = codes
                self.run_worker(
                    self._show_credential_review(codes),
                    exclusive=True,
                    group="backup-credential-review",
                )
            if (
                current["kind"] == "restore"
                and current["operation_id"] == self._requested_restore_operation
                and codes
                and codes != self._restore_review_codes_seen
            ):
                self._restore_review_codes_seen = codes
                self.run_worker(
                    self._show_restore_credential_review(current["operation_id"], codes),
                    exclusive=True,
                    group="restore-credential-review",
                )
            if (
                current["kind"] == "later_rollback"
                and current["operation_id"] == self._requested_rollback_operation
                and self._rollback_selection == self._later_selection()
                and self._mode == "copies"
                and codes and codes != self._later_review_codes_seen
            ):
                self._later_review_codes_seen = codes
                self.run_worker(
                    self._show_later_credential_review(
                        self._revision, self._rollback_selection, codes, pending=True
                    ),
                    exclusive=True, group="later-credential-review",
                )
        self.query_one("#backup-status", Static).update(text)
        self.query_one("#backup-cancel", Button).disabled = (
            current["state"] != "running"
        )
        if (
            current["state"] != "running"
            and self._last_terminal != current["operation_id"]
        ):
            self._last_terminal = current["operation_id"]
            if self._mode in ("copies", "profiles"):
                self._refresh_list(self.app, self._revision, self._mode)
            elif self._mode == "media" and current["kind"] == "recovered_media":
                self._revision += 1
                self._load_media(self.app, self._revision, self._media_offset)

    async def _show_credential_review(self, codes):
        self._invalidate()
        area = self.query_one("#backup-credential-review", Vertical)
        await area.remove_children()
        await area.mount(
            Static(
                "Capture found these unavailable credential scopes. Review each omission, then review coverage again.",
                markup=False,
            )
        )
        for code in codes:
            await area.mount(
                Checkbox(
                    Text("Acknowledge omission: " + code),
                    name=code,
                    classes="backup-acknowledge-credential",
                )
            )

    async def _show_restore_credential_review(self, operation, codes):
        if operation != self._requested_restore_operation or codes != self._restore_review_codes_seen:
            return
        self._invalidate()
        area = self.query_one("#backup-restore-credential-review", Vertical)
        await area.remove_children()
        if operation != self._requested_restore_operation or codes != self._restore_review_codes_seen:
            return
        area.display = True
        await area.mount(
            Static(
                "The safety copy could not include these credentials. No replacement was published. "
                "In Recovery copies, choose Abort untouched replacement. Then return here, "
                "review each omission, enter a new rollback password, and review restore again.",
                markup=False,
            ),
            *(
                Checkbox(
                    Text("Acknowledge safety-copy omission: " + code),
                    name=code,
                    classes="backup-acknowledge-restore-credential",
                )
                for code in codes
            ),
        )

    async def _show_later_credential_review(
        self, revision, selection, codes, *, pending
    ):
        if (
            not self.is_mounted
            or self._mode != "copies"
            or revision != self._revision
            or selection != self._later_selection()
            or codes != self._later_review_codes_seen
        ):
            return
        self._invalidate()
        revision = self._revision
        area = self.query_one("#backup-later-credential-review", Vertical)
        await area.remove_children()
        if (
            not self.is_mounted
            or self._mode != "copies"
            or revision != self._revision
            or selection != self._later_selection()
            or codes != self._later_review_codes_seen
        ):
            return
        area.display = True
        await area.mount(
            Static(
                (
                    "The new safety copy could not include these credentials. No later rollback was published. "
                    "Choose Abort untouched replacement above. "
                    if pending
                    else "Review these unavailable credential scopes. "
                )
                + "Then select each omission you accept, re-enter the old copy password, and review later rollback again.",
                markup=False,
            ),
            *(
                Checkbox(
                    Text("Acknowledge safety-copy omission: " + code),
                    name=code,
                    classes="backup-acknowledge-later-credential",
                )
                for code in codes
            ),
        )

        if pending and self._mode == "copies" and selection == self._later_selection():
            # The omission review invalidated the terminal poll's earlier list.
            self._refresh_list(self.app, self._revision, "copies")

    def _deliver(self, app, callback, *args):
        """Discard a read-only preview after its view or application exits."""
        if not self.is_mounted or not app.is_running:
            return
        try:
            app.call_from_thread(callback, *args)
        except RuntimeError:
            if app.is_running:
                raise

    @work(exclusive=True, thread=True, group="backup-summary")
    def _load_summary(self, app, operation):
        try:
            summary = self.service.summary(operation)
        except (OSError, ValueError, RuntimeError):
            self._deliver(app, self._summary_ready, operation, None)
        else:
            self._deliver(app, self._summary_ready, operation, summary)

    async def _summary_ready(self, operation, summary):
        if not self.is_mounted or operation != self._summary_requested:
            return
        if summary is None:
            self.query_one("#backup-inspection-summary", Static).update(
                "Archive summary could not be verified. Inspect the archive again."
            )
            return
        rows = [
            "Archive verified",
            f"Files: {summary['file_count']} · Payload bytes: {summary['payload_bytes']}",
            f"Consistency: {summary['consistency']}",
            f"Credentials: {summary['credential_policy']}",
            "Owners: " + ", ".join(summary["owners"]),
            "Required capabilities: " + ", ".join(summary["required_capabilities"]),
        ]
        rows.extend(
            f"Excluded {key}: {reason}" for key, reason in summary["exclusions"]
        )
        rows.extend(summary["report"])
        self.query_one("#backup-inspection-summary", Static).update("\n".join(rows))
        slots = self.query_one("#backup-destination-slots", Vertical)
        await slots.remove_children()
        for index, slot in enumerate(summary["destination_slots"]):
            await slots.mount(
                Static(f"{slot['logical_id']} ({slot['kind']})", markup=False),
                Input(
                    placeholder="Absolute local directory", id=f"backup-root-{index}"
                ),
            )
        for index, profile in enumerate(summary["profile_ids"]):
            await slots.mount(
                Static(f"New display name for {profile}", markup=False),
                Input(
                    placeholder="Optional local profile name",
                    id=f"backup-profile-name-{index}",
                ),
            )
        groups = self.query_one("#backup-inert-groups", Vertical)
        await groups.remove_children()
        for index, group in enumerate(summary["dependency_groups"]):
            await groups.mount(
                Checkbox(
                    Text(f"{group['group_id']}: {len(group['members'])} members"),
                    id=f"backup-inert-group-{index}",
                )
            )
        if not self.is_mounted or operation != self._summary_requested:
            return
        self._inspection_id, self._inspection_summary = operation, summary
        self.query_one("#backup-restore-form").display = self._mode == "inspect"
        self.query_one("#backup-inert-form").display = self._mode == "inspect"

    @on(Button.Pressed, "#backup-review-extraction")
    def _review_extraction(self):
        if self._inspection_id is None:
            return
        groups = tuple(
            group["group_id"]
            for index, group in enumerate(self._inspection_summary["dependency_groups"])
            if self.query_one(f"#backup-inert-group-{index}", Checkbox).value
        )
        destination = Path(self._input("backup-inert-destination")).expanduser()
        if not groups or not destination.is_absolute():
            self.query_one("#backup-inert-preview", Static).update(
                "Select at least one group and a new absolute directory."
            )
            return
        self._invalidate()
        self._preview_extraction(
            self.app, self._revision, self._inspection_id, groups, destination
        )

    @work(exclusive=True, thread=True, group="backup-extraction-preview")
    def _preview_extraction(self, app, revision, inspection, groups, destination):
        try:
            operation = self.service.start_extraction_preview(
                inspection, group_ids=groups, destination=destination
            )
            status = self.service.wait(operation)
            plan = status["result"].get("plan")
            issue = (
                ", ".join(status["issues"]) if status["state"] != "succeeded" else None
            )
        except (OSError, ValueError, RuntimeError) as error:
            plan, issue = None, self.service.issue_code(error)
        self._deliver(app, self._extraction_ready, revision, inspection, plan, issue)

    def _extraction_ready(self, revision, inspection, plan, issue):
        if (
            not self.is_mounted
            or revision != self._revision
            or inspection != self._inspection_id
        ):
            return
        self._extraction_plan = plan
        self.query_one("#backup-start-extraction", Button).disabled = plan is None
        text = "Extraction review refused: " + str(issue)
        if plan is not None:
            text = (
                f"Inert extraction to {plan.destination}\n"
                f"Groups: {', '.join(plan.group_ids)}\n"
                f"Payload: {plan.payload_bytes} bytes\n"
                f"Unselected dependencies: {plan.unselected_dependencies}"
            )
        self.query_one("#backup-inert-preview", Static).update(text)

    @on(Button.Pressed, "#backup-start-extraction")
    def _start_extraction(self):
        if self._inspection_id is None or self._extraction_plan is None:
            return
        try:
            self.service.start_extraction(self._inspection_id, self._extraction_plan)
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-inert-preview", Static).update(
                "Extraction could not start: " + self.service.issue_code(error)
            )
        self._invalidate()
        self._refresh_status()

    @on(Button.Pressed, "#backup-review-restore")
    def _review_restore(self):
        if self._inspection_id is None:
            return
        mode = self.query_one("#backup-restore-mode", Select).value
        if mode == "replace" and self._requires_recovery_restart():
            self.query_one("#backup-message", Static).update(
                "Continue in recovery mode before reviewing replacement."
            )
            return
        destinations = {}
        for index, slot in enumerate(self._inspection_summary["destination_slots"]):
            path = Path(self._input(f"backup-root-{index}")).expanduser()
            if not path.is_absolute():
                self.query_one("#backup-message", Static).update(
                    "Choose an absolute local directory for every destination."
                )
                return
            destinations[slot["logical_id"]] = path
        names = {
            profile: self._input(f"backup-profile-name-{index}").strip()
            for index, profile in enumerate(self._inspection_summary["profile_ids"])
            if self._input(f"backup-profile-name-{index}").strip()
        }
        target = (
            Path(self._input("backup-target-config")).expanduser()
            if mode == "replace"
            else None
        )
        if target is not None and not target.is_absolute():
            self.query_one("#backup-message", Static).update(
                "Choose the existing local profile configuration to replace."
            )
            return
        self._invalidate()
        self._preview_restore(
            self.app,
            self._revision,
            self._inspection_id,
            mode,
            destinations,
            names,
            target,
            tuple(
                box.name
                for box in self.query(".backup-acknowledge-restore-credential")
                if mode == "replace" and box.value and box.name in self._restore_review_codes_seen
            ),
            tuple(
                box.name
                for box in self.query(".backup-safety-member")
                if mode == "replace" and box.value
                and box.name in {row[0] for row in self._safety_scope_seen}
            ),
        )

    @work(exclusive=True, thread=True, group="backup-restore-preview")
    def _preview_restore(
        self, app, revision, inspection, mode, destinations, names, target, acknowledged, safety_scope
    ):
        try:
            inventory = (
                None
                if target is None
                else self.service.preview_backup((target,), options={})
            )
            plan = self.service.preview_restore(
                inspection,
                mode=mode,
                destinations=destinations,
                target=inventory,
                profile_names=names,
                acknowledged_credential_issues=acknowledged,
                safety_scope=safety_scope,
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._deliver(
                app, self._restore_ready, revision, None, self.service.issue_code(error)
            )
        else:
            self._deliver(app, self._restore_ready, revision, plan, None)

    async def _restore_ready(self, revision, plan, issue):
        if not self.is_mounted or revision != self._revision:
            return
        if plan is None:
            self.query_one("#backup-restore-preview", Static).update(
                "Restore review refused: " + issue
            )
            return
        choices = tuple(
            (item.logical_id, item.owner, str(item.path))
            for item in (plan.target.items if plan.target is not None else ())
            if item.path is not None
            and item.status in {"included", "included_directory"}
            and (item.logical_id, item.path) in plan.preserve
        )
        if choices != self._safety_scope_seen:
            area = self.query_one("#backup-safety-scope", Vertical)
            await area.remove_children()
            if not self.is_mounted or revision != self._revision:
                return
            self._safety_scope_seen = choices
            area.display = bool(choices)
            if choices:
                await area.mount(
                    Static(
                        "Optional additions to the encrypted before-replacement copy. "
                        "Select preserved files and their declared folders when restored data depends on them. "
                        "These live files stay in place. Review restore again after selecting.",
                        markup=False,
                    ),
                    *(
                        Checkbox(Text(f"{owner}: {path}"), name=key, classes="backup-safety-member")
                        for key, owner, path in choices
                    ),
                )
            if not self.is_mounted or revision != self._revision:
                return
            if plan.safety_scope:
                self.query_one("#backup-restore-preview", Static).update(
                    "Safety-copy sources changed. Select additions and review restore again."
                )
                return
        self._restore_plan = plan
        rows = [
            "Reviewed local restore plan",
            "Restored execution remains inactive until owner review and setup.",
        ]
        for label, entries in (
            ("Restore", plan.restore),
            ("Retire", plan.retire),
            ("Preserve", plan.preserve),
        ):
            rows.extend(f"{label}: {key} → {path}" for key, path in entries)
        rows.extend(f"Issue: {issue}" for issue in plan.issues)
        rows.extend(f"Additional safety-copy source: {key}" for key in plan.safety_scope)
        rows.extend(
            f"Acknowledged safety-copy credential omission: {issue}"
            for issue in plan.acknowledged_credential_issues
        )
        rows.extend(
            f"Metadata: {key}: {old} → {new}" for key, old, new in plan.metadata
        )
        self.query_one("#backup-restore-preview", Static).update("\n".join(rows))
        self.query_one("#backup-start-restore", Button).disabled = False

    @on(Button.Pressed, "#backup-start-restore")
    def _start_restore(self):
        if self._restore_plan is None:
            return
        plan = self._restore_plan
        if plan.mode == "replace" and self._requires_recovery_restart():
            return
        password = None
        if plan.mode == "replace":
            password = self._input("backup-rollback-password")
            if not password or password != self._input("backup-rollback-confirm"):
                self.query_one("#backup-message", Static).update(
                    "Enter and confirm the rollback password before replacement."
                )
                return
            password = password.encode()
        try:
            operation = self.service.start_restore(
                self._inspection_id, plan, rollback_password=password
            )
            self._requested_restore_operation = operation if plan.mode == "replace" else None
        except (OSError, ValueError, RuntimeError) as error:
            self.query_one("#backup-message", Static).update(
                "Restore could not start: " + self.service.issue_code(error)
            )
        finally:
            self._clear_passwords()
        self._invalidate()
        self._refresh_status()

    def _requires_recovery_restart(self):
        return callable(getattr(self.app, "request_recovery_restart", None))

    def _sync_replacement_host(self):
        required = (
            self.query_one("#backup-restore-mode", Select).value == "replace"
            and self._requires_recovery_restart()
        )
        self.query_one("#backup-restart-note").display = required
        self.query_one("#backup-restart").display = required
        self.query_one("#backup-review-restore", Button).disabled = required

    @on(Button.Pressed, "#backup-restart")
    def _restart_for_replacement(self):
        if not self._requires_recovery_restart():
            return
        target = self._input("backup-target-config").strip()
        if not target and self.config_paths:
            target = str(self.config_paths[0])
        target = Path(target).expanduser()
        if not target.is_absolute():
            self.query_one("#backup-message", Static).update(
                "Choose the existing local configuration before continuing."
            )
            return
        source = self._input("backup-source").strip()
        archive = Path(source).expanduser() if source else None
        self._clear_passwords()
        self.app.request_recovery_restart(archive, target)

    @on(Button.Pressed, "#backup-cancel")
    def _cancel(self):
        current = self.service.current()
        if current and current["state"] == "running":
            self.service.cancel(current["operation_id"])
            self.query_one("#backup-message", Static).update(
                "Cancellation requested; waiting for a safe stopping point."
            )

    @on(Button.Pressed, "#backup-pick-source")
    def _pick_source(self):
        self.app.push_screen(
            FileOpen(title="Choose a backup archive"),
            lambda path: self._set_path("backup-source", path),
        )

    @on(Button.Pressed, "#backup-pick-destination")
    def _pick_destination(self):
        self.app.push_screen(
            FileSave(title="Choose a new backup file", can_overwrite=False),
            lambda path: self._set_path("backup-destination", path),
        )

    def _set_path(self, identifier, path):
        if path is not None and self.is_mounted:
            self.query_one("#" + identifier, Input).value = str(Path(path).absolute())

    @on(Button.Pressed, "#backup-add-profile")
    def _add_profile(self):
        self.app.push_screen(
            FileOpen(title="Choose a profile configuration"), self._profile_chosen
        )

    def _profile_chosen(self, path):
        if path is not None and self.is_mounted:
            self.config_paths = tuple(
                dict.fromkeys((*self.config_paths, Path(path).absolute()))
            )
            self.query_one("#backup-profiles", Static).update(self._profile_text())
            self._forget_credential_review()
            self._invalidate()

    @on(Button.Pressed, "#backup-add-external")
    def _add_external(self):
        self.app.push_screen(
            SelectDirectory(title="Choose an external folder"), self._external_chosen
        )

    def _external_chosen(self, path):
        if path is not None and self.is_mounted:
            self.external_roots = tuple(
                dict.fromkeys((*self.external_roots, Path(path).absolute()))
            )
            self.query_one("#backup-external-roots", Static).update(
                "\n".join(map(str, self.external_roots))
            )
            self._forget_credential_review()
            self._invalidate()
