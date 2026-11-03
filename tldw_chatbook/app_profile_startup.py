"""Existing owned operations, loaded only on their actual execution path."""


def _stamp_new_profile_library_lifecycle(self, *, source_globals) -> None:
    """Persist the Library lifecycle at profile CREATION (task-32059).

    ``coerce_library_lifecycle`` reads an absent
    ``[library.rail_state] lifecycle`` as ``expanded`` for any profile the
    current run did not create -- so a user who completed first-run setup,
    quit, and relaunched before ever opening Library never saw the
    documented compact "Get started" rail. Writing ``unknown`` here (the
    same value the screen would have derived on a first visit in THIS run)
    makes the second launch read the fact instead of inferring it from a
    missing key. A profile that already carries a lifecycle is untouched.
    """
    library_config = self.app_config.get("library")
    if not isinstance(library_config, dict):
        library_config = {}
        self.app_config["library"] = library_config
    rail_state = library_config.get("rail_state")
    if not isinstance(rail_state, dict):
        rail_state = {}
        library_config["rail_state"] = rail_state
    if "lifecycle" in rail_state:
        return
    # One name for the value both the in-memory state and the file get:
    # "unknown" is LibraryLifecycle.UNKNOWN.value, spelled out rather than
    # imported: pulling the Library package in here would put its modules
    # on every boot for one string (see the module-census ratchet).
    lifecycle = "unknown"
    rail_state["lifecycle"] = lifecycle
    try:
        # `save_setting_to_cli_config` REPORTS a write failure rather than
        # raising it, and an unstamped profile reads back as `expanded` on
        # the next launch -- so the return value is the failure signal.
        saved = source_globals["save_setting_to_cli_config"](
            "library.rail_state", "lifecycle", lifecycle
        )
    except Exception:
        saved = False
    if not saved:
        # A profile whose config cannot be written still gets the correct
        # in-memory lifecycle for this run; boot must not fail over it.
        source_globals["logger"].warning(
            "Could not stamp the Library lifecycle for a new profile."
        )
