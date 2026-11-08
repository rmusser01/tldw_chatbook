"""Existing config producer records, captured against the supplied owner namespace."""


def record_config_sources(namespace, entries):
    from types import FunctionType

    rows = []
    for owner, name in entries:
        original = (
            owner[name] if type(owner) is dict else getattr(owner, name)
        )  # exact stock compatibility boundary
        function = getattr(original, "__func__", original)
        records = []
        while type(function) is FunctionType:
            records.append(
                (
                    function,
                    function.__code__,
                    function.__globals__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    tuple((function.__kwdefaults__ or {}).items()),
                    function.__closure__,
                    tuple(
                        (cell, cell.cell_contents)
                        for cell in function.__closure__ or ()
                    ),
                    vars(function).get("__wrapped__"),
                )
            )
            function = vars(function).get("__wrapped__")
        rows.append((owner, name, original, tuple(records)))
    return namespace, tuple(rows)


def record_first_send_sources(namespace):
    return record_config_sources(
        namespace,
        tuple(
            (namespace, name)
            for name in (
                "current_config_identity",
                "save_setting_to_cli_config",
                "save_settings_to_cli_config",
                "apply_settings_mutation_to_cli_config",
                "_apply_literal_settings_transaction_locked",
                "_config_write_lock",
                "_read_raw_cli_config_unlocked",
                "_atomic_config_values_from_raw",
                "_apply_literal_mutation_unlocked",
                "_config_data_for_persistence",
                "_write_raw_cli_config_unlocked",
                "_publish_runtime_config_unlocked",
                "_install_bootstrap_cache_from_raw",
                "load_settings",
                "_current_settings_view",
                "_load_cli_config_bootstrap_unlocked",
                "create_private_text",
                "_get_effective_config_path",
                "_settings_rebuild_lock",
                "_config_file_lock",
            )
        ),
    )
