"""Support ``python -m tldw_chatbook`` without eager application imports."""

from tldw_chatbook.cli import main_cli_runner


if __name__ == "__main__":
    from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

    admit_startup()
    raise SystemExit(main_cli_runner())
