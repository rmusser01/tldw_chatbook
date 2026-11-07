"""Empty schema dumps from AgentRunsDB at 6f0f4e04ef (v12) and 3d271626a6 (v15)."""

from pathlib import Path


def install_historical(connection, version):
    """Install original DDL/stamps and the released standalone successors."""
    start = 12 if version == 12 else 15
    fixture = Path(__file__).with_name(f"agent_runs_v{start}.sql")
    connection.executescript(fixture.read_text())
    migrations = Path(__file__).resolve().parents[3] / "tldw_chatbook/DB/migrations"
    for number in range(start, version):
        (script,) = migrations.glob(f"agent_runs_v{number}_to_v{number + 1}_*.sql")
        connection.executescript(script.read_text())
    assert [
        row[0]
        for row in connection.execute(
            "SELECT version FROM schema_version ORDER BY version"
        )
    ] == list(range(4, version + 1))
