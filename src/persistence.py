import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from models import AnalysisRun

DEFAULT_DB_PATH = Path("data") / "venturelens.db"


@dataclass(frozen=True, slots=True)
class RunSummary:
    id: str
    startup_name: str
    status: str
    created_at: str


@contextmanager
def _connect(db_path: str | Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        yield connection
    finally:
        connection.close()


def save_run(run: AnalysisRun, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO runs (id, status, created_at, data)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                created_at = excluded.created_at,
                data = excluded.data
            """,
            (run.id, run.status, run.created_at, run.model_dump_json()),
        )
        connection.commit()


def load_run(run_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> AnalysisRun | None:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()

    if row is None:
        return None

    return AnalysisRun.model_validate_json(row[0])


def list_runs(db_path: str | Path = DEFAULT_DB_PATH) -> list[RunSummary]:
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT id, status, created_at, data FROM runs ORDER BY created_at DESC").fetchall()

    return [
        RunSummary(
            id=row[0],
            startup_name=AnalysisRun.model_validate_json(row[3]).input.startup_name,
            status=row[1],
            created_at=row[2],
        )
        for row in rows
    ]
