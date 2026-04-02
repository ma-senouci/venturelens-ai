import sqlite3
from contextlib import closing
from pathlib import Path

from persistence import RunSummary, list_runs, load_run, save_run


def _snapshot_file(path: Path) -> tuple[bool, bytes | None]:
    if not path.exists():
        return False, None

    return True, path.read_bytes()


def _build_run(sample_analysis_run, *, run_id: str, startup_name: str, status: str, created_at: str):
    run_input = sample_analysis_run.input.model_copy(update={"startup_name": startup_name})
    return sample_analysis_run.model_copy(
        update={
            "id": run_id,
            "status": status,
            "created_at": created_at,
            "input": run_input,
        }
    )


class TestPersistence:
    def test_save_run_bootstraps_schema_without_touching_default_app_database(
        self, isolated_db_path, sample_analysis_run
    ):
        real_app_db_path = Path(__file__).resolve().parent.parent / "data" / "venturelens.db"
        real_app_db_snapshot = _snapshot_file(real_app_db_path)

        assert not isolated_db_path.parent.exists()

        save_run(sample_analysis_run, db_path=isolated_db_path)

        assert isolated_db_path.parent.exists()
        assert isolated_db_path.exists()
        assert _snapshot_file(real_app_db_path) == real_app_db_snapshot

        with closing(sqlite3.connect(isolated_db_path)) as connection:
            columns = connection.execute("PRAGMA table_info(runs)").fetchall()

        assert [(column[1], column[2], column[5]) for column in columns] == [
            ("id", "TEXT", 1),
            ("status", "TEXT", 0),
            ("created_at", "TEXT", 0),
            ("data", "TEXT", 0),
        ]

    def test_save_run_then_load_run_round_trips_analysis_run(self, isolated_db_path, sample_analysis_run):
        save_run(sample_analysis_run, db_path=isolated_db_path)

        restored = load_run(sample_analysis_run.id, db_path=isolated_db_path)

        assert restored == sample_analysis_run

    def test_load_run_returns_none_for_unknown_id(self, isolated_db_path):
        assert load_run("2d2257da-fc96-49ea-a1a6-b4ce2f106a5a", db_path=isolated_db_path) is None

    def test_list_runs_returns_newest_first_summaries(self, isolated_db_path, sample_analysis_run):
        oldest_run = _build_run(
            sample_analysis_run,
            run_id="fcb4f4f2-7a7f-4338-91e8-d97197b4f0b0",
            startup_name="OldCo",
            status="failed",
            created_at="2026-03-29T08:00:00+00:00",
        )
        newest_run = _build_run(
            sample_analysis_run,
            run_id="01fdfece-4fbd-4822-ab2b-5b66ac218048",
            startup_name="NewCo",
            status="complete",
            created_at="2026-03-31T18:30:00+00:00",
        )
        middle_run = _build_run(
            sample_analysis_run,
            run_id="9d5a8185-dc28-4721-a21f-6038697a0302",
            startup_name="MidCo",
            status="partial",
            created_at="2026-03-30T12:15:00+00:00",
        )

        save_run(middle_run, db_path=isolated_db_path)
        save_run(oldest_run, db_path=isolated_db_path)
        save_run(newest_run, db_path=isolated_db_path)

        assert list_runs(db_path=isolated_db_path) == [
            RunSummary(
                id="01fdfece-4fbd-4822-ab2b-5b66ac218048",
                startup_name="NewCo",
                status="complete",
                created_at="2026-03-31T18:30:00+00:00",
            ),
            RunSummary(
                id="9d5a8185-dc28-4721-a21f-6038697a0302",
                startup_name="MidCo",
                status="partial",
                created_at="2026-03-30T12:15:00+00:00",
            ),
            RunSummary(
                id="fcb4f4f2-7a7f-4338-91e8-d97197b4f0b0",
                startup_name="OldCo",
                status="failed",
                created_at="2026-03-29T08:00:00+00:00",
            ),
        ]

    def test_save_run_updates_existing_row_for_same_id(self, isolated_db_path, sample_analysis_run):
        save_run(sample_analysis_run, db_path=isolated_db_path)

        updated_run = sample_analysis_run.model_copy(update={"status": "partial"})
        save_run(updated_run, db_path=isolated_db_path)

        with closing(sqlite3.connect(isolated_db_path)) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE id = ?",
                (sample_analysis_run.id,),
            ).fetchone()[0]
            stored_status = connection.execute(
                "SELECT status FROM runs WHERE id = ?",
                (sample_analysis_run.id,),
            ).fetchone()[0]

        assert row_count == 1
        assert stored_status == "partial"
        assert load_run(sample_analysis_run.id, db_path=isolated_db_path) == updated_run
