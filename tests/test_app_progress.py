import logging
from unittest.mock import patch

from intake import create_analysis_run
from models import AgentFindings, StageResult
from orchestrator import PIPELINE_STAGES, PipelineResult


def _make_stage_results(statuses):
    results = []
    for stage_name in PIPELINE_STAGES:
        status = statuses.get(stage_name, "completed")
        error = f"{stage_name} failed" if status == "failed" else None
        findings = (
            AgentFindings(
                sources=[f"https://example.com/{stage_name}"],
                confidence=0.8,
                key_findings=[f"{stage_name} finding"],
                evidence_gaps=[],
            )
            if status == "completed"
            else None
        )
        results.append(StageResult(stage_name=stage_name, status=status, error=error, findings=findings))
    return results


class TestFinalizeRunStatusTransitions:
    def test_complete_pipeline_sets_run_status_to_complete(self, sample_run_input):
        analysis_run = create_analysis_run(sample_run_input)
        assert analysis_run.status == "running"

        stage_results = _make_stage_results({})
        pipeline_result = PipelineResult(status="complete", stage_results=stage_results)

        fake_session = {"analysis_run": analysis_run}
        saved_runs = []

        with patch("app.st") as mock_st, patch("app.save_run", side_effect=lambda run: saved_runs.append(run)):
            mock_st.session_state = fake_session
            from app import _finalize_run

            _finalize_run(pipeline_result)

        updated = fake_session["analysis_run"]
        assert updated.status == "complete"
        assert len(updated.stage_results) == 4


class TestRunPersistence:
    def test_completed_run_is_persisted_to_sqlite(self, sample_run_input, isolated_db_path):
        analysis_run = create_analysis_run(sample_run_input)
        stage_results = _make_stage_results({})
        pipeline_result = PipelineResult(status="complete", stage_results=stage_results)

        fake_session = {"analysis_run": analysis_run}

        with patch("app.st") as mock_st:
            mock_st.session_state = fake_session
            with patch("app.save_run") as mock_save:
                from app import _finalize_run

                _finalize_run(pipeline_result)

                mock_save.assert_called_once()
                persisted_run = mock_save.call_args[0][0]
                assert persisted_run.status == "complete"
                assert persisted_run.id == analysis_run.id
                assert len(persisted_run.stage_results) == 4

    def test_finalize_does_not_run_twice_for_same_run(self, sample_run_input):
        analysis_run = create_analysis_run(sample_run_input)
        pipeline_result = PipelineResult(status="complete", stage_results=_make_stage_results({}))

        fake_session = {"analysis_run": analysis_run}

        with patch("app.st") as mock_st, patch("app.save_run") as mock_save:
            mock_st.session_state = fake_session
            from app import _finalize_run

            _finalize_run(pipeline_result)
            _finalize_run(pipeline_result)

            mock_save.assert_called_once()

    def test_save_run_failure_logs_and_shows_error(self, sample_run_input, caplog):
        analysis_run = create_analysis_run(sample_run_input)
        pipeline_result = PipelineResult(status="complete", stage_results=_make_stage_results({}))

        fake_session = {"analysis_run": analysis_run}

        with patch("app.st") as mock_st, patch("app.save_run", side_effect=RuntimeError("disk full")):
            mock_st.session_state = fake_session
            from app import _finalize_run

            with caplog.at_level(logging.ERROR):
                _finalize_run(pipeline_result)

            mock_st.error.assert_called_once_with("Failed to save the completed analysis run.")

        updated = fake_session["analysis_run"]
        assert updated.status == "complete"
        assert "disk full" in caplog.text
