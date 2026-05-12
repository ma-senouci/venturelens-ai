import logging
from unittest.mock import patch

from intake import create_analysis_run
from models import AgentFindings, CriticFindings, StageResult
from orchestrator import CRITIC_STAGE, PIPELINE_STAGES, PipelineResult


def _make_stage_results(statuses=None):
    statuses = statuses or {}
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


def _make_critic_stage_result(status="completed", error=None):
    findings = (
        CriticFindings(
            contradictions=["Needs review"],
            weak_assumptions=[],
            unsupported_claims=[],
            open_questions=[],
            sources=["https://example.com/review"],
            confidence=0.7,
        )
        if status == "completed"
        else None
    )
    return StageResult(stage_name=CRITIC_STAGE, status=status, error=error, findings=findings)


class TestRenderProgressDisplay:
    def test_shows_pending_critic_row_while_pipeline_running(self):
        fake_session = {
            "pipeline_progress": {
                "stage_results": [_make_stage_results()[0]],
                "pipeline_result": None,
            }
        }

        with patch("app.st") as mock_st:
            mock_st.session_state = fake_session
            from app import render_progress_display

            render_progress_display()

        write_calls = [call[0][0] for call in mock_st.write.call_args_list]
        assert any("Independent Review" in call for call in write_calls)

    def test_hides_critic_row_after_finished_run_when_no_critic_stage_exists(self):
        fake_session = {
            "pipeline_progress": {
                "stage_results": _make_stage_results(),
                "pipeline_result": PipelineResult(status="complete", stage_results=_make_stage_results()),
            }
        }

        with patch("app.st") as mock_st, patch("app._finalize_run") as mock_finalize:
            mock_st.session_state = fake_session
            from app import render_progress_display

            render_progress_display()

        write_calls = [call[0][0] for call in mock_st.write.call_args_list]
        assert all("Independent Review" not in call for call in write_calls)
        mock_finalize.assert_called_once()

    def test_shows_critic_failure_status_when_critic_stage_failed(self):
        stage_results = _make_stage_results()
        critic_result = _make_critic_stage_result(status="failed", error="critic timeout")
        stage_results.append(critic_result)
        fake_session = {
            "pipeline_progress": {
                "stage_results": stage_results,
                "pipeline_result": PipelineResult(status="partial", stage_results=stage_results),
            }
        }

        with patch("app.st") as mock_st, patch("app._finalize_run"):
            mock_st.session_state = fake_session
            from app import render_progress_display

            render_progress_display()

        write_calls = [call[0][0] for call in mock_st.write.call_args_list]
        caption_calls = [call[0][0] for call in mock_st.caption.call_args_list]
        assert any("Independent Review" in call for call in write_calls)
        assert "Error: critic timeout" in caption_calls


class TestFinalizeRunStatusTransitions:
    def test_complete_pipeline_sets_run_status_to_complete(self, sample_run_input):
        analysis_run = create_analysis_run(sample_run_input)
        assert analysis_run.status == "running"

        stage_results = _make_stage_results()
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
        stage_results = _make_stage_results()
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
        pipeline_result = PipelineResult(status="complete", stage_results=_make_stage_results())

        fake_session = {"analysis_run": analysis_run}

        with patch("app.st") as mock_st, patch("app.save_run") as mock_save:
            mock_st.session_state = fake_session
            from app import _finalize_run

            _finalize_run(pipeline_result)
            _finalize_run(pipeline_result)

            mock_save.assert_called_once()

    def test_save_run_failure_logs_and_shows_error(self, sample_run_input, caplog):
        analysis_run = create_analysis_run(sample_run_input)
        pipeline_result = PipelineResult(status="complete", stage_results=_make_stage_results())

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
