import logging

from models import AgentFindings, RunInput, StageResult
from orchestrator import PIPELINE_STAGES, PipelineResult


def _make_fake_agents():
    def agent(run_input: RunInput) -> AgentFindings:
        return AgentFindings(
            sources=[f"https://example.com/{run_input.startup_name}"],
            confidence=0.8,
            key_findings=["finding"],
            evidence_gaps=[],
        )

    return {stage: agent for stage in PIPELINE_STAGES}


def _make_fake_pipeline_result(status="complete"):
    return PipelineResult(
        status=status,
        stage_results=[StageResult(stage_name=stage, status="completed") for stage in PIPELINE_STAGES],
    )


class TestRunAnalysisPipeline:
    def test_returns_pipeline_result_with_correct_status(self, sample_run_input, monkeypatch):
        expected = _make_fake_pipeline_result("complete")
        monkeypatch.setattr("pipeline_runner.build_all_research_agents", lambda _settings: _make_fake_agents())
        monkeypatch.setattr(
            "pipeline_runner.run_pipeline",
            lambda run_input, agents, on_stage_update=None, critic_agent=None, recommendation_agent=None: expected,
        )

        from config import Settings
        from pipeline_runner import run_analysis_pipeline

        settings = Settings(openai_api_key="k", openai_model_name="m", serper_api_key="s")
        result = run_analysis_pipeline(sample_run_input, settings, on_stage_update=lambda _: None)

        assert result is expected
        assert result.status == "complete"

    def test_passes_callback_and_critic_to_run_pipeline(self, sample_run_input, monkeypatch):
        captured = {}

        def fake_run_pipeline(run_input, agents, on_stage_update=None, critic_agent=None, recommendation_agent=None):
            captured["cb"] = on_stage_update
            captured["critic"] = critic_agent
            captured["recommendation"] = recommendation_agent
            return _make_fake_pipeline_result()

        monkeypatch.setattr("pipeline_runner.build_all_research_agents", lambda _settings: _make_fake_agents())
        monkeypatch.setattr("pipeline_runner.run_pipeline", fake_run_pipeline)

        from config import Settings
        from pipeline_runner import run_analysis_pipeline

        def sentinel(_):
            return None

        settings = Settings(openai_api_key="k", openai_model_name="m", serper_api_key="s")
        run_analysis_pipeline(sample_run_input, settings, on_stage_update=sentinel)

        assert captured["cb"] is sentinel
        assert callable(captured["critic"])
        assert callable(captured["recommendation"])


class TestBuildSessionStateCallback:
    def test_writes_stage_result_to_dict(self):
        from pipeline_runner import build_session_state_callback

        progress = {"stage_results": [], "pipeline_result": None}
        callback = build_session_state_callback(progress)
        stage_result = StageResult(stage_name="market", status="completed")

        callback(stage_result)

        assert len(progress["stage_results"]) == 1
        assert progress["stage_results"][0] is stage_result


class TestStartPipelineThread:
    def test_thread_writes_final_result_to_progress_dict(self, sample_run_input, monkeypatch):
        expected = _make_fake_pipeline_result("complete")
        monkeypatch.setattr("pipeline_runner.build_all_research_agents", lambda _settings: _make_fake_agents())
        monkeypatch.setattr(
            "pipeline_runner.run_pipeline",
            lambda run_input, agents, on_stage_update=None, critic_agent=None, recommendation_agent=None: expected,
        )

        from config import Settings
        from pipeline_runner import start_pipeline_thread

        progress = {"stage_results": [], "pipeline_result": None}
        settings = Settings(openai_api_key="k", openai_model_name="m", serper_api_key="s")
        thread = start_pipeline_thread(sample_run_input, settings, progress)
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert progress["pipeline_result"] is expected

    def test_thread_is_daemon(self, sample_run_input, monkeypatch):
        monkeypatch.setattr("pipeline_runner.build_all_research_agents", lambda _settings: _make_fake_agents())
        monkeypatch.setattr(
            "pipeline_runner.run_pipeline",
            lambda run_input, agents, on_stage_update=None, critic_agent=None, recommendation_agent=None: (
                _make_fake_pipeline_result()
            ),
        )

        from config import Settings
        from pipeline_runner import start_pipeline_thread

        progress = {"stage_results": [], "pipeline_result": None}
        settings = Settings(openai_api_key="k", openai_model_name="m", serper_api_key="s")
        thread = start_pipeline_thread(sample_run_input, settings, progress)
        assert thread.daemon is True
        thread.join(timeout=5)

    def test_thread_error_does_not_raise_logs_instead(self, sample_run_input, monkeypatch, caplog):
        def exploding_run_pipeline(
            run_input,
            agents,
            on_stage_update=None,
            critic_agent=None,
            recommendation_agent=None,
        ):
            raise RuntimeError("catastrophic failure")

        monkeypatch.setattr("pipeline_runner.build_all_research_agents", lambda _settings: _make_fake_agents())
        monkeypatch.setattr("pipeline_runner.run_pipeline", exploding_run_pipeline)

        from config import Settings
        from pipeline_runner import start_pipeline_thread

        progress = {"stage_results": [], "pipeline_result": None}
        settings = Settings(openai_api_key="k", openai_model_name="m", serper_api_key="s")

        with caplog.at_level(logging.ERROR):
            thread = start_pipeline_thread(sample_run_input, settings, progress)
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert progress["pipeline_result"] is not None
        assert progress["pipeline_result"].status == "failed"
        assert "catastrophic failure" in caplog.text
