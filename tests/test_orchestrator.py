import logging
from collections.abc import Callable

from models import AgentFindings, StageResult
from orchestrator import PIPELINE_STAGES, PipelineResult, _resolve_pipeline_status, run_pipeline


def make_success_agent(stage_name: str) -> Callable:
    def agent(_run_input) -> AgentFindings:
        return AgentFindings(
            sources=[f"https://example.com/{stage_name}"],
            confidence=0.8,
            key_findings=[f"{stage_name} finding"],
            evidence_gaps=[],
        )

    return agent


def make_failing_agent(error: Exception) -> Callable:
    def agent(_run_input):
        raise error

    return agent


def build_success_agents() -> dict[str, Callable]:
    return {stage_name: make_success_agent(stage_name) for stage_name in PIPELINE_STAGES}


def test_run_pipeline_returns_complete_when_all_stages_succeed(sample_run_input):
    result = run_pipeline(sample_run_input, build_success_agents())

    assert isinstance(result, PipelineResult)
    assert result.status == "complete"
    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES
    assert all(stage_result.status == "completed" for stage_result in result.stage_results)
    assert all(stage_result.error is None for stage_result in result.stage_results)
    assert all(stage_result.findings is not None for stage_result in result.stage_results)


def test_run_pipeline_returns_partial_when_one_stage_fails(sample_run_input):
    agents = build_success_agents()
    agents["competition"] = make_failing_agent(RuntimeError("competition timeout"))

    result = run_pipeline(sample_run_input, agents)

    assert result.status == "partial"
    assert [stage_result.status for stage_result in result.stage_results] == [
        "completed",
        "failed",
        "completed",
        "completed",
    ]
    failed_stage = result.stage_results[1]
    assert failed_stage.stage_name == "competition"
    assert failed_stage.error == "competition timeout"
    assert failed_stage.findings is None


def test_run_pipeline_returns_failed_when_all_stages_fail(sample_run_input):
    agents = {stage_name: make_failing_agent(RuntimeError(f"{stage_name} failed")) for stage_name in PIPELINE_STAGES}

    result = run_pipeline(sample_run_input, agents)

    assert result.status == "failed"
    assert len(result.stage_results) == 4
    assert all(stage_result.status == "failed" for stage_result in result.stage_results)
    assert [stage_result.error for stage_result in result.stage_results] == [
        "market failed",
        "competition failed",
        "product failed",
        "risk failed",
    ]


def test_run_pipeline_calls_callback_once_per_stage_in_order(sample_run_input):
    callback_results: list[StageResult] = []

    result = run_pipeline(sample_run_input, build_success_agents(), on_stage_update=callback_results.append)

    assert len(callback_results) == 4
    assert [stage_result.stage_name for stage_result in callback_results] == PIPELINE_STAGES
    assert [stage_result.model_dump() for stage_result in callback_results] == [
        stage_result.model_dump() for stage_result in result.stage_results
    ]


def test_run_pipeline_passes_success_and_failure_payloads_to_callback(sample_run_input):
    agents = build_success_agents()
    agents["product"] = make_failing_agent(TimeoutError("product timed out"))
    callback_results: list[StageResult] = []

    run_pipeline(sample_run_input, agents, on_stage_update=callback_results.append)

    assert [stage_result.stage_name for stage_result in callback_results] == PIPELINE_STAGES
    assert [stage_result.status for stage_result in callback_results] == [
        "completed",
        "completed",
        "failed",
        "completed",
    ]
    assert callback_results[0].findings is not None
    assert callback_results[0].error is None
    assert callback_results[2].findings is None
    assert callback_results[2].error == "product timed out"


def test_run_pipeline_logs_failures_with_logging_error(sample_run_input, caplog):
    agents = build_success_agents()
    agents["market"] = make_failing_agent(OSError("network down"))

    with caplog.at_level(logging.ERROR):
        result = run_pipeline(sample_run_input, agents)

    assert result.status == "partial"
    assert "Stage 'market' failed: network down" in caplog.text


def test_run_pipeline_raises_for_missing_agent_registration(sample_run_input):
    agents = build_success_agents()
    agents.pop("risk")

    try:
        run_pipeline(sample_run_input, agents)
    except ValueError as exc:
        assert str(exc) == "Missing agent registrations for stages: risk"
    else:
        raise AssertionError("run_pipeline should fail fast when a stage registration is missing")


def test_run_pipeline_treats_none_findings_as_stage_failure(sample_run_input, caplog):
    def none_agent(_run_input):
        return None

    agents = build_success_agents()
    agents["competition"] = none_agent

    with caplog.at_level(logging.ERROR):
        result = run_pipeline(sample_run_input, agents)

    failed_stage = result.stage_results[1]
    assert result.status == "partial"
    assert failed_stage.stage_name == "competition"
    assert failed_stage.status == "failed"
    assert failed_stage.findings is None
    assert failed_stage.error == "Stage 'competition' returned no findings"
    assert "Stage 'competition' failed: Stage 'competition' returned no findings" in caplog.text


def test_run_pipeline_continues_when_stage_callback_fails(sample_run_input, caplog):
    callback_results: list[StageResult] = []

    def failing_callback(stage_result: StageResult) -> None:
        callback_results.append(stage_result)
        raise RuntimeError("callback broke")

    with caplog.at_level(logging.ERROR):
        result = run_pipeline(sample_run_input, build_success_agents(), on_stage_update=failing_callback)

    assert result.status == "complete"
    assert len(result.stage_results) == 4
    assert len(callback_results) == 4
    assert "Stage update callback failed for 'market': callback broke" in caplog.text
    assert "Stage update callback failed for 'risk': callback broke" in caplog.text


def test_run_pipeline_continues_after_early_failure(sample_run_input):
    execution_order: list[str] = []

    def make_tracking_success_agent(stage_name: str) -> Callable:
        def agent(_run_input) -> AgentFindings:
            execution_order.append(stage_name)
            return AgentFindings(
                sources=[f"https://example.com/{stage_name}"],
                confidence=0.8,
                key_findings=[f"{stage_name} finding"],
                evidence_gaps=[],
            )

        return agent

    def failing_agent(_run_input):
        execution_order.append("market")
        raise RuntimeError("market failed")

    agents = {stage_name: make_tracking_success_agent(stage_name) for stage_name in PIPELINE_STAGES}
    agents["market"] = failing_agent

    result = run_pipeline(sample_run_input, agents)

    assert execution_order == PIPELINE_STAGES
    assert len(result.stage_results) == 4
    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES
    assert result.stage_results[0].status == "failed"
    assert all(stage_result.status == "completed" for stage_result in result.stage_results[1:])


def test_run_pipeline_accepts_no_op_callback(sample_run_input):
    result = run_pipeline(sample_run_input, build_success_agents(), on_stage_update=lambda _stage_result: None)

    assert result.status == "complete"


def test_resolve_pipeline_status_uses_stage_results_length():
    stage_results = [StageResult(stage_name="market", status="completed", findings=None)]

    assert _resolve_pipeline_status(stage_results) == "complete"
