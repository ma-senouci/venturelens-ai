import logging
from collections.abc import Callable

from models import AgentFindings, CriticFindings, MemoOutput, StageResult
from orchestrator import (
    CRITIC_STAGE,
    PIPELINE_STAGES,
    RECOMMENDATION_STAGE,
    PipelineResult,
    _resolve_pipeline_status,
    run_pipeline,
)


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


def make_success_critic() -> Callable:
    def critic(_run_input, _stage_results) -> CriticFindings:
        return CriticFindings(
            contradictions=["Test contradiction"],
            weak_assumptions=[],
            unsupported_claims=[],
            open_questions=[],
            sources=["https://example.com/review"],
            confidence=0.8,
        )

    return critic


def make_failing_critic(error: Exception) -> Callable:
    def critic(_run_input, _stage_results):
        raise error

    return critic


def make_success_recommendation() -> Callable:
    def recommendation(_run_input, _stage_results) -> MemoOutput:
        findings = AgentFindings(
            sources=["https://example.com/market"],
            confidence=0.8,
            key_findings=["market finding"],
            evidence_gaps=[],
        )
        return MemoOutput(
            executive_summary="Recommendation summary",
            research_findings={"market": findings},
            independent_review=None,
            recommendation="Watch",
            confidence=0.65,
            confidence_factors=["Independent Review not available"],
            unresolved_risks=["Open risk"],
            open_questions=["Open question"],
            sources=["https://example.com/market"],
        )

    return recommendation


def make_failing_recommendation(error: Exception) -> Callable:
    def recommendation(_run_input, _stage_results):
        raise error

    return recommendation


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


def test_run_pipeline_with_critic_runs_after_research_and_includes_critic(sample_run_input):
    result = run_pipeline(sample_run_input, build_success_agents(), critic_agent=make_success_critic())

    assert result.status == "complete"
    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES + [CRITIC_STAGE]
    critic_result = result.stage_results[-1]
    assert critic_result.status == "completed"
    assert isinstance(critic_result.findings, CriticFindings)
    assert critic_result.findings.contradictions == ["Test contradiction"]


def test_run_pipeline_with_critic_emits_callback_for_critic_stage(sample_run_input):
    callback_results: list[StageResult] = []

    run_pipeline(
        sample_run_input,
        build_success_agents(),
        on_stage_update=callback_results.append,
        critic_agent=make_success_critic(),
    )

    assert [stage_result.stage_name for stage_result in callback_results] == PIPELINE_STAGES + [CRITIC_STAGE]
    assert callback_results[-1].status == "completed"


def test_run_pipeline_with_critic_skips_when_all_research_stages_fail(sample_run_input):
    agents = {stage_name: make_failing_agent(RuntimeError(f"{stage_name} failed")) for stage_name in PIPELINE_STAGES}

    result = run_pipeline(sample_run_input, agents, critic_agent=make_success_critic())

    assert result.status == "failed"
    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES + [CRITIC_STAGE]
    critic_result = result.stage_results[-1]
    assert critic_result.status == "failed"
    assert critic_result.error == "No completed research findings to review"


def test_run_pipeline_with_critic_runs_on_partial_results(sample_run_input):
    agents = build_success_agents()
    agents["competition"] = make_failing_agent(RuntimeError("competition timeout"))

    result = run_pipeline(sample_run_input, agents, critic_agent=make_success_critic())

    assert result.status == "partial"
    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES + [CRITIC_STAGE]
    assert result.stage_results[1].status == "failed"
    assert result.stage_results[-1].status == "completed"


def test_run_pipeline_handles_critic_failure_gracefully(sample_run_input, caplog):
    with caplog.at_level(logging.ERROR):
        result = run_pipeline(
            sample_run_input,
            build_success_agents(),
            critic_agent=make_failing_critic(RuntimeError("critic timeout")),
        )

    assert result.status == "partial"
    critic_result = result.stage_results[-1]
    assert critic_result.stage_name == CRITIC_STAGE
    assert critic_result.status == "failed"
    assert critic_result.error == "critic timeout"
    assert "Stage 'critic' failed: critic timeout" in caplog.text


def test_run_pipeline_treats_none_critic_findings_as_failure(sample_run_input, caplog):
    def none_critic(_run_input, _stage_results):
        return None

    with caplog.at_level(logging.ERROR):
        result = run_pipeline(sample_run_input, build_success_agents(), critic_agent=none_critic)

    assert result.status == "partial"
    critic_result = result.stage_results[-1]
    assert critic_result.stage_name == CRITIC_STAGE
    assert critic_result.status == "failed"
    assert critic_result.error == "Critic agent returned no findings"
    assert "Stage 'critic' failed: Critic agent returned no findings" in caplog.text


def test_run_pipeline_with_recommendation_runs_after_critic_and_is_included(sample_run_input):
    result = run_pipeline(
        sample_run_input,
        build_success_agents(),
        critic_agent=make_success_critic(),
        recommendation_agent=make_success_recommendation(),
    )

    assert result.status == "complete"
    assert [stage_result.stage_name for stage_result in result.stage_results] == (
        PIPELINE_STAGES + [CRITIC_STAGE, RECOMMENDATION_STAGE]
    )
    recommendation_result = result.stage_results[-1]
    assert recommendation_result.status == "completed"
    assert recommendation_result.findings is None
    assert recommendation_result.error is None


def test_run_pipeline_with_recommendation_emits_callback(sample_run_input):
    callback_results: list[StageResult] = []

    run_pipeline(
        sample_run_input,
        build_success_agents(),
        on_stage_update=callback_results.append,
        critic_agent=make_success_critic(),
        recommendation_agent=make_success_recommendation(),
    )

    assert [stage_result.stage_name for stage_result in callback_results] == (
        PIPELINE_STAGES + [CRITIC_STAGE, RECOMMENDATION_STAGE]
    )
    assert callback_results[-1].status == "completed"


def test_run_pipeline_with_recommendation_skips_when_all_research_fail(sample_run_input):
    agents = {stage_name: make_failing_agent(RuntimeError(f"{stage_name} failed")) for stage_name in PIPELINE_STAGES}

    result = run_pipeline(
        sample_run_input,
        agents,
        critic_agent=make_success_critic(),
        recommendation_agent=make_success_recommendation(),
    )

    assert result.status == "failed"
    assert [stage_result.stage_name for stage_result in result.stage_results] == (
        PIPELINE_STAGES + [CRITIC_STAGE, RECOMMENDATION_STAGE]
    )
    recommendation_result = result.stage_results[-1]
    assert recommendation_result.status == "failed"
    assert recommendation_result.error == "No completed research findings to synthesize"


def test_run_pipeline_with_recommendation_runs_when_critic_fails(sample_run_input):
    result = run_pipeline(
        sample_run_input,
        build_success_agents(),
        critic_agent=make_failing_critic(RuntimeError("critic timeout")),
        recommendation_agent=make_success_recommendation(),
    )

    assert result.status == "partial"
    assert [stage_result.stage_name for stage_result in result.stage_results] == (
        PIPELINE_STAGES + [CRITIC_STAGE, RECOMMENDATION_STAGE]
    )
    assert result.stage_results[-2].status == "failed"
    assert result.stage_results[-1].status == "completed"


def test_run_pipeline_is_backward_compatible_when_recommendation_agent_omitted(sample_run_input):
    result = run_pipeline(sample_run_input, build_success_agents(), critic_agent=make_success_critic())

    assert [stage_result.stage_name for stage_result in result.stage_results] == PIPELINE_STAGES + [CRITIC_STAGE]


def test_run_pipeline_handles_recommendation_failure_gracefully(sample_run_input, caplog):
    with caplog.at_level(logging.ERROR):
        result = run_pipeline(
            sample_run_input,
            build_success_agents(),
            critic_agent=make_success_critic(),
            recommendation_agent=make_failing_recommendation(RuntimeError("recommendation timeout")),
        )

    assert result.status == "partial"
    recommendation_result = result.stage_results[-1]
    assert recommendation_result.stage_name == RECOMMENDATION_STAGE
    assert recommendation_result.status == "failed"
    assert recommendation_result.error == "recommendation timeout"
    assert "Stage 'recommendation' failed: recommendation timeout" in caplog.text
