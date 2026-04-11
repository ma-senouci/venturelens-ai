import logging
from collections.abc import Callable
from dataclasses import dataclass

from models import AgentFindings, RunInput, StageResult

logger = logging.getLogger(__name__)

PIPELINE_STAGES = ["market", "competition", "product", "risk"]
_AGENT_ERRORS = (ValueError, RuntimeError, TimeoutError, OSError)
_CALLBACK_ERRORS = (AttributeError, LookupError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    status: str
    stage_results: list[StageResult]


def _resolve_pipeline_status(stage_results: list[StageResult]) -> str:
    total_stages = len(stage_results)
    completed_count = sum(1 for stage_result in stage_results if stage_result.status == "completed")

    if total_stages == 0:
        return "failed"
    if completed_count == total_stages:
        return "complete"
    if completed_count == 0:
        return "failed"
    return "partial"


def _validate_run_agents(run_agents: dict[str, Callable[[RunInput], AgentFindings]]) -> None:
    missing_stage_names = [stage_name for stage_name in PIPELINE_STAGES if not callable(run_agents.get(stage_name))]
    if missing_stage_names:
        missing = ", ".join(missing_stage_names)
        raise ValueError(f"Missing agent registrations for stages: {missing}")


def _emit_stage_update(
    on_stage_update: Callable[[StageResult], None],
    stage_result: StageResult,
) -> None:
    try:
        on_stage_update(stage_result)
    except _CALLBACK_ERRORS as exc:
        logger.error("Stage update callback failed for '%s': %s", stage_result.stage_name, exc)


def run_pipeline(
    run_input: RunInput,
    run_agents: dict[str, Callable[[RunInput], AgentFindings]],
    on_stage_update: Callable[[StageResult], None] = lambda _stage_result: None,
) -> PipelineResult:
    _validate_run_agents(run_agents)
    stage_results: list[StageResult] = []

    for stage_name in PIPELINE_STAGES:
        try:
            agent = run_agents[stage_name]
            findings = agent(run_input)
            if findings is None:
                raise ValueError(f"Stage '{stage_name}' returned no findings")
            stage_result = StageResult(stage_name=stage_name, status="completed", findings=findings)
        except _AGENT_ERRORS as exc:
            logger.error("Stage '%s' failed: %s", stage_name, exc)
            stage_result = StageResult(stage_name=stage_name, status="failed", error=str(exc))

        stage_results.append(stage_result)
        _emit_stage_update(on_stage_update, stage_result)

    return PipelineResult(status=_resolve_pipeline_status(stage_results), stage_results=stage_results)
