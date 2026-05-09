import logging
import threading
from collections.abc import Callable

from agents import build_all_research_agents, build_critic_agent
from config import Settings
from memo_generator import build_recommendation_agent
from models import RunInput, StageResult
from orchestrator import PipelineResult, run_pipeline

logger = logging.getLogger(__name__)


def build_session_state_callback(
    progress: dict,
) -> Callable[[StageResult], None]:

    def on_stage_update(stage_result: StageResult) -> None:
        progress["stage_results"].append(stage_result)

    return on_stage_update


def run_analysis_pipeline(
    run_input: RunInput,
    settings: Settings,
    on_stage_update: Callable[[StageResult], None] = lambda _stage_result: None,
) -> PipelineResult:
    agents = build_all_research_agents(settings)
    critic = build_critic_agent(settings)
    recommendation = build_recommendation_agent(settings)
    return run_pipeline(
        run_input,
        agents,
        on_stage_update=on_stage_update,
        critic_agent=critic,
        recommendation_agent=recommendation,
    )


def start_pipeline_thread(
    run_input: RunInput,
    settings: Settings,
    progress: dict,
) -> threading.Thread:
    callback = build_session_state_callback(progress)

    def _target() -> None:
        try:
            result = run_analysis_pipeline(run_input, settings, on_stage_update=callback)
        except Exception:
            logger.exception("Pipeline thread failed")
            result = PipelineResult(status="failed", stage_results=[])
        progress["pipeline_result"] = result

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread
