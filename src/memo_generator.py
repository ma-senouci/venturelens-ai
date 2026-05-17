import logging
import os
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from config import OPENAI_API_KEY_ENV_VAR, OPENAI_MODEL_NAME_ENV_VAR, Settings
from models import AgentFindings, CriticFindings, MemoOutput, RunInput, StageResult

logger = logging.getLogger(__name__)

_RESEARCH_STAGE_LABELS = {
    "market": "Market Research",
    "competition": "Competition Analysis",
    "product": "Product Positioning",
    "risk": "Risk Assessment",
}

_RECOMMENDATION_PROMPT = """\
You are an investment recommendation analyst. Given the following startup information,
research findings, and independent review, produce a concise executive summary and
one recommendation: Invest, Watch, or Pass.

Focus on:
- Evidence strength across all completed research perspectives
- Contradictions, weak assumptions, and unsupported claims from the Independent Review
- Unresolved risks, open questions, and missing perspectives
- A defensible recommendation grounded in the available evidence

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Research and review context:
{recommendation_context}

Return only structured output with:
- executive_summary: concise synthesis for an analyst
- recommendation: Invest, Watch, or Pass
Do not invent confidence scores, sources, or rationale fields outside the schema."""


class _LLMRecommendation(BaseModel):
    executive_summary: str
    recommendation: Literal["Invest", "Watch", "Pass"]


def _load_crewai_runtime():
    from crewai import Agent, Crew, Process, Task

    return Agent, Crew, Process, Task


def _set_agent_env_vars(settings: Settings) -> None:
    os.environ[OPENAI_API_KEY_ENV_VAR] = settings.openai_api_key
    os.environ[OPENAI_MODEL_NAME_ENV_VAR] = settings.openai_model_name


def _format_prompt(template: str, run_input: RunInput) -> str:
    replacements = {
        "{startup_name}": run_input.startup_name,
        "{website_url}": run_input.website_url or "N/A",
        "{description}": run_input.description or "N/A",
        "{thesis}": run_input.thesis or "N/A",
        "{analysis_focus}": run_input.analysis_focus or "N/A",
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def _normalize_string_list(items: list[str]) -> list[str]:
    seen: dict[str, bool] = {}
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen[stripped] = True
    return list(seen.keys())


def _extract_research_findings(stage_results: list[StageResult]) -> dict[str, AgentFindings]:
    return {
        stage_result.stage_name: stage_result.findings
        for stage_result in stage_results
        if stage_result.status == "completed"
        and isinstance(stage_result.findings, AgentFindings)
        and stage_result.stage_name in _RESEARCH_STAGE_LABELS
    }


def _extract_critic_findings(stage_results: list[StageResult]) -> CriticFindings | None:
    for stage_result in stage_results:
        if (
            stage_result.stage_name == "critic"
            and stage_result.status == "completed"
            and isinstance(stage_result.findings, CriticFindings)
        ):
            return stage_result.findings
    return None


def _format_recommendation_context(stage_results: list[StageResult]) -> str:
    serialized_sections: list[str] = []
    stage_results_by_name = {stage_result.stage_name: stage_result for stage_result in stage_results}

    for stage_name, stage_label in _RESEARCH_STAGE_LABELS.items():
        stage_result = stage_results_by_name.get(stage_name)
        if (
            stage_result is None
            or stage_result.status != "completed"
            or not isinstance(stage_result.findings, AgentFindings)
        ):
            serialized_sections.append(f"{stage_label}: FAILED - not available for synthesis")
            continue

        findings = stage_result.findings
        lines = [f"{stage_label}:"]

        if findings.key_findings:
            lines.append("Key findings:")
            lines.extend(f"{index}. {finding}" for index, finding in enumerate(findings.key_findings, 1))
        else:
            lines.append("Key findings: None provided")

        if findings.evidence_gaps:
            lines.append("Evidence gaps:")
            lines.extend(f"- {gap}" for gap in findings.evidence_gaps)

        if findings.sources:
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in findings.sources)
        else:
            lines.append("Sources: None provided")

        lines.append(f"Confidence: {findings.confidence:.2f}")
        serialized_sections.append("\n".join(lines))

    critic_findings = _extract_critic_findings(stage_results)
    if critic_findings is None:
        serialized_sections.append("Independent Review: FAILED - not available for synthesis")
    else:
        review_lines = ["Independent Review:"]
        review_lines.append("Contradictions:" if critic_findings.contradictions else "Contradictions: None identified")
        review_lines.extend(f"- {item}" for item in critic_findings.contradictions)
        review_lines.append(
            "Weak assumptions:" if critic_findings.weak_assumptions else "Weak assumptions: None identified"
        )
        review_lines.extend(f"- {item}" for item in critic_findings.weak_assumptions)
        review_lines.append(
            "Unsupported claims:" if critic_findings.unsupported_claims else "Unsupported claims: None identified"
        )
        review_lines.extend(f"- {item}" for item in critic_findings.unsupported_claims)
        review_lines.append("Open questions:" if critic_findings.open_questions else "Open questions: None identified")
        review_lines.extend(f"- {item}" for item in critic_findings.open_questions)
        review_lines.append(
            "Missing perspectives:" if critic_findings.missing_perspectives else "Missing perspectives: None identified"
        )
        review_lines.extend(f"- {item}" for item in critic_findings.missing_perspectives)
        review_lines.append("Sources:" if critic_findings.sources else "Sources: None provided")
        review_lines.extend(f"- {item}" for item in critic_findings.sources)
        review_lines.append(f"Confidence: {critic_findings.confidence:.2f}")
        serialized_sections.append("\n".join(review_lines))

    return "\n\n".join(serialized_sections)


def _calculate_base_confidence(stage_results: list[StageResult]) -> float:
    research_findings = _extract_research_findings(stage_results)
    missing_research_count = len(_RESEARCH_STAGE_LABELS) - len(research_findings)
    critic_findings = _extract_critic_findings(stage_results)

    confidence = 1.0 - (missing_research_count * 0.15)
    if critic_findings is None:
        confidence -= 0.10
    else:
        confidence -= len(critic_findings.contradictions) * 0.05
        confidence -= len(critic_findings.weak_assumptions) * 0.03
        confidence -= len(critic_findings.unsupported_claims) * 0.02

    return max(0.1, min(1.0, confidence))


def _count_corroborating_findings(stage_results: list[StageResult]) -> int:
    findings_to_stages: dict[str, set[str]] = {}
    for stage_name, findings in _extract_research_findings(stage_results).items():
        for finding in findings.key_findings:
            normalized = finding.strip().casefold()
            if not normalized:
                continue
            findings_to_stages.setdefault(normalized, set()).add(stage_name)
    return sum(1 for stage_names in findings_to_stages.values() if len(stage_names) > 1)


def _build_confidence_factors(stage_results: list[StageResult]) -> list[str]:
    research_findings = _extract_research_findings(stage_results)
    critic_findings = _extract_critic_findings(stage_results)
    factors: list[str] = []

    if len(research_findings) == len(_RESEARCH_STAGE_LABELS):
        factors.append("All 4 research perspectives completed")
    elif research_findings:
        factors.append(f"{len(research_findings)} research perspectives completed")

    if critic_findings is None:
        factors.append("Independent Review not available")
    else:
        factors.append("Independent Review completed")

    corroborating_findings = _count_corroborating_findings(stage_results)
    if corroborating_findings:
        factors.append(f"{corroborating_findings} corroborating findings across perspectives")

    for stage_name, stage_label in _RESEARCH_STAGE_LABELS.items():
        if stage_name not in research_findings:
            factors.append(f"Missing perspective: {stage_label}")

    if critic_findings is not None:
        if critic_findings.contradictions:
            factors.append(f"{len(critic_findings.contradictions)} contradictions identified")
        if critic_findings.weak_assumptions:
            factors.append(f"{len(critic_findings.weak_assumptions)} weak assumptions flagged")
        if critic_findings.unsupported_claims:
            factors.append(f"{len(critic_findings.unsupported_claims)} unsupported claims found")

    evidence_gap_count = sum(len(findings.evidence_gaps) for findings in research_findings.values())
    if evidence_gap_count:
        factors.append(f"{evidence_gap_count} evidence gaps across research")

    return _normalize_string_list(factors)


def _collect_unresolved_risks(stage_results: list[StageResult]) -> list[str]:
    risks: list[str] = []
    research_findings = _extract_research_findings(stage_results)
    critic_findings = _extract_critic_findings(stage_results)

    risk_findings = research_findings.get("risk")
    if risk_findings is not None:
        risks.extend(risk_findings.key_findings)

    if critic_findings is not None:
        risks.extend(critic_findings.contradictions)
        risks.extend(critic_findings.weak_assumptions)
        risks.extend(critic_findings.unsupported_claims)

    return _normalize_string_list(risks)


def _collect_open_questions(stage_results: list[StageResult]) -> list[str]:
    questions: list[str] = []
    critic_findings = _extract_critic_findings(stage_results)

    if critic_findings is not None:
        questions.extend(critic_findings.open_questions)
        questions.extend(critic_findings.missing_perspectives)

    for findings in _extract_research_findings(stage_results).values():
        questions.extend(findings.evidence_gaps)

    return _normalize_string_list(questions)


def _collect_all_sources(stage_results: list[StageResult]) -> list[str]:
    sources: list[str] = []

    for findings in _extract_research_findings(stage_results).values():
        sources.extend(findings.sources)

    critic_findings = _extract_critic_findings(stage_results)
    if critic_findings is not None:
        sources.extend(critic_findings.sources)

    return _normalize_string_list(sources)


def _normalize_memo_output(
    memo: MemoOutput,
) -> MemoOutput:
    return MemoOutput(
        executive_summary=memo.executive_summary.strip(),
        research_findings=memo.research_findings,
        independent_review=memo.independent_review,
        recommendation=memo.recommendation,
        confidence=memo.confidence,
        confidence_factors=memo.confidence_factors,
        unresolved_risks=memo.unresolved_risks,
        open_questions=memo.open_questions,
        sources=memo.sources,
    )


def build_recommendation_agent(
    settings: Settings,
) -> Callable[[RunInput, list[StageResult]], MemoOutput]:
    def recommendation(run_input: RunInput, stage_results: list[StageResult]) -> MemoOutput:
        return generate_recommendation(run_input, settings, stage_results)

    return recommendation


def generate_recommendation(
    run_input: RunInput,
    settings: Settings,
    stage_results: list[StageResult],
) -> MemoOutput:
    Agent, Crew, Process, Task = _load_crewai_runtime()

    _set_agent_env_vars(settings)

    recommendation_agent = Agent(
        role="investment recommendation analyst",
        goal=f"Synthesize a recommendation memo for {run_input.startup_name}",
        backstory="You synthesize structured diligence findings into concise analyst-ready recommendations.",
        verbose=False,
    )

    recommendation_context = _format_recommendation_context(stage_results)
    task_description = _format_prompt(_RECOMMENDATION_PROMPT, run_input).replace(
        "{recommendation_context}",
        recommendation_context,
    )
    recommendation_task = Task(
        description=task_description,
        expected_output="Structured executive summary and recommendation",
        agent=recommendation_agent,
        output_pydantic=_LLMRecommendation,
    )

    crew = Crew(
        agents=[recommendation_agent],
        tasks=[recommendation_task],
        process=Process.sequential,
        verbose=False,
    )

    logger.info("Running recommendation agent for '%s'", run_input.startup_name)
    crew_output = crew.kickoff()
    llm_output = crew_output.pydantic
    if llm_output is None:
        raise ValueError("recommendation agent returned no structured output")

    base_confidence = _calculate_base_confidence(stage_results)
    confidence_factors = _build_confidence_factors(stage_results)
    unresolved_risks = _collect_unresolved_risks(stage_results)
    open_questions = _collect_open_questions(stage_results)
    sources = _collect_all_sources(stage_results)

    memo = MemoOutput.model_construct(
        executive_summary=llm_output.executive_summary,
        research_findings=_extract_research_findings(stage_results),
        independent_review=_extract_critic_findings(stage_results),
        recommendation=llm_output.recommendation,
        confidence=base_confidence,
        confidence_factors=confidence_factors,
        unresolved_risks=unresolved_risks,
        open_questions=open_questions,
        sources=sources,
    )

    return _normalize_memo_output(memo)
