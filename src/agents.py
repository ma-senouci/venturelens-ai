import logging
import os
from collections.abc import Callable

from config import Settings
from models import (
    AgentFindings,
    CompetitionFindings,
    CriticFindings,
    MarketFindings,
    ProductFindings,
    RiskFindings,
    RunInput,
    StageResult,
)

logger = logging.getLogger(__name__)

_MARKET_PROMPT = """\
You are a market research analyst. Given the following startup information, \
research the market opportunity using public web sources.

Focus on:
- Market size or demand signals (TAM/SAM only when supported by public sources)
- Growth tailwinds and timing indicators
- Customer adoption patterns and early signals
- Explicit gaps where data is unavailable or inconclusive

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Return structured findings with sources for every claim. \
If evidence is too weak to support a finding, convert it into an evidence gap \
and lower your confidence score instead of returning an unsupported claim."""

_COMPETITION_PROMPT = """\
You are a competitive intelligence analyst. Given the following startup information, \
research the competitive landscape using public web sources.

Focus on:
- Direct competitors and their positioning
- Adjacent alternatives or substitutes
- Differentiation claims and defensibility
- Barriers to entry and switching costs
- Explicit gaps where data is unavailable or inconclusive

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Return structured findings with sources for every claim. \
If evidence is too weak to support a finding, convert it into an evidence gap \
and lower your confidence score instead of returning an unsupported claim."""

_PRODUCT_PROMPT = """\
You are a product positioning analyst. Given the following startup information, \
evaluate the product's market positioning using public web sources.

Focus on:
- Product-market fit signals and validation evidence
- Unique value proposition clarity and defensibility
- Positioning relative to alternatives and substitutes
- Go-to-market approach and early traction indicators
- Explicit gaps where data is unavailable or inconclusive

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Return structured findings with sources for every claim. \
If evidence is too weak to support a finding, convert it into an evidence gap \
and lower your confidence score instead of returning an unsupported claim."""

_RISK_PROMPT = """\
You are a risk assessment analyst. Given the following startup information, \
identify and evaluate key risks using public web sources.

Focus on:
- Market risks (shrinking demand, regulatory headwinds, macro factors)
- Execution risks (team gaps, operational complexity, scaling challenges)
- Financial risks (burn rate concerns, funding dependency, unit economics)
- Technology risks (technical debt indicators, platform dependencies)
- Competitive risks (incumbent response, low barriers to entry)
- Explicit gaps where data is unavailable or inconclusive

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Return structured findings with sources for every claim. \
If evidence is too weak to support a finding, convert it into an evidence gap \
and lower your confidence score instead of returning an unsupported claim."""

_CRITIC_PROMPT = """\
You are an independent review analyst. Given the following startup information and \
completed research findings, pressure-test the research before it is shown to an analyst.

Focus on:
- Contradictions between perspectives
- Weak assumptions that overreach the available evidence
- Unsupported claims that are not sufficiently backed by sources
- Open questions that remain unresolved
- Missing research perspectives and the impact of those gaps on the review

Startup: {startup_name}
Website: {website_url}
Description: {description}
Thesis: {thesis}
Analysis focus: {analysis_focus}

Research findings to review:
{findings_context}

Return structured critic findings with concise statements, references to relevant sources, \
and a calibrated confidence score."""

_CRITIC_STAGE_LABELS = {
    "market": "Market Research",
    "competition": "Competition Analysis",
    "product": "Product Positioning",
    "risk": "Risk Assessment",
}


def _load_crewai_runtime():
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import ScrapeWebsiteTool, SerperDevTool

    return Agent, Crew, Process, Task, ScrapeWebsiteTool, SerperDevTool


def _set_agent_env_vars(settings: Settings) -> None:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    os.environ["OPENAI_MODEL_NAME"] = settings.openai_model_name
    os.environ["SERPER_API_KEY"] = settings.serper_api_key


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


def _normalize_findings(findings: AgentFindings) -> AgentFindings:
    key_findings = [f.strip() for f in findings.key_findings if f and f.strip()]
    evidence_gaps = [g.strip() for g in findings.evidence_gaps if g and g.strip()]
    sources = list(dict.fromkeys(s.strip() for s in findings.sources if s and s.strip()))
    confidence = max(0.0, min(1.0, findings.confidence))

    # Weak-evidence guardrail: demote unsupported findings to evidence gaps
    if key_findings and not sources:
        evidence_gaps = evidence_gaps + [f"Insufficient sourcing for: {f}" for f in key_findings]
        key_findings = []
        confidence = min(confidence, 0.2)

    if sources and key_findings and len(sources) < len(key_findings):
        overflow = key_findings[len(sources) :]
        key_findings = key_findings[: len(sources)]
        evidence_gaps = evidence_gaps + [f"Dropped (under-sourced): {f}" for f in overflow]
        confidence = min(confidence, 0.5)

    return type(findings)(
        key_findings=key_findings,
        evidence_gaps=evidence_gaps,
        sources=sources,
        confidence=confidence,
    )


def _normalize_critic_findings(findings: CriticFindings) -> CriticFindings:
    return CriticFindings(
        contradictions=[item.strip() for item in findings.contradictions if item and item.strip()],
        weak_assumptions=[item.strip() for item in findings.weak_assumptions if item and item.strip()],
        unsupported_claims=[item.strip() for item in findings.unsupported_claims if item and item.strip()],
        open_questions=[item.strip() for item in findings.open_questions if item and item.strip()],
        sources=list(dict.fromkeys(item.strip() for item in findings.sources if item and item.strip())),
        confidence=max(0.0, min(1.0, findings.confidence)),
    )


def validate_findings(findings: AgentFindings) -> AgentFindings:
    normalized = _normalize_findings(findings)
    if normalized.key_findings and len(normalized.sources) < len(normalized.key_findings):
        raise ValueError(
            f"Under-sourced result: {len(normalized.sources)} sources for {len(normalized.key_findings)} findings"
        )
    return normalized


def _format_findings_for_critic(stage_results: list[StageResult]) -> str:
    serialized_sections: list[str] = []
    stage_results_by_name = {stage_result.stage_name: stage_result for stage_result in stage_results}

    for stage_name, stage_label in _CRITIC_STAGE_LABELS.items():
        stage_result = stage_results_by_name.get(stage_name)
        if (
            stage_result is None
            or stage_result.status != "completed"
            or stage_result.findings is None
            or not isinstance(stage_result.findings, AgentFindings)
        ):
            serialized_sections.append(f"{stage_label}: FAILED - not available for review")
            continue

        findings = stage_result.findings
        key_findings = findings.key_findings or []
        evidence_gaps = findings.evidence_gaps or []
        sources = findings.sources or []

        lines = [f"{stage_label}:"]
        if key_findings:
            lines.append("Key findings:")
            lines.extend(f"{index}. {finding}" for index, finding in enumerate(key_findings, 1))
        else:
            lines.append("Key findings: None provided")

        if evidence_gaps:
            lines.append("Evidence gaps:")
            lines.extend(f"- {gap}" for gap in evidence_gaps)

        if sources:
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in sources)
        else:
            lines.append("Sources: None provided")

        lines.append(f"Confidence: {findings.confidence:.2f}")
        serialized_sections.append("\n".join(lines))

    return "\n\n".join(serialized_sections)


def _run_crew_for_findings(
    run_input: RunInput,
    settings: Settings,
    prompt_template: str,
    findings_cls: type[AgentFindings],
    agent_role: str,
) -> AgentFindings:
    Agent, Crew, Process, Task, ScrapeWebsiteTool, SerperDevTool = _load_crewai_runtime()

    _set_agent_env_vars(settings)

    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    researcher = Agent(
        role=agent_role,
        goal=f"Research {run_input.startup_name} and produce structured findings",
        backstory=f"You are a specialist {agent_role} with deep domain expertise.",
        tools=[search_tool, scrape_tool],
        verbose=False,
    )

    task_description = _format_prompt(prompt_template, run_input)
    research_task = Task(
        description=task_description,
        expected_output=f"Structured {agent_role} findings with sources and confidence",
        agent=researcher,
        output_pydantic=findings_cls,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False,
    )

    logger.info("Running %s agent for '%s'", agent_role, run_input.startup_name)
    crew_output = crew.kickoff()

    raw_findings = crew_output.pydantic
    if raw_findings is None:
        raise ValueError(f"{agent_role} agent returned no structured output")

    return validate_findings(raw_findings)


def _run_crew_for_critic(
    run_input: RunInput,
    settings: Settings,
    stage_results: list[StageResult],
) -> CriticFindings:
    Agent, Crew, Process, Task, ScrapeWebsiteTool, SerperDevTool = _load_crewai_runtime()

    _set_agent_env_vars(settings)

    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    reviewer = Agent(
        role="independent review analyst",
        goal=f"Pressure-test the research for {run_input.startup_name} and produce structured critique findings",
        backstory="You are an independent analyst focused on contradictions, weak assumptions, and unsupported claims.",
        tools=[search_tool, scrape_tool],
        verbose=False,
    )

    findings_context = _format_findings_for_critic(stage_results)
    task_description = _format_prompt(_CRITIC_PROMPT, run_input).replace("{findings_context}", findings_context)
    review_task = Task(
        description=task_description,
        expected_output=(
            "Structured critic findings with contradictions, weak assumptions, "
            "unsupported claims, open questions, sources, and confidence"
        ),
        agent=reviewer,
        output_pydantic=CriticFindings,
    )

    crew = Crew(
        agents=[reviewer],
        tasks=[review_task],
        process=Process.sequential,
        verbose=False,
    )

    logger.info("Running critic agent for '%s'", run_input.startup_name)
    crew_output = crew.kickoff()

    raw_findings = crew_output.pydantic
    if raw_findings is None:
        raise ValueError("critic agent returned no structured output")

    return _normalize_critic_findings(raw_findings)


def run_market_agent(run_input: RunInput, settings: Settings) -> MarketFindings:
    return _run_crew_for_findings(
        run_input=run_input,
        settings=settings,
        prompt_template=_MARKET_PROMPT,
        findings_cls=MarketFindings,
        agent_role="market research analyst",
    )


def run_competition_agent(run_input: RunInput, settings: Settings) -> CompetitionFindings:
    return _run_crew_for_findings(
        run_input=run_input,
        settings=settings,
        prompt_template=_COMPETITION_PROMPT,
        findings_cls=CompetitionFindings,
        agent_role="competitive intelligence analyst",
    )


def run_product_agent(run_input: RunInput, settings: Settings) -> ProductFindings:
    return _run_crew_for_findings(
        run_input=run_input,
        settings=settings,
        prompt_template=_PRODUCT_PROMPT,
        findings_cls=ProductFindings,
        agent_role="product positioning analyst",
    )


def run_risk_agent(run_input: RunInput, settings: Settings) -> RiskFindings:
    return _run_crew_for_findings(
        run_input=run_input,
        settings=settings,
        prompt_template=_RISK_PROMPT,
        findings_cls=RiskFindings,
        agent_role="risk assessment analyst",
    )


def run_critic_agent(
    run_input: RunInput,
    settings: Settings,
    stage_results: list[StageResult],
) -> CriticFindings:
    return _run_crew_for_critic(
        run_input=run_input,
        settings=settings,
        stage_results=stage_results,
    )


def build_all_research_agents(
    settings: Settings,
) -> dict[str, Callable[[RunInput], AgentFindings]]:
    return {
        "market": lambda run_input: run_market_agent(run_input, settings),
        "competition": lambda run_input: run_competition_agent(run_input, settings),
        "product": lambda run_input: run_product_agent(run_input, settings),
        "risk": lambda run_input: run_risk_agent(run_input, settings),
    }


def build_critic_agent(
    settings: Settings,
) -> Callable[[RunInput, list[StageResult]], CriticFindings]:
    def critic(run_input: RunInput, stage_results: list[StageResult]) -> CriticFindings:
        return run_critic_agent(run_input, settings, stage_results)

    return critic
