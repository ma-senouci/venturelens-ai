import logging
import os
from collections.abc import Callable

from config import Settings
from models import (
    AgentFindings,
    CompetitionFindings,
    MarketFindings,
    ProductFindings,
    RiskFindings,
    RunInput,
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


def validate_findings(findings: AgentFindings) -> AgentFindings:
    normalized = _normalize_findings(findings)
    if normalized.key_findings and len(normalized.sources) < len(normalized.key_findings):
        raise ValueError(
            f"Under-sourced result: {len(normalized.sources)} sources for {len(normalized.key_findings)} findings"
        )
    return normalized


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


def build_all_research_agents(
    settings: Settings,
) -> dict[str, Callable[[RunInput], AgentFindings]]:
    return {
        "market": lambda run_input: run_market_agent(run_input, settings),
        "competition": lambda run_input: run_competition_agent(run_input, settings),
        "product": lambda run_input: run_product_agent(run_input, settings),
        "risk": lambda run_input: run_risk_agent(run_input, settings),
    }
