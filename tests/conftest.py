import pytest

from models import AgentFindings, AnalysisRun, CriticFindings, MemoOutput, RunInput, StageResult


@pytest.fixture
def sample_run_input():
    return RunInput(startup_name="Acme Corp", website_url="https://acme.com", description="Rocket delivery startup")


@pytest.fixture
def sample_agent_findings():
    return AgentFindings(
        sources=["https://crunchbase.com/acme", "https://techcrunch.com/acme"],
        confidence=0.82,
        key_findings=["Growing TAM", "Strong founding team", "Early revenue traction"],
        evidence_gaps=["No public financials", "Limited competitive data"],
    )


@pytest.fixture
def sample_analysis_run(sample_run_input, sample_agent_findings):
    market_stage = StageResult(stage_name="market_research", status="completed", findings=sample_agent_findings)
    critic = CriticFindings(
        contradictions=["Market sizing assumptions are inconsistent"],
        weak_assumptions=["Customer acquisition costs stay flat"],
        unsupported_claims=["Enterprise demand is already proven"],
        open_questions=["How repeatable is the sales motion?"],
        sources=["https://example.com/review"],
        confidence=0.74,
    )
    critic_stage = StageResult(stage_name="critic", status="completed", findings=critic)
    memo = MemoOutput(
        executive_summary="Promising market with diligence gaps to resolve.",
        research_findings={"market_research": sample_agent_findings},
        independent_review=critic,
        recommendation="Watch",
        confidence=0.78,
        confidence_factors=["Large market", "Early traction"],
        unresolved_risks=["Unclear unit economics"],
        open_questions=["What is the 12-month retention profile?"],
        sources=[*sample_agent_findings.sources, "https://example.com/review"],
    )
    return AnalysisRun(
        status="complete",
        input=sample_run_input,
        stage_results=[market_stage, critic_stage],
        memo=memo,
    )
