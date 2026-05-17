from dataclasses import dataclass

import pytest

from models import CompetitionFindings, CriticFindings, MarketFindings, RunInput, StageResult


@dataclass
class FakeCrewOutput:
    pydantic: object = None


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCrew:
    _kickoff_return: FakeCrewOutput | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def kickoff(self):
        if FakeCrew._kickoff_return is not None:
            return FakeCrew._kickoff_return
        return FakeCrewOutput()


class FakeProcess:
    sequential = "sequential"


class FakeSerperDevTool:
    def __init__(self, **kwargs):
        pass


class FakeScrapeWebsiteTool:
    def __init__(self, **kwargs):
        pass


def _fake_load_crewai_runtime():
    return FakeAgent, FakeCrew, FakeProcess, FakeTask, FakeScrapeWebsiteTool, FakeSerperDevTool


def _make_completed_stages() -> list[StageResult]:
    return [
        StageResult(
            stage_name="market",
            status="completed",
            findings=MarketFindings(
                key_findings=["Growing TAM"],
                evidence_gaps=[],
                sources=["https://example.com/market"],
                confidence=0.85,
            ),
        ),
        StageResult(
            stage_name="competition",
            status="completed",
            findings=CompetitionFindings(
                key_findings=["Two direct competitors"],
                evidence_gaps=["No pricing data"],
                sources=["https://example.com/comp"],
                confidence=0.75,
            ),
        ),
    ]


@pytest.fixture(autouse=True)
def _patch_crewai_runtime(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "_load_crewai_runtime", _fake_load_crewai_runtime)


@pytest.fixture(autouse=True)
def _reset_fake_crew_state():
    yield
    FakeCrew._kickoff_return = None


@pytest.fixture(autouse=True)
def _patch_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")


@pytest.fixture
def fake_settings():
    from config import Settings

    return Settings(
        openai_api_key="test-key",
        openai_model_name="gpt-4o-mini",
        serper_api_key="test-serper-key",
    )


@pytest.fixture
def basic_run_input():
    return RunInput(
        startup_name="TestCo",
        website_url="https://testco.io",
        description="A widget company",
    )


def test_run_critic_agent_returns_critic_findings(basic_run_input, fake_settings):
    from agents import run_critic_agent

    expected = CriticFindings(
        contradictions=["Market claims large TAM but competition shows crowded space"],
        weak_assumptions=["Growth rate extrapolation is linear"],
        unsupported_claims=["Enterprise readiness not evidenced"],
        open_questions=["What is the actual customer retention?"],
        sources=["https://example.com/market", "https://example.com/comp"],
        confidence=0.72,
        missing_perspectives=["No customer interview evidence"],
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    result = run_critic_agent(basic_run_input, fake_settings, _make_completed_stages())

    assert isinstance(result, CriticFindings)
    assert result.contradictions == ["Market claims large TAM but competition shows crowded space"]
    assert result.weak_assumptions == ["Growth rate extrapolation is linear"]
    assert result.unsupported_claims == ["Enterprise readiness not evidenced"]
    assert result.open_questions == ["What is the actual customer retention?"]
    assert result.sources == ["https://example.com/market", "https://example.com/comp"]
    assert result.confidence == 0.72
    assert result.missing_perspectives == ["No customer interview evidence"]


def test_format_findings_for_critic_includes_completed_and_failed_stages():
    from agents import _format_findings_for_critic

    stage_results = _make_completed_stages() + [StageResult(stage_name="product", status="failed", error="timeout")]

    result = _format_findings_for_critic(stage_results)

    assert "Market Research:" in result
    assert "1. Growing TAM" in result
    assert "Competition Analysis:" in result
    assert "- No pricing data" in result
    assert "Product Positioning: FAILED - not available for review" in result
    assert "Risk Assessment: FAILED - not available for review" in result


def test_normalize_critic_findings_strips_dedupes_and_clamps():
    from agents import _normalize_critic_findings

    normalized = _normalize_critic_findings(
        CriticFindings.model_construct(
            contradictions=["  A conflict  ", "", " "],
            weak_assumptions=["  Fragile assumption ", ""],
            unsupported_claims=["  Unproven claim  "],
            open_questions=["  What is churn?  ", ""],
            sources=[" https://example.com/a ", "https://example.com/a", ""],
            confidence=3.0,
            missing_perspectives=["  No pricing validation  ", "", "No pricing validation"],
        )
    )

    assert normalized.contradictions == ["A conflict"]
    assert normalized.weak_assumptions == ["Fragile assumption"]
    assert normalized.unsupported_claims == ["Unproven claim"]
    assert normalized.open_questions == ["What is churn?"]
    assert normalized.sources == ["https://example.com/a"]
    assert normalized.confidence == 1.0
    assert normalized.missing_perspectives == ["No pricing validation"]


def test_run_critic_agent_raises_when_no_pydantic_output(basic_run_input, fake_settings):
    from agents import run_critic_agent

    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=None)

    with pytest.raises(ValueError, match="critic agent returned no structured output"):
        run_critic_agent(basic_run_input, fake_settings, _make_completed_stages())
