from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from models import CompetitionFindings, MarketFindings, ProductFindings, RiskFindings, RunInput


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


def test_market_agent_returns_market_findings(basic_run_input, fake_settings):
    from agents import run_market_agent

    expected = MarketFindings(
        key_findings=["Growing TAM"],
        evidence_gaps=[],
        sources=["https://example.com/market"],
        confidence=0.85,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    result = run_market_agent(basic_run_input, fake_settings)

    assert isinstance(result, MarketFindings)
    assert result.key_findings == ["Growing TAM"]
    assert result.sources == ["https://example.com/market"]
    assert result.confidence == 0.85


def test_competition_agent_returns_competition_findings(basic_run_input, fake_settings):
    from agents import run_competition_agent

    expected = CompetitionFindings(
        key_findings=["Two direct competitors"],
        evidence_gaps=[],
        sources=["https://example.com/comp"],
        confidence=0.75,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    result = run_competition_agent(basic_run_input, fake_settings)

    assert isinstance(result, CompetitionFindings)
    assert result.key_findings == ["Two direct competitors"]
    assert result.sources == ["https://example.com/comp"]
    assert result.confidence == 0.75


def test_insufficient_info_preserves_evidence_gaps_and_lowers_confidence(basic_run_input, fake_settings):
    from agents import run_market_agent

    sparse = MarketFindings(
        key_findings=[],
        evidence_gaps=["No public market reports found", "TAM data unavailable"],
        sources=[],
        confidence=0.3,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=sparse)

    result = run_market_agent(basic_run_input, fake_settings)

    assert isinstance(result, MarketFindings)
    assert len(result.key_findings) == 0
    assert "No public market reports found" in result.evidence_gaps
    assert "TAM data unavailable" in result.evidence_gaps
    assert result.confidence == 0.3


def test_unsourced_findings_demoted_to_evidence_gaps(basic_run_input, fake_settings):
    from agents import run_market_agent

    unsourced = MarketFindings(
        key_findings=["Claim without backing"],
        evidence_gaps=[],
        sources=[],
        confidence=0.9,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=unsourced)

    result = run_market_agent(basic_run_input, fake_settings)

    assert len(result.key_findings) == 0
    assert any("Claim without backing" in gap for gap in result.evidence_gaps)
    assert result.confidence <= 0.2


def test_under_sourced_findings_trimmed_to_match_sources(basic_run_input, fake_settings):
    from agents import run_market_agent

    over_claimed = MarketFindings(
        key_findings=["Finding A", "Finding B", "Finding C"],
        evidence_gaps=[],
        sources=["https://example.com/only-one"],
        confidence=0.9,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=over_claimed)

    result = run_market_agent(basic_run_input, fake_settings)

    assert len(result.key_findings) <= len(result.sources)
    assert result.key_findings == ["Finding A"]
    assert any("Finding B" in gap for gap in result.evidence_gaps)
    assert any("Finding C" in gap for gap in result.evidence_gaps)
    assert result.confidence <= 0.5


def test_duplicate_sources_deduped(basic_run_input, fake_settings):
    from agents import run_market_agent

    with_dupes = MarketFindings(
        key_findings=["Finding"],
        evidence_gaps=[],
        sources=["https://example.com/a", "https://example.com/a", "https://example.com/a"],
        confidence=0.8,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=with_dupes)

    result = run_market_agent(basic_run_input, fake_settings)

    assert result.sources == ["https://example.com/a"]
    assert len(result.key_findings) <= len(result.sources)


def test_blank_strings_stripped_from_findings(basic_run_input, fake_settings):
    from agents import run_market_agent

    with_blanks = MarketFindings(
        key_findings=["Real finding", "", "  "],
        evidence_gaps=["Real gap", ""],
        sources=["https://example.com/a", "", "https://example.com/b"],
        confidence=0.8,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=with_blanks)

    result = run_market_agent(basic_run_input, fake_settings)

    assert "" not in result.key_findings
    assert "  " not in result.key_findings
    assert "" not in result.evidence_gaps
    assert "" not in result.sources
    assert result.key_findings == ["Real finding"]
    assert result.evidence_gaps == ["Real gap"]
    assert result.sources == ["https://example.com/a", "https://example.com/b"]


def test_agents_module_importable_without_crewai(monkeypatch):
    import importlib
    import sys

    if "agents" in sys.modules:
        saved = sys.modules.pop("agents")
    else:
        saved = None

    blocker = MagicMock()
    monkeypatch.setitem(sys.modules, "crewai", blocker)
    monkeypatch.setitem(sys.modules, "crewai_tools", blocker)

    try:
        mod = importlib.import_module("agents")
        assert mod is not None
        assert hasattr(mod, "_load_crewai_runtime")
    finally:
        if saved is not None:
            sys.modules["agents"] = saved


def test_no_none_pydantic_raises_value_error(basic_run_input, fake_settings):
    from agents import run_market_agent

    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=None)

    with pytest.raises(ValueError, match="returned no structured output"):
        run_market_agent(basic_run_input, fake_settings)


def test_product_agent_returns_product_findings(basic_run_input, fake_settings):
    from agents import run_product_agent

    expected = ProductFindings(
        key_findings=["Strong PMF signals"],
        evidence_gaps=[],
        sources=["https://example.com/product"],
        confidence=0.8,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    result = run_product_agent(basic_run_input, fake_settings)

    assert isinstance(result, ProductFindings)
    assert result.key_findings == ["Strong PMF signals"]
    assert result.sources == ["https://example.com/product"]
    assert result.confidence == 0.8


def test_risk_agent_returns_risk_findings(basic_run_input, fake_settings):
    from agents import run_risk_agent

    expected = RiskFindings(
        key_findings=["High burn rate"],
        evidence_gaps=[],
        sources=["https://example.com/risk"],
        confidence=0.7,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    result = run_risk_agent(basic_run_input, fake_settings)

    assert isinstance(result, RiskFindings)
    assert result.key_findings == ["High burn rate"]
    assert result.sources == ["https://example.com/risk"]
    assert result.confidence == 0.7


def test_build_all_research_agents_returns_all_four_keys(basic_run_input, fake_settings):
    from agents import build_all_research_agents

    expected = ProductFindings(
        key_findings=["PMF confirmed"],
        evidence_gaps=[],
        sources=["https://example.com/product"],
        confidence=0.8,
    )
    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=expected)

    agent_map = build_all_research_agents(fake_settings)

    assert set(agent_map.keys()) == {"market", "competition", "product", "risk"}
    for key in ("market", "competition", "product", "risk"):
        assert callable(agent_map[key])

    result = agent_map["product"](basic_run_input)
    assert isinstance(result, ProductFindings)
    assert result.key_findings == ["PMF confirmed"]


def test_build_critic_agent_delegates_to_run_critic_agent(basic_run_input, fake_settings, monkeypatch):
    from models import CriticFindings, StageResult

    captured = {}
    expected = CriticFindings(
        contradictions=["Needs review"],
        weak_assumptions=[],
        unsupported_claims=[],
        open_questions=[],
        sources=["https://example.com/review"],
        confidence=0.7,
    )

    def fake_run_critic_agent(run_input, settings, stage_results):
        captured["run_input"] = run_input
        captured["settings"] = settings
        captured["stage_results"] = stage_results
        return expected

    monkeypatch.setattr("agents.run_critic_agent", fake_run_critic_agent)

    from agents import build_critic_agent

    critic = build_critic_agent(fake_settings)
    stage_results = [StageResult(stage_name="market", status="completed")]

    result = critic(basic_run_input, stage_results)

    assert result is expected
    assert captured["run_input"] == basic_run_input
    assert captured["settings"] == fake_settings
    assert captured["stage_results"] == stage_results
