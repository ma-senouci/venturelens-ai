from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel

from models import (
    CompetitionFindings,
    CriticFindings,
    MarketFindings,
    MemoOutput,
    ProductFindings,
    RiskFindings,
    RunInput,
    StageResult,
)


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


class FakeRecommendationPayload(BaseModel):
    executive_summary: str
    recommendation: Literal["Invest", "Watch", "Pass"]


def _fake_load_crewai_runtime():
    return FakeAgent, FakeCrew, FakeProcess, FakeTask


def _make_stage_results(
    *,
    include_product: bool = True,
    include_risk: bool = True,
    include_critic: bool = True,
) -> list[StageResult]:
    stage_results = [
        StageResult(
            stage_name="market",
            status="completed",
            findings=MarketFindings(
                key_findings=["Growing TAM", "Fast pilot traction"],
                evidence_gaps=["Need pricing data"],
                sources=["https://example.com/shared", "https://example.com/market"],
                confidence=0.9,
            ),
        ),
        StageResult(
            stage_name="competition",
            status="completed",
            findings=CompetitionFindings(
                key_findings=["Fast pilot traction", "Crowded space"],
                evidence_gaps=["No pricing benchmarks"],
                sources=["https://example.com/shared", "https://example.com/competition"],
                confidence=0.8,
            ),
        ),
        StageResult(
            stage_name="product",
            status="completed" if include_product else "failed",
            error=None if include_product else "timeout",
            findings=(
                ProductFindings(
                    key_findings=["Strong PMF signal"],
                    evidence_gaps=["Limited retention proof"],
                    sources=["https://example.com/product"],
                    confidence=0.76,
                )
                if include_product
                else None
            ),
        ),
        StageResult(
            stage_name="risk",
            status="completed" if include_risk else "failed",
            error=None if include_risk else "timeout",
            findings=(
                RiskFindings(
                    key_findings=["Customer concentration risk"],
                    evidence_gaps=["No churn disclosure"],
                    sources=["https://example.com/risk", "https://example.com/shared"],
                    confidence=0.71,
                )
                if include_risk
                else None
            ),
        ),
    ]

    if include_critic:
        stage_results.append(
            StageResult(
                stage_name="critic",
                status="completed",
                findings=CriticFindings(
                    contradictions=["Market growth conflicts with churn signals"],
                    weak_assumptions=["Assumes enterprise expansion is easy"],
                    unsupported_claims=["Category leadership is unproven"],
                    open_questions=["What is retention by cohort?"],
                    sources=["https://example.com/review", "https://example.com/shared"],
                    confidence=0.67,
                ),
            )
        )

    return stage_results


@pytest.fixture(autouse=True)
def _patch_crewai_runtime(monkeypatch):
    import memo_generator

    monkeypatch.setattr(memo_generator, "_load_crewai_runtime", _fake_load_crewai_runtime)


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


def test_generate_recommendation_returns_memo_output(basic_run_input, fake_settings):
    from memo_generator import generate_recommendation

    FakeCrew._kickoff_return = FakeCrewOutput(
        pydantic=FakeRecommendationPayload(
            executive_summary="Strong demand signals with manageable diligence gaps.",
            recommendation="Invest",
        )
    )

    result = generate_recommendation(basic_run_input, fake_settings, _make_stage_results())

    assert isinstance(result, MemoOutput)
    assert result.executive_summary == "Strong demand signals with manageable diligence gaps."
    assert result.recommendation == "Invest"
    assert result.independent_review is not None
    assert set(result.research_findings.keys()) == {"market", "competition", "product", "risk"}


def test_generate_recommendation_returns_memo_output_when_critic_missing(basic_run_input, fake_settings):
    from memo_generator import generate_recommendation

    FakeCrew._kickoff_return = FakeCrewOutput(
        pydantic=FakeRecommendationPayload(
            executive_summary="Promising, but missing the independent review signal.",
            recommendation="Watch",
        )
    )

    result = generate_recommendation(basic_run_input, fake_settings, _make_stage_results(include_critic=False))

    assert isinstance(result, MemoOutput)
    assert result.recommendation == "Watch"
    assert result.independent_review is None
    assert "Independent Review not available" in result.confidence_factors


def test_generate_recommendation_raises_when_llm_returns_no_pydantic(basic_run_input, fake_settings):
    from memo_generator import generate_recommendation

    FakeCrew._kickoff_return = FakeCrewOutput(pydantic=None)

    with pytest.raises(ValueError, match="recommendation agent returned no structured output"):
        generate_recommendation(basic_run_input, fake_settings, _make_stage_results())


def test_confidence_is_lowered_when_perspectives_are_missing(basic_run_input, fake_settings):
    from memo_generator import generate_recommendation

    FakeCrew._kickoff_return = FakeCrewOutput(
        pydantic=FakeRecommendationPayload(
            executive_summary="Complete run summary.",
            recommendation="Invest",
        )
    )
    complete = generate_recommendation(basic_run_input, fake_settings, _make_stage_results())

    FakeCrew._kickoff_return = FakeCrewOutput(
        pydantic=FakeRecommendationPayload(
            executive_summary="Partial run summary.",
            recommendation="Watch",
        )
    )
    partial = generate_recommendation(
        basic_run_input,
        fake_settings,
        _make_stage_results(include_product=False),
    )

    assert partial.confidence < complete.confidence


def test_confidence_factors_include_positive_and_negative_signals():
    from memo_generator import _build_confidence_factors

    factors = _build_confidence_factors(_make_stage_results(include_product=False))

    assert "Independent Review completed" in factors
    assert "Missing perspective: Product Positioning" in factors
    assert any("contradictions identified" in factor for factor in factors)


def test_unresolved_risks_are_collected_from_critic_and_risk_agent():
    from memo_generator import _collect_unresolved_risks

    risks = _collect_unresolved_risks(_make_stage_results())

    assert "Customer concentration risk" in risks
    assert "Market growth conflicts with churn signals" in risks


def test_open_questions_merge_critic_questions_with_evidence_gaps():
    from memo_generator import _collect_open_questions

    questions = _collect_open_questions(_make_stage_results())

    assert "What is retention by cohort?" in questions
    assert "Need pricing data" in questions
    assert "Limited retention proof" in questions


def test_sources_are_deduplicated_across_all_stages():
    from memo_generator import _collect_all_sources

    sources = _collect_all_sources(_make_stage_results())

    assert sources == [
        "https://example.com/shared",
        "https://example.com/market",
        "https://example.com/competition",
        "https://example.com/product",
        "https://example.com/risk",
        "https://example.com/review",
    ]


def test_format_recommendation_context_includes_research_and_critic_findings():
    from memo_generator import _format_recommendation_context

    context = _format_recommendation_context(_make_stage_results())

    assert "Market Research:" in context
    assert "1. Growing TAM" in context
    assert "Independent Review:" in context
    assert "Market growth conflicts with churn signals" in context


def test_calculate_base_confidence_exact_values():
    from memo_generator import _calculate_base_confidence

    # Full run: 1.0 - 0.05 (1 contradiction) - 0.03 (1 weak assumption) - 0.02 (1 unsupported claim)
    assert _calculate_base_confidence(_make_stage_results()) == pytest.approx(0.90)

    # Partial: 1.0 - 0.15 (missing product) - 0.10 (no critic)
    partial_confidence = _calculate_base_confidence(_make_stage_results(include_product=False, include_critic=False))
    assert partial_confidence == pytest.approx(0.75)


def test_build_recommendation_agent_delegates_to_generate_recommendation(basic_run_input, fake_settings, monkeypatch):
    captured = {}
    expected = object()

    def fake_generate_recommendation(run_input, settings, stage_results):
        captured["run_input"] = run_input
        captured["settings"] = settings
        captured["stage_results"] = stage_results
        return expected

    monkeypatch.setattr("memo_generator.generate_recommendation", fake_generate_recommendation)

    from memo_generator import build_recommendation_agent

    recommendation = build_recommendation_agent(fake_settings)
    stage_results = [StageResult(stage_name="market", status="completed")]

    result = recommendation(basic_run_input, stage_results)

    assert result is expected
    assert captured["run_input"] == basic_run_input
    assert captured["settings"] == fake_settings
    assert captured["stage_results"] == stage_results
