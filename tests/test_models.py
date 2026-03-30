import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from models import (
    AgentFindings,
    AnalysisRun,
    CompetitionFindings,
    CriticFindings,
    MarketFindings,
    MemoOutput,
    ProductFindings,
    RiskFindings,
    RunInput,
    StageResult,
)


def _assert_uuid4_string(value: str) -> None:
    parsed = uuid.UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def _assert_iso8601_datetime_string(value: str) -> None:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    assert "T" in value
    assert parsed.tzinfo is not None


class TestRunInput:
    def test_minimal_creation_defaults_optionals_to_none(self):
        run = RunInput(startup_name="Acme Corp")
        assert run.startup_name == "Acme Corp"
        assert run.website_url is None
        assert run.description is None
        assert run.thesis is None
        assert run.analysis_focus is None

    def test_auto_generates_uuid4_ids_and_iso8601_timestamps(self):
        run_a = RunInput(startup_name="A")
        run_b = RunInput(startup_name="B")
        _assert_uuid4_string(run_a.id)
        _assert_uuid4_string(run_b.id)
        _assert_iso8601_datetime_string(run_a.created_at)
        _assert_iso8601_datetime_string(run_b.created_at)
        assert run_a.id != run_b.id

    def test_startup_name_required(self):
        with pytest.raises(ValidationError):
            RunInput()

    @pytest.mark.parametrize(("field_name", "value"), [("id", "not-a-uuid"), ("created_at", "not-a-date")])
    def test_rejects_invalid_contract_strings(self, field_name, value):
        with pytest.raises(ValidationError):
            RunInput(startup_name="Acme Corp", **{field_name: value})

    def test_round_trip_preserves_all_fields(self):
        original = RunInput(startup_name="Acme Corp", website_url="https://acme.com")
        restored = RunInput.model_validate_json(original.model_dump_json())
        assert original == restored


class TestAgentFindings:
    def test_sources_and_confidence_are_required(self):
        with pytest.raises(ValidationError):
            AgentFindings(key_findings=["Strong growth"], evidence_gaps=["No financials"])

    def test_confidence_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentFindings(sources=[], confidence=-0.1, key_findings=[], evidence_gaps=[])
        with pytest.raises(ValidationError):
            AgentFindings(sources=[], confidence=1.1, key_findings=[], evidence_gaps=[])

    def test_round_trip_json(self):
        original = AgentFindings(sources=["s1"], confidence=0.75, key_findings=["f1"], evidence_gaps=["g1"])
        restored = AgentFindings.model_validate_json(original.model_dump_json())
        assert original == restored


class TestSpecializedFindings:
    def test_specialized_findings_share_agent_contract(self):
        for cls in (MarketFindings, CompetitionFindings, ProductFindings, RiskFindings):
            finding = cls(sources=["s1"], confidence=0.5, key_findings=["f1"], evidence_gaps=["g1"])
            assert isinstance(finding, AgentFindings)
            restored = cls.model_validate_json(finding.model_dump_json())
            assert finding == restored


class TestCriticFindings:
    def test_round_trip_json(self):
        original = CriticFindings(
            contradictions=["c1"],
            weak_assumptions=["w1"],
            unsupported_claims=["u1"],
            open_questions=["q1"],
            sources=["s1"],
            confidence=0.7,
        )
        restored = CriticFindings.model_validate_json(original.model_dump_json())
        assert original == restored


class TestStageResult:
    def test_completed_with_findings(self):
        findings = AgentFindings(sources=["s1"], confidence=0.8, key_findings=["f1"], evidence_gaps=[])
        stage_result = StageResult(stage_name="market_research", status="completed", findings=findings)
        assert stage_result.findings is not None
        assert stage_result.error is None

    def test_failed_with_error(self):
        stage_result = StageResult(stage_name="market_research", status="failed", error="API timeout")
        assert stage_result.error == "API timeout"
        assert stage_result.findings is None

    def test_completed_with_critic_findings(self):
        critic = CriticFindings(
            contradictions=["c1"],
            weak_assumptions=["w1"],
            unsupported_claims=[],
            open_questions=["q1"],
            sources=["s1"],
            confidence=0.6,
        )
        stage_result = StageResult(stage_name="critic", status="completed", findings=critic)
        assert isinstance(stage_result.findings, CriticFindings)
        assert stage_result.findings.contradictions == ["c1"]


class TestMemoOutput:
    def test_round_trip_json(self):
        findings = AgentFindings(sources=["s1"], confidence=0.8, key_findings=["f1"], evidence_gaps=[])
        original = MemoOutput(
            executive_summary="Summary",
            research_findings={"market": findings},
            independent_review=None,
            recommendation="Pass",
            confidence=0.3,
            confidence_factors=["Weak market"],
            unresolved_risks=["No traction"],
            open_questions=["Revenue model?"],
            sources=["s1"],
        )
        restored = MemoOutput.model_validate_json(original.model_dump_json())
        assert original == restored


class TestAnalysisRun:
    def test_full_nested_structure(self):
        run_input = RunInput(startup_name="Acme Corp")
        findings = AgentFindings(sources=["s1"], confidence=0.8, key_findings=["f1"], evidence_gaps=[])
        stage = StageResult(stage_name="market", status="completed", findings=findings)
        analysis = AnalysisRun(status="running", input=run_input, stage_results=[stage])
        assert analysis.status == "running"
        assert analysis.input.startup_name == "Acme Corp"
        assert len(analysis.stage_results) == 1
        assert analysis.memo is None
        _assert_uuid4_string(analysis.id)
        _assert_iso8601_datetime_string(analysis.created_at)

    @pytest.mark.parametrize(("field_name", "value"), [("id", "not-a-uuid"), ("created_at", "not-a-date")])
    def test_rejects_invalid_contract_strings(self, field_name, value):
        with pytest.raises(ValidationError):
            AnalysisRun(
                status="pending",
                input=RunInput(startup_name="Acme Corp"),
                stage_results=[],
                **{field_name: value},
            )

    def test_round_trip_preserves_nested_ids_and_timestamps(self, sample_analysis_run):
        original = sample_analysis_run
        restored = AnalysisRun.model_validate_json(original.model_dump_json())
        assert original == restored
        assert original.id == restored.id
        assert original.created_at == restored.created_at
        assert original.input.id == restored.input.id
        assert original.input.created_at == restored.input.created_at
