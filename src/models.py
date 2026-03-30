import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

UUID4_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"


def _validate_iso8601_datetime(value: str) -> str:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("must be a valid ISO 8601 datetime string") from exc

    if "T" not in value or parsed.tzinfo is None:
        raise ValueError("must be a timezone-aware ISO 8601 datetime string")

    return value


ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]
UUID4String = Annotated[str, Field(pattern=UUID4_PATTERN)]
ISO8601DateTimeString = Annotated[str, AfterValidator(_validate_iso8601_datetime)]


class RunInput(BaseModel):
    startup_name: str
    website_url: str | None = None
    description: str | None = None
    thesis: str | None = None
    analysis_focus: str | None = None
    id: UUID4String = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: ISO8601DateTimeString = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AgentFindings(BaseModel):
    sources: list[str]
    confidence: ConfidenceScore
    key_findings: list[str]
    evidence_gaps: list[str]


class MarketFindings(AgentFindings):
    pass


class CompetitionFindings(AgentFindings):
    pass


class ProductFindings(AgentFindings):
    pass


class RiskFindings(AgentFindings):
    pass


class CriticFindings(BaseModel):
    contradictions: list[str]
    weak_assumptions: list[str]
    unsupported_claims: list[str]
    open_questions: list[str]
    sources: list[str]
    confidence: ConfidenceScore


class StageResult(BaseModel):
    stage_name: str
    status: Literal["completed", "failed", "in_progress"]
    error: str | None = None
    findings: AgentFindings | CriticFindings | None = None


class MemoOutput(BaseModel):
    executive_summary: str
    research_findings: dict[str, AgentFindings]
    independent_review: CriticFindings | None = None
    recommendation: Literal["Invest", "Watch", "Pass"]
    confidence: ConfidenceScore
    confidence_factors: list[str]
    unresolved_risks: list[str]
    open_questions: list[str]
    sources: list[str]


class AnalysisRun(BaseModel):
    id: UUID4String = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["pending", "running", "complete", "partial", "failed"]
    created_at: ISO8601DateTimeString = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    input: RunInput
    stage_results: list[StageResult]
    memo: MemoOutput | None = None
